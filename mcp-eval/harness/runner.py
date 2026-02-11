"""
Main evaluation runner.

Orchestrates the eval pipeline:
1. Load tool schemas from registry
2. Load eval cases
3. For each case: inject tools, call model, optionally run e2e with mock execution
4. Score results
5. Output metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .adapters import ModelAdapter, create_adapter
from .mock_mcp import MockMCPExecutor, resolve_server_for_tool
from .scorer import (
    compute_aggregate_metrics,
    judge_e2e_with_llm,
    score_sequence,
    score_tool_selection,
)
from .types import (
    AggregateMetrics,
    EvalCase,
    EvalResult,
    EvalScenario,
    ToolCallResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# System prompt for eval
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to various tools.
When the user asks you to do something, use the appropriate tool(s) to help them.
Always prefer using tools over providing general advice when a relevant tool is available.
If multiple tools are needed, call them in the logical order."""


# ============================================================================
# Eval Runner
# ============================================================================


class EvalRunner:
    """Main evaluation runner that orchestrates the eval pipeline."""

    def __init__(
        self,
        adapter: ModelAdapter,
        model: str,
        registry_dir: str | Path,
        mock_responses_dir: str | Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_concurrent: int = 5,
        enable_e2e: bool = False,
        max_e2e_turns: int = 3,
        cache_dir: str | Path | None = None,
        judge_adapter: ModelAdapter | None = None,
        judge_model: str = "gpt-4o",
    ):
        self.adapter = adapter
        self.model = model
        self.registry_dir = Path(registry_dir)
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.enable_e2e = enable_e2e
        self.max_e2e_turns = max_e2e_turns
        self.judge_adapter = judge_adapter
        self.judge_model = judge_model

        # Mock executor
        self.mock_executor: MockMCPExecutor | None = None
        if mock_responses_dir:
            self.mock_executor = MockMCPExecutor(mock_responses_dir)

        # Cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Schema cache
        self._schema_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Schema loading
    # ------------------------------------------------------------------

    def load_schemas(self, server_names: list[str]) -> list[dict]:
        """Load tool schemas for the given server names."""
        schemas = []
        for name in server_names:
            if name in self._schema_cache:
                schemas.append(self._schema_cache[name])
                continue

            # Search across domain directories
            found = False
            for domain_dir in self.registry_dir.iterdir():
                if not domain_dir.is_dir():
                    continue
                schema_file = domain_dir / f"{name}.json"
                if schema_file.exists():
                    with open(schema_file) as f:
                        schema = json.load(f)
                        self._schema_cache[name] = schema
                        schemas.append(schema)
                        found = True
                        break

            if not found:
                logger.warning(f"Schema not found for server: {name}")

        return schemas

    # ------------------------------------------------------------------
    # Case loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_eval_cases(eval_file: str | Path) -> list[EvalCase]:
        """Load eval cases from a JSONL file."""
        cases = []
        with open(eval_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    data = json.loads(line)
                    cases.append(EvalCase.from_dict(data))
        return cases

    @staticmethod
    def load_eval_cases_multi(eval_files: list[str | Path]) -> list[EvalCase]:
        """Load eval cases from multiple JSONL files."""
        cases = []
        for f in eval_files:
            cases.extend(EvalRunner.load_eval_cases(f))
        return cases

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, case: EvalCase, schemas: list[dict]) -> str:
        """Generate a cache key for a model call."""
        schema_str = json.dumps([s["server"] for s in schemas], sort_keys=True)
        key_str = f"{self.model}|{case.query}|{schema_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> dict | None:
        """Try to load a cached model response."""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, data: dict) -> None:
        """Save a model response to cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Single case evaluation
    # ------------------------------------------------------------------

    async def eval_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single case.

        Returns:
            EvalResult with predictions and scores
        """
        result = EvalResult(
            case_id=case.id,
            scenario=case.scenario,
            model=self.model,
        )

        try:
            # 1. Load schemas for available servers
            schemas = self.load_schemas(case.available_servers)
            if not schemas:
                result.error = f"No schemas found for servers: {case.available_servers}"
                return result

            # 2. Format tools for the model
            tools = self.adapter.format_tools(schemas)

            # 3. Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": case.query},
            ]

            # 4. Check cache
            cache_key = self._cache_key(case, schemas)
            cached = self._get_cached_result(cache_key)

            if cached:
                tool_calls = [
                    ToolCallResult(tool_name=tc["tool_name"], arguments=tc["arguments"])
                    for tc in cached.get("tool_calls", [])
                ]
                text_response = cached.get("text_response")
                latency_ms = cached.get("latency_ms", 0.0)
            else:
                # 5. Call model
                tool_calls, text_response, latency_ms = await self.adapter.call_with_tools(
                    messages=messages,
                    tools=tools,
                    model=self.model,
                )

                # Save to cache
                self._save_cache(cache_key, {
                    "tool_calls": [
                        {"tool_name": tc.tool_name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ],
                    "text_response": text_response,
                    "latency_ms": latency_ms,
                })

            result.predicted_tool_calls = tool_calls
            result.final_response = text_response
            result.latency_ms = latency_ms

            # 6. E2E: execute mock tools and continue conversation
            tool_results: list[str] = []
            if self.enable_e2e and self.mock_executor and tool_calls:
                tool_results = await self._run_e2e_loop(
                    case, schemas, tools, messages, tool_calls, result
                )

            # 7. Score
            self._score_result(case, result, schemas, tool_results)

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error evaluating case {case.id}: {e}", exc_info=True)

        return result

    async def _run_e2e_loop(
        self,
        case: EvalCase,
        schemas: list[dict],
        tools: list[dict],
        messages: list[dict],
        initial_tool_calls: list[ToolCallResult],
        result: EvalResult,
    ) -> list[str]:
        """Run the e2e mock execution loop.

        Feeds mock tool results back to the model until it produces a final
        text response or we hit the max turns limit.
        """
        tool_results: list[str] = []
        current_tool_calls = initial_tool_calls
        all_tool_calls = list(initial_tool_calls)

        for turn in range(self.max_e2e_turns):
            if not current_tool_calls:
                break

            # Build assistant message with tool calls (for conversation history)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
            if hasattr(current_tool_calls[0], "raw_response") and current_tool_calls[0].raw_response:
                # OpenAI format
                assistant_msg["tool_calls"] = [tc.raw_response for tc in current_tool_calls]
            messages.append(assistant_msg)

            # Execute each tool call and add results to messages
            for tc in current_tool_calls:
                server = resolve_server_for_tool(tc.tool_name, schemas) or "unknown"
                mock_result = self.mock_executor.execute(server, tc.tool_name, tc.arguments)
                tool_results.append(mock_result)

                # Get tool call ID from raw response
                tc_id = "call_mock"
                if tc.raw_response and isinstance(tc.raw_response, dict):
                    tc_id = tc.raw_response.get("id", "call_mock")

                tool_msg = self.adapter.format_tool_result(tc_id, tc.tool_name, mock_result)
                messages.append(tool_msg)

            # Call model again with tool results
            new_tool_calls, text_response, _ = await self.adapter.call_with_tools(
                messages=messages,
                tools=tools,
                model=self.model,
            )

            if text_response:
                result.final_response = text_response

            if new_tool_calls:
                all_tool_calls.extend(new_tool_calls)
                current_tool_calls = new_tool_calls
            else:
                break

        result.predicted_tool_calls = all_tool_calls
        return tool_results

    def _score_result(
        self,
        case: EvalCase,
        result: EvalResult,
        schemas: list[dict],
        tool_results: list[str],
    ) -> None:
        """Score a single eval result."""
        if case.scenario in (EvalScenario.SINGLE_TOOL, EvalScenario.SERVER_ROUTING):
            if case.expected:
                scores = score_tool_selection(
                    result.predicted_tool_calls, case.expected, schemas
                )
                result.server_correct = scores["server_correct"]
                result.tool_correct = scores["tool_correct"]
                result.tool_in_top3 = scores["tool_in_top3"]
                result.args_exact_match = scores["args_exact_match"]
                result.args_schema_valid = scores["args_schema_valid"]

        elif case.scenario == EvalScenario.MULTI_TOOL:
            if case.expected_sequence:
                # Score first tool
                first_expected = case.expected_sequence[0]
                scores = score_tool_selection(
                    result.predicted_tool_calls, first_expected, schemas
                )
                result.server_correct = scores["server_correct"]
                result.tool_correct = scores["tool_correct"]
                result.tool_in_top3 = scores["tool_in_top3"]
                result.args_exact_match = scores["args_exact_match"]
                result.args_schema_valid = scores["args_schema_valid"]

                # Score full sequence
                result.sequence_correct = score_sequence(
                    result.predicted_tool_calls, case.expected_sequence, schemas
                )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    async def eval_batch(
        self,
        cases: list[EvalCase],
        progress_callback: Any = None,
    ) -> list[EvalResult]:
        """Evaluate a batch of cases with concurrency control.

        Args:
            cases: List of eval cases to evaluate
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of eval results
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[EvalResult] = []
        completed = 0

        async def _eval_with_semaphore(case: EvalCase) -> EvalResult:
            nonlocal completed
            async with semaphore:
                result = await self.eval_case(case)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(cases))
                return result

        tasks = [_eval_with_semaphore(case) for case in cases]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def run_eval(
        self,
        eval_files: list[str | Path],
        output_file: str | Path | None = None,
        scenarios: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> tuple[list[EvalResult], AggregateMetrics]:
        """Run a full evaluation.

        Args:
            eval_files: Paths to JSONL eval files
            output_file: Optional path to save results JSON
            scenarios: Optional filter by scenario types
            tags: Optional filter by tags
            limit: Optional limit on number of cases

        Returns:
            Tuple of (results list, aggregate metrics)
        """
        # Load cases
        cases = self.load_eval_cases_multi(eval_files)
        logger.info(f"Loaded {len(cases)} eval cases from {len(eval_files)} files")

        # Filter
        if scenarios:
            cases = [c for c in cases if c.scenario.value in scenarios]
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        if limit:
            cases = cases[:limit]

        logger.info(f"Running {len(cases)} cases with model={self.model}, e2e={self.enable_e2e}")

        # Run
        def progress(done: int, total: int) -> None:
            if done % 5 == 0 or done == total:
                logger.info(f"  Progress: {done}/{total}")

        results = await self.eval_batch(cases, progress_callback=progress)

        # E2E judging
        if self.enable_e2e and self.judge_adapter:
            logger.info("Running LLM-as-judge for e2e scoring...")
            for i, (result, case) in enumerate(zip(results, cases)):
                if result.final_response and not result.error:
                    score, reasoning = await judge_e2e_with_llm(
                        case=case,
                        result=result,
                        tool_results=[],
                        adapter=self.judge_adapter,
                        model=self.judge_model,
                    )
                    result.e2e_score = score
                    result.e2e_correct = score >= 0.7

        # Compute metrics
        metrics = compute_aggregate_metrics(results, cases)

        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "model": self.model,
                "total_cases": len(cases),
                "metrics": metrics.to_dict(),
                "results": [r.to_dict() for r in results],
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results, metrics
