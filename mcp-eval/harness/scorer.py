"""
Scoring module for MCP tool eval.

Computes per-case and aggregate metrics for tool selection accuracy,
argument correctness, sequence matching, and optional LLM-as-judge e2e scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import jsonschema

from .types import (
    AggregateMetrics,
    EvalCase,
    EvalResult,
    EvalScenario,
    ExpectedToolCall,
    ToolCallResult,
)


# ============================================================================
# Per-case scoring
# ============================================================================


def score_tool_selection(
    predicted: list[ToolCallResult],
    expected: ExpectedToolCall,
    schemas: list[dict],
) -> dict[str, bool | None]:
    """Score a single tool call prediction against expected.

    Returns dict with keys: server_correct, tool_correct, tool_in_top3,
    args_exact_match, args_schema_valid.
    """
    result: dict[str, bool | None] = {
        "server_correct": None,
        "tool_correct": None,
        "tool_in_top3": None,
        "args_exact_match": None,
        "args_schema_valid": None,
    }

    if not predicted:
        result["server_correct"] = False
        result["tool_correct"] = False
        result["tool_in_top3"] = False
        result["args_exact_match"] = False
        result["args_schema_valid"] = False
        return result

    # First tool call
    first_call = predicted[0]

    # Tool name match
    result["tool_correct"] = first_call.tool_name == expected.tool

    # Top-3 match
    top3_names = [tc.tool_name for tc in predicted[:3]]
    result["tool_in_top3"] = expected.tool in top3_names

    # Server match — find which server the predicted tool belongs to
    predicted_server = _find_server_for_tool(first_call.tool_name, schemas)
    result["server_correct"] = predicted_server == expected.server

    # Argument scoring (only if tool is correct)
    if result["tool_correct"]:
        result["args_exact_match"] = _score_args_exact(
            first_call.arguments, expected.arguments
        )
        result["args_schema_valid"] = _validate_args_schema(
            first_call.tool_name, first_call.arguments, schemas
        )
    else:
        # If wrong tool, args can't be correct
        result["args_exact_match"] = False
        # But we can still check schema validity of what was called
        result["args_schema_valid"] = _validate_args_schema(
            first_call.tool_name, first_call.arguments, schemas
        )

    return result


def score_sequence(
    predicted: list[ToolCallResult],
    expected_sequence: list[ExpectedToolCall],
    schemas: list[dict],
) -> bool:
    """Score multi-tool sequence: all tools in the correct order."""
    if len(predicted) < len(expected_sequence):
        return False

    for i, expected in enumerate(expected_sequence):
        if i >= len(predicted):
            return False
        if predicted[i].tool_name != expected.tool:
            return False

    return True


def _find_server_for_tool(tool_name: str, schemas: list[dict]) -> str | None:
    """Look up which server a tool belongs to."""
    for schema in schemas:
        for tool in schema.get("tools", []):
            if tool["name"] == tool_name:
                return schema["server"]
    return None


def _score_args_exact(
    predicted_args: dict[str, Any],
    expected_args: dict[str, Any],
) -> bool:
    """Check if predicted arguments match expected arguments.

    Only checks expected keys — extra predicted keys are allowed.
    Values are compared with normalization (lowercase strings, sorted lists).
    """
    if not expected_args:
        return True  # No expected args means any args are fine

    for key, expected_val in expected_args.items():
        if key not in predicted_args:
            return False
        predicted_val = predicted_args[key]
        if not _values_match(predicted_val, expected_val):
            return False

    return True


def _values_match(predicted: Any, expected: Any) -> bool:
    """Compare two values with normalization."""
    # String comparison: case-insensitive, strip whitespace
    if isinstance(expected, str) and isinstance(predicted, str):
        return predicted.strip().lower() == expected.strip().lower()

    # List comparison: order-insensitive for simple lists
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(predicted) != len(expected):
            return False
        # Try sorted comparison for lists of strings
        try:
            return sorted(str(x).lower() for x in predicted) == sorted(
                str(x).lower() for x in expected
            )
        except TypeError:
            return predicted == expected

    # Dict comparison: recursive
    if isinstance(expected, dict) and isinstance(predicted, dict):
        return all(
            k in predicted and _values_match(predicted[k], v)
            for k, v in expected.items()
        )

    # Numeric comparison with tolerance
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        if expected == 0:
            return predicted == 0
        return abs(predicted - expected) / max(abs(expected), 1e-9) < 0.01

    # Direct comparison
    return predicted == expected


def _validate_args_schema(
    tool_name: str,
    arguments: dict[str, Any],
    schemas: list[dict],
) -> bool:
    """Validate tool arguments against the tool's JSON Schema."""
    # Find the tool's input schema
    input_schema = None
    for schema in schemas:
        for tool in schema.get("tools", []):
            if tool["name"] == tool_name:
                input_schema = tool.get("inputSchema")
                break
        if input_schema:
            break

    if not input_schema:
        return True  # Can't validate if schema not found

    try:
        jsonschema.validate(instance=arguments, schema=input_schema)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        return True  # Schema itself is invalid, don't penalize


# ============================================================================
# LLM-as-judge for e2e accuracy
# ============================================================================

E2E_JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response to a user query.
The assistant was given access to specific tools and used them to answer the query.

User query: {query}

Tool calls made: {tool_calls}

Tool results received: {tool_results}

Final assistant response: {final_response}

Expected behavior: The assistant should have called {expected_tools} and provided an accurate, helpful response.

Rate the overall quality of the response on a scale of 0.0 to 1.0:
- 1.0: Perfect — correct tools used, accurate interpretation, helpful response
- 0.7-0.9: Good — mostly correct with minor issues
- 0.4-0.6: Partial — some correct elements but significant gaps
- 0.1-0.3: Poor — wrong tools or very inaccurate
- 0.0: Complete failure

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""


async def judge_e2e_with_llm(
    case: EvalCase,
    result: EvalResult,
    tool_results: list[str],
    adapter: Any,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """Use LLM-as-judge for end-to-end evaluation.

    Args:
        case: The evaluation case
        result: The evaluation result with predictions
        tool_results: Mock tool execution results
        adapter: Model adapter for the judge LLM
        model: Model to use as judge

    Returns:
        Tuple of (score 0-1, reasoning string)
    """
    # Build expected tools description
    if case.expected:
        expected_tools = f"{case.expected.server}.{case.expected.tool}"
    elif case.expected_sequence:
        expected_tools = " -> ".join(
            f"{e.server}.{e.tool}" for e in case.expected_sequence
        )
    else:
        expected_tools = "unknown"

    tool_calls_str = json.dumps(
        [{"tool": tc.tool_name, "args": tc.arguments} for tc in result.predicted_tool_calls],
        indent=2,
    )

    prompt = E2E_JUDGE_PROMPT.format(
        query=case.query,
        tool_calls=tool_calls_str,
        tool_results=json.dumps(tool_results[:3], indent=2),  # Limit for context
        final_response=result.final_response or "(no response)",
        expected_tools=expected_tools,
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        _, text, _ = await adapter.call_with_tools(messages, [], model)

        if text:
            # Parse JSON from response
            # Handle potential markdown wrapping
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            return float(parsed.get("score", 0.0)), parsed.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Judge error: {e}"

    return 0.0, "Failed to parse judge response"


# ============================================================================
# Aggregate scoring
# ============================================================================


def compute_aggregate_metrics(
    results: list[EvalResult],
    cases: list[EvalCase] | None = None,
) -> AggregateMetrics:
    """Compute aggregate metrics from individual eval results.

    Args:
        results: List of per-case evaluation results
        cases: Optional list of eval cases (for tag-based breakdown)

    Returns:
        AggregateMetrics with overall and per-tag/scenario breakdowns
    """
    metrics = AggregateMetrics()
    metrics.total_cases = len(results)

    if not results:
        return metrics

    # Build case lookup for tags
    case_lookup: dict[str, EvalCase] = {}
    if cases:
        case_lookup = {c.id: c for c in cases}

    # Aggregate counters
    def _safe_count(values: list[bool | None]) -> tuple[int, int]:
        """Count True and total non-None values."""
        filtered = [v for v in values if v is not None]
        return sum(1 for v in filtered if v), len(filtered)

    server_vals = [r.server_correct for r in results]
    tool_vals = [r.tool_correct for r in results]
    top3_vals = [r.tool_in_top3 for r in results]
    arg_exact_vals = [r.args_exact_match for r in results]
    arg_schema_vals = [r.args_schema_valid for r in results]
    seq_vals = [r.sequence_correct for r in results]
    e2e_vals = [r.e2e_correct for r in results]
    e2e_scores = [r.e2e_score for r in results if r.e2e_score is not None]
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    errors = [r for r in results if r.error is not None]

    def _rate(values: list[bool | None]) -> float:
        correct, total = _safe_count(values)
        return correct / total if total > 0 else 0.0

    metrics.server_accuracy = _rate(server_vals)
    metrics.tool_accuracy = _rate(tool_vals)
    metrics.tool_accuracy_at_3 = _rate(top3_vals)
    metrics.arg_exact_match_rate = _rate(arg_exact_vals)
    metrics.arg_schema_valid_rate = _rate(arg_schema_vals)
    metrics.sequence_accuracy = _rate(seq_vals)
    metrics.e2e_accuracy = _rate(e2e_vals)
    metrics.avg_e2e_score = sum(e2e_scores) / len(e2e_scores) if e2e_scores else 0.0
    metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    metrics.error_rate = len(errors) / len(results) if results else 0.0

    # Per-scenario breakdown
    by_scenario: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario.value].append(r)

    for scenario, scenario_results in by_scenario.items():
        metrics.by_scenario[scenario] = {
            "count": len(scenario_results),
            "server_accuracy": _rate([r.server_correct for r in scenario_results]),
            "tool_accuracy": _rate([r.tool_correct for r in scenario_results]),
            "arg_exact_match_rate": _rate([r.args_exact_match for r in scenario_results]),
        }

    # Per-tag breakdown
    if case_lookup:
        by_tag: dict[str, list[EvalResult]] = defaultdict(list)
        for r in results:
            case = case_lookup.get(r.case_id)
            if case:
                for tag in case.tags:
                    by_tag[tag].append(r)

        for tag, tag_results in by_tag.items():
            metrics.by_tag[tag] = {
                "count": len(tag_results),
                "server_accuracy": _rate([r.server_correct for r in tag_results]),
                "tool_accuracy": _rate([r.tool_correct for r in tag_results]),
            }

    return metrics
