#!/usr/bin/env python3
"""
CLI entrypoint for MCP Tool Eval Framework.

Usage:
    # Run full eval with default config
    python run_eval.py

    # Run with specific config
    python run_eval.py --config config.yaml

    # Run only specific models
    python run_eval.py --models gpt-4o claude-sonnet

    # Run only specific scenarios
    python run_eval.py --scenarios single_tool server_routing

    # Filter by tags
    python run_eval.py --tags office easy

    # Limit number of cases
    python run_eval.py --limit 10

    # Enable e2e evaluation
    python run_eval.py --e2e

    # List available schemas
    python run_eval.py --list-schemas

    # List eval cases
    python run_eval.py --list-cases

    # Generate schemas from template
    python run_eval.py --generate-schemas --template registry/templates/office_template.yaml --count 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.adapters import create_adapter
from harness.runner import EvalRunner
from harness.types import AggregateMetrics
from registry.generator import generate_deterministic_from_template, list_schemas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_metrics(metrics: AggregateMetrics, model_name: str) -> None:
    """Print formatted metrics to console."""
    print(f"\n{'='*60}")
    print(f"  Results for: {model_name}")
    print(f"{'='*60}")
    print(f"  Total cases:          {metrics.total_cases}")
    print(f"  Server accuracy:      {metrics.server_accuracy:.1%}")
    print(f"  Tool accuracy:        {metrics.tool_accuracy:.1%}")
    print(f"  Tool accuracy @3:     {metrics.tool_accuracy_at_3:.1%}")
    print(f"  Arg exact match:      {metrics.arg_exact_match_rate:.1%}")
    print(f"  Arg schema valid:     {metrics.arg_schema_valid_rate:.1%}")
    print(f"  Sequence accuracy:    {metrics.sequence_accuracy:.1%}")
    if metrics.avg_e2e_score > 0:
        print(f"  E2E accuracy:         {metrics.e2e_accuracy:.1%}")
        print(f"  Avg E2E score:        {metrics.avg_e2e_score:.2f}")
    print(f"  Avg latency:          {metrics.avg_latency_ms:.0f}ms")
    print(f"  Error rate:           {metrics.error_rate:.1%}")

    if metrics.by_scenario:
        print(f"\n  Per-scenario breakdown:")
        for scenario, data in sorted(metrics.by_scenario.items()):
            print(f"    {scenario:<20} n={data['count']:<4}  "
                  f"server={data['server_accuracy']:.1%}  "
                  f"tool={data['tool_accuracy']:.1%}")

    if metrics.by_tag:
        print(f"\n  Per-tag breakdown:")
        for tag, data in sorted(metrics.by_tag.items()):
            print(f"    {tag:<20} n={data['count']:<4}  "
                  f"server={data['server_accuracy']:.1%}  "
                  f"tool={data['tool_accuracy']:.1%}")

    print(f"{'='*60}\n")


async def run_eval_for_model(
    model_config: dict,
    config: dict,
    eval_files: list[str],
    args: argparse.Namespace,
) -> tuple[str, AggregateMetrics]:
    """Run evaluation for a single model configuration."""
    model_name = model_config["name"]
    provider = model_config["provider"]
    model_id = model_config["model"]

    logger.info(f"Starting eval for {model_name} ({provider}/{model_id})")

    # Create adapter
    adapter = create_adapter(
        provider=provider,
        api_key=model_config.get("api_key"),
        base_url=model_config.get("base_url"),
    )

    # Runner settings
    runner_config = config.get("runner", {})
    e2e_config = config.get("e2e", {})

    # E2E settings
    enable_e2e = args.e2e or e2e_config.get("enabled", False)
    judge_adapter = None
    if enable_e2e:
        judge_provider = e2e_config.get("judge_provider", "openai")
        judge_adapter = create_adapter(provider=judge_provider)

    # Create runner
    runner = EvalRunner(
        adapter=adapter,
        model=model_id,
        registry_dir=PROJECT_ROOT / config["registry"]["schemas_dir"],
        mock_responses_dir=PROJECT_ROOT / config["eval_sets"]["mock_responses_dir"] if enable_e2e else None,
        system_prompt=runner_config.get("system_prompt", ""),
        max_concurrent=runner_config.get("max_concurrent", 5),
        enable_e2e=enable_e2e,
        max_e2e_turns=e2e_config.get("max_turns", 3),
        cache_dir=PROJECT_ROOT / runner_config["cache_dir"] if runner_config.get("cache_dir") else None,
        judge_adapter=judge_adapter,
        judge_model=e2e_config.get("judge_model", "gpt-4o"),
    )

    # Determine filters
    scenarios = args.scenarios or config.get("filters", {}).get("scenarios")
    tags = args.tags or config.get("filters", {}).get("tags")
    limit = args.limit or config.get("filters", {}).get("limit")

    # Output file
    output_config = config.get("output", {})
    results_dir = PROJECT_ROOT / output_config.get("results_dir", "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"{model_name}_{timestamp}.json"

    # Run
    results, metrics = await runner.run_eval(
        eval_files=[PROJECT_ROOT / f for f in eval_files],
        output_file=output_file,
        scenarios=scenarios,
        tags=tags,
        limit=limit,
    )

    print_metrics(metrics, model_name)
    return model_name, metrics


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    config = load_config(args.config)

    # Handle utility commands
    if args.list_schemas:
        list_schemas(PROJECT_ROOT / config["registry"]["schemas_dir"])
        return

    if args.list_cases:
        eval_files = config["eval_sets"]["files"]
        total = 0
        for f in eval_files:
            cases = EvalRunner.load_eval_cases(PROJECT_ROOT / f)
            print(f"\n{f}: {len(cases)} cases")
            for case in cases[:5]:
                print(f"  [{case.id}] {case.scenario.value}: {case.query[:60]}...")
            if len(cases) > 5:
                print(f"  ... and {len(cases) - 5} more")
            total += len(cases)
        print(f"\nTotal: {total} eval cases")
        return

    if args.generate_schemas:
        if not args.template:
            print("Error: --template required for --generate-schemas")
            sys.exit(1)
        generate_deterministic_from_template(
            str(PROJECT_ROOT / args.template),
            count=args.count or 10,
            output_dir=PROJECT_ROOT / config["registry"]["schemas_dir"] / "generated",
        )
        return

    # Determine which models to run
    all_models = config.get("models", [])
    if args.models:
        models_to_run = [m for m in all_models if m["name"] in args.models]
        if not models_to_run:
            print(f"Error: No matching models found. Available: {[m['name'] for m in all_models]}")
            sys.exit(1)
    else:
        models_to_run = all_models

    if not models_to_run:
        print("Error: No models configured. Edit config.yaml to add models.")
        sys.exit(1)

    eval_files = config["eval_sets"]["files"]

    # Run eval for each model
    all_metrics: dict[str, AggregateMetrics] = {}
    for model_config in models_to_run:
        name, metrics = await run_eval_for_model(model_config, config, eval_files, args)
        all_metrics[name] = metrics

    # Print comparison if multiple models
    if len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print(f"  Model Comparison")
        print(f"{'='*80}")
        header = f"  {'Model':<25} {'Server%':>8} {'Tool%':>8} {'Tool@3%':>8} {'ArgMatch%':>10} {'Latency':>8}"
        print(header)
        print(f"  {'-'*75}")
        for name, m in all_metrics.items():
            print(f"  {name:<25} {m.server_accuracy:>7.1%} {m.tool_accuracy:>7.1%} "
                  f"{m.tool_accuracy_at_3:>7.1%} {m.arg_exact_match_rate:>9.1%} "
                  f"{m.avg_latency_ms:>7.0f}ms")
        print(f"{'='*80}\n")

        # Save comparison
        output_config = config.get("output", {})
        results_dir = PROJECT_ROOT / output_config.get("results_dir", "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = results_dir / f"comparison_{timestamp}.json"
        with open(comparison_file, "w") as f:
            json.dump(
                {name: m.to_dict() for name, m in all_metrics.items()},
                f,
                indent=2,
            )
        logger.info(f"Comparison saved to {comparison_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP Tool Eval Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )

    # Model selection
    parser.add_argument(
        "--models", nargs="+", type=str,
        help="Model names to evaluate (from config.yaml). Default: all models.",
    )

    # Filtering
    parser.add_argument(
        "--scenarios", nargs="+", type=str,
        choices=["single_tool", "multi_tool", "server_routing"],
        help="Filter by scenario types",
    )
    parser.add_argument("--tags", nargs="+", type=str, help="Filter by tags")
    parser.add_argument("--limit", type=int, help="Limit number of eval cases")

    # E2E
    parser.add_argument("--e2e", action="store_true", help="Enable end-to-end evaluation")

    # Utility commands
    parser.add_argument("--list-schemas", action="store_true", help="List all tool schemas")
    parser.add_argument("--list-cases", action="store_true", help="List eval cases")

    # Schema generation
    parser.add_argument("--generate-schemas", action="store_true",
                        help="Generate schemas from template (deterministic)")
    parser.add_argument("--template", type=str, help="Template file for schema generation")
    parser.add_argument("--count", type=int, help="Number of schemas to generate")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
