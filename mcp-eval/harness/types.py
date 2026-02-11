"""Shared data types for the MCP tool eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvalScenario(str, Enum):
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    SERVER_ROUTING = "server_routing"


@dataclass
class ExpectedToolCall:
    """Ground-truth expected tool call."""
    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCase:
    """A single evaluation case."""
    id: str
    scenario: EvalScenario
    query: str
    available_servers: list[str]
    tags: list[str] = field(default_factory=list)

    # For single_tool and server_routing
    expected: ExpectedToolCall | None = None

    # For multi_tool
    expected_sequence: list[ExpectedToolCall] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        expected = None
        if "expected" in data:
            expected = ExpectedToolCall(**data["expected"])

        expected_sequence = None
        if "expected_sequence" in data:
            expected_sequence = [ExpectedToolCall(**e) for e in data["expected_sequence"]]

        return cls(
            id=data["id"],
            scenario=EvalScenario(data["scenario"]),
            query=data["query"],
            available_servers=data["available_servers"],
            tags=data.get("tags", []),
            expected=expected,
            expected_sequence=expected_sequence,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "scenario": self.scenario.value,
            "query": self.query,
            "available_servers": self.available_servers,
        }
        if self.tags:
            d["tags"] = self.tags
        if self.expected:
            d["expected"] = {
                "server": self.expected.server,
                "tool": self.expected.tool,
                "arguments": self.expected.arguments,
            }
        if self.expected_sequence:
            d["expected_sequence"] = [
                {"server": e.server, "tool": e.tool, "arguments": e.arguments}
                for e in self.expected_sequence
            ]
        return d


@dataclass
class ToolCallResult:
    """Result from a model's tool call response."""
    tool_name: str
    arguments: dict[str, Any]
    raw_response: Any = None


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    case_id: str
    scenario: EvalScenario

    # Model outputs
    predicted_tool_calls: list[ToolCallResult] = field(default_factory=list)
    final_response: str | None = None

    # Scores (filled by scorer)
    server_correct: bool | None = None
    tool_correct: bool | None = None
    tool_in_top3: bool | None = None
    args_exact_match: bool | None = None
    args_schema_valid: bool | None = None
    sequence_correct: bool | None = None
    e2e_correct: bool | None = None
    e2e_score: float | None = None  # 0-1 for LLM-as-judge

    # Metadata
    model: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario.value,
            "model": self.model,
            "predicted_tool_calls": [
                {"tool_name": tc.tool_name, "arguments": tc.arguments}
                for tc in self.predicted_tool_calls
            ],
            "final_response": self.final_response,
            "server_correct": self.server_correct,
            "tool_correct": self.tool_correct,
            "tool_in_top3": self.tool_in_top3,
            "args_exact_match": self.args_exact_match,
            "args_schema_valid": self.args_schema_valid,
            "sequence_correct": self.sequence_correct,
            "e2e_correct": self.e2e_correct,
            "e2e_score": self.e2e_score,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all eval cases."""
    total_cases: int = 0
    server_accuracy: float = 0.0
    tool_accuracy: float = 0.0
    tool_accuracy_at_3: float = 0.0
    arg_exact_match_rate: float = 0.0
    arg_schema_valid_rate: float = 0.0
    sequence_accuracy: float = 0.0
    e2e_accuracy: float = 0.0
    avg_e2e_score: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0

    # Per-tag breakdown
    by_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-scenario breakdown
    by_scenario: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "server_accuracy": round(self.server_accuracy, 4),
            "tool_accuracy": round(self.tool_accuracy, 4),
            "tool_accuracy_at_3": round(self.tool_accuracy_at_3, 4),
            "arg_exact_match_rate": round(self.arg_exact_match_rate, 4),
            "arg_schema_valid_rate": round(self.arg_schema_valid_rate, 4),
            "sequence_accuracy": round(self.sequence_accuracy, 4),
            "e2e_accuracy": round(self.e2e_accuracy, 4),
            "avg_e2e_score": round(self.avg_e2e_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "by_tag": self.by_tag,
            "by_scenario": self.by_scenario,
        }
