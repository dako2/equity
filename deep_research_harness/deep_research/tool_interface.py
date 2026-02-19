"""
Abstract Tool interface for the deep research harness.

Integrators must subclass ``Tool`` and implement the abstract properties/methods.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

from deep_research.models import (
    Placement,
    ToolResponse,
)
from deep_research.utils import Emitter

TOverride = TypeVar("TOverride")


class Tool(abc.ABC, Generic[TOverride]):
    """Abstract base for tools callable by the deep research agent."""

    def __init__(self, emitter: Emitter | None = None) -> None:
        self._emitter = emitter

    @property
    def emitter(self) -> Emitter:
        if self._emitter is None:
            raise ValueError(
                f"Emitter not set on tool {self.name}. Call set_emitter() first."
            )
        return self._emitter

    def set_emitter(self, emitter: Emitter) -> None:
        self._emitter = emitter

    @property
    @abc.abstractmethod
    def id(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name as passed to the LLM in the tool JSON schema."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """Human-readable display name."""
        ...

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abc.abstractmethod
    def tool_definition(self) -> dict[str, Any]:
        """Full JSON-schema tool definition for the LLM."""
        ...

    @abc.abstractmethod
    def run(
        self,
        placement: Placement,
        override_kwargs: TOverride,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """Execute the tool and return its response."""
        ...
