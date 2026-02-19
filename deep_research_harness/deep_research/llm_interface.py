"""
Abstract LLM interface for the deep research harness.

Integrators must subclass ``LLM`` and implement ``stream`` / ``invoke``.

Reference implementations:
  - ``LiteLLMAdapter`` — generic multi-provider adapter backed by litellm.
  - ``GrokAdapter``     — xAI Grok adapter with native server-side
    ``web_search`` and ``x_search`` tools.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import Enum
from typing import Any

from pydantic import BaseModel

from deep_research.models import (
    ChatCompletionMessage,
    Delta,
    ModelResponseStream,
    ReasoningEffort,
    StreamingChoice,
    ToolChoiceOptions,
    Usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / Identity
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    model_name: str
    model_provider: str = "openai"
    max_input_tokens: int = 128_000
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_retries: int = 3
    request_timeout: int = 600  # seconds


class LLMUserIdentity(BaseModel):
    """Optional user identity passed through to the LLM provider."""

    user_id: str
    user_email: str | None = None
    user_name: str | None = None


# ---------------------------------------------------------------------------
# Abstract LLM
# ---------------------------------------------------------------------------


class LLM(ABC):
    """Abstract base class for language model backends.

    Subclasses must implement ``stream`` (returns a *synchronous* generator
    of ``ModelResponseStream`` chunks) and optionally ``invoke``.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def stream(
        self,
        prompt: list[ChatCompletionMessage] | ChatCompletionMessage,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        timeout_override: int | None = None,
    ) -> Generator[ModelResponseStream, None, None]:
        """Stream LLM response chunks synchronously."""
        ...

    def invoke(
        self,
        prompt: list[ChatCompletionMessage] | ChatCompletionMessage,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        timeout_override: int | None = None,
    ) -> Any:
        """Non-streaming invocation (default: consume the stream)."""
        accumulated_content = ""
        for chunk in self.stream(
            prompt=prompt,
            tools=tools,
            tool_choice=tool_choice,
            structured_response_format=structured_response_format,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            user_identity=user_identity,
            timeout_override=timeout_override,
        ):
            if chunk.choice.delta.content:
                accumulated_content += chunk.choice.delta.content
        return accumulated_content


# ---------------------------------------------------------------------------
# LiteLLM Adapter (reference implementation)
# ---------------------------------------------------------------------------


class LiteLLMAdapter(LLM):
    """Concrete LLM backed by litellm.completion (streaming)."""

    def stream(
        self,
        prompt: list[ChatCompletionMessage] | ChatCompletionMessage,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        timeout_override: int | None = None,
    ) -> Generator[ModelResponseStream, None, None]:
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "litellm is required for LiteLLMAdapter. Install it with: pip install litellm"
            ) from exc

        # Normalise prompt into list of dicts
        if not isinstance(prompt, list):
            prompt = [prompt]
        messages = [m.model_dump() for m in prompt]

        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "timeout": timeout_override or self.config.request_timeout,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice.value
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = litellm.completion(**kwargs)
        for chunk in response:
            delta_data = chunk.choices[0].delta if chunk.choices else None
            if delta_data is None:
                continue

            tool_calls_list = []
            if hasattr(delta_data, "tool_calls") and delta_data.tool_calls:
                from deep_research.models import (
                    ChatCompletionDeltaToolCall,
                    StreamFunctionCall,
                )

                for tc in delta_data.tool_calls:
                    func = None
                    if hasattr(tc, "function") and tc.function:
                        func = StreamFunctionCall(
                            name=getattr(tc.function, "name", None),
                            arguments=getattr(tc.function, "arguments", None),
                        )
                    tool_calls_list.append(
                        ChatCompletionDeltaToolCall(
                            id=getattr(tc, "id", None),
                            index=getattr(tc, "index", 0),
                            function=func,
                        )
                    )

            delta = Delta(
                content=getattr(delta_data, "content", None),
                reasoning_content=getattr(delta_data, "reasoning_content", None),
                tool_calls=tool_calls_list,
            )

            usage_obj = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage_obj = Usage(
                    completion_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                    prompt_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    total_tokens=getattr(chunk.usage, "total_tokens", 0) or 0,
                )

            yield ModelResponseStream(
                id=getattr(chunk, "id", ""),
                created=str(getattr(chunk, "created", "")),
                choice=StreamingChoice(
                    finish_reason=getattr(chunk.choices[0], "finish_reason", None)
                    if chunk.choices
                    else None,
                    delta=delta,
                ),
                usage=usage_obj,
            )


# ---------------------------------------------------------------------------
# Grok native server-side tool types
# ---------------------------------------------------------------------------


class GrokServerTool(str, Enum):
    """Native server-side tools supported by the xAI Grok API.

    These are *not* function-calling tools — they are executed on the
    xAI server and their results are folded into the model's response
    automatically.
    """

    LIVE_SEARCH = "live_search"
    X_SEARCH = "x_search"


# ---------------------------------------------------------------------------
# Grok Adapter (xAI — OpenAI-compatible with native search)
# ---------------------------------------------------------------------------


class GrokAdapter(LLM):
    """Concrete LLM backed by the xAI Grok API (OpenAI-compatible).

    This adapter uses the ``openai`` Python SDK pointed at the xAI
    endpoint (``https://api.x.ai/v1``).  In addition to regular
    function-calling tools it automatically injects xAI's native
    server-side ``web_search`` and/or ``x_search`` tools into every
    request so that the model can search the web and X (Twitter)
    autonomously.

    Parameters
    ----------
    config : LLMConfig
        Must include ``api_key`` (your xAI API key).
        ``model_name`` defaults to ``"grok-4-1-fast"`` and
        ``api_base`` defaults to ``"https://api.x.ai/v1"``.
    server_tools : list[GrokServerTool] | None
        Which native server-side tools to enable.  Defaults to
        ``[GrokServerTool.LIVE_SEARCH, GrokServerTool.X_SEARCH]``.

    Example
    -------
    >>> llm = GrokAdapter(
    ...     LLMConfig(
    ...         model_name="grok-4-1-fast",
    ...         api_key="xai-...",
    ...     ),
    ... )
    """

    XAI_API_BASE = "https://api.x.ai/v1"

    def __init__(
        self,
        config: LLMConfig,
        server_tools: list[GrokServerTool] | None = None,
    ) -> None:
        # Default model name and api_base for Grok
        if not config.api_base:
            config.api_base = self.XAI_API_BASE
        if not config.model_name:
            config.model_name = "grok-4-1-fast"
        # Grok-4-1-fast has a 2M context window
        if config.max_input_tokens == 128_000:  # still at default
            config.max_input_tokens = 131_072  # use a practical default
        super().__init__(config)

        if server_tools is None:
            server_tools = [GrokServerTool.LIVE_SEARCH, GrokServerTool.X_SEARCH]
        self._server_tools = server_tools

    def _build_native_tool_entries(self) -> list[dict[str, Any]]:
        """Build the native server-side tool entries for the request."""
        return [{"type": tool.value} for tool in self._server_tools]

    def stream(
        self,
        prompt: list[ChatCompletionMessage] | ChatCompletionMessage,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        structured_response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        timeout_override: int | None = None,
    ) -> Generator[ModelResponseStream, None, None]:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for GrokAdapter. Install it with: pip install openai"
            ) from exc

        from deep_research.models import (
            ChatCompletionDeltaToolCall,
            StreamFunctionCall,
        )

        # Normalise prompt into list of dicts
        if not isinstance(prompt, list):
            prompt = [prompt]
        messages = [m.model_dump() for m in prompt]

        # Merge native server-side tools with any function-calling tools
        all_tools: list[dict[str, Any]] = list(self._build_native_tool_entries())
        if tools:
            all_tools.extend(tools)

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=timeout_override or self.config.request_timeout,
        )

        create_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
        }
        if all_tools:
            create_kwargs["tools"] = all_tools
            # For native tools combined with function tools, use "auto"
            # so the model can choose between server-side and function tools
            create_kwargs["tool_choice"] = tool_choice.value
        if max_tokens:
            create_kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**create_kwargs)

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta_data = choice.delta
            if delta_data is None:
                continue

            tool_calls_list: list[ChatCompletionDeltaToolCall] = []
            if hasattr(delta_data, "tool_calls") and delta_data.tool_calls:
                for tc in delta_data.tool_calls:
                    func = None
                    if hasattr(tc, "function") and tc.function:
                        func = StreamFunctionCall(
                            name=getattr(tc.function, "name", None),
                            arguments=getattr(tc.function, "arguments", None),
                        )
                    tool_calls_list.append(
                        ChatCompletionDeltaToolCall(
                            id=getattr(tc, "id", None),
                            index=getattr(tc, "index", 0),
                            function=func,
                        )
                    )

            delta = Delta(
                content=getattr(delta_data, "content", None),
                reasoning_content=getattr(delta_data, "reasoning_content", None),
                tool_calls=tool_calls_list,
            )

            usage_obj = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage_obj = Usage(
                    completion_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                    prompt_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    total_tokens=getattr(chunk.usage, "total_tokens", 0) or 0,
                )

            yield ModelResponseStream(
                id=getattr(chunk, "id", ""),
                created=str(getattr(chunk, "created", "")),
                choice=StreamingChoice(
                    finish_reason=getattr(choice, "finish_reason", None),
                    delta=delta,
                ),
                usage=usage_obj,
            )
