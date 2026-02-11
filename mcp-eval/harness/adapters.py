"""
Model adapters for calling different LLM providers with tool schemas.

Supports:
- OpenAI (GPT-4o, o3, etc.)
- Anthropic (Claude)
- OpenAI-compatible endpoints (vLLM, ollama, Together, etc.)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from .types import ToolCallResult


class ModelAdapter(ABC):
    """Base adapter for calling models with tool schemas."""

    @abstractmethod
    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[list[ToolCallResult], str | None, float]:
        """Call the model with tools and return (tool_calls, text_response, latency_ms).

        Args:
            messages: Chat messages in OpenAI format
            tools: Tool definitions in the provider's format
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (list of tool calls, optional text response, latency in ms)
        """
        ...

    @abstractmethod
    def format_tools(self, mcp_schemas: list[dict]) -> list[dict]:
        """Convert MCP tool schemas to the provider's native tool format.

        Args:
            mcp_schemas: List of MCP server schemas with 'tools' arrays

        Returns:
            List of tool definitions in the provider's format
        """
        ...

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool execution result as a message for the provider.

        Args:
            tool_call_id: The ID of the tool call (from the model's response)
            tool_name: Name of the tool that was called
            result: String result from mock execution

        Returns:
            Message dict in the provider's format
        """
        ...


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        self.client = openai.AsyncOpenAI(**kwargs)

    def format_tools(self, mcp_schemas: list[dict]) -> list[dict]:
        """Convert MCP schemas to OpenAI function calling format."""
        openai_tools = []
        for schema in mcp_schemas:
            for tool in schema.get("tools", []):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                })
        return openai_tools

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[list[ToolCallResult], str | None, float]:
        start = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        tool_calls: list[ToolCallResult] = []
        text_response: str | None = message.content

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}

                tool_calls.append(ToolCallResult(
                    tool_name=tc.function.name,
                    arguments=args,
                    raw_response={
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    },
                ))

        return tool_calls, text_response, latency_ms


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key

        self.client = anthropic.AsyncAnthropic(**kwargs)

    def format_tools(self, mcp_schemas: list[dict]) -> list[dict]:
        """Convert MCP schemas to Anthropic tool use format."""
        anthropic_tools = []
        for schema in mcp_schemas:
            for tool in schema.get("tools", []):
                anthropic_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                })
        return anthropic_tools

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }
            ],
        }

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[list[ToolCallResult], str | None, float]:
        start = time.perf_counter()

        # Extract system message if present
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        tool_calls: list[ToolCallResult] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallResult(
                    tool_name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                    raw_response={
                        "id": block.id,
                        "type": block.type,
                        "name": block.name,
                        "input": block.input,
                    },
                ))

        text_response = "\n".join(text_parts) if text_parts else None
        return tool_calls, text_response, latency_ms


def create_adapter(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelAdapter:
    """Factory function to create a model adapter.

    Args:
        provider: One of 'openai', 'anthropic', 'vllm', 'ollama', 'together', 'openai_compatible'
        api_key: API key for the provider
        base_url: Custom base URL (for self-hosted / compatible endpoints)
        **kwargs: Additional provider-specific arguments

    Returns:
        A ModelAdapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(api_key=api_key, base_url=base_url, **kwargs)
    elif provider == "anthropic":
        return AnthropicAdapter(api_key=api_key, **kwargs)
    elif provider in ("vllm", "ollama", "together", "openai_compatible"):
        if not base_url:
            default_urls = {
                "vllm": "http://localhost:8000/v1",
                "ollama": "http://localhost:11434/v1",
                "together": "https://api.together.xyz/v1",
            }
            base_url = default_urls.get(provider)
        return OpenAIAdapter(api_key=api_key or "not-needed", base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', 'vllm', 'ollama', 'together', or 'openai_compatible'")
