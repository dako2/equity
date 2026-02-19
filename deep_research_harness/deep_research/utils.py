"""
Utility functions for the deep research harness.

Includes:
  - Emitter: thread-safe packet queue for streaming
  - ChatStateContainer: thread-safe state accumulation
  - Text processing (JSON extraction)
  - Concurrency helpers
  - Prompt helpers
  - Think-tool token processor
  - Special tool call detection
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from deep_research.models import (
    ChatCompletionDeltaToolCall,
    ChatMessageSimple,
    CitationMapping,
    Delta,
    FunctionCall,
    MessageType,
    OverallStop,
    Packet,
    PacketException,
    Placement,
    SearchDoc,
    SpecialToolCalls,
    ToolCallInfo,
    ToolCallKickoff,
)
from deep_research.prompts import (
    GENERATE_REPORT_TOOL_NAME,
    THINK_TOOL_NAME,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")

# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class Emitter:
    """Thread-safe packet emitter using a Queue."""

    def __init__(self, bus: Queue[Packet] | None = None) -> None:
        self.bus: Queue[Packet] = bus or Queue()

    def emit(self, packet: Packet) -> None:
        self.bus.put(packet)


def get_default_emitter() -> Emitter:
    return Emitter()


# ---------------------------------------------------------------------------
# ChatStateContainer
# ---------------------------------------------------------------------------

SearchDocKey = str | tuple[str, int, tuple[str, ...]]


class ChatStateContainer:
    """Container for accumulating state during LLM loop execution.

    Thread-safe: all write operations are protected by a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tool_calls: list[ToolCallInfo] = []
        self.reasoning_tokens: str | None = None
        self.answer_tokens: str | None = None
        self.citation_to_doc: CitationMapping = {}
        self.is_clarification: bool = False
        self.pre_answer_processing_time: float | None = None
        self._all_search_docs: dict[SearchDocKey, SearchDoc] = {}
        self._emitted_citations: set[int] = set()

    def add_tool_call(self, tool_call: ToolCallInfo) -> None:
        with self._lock:
            self.tool_calls.append(tool_call)

    def set_reasoning_tokens(self, reasoning: str | None) -> None:
        with self._lock:
            self.reasoning_tokens = reasoning

    def set_answer_tokens(self, answer: str | None) -> None:
        with self._lock:
            self.answer_tokens = answer

    def set_citation_mapping(self, citation_to_doc: CitationMapping) -> None:
        with self._lock:
            self.citation_to_doc = citation_to_doc

    def set_is_clarification(self, is_clarification: bool) -> None:
        with self._lock:
            self.is_clarification = is_clarification

    def get_answer_tokens(self) -> str | None:
        with self._lock:
            return self.answer_tokens

    def get_reasoning_tokens(self) -> str | None:
        with self._lock:
            return self.reasoning_tokens

    def get_tool_calls(self) -> list[ToolCallInfo]:
        with self._lock:
            return self.tool_calls.copy()

    def get_citation_to_doc(self) -> CitationMapping:
        with self._lock:
            return self.citation_to_doc.copy()

    def get_is_clarification(self) -> bool:
        with self._lock:
            return self.is_clarification

    def set_pre_answer_processing_time(self, duration: float | None) -> None:
        with self._lock:
            self.pre_answer_processing_time = duration

    def get_pre_answer_processing_time(self) -> float | None:
        with self._lock:
            return self.pre_answer_processing_time

    @staticmethod
    def create_search_doc_key(
        search_doc: SearchDoc, use_simple_key: bool = True
    ) -> SearchDocKey:
        if use_simple_key:
            return search_doc.document_id
        match_highlights_tuple = tuple(sorted(search_doc.match_highlights or []))
        return (search_doc.document_id, search_doc.chunk_ind, match_highlights_tuple)

    def add_search_docs(
        self, search_docs: list[SearchDoc], use_simple_key: bool = True
    ) -> None:
        with self._lock:
            for doc in search_docs:
                key = self.create_search_doc_key(doc, use_simple_key)
                if key not in self._all_search_docs:
                    self._all_search_docs[key] = doc

    def get_all_search_docs(self) -> dict[SearchDocKey, SearchDoc]:
        with self._lock:
            return self._all_search_docs.copy()

    def add_emitted_citation(self, citation_num: int) -> None:
        with self._lock:
            self._emitted_citations.add(citation_num)

    def get_emitted_citations(self) -> set[int]:
        with self._lock:
            return self._emitted_citations.copy()


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def find_all_json_objects(text: str) -> list[dict]:
    """Find all JSON objects in text using balanced brace matching."""
    json_objects: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            brace_count = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == "{":
                    brace_count += 1
                elif text[j] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                json_objects.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break
        i += 1
    return json_objects


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------


class CallableProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def run_functions_tuples_in_parallel(
    functions_with_args: Sequence[tuple[CallableProtocol, tuple[Any, ...]]],
    allow_failures: bool = False,
    max_workers: int | None = None,
    timeout: float | None = None,
    timeout_callback: (
        Callable[[int, CallableProtocol, tuple[Any, ...]], Any] | None
    ) = None,
) -> list[Any]:
    """Execute functions in parallel, preserving result order."""
    workers = (
        min(max_workers, len(functions_with_args))
        if max_workers is not None
        else len(functions_with_args)
    )
    if workers <= 0:
        return []

    results: list[tuple[int, Any]] = []
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        future_to_index = {
            executor.submit(contextvars.copy_context().run, func, *args): i
            for i, (func, args) in enumerate(functions_with_args)
        }

        if timeout is not None:
            done, not_done = wait(future_to_index.keys(), timeout=timeout)
            for future in done:
                index = future_to_index[future]
                try:
                    results.append((index, future.result()))
                except Exception as e:
                    logger.exception(f"Function at index {index} failed: {e}")
                    results.append((index, None))
                    if not allow_failures:
                        raise
            for future in not_done:
                index = future_to_index[future]
                func, args = functions_with_args[index]
                if timeout_callback:
                    results.append((index, timeout_callback(index, func, args)))
                else:
                    results.append((index, None))
                    if not allow_failures:
                        raise TimeoutError(
                            f"Function at index {index} timed out after {timeout}s"
                        )
                future.cancel()
        else:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results.append((index, future.result()))
                except Exception as e:
                    logger.exception(f"Function at index {index} failed: {e}")
                    results.append((index, None))
                    if not allow_failures:
                        raise
    finally:
        executor.shutdown(wait=(timeout is None))

    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def get_current_llm_day_time(
    include_day_of_week: bool = True,
    full_sentence: bool = True,
    include_hour_min: bool = False,
) -> str:
    current_datetime = datetime.now()
    formatted_datetime = (
        current_datetime.strftime("%B %d, %Y %H:%M")
        if include_hour_min
        else current_datetime.strftime("%B %d, %Y")
    )
    day_of_week = current_datetime.strftime("%A")
    if full_sentence:
        return f"The current day and time is {day_of_week} {formatted_datetime}"
    if include_day_of_week:
        return f"{day_of_week} {formatted_datetime}"
    return formatted_datetime


# ---------------------------------------------------------------------------
# Message history construction
# ---------------------------------------------------------------------------


def construct_message_history(
    system_prompt: ChatMessageSimple | None,
    simple_chat_history: list[ChatMessageSimple],
    reminder_message: ChatMessageSimple | None,
    available_tokens: int,
    last_n_user_messages: int | None = None,
) -> list[ChatMessageSimple]:
    """Build message history list that fits within the token budget.

    Simplified version of Onyx's construct_message_history that doesn't
    handle project files or custom agent prompts.
    """
    if last_n_user_messages is not None and last_n_user_messages <= 0:
        raise ValueError("last_n_user_messages must be > 0")

    history_token_budget = available_tokens
    history_token_budget -= system_prompt.token_count if system_prompt else 0
    history_token_budget -= reminder_message.token_count if reminder_message else 0

    if history_token_budget < 0:
        raise ValueError("Not enough tokens to construct message history")

    if system_prompt:
        system_prompt.should_cache = True

    if not simple_chat_history:
        result: list[ChatMessageSimple] = []
        if system_prompt:
            result.append(system_prompt)
        if reminder_message:
            result.append(reminder_message)
        return result

    # Optionally filter to last N user messages
    chat_history = list(simple_chat_history)
    if last_n_user_messages is not None:
        user_msg_indices = [
            i
            for i, msg in enumerate(chat_history)
            if msg.message_type == MessageType.USER
        ]
        if user_msg_indices and len(user_msg_indices) > last_n_user_messages:
            nth_idx = user_msg_indices[-last_n_user_messages]
            chat_history = chat_history[nth_idx:]

    # Find last user message
    last_user_msg_index = None
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i].message_type == MessageType.USER:
            last_user_msg_index = i
            break

    if last_user_msg_index is None:
        raise ValueError("No user message found in simple_chat_history")

    # Ensure the last user message + after fit in budget, then add earlier history
    must_include = chat_history[last_user_msg_index:]
    must_include_tokens = sum(m.token_count for m in must_include)

    earlier_history = chat_history[:last_user_msg_index]
    remaining_budget = history_token_budget - must_include_tokens

    included_earlier: list[ChatMessageSimple] = []
    for msg in reversed(earlier_history):
        if remaining_budget - msg.token_count < 0:
            break
        included_earlier.insert(0, msg)
        remaining_budget -= msg.token_count

    result = []
    if system_prompt:
        result.append(system_prompt)
    result.extend(included_earlier)
    result.extend(must_include)
    if reminder_message:
        result.append(reminder_message)
    return result


# ---------------------------------------------------------------------------
# Think-tool token processor (for non-reasoning models)
# ---------------------------------------------------------------------------

JSON_PREFIX_WITH_SPACE = '{"reasoning": "'
JSON_PREFIX_NO_SPACE = '{"reasoning":"'


class ThinkToolProcessorState(BaseModel):
    think_tool_found: bool = False
    think_tool_index: int | None = None
    think_tool_id: str | None = None
    full_arguments: str = ""
    accumulated_args: str = ""
    json_prefix_stripped: bool = False
    buffer: str = ""


def _unescape_json_string(s: str) -> str:
    placeholder = "\x00ESCAPED_BACKSLASH\x00"
    result = s.replace("\\\\", placeholder)
    result = result.replace("\\n", "\n")
    result = result.replace("\\r", "\r")
    result = result.replace("\\t", "\t")
    result = result.replace('\\"', '"')
    result = result.replace(placeholder, "\\")
    return result


def _extract_reasoning_chunk(state: ThinkToolProcessorState) -> str | None:
    if not state.json_prefix_stripped:
        for prefix in [JSON_PREFIX_WITH_SPACE, JSON_PREFIX_NO_SPACE]:
            prefix_pos = state.accumulated_args.find(prefix)
            if prefix_pos != -1:
                content_start = prefix_pos + len(prefix)
                state.buffer = state.accumulated_args[content_start:]
                state.accumulated_args = ""
                state.json_prefix_stripped = True
                break
        if not state.json_prefix_stripped:
            return None
    else:
        state.buffer += state.accumulated_args
        state.accumulated_args = ""

    holdback = 3
    if len(state.buffer) <= holdback:
        return None

    to_emit = state.buffer[:-holdback]
    remaining = state.buffer[-holdback:]

    if to_emit and to_emit[-1] == "\\":
        remaining = to_emit[-1] + remaining
        to_emit = to_emit[:-1]

    state.buffer = remaining
    if to_emit:
        to_emit = _unescape_json_string(to_emit)
    return to_emit if to_emit else None


def create_think_tool_token_processor() -> (
    Callable[[Delta | None, Any], tuple[Delta | None, Any]]
):
    """Create a processor that converts think_tool calls to reasoning_content."""

    def process_token(delta: Delta | None, state: Any) -> tuple[Delta | None, Any]:
        if state is None:
            state = ThinkToolProcessorState()

        if delta is None:
            if state.think_tool_found and state.think_tool_id:
                complete_tool_call = ChatCompletionDeltaToolCall(
                    id=state.think_tool_id,
                    index=state.think_tool_index or 0,
                    type="function",
                    function=FunctionCall(
                        name=THINK_TOOL_NAME,
                        arguments=state.full_arguments,
                    ),
                )
                return Delta(tool_calls=[complete_tool_call]), state
            return None, state

        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function and tool_call.function.name == THINK_TOOL_NAME:
                    state.think_tool_found = True
                    state.think_tool_index = tool_call.index

                if (
                    state.think_tool_found
                    and tool_call.index == state.think_tool_index
                    and tool_call.id
                ):
                    state.think_tool_id = tool_call.id

                if (
                    state.think_tool_found
                    and tool_call.index == state.think_tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    state.full_arguments += tool_call.function.arguments
                    state.accumulated_args += tool_call.function.arguments
                    reasoning_chunk = _extract_reasoning_chunk(state)
                    if reasoning_chunk:
                        return Delta(reasoning_content=reasoning_chunk), state

        if state.think_tool_found:
            return None, state

        return delta, state

    return process_token


# ---------------------------------------------------------------------------
# Special tool call detection
# ---------------------------------------------------------------------------


def check_special_tool_calls(tool_calls: list[ToolCallKickoff]) -> SpecialToolCalls:
    think_tool_call: ToolCallKickoff | None = None
    generate_report_tool_call: ToolCallKickoff | None = None

    for tool_call in tool_calls:
        if tool_call.tool_name == THINK_TOOL_NAME:
            think_tool_call = tool_call
        elif tool_call.tool_name == GENERATE_REPORT_TOOL_NAME:
            generate_report_tool_call = tool_call

    return SpecialToolCalls(
        think_tool_call=think_tool_call,
        generate_report_tool_call=generate_report_tool_call,
    )


# ---------------------------------------------------------------------------
# Generate tools description
# ---------------------------------------------------------------------------


def generate_tools_description(tools: list[Any]) -> str:
    """Generate a comma-separated description of tool names."""
    return ", ".join(getattr(t, "name", str(t)) for t in tools)
