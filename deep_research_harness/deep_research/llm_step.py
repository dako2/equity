"""
LLM step execution for the deep research harness.

Provides ``run_llm_step_pkt_generator`` and ``run_llm_step`` which stream
an LLM turn as ``Packet`` objects while accumulating reasoning, answer text,
and tool calls.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable, Generator
from html import unescape
from typing import Any

from deep_research.citation_processor import DynamicCitationProcessor
from deep_research.llm_interface import LLM, LLMConfig, LLMUserIdentity
from deep_research.models import (
    AgentResponseDelta,
    AgentResponseStart,
    AssistantMessage,
    ChatCompletionMessage,
    ChatMessageSimple,
    CitationInfo,
    Delta,
    FunctionCall,
    LlmStepResult,
    MessageType,
    Packet,
    Placement,
    ReasoningDelta,
    ReasoningDone,
    ReasoningEffort,
    ReasoningStart,
    SearchDoc,
    SystemMessage,
    ToolCall,
    ToolCallKickoff,
    ToolChoiceOptions,
    ToolMessage,
    UserMessage,
)
from deep_research.utils import ChatStateContainer, Emitter, find_all_json_objects

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML tool call parsing helpers
# ---------------------------------------------------------------------------

_XML_INVOKE_BLOCK_RE = re.compile(
    r"<invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</invoke>",
    re.IGNORECASE | re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter\b(?P<attrs>[^>]*)>(?P<value>.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_CALLS_OPEN_MARKER = "<function_calls"
_FUNCTION_CALLS_CLOSE_MARKER = "</function_calls>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_llm_output(value: str) -> str:
    return "".join(c for c in value if c != "\x00" and not ("\ud800" <= c <= "\udfff"))


def _try_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("{") and stripped.endswith("}"))
    ):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _parse_tool_args_to_dict(raw_args: Any) -> dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return {
            k: _try_parse_json_string(
                _sanitize_llm_output(v) if isinstance(v, str) else v
            )
            for k, v in raw_args.items()
        }
    if not isinstance(raw_args, str):
        return {}
    raw_args = _sanitize_llm_output(raw_args)
    try:
        parsed1: Any = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed1, dict):
        return {k: _try_parse_json_string(v) for k, v in parsed1.items()}
    if isinstance(parsed1, str):
        try:
            parsed2: Any = json.loads(parsed1)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed2, dict):
            return {k: _try_parse_json_string(v) for k, v in parsed2.items()}
    return {}


def _update_tool_call_with_delta(
    tool_calls_in_progress: dict[int, dict[str, Any]],
    tool_call_delta: Any,
) -> None:
    index = tool_call_delta.index
    if index not in tool_calls_in_progress:
        tool_calls_in_progress[index] = {
            "id": f"fallback_{uuid.uuid4().hex}",
            "name": None,
            "arguments": "",
        }
    if tool_call_delta.id:
        tool_calls_in_progress[index]["id"] = tool_call_delta.id
    if tool_call_delta.function:
        if tool_call_delta.function.name:
            tool_calls_in_progress[index]["name"] = tool_call_delta.function.name
        if tool_call_delta.function.arguments:
            tool_calls_in_progress[index]["arguments"] += (
                tool_call_delta.function.arguments
            )


def _extract_tool_call_kickoffs(
    id_to_tool_call_map: dict[int, dict[str, Any]],
    turn_index: int,
    tab_index: int | None = None,
    sub_turn_index: int | None = None,
) -> list[ToolCallKickoff]:
    tool_calls: list[ToolCallKickoff] = []
    tab_index_calc = 0
    for tool_call_data in id_to_tool_call_map.values():
        if tool_call_data.get("id") and tool_call_data.get("name"):
            tool_args = _parse_tool_args_to_dict(tool_call_data.get("arguments"))
            tool_calls.append(
                ToolCallKickoff(
                    tool_call_id=tool_call_data["id"],
                    tool_name=tool_call_data["name"],
                    tool_args=tool_args,
                    placement=Placement(
                        turn_index=turn_index,
                        tab_index=tab_index_calc if tab_index is None else tab_index,
                        sub_turn_index=sub_turn_index,
                    ),
                )
            )
            tab_index_calc += 1
    return tool_calls


# ---------------------------------------------------------------------------
# XML content filter
# ---------------------------------------------------------------------------


class _XmlToolCallContentFilter:
    def __init__(self) -> None:
        self._pending = ""
        self._inside_function_calls_block = False

    def _matching_open_marker_prefix_len(self, text: str) -> int:
        max_len = min(len(text), len(_FUNCTION_CALLS_OPEN_MARKER) - 1)
        text_lower = text.lower()
        for candidate_len in range(max_len, 0, -1):
            if text_lower.endswith(_FUNCTION_CALLS_OPEN_MARKER[:candidate_len]):
                return candidate_len
        return 0

    def process(self, content: str) -> str:
        if not content:
            return ""
        self._pending += content
        output_parts: list[str] = []
        while self._pending:
            pending_lower = self._pending.lower()
            if self._inside_function_calls_block:
                end_idx = pending_lower.find(_FUNCTION_CALLS_CLOSE_MARKER)
                if end_idx == -1:
                    return "".join(output_parts)
                self._pending = self._pending[
                    end_idx + len(_FUNCTION_CALLS_CLOSE_MARKER) :
                ]
                self._inside_function_calls_block = False
                continue

            start_idx = self._find_open_marker(pending_lower)
            if start_idx == -1:
                tail_len = self._matching_open_marker_prefix_len(self._pending)
                emit_upto = len(self._pending) - tail_len
                if emit_upto > 0:
                    output_parts.append(self._pending[:emit_upto])
                    self._pending = self._pending[emit_upto:]
                return "".join(output_parts)

            if start_idx > 0:
                output_parts.append(self._pending[:start_idx])
            self._pending = self._pending[start_idx:]
            self._inside_function_calls_block = True
        return "".join(output_parts)

    def flush(self) -> str:
        if self._inside_function_calls_block:
            self._pending = ""
            self._inside_function_calls_block = False
            return ""
        remaining = self._pending
        self._pending = ""
        return remaining

    @staticmethod
    def _find_open_marker(text_lower: str) -> int:
        search_from = 0
        while True:
            idx = text_lower.find(_FUNCTION_CALLS_OPEN_MARKER, search_from)
            if idx == -1:
                return -1
            follower_pos = idx + len(_FUNCTION_CALLS_OPEN_MARKER)
            follower = text_lower[follower_pos] if follower_pos < len(text_lower) else None
            if follower is None or follower in {">", " ", "\t", "\n", "\r"}:
                return idx
            search_from = idx + 1


# ---------------------------------------------------------------------------
# Fallback tool call extraction from text
# ---------------------------------------------------------------------------


def _extract_xml_attribute(attrs: str, attr_name: str) -> str | None:
    attr_match = re.search(
        rf"""\b{re.escape(attr_name)}\s*=\s*(['"])(.*?)\1""",
        attrs,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not attr_match:
        return None
    return _sanitize_llm_output(unescape(attr_match.group(2).strip()))


def _parse_xml_parameter_value(raw_value: str, string_attr: str | None) -> Any:
    value = _sanitize_llm_output(unescape(raw_value).strip())
    if string_attr and string_attr.lower() == "true":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _resolve_tool_arguments(obj: dict[str, Any]) -> dict[str, Any] | None:
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if isinstance(arguments, dict):
        return arguments
    return None


def _try_match_json_to_tool(
    json_obj: dict[str, Any],
    tool_name_to_def: dict[str, dict],
) -> tuple[str, dict[str, Any]] | None:
    if "name" in json_obj and json_obj["name"] in tool_name_to_def:
        arguments = _resolve_tool_arguments(json_obj)
        if arguments is not None:
            return (json_obj["name"], arguments)

    if "function" in json_obj and isinstance(json_obj["function"], dict):
        func_obj = json_obj["function"]
        if "name" in func_obj and func_obj["name"] in tool_name_to_def:
            arguments = _resolve_tool_arguments(func_obj)
            if arguments is not None:
                return (func_obj["name"], arguments)

    for tool_name in tool_name_to_def:
        if tool_name in json_obj and isinstance(json_obj[tool_name], dict):
            return (tool_name, json_obj[tool_name])

    for tool_name, func_def in tool_name_to_def.items():
        params = func_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        if not properties:
            continue
        if all(req in json_obj for req in required):
            matching_props = [prop for prop in properties if prop in json_obj]
            if matching_props:
                filtered_args = {k: v for k, v in json_obj.items() if k in properties}
                return (tool_name, filtered_args)

    return None


def extract_tool_calls_from_response_text(
    response_text: str | None,
    tool_definitions: list[dict],
    placement: Placement,
) -> list[ToolCallKickoff]:
    """Fallback: extract tool calls from LLM text by matching JSON / XML."""
    if not response_text or not tool_definitions:
        return []

    tool_name_to_def: dict[str, dict] = {}
    for tool_def in tool_definitions:
        if tool_def.get("type") == "function" and "function" in tool_def:
            func_def = tool_def["function"]
            name = func_def.get("name")
            if name:
                tool_name_to_def[name] = func_def
    if not tool_name_to_def:
        return []

    matched: list[tuple[str, dict[str, Any]]] = []
    json_objects = find_all_json_objects(response_text)
    for json_obj in json_objects:
        m = _try_match_json_to_tool(json_obj, tool_name_to_def)
        if m:
            matched.append(m)

    if not matched:
        for invoke_match in _XML_INVOKE_BLOCK_RE.finditer(response_text):
            invoke_attrs = invoke_match.group("attrs")
            tool_name = _extract_xml_attribute(invoke_attrs, "name")
            if not tool_name or tool_name not in tool_name_to_def:
                continue
            tool_args: dict[str, Any] = {}
            invoke_body = invoke_match.group("body")
            for pm in _XML_PARAMETER_RE.finditer(invoke_body):
                pname = _extract_xml_attribute(pm.group("attrs"), "name")
                if pname:
                    string_attr = _extract_xml_attribute(pm.group("attrs"), "string")
                    tool_args[pname] = _parse_xml_parameter_value(
                        pm.group("value"), string_attr
                    )
            matched.append((tool_name, tool_args))

    result: list[ToolCallKickoff] = []
    for tab_idx, (tool_name, tool_args) in enumerate(matched):
        result.append(
            ToolCallKickoff(
                tool_call_id=f"extracted_{uuid.uuid4().hex[:8]}",
                tool_name=tool_name,
                tool_args=tool_args,
                placement=Placement(
                    turn_index=placement.turn_index,
                    tab_index=tab_idx,
                    sub_turn_index=placement.sub_turn_index,
                ),
            )
        )
    return result


# ---------------------------------------------------------------------------
# History translation
# ---------------------------------------------------------------------------


def translate_history_to_llm_format(
    history: list[ChatMessageSimple],
    llm_config: LLMConfig,
) -> list[ChatCompletionMessage]:
    """Convert ChatMessageSimple list to ChatCompletionMessage list."""
    messages: list[ChatCompletionMessage] = []
    for msg in history:
        if msg.message_type == MessageType.SYSTEM:
            messages.append(SystemMessage(content=msg.message))
        elif msg.message_type == MessageType.USER:
            messages.append(UserMessage(content=msg.message))
        elif msg.message_type == MessageType.USER_REMINDER:
            messages.append(UserMessage(content=msg.message))
        elif msg.message_type == MessageType.ASSISTANT:
            tc_list = None
            if msg.tool_calls:
                tc_list = [
                    ToolCall(
                        id=tc.tool_call_id,
                        function=FunctionCall(
                            name=tc.tool_name,
                            arguments=json.dumps(tc.tool_arguments),
                        ),
                    )
                    for tc in msg.tool_calls
                ]
            messages.append(
                AssistantMessage(
                    content=msg.message or None,
                    tool_calls=tc_list,
                )
            )
        elif msg.message_type == MessageType.TOOL_CALL_RESPONSE:
            if not msg.tool_call_id:
                raise ValueError("Tool call response missing tool_call_id")
            messages.append(
                ToolMessage(content=msg.message, tool_call_id=msg.tool_call_id)
            )
    return messages


# ---------------------------------------------------------------------------
# Increment turn helpers
# ---------------------------------------------------------------------------


def _increment_turns(
    turn_index: int, sub_turn_index: int | None
) -> tuple[int, int | None]:
    if sub_turn_index is None:
        return turn_index + 1, None
    return turn_index, sub_turn_index + 1


def _delta_has_action(delta: Delta) -> bool:
    return bool(delta.content or delta.reasoning_content or delta.tool_calls)


# ---------------------------------------------------------------------------
# Core LLM step
# ---------------------------------------------------------------------------


def run_llm_step_pkt_generator(
    history: list[ChatMessageSimple],
    tool_definitions: list[dict],
    tool_choice: ToolChoiceOptions,
    llm: LLM,
    placement: Placement,
    state_container: ChatStateContainer | None,
    citation_processor: DynamicCitationProcessor | None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    final_documents: list[SearchDoc] | None = None,
    user_identity: LLMUserIdentity | None = None,
    custom_token_processor: (
        Callable[[Delta | None, Any], tuple[Delta | None, Any]] | None
    ) = None,
    max_tokens: int | None = None,
    use_existing_tab_index: bool = False,
    is_deep_research: bool = False,
    pre_answer_processing_time: float | None = None,
    timeout_override: int | None = None,
) -> Generator[Packet, None, tuple[LlmStepResult, bool]]:
    """Run an LLM step and stream the response as packets.

    Yields Packet objects. On StopIteration, .value is (LlmStepResult, has_reasoned).
    """
    turn_index = placement.turn_index
    tab_index = placement.tab_index
    sub_turn_index = placement.sub_turn_index

    def _current_placement() -> Placement:
        return Placement(
            turn_index=turn_index,
            tab_index=tab_index,
            sub_turn_index=sub_turn_index,
        )

    llm_msg_history = translate_history_to_llm_format(history, llm.config)
    has_reasoned = False

    id_to_tool_call_map: dict[int, dict[str, Any]] = {}
    reasoning_start = False
    answer_start = False
    accumulated_reasoning = ""
    accumulated_answer = ""
    accumulated_raw_answer = ""
    xml_filter = _XmlToolCallContentFilter()
    processor_state: Any = None

    def _emit_citation_results(
        results: Generator[str | CitationInfo, None, None],
    ) -> Generator[Packet, None, None]:
        nonlocal accumulated_answer
        for result in results:
            if isinstance(result, str):
                accumulated_answer += result
                if state_container:
                    state_container.set_answer_tokens(accumulated_answer)
                yield Packet(
                    placement=_current_placement(),
                    obj=AgentResponseDelta(content=result),
                )
            elif isinstance(result, CitationInfo):
                yield Packet(placement=_current_placement(), obj=result)
                if state_container:
                    state_container.add_emitted_citation(result.citation_number)

    def _close_reasoning_if_active() -> Generator[Packet, None, None]:
        nonlocal reasoning_start, has_reasoned, turn_index, sub_turn_index
        if reasoning_start:
            yield Packet(
                placement=Placement(
                    turn_index=turn_index,
                    tab_index=tab_index,
                    sub_turn_index=sub_turn_index,
                ),
                obj=ReasoningDone(),
            )
            has_reasoned = True
            turn_index, sub_turn_index = _increment_turns(turn_index, sub_turn_index)
            reasoning_start = False

    def _emit_content_chunk(content_chunk: str) -> Generator[Packet, None, None]:
        nonlocal accumulated_answer, accumulated_reasoning, answer_start, reasoning_start
        nonlocal turn_index, sub_turn_index

        if is_deep_research and tool_choice == ToolChoiceOptions.REQUIRED:
            accumulated_reasoning += content_chunk
            if state_container:
                state_container.set_reasoning_tokens(accumulated_reasoning)
            if not reasoning_start:
                yield Packet(placement=_current_placement(), obj=ReasoningStart())
            yield Packet(
                placement=_current_placement(),
                obj=ReasoningDelta(reasoning=content_chunk),
            )
            reasoning_start = True
            return

        yield from _close_reasoning_if_active()

        if not answer_start:
            if state_container and pre_answer_processing_time is not None:
                state_container.set_pre_answer_processing_time(pre_answer_processing_time)
            yield Packet(
                placement=_current_placement(),
                obj=AgentResponseStart(
                    final_documents=final_documents,
                    pre_answer_processing_seconds=pre_answer_processing_time,
                ),
            )
            answer_start = True

        if citation_processor:
            yield from _emit_citation_results(
                citation_processor.process_token(content_chunk)
            )
        else:
            accumulated_answer += content_chunk
            if state_container:
                state_container.set_answer_tokens(accumulated_answer)
            yield Packet(
                placement=_current_placement(),
                obj=AgentResponseDelta(content=content_chunk),
            )

    for packet in llm.stream(
        prompt=llm_msg_history,
        tools=tool_definitions if tool_definitions else None,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        user_identity=user_identity,
        timeout_override=timeout_override,
    ):
        delta = packet.choice.delta

        if (
            delta.content is None
            and delta.reasoning_content is None
            and not delta.tool_calls
        ):
            continue

        if custom_token_processor:
            modified_delta, processor_state = custom_token_processor(
                delta, processor_state
            )
            if modified_delta is None:
                continue
            delta = modified_delta

        if delta.reasoning_content:
            accumulated_reasoning += delta.reasoning_content
            if state_container:
                state_container.set_reasoning_tokens(accumulated_reasoning)
            if not reasoning_start:
                yield Packet(placement=_current_placement(), obj=ReasoningStart())
            yield Packet(
                placement=_current_placement(),
                obj=ReasoningDelta(reasoning=delta.reasoning_content),
            )
            reasoning_start = True

        if delta.content:
            accumulated_raw_answer += delta.content
            filtered = xml_filter.process(delta.content)
            if filtered:
                yield from _emit_content_chunk(filtered)

        if delta.tool_calls:
            yield from _close_reasoning_if_active()
            for tc_delta in delta.tool_calls:
                _update_tool_call_with_delta(id_to_tool_call_map, tc_delta)

    # Flush XML filter
    tail = xml_filter.flush()
    if tail:
        yield from _emit_content_chunk(tail)

    # Flush custom processor
    if custom_token_processor:
        flush_delta, processor_state = custom_token_processor(None, processor_state)
        if flush_delta and flush_delta.tool_calls:
            for tc_delta in flush_delta.tool_calls:
                _update_tool_call_with_delta(id_to_tool_call_map, tc_delta)

    tool_calls = _extract_tool_call_kickoffs(
        id_to_tool_call_map=id_to_tool_call_map,
        turn_index=turn_index,
        tab_index=tab_index if use_existing_tab_index else None,
        sub_turn_index=sub_turn_index,
    )

    yield from _close_reasoning_if_active()

    if citation_processor:
        yield from _emit_citation_results(citation_processor.process_token(None))

    return (
        LlmStepResult(
            reasoning=accumulated_reasoning if accumulated_reasoning else None,
            answer=accumulated_answer if accumulated_answer else None,
            tool_calls=tool_calls if tool_calls else None,
            raw_answer=accumulated_raw_answer if accumulated_raw_answer else None,
        ),
        has_reasoned,
    )


def run_llm_step(
    emitter: Emitter,
    history: list[ChatMessageSimple],
    tool_definitions: list[dict],
    tool_choice: ToolChoiceOptions,
    llm: LLM,
    placement: Placement,
    state_container: ChatStateContainer | None,
    citation_processor: DynamicCitationProcessor | None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    final_documents: list[SearchDoc] | None = None,
    user_identity: LLMUserIdentity | None = None,
    custom_token_processor: (
        Callable[[Delta | None, Any], tuple[Delta | None, Any]] | None
    ) = None,
    max_tokens: int | None = None,
    use_existing_tab_index: bool = False,
    is_deep_research: bool = False,
    pre_answer_processing_time: float | None = None,
    timeout_override: int | None = None,
) -> tuple[LlmStepResult, bool]:
    """Consume the packet generator, emitting each packet, and return the final result."""
    gen = run_llm_step_pkt_generator(
        history=history,
        tool_definitions=tool_definitions,
        tool_choice=tool_choice,
        llm=llm,
        placement=placement,
        state_container=state_container,
        citation_processor=citation_processor,
        reasoning_effort=reasoning_effort,
        final_documents=final_documents,
        user_identity=user_identity,
        custom_token_processor=custom_token_processor,
        max_tokens=max_tokens,
        use_existing_tab_index=use_existing_tab_index,
        is_deep_research=is_deep_research,
        pre_answer_processing_time=pre_answer_processing_time,
        timeout_override=timeout_override,
    )
    while True:
        try:
            packet = next(gen)
            emitter.emit(packet)
        except StopIteration as e:
            return e.value
