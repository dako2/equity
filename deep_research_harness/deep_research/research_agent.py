"""
Research agent execution for the deep research harness.

Each research agent is assigned a specific research topic and iteratively
calls tools (web search, internal search, URL opening) until it gathers
enough information, then generates an intermediate report.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from deep_research.citation_processor import (
    CitationMapping,
    DynamicCitationProcessor,
    CitationMode,
    collapse_citations,
    update_citation_processor_from_tool_response,
)
from deep_research.llm_interface import LLM, LLMUserIdentity
from deep_research.llm_step import run_llm_step, run_llm_step_pkt_generator
from deep_research.models import (
    AgentResponseDelta,
    AgentResponseStart,
    ChatMessageSimple,
    CombinedResearchAgentCallResult,
    IntermediateReportCitedDocs,
    IntermediateReportDelta,
    IntermediateReportStart,
    LlmStepResult,
    MessageType,
    Packet,
    PacketException,
    Placement,
    ReasoningEffort,
    ResearchAgentCallResult,
    ResearchAgentStart,
    SearchDocsResponse,
    SectionEnd,
    ToolCallKickoff,
    ToolCallSimple,
    ToolChoiceOptions,
    ToolResponse,
)
from deep_research.prompts import (
    MAX_RESEARCH_CYCLES,
    OPEN_URL_REMINDER_RESEARCH_AGENT,
    OPEN_URLS_TOOL_DESCRIPTION,
    OPEN_URLS_TOOL_DESCRIPTION_REASONING,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT_REASONING,
    RESEARCH_AGENT_TASK_KEY,
    RESEARCH_REPORT_PROMPT,
    THINK_TOOL_RESPONSE_MESSAGE,
    THINK_TOOL_RESPONSE_TOKEN_COUNT,
    USER_REPORT_QUERY,
    WEB_SEARCH_TOOL_DESCRIPTION,
    get_research_agent_additional_tool_definitions,
)
from deep_research.tool_interface import Tool
from deep_research.utils import (
    ChatStateContainer,
    Emitter,
    check_special_tool_calls,
    construct_message_history,
    create_think_tool_token_processor,
    generate_tools_description,
    get_current_llm_day_time,
    run_functions_tuples_in_parallel,
)

logger = logging.getLogger(__name__)

# Timeouts
RESEARCH_AGENT_TIMEOUT_SECONDS = 30 * 60
RESEARCH_AGENT_TIMEOUT_MESSAGE = "Research Agent timed out after 30 minutes"
RESEARCH_AGENT_FORCE_REPORT_SECONDS = 12 * 60
MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS = 10000


# ---------------------------------------------------------------------------
# Tool runner (simplified)
# ---------------------------------------------------------------------------


def _run_tool_calls(
    tool_calls: list[ToolCallKickoff],
    tools: list[Tool],
) -> list[ToolResponse]:
    """Run tool calls against available tools and return responses."""
    tools_by_name = {tool.name: tool for tool in tools}
    responses: list[ToolResponse] = []
    for tc in tool_calls:
        tool = tools_by_name.get(tc.tool_name)
        if tool is None:
            logger.warning(f"Tool '{tc.tool_name}' not found, skipping")
            continue
        try:
            resp = tool.run(placement=tc.placement, override_kwargs=None, **tc.tool_args)
            resp.tool_call = tc
            responses.append(resp)
        except Exception as e:
            logger.error(f"Tool '{tc.tool_name}' failed: {e}")
    return responses


# ---------------------------------------------------------------------------
# Intermediate report generation
# ---------------------------------------------------------------------------


def generate_intermediate_report(
    research_topic: str,
    history: list[ChatMessageSimple],
    llm: LLM,
    token_counter: Callable[[str], int],
    citation_processor: DynamicCitationProcessor,
    user_identity: LLMUserIdentity | None,
    emitter: Emitter,
    placement: Placement,
) -> str:
    """Generate an intermediate report from research findings."""
    state_container = ChatStateContainer()
    system_prompt = ChatMessageSimple(
        message=RESEARCH_REPORT_PROMPT,
        token_count=token_counter(RESEARCH_REPORT_PROMPT),
        message_type=MessageType.SYSTEM,
    )

    reminder_str = USER_REPORT_QUERY.format(research_topic=research_topic)
    reminder_message = ChatMessageSimple(
        message=reminder_str,
        token_count=token_counter(reminder_str),
        message_type=MessageType.USER,
    )

    research_history = construct_message_history(
        system_prompt=system_prompt,
        simple_chat_history=history,
        reminder_message=reminder_message,
        available_tokens=llm.config.max_input_tokens,
    )

    gen = run_llm_step_pkt_generator(
        history=research_history,
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.NONE,
        llm=llm,
        placement=placement,
        citation_processor=citation_processor,
        state_container=state_container,
        reasoning_effort=ReasoningEffort.LOW,
        max_tokens=MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS,
        use_existing_tab_index=True,
        is_deep_research=True,
        timeout_override=300,
    )

    while True:
        try:
            packet = next(gen)
            if isinstance(packet.obj, AgentResponseStart):
                emitter.emit(
                    Packet(placement=placement, obj=IntermediateReportStart())
                )
            elif isinstance(packet.obj, AgentResponseDelta):
                emitter.emit(
                    Packet(
                        placement=placement,
                        obj=IntermediateReportDelta(content=packet.obj.content),
                    )
                )
            else:
                emitter.emit(Packet(placement=placement, obj=packet.obj))
        except StopIteration as e:
            llm_step_result, _ = e.value
            emitter.emit(
                Packet(
                    placement=placement,
                    obj=IntermediateReportCitedDocs(
                        cited_docs=list(
                            citation_processor.get_seen_citations().values()
                        )
                    ),
                )
            )
            emitter.emit(Packet(placement=placement, obj=SectionEnd()))
            break

    llm_step_result = cast(LlmStepResult, llm_step_result)
    final_report = llm_step_result.answer
    if final_report is None:
        raise ValueError(
            f"LLM failed to generate report for topic: {research_topic}"
        )
    return final_report


# ---------------------------------------------------------------------------
# Single research agent call
# ---------------------------------------------------------------------------


def run_research_agent_call(
    research_agent_call: ToolCallKickoff,
    parent_tool_call_id: str,
    tools: list[Tool],
    emitter: Emitter,
    state_container: ChatStateContainer,
    llm: LLM,
    is_reasoning_model: bool,
    token_counter: Callable[[str], int],
    user_identity: LLMUserIdentity | None,
) -> ResearchAgentCallResult | None:
    turn_index = research_agent_call.placement.turn_index
    tab_index = research_agent_call.placement.tab_index
    try:
        start_time = time.monotonic()

        citation_processor = DynamicCitationProcessor(
            citation_mode=CitationMode.KEEP_MARKERS
        )

        research_cycle_count = 0
        llm_cycle_count = 0
        current_tools = tools
        reasoning_cycles = 0
        just_ran_web_search = False

        research_topic = research_agent_call.tool_args[RESEARCH_AGENT_TASK_KEY]

        emitter.emit(
            Packet(
                placement=Placement(turn_index=turn_index, tab_index=tab_index),
                obj=ResearchAgentStart(research_task=research_topic),
            )
        )

        initial_user_message = ChatMessageSimple(
            message=research_topic,
            token_count=token_counter(research_topic),
            message_type=MessageType.USER,
        )
        msg_history: list[ChatMessageSimple] = [initial_user_message]
        most_recent_reasoning: str | None = None

        while research_cycle_count <= MAX_RESEARCH_CYCLES:
            elapsed = time.monotonic() - start_time
            if elapsed > RESEARCH_AGENT_FORCE_REPORT_SECONDS:
                logger.info(
                    f"Research agent exceeded {RESEARCH_AGENT_FORCE_REPORT_SECONDS}s, forcing report"
                )
                break
            if research_cycle_count == MAX_RESEARCH_CYCLES:
                break

            tools_by_name = {tool.name: tool for tool in current_tools}
            tools_description = generate_tools_description(current_tools)

            web_search_tip = (
                WEB_SEARCH_TOOL_DESCRIPTION
                if any(t.name == "web_search" for t in current_tools)
                else ""
            )
            open_urls_tip = (
                OPEN_URLS_TOOL_DESCRIPTION
                if any(t.name == "open_urls" for t in current_tools)
                else ""
            )
            if is_reasoning_model and open_urls_tip:
                open_urls_tip = OPEN_URLS_TOOL_DESCRIPTION_REASONING

            system_prompt_template = (
                RESEARCH_AGENT_PROMPT_REASONING
                if is_reasoning_model
                else RESEARCH_AGENT_PROMPT
            )
            system_prompt_str = system_prompt_template.format(
                available_tools=tools_description,
                current_datetime=get_current_llm_day_time(full_sentence=False),
                current_cycle_count=research_cycle_count,
                optional_internal_search_tool_description="",
                optional_web_search_tool_description=web_search_tip,
                optional_open_url_tool_description=open_urls_tip,
            )

            system_prompt = ChatMessageSimple(
                message=system_prompt_str,
                token_count=token_counter(system_prompt_str),
                message_type=MessageType.SYSTEM,
            )

            reminder_message = None
            if just_ran_web_search:
                reminder_message = ChatMessageSimple(
                    message=OPEN_URL_REMINDER_RESEARCH_AGENT,
                    token_count=100,
                    message_type=MessageType.USER,
                )

            constructed_history = construct_message_history(
                system_prompt=system_prompt,
                simple_chat_history=msg_history,
                reminder_message=reminder_message,
                available_tokens=llm.config.max_input_tokens,
            )

            research_agent_tools = get_research_agent_additional_tool_definitions(
                include_think_tool=not is_reasoning_model
            )
            custom_processor = (
                create_think_tool_token_processor()
                if not is_reasoning_model
                else None
            )

            llm_step_result, has_reasoned = run_llm_step(
                emitter=emitter,
                history=constructed_history,
                tool_definitions=[tool.tool_definition() for tool in current_tools]
                + research_agent_tools,
                tool_choice=ToolChoiceOptions.REQUIRED,
                llm=llm,
                placement=Placement(
                    turn_index=turn_index,
                    tab_index=tab_index,
                    sub_turn_index=llm_cycle_count + reasoning_cycles,
                ),
                citation_processor=None,
                state_container=None,
                reasoning_effort=ReasoningEffort.LOW,
                custom_token_processor=custom_processor,
                use_existing_tab_index=True,
                is_deep_research=True,
                max_tokens=1000,
            )
            if has_reasoned:
                reasoning_cycles += 1

            tool_calls = llm_step_result.tool_calls or []
            if tool_calls:
                first_type = tool_calls[0].tool_name
                tool_calls = [tc for tc in tool_calls if tc.tool_name == first_type]

            just_ran_web_search = False
            special = check_special_tool_calls(tool_calls)

            if special.generate_report_tool_call:
                report = generate_intermediate_report(
                    research_topic=research_topic,
                    history=msg_history,
                    llm=llm,
                    token_counter=token_counter,
                    citation_processor=citation_processor,
                    user_identity=user_identity,
                    emitter=emitter,
                    placement=Placement(turn_index=turn_index, tab_index=tab_index),
                )
                return ResearchAgentCallResult(
                    intermediate_report=report,
                    citation_mapping=citation_processor.get_seen_citations(),
                )
            elif special.think_tool_call:
                tc = special.think_tool_call
                tc_msg = tc.to_msg_str()
                tc_tokens = token_counter(tc_msg)
                think_simple = ToolCallSimple(
                    tool_call_id=tc.tool_call_id,
                    tool_name=tc.tool_name,
                    tool_arguments=tc.tool_args,
                    token_count=tc_tokens,
                )
                msg_history.append(
                    ChatMessageSimple(
                        message="",
                        token_count=tc_tokens,
                        message_type=MessageType.ASSISTANT,
                        tool_calls=[think_simple],
                    )
                )
                msg_history.append(
                    ChatMessageSimple(
                        message=THINK_TOOL_RESPONSE_MESSAGE,
                        token_count=THINK_TOOL_RESPONSE_TOKEN_COUNT,
                        message_type=MessageType.TOOL_CALL_RESPONSE,
                        tool_call_id=tc.tool_call_id,
                    )
                )
                reasoning_cycles += 1
                most_recent_reasoning = llm_step_result.reasoning
                continue
            else:
                tool_responses = _run_tool_calls(tool_calls, current_tools)
                if tool_calls and not tool_responses:
                    research_cycle_count += 1
                    llm_cycle_count += 1
                    continue

                valid_responses = [tr for tr in tool_responses if tr.tool_call is not None]

                if valid_responses:
                    tc_simples: list[ToolCallSimple] = []
                    for tr in valid_responses:
                        tc = tr.tool_call
                        assert tc is not None
                        tc_msg = tc.to_msg_str()
                        tc_simples.append(
                            ToolCallSimple(
                                tool_call_id=tc.tool_call_id,
                                tool_name=tc.tool_name,
                                tool_arguments=tc.tool_args,
                                token_count=token_counter(tc_msg),
                            )
                        )
                    msg_history.append(
                        ChatMessageSimple(
                            message="",
                            token_count=sum(t.token_count for t in tc_simples),
                            message_type=MessageType.ASSISTANT,
                            tool_calls=tc_simples,
                        )
                    )

                for tr in valid_responses:
                    tc = tr.tool_call
                    assert tc is not None

                    update_citation_processor_from_tool_response(
                        tool_response=tr,
                        citation_processor=citation_processor,
                    )

                    if isinstance(tr.rich_response, SearchDocsResponse):
                        if tr.rich_response.search_docs:
                            state_container.add_search_docs(tr.rich_response.search_docs)
                        if tc.tool_name == "web_search" and tr.rich_response.search_docs:
                            just_ran_web_search = True

                    msg_history.append(
                        ChatMessageSimple(
                            message=tr.llm_facing_response,
                            token_count=token_counter(tr.llm_facing_response),
                            message_type=MessageType.TOOL_CALL_RESPONSE,
                            tool_call_id=tc.tool_call_id,
                        )
                    )

                most_recent_reasoning = None
                llm_cycle_count += 1
                research_cycle_count += 1

        # Fallback: generate report from everything so far
        report = generate_intermediate_report(
            research_topic=research_topic,
            history=msg_history,
            llm=llm,
            token_counter=token_counter,
            citation_processor=citation_processor,
            user_identity=user_identity,
            emitter=emitter,
            placement=Placement(turn_index=turn_index, tab_index=tab_index),
        )
        return ResearchAgentCallResult(
            intermediate_report=report,
            citation_mapping=citation_processor.get_seen_citations(),
        )

    except Exception as e:
        logger.error(f"Error in research agent: {e}")
        emitter.emit(
            Packet(
                placement=Placement(turn_index=turn_index, tab_index=tab_index),
                obj=PacketException(exception=e),
            )
        )
        return None


# ---------------------------------------------------------------------------
# Parallel research agent calls
# ---------------------------------------------------------------------------


def _on_research_agent_timeout(
    index: int,
    func: Any,
    args: tuple[Any, ...],
) -> ResearchAgentCallResult:
    research_agent_call: ToolCallKickoff = args[0]
    research_task = research_agent_call.tool_args.get(RESEARCH_AGENT_TASK_KEY, "unknown")
    logger.warning(
        f"Research agent timed out for task: {research_task}"
    )
    return ResearchAgentCallResult(
        intermediate_report=RESEARCH_AGENT_TIMEOUT_MESSAGE,
        citation_mapping={},
    )


def run_research_agent_calls(
    research_agent_calls: list[ToolCallKickoff],
    parent_tool_call_ids: list[str],
    tools: list[Tool],
    emitter: Emitter,
    state_container: ChatStateContainer,
    llm: LLM,
    is_reasoning_model: bool,
    token_counter: Callable[[str], int],
    citation_mapping: CitationMapping,
    user_identity: LLMUserIdentity | None = None,
) -> CombinedResearchAgentCallResult:
    """Run multiple research agents in parallel and merge results."""
    functions_with_args = [
        (
            run_research_agent_call,
            (
                call,
                parent_id,
                tools,
                emitter,
                state_container,
                llm,
                is_reasoning_model,
                token_counter,
                user_identity,
            ),
        )
        for call, parent_id in zip(research_agent_calls, parent_tool_call_ids)
    ]

    results = run_functions_tuples_in_parallel(
        functions_with_args,
        allow_failures=False,
        timeout=RESEARCH_AGENT_TIMEOUT_SECONDS,
        timeout_callback=_on_research_agent_timeout,
    )

    updated_mapping = citation_mapping
    updated_answers: list[str | None] = []

    for result in results:
        if result is None:
            updated_answers.append(None)
            continue
        updated_answer, updated_mapping = collapse_citations(
            answer_text=result.intermediate_report,
            existing_citation_mapping=updated_mapping,
            new_citation_mapping=result.citation_mapping,
        )
        updated_answers.append(updated_answer)

    return CombinedResearchAgentCallResult(
        intermediate_reports=updated_answers,
        citation_mapping=updated_mapping,
    )
