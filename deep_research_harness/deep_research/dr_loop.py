"""
Deep Research orchestration loop.

This is the main entry-point for running a deep research session.
It proceeds through three phases:

1. **Clarification** (optional) — ask the user for clarifying questions.
2. **Research Plan** — generate a numbered research plan.
3. **Research Execution** — iterate: dispatch research agents, gather
   intermediate reports, then generate the final report.

All streaming output is emitted as ``Packet`` objects through the
``Emitter``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import cast

from deep_research.citation_processor import (
    CitationMapping,
    DynamicCitationProcessor,
)
from deep_research.llm_interface import LLM, LLMUserIdentity
from deep_research.llm_step import run_llm_step, run_llm_step_pkt_generator
from deep_research.models import (
    AgentResponseDelta,
    AgentResponseStart,
    ChatMessageSimple,
    DeepResearchPlanDelta,
    DeepResearchPlanStart,
    LlmStepResult,
    MessageType,
    OverallStop,
    Packet,
    Placement,
    SectionEnd,
    ToolCallKickoff,
    ToolCallSimple,
    ToolChoiceOptions,
    TopLevelBranching,
)
from deep_research.prompts import (
    CLARIFICATION_PROMPT,
    FINAL_REPORT_PROMPT,
    FIRST_CYCLE_REMINDER,
    FIRST_CYCLE_REMINDER_TOKENS,
    INTERNAL_SEARCH_CLARIFICATION_GUIDANCE,
    INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE,
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_PROMPT_REASONING,
    RESEARCH_AGENT_TOOL_NAME,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_PLAN_REMINDER,
    THINK_TOOL_RESPONSE_MESSAGE,
    THINK_TOOL_RESPONSE_TOKEN_COUNT,
    USER_FINAL_REPORT_QUERY,
    get_clarification_tool_definitions,
    get_orchestrator_tools,
)
from deep_research.research_agent import run_research_agent_calls
from deep_research.tool_interface import Tool
from deep_research.utils import (
    ChatStateContainer,
    Emitter,
    check_special_tool_calls,
    construct_message_history,
    create_think_tool_token_processor,
    get_current_llm_day_time,
)
from deep_research.models import ToolCallInfo, ReasoningEffort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_USER_MESSAGES_FOR_CONTEXT = 5
MAX_FINAL_REPORT_TOKENS = 20000
DEEP_RESEARCH_FORCE_REPORT_SECONDS = 30 * 60

MAX_ORCHESTRATOR_CYCLES = 8
MAX_ORCHESTRATOR_CYCLES_REASONING = 4


# ---------------------------------------------------------------------------
# Final report generation
# ---------------------------------------------------------------------------


def generate_final_report(
    history: list[ChatMessageSimple],
    research_plan: str,
    llm: LLM,
    token_counter: Callable[[str], int],
    state_container: ChatStateContainer,
    emitter: Emitter,
    turn_index: int,
    citation_mapping: CitationMapping,
    user_identity: LLMUserIdentity | None,
    saved_reasoning: str | None = None,
    pre_answer_processing_time: float | None = None,
) -> bool:
    """Generate the final research report.

    Returns True if reasoning occurred (turn_index was incremented).
    """
    final_report_prompt = FINAL_REPORT_PROMPT.format(
        current_datetime=get_current_llm_day_time(full_sentence=False),
    )
    system_prompt = ChatMessageSimple(
        message=final_report_prompt,
        token_count=token_counter(final_report_prompt),
        message_type=MessageType.SYSTEM,
    )
    final_reminder = USER_FINAL_REPORT_QUERY.format(research_plan=research_plan)
    reminder_message = ChatMessageSimple(
        message=final_reminder,
        token_count=token_counter(final_reminder),
        message_type=MessageType.USER_REMINDER,
    )
    final_report_history = construct_message_history(
        system_prompt=system_prompt,
        simple_chat_history=history,
        reminder_message=reminder_message,
        available_tokens=llm.config.max_input_tokens,
    )

    citation_processor = DynamicCitationProcessor()
    citation_processor.update_citation_mapping(citation_mapping)

    final_documents = list(citation_processor.citation_to_doc.values())

    llm_step_result, has_reasoned = run_llm_step(
        emitter=emitter,
        history=final_report_history,
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.NONE,
        llm=llm,
        placement=Placement(turn_index=turn_index),
        citation_processor=citation_processor,
        state_container=state_container,
        final_documents=final_documents,
        user_identity=user_identity,
        max_tokens=MAX_FINAL_REPORT_TOKENS,
        is_deep_research=True,
        pre_answer_processing_time=pre_answer_processing_time,
        timeout_override=300,
    )

    state_container.set_citation_mapping(citation_processor.citation_to_doc)

    final_report = llm_step_result.answer
    if final_report is None:
        raise ValueError("LLM failed to generate the final deep research report")

    if saved_reasoning:
        state_container.set_reasoning_tokens(saved_reasoning)

    return has_reasoned


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------


def run_deep_research(
    emitter: Emitter,
    state_container: ChatStateContainer,
    simple_chat_history: list[ChatMessageSimple],
    tools: list[Tool],
    llm: LLM,
    token_counter: Callable[[str], int],
    is_reasoning_model: bool = False,
    skip_clarification: bool = False,
    include_internal_search: bool = False,
    user_identity: LLMUserIdentity | None = None,
) -> None:
    """Run the full deep-research loop.

    Parameters
    ----------
    emitter:
        Packet emitter for streaming results to the caller.
    state_container:
        Mutable container that accumulates reasoning, answer, citations, and
        tool-call state across all phases.
    simple_chat_history:
        The conversation history up to this point (including the user query).
    tools:
        Available tools (web search, internal search, open URL …).
    llm:
        The language model to use for all prompts.
    token_counter:
        A callable that returns the token count for a string.
    is_reasoning_model:
        Whether the LLM is a "reasoning" model (e.g. o1).  When ``True``
        the orchestrator skips think-tool usage and uses fewer cycles.
    skip_clarification:
        If ``True``, skip the clarification phase entirely.
    include_internal_search:
        If ``True``, include internal-search guidance in the prompts.
    user_identity:
        Optional user identity for the LLM provider.
    """
    if llm.config.max_input_tokens < 50_000:
        raise RuntimeError(
            "Cannot run Deep Research with an LLM that has < 50,000 max input tokens"
        )

    processing_start_time = time.monotonic()
    available_tokens = llm.config.max_input_tokens

    llm_step_result: LlmStepResult | None = None
    orchestrator_start_turn_index = 1

    # ---------------------------------------------------------------
    # PHASE 1: Clarification (optional)
    # ---------------------------------------------------------------
    if not skip_clarification:
        internal_search_clarification_guidance = (
            INTERNAL_SEARCH_CLARIFICATION_GUIDANCE
            if include_internal_search
            else ""
        )
        clarification_prompt = CLARIFICATION_PROMPT.format(
            current_datetime=get_current_llm_day_time(full_sentence=False),
            internal_search_clarification_guidance=internal_search_clarification_guidance,
        )
        system_prompt = ChatMessageSimple(
            message=clarification_prompt,
            token_count=300,
            message_type=MessageType.SYSTEM,
        )

        truncated = construct_message_history(
            system_prompt=system_prompt,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            available_tokens=available_tokens,
            last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT,
        )

        clarification_tool_duration = time.monotonic() - processing_start_time
        llm_step_result, _ = run_llm_step(
            emitter=emitter,
            history=truncated,
            tool_definitions=get_clarification_tool_definitions(),
            tool_choice=ToolChoiceOptions.AUTO,
            llm=llm,
            placement=Placement(turn_index=0),
            citation_processor=None,
            state_container=state_container,
            user_identity=user_identity,
            is_deep_research=True,
            pre_answer_processing_time=clarification_tool_duration,
        )

        if not llm_step_result.tool_calls:
            state_container.set_is_clarification(True)
            emitter.emit(
                Packet(
                    placement=Placement(turn_index=0),
                    obj=OverallStop(stop_reason="clarification"),
                )
            )
            return

    # ---------------------------------------------------------------
    # PHASE 2: Research Plan
    # ---------------------------------------------------------------
    system_prompt = ChatMessageSimple(
        message=RESEARCH_PLAN_PROMPT.format(
            current_datetime=get_current_llm_day_time(full_sentence=False)
        ),
        token_count=300,
        message_type=MessageType.SYSTEM,
    )
    plan_reminder = ChatMessageSimple(
        message=RESEARCH_PLAN_REMINDER,
        token_count=100,
        message_type=MessageType.USER,
    )
    truncated = construct_message_history(
        system_prompt=system_prompt,
        simple_chat_history=simple_chat_history + [plan_reminder],
        reminder_message=None,
        available_tokens=available_tokens,
        last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT + 1,
    )

    plan_gen = run_llm_step_pkt_generator(
        history=truncated,
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.NONE,
        llm=llm,
        placement=Placement(turn_index=0),
        citation_processor=None,
        state_container=state_container,
        user_identity=user_identity,
        is_deep_research=True,
    )

    while True:
        try:
            packet = next(plan_gen)
            if isinstance(packet.obj, AgentResponseStart):
                emitter.emit(
                    Packet(placement=packet.placement, obj=DeepResearchPlanStart())
                )
            elif isinstance(packet.obj, AgentResponseDelta):
                emitter.emit(
                    Packet(
                        placement=packet.placement,
                        obj=DeepResearchPlanDelta(content=packet.obj.content),
                    )
                )
            else:
                emitter.emit(packet)
        except StopIteration as e:
            llm_step_result, reasoned = e.value
            emitter.emit(
                Packet(
                    placement=Placement(turn_index=1 if reasoned else 0),
                    obj=SectionEnd(),
                )
            )
            if reasoned:
                orchestrator_start_turn_index += 1
            break

    llm_step_result = cast(LlmStepResult, llm_step_result)
    research_plan = llm_step_result.answer
    if research_plan is None:
        raise RuntimeError("Deep Research failed to generate a research plan")

    # ---------------------------------------------------------------
    # PHASE 3: Research Execution
    # ---------------------------------------------------------------
    max_cycles = (
        MAX_ORCHESTRATOR_CYCLES
        if not is_reasoning_model
        else MAX_ORCHESTRATOR_CYCLES_REASONING
    )

    prompt_template = (
        ORCHESTRATOR_PROMPT
        if not is_reasoning_model
        else ORCHESTRATOR_PROMPT_REASONING
    )

    internal_search_guidance = (
        INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE
        if include_internal_search
        else ""
    )

    # Pre-compute approximate token count for the orchestrator prompt
    sample_prompt = prompt_template.format(
        current_datetime=get_current_llm_day_time(full_sentence=False),
        current_cycle_count=1,
        max_cycles=max_cycles,
        research_plan=research_plan,
        internal_search_research_task_guidance=internal_search_guidance,
    )
    orchestration_tokens = token_counter(sample_prompt)

    reasoning_cycles = 0
    most_recent_reasoning: str | None = None
    citation_mapping: CitationMapping = {}
    final_turn_index = orchestrator_start_turn_index

    for cycle in range(max_cycles):
        elapsed = time.monotonic() - processing_start_time
        timed_out = elapsed > DEEP_RESEARCH_FORCE_REPORT_SECONDS
        is_last_cycle = cycle == max_cycles - 1

        if timed_out or is_last_cycle:
            if timed_out:
                logger.info(
                    f"Deep research exceeded {DEEP_RESEARCH_FORCE_REPORT_SECONDS}s, forcing report"
                )
            report_turn = orchestrator_start_turn_index + cycle + reasoning_cycles
            report_reasoned = generate_final_report(
                history=simple_chat_history,
                research_plan=research_plan,
                llm=llm,
                token_counter=token_counter,
                state_container=state_container,
                emitter=emitter,
                turn_index=report_turn,
                citation_mapping=citation_mapping,
                user_identity=user_identity,
                pre_answer_processing_time=elapsed,
            )
            final_turn_index = report_turn + (1 if report_reasoned else 0)
            break

        # Cycle-1 reminder to ensure thorough exploration
        first_cycle_reminder = None
        if cycle == 1:
            first_cycle_reminder = ChatMessageSimple(
                message=FIRST_CYCLE_REMINDER,
                token_count=FIRST_CYCLE_REMINDER_TOKENS,
                message_type=MessageType.USER_REMINDER,
            )

        research_agent_calls: list[ToolCallKickoff] = []

        orchestrator_prompt = prompt_template.format(
            current_datetime=get_current_llm_day_time(full_sentence=False),
            current_cycle_count=cycle,
            max_cycles=max_cycles,
            research_plan=research_plan,
            internal_search_research_task_guidance=internal_search_guidance,
        )

        sys_prompt = ChatMessageSimple(
            message=orchestrator_prompt,
            token_count=orchestration_tokens,
            message_type=MessageType.SYSTEM,
        )

        truncated = construct_message_history(
            system_prompt=sys_prompt,
            simple_chat_history=simple_chat_history,
            reminder_message=first_cycle_reminder,
            available_tokens=available_tokens,
            last_n_user_messages=MAX_USER_MESSAGES_FOR_CONTEXT,
        )

        custom_processor = (
            create_think_tool_token_processor()
            if not is_reasoning_model
            else None
        )

        llm_step_result, has_reasoned = run_llm_step(
            emitter=emitter,
            history=truncated,
            tool_definitions=get_orchestrator_tools(
                include_think_tool=not is_reasoning_model,
            ),
            tool_choice=ToolChoiceOptions.REQUIRED,
            llm=llm,
            placement=Placement(
                turn_index=orchestrator_start_turn_index + cycle + reasoning_cycles,
            ),
            citation_processor=DynamicCitationProcessor(),
            state_container=state_container,
            user_identity=user_identity,
            custom_token_processor=custom_processor,
            is_deep_research=True,
            max_tokens=1024,
        )
        if has_reasoned:
            reasoning_cycles += 1

        tool_calls = llm_step_result.tool_calls or []

        if not tool_calls and cycle == 0:
            raise RuntimeError("Deep Research failed to generate any research tasks")

        if not tool_calls:
            logger.warning("No tool calls returned — forcing final report")
            report_turn = orchestrator_start_turn_index + cycle + reasoning_cycles
            report_reasoned = generate_final_report(
                history=simple_chat_history,
                research_plan=research_plan,
                llm=llm,
                token_counter=token_counter,
                state_container=state_container,
                emitter=emitter,
                turn_index=report_turn,
                citation_mapping=citation_mapping,
                user_identity=user_identity,
                pre_answer_processing_time=time.monotonic() - processing_start_time,
            )
            final_turn_index = report_turn + (1 if report_reasoned else 0)
            break

        special = check_special_tool_calls(tool_calls)

        if special.generate_report_tool_call:
            report_turn = special.generate_report_tool_call.placement.turn_index
            report_reasoned = generate_final_report(
                history=simple_chat_history,
                research_plan=research_plan,
                llm=llm,
                token_counter=token_counter,
                state_container=state_container,
                emitter=emitter,
                turn_index=report_turn,
                citation_mapping=citation_mapping,
                user_identity=user_identity,
                saved_reasoning=most_recent_reasoning,
                pre_answer_processing_time=time.monotonic() - processing_start_time,
            )
            final_turn_index = report_turn + (1 if report_reasoned else 0)
            break

        elif special.think_tool_call:
            tc = special.think_tool_call
            most_recent_reasoning = state_container.get_reasoning_tokens()
            tc_msg = tc.to_msg_str()
            tc_tokens = token_counter(tc_msg)

            think_simple = ToolCallSimple(
                tool_call_id=tc.tool_call_id,
                tool_name=tc.tool_name,
                tool_arguments=tc.tool_args,
                token_count=tc_tokens,
            )
            simple_chat_history.append(
                ChatMessageSimple(
                    message="",
                    token_count=tc_tokens,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=[think_simple],
                )
            )
            simple_chat_history.append(
                ChatMessageSimple(
                    message=THINK_TOOL_RESPONSE_MESSAGE,
                    token_count=THINK_TOOL_RESPONSE_TOKEN_COUNT,
                    message_type=MessageType.TOOL_CALL_RESPONSE,
                    tool_call_id=tc.tool_call_id,
                )
            )
            continue

        else:
            # Collect research_agent tool calls
            for tc in tool_calls:
                if tc.tool_name != RESEARCH_AGENT_TOOL_NAME:
                    logger.warning(f"Unexpected tool call: {tc.tool_name}")
                    continue
                research_agent_calls.append(tc)

            if not research_agent_calls:
                logger.warning("No research agent calls — forcing final report")
                report_turn = orchestrator_start_turn_index + cycle + reasoning_cycles
                report_reasoned = generate_final_report(
                    history=simple_chat_history,
                    research_plan=research_plan,
                    llm=llm,
                    token_counter=token_counter,
                    state_container=state_container,
                    emitter=emitter,
                    turn_index=report_turn,
                    citation_mapping=citation_mapping,
                    user_identity=user_identity,
                    pre_answer_processing_time=time.monotonic() - processing_start_time,
                )
                final_turn_index = report_turn + (1 if report_reasoned else 0)
                break

            if len(research_agent_calls) > 1:
                emitter.emit(
                    Packet(
                        placement=Placement(
                            turn_index=research_agent_calls[0].placement.turn_index,
                        ),
                        obj=TopLevelBranching(
                            num_parallel_branches=len(research_agent_calls),
                        ),
                    )
                )

            research_results = run_research_agent_calls(
                research_agent_calls=research_agent_calls,
                parent_tool_call_ids=[tc.tool_call_id for tc in tool_calls],
                tools=tools,
                emitter=emitter,
                state_container=state_container,
                llm=llm,
                is_reasoning_model=is_reasoning_model,
                token_counter=token_counter,
                citation_mapping=citation_mapping,
                user_identity=user_identity,
            )

            citation_mapping = research_results.citation_mapping

            # Build ASSISTANT message with all tool calls
            tc_simples: list[ToolCallSimple] = []
            for agent_call in research_agent_calls:
                msg = agent_call.to_msg_str()
                tc_simples.append(
                    ToolCallSimple(
                        tool_call_id=agent_call.tool_call_id,
                        tool_name=agent_call.tool_name,
                        tool_arguments=agent_call.tool_args,
                        token_count=token_counter(msg),
                    )
                )

            simple_chat_history.append(
                ChatMessageSimple(
                    message="",
                    token_count=sum(t.token_count for t in tc_simples),
                    message_type=MessageType.ASSISTANT,
                    tool_calls=tc_simples,
                )
            )

            # Add TOOL_CALL_RESPONSE messages for each intermediate report
            for tab_idx, report in enumerate(research_results.intermediate_reports):
                if report is None:
                    logger.error(
                        f"Research agent at tab_index {tab_idx} failed, skipping"
                    )
                    continue

                current_tc = research_agent_calls[tab_idx]
                tool_call_info = ToolCallInfo(
                    parent_tool_call_id=None,
                    turn_index=orchestrator_start_turn_index + cycle + reasoning_cycles,
                    tab_index=tab_idx,
                    tool_name=current_tc.tool_name,
                    tool_call_id=current_tc.tool_call_id,
                    tool_id=0,  # No DB lookup; placeholder
                    reasoning_tokens=llm_step_result.reasoning or most_recent_reasoning,
                    tool_call_arguments=current_tc.tool_args,
                    tool_call_response=report,
                    search_docs=None,
                )
                state_container.add_tool_call(tool_call_info)

                simple_chat_history.append(
                    ChatMessageSimple(
                        message=report,
                        token_count=token_counter(report),
                        message_type=MessageType.TOOL_CALL_RESPONSE,
                        tool_call_id=current_tc.tool_call_id,
                    )
                )

            most_recent_reasoning = None

    # ---------------------------------------------------------------
    # Emit final stop
    # ---------------------------------------------------------------
    emitter.emit(
        Packet(
            placement=Placement(turn_index=final_turn_index),
            obj=OverallStop(),
        )
    )
