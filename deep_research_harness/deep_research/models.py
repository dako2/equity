"""
Core data models for the deep research harness.

These are self-contained Pydantic models that replace the various Onyx-internal
types. Everything the harness needs is defined here so there are zero imports
from the parent project.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageType(str, Enum):
    """Message roles following the OpenAI Chat Completions convention."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL_RESPONSE = "tool_call_response"
    USER_REMINDER = "user_reminder"


class ToolChoiceOptions(str, Enum):
    REQUIRED = "required"
    AUTO = "auto"
    NONE = "none"


class ReasoningEffort(str, Enum):
    AUTO = "auto"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CitationMode(Enum):
    """How citations should be handled in the output."""

    REMOVE = "remove"
    KEEP_MARKERS = "keep_markers"
    HYPERLINK = "hyperlink"


# ---------------------------------------------------------------------------
# Placement / Packet system
# ---------------------------------------------------------------------------


class Placement(BaseModel):
    """Identifies where a streaming packet belongs in the conversation UI."""

    turn_index: int
    tab_index: int = 0
    sub_turn_index: int | None = None


# ---------------------------------------------------------------------------
# Streaming packet types
# ---------------------------------------------------------------------------


class BaseObj(BaseModel):
    type: str = ""


class SectionEnd(BaseObj):
    type: Literal["section_end"] = "section_end"


class OverallStop(BaseObj):
    type: Literal["stop"] = "stop"
    stop_reason: str | None = None


class TopLevelBranching(BaseObj):
    type: Literal["top_level_branching"] = "top_level_branching"
    num_parallel_branches: int


class PacketException(BaseObj):
    type: Literal["error"] = "error"
    exception: Exception = Field(exclude=True)
    model_config = {"arbitrary_types_allowed": True}


class ReasoningStart(BaseObj):
    type: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDelta(BaseObj):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    reasoning: str


class ReasoningDone(BaseObj):
    type: Literal["reasoning_done"] = "reasoning_done"


class AgentResponseStart(BaseObj):
    type: Literal["message_start"] = "message_start"
    final_documents: list[SearchDoc] | None = None
    pre_answer_processing_seconds: float | None = None


class AgentResponseDelta(BaseObj):
    type: Literal["message_delta"] = "message_delta"
    content: str


class CitationInfo(BaseObj):
    type: Literal["citation_info"] = "citation_info"
    citation_number: int
    document_id: str


class DeepResearchPlanStart(BaseObj):
    type: Literal["deep_research_plan_start"] = "deep_research_plan_start"


class DeepResearchPlanDelta(BaseObj):
    type: Literal["deep_research_plan_delta"] = "deep_research_plan_delta"
    content: str


class ResearchAgentStart(BaseObj):
    type: Literal["research_agent_start"] = "research_agent_start"
    research_task: str


class IntermediateReportStart(BaseObj):
    type: Literal["intermediate_report_start"] = "intermediate_report_start"


class IntermediateReportDelta(BaseObj):
    type: Literal["intermediate_report_delta"] = "intermediate_report_delta"
    content: str


class IntermediateReportCitedDocs(BaseObj):
    type: Literal["intermediate_report_cited_docs"] = "intermediate_report_cited_docs"
    cited_docs: list[SearchDoc] | None = None


class SearchToolStart(BaseObj):
    type: Literal["search_tool_start"] = "search_tool_start"
    is_internet_search: bool = False


class SearchToolQueriesDelta(BaseObj):
    type: Literal["search_tool_queries_delta"] = "search_tool_queries_delta"
    queries: list[str]


class SearchToolDocumentsDelta(BaseObj):
    type: Literal["search_tool_documents_delta"] = "search_tool_documents_delta"
    documents: list[SearchDoc]


class OpenUrlStart(BaseObj):
    type: Literal["open_url_start"] = "open_url_start"


class OpenUrlUrls(BaseObj):
    type: Literal["open_url_urls"] = "open_url_urls"
    urls: list[str]


class OpenUrlDocuments(BaseObj):
    type: Literal["open_url_documents"] = "open_url_documents"
    documents: list[SearchDoc]


class ToolCallDebug(BaseObj):
    type: Literal["tool_call_debug"] = "tool_call_debug"
    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


# Discriminated union of all possible packet object types
PacketObj = Union[
    OverallStop,
    SectionEnd,
    TopLevelBranching,
    PacketException,
    AgentResponseStart,
    AgentResponseDelta,
    SearchToolStart,
    SearchToolQueriesDelta,
    SearchToolDocumentsDelta,
    OpenUrlStart,
    OpenUrlUrls,
    OpenUrlDocuments,
    ReasoningStart,
    ReasoningDelta,
    ReasoningDone,
    CitationInfo,
    ToolCallDebug,
    DeepResearchPlanStart,
    DeepResearchPlanDelta,
    ResearchAgentStart,
    IntermediateReportStart,
    IntermediateReportDelta,
    IntermediateReportCitedDocs,
]


class Packet(BaseModel):
    placement: Placement
    obj: Annotated[PacketObj, Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Search / Document models
# ---------------------------------------------------------------------------


class SearchDoc(BaseModel):
    """A search result document with metadata for citation tracking."""

    document_id: str
    link: str | None = None
    title: str | None = None
    snippet: str | None = None
    chunk_ind: int = 0
    match_highlights: list[str] | None = None


class SearchDocsResponse(BaseModel):
    """Response from a search tool containing documents and citation mapping."""

    search_docs: list[SearchDoc]
    displayed_docs: list[SearchDoc] | None = None
    citation_mapping: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Tool call models
# ---------------------------------------------------------------------------

TOOL_CALL_MSG_FUNC_NAME = "function_name"
TOOL_CALL_MSG_ARGUMENTS = "arguments"


class ToolCallKickoff(BaseModel):
    """Represents a tool call requested by the LLM."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    placement: Placement

    def to_msg_str(self) -> str:
        return json.dumps(
            {
                TOOL_CALL_MSG_FUNC_NAME: self.tool_name,
                TOOL_CALL_MSG_ARGUMENTS: self.tool_args,
            }
        )


class ToolCallSimple(BaseModel):
    """Simplified tool call for message history representation."""

    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    token_count: int = 0


class ToolResponse(BaseModel):
    """Response from running a tool."""

    rich_response: SearchDocsResponse | str | None = None
    llm_facing_response: str
    tool_call: ToolCallKickoff | None = None


class ToolCallInfo(BaseModel):
    """Full tool call info for state tracking."""

    parent_tool_call_id: str | None
    turn_index: int
    tab_index: int
    tool_name: str
    tool_call_id: str
    tool_id: int
    reasoning_tokens: str | None
    tool_call_arguments: dict[str, Any]
    tool_call_response: str
    search_docs: list[SearchDoc] | None = None
    generated_images: list[Any] | None = None


# ---------------------------------------------------------------------------
# Chat message models
# ---------------------------------------------------------------------------


class ChatMessageSimple(BaseModel):
    """Simplified chat message for building LLM conversation history."""

    message: str
    token_count: int
    message_type: MessageType
    image_files: list[Any] | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallSimple] | None = None
    should_cache: bool = False
    file_id: str | None = None


class LlmStepResult(BaseModel):
    """Result of a single LLM generation step."""

    reasoning: str | None
    answer: str | None
    tool_calls: list[ToolCallKickoff] | None
    raw_answer: str | None = None


# ---------------------------------------------------------------------------
# LLM message types (OpenAI-compatible)
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: FunctionCall


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[Any]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


ChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage
LanguageModelInput = list[ChatCompletionMessage] | ChatCompletionMessage


# ---------------------------------------------------------------------------
# Model response types (streaming)
# ---------------------------------------------------------------------------


class StreamFunctionCall(BaseModel):
    arguments: str | None = None
    name: str | None = None


class ChatCompletionDeltaToolCall(BaseModel):
    id: str | None = None
    index: int = 0
    type: str = "function"
    function: StreamFunctionCall | None = None


class Delta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] = Field(default_factory=list)


class StreamingChoice(BaseModel):
    finish_reason: str | None = None
    index: int = 0
    delta: Delta = Field(default_factory=Delta)


class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ModelResponseStream(BaseModel):
    id: str
    created: str
    choice: StreamingChoice
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Citation mapping alias
# ---------------------------------------------------------------------------

CitationMapping = dict[int, SearchDoc]


# ---------------------------------------------------------------------------
# Deep research specific models
# ---------------------------------------------------------------------------


class SpecialToolCalls(BaseModel):
    think_tool_call: ToolCallKickoff | None = None
    generate_report_tool_call: ToolCallKickoff | None = None


class ResearchAgentCallResult(BaseModel):
    intermediate_report: str
    citation_mapping: CitationMapping


class CombinedResearchAgentCallResult(BaseModel):
    intermediate_reports: list[str | None]
    citation_mapping: CitationMapping


class FileToolMetadata(BaseModel):
    """Lightweight metadata for files in context."""

    file_id: str
    filename: str
    approx_char_count: int
