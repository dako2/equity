"""
Dynamic Citation Processor for LLM Responses.

Self-contained citation processing with three modes:
  - REMOVE: strip citation markers from text
  - KEEP_MARKERS: preserve original [1], [2] markers unchanged
  - HYPERLINK: replace markers with [[1]](url) links and emit CitationInfo
"""

from __future__ import annotations

import re
from collections.abc import Generator
from typing import TypeAlias

from deep_research.models import (
    CitationInfo,
    CitationMode,
    SearchDoc,
)

import logging

logger = logging.getLogger(__name__)

TRIPLE_BACKTICK = "```"
STOP_STREAM_PAT = "<<END_OF_STREAM>>"

CitationMapping: TypeAlias = dict[int, SearchDoc]


def in_code_block(llm_text: str) -> bool:
    """Check if we're currently inside a code block by counting triple backticks."""
    count = llm_text.count(TRIPLE_BACKTICK)
    return count % 2 != 0


class DynamicCitationProcessor:
    """
    A citation processor that accepts dynamic citation mappings.

    This processor detects citations (e.g., [1], [2,3], [[4]]) from an LLM stream
    and handles them according to the configured CitationMode.
    """

    def __init__(
        self,
        citation_mode: CitationMode = CitationMode.HYPERLINK,
        stop_stream: str | None = STOP_STREAM_PAT,
    ) -> None:
        self.citation_to_doc: CitationMapping = {}
        self.seen_citations: CitationMapping = {}

        self.llm_out = ""
        self.curr_segment = ""
        self.hold = ""
        self.stop_stream = stop_stream
        self.citation_mode = citation_mode

        self.cited_documents_in_order: list[SearchDoc] = []
        self.cited_document_ids: set[str] = set()
        self.recent_cited_documents: set[str] = set()
        self.non_citation_count = 0

        self.possible_citation_pattern = re.compile(r"([\[【［]+(?:\d+,? ?)*$)")
        self.citation_pattern = re.compile(
            r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])"
        )

    def update_citation_mapping(
        self,
        citation_mapping: CitationMapping,
        update_duplicate_keys: bool = False,
    ) -> None:
        if update_duplicate_keys:
            self.citation_to_doc.update(citation_mapping)
        else:
            duplicate_keys = set(citation_mapping.keys()) & set(self.citation_to_doc.keys())
            non_duplicate_mapping = {
                k: v for k, v in citation_mapping.items() if k not in duplicate_keys
            }
            self.citation_to_doc.update(non_duplicate_mapping)

    def process_token(
        self, token: str | None
    ) -> Generator[str | CitationInfo, None, None]:
        if token is None:
            if self.curr_segment:
                yield self.curr_segment
            return

        if self.stop_stream:
            next_hold = self.hold + token
            if self.stop_stream in next_hold:
                stop_pos = next_hold.find(self.stop_stream)
                text_before_stop = next_hold[:stop_pos]
                if text_before_stop:
                    self.hold = ""
                    token = text_before_stop
                else:
                    return
            elif next_hold == self.stop_stream[: len(next_hold)]:
                self.hold = next_hold
                return
            else:
                token = next_hold
                self.hold = ""

        self.curr_segment += token
        self.llm_out += token

        if "`" in self.curr_segment:
            if self.curr_segment.endswith("`"):
                pass
            elif "```" in self.curr_segment:
                parts = self.curr_segment.split("```")
                if len(parts) > 1 and len(parts[1]) > 0:
                    piece_that_comes_after = parts[1][0]
                    if piece_that_comes_after == "\n" and in_code_block(self.llm_out):
                        self.curr_segment = self.curr_segment.replace("```", "```plaintext")

        citation_matches = list(self.citation_pattern.finditer(self.curr_segment))
        possible_citation_found = bool(
            re.search(self.possible_citation_pattern, self.curr_segment)
        )

        result = ""
        if citation_matches and not in_code_block(self.llm_out):
            match_idx = 0
            for match in citation_matches:
                match_span = match.span()
                intermatch_str = self.curr_segment[match_idx : match_span[0]]
                self.non_citation_count += len(intermatch_str)
                match_idx = match_span[1]

                if intermatch_str:
                    has_leading_space = intermatch_str[-1].isspace()
                else:
                    if match_idx > 0:
                        has_leading_space = True
                    else:
                        segment_start_idx = len(self.llm_out) - len(self.curr_segment)
                        if segment_start_idx > 0:
                            has_leading_space = self.llm_out[segment_start_idx - 1].isspace()
                        else:
                            has_leading_space = False

                if self.non_citation_count > 5:
                    self.recent_cited_documents.clear()

                citation_text, citation_info_list = self._process_citation(match, has_leading_space)

                if self.citation_mode == CitationMode.HYPERLINK:
                    if intermatch_str:
                        yield intermatch_str
                    for citation in citation_info_list:
                        yield citation
                    if citation_text:
                        yield citation_text
                elif self.citation_mode == CitationMode.KEEP_MARKERS:
                    if intermatch_str:
                        yield intermatch_str
                    yield match.group()
                else:  # REMOVE
                    if intermatch_str:
                        remaining_text = self.curr_segment[match_span[1] :]
                        if intermatch_str[-1].isspace() and remaining_text:
                            first_char = remaining_text[0]
                            if first_char.isspace() or first_char in ".,;:!?)]}":
                                intermatch_str = intermatch_str.rstrip()
                        if intermatch_str:
                            yield intermatch_str

                self.non_citation_count = 0

            self.curr_segment = self.curr_segment[match_idx:]
            self.non_citation_count = len(self.curr_segment)

        if not possible_citation_found:
            result += self.curr_segment
            self.non_citation_count += len(self.curr_segment)
            self.curr_segment = ""

        if result:
            yield result

    def _process_citation(
        self, match: re.Match, has_leading_space: bool  # type: ignore[type-arg]
    ) -> tuple[str, list[CitationInfo]]:
        citation_str: str = match.group()
        formatted = match.lastindex == 1

        citation_info_list: list[CitationInfo] = []
        formatted_citation_parts: list[str] = []

        citation_content = citation_str[2:-2] if formatted else citation_str[1:-1]

        for num_str in citation_content.split(","):
            num_str = num_str.strip()
            if not num_str:
                continue

            try:
                num = int(num_str)
            except ValueError:
                continue

            if num not in self.citation_to_doc:
                continue

            search_doc = self.citation_to_doc[num]
            doc_id = search_doc.document_id
            link = search_doc.link or ""

            self.seen_citations[num] = search_doc

            if self.citation_mode != CitationMode.HYPERLINK:
                continue

            formatted_citation_parts.append(f"[[{num}]]({link})")

            if doc_id in self.recent_cited_documents:
                continue
            self.recent_cited_documents.add(doc_id)

            if doc_id not in self.cited_document_ids:
                self.cited_document_ids.add(doc_id)
                self.cited_documents_in_order.append(search_doc)
                citation_info_list.append(
                    CitationInfo(citation_number=num, document_id=doc_id)
                )

        formatted_citation_text = " ".join(formatted_citation_parts)
        if formatted_citation_text and not has_leading_space:
            formatted_citation_text = " " + formatted_citation_text

        return formatted_citation_text, citation_info_list

    def get_cited_documents(self) -> list[SearchDoc]:
        return self.cited_documents_in_order

    def get_seen_citations(self) -> CitationMapping:
        return self.seen_citations

    @property
    def num_cited_documents(self) -> int:
        return len(self.cited_document_ids)

    def get_next_citation_number(self) -> int:
        if not self.citation_to_doc:
            return 1
        return max(self.citation_to_doc.keys()) + 1


# ---------------------------------------------------------------------------
# Citation utilities
# ---------------------------------------------------------------------------


def update_citation_processor_from_tool_response(
    tool_response: "ToolResponse",  # forward ref
    citation_processor: DynamicCitationProcessor,
    citeable_tool_names: set[str] | None = None,
) -> None:
    """Update citation processor if this was a citeable tool with a SearchDocsResponse."""
    from deep_research.models import ToolResponse as TR, SearchDocsResponse

    if not isinstance(tool_response, TR):
        return
    if tool_response.tool_call is None:
        return

    # Default citeable tools
    if citeable_tool_names is None:
        citeable_tool_names = {"web_search", "internal_search", "open_urls"}

    if tool_response.tool_call.tool_name in citeable_tool_names:
        if isinstance(tool_response.rich_response, SearchDocsResponse):
            search_response = tool_response.rich_response
            citation_to_doc: CitationMapping = {}
            for citation_num, doc_id in search_response.citation_mapping.items():
                matching_doc = next(
                    (doc for doc in search_response.search_docs if doc.document_id == doc_id),
                    None,
                )
                if matching_doc:
                    citation_to_doc[citation_num] = matching_doc
            citation_processor.update_citation_mapping(citation_to_doc)


def extract_citation_order_from_text(text: str) -> list[int]:
    """Extract citation numbers from text in order of first appearance."""
    citation_pattern = re.compile(
        r"([\[【［]{2}(\d+)[\]】］]{2})|([\[【［]([\d]+(?: *, *\d+)*)[\]】］])"
    )
    seen: set[int] = set()
    order: list[int] = []

    for match in citation_pattern.finditer(text):
        if match.group(2):
            nums_str = match.group(2)
        elif match.group(4):
            nums_str = match.group(4)
        else:
            continue

        for num_str in nums_str.split(","):
            num_str = num_str.strip()
            if num_str:
                try:
                    num = int(num_str)
                    if num not in seen:
                        seen.add(num)
                        order.append(num)
                except ValueError:
                    continue
    return order


def collapse_citations(
    answer_text: str,
    existing_citation_mapping: CitationMapping,
    new_citation_mapping: CitationMapping,
) -> tuple[str, CitationMapping]:
    """Collapse citations to use smallest possible numbers, merging with existing mapping."""
    doc_id_to_existing_citation: dict[str, int] = {
        doc.document_id: citation_num
        for citation_num, doc in existing_citation_mapping.items()
    }

    next_citation_num = max(existing_citation_mapping.keys()) + 1 if existing_citation_mapping else 1

    old_to_new: dict[int, int] = {}
    additional_mappings: CitationMapping = {}

    for old_num, search_doc in new_citation_mapping.items():
        doc_id = search_doc.document_id

        if doc_id in doc_id_to_existing_citation:
            old_to_new[old_num] = doc_id_to_existing_citation[doc_id]
        else:
            existing_new_num = None
            for mapped_old, mapped_new in old_to_new.items():
                if (
                    mapped_old in new_citation_mapping
                    and new_citation_mapping[mapped_old].document_id == doc_id
                ):
                    existing_new_num = mapped_new
                    break

            if existing_new_num is not None:
                old_to_new[old_num] = existing_new_num
            else:
                old_to_new[old_num] = next_citation_num
                additional_mappings[next_citation_num] = search_doc
                next_citation_num += 1

    citation_pattern = re.compile(
        r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])"
    )

    def replace_citation(match: re.Match) -> str:  # type: ignore[type-arg]
        citation_str = match.group()
        if (
            citation_str.startswith("[[")
            or citation_str.startswith("【【")
            or citation_str.startswith("［［")
        ):
            open_bracket = citation_str[:2]
            close_bracket = citation_str[-2:]
            content = citation_str[2:-2]
        else:
            open_bracket = citation_str[0]
            close_bracket = citation_str[-1]
            content = citation_str[1:-1]

        new_nums = []
        for num_str in content.split(","):
            num_str = num_str.strip()
            if not num_str:
                continue
            try:
                num = int(num_str)
                if num in old_to_new:
                    new_nums.append(str(old_to_new[num]))
                else:
                    new_nums.append(num_str)
            except ValueError:
                new_nums.append(num_str)

        new_content = ", ".join(new_nums)
        return f"{open_bracket}{new_content}{close_bracket}"

    updated_text = citation_pattern.sub(replace_citation, answer_text)
    combined_mapping: CitationMapping = dict(existing_citation_mapping)
    combined_mapping.update(additional_mappings)

    return updated_text, combined_mapping
