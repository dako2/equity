#!/usr/bin/env python3
"""
Minimal example showing how to run the deep research harness.

Requirements
------------
pip install -r requirements.txt

Set the appropriate API key environment variable for your chosen provider.

Usage
-----
    # Using GPT-4o via LiteLLM (default):
    OPENAI_API_KEY=sk-... python example.py "What are the latest advances in quantum computing?"

    # Using xAI Grok-4-1-fast with native web_search + x_search:
    DR_PROVIDER=grok XAI_API_KEY=xai-... python example.py "What are the latest advances in quantum computing?"

Environment Variables
---------------------
    DR_PROVIDER   : "litellm" (default) or "grok"
    DR_MODEL      : Model name override (default depends on provider)
    OPENAI_API_KEY: Required for LiteLLM / OpenAI
    XAI_API_KEY   : Required for Grok
"""

from __future__ import annotations

import os
import sys
import threading
from queue import Empty

# ---------------------------------------------------------------------------
# 1. Ensure the package is importable when running from this directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from deep_research.dr_loop import run_deep_research  # noqa: E402
from deep_research.llm_interface import (  # noqa: E402
    GrokAdapter,
    GrokServerTool,
    LiteLLMAdapter,
    LLMConfig,
)
from deep_research.models import (  # noqa: E402
    AgentResponseDelta,
    AgentResponseStart,
    ChatMessageSimple,
    DeepResearchPlanDelta,
    DeepResearchPlanStart,
    IntermediateReportDelta,
    IntermediateReportStart,
    MessageType,
    OverallStop,
    Packet,
    ReasoningDelta,
    ReasoningStart,
    ResearchAgentStart,
    SectionEnd,
)
from deep_research.tool_interface import Tool  # noqa: E402
from deep_research.utils import ChatStateContainer, Emitter  # noqa: E402

# ---------------------------------------------------------------------------
# 2. (Optional) Implement a concrete Tool — e.g. a simple web-search stub
# ---------------------------------------------------------------------------
# For a real use-case you would implement Tool subclasses that wrap APIs
# such as Google Search, Bing, or an internal document search.
#
# When using GrokAdapter, the model has *native* server-side web_search and
# x_search — so external tool implementations are optional.  The model will
# autonomously search the web and X (Twitter) via its built-in capabilities.
#
# class WebSearchTool(Tool):
#     @property
#     def id(self) -> int: return 1
#     @property
#     def name(self) -> str: return "web_search"
#     @property
#     def description(self) -> str: return "Search the web"
#     @property
#     def display_name(self) -> str: return "Web Search"
#     def tool_definition(self) -> dict: ...
#     def run(self, placement, override_kwargs, **llm_kwargs) -> ToolResponse: ...

# ---------------------------------------------------------------------------
# 3. Configure and run
# ---------------------------------------------------------------------------


def simple_token_counter(text: str) -> int:
    """Rough token estimator (~4 chars per token)."""
    return max(1, len(text) // 4)


def drain_emitter(emitter: Emitter) -> None:
    """Background thread that prints packets as they arrive."""
    while True:
        try:
            pkt: Packet = emitter.bus.get(timeout=1.0)
        except Empty:
            continue

        obj = pkt.obj

        if isinstance(obj, DeepResearchPlanStart):
            print("\n📋 Research Plan:\n", end="", flush=True)
        elif isinstance(obj, DeepResearchPlanDelta):
            print(obj.content, end="", flush=True)
        elif isinstance(obj, ResearchAgentStart):
            print(f"\n\n🔍 Research Agent: {obj.research_task}", flush=True)
        elif isinstance(obj, IntermediateReportStart):
            print("\n📝 Intermediate Report:\n", end="", flush=True)
        elif isinstance(obj, IntermediateReportDelta):
            print(obj.content, end="", flush=True)
        elif isinstance(obj, ReasoningStart):
            print("\n💭 Reasoning: ", end="", flush=True)
        elif isinstance(obj, ReasoningDelta):
            print(obj.reasoning, end="", flush=True)
        elif isinstance(obj, AgentResponseStart):
            print("\n\n📄 Final Report:\n", end="", flush=True)
        elif isinstance(obj, AgentResponseDelta):
            print(obj.content, end="", flush=True)
        elif isinstance(obj, SectionEnd):
            print("\n--- section end ---", flush=True)
        elif isinstance(obj, OverallStop):
            print("\n\n✅ Done!", flush=True)
            return


def build_llm(provider: str) -> GrokAdapter | LiteLLMAdapter:
    """Build the LLM adapter based on the chosen provider."""
    if provider == "grok":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            print("❌ XAI_API_KEY environment variable is required for Grok.", file=sys.stderr)
            sys.exit(1)

        model = os.getenv("DR_MODEL", "grok-4-1-fast")
        config = LLMConfig(
            model_name=model,
            model_provider="xai",
            api_key=api_key,
            api_base="https://api.x.ai/v1",
            max_input_tokens=131_072,
        )
        # Enable both native server-side tools: web search + X search
        llm = GrokAdapter(
            config,
            server_tools=[GrokServerTool.LIVE_SEARCH, GrokServerTool.X_SEARCH],
        )
        print(f"  Provider : xAI Grok ({model})")
        print(f"  Endpoint : {config.api_base}")
        print(f"  Native tools : web_search ✓, x_search ✓")
        return llm
    else:
        # Default: LiteLLM (works with OpenAI, Anthropic, etc.)
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("DR_MODEL", "gpt-4o")
        config = LLMConfig(
            model_name=model,
            api_key=api_key,
        )
        llm = LiteLLMAdapter(config)
        print(f"  Provider : LiteLLM ({model})")
        return llm


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest advances in quantum computing?"
    provider = os.getenv("DR_PROVIDER", "litellm").lower()

    print(f"🚀 Running deep research for: {query!r}\n")

    # --- LLM ----------------------------------------------------------
    llm = build_llm(provider)

    # --- State --------------------------------------------------------
    emitter = Emitter()
    state = ChatStateContainer()
    history: list[ChatMessageSimple] = [
        ChatMessageSimple(
            message=query,
            token_count=simple_token_counter(query),
            message_type=MessageType.USER,
        )
    ]

    # --- Tools --------------------------------------------------------
    # When using GrokAdapter, the model has native server-side web_search
    # and x_search — no external tool implementations needed for search.
    # For LiteLLM you would add your own Tool implementations here.
    tools: list[Tool] = []

    # --- Run ----------------------------------------------------------
    print()
    reader = threading.Thread(target=drain_emitter, args=(emitter,), daemon=True)
    reader.start()

    try:
        run_deep_research(
            emitter=emitter,
            state_container=state,
            simple_chat_history=history,
            tools=tools,
            llm=llm,
            token_counter=simple_token_counter,
            skip_clarification=True,
        )
    except Exception as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        raise

    reader.join(timeout=5)
    print()


if __name__ == "__main__":
    main()
