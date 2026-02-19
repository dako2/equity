"""
deep_research — portable deep-research harness.

Quick start (LiteLLM / OpenAI)
-------------------------------

.. code-block:: python

    from deep_research.llm_interface import LiteLLMAdapter, LLMConfig
    from deep_research.dr_loop import run_deep_research
    from deep_research.utils import Emitter, ChatStateContainer
    from deep_research.models import ChatMessageSimple, MessageType

    llm = LiteLLMAdapter(LLMConfig(model_name="gpt-4o"))
    emitter = Emitter()
    state = ChatStateContainer()
    history = [ChatMessageSimple(message="...", token_count=10, message_type=MessageType.USER)]
    run_deep_research(emitter=emitter, state_container=state, simple_chat_history=history,
                      tools=[], llm=llm, token_counter=lambda s: len(s) // 4)

Quick start (xAI Grok with native search)
------------------------------------------

.. code-block:: python

    from deep_research.llm_interface import GrokAdapter, GrokServerTool, LLMConfig
    from deep_research.dr_loop import run_deep_research
    from deep_research.utils import Emitter, ChatStateContainer
    from deep_research.models import ChatMessageSimple, MessageType

    llm = GrokAdapter(
        LLMConfig(model_name="grok-4-1-fast", api_key="xai-..."),
        server_tools=[GrokServerTool.LIVE_SEARCH, GrokServerTool.X_SEARCH],
    )
    emitter = Emitter()
    state = ChatStateContainer()
    history = [ChatMessageSimple(message="...", token_count=10, message_type=MessageType.USER)]
    run_deep_research(emitter=emitter, state_container=state, simple_chat_history=history,
                      tools=[], llm=llm, token_counter=lambda s: len(s) // 4)
"""

from deep_research.dr_loop import run_deep_research  # noqa: F401
from deep_research.llm_interface import (  # noqa: F401
    LLM,
    GrokAdapter,
    GrokServerTool,
    LiteLLMAdapter,
    LLMConfig,
)
from deep_research.utils import ChatStateContainer, Emitter  # noqa: F401

__all__ = [
    "run_deep_research",
    "LLM",
    "GrokAdapter",
    "GrokServerTool",
    "LiteLLMAdapter",
    "LLMConfig",
    "ChatStateContainer",
    "Emitter",
]
