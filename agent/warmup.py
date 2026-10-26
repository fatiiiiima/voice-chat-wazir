"""Pre-fill the LLM server's prompt cache before the first caller arrives.

llama.cpp keeps a separate cache per slot (one slot per simultaneous
conversation). An uncached slot must process Wazir's whole system prompt and
tool definitions first, which takes ~2 s on the Arc iGPU, versus ~60 ms when
cached. We send one tiny request per slot, built by Pipecat exactly as a real
call would be, so every slot holds the prompt from the start.
"""

import asyncio
import time

from loguru import logger

from agent.bot import make_context, make_llm
from agent.config import settings


async def warm_llm_cache(slots: int = settings.llm_slots) -> None:
    llm = make_llm()
    context = make_context()
    context.add_message({"role": "user", "content": "Hi"})
    invocation = await llm.get_llm_adapter().get_llm_invocation_params(
        context,
        system_instruction=llm._settings.system_instruction,
        convert_developer_to_user=not llm.supports_developer_role,
    )
    params = llm.build_chat_completion_params(invocation)
    params.update(stream=False, max_tokens=1)
    params.pop("stream_options", None)

    t0 = time.perf_counter()
    # Concurrent requests land on different slots.
    await asyncio.gather(*(llm._client.chat.completions.create(**params) for _ in range(slots)))
    logger.info(f"LLM prompt cache warmed on {slots} slots in {time.perf_counter() - t0:.1f}s")
