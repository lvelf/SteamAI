import os
import asyncio
from functools import lru_cache
import fastapi_poe as fp

MODEL = os.environ.get("POE_MODEL", "GPT-4o-Mini")
POE_API_KEY = os.environ.get("POE_API_KEY", "")

async def transform_game_description(user_input: str) -> str:
    prompt = f"""Rewrite the user's casual game description into ONE concise, classification-friendly sentence.

Requirements:
- MUST include: (1) core genre or gameplay loop, (2) perspective/view (e.g., first-person/third-person/isometric/2D), (3) key mechanics (2-3), (4) setting/theme (1 short phrase), (5) player mode if mentioned (single-player/co-op/PvP).
- Avoid vague marketing words (fun, exciting, amazing, immersive, stunning).
- Do NOT mention any game titles, platform, price, or "Steam".
- Output only the rewritten sentence, no quotes, no prefix.

User input: "{user_input}"
Rewritten:"""

    message = fp.ProtocolMessage(role="user", content=prompt)
    full_response = ""

    async for partial in fp.get_bot_response(
        messages=[message],
        bot_name=MODEL,
        api_key=POE_API_KEY,
    ):
        full_response += partial.text

    response = full_response.strip()

    prefixes = ["Output:", "output:", "Transformed:", "Description:"]
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    return response


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
       
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return asyncio.run(coro)


@lru_cache(maxsize=512)
def transform_sync(user_input: str) -> str:
    if not POE_API_KEY:
        return user_input
    return _run_async(transform_game_description(user_input))