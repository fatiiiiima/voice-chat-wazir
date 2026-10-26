"""HTTP server: serves the web widget and does WebRTC signalling.

Run with:  uv run python -m agent.server
Then open: http://127.0.0.1:7860
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from agent.bot import run_bot
from agent.config import settings
from agent.models import preload

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

webrtc_handler = SmallWebRTCRequestHandler()
sessions: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading speech models (first run downloads them, this can take a few minutes)...")
    await asyncio.to_thread(preload)
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.llm_base_url}/models")
            logger.info(f"LLM server reachable at {settings.llm_base_url}: {r.status_code}")
    except Exception as e:
        logger.warning(f"LLM server NOT reachable at {settings.llm_base_url} ({e}). Start it first.")
    logger.info(f"Ready: open http://{settings.host}:{settings.port} in Chrome or Edge")
    yield
    await webrtc_handler.close()
    for task in sessions:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/api/offer")
async def offer(request: Request):
    body = await request.json()

    async def on_connection(connection: SmallWebRTCConnection):
        task = asyncio.create_task(run_bot(connection))
        sessions.add(task)
        task.add_done_callback(sessions.discard)

    return await webrtc_handler.handle_web_request(
        SmallWebRTCRequest.from_dict(body), on_connection
    )


def main():
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
