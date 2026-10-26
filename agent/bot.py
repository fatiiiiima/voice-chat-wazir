"""One voice session: WebRTC audio in -> VAD/turn -> STT -> LLM -> TTS -> WebRTC audio out."""

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from agent import models  # noqa: F401  (installs the shared-Kokoro cache)
from agent.config import settings
from agent.prompts import GREETING, build_system_prompt
from agent.stt_parakeet import ParakeetSTTService
from agent.timing import TurnTimingObserver
from agent.tts_kokoro import CachedKokoroTTSService
from agent.tools import book_test_drive_impl


async def book_test_drive(
    params: FunctionCallParams, name: str, phone: str, car: str, date_time: str, branch: str
):
    """Book a test drive for the customer. Only call this after the customer has confirmed all details.

    Args:
        name: The customer's full name.
        phone: The customer's phone number, digits only.
        car: The car from the inventory, e.g. "2022 BMW X3 xDrive30i".
        date_time: Explicit date and time, e.g. "Tuesday 28 October 2026 at 17:00".
        branch: One of "AGMC Sheikh Zayed Road", "AGMC Motor City", "AGMC Sharjah".
    """
    result = book_test_drive_impl(name, phone, car, date_time, branch)
    await params.result_callback(result)


def make_tts() -> CachedKokoroTTSService:
    return CachedKokoroTTSService(
        extra_phrases=[GREETING],
        settings=CachedKokoroTTSService.Settings(voice=settings.tts_voice, speed=settings.tts_speed)
    )


def make_llm() -> OpenAILLMService:
    return OpenAILLMService(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        settings=OpenAILLMService.Settings(
            model=settings.llm_model,
            system_instruction=build_system_prompt(),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ),
    )


def make_context() -> LLMContext:
    return LLMContext(tools=[book_test_drive])


async def run_bot(connection: SmallWebRTCConnection):
    session_id = connection.pc_id
    logger.info(f"[{session_id}] starting session")

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = ParakeetSTTService()

    llm = make_llm()
    tts = make_tts()
    context = make_context()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Silero decides "speech stopped" after vad_stop_secs of silence, then the
            # default Smart-Turn v3 model decides whether the user actually finished.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=settings.vad_stop_secs)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker: PipelineWorker | None = None

    async def send_to_client(message: dict):
        if worker:
            await worker.queue_frame(OutputTransportMessageUrgentFrame(message=message))

    timing = TurnTimingObserver(session_id, settings.log_dir, send=send_to_client)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[timing],
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[{session_id}] client connected")
        # A fixed greeting needs no LLM call, so the user hears Wazir immediately.
        await worker.queue_frame(TTSSpeakFrame(GREETING))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[{session_id}] client disconnected")
        await worker.cancel()

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        if isinstance(message, dict) and message.get("type") == "client-latency":
            timing.add_client_latency(float(message.get("ms", 0)), message.get("turn"))

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    logger.info(f"[{session_id}] session ended")
