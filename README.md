# Wazir: AGMC local voice agent

A voice assistant for AGMC's used-car website. A customer clicks **Start talking**, asks about cars, and can book a test drive. Every model runs on your own machine: no cloud APIs.

```
Browser mic ──WebRTC──► Silero VAD + Smart-Turn ─► Parakeet STT ─► Qwen3-4B (llama.cpp, Arc GPU) ─► Kokoro TTS ──WebRTC──► Browser speaker
   (echo cancellation)        (CPU)                  (CPU)            streaming tokens + tool call       (CPU)
```

| Stage | Model / tool | Runs on |
|---|---|---|
| Transport | WebRTC (Pipecat SmallWebRTC), browser echo cancellation | – |
| End of speech | Silero VAD (200 ms silence) + Smart-Turn v3 (did they finish the sentence?) | CPU |
| Speech-to-text | NVIDIA Parakeet TDT 0.6B v2, ONNX int8 | CPU |
| LLM | Qwen3-4B-Instruct-2507, 4-bit GGUF, served by llama.cpp (Vulkan) | Intel Arc GPU |
| Text-to-speech | Kokoro-82M, voice `af_heart` (female) | CPU |
| Orchestration | Pipecat 1.14 (barge-in, turn-taking, streaming) | CPU |

**Barge-in:** if you talk while Wazir is speaking, she stops immediately, the LLM request is cancelled, and her unspoken words are dropped from the conversation history.

**Tool:** `book_test_drive(name, phone, car, date_time, branch)` validates the details against the mock inventory and the three branches, then appends the booking to `logs/bookings.jsonl`.

---

## 1. Set up on Windows (once)

You need Python (any recent version: setup downloads its own Python 3.12 for the project) and an up-to-date Intel Arc graphics driver.

```powershell
git clone https://github.com/fatiiiiima/voice-chat-wazir.git
cd voice-chat-wazir
git checkout claude/busy-keller-aqf202
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Setup installs the Python packages, downloads llama.cpp, and downloads the speech models (about 1 GB). It takes a few minutes.

## 2. Run it (two terminals)

**Terminal 1: the LLM on your Arc GPU.** The first run downloads the ~2.5 GB model.
```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_llm.ps1
```
Wait until it prints `server is listening on http://127.0.0.1:8080`.

**Terminal 2: the voice agent.**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_agent.ps1
```
Wait for `Ready: open http://127.0.0.1:7860`, open that address in **Chrome or Edge**, click **Start talking**, and allow the microphone.

If Windows Firewall asks about Python, click **Allow**. WebRTC audio needs it.

## 3. Test latency and send the results

Do all three steps with both terminals from step 2 still running. Open a **third** PowerShell window in the repo folder.

**A. The LLM on its own** (about 1 minute):
```powershell
python -m uv run python scripts/bench_llm.py
```

**B. Talk to Wazir yourself for 2–3 minutes** in the browser. Ask about cars, ask a price, book a test drive, and interrupt her once. The table under the chat fills in for every turn. Then print the summary:
```powershell
python -m uv run python scripts/latency_report.py
```

**C. Simulated callers.** Run one caller, then three at once. Close the browser tab first so it doesn't hear the test audio.
```powershell
python -m uv run python scripts/loadtest.py
python -m uv run python scripts/loadtest.py --sessions 3
python -m uv run python scripts/latency_report.py --last 12
```

**Send back:**
- the output of A, B and C (copy-paste the text);
- your exact CPU model and RAM (Task Manager → Performance → CPU / Memory);
- the file `logs\turns.jsonl`.

Those are enough to see exactly which stage is slow on your hardware.

### Reading the numbers

All times are in milliseconds, measured from **the moment you stopped talking**.

| Column | Meaning |
|---|---|
| VAD | Fixed silence window before we even consider you finished (`VAD_STOP_SECS`, 200 ms) |
| Turn+STT | Smart-Turn deciding you're done, with speech-to-text running in parallel |
| LLM 1st tok | Time until the LLM produces its first word |
| TTS 1st audio | First word to first audio. Near 0 when the reply starts with a pre-rendered opener like "Sure!" |
| Out | Handing audio to WebRTC |
| Server total | Sum of the above |
| **Heard (browser)** | Measured in the browser: your mic going quiet until Wazir's voice actually plays. **This is the number the customer experiences**, and it runs about 150–250 ms above the server total. |

---

## Configuration (`.env`)

Copy `.env.example` to `.env` (setup does this). The useful settings:

| Setting | Default | Notes |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Any OpenAI-compatible server: llama.cpp, OpenVINO Model Server, or vLLM later |
| `VAD_STOP_SECS` | `0.2` | Lower is faster but cuts people off more often |
| `TTS_VOICE` | `af_heart` | Other female voices: `af_bella`, `af_nicole`, `bf_emma` |
| `TTS_THREADS`, `STT_THREADS` | auto | CPU threads per model. Auto is about 40% / 25% of cores, so one caller's TTS can't starve another caller's turn detection |
| `STT_QUANTIZATION` | `int8` | Set it empty for full-precision Parakeet |

To try a different LLM, set `LLM_GGUF` before `start_llm.ps1`, for example `$env:LLM_GGUF="unsloth/Qwen3-1.7B-GGUF:Q4_K_M"`.

## Project layout

```
agent/
  server.py        HTTP + WebRTC signalling, serves the widget
  bot.py           the per-call pipeline (VAD → STT → LLM → TTS) and the book_test_drive tool
  prompts.py       Wazir's system prompt
  tools.py         mock inventory, branches, booking validation
  stt_parakeet.py  Parakeet speech-to-text
  tts_kokoro.py    Kokoro TTS with pre-rendered openers and greeting
  timing.py        per-turn latency logging → logs/turns.jsonl
  models.py        loads models once, shared by all calls
web/index.html     the widget (mic, transcript, latency table)
scripts/           setup / start scripts, LLM benchmark, load test, latency report
deploy/            docker-compose for the future Linux GPU server (vLLM)
tests/             unit tests: python -m uv run pytest
```

## Known limits of this version

- **Concurrency on the laptop:** STT and TTS run on the CPU and share its cores. Expect good latency for 1–2 simultaneous callers. More callers slow down turn detection, which the load test shows. Moving STT/TTS to GPU batching is the scaling phase.
- **Booking-confirmation turns** take one extra LLM round trip (the tool call), so they run slower than normal turns.
- **Not in v1:** price negotiation, a real database, phone/WhatsApp calls, scaling.
