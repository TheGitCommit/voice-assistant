# Server

GPU-accelerated voice processing server: Whisper STT → Llama LLM → Kokoro/Piper TTS.

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (recommended)
- 16GB RAM
- [llama.cpp](https://github.com/ggerganov/llama.cpp) binary + GGUF model
- [Kokoro TTS](https://github.com/hexgrad/kokoro-82M) or [Piper](https://github.com/rhasspy/piper) binary + ONNX voice

## Setup

```bash
cd server
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Configure paths in `.env` (in project root):

```bash
LLAMA_EXE_PATH=C:\path\to\llama-server.exe
LLAMA_MODEL_PATH=C:\path\to\model.gguf

TTS_PROVIDER=kokoro  # or "piper"
KOKORO_VOICE=af_heart
PIPER_EXE_PATH=./server/piper/piper.exe
PIPER_MODEL_PATH=./server/piper/models/en_US-amy-medium.onnx
```

## Run

```bash
python server_main.py
```

The server will:
- Start Llama.cpp server
- Load Whisper model
- Listen on `ws://0.0.0.0:8000/ws/audio`
- Health check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

## Architecture

```
Audio ──▶ VAD (Silero) ──▶ Whisper (STT) ──▶ Llama (LLM) ──▶ TTS ──▶ Audio Out
```

The pipeline processes audio in real-time:
1. **VAD** detects speech boundaries (start/stop)
2. **Whisper** transcribes speech to text
3. **Llama** generates response (streaming tokens)
4. **TTS** synthesizes speech (Kokoro or Piper)
5. Audio streams back to client

## Structure

```
server/
├── core/
│   ├── audio_processor.py   # Pipeline orchestrator
│   └── vad.py               # Silero VAD (speech boundary detection)
├── inference/
│   ├── whisper_stt.py       # Faster Whisper (async transcription)
│   ├── llm_client.py        # Llama.cpp HTTP client + conversation history
│   ├── llama_process_manager.py  # Process lifecycle + health monitor
│   ├── kokoro_tts.py        # Kokoro TTS
│   ├── piper_tts.py         # Piper TTS
│   └── tts_factory.py       # TTS provider factory
├── networking/
│   ├── websocket_server.py  # FastAPI endpoint
│   └── websocket_connection.py  # Connection handler
├── utils/
│   ├── latency_monitor.py   # Performance tracking
│   ├── logging_utils.py     # Rate-limited logging
│   └── retry.py             # Retry decorators
├── config.py                # Server configuration
└── server_main.py           # Entry point
```

## WebSocket Protocol

**Client → Server**
| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary | PCM float32, 16kHz, mono, continuous stream |
| Hello | JSON | `{"type":"hello","sample_rate":16000,"channels":1}` |
| Interrupt | JSON | `{"type":"interrupt"}` - Cancel current pipeline |

**Server → Client**
| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary | PCM16LE, 22050Hz (Piper) or 24000Hz (Kokoro), mono |
| Transcription | JSON | `{"type":"transcription","text":"..."}` |
| Response | JSON | `{"type":"llm_response","text":"..."}` |
| TTS state | JSON | `{"type":"tts_start"}` / `{"type":"tts_stop"}` |

## Features

- **Barge-in**: Client can send `{"type":"interrupt"}` to cancel current pipeline
- **Auto-recovery**: Llama process restarted on crash (5 retries, 30s health check)
- **Retry logic**: LLM (3x exponential), TTS (2x fixed delay)
- **TTS prefetching**: Next sentence synthesized during playback
- **Streaming**: LLM tokens streamed to client as generated
- **Conversation history**: In-memory history maintained during connection

## Config Options

Edit `server/config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `whisper.model_size` | `distil-small.en` | Options: tiny.en, distil-small.en, small.en, medium.en |
| `llama.gpu_layers` | -1 | -1 = all on GPU, reduce if OOM |
| `llama.context_size` | 4096 | Context window size |
| `llama.max_history_turns` | 10 | Conversation turns to keep in memory |
| `vad.speech_threshold` | 0.45 | Lower = more sensitive (0.0-1.0) |
| `vad.silence_frames_required` | 10 | Frames of silence to end utterance (~320ms) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Llama won't start | Check paths in `.env`, verify CUDA, check model file |
| CUDA OOM | Reduce `gpu_layers`, use smaller model |
| Bad transcription | Check mic input, tune VAD threshold |
| Slow response | Use smaller models, increase GPU layers |
| Connection timeout | Increase `request_timeout_seconds` in config |
