# Server

GPU-accelerated voice processing server: Whisper STT → Llama LLM → Piper TTS.

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA
- 16GB RAM
- [llama.cpp](https://github.com/ggerganov/llama.cpp) binary + GGUF model
- [Piper](https://github.com/rhasspy/piper) binary + ONNX voice

## Setup

```bash
cd server
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Configure paths in `.env`:
```bash
LLAMA_EXE_PATH=C:\path\to\llama-server.exe
LLAMA_MODEL_PATH=C:\path\to\model.gguf
PIPER_EXE_PATH=./piper/piper.exe
PIPER_MODEL_PATH=./piper/models/en_US-amy-medium.onnx
```

## Run

```bash
python server_main.py
# WebSocket: ws://localhost:8000/ws/audio
# Health:    http://localhost:8000/health
```

## Structure

```
server/
├── core/
│   ├── audio_processor.py   # Pipeline: VAD → STT → LLM → TTS
│   └── vad.py               # Silero VAD (speech boundary detection)
├── inference/
│   ├── whisper_stt.py       # Faster Whisper (async transcription)
│   ├── llm_client.py        # Llama.cpp HTTP client + history
│   ├── llama_process_manager.py  # Process lifecycle + health monitor
│   └── piper_tts.py         # Piper subprocess
├── networking/
│   ├── websocket_server.py  # FastAPI endpoint
│   └── websocket_connection.py
├── utils/
│   ├── retry.py             # Retry decorators
│   └── timing.py            # Latency tracking
├── sessions/                # Saved conversations (JSON)
└── config.py
```

## WebSocket Protocol

**Client → Server**
| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary | PCM float32, 16kHz, mono, 20ms chunks |
| Hello | JSON | `{"type":"hello","sample_rate":16000}` |
| Interrupt | JSON | `{"type":"interrupt"}` |
| Load session | JSON | `{"type":"load_session","session_id":"..."}` |

**Server → Client**
| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary | PCM16LE, 22050Hz, mono |
| Transcription | JSON | `{"type":"transcription","text":"..."}` |
| Response | JSON | `{"type":"llm_response","text":"..."}` |
| TTS state | JSON | `{"type":"tts_start"}` / `{"type":"tts_stop"}` |

## Features

- **Barge-in**: Interrupt via `{"type":"interrupt"}` cancels pipeline
- **Auto-recovery**: Llama process restarted on crash (5 retries, 30s health check)
- **Retry**: LLM (3x exponential), TTS (2x fixed)
- **Prefetch**: Next TTS sentence synthesized during playback
- **Persistence**: Sessions saved to `sessions/{id}.json`

## Config Options

| Setting | Default | Notes |
|---------|---------|-------|
| `whisper.model_size` | `small.en` | tiny/small/medium/large-v3 |
| `llama.gpu_layers` | -1 | -1 = all on GPU |
| `llama.context_size` | 8192 | Context window |
| `vad.speech_threshold` | 0.45 | Lower = more sensitive |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Llama won't start | Check paths in `.env`, verify CUDA |
| CUDA OOM | Reduce `gpu_layers`, use smaller model |
| Bad transcription | Check mic, tune VAD threshold |
| Slow response | Use smaller models, increase GPU layers |
