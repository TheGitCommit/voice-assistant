# Voice Assistant

Local-first voice assistant with real-time streaming. All inference runs on your hardware—no cloud required.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Client                               │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────┐                            │
│  │ Microphone  │───▶│   VAD    │───▶ WebSocket ────────────▶│
│  └─────────────┘    │(WebRTC)  │                            │
│                     │          │                            │
│  ┌─────────────┐    │ Barge-in │                            │
│  │   Speaker   │◄───│ detection│◄─── TTS Audio ─────────────│
│  └─────────────┘    └──────────┘                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Local Server (GPU)                        │
├──────────────────────────────────────────────────────────────┤
│  Audio ──▶ VAD ──▶ Whisper ──▶ Llama ──▶ TTS ──▶ Audio Out │
│            (Silero)   (STT)      (LLM)    (Kokoro/Piper)    │
└──────────────────────────────────────────────────────────────┘
```

## Local Mode Pipeline

```
Audio In ──▶ VAD ──▶ Whisper ──▶ Llama ──▶ TTS ──▶ Audio Out
            (Silero)   (STT)      (LLM)    (Kokoro/Piper)
```

The server processes audio in real-time:
1. **VAD** detects when you start/stop speaking
2. **Whisper** transcribes speech to text
3. **Llama** generates response (streaming)
4. **TTS** synthesizes speech from text
5. Audio streams back to client

## Cloud Fallback

When the local server is unreachable, the client automatically switches to cloud APIs:
- **STT/TTS**: Deepgram (Nova-2 / Aura)
- **LLM**: DeepSeek Chat

## Features

- **Streaming pipeline**: Low-latency real-time processing
- **Barge-in support**: Interrupt responses mid-playback by speaking
- **Audio feedback**: Audible cues for listening/processing states
- **TTS prefetching**: Synthesizes next sentence while current one plays
- **Auto-recovery**: Llama server auto-restarts on crash
- **Retry logic**: Transient failures retry automatically
- **Cloud fallback**: Auto-switches to cloud when server is offline

## Prerequisites

### Server
- Python 3.10+
- CUDA-capable GPU (recommended)
- [Llama.cpp server](https://github.com/ggerganov/llama.cpp) binary
- GGUF model file
- [Kokoro TTS](https://github.com/hexgrad/kokoro-82M) or [Piper TTS](https://github.com/rhasspy/piper)

### Client
- Python 3.10+
- PortAudio (`sudo apt install portaudio19-dev` on Linux)
- Microphone and speakers

## Installation

### Server

```bash
cd server
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Configure paths in `.env` (see Configuration section below).

### Client

```bash
cd client
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
# === Server Config ===

# Llama.cpp server
LLAMA_EXE_PATH=C:\path\to\llama-server.exe
LLAMA_MODEL_PATH=C:\path\to\model.gguf

# TTS (Kokoro or Piper)
TTS_PROVIDER=kokoro  # or "piper"
KOKORO_VOICE=af_heart
PIPER_EXE_PATH=./server/piper/piper.exe
PIPER_MODEL_PATH=./server/piper/models/en_US-amy-medium.onnx

# === Client Config ===

# Server URL (default: ws://localhost:8000/ws/audio)
SERVER_URL=ws://192.168.1.100:8000/ws/audio

# === Cloud Fallback (optional) ===

ENABLE_CLOUD_FALLBACK=true
DEEPGRAM_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

### Server Parameters

Edit `server/config.py` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_layers` | -1 | GPU layers for Llama (-1 = all) |
| `context_size` | 4096 | LLM context window |
| `max_history_turns` | 10 | Conversation turns to keep in memory |
| `speech_threshold` | 0.45 | VAD sensitivity (lower = more sensitive) |

### Whisper Models

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny.en` | ~1GB | Fastest | Lower |
| `distil-small.en` | ~1.5GB | Fast | Good |
| `small.en` | ~2GB | Medium | Good |
| `medium.en` | ~5GB | Slow | Better |

Set in `server/config.py`:
```python
"whisper": WhisperConfig(model_size="distil-small.en"),
```

## Usage

### Start Server

```bash
cd server
python server_main.py
```

The server will:
1. Start the Llama.cpp inference server
2. Load Whisper model
3. Listen on `ws://0.0.0.0:8000/ws/audio`

### Start Client

```bash
cd client
python client_main.py
```

The client will:
1. Check server health
2. Fall back to cloud mode if server unreachable
3. Start capturing audio and streaming

### Interaction

Just speak naturally. The system will:
1. Detect when you start/stop speaking
2. Transcribe your speech
3. Generate a response
4. Play audio back

### Barge-in (Interruption)

You can interrupt the assistant mid-response by speaking. Say anything—"stop", "wait", "actually..."—and:

1. **Playback stops immediately** (client clears audio buffer)
2. **Server cancels** the current LLM/TTS pipeline
3. **Your interruption is processed** as a new query

### Audio Feedback

The client plays short tones to indicate state:
- **High beep (880Hz)**: System detected your voice, listening
- **Lower beep (440Hz)**: Transcription received, processing

## Project Structure

```
voice-assistant/
├── client/
│   ├── audio/
│   │   ├── audio_capture.py    # Microphone input
│   │   ├── audio_playback.py   # Speaker output
│   │   ├── feedback.py         # Audio feedback tones
│   │   └── vad.py              # WebRTC VAD (client-side)
│   ├── cloud_fallback/
│   │   ├── cloud_processor.py  # Cloud mode orchestrator
│   │   ├── deepgram_stt.py     # Deepgram STT client
│   │   ├── deepgram_tts.py     # Deepgram TTS client
│   │   └── deepseek_llm.py     # DeepSeek LLM client
│   ├── client_main.py          # Entry point
│   ├── config/                 # Client configuration
│   └── websocket_client.py     # WebSocket streaming + barge-in
│
├── server/
│   ├── core/
│   │   ├── audio_processor.py  # Pipeline orchestrator
│   │   └── vad.py              # Silero VAD
│   ├── inference/
│   │   ├── llama_process_manager.py  # Llama lifecycle + health monitor
│   │   ├── llm_client.py       # LLM client + conversation history
│   │   ├── kokoro_tts.py       # Kokoro TTS
│   │   ├── piper_tts.py        # Piper TTS
│   │   ├── tts_factory.py      # TTS provider factory
│   │   └── whisper_stt.py      # Faster Whisper STT
│   ├── networking/
│   │   ├── websocket_server.py     # FastAPI endpoint
│   │   └── websocket_connection.py # Connection handler
│   ├── utils/
│   │   ├── latency_monitor.py  # Performance tracking
│   │   ├── logging_utils.py    # Rate-limited logging
│   │   └── retry.py            # Retry decorators
│   ├── config.py               # Server configuration
│   └── server_main.py          # Entry point
│
└── readme.md
```

## Performance

Typical latencies on RTX 3080 + Ryzen 5800X:

| Component | Time |
|-----------|------|
| VAD | ~5ms |
| STT (distil-small.en) | 300-500ms |
| LLM (8B Q4) | 800-1500ms |
| TTS | 50-100ms |
| **End-to-end** | **1.5-2.5s** |

### Optimization Tips

1. **Use smaller models** for faster response
2. **Increase `gpu_layers`** to offload more to GPU
3. **Use `distil-small.en`** Whisper model (faster than `small.en`)
4. **Tune VAD thresholds** to reduce false triggers

## Reliability Features

### Llama Auto-Recovery

The server monitors the Llama.cpp process every 30 seconds. If it crashes:
1. Detects via HTTP health check failure
2. Attempts restart (up to 5 retries)
3. Logs status to console

### Retry Logic

Transient failures retry automatically:

| Component | Max Retries | Backoff |
|-----------|-------------|---------|
| LLM API | 3 | Exponential (1s, 2s, 4s) |
| TTS | 2 | Fixed 0.5s |

## Troubleshooting

### "Llama server not running"
- Check `LLAMA_EXE_PATH` and `LLAMA_MODEL_PATH` in `.env`
- Ensure the model file exists and is a valid GGUF
- Increase `startup_delay_seconds` if model is large

### No audio input
- Check microphone permissions
- Install PortAudio: `sudo apt install portaudio19-dev`
- List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### High latency
- Use smaller Whisper model (e.g., `distil-small.en`)
- Use quantized LLM (Q4_K_M recommended)
- Ensure GPU is being used (`device: cuda` in config)

### VAD cuts off speech
- Lower `speech_threshold` (more sensitive)
- Increase `silence_frames_required` (longer pause needed)

## License

MIT
