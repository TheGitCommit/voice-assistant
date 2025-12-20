# Voice Assistant

Local-first voice assistant with real-time streaming. All inference runs on your hardware—no cloud required.

## Architecture

```
┌─────────────────┐         WebSocket          ┌─────────────────────────────────┐
│     Client      │◄──────────────────────────►│            Server               │
│                 │                            │                                 │
│  ┌───────────┐  │   Audio chunks (binary)    │  ┌─────┐  ┌─────┐  ┌─────────┐  │
│  │ Microphone├──┼───────────────────────────►│  │ VAD ├──► STT ├──► Llama   │  │
│  └───────────┘  │                            │  │     │  │     │  │ (LLM)   │  │
│                 │   TTS audio (binary)       │  └─────┘  └─────┘  └────┬────┘  │
│  ┌───────────┐  │◄───────────────────────────┤                        │        │
│  │  Speaker  │◄─┤   Events (JSON)            │  ┌─────────────────────▼────┐   │
│  └───────────┘  │◄──────────────────────────►│  │        Piper TTS         │   │
└─────────────────┘                            │  └──────────────────────────┘   │
                                               └─────────────────────────────────┘
```

**Client** (lightweight, runs on Raspberry Pi):
- Captures microphone audio
- Streams to server via WebSocket
- Plays TTS responses

**Server** (GPU machine):
- Silero VAD: Detects speech boundaries
- Faster Whisper: Speech-to-text
- Llama.cpp: LLM inference
- Piper: Text-to-speech

## Features

- **Streaming pipeline**: LLM tokens → sentence detection → TTS → audio playback (low latency)
- **Barge-in support**: Interrupt responses mid-playback by speaking
- **Audio feedback**: Audible cues for listening/processing states
- **Conversation persistence**: Sessions saved to disk, restored on reconnect
- **TTS prefetching**: Synthesizes next sentence while current one plays
- **Auto-recovery**: Llama server auto-restarts on crash
- **Retry logic**: Transient failures retry automatically
- **Cloud fallback**: Auto-switches to Deepgram + DeepSeek when server is offline
- **Performance metrics**: Tracks STT/LLM/TTS latency per session

## Prerequisites

### Server
- Python 3.10+
- CUDA-capable GPU (recommended)
- [Llama.cpp server](https://github.com/ggerganov/llama.cpp) binary
- GGUF model file
- [Piper TTS](https://github.com/rhasspy/piper) binary + voice model

### Client
- Python 3.10+
- PortAudio (`sudo apt install portaudio19-dev` on Linux)
- Microphone and speakers

## Installation

### Server

```bash
cd server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install fastapi uvicorn websockets httpx python-dotenv
pip install faster-whisper torch silero-vad numpy

# Configure paths (see Configuration section)
cp .env.example .env
# Edit .env with your paths
```

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

# Piper TTS (optional, defaults to ./server/piper/)
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
| `context_size` | 8192 | LLM context window |
| `max_history_turns` | 10 | Conversation turns to keep |
| `speech_threshold` | 0.45 | VAD sensitivity (lower = more sensitive) |
| `silence_frames_required` | 10 | Frames of silence to end utterance (~320ms) |

### Whisper Models

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny.en` | ~1GB | Fastest | Lower |
| `small.en` | ~2GB | Fast | Good |
| `medium.en` | ~5GB | Medium | Better |
| `large-v3` | ~10GB | Slow | Best |

Set in `server/config.py`:
```python
"whisper": WhisperConfig(model_size="small.en"),
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

Console output:
```
You: What's the weather like?
Assistant: I don't have access to real-time weather data, but I can help you find...
```

### Barge-in (Interruption)

You can interrupt the assistant mid-response by speaking. Say anything—"stop", "wait", "actually..."—and:

1. **Playback stops immediately** (client clears audio buffer)
2. **Server cancels** the current LLM/TTS pipeline
3. **Your interruption is processed** as a new query

```
Assistant: [speaking] "The capital of France is Paris, which is known for..."
You: "Stop"
  → [audio stops]
  → [processes "stop" as new input]
Assistant: "Okay, stopping."
```

**Note**: Whatever you say to interrupt becomes the next query. If you just want silence, you'll need to say something like "be quiet" or "stop talking".

### Audio Feedback

The client plays short tones to indicate state:
- **High beep (880Hz)**: System detected your voice, listening
- **Lower beep (440Hz)**: Transcription received, processing

## Session Persistence

Conversations are automatically saved and can be restored:

### Auto-save
Sessions save to `sessions/{connection_id}.json` when:
- WebSocket disconnects
- After each LLM response

### Restore a Session
Send a session ID in the hello message:
```json
{"type": "hello", "session_id": "my-session-123"}
```

Or request it later:
```json
{"type": "load_session", "session_id": "my-session-123"}
```

### Session Files
```
sessions/
├── abc123.json      # Auto-generated IDs
├── my-session.json  # Custom session IDs
└── ...
```

Each file contains the full conversation history in JSON format.

## Project Structure

```
voice-assistant/
├── client/
│   ├── audio/
│   │   ├── audio_capture.py    # Microphone input
│   │   ├── audio_playback.py   # Speaker output + stop support
│   │   ├── feedback.py         # Audio feedback tones
│   │   └── vad.py              # WebRTC VAD (client-side)
│   ├── cloud_fallback/
│   │   ├── cloud_processor.py  # Cloud mode orchestrator
│   │   ├── deepgram_stt.py     # Deepgram STT client
│   │   ├── deepgram_tts.py     # Deepgram TTS client
│   │   └── deepseek_llm.py     # DeepSeek LLM client
│   ├── client_main.py          # Entry point
│   ├── config.py               # Client configuration
│   └── websocket_client.py     # WebSocket streaming + barge-in
│
├── server/
│   ├── core/
│   │   ├── audio_processor.py  # Pipeline + interruption + prefetch
│   │   └── vad.py              # Silero VAD
│   ├── inference/
│   │   ├── llama_process_manager.py  # Lifecycle + health monitor
│   │   ├── llm_client.py       # LLM client + history persistence
│   │   ├── piper_tts.py        # Piper TTS + retry
│   │   └── whisper_stt.py      # Faster Whisper + async
│   ├── networking/
│   │   ├── websocket_server.py     # FastAPI endpoint
│   │   └── websocket_connection.py # Connection + interrupt handler
│   ├── utils/
│   │   ├── logging_utils.py    # Rate-limited logging
│   │   ├── retry.py            # Retry decorators
│   │   └── timing.py           # Performance tracking
│   ├── config.py               # Server configuration
│   └── server_main.py          # Entry point
│
├── sessions/                   # Saved conversation history
├── PERFORMANCE.md              # Latency measurement docs
└── readme.md
```

## Performance

Typical latencies on RTX 3080 + Ryzen 5800X:

| Component | Time |
|-----------|------|
| VAD | ~5ms |
| STT (small.en) | 300-500ms |
| LLM (8B Q4) | 800-1500ms |
| TTS | 50-100ms |
| **End-to-end** | **1.5-2.5s** |

See [PERFORMANCE.md](PERFORMANCE.md) for measurement details.

### Optimization Tips

1. **Use smaller models** for faster response
2. **Increase `gpu_layers`** to offload more to GPU
3. **Lower `beam_size`** in Whisper config (trades accuracy for speed)
4. **Tune VAD thresholds** to reduce false triggers

### TTS Prefetching

The pipeline synthesizes the next sentence while the current one is playing:

```
Sentence 1: [TTS]───►[Streaming to client]
Sentence 2:      [TTS]───────────────────►[Streaming]
Sentence 3:           [TTS]───────────────────────────►...
```

This reduces perceived gaps between sentences. The prefetch queue holds up to 2 sentences ahead.

## Cloud Fallback

When the local server is unreachable, the client auto-switches to cloud APIs:

- **STT/TTS**: Deepgram (Nova-2 / Aura)
- **LLM**: DeepSeek Chat

Configure in `.env`:
```bash
ENABLE_CLOUD_FALLBACK=true
DEEPGRAM_API_KEY=...
DEEPSEEK_API_KEY=...
```

The client checks server health every connection attempt and falls back automatically.

## Reliability Features

### Llama Auto-Recovery

The server monitors the Llama.cpp process every 30 seconds. If it crashes:

1. Detects via HTTP health check failure
2. Attempts restart (up to 5 retries)
3. Logs status to console

No manual intervention needed for transient crashes.

### Retry Logic

Transient failures retry automatically:

| Component | Max Retries | Backoff |
|-----------|-------------|---------|
| LLM API | 3 | Exponential (1s, 2s, 4s) |
| TTS | 2 | Fixed 0.5s |

Only retries on connection/timeout errors, not on bad requests.

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
- Use smaller Whisper model
- Use quantized LLM (Q4_K_M recommended)
- Ensure GPU is being used (`device: cuda` in config)

### VAD cuts off speech
- Lower `speech_threshold` (more sensitive)
- Increase `silence_frames_required` (longer pause needed)

## License

MIT
