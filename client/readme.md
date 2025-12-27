# Client

Lightweight voice client for Raspberry Pi or any machine with mic/speaker. Streams audio to the server and plays responses.

## Requirements

- Python 3.10+
- PortAudio (`sudo apt install portaudio19-dev` on Linux)
- Microphone + speakers
- Network connection to server

## Setup

```bash
cd client
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Configure in `.env` (in project root):

```bash
# Server URL (default: ws://localhost:8000/ws/audio)
SERVER_URL=ws://192.168.1.100:8000/ws/audio

# Optional: cloud fallback
ENABLE_CLOUD_FALLBACK=true
DEEPGRAM_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

## Run

```bash
python client_main.py
```

The client will:
1. Check server health
2. Connect via WebSocket or fall back to cloud mode
3. Start capturing audio and streaming to server
4. Play TTS responses through speakers

## How It Works

```
Microphone → VAD → WebSocket → Server → TTS Audio → Speaker
                       ↑
              Barge-in interrupt
```

1. **Capture**: Microphone audio in 20ms chunks
2. **VAD**: Client-side VAD detects speech (for barge-in)
3. **Stream**: Sends audio to server via WebSocket (always-on)
4. **Receive**: Gets TTS audio + events from server
5. **Play**: Outputs audio to speaker

## Features

- **Always-on streaming**: Continuously streams audio to server
- **Barge-in**: Speak during playback to interrupt the assistant
- **Audio feedback**: Plays tones to indicate system state
- **Cloud fallback**: Auto-switches to cloud APIs if server unavailable

## Structure

```
client/
├── audio/
│   ├── audio_capture.py   # Microphone input (sounddevice)
│   ├── audio_playback.py  # Speaker output + stop support
│   ├── feedback.py        # Beep tones (listening/processing)
│   └── vad.py             # WebRTC VAD (client-side barge-in)
├── cloud_fallback/
│   ├── cloud_processor.py # Cloud mode orchestrator
│   ├── deepgram_stt.py    # Deepgram STT client
│   ├── deepgram_tts.py    # Deepgram TTS client
│   └── deepseek_llm.py    # DeepSeek LLM client
├── config/
│   └── config.py          # Client configuration
├── client_main.py         # Entry point
└── websocket_client.py    # WebSocket streaming + barge-in
```

## Barge-in

Speak during playback to interrupt:

1. Client-side VAD detects your voice while TTS is playing
2. Client stops speaker immediately
3. Client sends `{"type":"interrupt"}` to server
4. Server cancels current LLM/TTS pipeline
5. Your speech becomes the next query

## Audio Feedback

| Tone | Meaning |
|------|---------|
| High beep (880Hz) | Listening started (VAD detected speech) |
| Low beep (440Hz) | Processing your speech (transcription received) |

## Cloud Fallback

When server is unreachable, auto-switches to:
- **STT**: Deepgram Nova-2
- **LLM**: DeepSeek Chat
- **TTS**: Deepgram Aura

Configure API keys in `.env` to enable.

## Config Options

Edit `client/config/config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `capture.sample_rate` | 16000 | Must match server |
| `vad.aggressiveness` | 2 | 0-3, higher = less sensitive |
| `playback.sample_rate` | 24000 | Must match server TTS output |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No audio input | Check mic permissions, install PortAudio, list devices: `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| Connection failed | Verify `SERVER_URL`, check server is running, check firewall |
| Audio choppy | Check network latency, verify sample rates match |
| Barge-in not working | VAD may not detect speech—lower `vad.aggressiveness` |
| No audio output | Check speaker/headphone connection, verify audio device |
