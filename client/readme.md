# Client

Lightweight voice client for Raspberry Pi or any machine with mic/speaker.

## Requirements

- Python 3.10+
- PortAudio (`sudo apt install portaudio19-dev` on Linux)
- Microphone + speakers

## Setup

```bash
cd client
python -m venv venv && source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Configure in `.env`:
```bash
SERVER_URL=ws://192.168.1.100:8000/ws/audio

# Optional: cloud fallback
ENABLE_CLOUD_FALLBACK=true
DEEPGRAM_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Run

```bash
python client_main.py
```

## Structure

```
client/
├── audio/
│   ├── audio_capture.py   # Microphone input (sounddevice)
│   ├── audio_playback.py  # Speaker output + stop
│   ├── feedback.py        # Beep tones (listening/processing)
│   └── vad.py             # WebRTC VAD (client-side)
├── cloud_fallback/
│   ├── cloud_processor.py # Cloud mode orchestrator
│   ├── deepgram_stt.py    # Deepgram STT
│   ├── deepgram_tts.py    # Deepgram TTS
│   └── deepseek_llm.py    # DeepSeek LLM
├── websocket_client.py    # Main client + barge-in
└── config.py
```

## How It Works

```
Microphone → VAD → WebSocket → Server → TTS Audio → Speaker
                       ↑
              Barge-in interrupt
```

1. **Capture**: Mic audio in 20ms chunks
2. **VAD**: Detects speech start/end
3. **Stream**: Sends audio to server via WebSocket
4. **Receive**: Gets TTS audio + events
5. **Play**: Outputs to speaker

## Barge-in

Speak during playback to interrupt:

1. VAD detects your voice while TTS is playing
2. Client stops speaker, sends `{"type":"interrupt"}`
3. Server cancels current pipeline
4. Your words become the next query

## Audio Feedback

| Tone | Meaning |
|------|---------|
| High beep (880Hz) | Listening started |
| Low beep (440Hz) | Processing your speech |

## Cloud Fallback

When server is unreachable, auto-switches to:
- **STT**: Deepgram Nova-2
- **LLM**: DeepSeek Chat
- **TTS**: Deepgram Aura

See `cloud_fallback/architecture.md` for details.

## Config Options

| Setting | Default | Notes |
|---------|---------|-------|
| `capture.sample_rate` | 16000 | Must match server |
| `vad.aggressiveness` | 2 | 0-3, higher = less sensitive |
| `playback.sample_rate` | 22050 | Must match server TTS |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No audio input | Check mic permissions, install PortAudio |
| Connection failed | Verify `SERVER_URL`, check server is running |
| Audio choppy | Increase buffer size, check network |
| Barge-in not working | VAD may not be detecting speech—lower aggressiveness |

