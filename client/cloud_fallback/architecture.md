# Voice Assistant Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         Client                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────┐    ┌──────────────┐        │
│  │ Microphone  │───▶│   VAD    │───▶│  Mode Logic  │        │
│  └─────────────┘    │(WebRTC)  │    └──────┬───────┘        │
│                     │          │           │                 │
│                     │ Barge-in │  ┌────────┴────────┐       │
│                     │ detection│  │                 │       │
│                     └──────────┘  │   Server up?    │       │
│                                   │                 │       │
│                         ┌─────────┴─────────┐       │       │
│                         │                   │       │       │
│                        YES                 NO       │       │
│                         │                   │       │       │
│                         ▼                   ▼       │       │
│                  ┌─────────────┐    ┌─────────────┐ │       │
│                  │ LOCAL MODE  │    │ CLOUD MODE  │ │       │
│                  │ WebSocket   │    │ REST APIs   │ │       │
│                  └──────┬──────┘    └──────┬──────┘ │       │
│                         │                   │       │       │
│  ┌─────────────────────┴───────────────────┘       │       │
│  │                                                  │       │
│  ▼                                                  │       │
│  ┌─────────────┐    ┌──────────┐                   │       │
│  │   Speaker   │◄───│ Feedback │ (beep tones)      │       │
│  └─────────────┘    └──────────┘                   │       │
└──────────────────────────────────────────────────────────────┘
            │                               │
            │ WebSocket                     │ HTTPS
            ▼                               ▼
┌─────────────────────┐         ┌──────────────────────┐
│   Local Server      │         │   Cloud Services     │
│   (GPU Machine)     │         │                      │
├─────────────────────┤         ├──────────────────────┤
│                     │         │ Deepgram STT/TTS     │
│ Whisper → Llama →   │         │ DeepSeek LLM         │
│ Piper               │         │                      │
│                     │         │                      │
│ Sessions saved to   │         │                      │
│ disk (JSON)         │         │                      │
└─────────────────────┘         └──────────────────────┘
```

## Local Mode Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Server Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio In ──▶ VAD ──▶ Whisper ──▶ Llama ──▶ Piper ──▶ Audio Out    │
│              (Silero)   (STT)      (LLM)    (TTS)                   │
│                │                     │        │                     │
│                │                     │        └─► Prefetch queue    │
│                │                     │            (next sentence)   │
│                │                     │                              │
│                │                     └─► History ◄─► sessions/*.json│
│                │                                                    │
│                └─► Utterance boundary detection                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Reliability:                                                       │
│  • Llama health check every 30s, auto-restart on crash (5 retries) │
│  • LLM calls: 3 retries with exponential backoff                   │
│  • TTS calls: 2 retries with fixed delay                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Barge-in Flow

```
Client                                 Server
  │                                      │
  │◄──────── tts_start ──────────────────│  Server begins speaking
  │                                      │
  │  [User speaks during playback]       │
  │  [Client VAD detects speech]         │
  │                                      │
  │──────── {"type":"interrupt"} ───────▶│  Client sends interrupt
  │  [Stops speaker immediately]         │
  │                                      │  [Sets _interrupted=True]
  │                                      │  [Cancels LLM streaming]
  │                                      │  [Cancels TTS synthesis]
  │◄──────── tts_stop ───────────────────│
  │                                      │
  │  [User's speech captured by VAD]     │
  │──────── audio bytes ────────────────▶│  New utterance processed
  │                                      │
  │◄──────── transcription ──────────────│  "stop" (or whatever said)
  │◄──────── response audio ─────────────│  LLM responds to interrupt
```

## Dual VAD Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Client VAD (WebRTC)                                             │
│ Purpose: Barge-in detection                                     │
│ When: Runs continuously, checks if speech during TTS playback   │
│ Action: Sends interrupt signal, stops local playback            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Audio stream (always sending)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Server VAD (Silero)                                             │
│ Purpose: Utterance boundary detection                           │
│ When: Processes incoming audio to find speech start/end         │
│ Action: Buffers speech, triggers STT when silence detected      │
└─────────────────────────────────────────────────────────────────┘
```

## Audio Feedback

```
Event                    Tone              Meaning
─────────────────────────────────────────────────────
Transcription received   440Hz (low)       Processing your speech
```

## Session Persistence

```
┌──────────────────────────────────────────────────────────────┐
│                    Session Lifecycle                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Connect ──▶ Load session (if ID provided)                   │
│                     │                                        │
│                     ▼                                        │
│              ┌─────────────┐                                 │
│              │ Conversation│◄──┐                             │
│              │   History   │   │                             │
│              └──────┬──────┘   │                             │
│                     │          │                             │
│                     ▼          │                             │
│              Each LLM response │                             │
│              saves to disk ────┘                             │
│                     │                                        │
│                     ▼                                        │
│  Disconnect ──▶ Final save to sessions/{id}.json             │
│                                                              │
└──────────────────────────────────────────────────────────────┘

File format:
sessions/
├── abc123.json          # Auto-generated connection IDs
├── my-session.json      # Custom session IDs
└── ...

Restore via:
  {"type": "hello", "session_id": "my-session"}
  {"type": "load_session", "session_id": "my-session"}
```

## TTS Prefetching

```
Timeline ────────────────────────────────────────────────────────▶

Sentence 1:  [Synthesize]───▶[Stream to client]
Sentence 2:       [Synthesize]────────────────▶[Stream]
Sentence 3:            [Synthesize]─────────────────────▶[Stream]

Queue depth: 2 sentences ahead
Result: Minimal gaps between sentences
```

## Cloud Mode Pipeline

```
Client                              Cloud Services
┌─────────────┐                    
│ Microphone  │                    
└──────┬──────┘                    
       │                           
       ▼                           
┌─────────────┐                    ┌──────────────────┐
│ VAD         │                    │ Deepgram         │
│ (WebRTC)    │                    │ ┌──────────────┐ │
└──────┬──────┘                    │ │ Nova-2 STT   │ │
       │          POST /listen     │ └──────┬───────┘ │
       └─────────────────────────▶ │        │         │
                                   └────────┼─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ DeepSeek         │
                                   │ ┌──────────────┐ │
                                   │ │ Chat API     │ │
                                   │ └──────┬───────┘ │
                                   └────────┼─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ Deepgram         │
       ┌─────────────┐             │ ┌──────────────┐ │
       │ Speaker     │◄────────────│ │ Aura TTS     │ │
       └─────────────┘  GET audio  │ └──────────────┘ │
                                   └──────────────────┘
```

## Decision Flow

```
┌─────────────────────────────────────┐
│         Client Startup              │
└─────────────┬───────────────────────┘
              │
              ▼
     ┌────────────────────┐
     │ Load .env config   │
     └─────────┬──────────┘
               │
               ▼
     ┌──────────────────────────┐
     │ GET /health (2s timeout) │
     └─────────┬────────────────┘
               │
        ┌──────┴──────┐
        │             │
    SUCCESS         FAIL
        │             │
        ▼             ▼
┌──────────────┐  ┌────────────────┐
│  LOCAL MODE  │  │ Cloud keys     │
│              │  │ configured?    │
│ • WebSocket  │  └────────┬───────┘
│ • Low latency│           │
│ • Private    │     ┌─────┴─────┐
│ • Free       │     │           │
└──────────────┘    YES          NO
                     │           │
                     ▼           ▼
              ┌──────────┐  ┌────────┐
              │  CLOUD   │  │ Retry  │
              │  MODE    │  │ in 5s  │
              │          │  └────────┘
              │ • HTTPS  │
              │ • Higher │
              │   latency│
              │ • Paid   │
              └──────────┘
```

## WebSocket Protocol Summary

### Client → Server

| Message | Format | Purpose |
|---------|--------|---------|
| Audio | Binary (float32, 16kHz) | Continuous mic stream |
| Hello | `{"type":"hello","session_id":"..."}` | Init + optional session restore |
| Interrupt | `{"type":"interrupt"}` | Stop current response |
| Load session | `{"type":"load_session","session_id":"..."}` | Restore conversation |

### Server → Client

| Message | Format | Purpose |
|---------|--------|---------|
| Audio | Binary (PCM16, 22kHz) | TTS response |
| Transcription | `{"type":"transcription","text":"..."}` | What user said |
| LLM response | `{"type":"llm_response","text":"..."}` | Full response text |
| TTS start | `{"type":"tts_start"}` | Playback beginning |
| TTS stop | `{"type":"tts_stop"}` | Playback ended |
