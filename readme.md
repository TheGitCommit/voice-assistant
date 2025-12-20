# Local AI Assistant

**Fully local, privacy-focused voice assistant** with real-time streaming capabilities.

---

## 🚀 Key Features

* **Real-time Pipeline**: Continuous audio streaming with energy-based Voice Activity Detection (VAD) for natural interaction.
* **Cloud-Free Privacy**: All processing (STT, LLM, and TTS) occurs locally; no data is sent to external APIs.
* **Low Latency**: Optimized WebSocket streaming and asynchronous processing ensure quick response times.
* **Distributed Design**: Offloads heavy model computation to a server while maintaining a responsive edge client.

---

## 🛠 Technical Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **STT** | Faster Whisper | High-speed, local speech-to-text transcription. |
| **LLM** | Llama.cpp (GGUF) | Local LLM hosting for context-aware responses. |
| **TTS** | Piper | High-quality, neural text-to-speech synthesis. |
| **Streaming** | WebSockets | Full-duplex binary data transfer for real-time audio. |
| **Orchestration** | Python & Asyncio | Non-blocking management of hardware and network tasks. |

---

## ⚙️ How It Works

The system follows a synchronized streaming loop to minimize wait times:

1.  **Client**: Captures raw audio chunks and streams them via WebSockets.
2.  **Server**: Monitors the stream with VAD to detect user utterances.
3.  **Inference**: Transcribes speech with Whisper, generates text with Llama, and synthesizes a response via Piper.
4.  **Playback**: The resulting audio is streamed back and played instantly on the client.

---

## 📂 Project Structure

* **/client**: Lightweight Python scripts for audio I/O and WebSocket communication (ideal for Raspberry Pi or edge devices).
* **/server**: Inference engine hosting Whisper, Llama.cpp, and Piper models (designed for GPU-enabled machines).
