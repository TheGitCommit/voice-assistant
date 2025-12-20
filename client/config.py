import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class ServerConfig:
    """ """

    url: str


@dataclass(frozen=True)
class AudioCaptureConfig:
    """
    Configuration class for audio capture settings.

    This class provides a structured way to define and manage settings for audio
    capture operations. It includes attributes such as sample rate, number of
    channels, chunk size, maximum queue size, and data type of the audio samples.

    :ivar sample_rate: Sample rate (in Hz) for the audio capture.
    :type sample_rate: int
    :ivar channels: Number of audio channels to capture.
    :type channels: int
    :ivar chunk_size: Size of each audio chunk in samples.
    :type chunk_size: int
    :ivar queue_maxsize: Maximum size of the queue for audio chunks.
    :type queue_maxsize: int
    :ivar dtype: Data type of the audio samples (e.g., "float32").
    :type dtype: str
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320  # 20ms at 16kHz
    queue_maxsize: int = 20
    dtype: str = "float32"


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """
    Configuration class for audio playback.

    This class encapsulates settings related to audio playback functionality, including
    sample rate, number of audio channels, and data type. It is implemented as an
    immutable dataclass to ensure that the configuration cannot be modified after its
    creation.

    :ivar sample_rate: The sample rate for audio playback in Hertz. By default, it is
        set to 22050, which is the default output rate for Piper TTS.
    :type sample_rate: int
    :ivar channels: The number of audio channels. Defaults to 1 for mono audio output.
    :type channels: int
    :ivar dtype: The data type of the audio samples. This indicates the format of
        the server's PCM output and is set as "int16" by default.
    :type dtype: str
    """

    sample_rate: int = 22050  # Piper TTS default output rate
    channels: int = 1
    dtype: str = "int16"  # Server sends PCM16


@dataclass(frozen=True)
class VADConfig:
    """
    Configuration for Voice Activity Detection (VAD).

    This data class encapsulates configuration parameters for VAD. It defines
    the aggressiveness level of detection, the duration of the audio frames
    to be analyzed, and the limit for silence frames before an utterance is
    considered ended.

    :ivar aggressiveness: Defines the WebRTC VAD aggressiveness level. Accepts
        values from 0 (least aggressive) up to 3 (most aggressive).
    :ivar frame_duration_ms: Specifies the duration of audio frames in
        milliseconds that will be processed by the VAD.
    :ivar silence_limit_frames: Indicates the number of consecutive silent
        frames required to deem the end of an utterance. Approximately 200ms
        of silence is used to end utterances when the frame duration is 20ms.
    """

    # WebRTC VAD aggressiveness: 0 (least) to 3 (most aggressive)
    aggressiveness: int = 2
    frame_duration_ms: int = 20
    silence_limit_frames: int = 10  # ~200ms of silence ends utterance


@dataclass(frozen=True)
class ClientConfig:
    """
    Represents the configuration for a client.

    This class is used to encapsulate the configuration details for a client,
    which includes server settings, audio capture configuration, audio playback
    options, and voice activity detection (VAD) settings. The class is immutable
    due to the `frozen=True` behavior of the dataclass for ensuring configuration
    integrity.

    :ivar server: Configuration details for the server.
    :type server: ServerConfig
    :ivar capture: Configuration details for audio capture functionality.
    :type capture: AudioCaptureConfig
    :ivar playback: Configuration data for handling audio playback.
    :type playback: AudioPlaybackConfig
    :ivar vad: Configuration for managing voice activity detection (VAD).
    :type vad: VADConfig
    """

    server: ServerConfig
    capture: AudioCaptureConfig
    playback: AudioPlaybackConfig
    vad: VADConfig


DEFAULT_CONFIG = ClientConfig(
    server=ServerConfig(
        url=os.getenv("SERVER_URL", "ws://localhost:8000/ws/audio"),
    ),
    capture=AudioCaptureConfig(),
    playback=AudioPlaybackConfig(),
    vad=VADConfig(),
)
