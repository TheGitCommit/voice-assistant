import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class DeepgramConfig:
    """
    Configuration for the Deepgram API.

    This frozen dataclass provides configuration settings for interacting with the
    Deepgram API. It includes attributes for controlling speech-to-text (STT) and
    text-to-speech (TTS) behavior, such as models, language, encoding, and sample
    rate. Instances of this dataclass are immutable.

    :ivar api_key: Deepgram API key for authentication.
    :type api_key: str
    :ivar stt_model: Identifier for the model to be used for speech-to-text
        processing.
    :type stt_model: str
    :ivar stt_language: Language code for the input audio during speech-to-text
        processing.
    :type stt_language: str
    :ivar tts_model: Identifier for the model to be used for text-to-speech
        synthesis.
    :type tts_model: str
    :ivar tts_encoding: Encoding format for the audio output during
        text-to-speech synthesis.
    :type tts_encoding: str
    :ivar tts_sample_rate: Sample rate (in Hz) for the audio output during
        text-to-speech synthesis.
    :type tts_sample_rate: int
    """

    api_key: str

    # STT settings
    stt_model: str = "nova-2"
    stt_language: str = "en-US"

    # TTS settings
    tts_model: str = "aura-asteria-en"
    tts_encoding: str = "linear16"
    tts_sample_rate: int = 24000


@dataclass(frozen=True)
class DeepSeekConfig:
    """
    Configuration class for DeepSeek API integration.

    This class is a dataclass that holds configuration details required for
    interacting with the DeepSeek API. It includes attributes for
    authentication, model selection, base URL, and settings for token limits
    and temperature control. The class is immutable due to the frozen nature
    of the dataclass decorator, ensuring the configuration remains consistent
    once instantiated.

    :ivar api_key: API key for authenticating requests to the DeepSeek API.
    :type api_key: str
    :ivar model: The model to be used for DeepSeek interactions. Default is
        "deepseek-chat".
    :type model: str
    :ivar base_url: The base URL of the DeepSeek API. Default is
        "https://api.deepseek.com".
    :type base_url: str
    :ivar max_tokens: The maximum number of tokens to be used in API responses.
        Default is 1000.
    :type max_tokens: int
    :ivar temperature: Sampling temperature for response generation, which
        controls randomness. Default is 0.7.
    :type temperature: float
    """

    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass(frozen=True)
class CloudConfig:
    """
    Represents the configuration settings for cloud services.

    This class defines the configuration required for interacting with different
    cloud-based services. It includes specific configurations for Deepgram and
    DeepSeek alongside general connection settings for fallback mechanisms, server
    health checks, and API request timeouts.

    :ivar deepgram: Configuration settings specific to Deepgram service.
    :type deepgram: DeepgramConfig
    :ivar deepseek: Configuration settings specific to DeepSeek service.
    :type deepseek: DeepSeekConfig
    :ivar enable_fallback: Determines if fallback mechanisms should be enabled
        when the primary service is unreachable. Defaults to True.
    :type enable_fallback: bool
    :ivar server_health_check_timeout: The timeout duration (in seconds) for
        checking server health. Defaults to 5.0 seconds.
    :type server_health_check_timeout: float
    :ivar api_request_timeout: The timeout duration (in seconds) for API requests.
        Defaults to 30.0 seconds.
    :type api_request_timeout: float
    """

    deepgram: DeepgramConfig
    deepseek: DeepSeekConfig

    # Connection settings
    enable_fallback: bool = True
    server_health_check_timeout: float = 5.0
    api_request_timeout: float = 30.0


CLOUD_CONFIG = CloudConfig(
    deepgram=DeepgramConfig(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
    ),
    deepseek=DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    ),
    enable_fallback=os.getenv("ENABLE_CLOUD_FALLBACK", "true").lower() == "true",
)


def is_cloud_configured() -> bool:
    """
    Determines if the cloud configuration is properly set up.

    This function checks whether the required API keys for `deepgram` and
    `deepseek` services, as well as the fallback enablement flag, are
    configured in the `CLOUD_CONFIG`. It returns a boolean value indicating
    whether the cloud services are ready for use.

    :return: True if the cloud configuration is properly set up,
        otherwise False.
    :rtype: bool
    """
    return bool(
        CLOUD_CONFIG.deepgram.api_key
        and CLOUD_CONFIG.deepseek.api_key
        and CLOUD_CONFIG.enable_fallback
    )
