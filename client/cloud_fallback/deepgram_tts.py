import logging
from typing import Optional

import httpx

from client.cloud_fallback.cloud_config import DeepgramConfig

logger = logging.getLogger(__name__)


class DeepgramTTS:
    def __init__(self, config: DeepgramConfig):
        self.config = config

        if not config.api_key:
            raise ValueError("Deepgram API key not configured")

        logger.info("DeepgramTTS initialized: model=%s", config.tts_model)

    async def synthesize(self, text: str) -> Optional[bytes]:

        if not text or not text.strip():
            return None

        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
        }

        params = {
            "model": self.config.tts_model,
            "encoding": self.config.tts_encoding,
            "sample_rate": self.config.tts_sample_rate,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Sending TTS request to Deepgram: text_len=%d", len(text))

                response = await client.post(
                    "https://api.deepgram.com/v1/speak",
                    headers=headers,
                    params=params,
                    json=payload,
                )
                response.raise_for_status()

                audio_bytes = response.content

                if not audio_bytes:
                    logger.warning("Deepgram produced empty audio")
                    return None

                logger.info(
                    "Deepgram TTS complete: text_len=%d audio_bytes=%d",
                    len(text),
                    len(audio_bytes),
                )

                return audio_bytes

        except httpx.TimeoutException:
            logger.error("Deepgram TTS request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Deepgram TTS HTTP error: status=%d", e.response.status_code)
            return None
        except Exception:
            logger.exception("Deepgram TTS failed")
            return None
