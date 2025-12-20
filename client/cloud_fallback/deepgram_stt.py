import logging
from typing import Optional

import httpx
import numpy as np

from client.cloud_fallback.cloud_config import DeepgramConfig

logger = logging.getLogger(__name__)


class DeepgramSTT:
    def __init__(self, config: DeepgramConfig):
        self.config = config

        if not config.api_key:
            raise ValueError("Deepgram API key not configured")

        logger.info("DeepgramSTT initialized: model=%s", config.stt_model)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            headers = {
                "Authorization": f"Token {self.config.api_key}",
                "Content-Type": "audio/raw",
            }

            params = {
                "model": self.config.stt_model,
                "language": self.config.stt_language,
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "channels": 1,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Sending audio to Deepgram: %d bytes", len(audio_bytes))

                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    headers=headers,
                    params=params,
                    content=audio_bytes,
                )
                response.raise_for_status()

                data = response.json()

                transcript = ""
                if "results" in data and "channels" in data["results"]:
                    channels = data["results"]["channels"]
                    if channels and "alternatives" in channels[0]:
                        alternatives = channels[0]["alternatives"]
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")

                logger.info(
                    "Deepgram transcription: text_len=%d",
                    len(transcript),
                )

                return transcript.strip()

        except httpx.HTTPStatusError as e:
            logger.error("Deepgram HTTP error: status=%d", e.response.status_code)
            return ""
        except httpx.TimeoutException:
            logger.error("Deepgram request timed out")
            return ""
        except Exception:
            logger.exception("Deepgram transcription failed")
            return ""
