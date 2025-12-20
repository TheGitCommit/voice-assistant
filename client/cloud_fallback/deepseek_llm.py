import logging
from typing import Optional

import httpx

from client.cloud_fallback.cloud_config import DeepSeekConfig

logger = logging.getLogger(__name__)


class DeepSeekLLM:
    def __init__(self, config: DeepSeekConfig):
        self.config = config

        if not config.api_key:
            raise ValueError("DeepSeek API key not configured")

        logger.info("DeepSeekLLM initialized: model=%s", config.model)

    async def get_completion(self, prompt: str) -> Optional[str]:
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to DeepSeek")
            return None

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug("Sending request to DeepSeek: prompt_len=%d", len(prompt))

                response = await client.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                completion = data["choices"][0]["message"]["content"]

                logger.info(
                    "DeepSeek completion: prompt_len=%d response_len=%d",
                    len(prompt),
                    len(completion),
                )

                return completion

        except httpx.TimeoutException:
            logger.error("DeepSeek request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("DeepSeek HTTP error: status=%d", e.response.status_code)
            return None
        except (KeyError, ValueError):
            logger.exception("DeepSeek response parsing failed")
            return None
        except Exception:
            logger.exception("DeepSeek request failed")
            return None
