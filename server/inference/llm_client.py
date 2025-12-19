"""
Client for interacting with the local Llama.cpp server.
"""

import logging
from typing import Optional

import httpx

from server.config import LlamaConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """HTTP client for Llama.cpp chat completions API."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.endpoint = config.endpoint_url
        self.timeout = config.request_timeout_seconds

    async def get_completion(self, prompt: str) -> Optional[str]:
        """
        Get a chat completion from the LLM.

        Args:
            prompt: User prompt text

        Returns:
            LLM response text, or None if request fails
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to LLM")
            return None

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient() as client:
                logger.debug("Sending LLM request: prompt_len=%d", len(prompt))

                response = await client.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                completion = data["choices"][0]["message"]["content"]

                logger.info(
                    "LLM completion received: prompt_len=%d response_len=%d",
                    len(prompt),
                    len(completion),
                )

                return completion

        except httpx.TimeoutException:
            logger.error("LLM request timed out after %.1fs", self.timeout)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("LLM HTTP error: status=%d", e.response.status_code)
            return None
        except (KeyError, ValueError):
            logger.exception("LLM response parsing failed")
            return None
        except Exception:
            logger.exception("LLM request failed")
            return None
