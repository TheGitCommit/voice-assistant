import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from server.config import LlamaConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0


class LLMClient:
    """Llama.cpp streaming chat client with conversation history (in-memory only)."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.endpoint = config.endpoint_url
        self.timeout = config.request_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

        self.system_instruction = (
            "You are a helpful voice assistant. "
            "IMPORTANT: You are receiving input from a Speech-to-Text (STT) system. "
            "Transcription may contain phonetic errors, missing words, or lack punctuation. "
            "If the input seems nonsensical, use the conversation context to infer the most likely intended meaning. "
            "Ignore conversational filler like 'um' or 'uh' and ignore unintentional repetitions. "
            "If the input is completely incoherent, politely ask for clarification. "
            "\n\n"
            "VOICE OUTPUT RULES: Your output will be converted to audio. "
            "Do NOT use Markdown formatting (no asterisks, bolding, or headers). "
            "Do NOT use lists or bullet points. Speak naturally using transition words like 'First' or 'Next'. "
            "Keep responses concise, conversational, and friendly."
        )

        self.history: list[dict[str, str]] = []

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        logger.debug("History updated: role=%s length=%d", role, len(self.history))

    def clear_history(self) -> None:
        self.history.clear()
        logger.info("Chat history cleared")

    def get_history_summary(self) -> str:
        return f"{len(self.history)} messages ({len(self.history) // 2} exchanges)"

    def trim_history(self, max_turns: int) -> None:
        max_messages = max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
            logger.info(
                "History trimmed to %d messages (%d turns)",
                len(self.history),
                max_turns,
            )

    async def stream_completion(self, prompt: str) -> AsyncGenerator[str, None]:
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to LLM")
            return

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        if len(self.history) > self.config.max_history_turns * 2:
            self.trim_history(self.config.max_history_turns)

        self.add_to_history("user", prompt)

        messages = [{"role": "system", "content": self.system_instruction}]
        messages.extend(self.history)

        payload = {
            "messages": messages,
            "stream": True,
        }

        start_time = time.perf_counter()
        accumulated_response = ""
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    "Sending streaming LLM request (attempt %d/%d): prompt_len=%d",
                    attempt,
                    MAX_RETRIES,
                    len(prompt),
                )

                async with self._client.stream(
                    "POST", self.endpoint, json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0]["delta"].get(
                                        "content", ""
                                    )
                                    if delta:
                                        accumulated_response += delta
                                        yield delta
                            except (json.JSONDecodeError, KeyError):
                                continue

                if accumulated_response:
                    self.add_to_history("assistant", accumulated_response)

                duration = time.perf_counter() - start_time
                logger.info(
                    "LLM streaming completed in %.3fs, history now: %s",
                    duration,
                    self.get_history_summary(),
                )
                return

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s. Retrying...",
                        attempt,
                        MAX_RETRIES,
                        e,
                    )
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    logger.error(
                        "LLM request failed after %d attempts: %s", MAX_RETRIES, e
                    )

            except httpx.HTTPStatusError as e:
                logger.error("LLM HTTP error: %d", e.response.status_code)
                return

            except Exception:
                logger.exception("LLM streaming failed")
                return

        if last_error:
            self.history.pop()  # Remove the user message we added

    async def get_completion(self, prompt: str) -> Optional[str]:
        full_text = ""
        async for chunk in self.stream_completion(prompt):
            full_text += chunk
        return full_text if full_text else None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
