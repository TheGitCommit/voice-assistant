import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from server.config import LlamaConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0
SESSIONS_DIR = Path("sessions")


class LLMClient:
    """Llama.cpp streaming chat client with conversation history and persistence."""

    def __init__(self, config: LlamaConfig, session_id: Optional[str] = None):
        self.config = config
        self.endpoint = config.endpoint_url
        self.timeout = config.request_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id = session_id

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

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    def save_history(self, session_id: Optional[str] = None) -> bool:
        """Save conversation history to a file. Returns True if successful."""
        sid = session_id or self._session_id
        if not sid:
            logger.warning("No session ID provided, cannot save history")
            return False

        if not self.history:
            logger.debug("No history to save")
            return True

        try:
            SESSIONS_DIR.mkdir(exist_ok=True)
            filepath = SESSIONS_DIR / f"{sid}.json"

            data = {
                "session_id": sid,
                "history": self.history,
                "saved_at": time.time(),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(
                "Saved history to %s (%d messages)", filepath, len(self.history)
            )
            return True

        except Exception:
            logger.exception("Failed to save history")
            return False

    def load_history(self, session_id: Optional[str] = None) -> bool:
        """Load conversation history from a file. Returns True if successful."""
        sid = session_id or self._session_id
        if not sid:
            logger.warning("No session ID provided, cannot load history")
            return False

        filepath = SESSIONS_DIR / f"{sid}.json"

        if not filepath.exists():
            logger.debug("No saved history found for session %s", sid)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.history = data.get("history", [])
            self._session_id = sid
            logger.info(
                "Loaded history from %s (%d messages)", filepath, len(self.history)
            )
            return True

        except Exception:
            logger.exception("Failed to load history")
            return False

    def delete_history(self, session_id: Optional[str] = None) -> bool:
        """Delete saved history file. Returns True if successful."""
        sid = session_id or self._session_id
        if not sid:
            return False

        filepath = SESSIONS_DIR / f"{sid}.json"

        try:
            if filepath.exists():
                filepath.unlink()
                logger.info("Deleted history file: %s", filepath)
            return True
        except Exception:
            logger.exception("Failed to delete history file")
            return False

    @staticmethod
    def list_sessions() -> list[str]:
        """List all saved session IDs."""
        if not SESSIONS_DIR.exists():
            return []
        return [f.stem for f in SESSIONS_DIR.glob("*.json")]

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
