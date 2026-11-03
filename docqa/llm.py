"""LLM interface and the Groq implementation with retries (FR-7, FR-9)."""

import time
from typing import Callable, Protocol

import groq

from docqa.config import Settings
from docqa.errors import DocQAError

BUSY_MESSAGE = "The AI service is busy. Please wait a minute and try again."
REJECTED_MESSAGE = "The AI service rejected the request. Check GROQ_API_KEY and GROQ_MODEL."

# Rate limits, dropped connections and timeouts are worth retrying.
_RETRYABLE = (groq.RateLimitError, groq.APIConnectionError)


class LLM(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client=None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._client = client if client is not None else groq.Groq(api_key=api_key)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        if json_mode:
            system = f"{system}\nReply with JSON only."
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    timeout=30,
                )
                return response.choices[0].message.content or ""
            except _RETRYABLE:
                if attempt == self.max_attempts:
                    raise DocQAError(BUSY_MESSAGE) from None
                self._sleep(5 * attempt)  # 5 seconds, then 10
            except groq.APIError:
                # Message never includes the key; `from None` keeps it out of tracebacks.
                raise DocQAError(REJECTED_MESSAGE) from None
        raise DocQAError(BUSY_MESSAGE)  # unreachable unless max_attempts < 1


def build_llm(settings: Settings) -> LLM:
    return GroqLLM(settings.groq_api_key, settings.groq_model)
