"""OpenAI chat completions provider with reasoning-model support."""
from __future__ import annotations

import os

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import ChatResult


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, prompt: str, max_tokens: int, temperature: float) -> ChatResult:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.reasoning_effort:
            # Reasoning models require max_completion_tokens; temperature is fixed
            # to 1 by the API and must be omitted.
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "")

        usage = resp.usage
        prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
        completion_tok = getattr(usage, "completion_tokens", 0) or 0
        # Reasoning tokens are billed under usage.completion_tokens_details.
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tok = getattr(details, "reasoning_tokens", 0) if details else 0

        return ChatResult(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            reasoning_tokens=reasoning_tok or 0,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )

    async def chat(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> ChatResult:
        return await self._call(prompt, max_tokens, temperature)
