"""DeepSeek chat completions provider (OpenAI-compatible endpoint)."""
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


class DeepSeekProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.model = model
        # Long httpx timeout to absorb reasoning-model tail latency.
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
            timeout=timeout,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, prompt: str, max_tokens: int, temperature: float) -> ChatResult:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        # DeepSeek reasoning models expose chain-of-thought under reasoning_content.
        reasoning = getattr(msg, "reasoning_content", "") or ""
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_text=reasoning,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )

    async def chat(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> ChatResult:
        return await self._call(prompt, max_tokens, temperature)
