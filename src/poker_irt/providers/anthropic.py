"""Anthropic Claude provider with extended-thinking support."""
from __future__ import annotations

import os

from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import ChatResult


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking_budget: int | None = None,
    ):
        self.model = model
        self.thinking_budget = thinking_budget
        self.client = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

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
            "max_tokens": max_tokens,
        }
        if self.thinking_budget:
            # Anthropic requires max_tokens > budget_tokens and budget_tokens >= 1024.
            # Reserve >=100 tokens for the final answer; fall back to no thinking
            # if max_tokens leaves too little room.
            budget = min(self.thinking_budget, max_tokens - 100)
            if budget >= 1024:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                kwargs["temperature"] = 1.0   # required with thinking enabled
            else:
                kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = temperature

        resp = await self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))

        usage = getattr(resp, "usage", None)
        return ChatResult(
            text="\n".join(p for p in text_parts if p),
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            reasoning_text="\n".join(p for p in thinking_parts if p),
            finish_reason=getattr(resp, "stop_reason", "") or "",
        )

    async def chat(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> ChatResult:
        return await self._call(prompt, max_tokens, temperature)
