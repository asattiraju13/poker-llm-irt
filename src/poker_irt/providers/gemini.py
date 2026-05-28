"""Google Gemini provider via the ``google-genai`` SDK."""
from __future__ import annotations

import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import ChatResult


class GeminiProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking_level: str | None = None,
    ):
        from google import genai

        self.model = model
        self.thinking_level = thinking_level
        self.client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, prompt: str, max_tokens: int, temperature: float) -> ChatResult:
        from google.genai import types

        if self.thinking_level:
            # Translate symbolic ``thinking_level`` to the integer budget the
            # 2.5-series API expects (-1 = dynamic).
            level_map = {"low": 1024, "medium": 4096, "high": -1}
            budget = level_map.get(self.thinking_level, -1)
            if budget > 0:
                budget = min(budget, max_tokens - 100)
            thinking_config = types.ThinkingConfig(thinking_budget=budget)
        else:
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=thinking_config,
        )

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
        completion_tok = getattr(usage, "candidates_token_count", 0) or 0
        reasoning_tok = getattr(usage, "thoughts_token_count", 0) or 0

        finish = ""
        if getattr(resp, "candidates", None):
            finish = getattr(resp.candidates[0], "finish_reason", "") or ""
            finish = str(finish) if finish else ""

        return ChatResult(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            reasoning_tokens=reasoning_tok,
            finish_reason=finish,
        )

    async def chat(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> ChatResult:
        return await self._call(prompt, max_tokens, temperature)
