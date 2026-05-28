"""Provider protocol and shared types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_text: str = ""
    reasoning_tokens: int = 0
    finish_reason: str = ""


class ChatProvider(Protocol):
    model: str

    async def chat(self, prompt: str, max_tokens: int, temperature: float) -> ChatResult: ...
