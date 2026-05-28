"""Provider registry. Dispatch on ``ModelSpec.provider``."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ChatProvider, ChatResult

if TYPE_CHECKING:
    from ..config import ModelSpec


def get_provider(spec: "ModelSpec") -> ChatProvider:
    p = spec.provider
    if p == "together":
        from .together import TogetherProvider
        return TogetherProvider(model=spec.model_path)
    if p == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(model=spec.model_path, reasoning_effort=spec.reasoning_effort)
    if p == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(model=spec.model_path, thinking_budget=spec.thinking_budget)
    if p == "google":
        from .gemini import GeminiProvider
        return GeminiProvider(model=spec.model_path, thinking_level=spec.thinking_level)
    if p == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(model=spec.model_path)
    raise ValueError(f"unknown provider {p!r} (model_id={spec.model_id!r})")


__all__ = ["ChatProvider", "ChatResult", "get_provider"]
