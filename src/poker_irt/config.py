"""Run configuration and named model panels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    """One model in a panel: provider, model path, and reasoning/thinking config."""
    model_id: str
    provider: str
    model_path: str
    max_tokens: int = 200
    reasoning_effort: str | None = None   # OpenAI: "minimal"|"low"|"medium"|"high"
    thinking_budget: int | None = None    # Anthropic: extended-thinking budget tokens
    thinking_level: str | None = None     # Gemini: "low"|"medium"|"high"


@dataclass
class RunConfig:
    run_name: str
    models: list[ModelSpec]
    seed: int = 42
    temperature: float = 0.0
    concurrency: int = 8
    max_tokens_override: int | None = None


SANITY_PANEL = [
    ModelSpec("together:Qwen/Qwen2.5-7B-Instruct-Turbo",
              "together", "Qwen/Qwen2.5-7B-Instruct-Turbo", max_tokens=120),
    ModelSpec("together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens=120),
    ModelSpec("together:deepseek-ai/DeepSeek-V4-Pro",
              "together", "deepseek-ai/DeepSeek-V4-Pro", max_tokens=4000),
]

# Main analysis panel. All open-weight models served via Together's serverless
# tier; closed frontier models via their native APIs.
MAIN_PANEL = [
    ModelSpec("openai:gpt-5-mini",
              "openai", "gpt-5-mini",
              max_tokens=6000, reasoning_effort="medium"),
    ModelSpec("anthropic:claude-sonnet-4-6",
              "anthropic", "claude-sonnet-4-6",
              max_tokens=4000, thinking_budget=2000),
    ModelSpec("anthropic:claude-haiku-4-5",
              "anthropic", "claude-haiku-4-5",
              max_tokens=200),
    ModelSpec("together:Qwen/Qwen3-235B-A22B-Instruct-tput",
              "together", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
              max_tokens=200),
    ModelSpec("together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
              max_tokens=200),
    ModelSpec("together:Qwen/Qwen2.5-7B-Instruct-Turbo",
              "together", "Qwen/Qwen2.5-7B-Instruct-Turbo",
              max_tokens=200),
    ModelSpec("google:gemini-2.5-flash-lite",
              "google", "gemini-2.5-flash-lite",
              max_tokens=200),
]


MODEL_PANELS: dict[str, list[ModelSpec]] = {
    "sanity": SANITY_PANEL,
    "main": MAIN_PANEL,
}
