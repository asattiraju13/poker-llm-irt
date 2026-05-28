"""Consolidate per-(model, split) parquet shards into long and wide response matrices."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PANEL_ORDER = [
    "openai:gpt-5-mini",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-haiku-4-5",
    "google:gemini-2.5-pro",
    "google:gemini-2.5-flash-lite",
    "together:Qwen/Qwen3-235B-A22B-Instruct-tput",
    "together:deepseek-ai/DeepSeek-V4-Pro",
    "deepseek:deepseek-reasoner",
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together:Qwen/Qwen2.5-7B-Instruct-Turbo",
]


def load_long(run_name: str, data_root: str | Path = "data/responses") -> pd.DataFrame:
    """Concatenate all parquet shards under ``data_root/run_name`` into one long DataFrame."""
    shard_dir = Path(data_root) / run_name
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {shard_dir}/")
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns:
        df = df.sort_values("ts").drop_duplicates(subset=["model_id", "item_id"], keep="last")
    return df.reset_index(drop=True)


def to_wide(long: pd.DataFrame, value: str = "aa") -> pd.DataFrame:
    """Pivot to a wide matrix: rows = item_id, columns = model_id, values = ``value``."""
    wide = long.pivot_table(index="item_id", columns="model_id", values=value,
                            aggfunc="first")
    cols = [m for m in PANEL_ORDER if m in wide.columns]
    extras = [c for c in wide.columns if c not in PANEL_ORDER]
    wide = wide[cols + extras]
    return wide


def sanity_check(long: pd.DataFrame) -> dict:
    """Return basic counts and per-model summary statistics for diagnostics."""
    out: dict = {}
    out["n_rows"] = len(long)
    out["n_unique_items"] = long["item_id"].nunique()
    out["n_models"] = long["model_id"].nunique()
    out["models"] = sorted(long["model_id"].unique().tolist())
    out["per_model_n"] = long.groupby("model_id").size().to_dict()
    out["per_split_n"] = long.groupby(["model_id", "split"]).size().unstack(fill_value=0).to_dict()
    out["aa_mean_per_model"] = long.groupby("model_id")["aa"].mean().round(4).to_dict()
    out["parse_ok_per_model"] = long.groupby("model_id")["parse_ok"].mean().round(4).to_dict()
    out["err_rate_per_model"] = (
        long.assign(is_err=long["error"].astype(str).str.len() > 0)
            .groupby("model_id")["is_err"].mean().round(4).to_dict()
    )
    return out
