"""Eval orchestrator: stream JSONL during a run, materialize parquet shards at the end.

The JSONL log is the resumable source of truth: on resume, items with a
successful row are skipped and items with only error rows are retried.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from .config import ModelSpec, RunConfig
from .data import Item, load_test_items
from .parsing import parse_action, parse_gto_label
from .prompting import build_user_message
from .providers import get_provider
from .scoring import score


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")


def _load_done_ids(jsonl_path: Path) -> set[str]:
    """Item IDs that already have at least one error-free row in the JSONL."""
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("error"):
                    done.add(row["item_id"])
            except Exception:
                continue
    return done


async def _eval_one_model(
    spec: ModelSpec,
    items: list[Item],
    out_dir: Path,
    cfg: RunConfig,
) -> None:
    jsonl_path = out_dir / "_logs" / f"{_safe_name(spec.model_id)}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done_ids(jsonl_path)
    remaining = [it for it in items if it.item_id not in done_ids]
    if not remaining:
        print(f"[{spec.model_id}] all {len(items)} items already complete; skipping")
        return
    print(f"[{spec.model_id}] {len(remaining)} of {len(items)} to run "
          f"({len(done_ids)} resumed)")

    provider = get_provider(spec)
    max_tokens = cfg.max_tokens_override or spec.max_tokens
    sem = asyncio.Semaphore(cfg.concurrency)
    write_lock = asyncio.Lock()

    async def _do_one(item: Item) -> None:
        async with sem:
            t0 = time.time()
            error = ""
            raw = ""
            reasoning_text = ""
            pt = ct = rt = 0
            finish = ""
            try:
                result = await provider.chat(
                    prompt=build_user_message(item),
                    max_tokens=max_tokens,
                    temperature=cfg.temperature,
                )
                raw = result.text
                reasoning_text = result.reasoning_text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                rt = result.reasoning_tokens
                finish = result.finish_reason
            except Exception as e:
                error = f"{type(e).__name__}: {e}"[:300]
            latency = time.time() - t0

            gto = parse_gto_label(item.gto_label)
            parsed = parse_action(raw) if raw else None
            aa, em = score(parsed, gto)

            row = {
                "model_id": spec.model_id,
                "item_id": item.item_id,
                "split": item.split,
                "gto_label": item.gto_label,
                "gto_action_class": gto[0],
                "gto_sizing": gto[1],
                "raw_output": raw,
                "reasoning_output": reasoning_text,
                "parsed_action_class": parsed[0] if parsed else "",
                "parsed_sizing": parsed[1] if parsed else None,
                "aa": aa,
                "em": em,
                "parse_ok": parsed is not None,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "reasoning_tokens": rt,
                "latency_s": latency,
                "finish_reason": finish,
                "error": error,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            async with write_lock:
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    tasks = [_do_one(it) for it in remaining]
    await atqdm.gather(*tasks, desc=spec.model_id)


def _materialize_parquet(jsonl_path: Path, out_dir: Path, model_id: str) -> None:
    """Convert the JSONL log into per-split parquet shards (latest attempt per item_id)."""
    if not jsonl_path.exists():
        return
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("ts").drop_duplicates("item_id", keep="last")
    name = _safe_name(model_id)
    for split, sub in df.groupby("split"):
        sub.to_parquet(out_dir / f"{name}__{split}.parquet", index=False)


def _print_summary(out_dir: Path, specs: list[ModelSpec]) -> None:
    rows = []
    for spec in specs:
        jsonl = out_dir / "_logs" / f"{_safe_name(spec.model_id)}.jsonl"
        if not jsonl.exists():
            continue
        data = []
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        if not data:
            continue
        df = pd.DataFrame(data)
        df = df.sort_values("ts").drop_duplicates("item_id", keep="last")
        rows.append({
            "model_id": spec.model_id,
            "n": len(df),
            "AA%": round(df["aa"].mean() * 100, 2),
            "EM%": round(df["em"].mean() * 100, 2),
            "parse_ok%": round(df["parse_ok"].mean() * 100, 2),
            "err%": round((df["error"] != "").mean() * 100, 2),
            "mean_lat_s": round(df["latency_s"].mean(), 2),
            "in_tok": int(df["prompt_tokens"].sum()),
            "out_tok": int(df["completion_tokens"].sum()),
            "rt_tok": int(df.get("reasoning_tokens", pd.Series([0])).sum()),
        })
    if not rows:
        return
    print("\n=== Summary ===")
    print(pd.DataFrame(rows).to_string(index=False))


def run(cfg: RunConfig) -> None:
    out_dir = Path("data/responses") / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    sampled = load_test_items()
    by_split = {s: sum(1 for it in sampled if it.split == s) for s in {it.split for it in sampled}}
    print(f"loaded {len(sampled)} items: {by_split}")

    for spec in cfg.models:
        asyncio.run(_eval_one_model(spec, sampled, out_dir, cfg))

    for spec in cfg.models:
        jsonl = out_dir / "_logs" / f"{_safe_name(spec.model_id)}.jsonl"
        _materialize_parquet(jsonl, out_dir, spec.model_id)

    _print_summary(out_dir, cfg.models)
