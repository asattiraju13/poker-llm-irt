"""Shared logging setup and per-run metrics helper."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(run_name: str, script_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to write to ``outputs/<run_name>/logs/<script>_<ts>.log``."""
    log_dir = Path(f"outputs/{run_name}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{ts}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)

    log = logging.getLogger(script_name)
    log.info(f"=== {script_name} started; log -> {log_path} ===")
    return log


def write_metrics(run_name: str, script_name: str, metrics: dict) -> Path:
    """Append a metrics record to ``outputs/<run_name>/metrics.jsonl``."""
    out_dir = Path(f"outputs/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.jsonl"
    record = {
        "ts": datetime.now().isoformat(),
        "script": script_name,
        **metrics,
    }
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path
