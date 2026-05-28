"""PokerBench test-split loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Item:
    item_id: str
    split: str           # "preflop_test" | "postflop_test"
    instruction: str
    gto_label: str


# 1,000 preflop + 10,000 postflop = 11,000 PokerBench test items.
TEST_FILES = {
    "preflop_test":  "data/raw/pokerbench/preflop_1k_test_set_prompt_and_label.json",
    "postflop_test": "data/raw/pokerbench/postflop_10k_test_set_prompt_and_label.json",
}


def load_test_items(root: str | Path = ".") -> list[Item]:
    """Load the full PokerBench test split (1k preflop + 10k postflop)."""
    root = Path(root)
    items: list[Item] = []
    for split, rel in TEST_FILES.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python -c 'from huggingface_hub import "
                f"snapshot_download; snapshot_download(\"RZ412/PokerBench\", "
                f"repo_type=\"dataset\", local_dir=\"data/raw/pokerbench\")'"
            )
        with path.open() as f:
            data = json.load(f)
        for idx, row in enumerate(data):
            items.append(Item(
                item_id=f"{split}:{idx}",
                split=split,
                instruction=row["instruction"],
                gto_label=row["output"],
            ))
    return items
