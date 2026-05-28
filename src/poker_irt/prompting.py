"""Prompt construction. Currently a passthrough of the PokerBench instruction."""
from __future__ import annotations

from .data import Item


def build_user_message(item: Item) -> str:
    return item.instruction
