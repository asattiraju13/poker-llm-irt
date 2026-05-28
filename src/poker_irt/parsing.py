"""Regex parsing for GTO labels and free-form LLM outputs.

Action vocabulary: ``{fold, check, call, bet, allin}``. ``raise`` normalizes
to ``bet``. Sizing is a float for bet/raise and ``None`` otherwise.
"""
from __future__ import annotations

import re

# Three alternation branches; iterate finditer and keep the LAST match so that
# models which think aloud before committing produce their final action.
ACTION_RE = re.compile(
    r"\b(?P<bet>bet|raise)\b\s*(?:[a-z']+\s+){0,3}(?P<size>\d+(?:\.\d+)?)?"
    r"|\b(?P<allin>all[- ]?in|allin|shove)\b"
    r"|\b(?P<simple>fold|check|call)\b",
    re.IGNORECASE,
)


def parse_action(text: str) -> tuple[str, float | None] | None:
    """Return ``(action_class, sizing)`` of the last recognizable action in ``text``."""
    if not text:
        return None
    last: tuple[str, float | None] | None = None
    for m in ACTION_RE.finditer(text.lower()):
        if m.group("bet"):
            size = float(m.group("size")) if m.group("size") else None
            last = ("bet", size)
        elif m.group("allin"):
            last = ("allin", None)
        elif m.group("simple"):
            last = (m.group("simple").lower(), None)
    return last


def parse_gto_label(label: str) -> tuple[str, float | None]:
    """Parse a PokerBench ``output`` value such as ``'call'`` or ``'bet 24'``."""
    s = (label or "").strip().lower()
    if s in {"all in", "allin", "all-in", "shove"}:
        return ("allin", None)
    parts = s.split()
    if not parts:
        return ("", None)
    action = parts[0]
    if action == "raise":
        action = "bet"
    if action in {"fold", "check", "call"}:
        return (action, None)
    if action == "bet":
        size = float(parts[1]) if len(parts) > 1 else None
        return ("bet", size)
    return (action, None)
