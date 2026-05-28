"""Phase 1 item-feature extraction from PokerBench ``instruction`` text.

Extracts game-phase / position / hero-cards / board / preflop-action /
pot-size features via focused regexes. Each extractor returns ``None`` on
parse failure rather than raising; ``features_for_item`` returns a flat dict.
"""
from __future__ import annotations

import re

POSITIONS = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}

RANK_MAP = {
    "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7,
    "Eight": 8, "Nine": 9, "Ten": 10, "Jack": 11, "Queen": 12,
    "King": 13, "Ace": 14,
}
SUIT_MAP = {"Heart": "H", "Diamond": "D", "Spade": "S", "Club": "C"}


# --- card-string parsing ---

_CARD_RE = re.compile(r"\b(Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Jack|Queen|King|Ace)\s+[Oo]f\s+(Heart|Diamond|Spade|Club)\b")


def parse_card(s: str) -> tuple[int, str] | None:
    """Parse a card description like 'King of Heart' to ``(13, 'H')``."""
    if not s:
        return None
    m = _CARD_RE.search(s)
    if not m:
        return None
    return RANK_MAP[m.group(1)], SUIT_MAP[m.group(2)]


# --- field extractors ---

def extract_game_phase(instruction: str) -> str:
    """Return the latest street described in the instruction: preflop/flop/turn/river."""
    if "The river comes" in instruction:
        return "river"
    if "The turn comes" in instruction:
        return "turn"
    if "The flop comes" in instruction:
        return "flop"
    return "preflop"


def extract_hero_position(instruction: str) -> str | None:
    """Return the seat label (e.g. ``BTN``) of the acting player, or ``None`` if unparseable."""
    m = re.search(r"your position is (\w+)", instruction)
    return m.group(1) if m and m.group(1) in POSITIONS else None


def extract_hero_cards(instruction: str) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    """Parse the hero's two hole cards from the ``your holding is [...]`` field."""
    m = re.search(r"your holding is \[([^\[\]]+?)\]", instruction)
    if not m:
        return (None, None)
    inner = m.group(1)
    # find both cards in the bracket
    cards = list(_CARD_RE.finditer(inner))
    if len(cards) < 2:
        return (None, None)
    c1 = (RANK_MAP[cards[0].group(1)], SUIT_MAP[cards[0].group(2)])
    c2 = (RANK_MAP[cards[1].group(1)], SUIT_MAP[cards[1].group(2)])
    return (c1, c2)


def extract_board_cards(instruction: str) -> list[tuple[int, str]]:
    """Return board cards in dealing order (flop1, flop2, flop3, [turn], [river])."""
    cards: list[tuple[int, str]] = []
    # Flop: "The flop comes X, Y, and Z" (sometimes with periods/commas after)
    m = re.search(r"The flop comes\s+(.+?)(?:,\s+then|\.|$)", instruction, re.DOTALL)
    if m:
        chunk = m.group(1)
        for cm in _CARD_RE.finditer(chunk):
            cards.append((RANK_MAP[cm.group(1)], SUIT_MAP[cm.group(2)]))
    # Turn: "The turn comes X"
    m = re.search(r"The turn comes\s+(.+?)(?:,\s+then|\.|$)", instruction, re.DOTALL)
    if m:
        cm = _CARD_RE.search(m.group(1))
        if cm:
            cards.append((RANK_MAP[cm.group(1)], SUIT_MAP[cm.group(2)]))
    # River: "The river comes X"
    m = re.search(r"The river comes\s+(.+?)(?:,\s+then|\.|$)", instruction, re.DOTALL)
    if m:
        cm = _CARD_RE.search(m.group(1))
        if cm:
            cards.append((RANK_MAP[cm.group(1)], SUIT_MAP[cm.group(2)]))
    return cards


def extract_pot_size(instruction: str) -> float | None:
    """Return the pot size in chips at the decision point, or ``None`` if missing."""
    m = re.search(r"current pot size is\s+(\d+(?:\.\d+)?)", instruction)
    return float(m.group(1)) if m else None


def extract_blinds(instruction: str) -> tuple[float | None, float | None]:
    """Return ``(small_blind, big_blind)`` chip values from the instruction header."""
    sb = re.search(r"small blind is\s+(\d+(?:\.\d+)?)", instruction)
    bb = re.search(r"big blind is\s+(\d+(?:\.\d+)?)", instruction)
    return (float(sb.group(1)) if sb else None,
            float(bb.group(1)) if bb else None)


def extract_num_players(instruction: str) -> int | None:
    """Return the table size (e.g. 6 for 6-handed) or ``None`` if unspecified."""
    m = re.search(r"(\d+)-handed", instruction)
    return int(m.group(1)) if m else None


def extract_starting_stack(instruction: str) -> float | None:
    """Return the effective starting stack in chips, or ``None`` if unparseable."""
    m = re.search(r"Everyone started with\s+(\d+(?:\.\d+)?)\s+chips", instruction)
    return float(m.group(1)) if m else None


def extract_preflop_action_chunk(instruction: str) -> str | None:
    """Substring describing the preflop action sequence (e.g. ``HJ raise 2.0, CO call``)."""
    m = re.search(r"Before the flop,\s*(.+?)\.\s*(?:Assume|The flop|Now|$)",
                  instruction, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_preflop_aggressor(preflop_chunk: str | None) -> str | None:
    """Return the position label of the first preflop raiser, or ``None``."""
    if not preflop_chunk:
        return None
    m = re.search(r"\b([A-Z]{2,3})\s+raise\b", preflop_chunk)
    return m.group(1) if m and m.group(1) in POSITIONS else None


def extract_num_preflop_raises(preflop_chunk: str | None) -> int:
    """Count the number of raise/all-in tokens in the preflop action chunk."""
    if not preflop_chunk:
        return 0
    return len(re.findall(r"\braise\b", preflop_chunk)) + len(re.findall(r"\ball in\b", preflop_chunk))


def extract_postflop_action_count(instruction: str) -> int:
    """Count postflop action tokens (bet/raise/check/call/fold) up to the hero's turn."""
    flop_idx = instruction.find("The flop comes")
    if flop_idx < 0:
        return 0
    # Slice from the flop intro up to the "Now it is your turn" marker so
    # we only count actions taken by other players before the hero acts.
    end = instruction.find("Now it is your turn", flop_idx)
    chunk = instruction[flop_idx:end if end >= 0 else len(instruction)]
    return len(re.findall(r"\b(bet|raise|check|call|fold)\b", chunk))


# --- top-level feature dict per item ---

def features_for_item(item_id: str, instruction: str) -> dict:
    """Return a flat ``{feature_name: value}`` dict combining all extractors above."""
    feats: dict = {"item_id": item_id}

    feats["game_phase"] = extract_game_phase(instruction)
    feats["hero_position"] = extract_hero_position(instruction)
    feats["num_players_table"] = extract_num_players(instruction)
    feats["starting_stack"] = extract_starting_stack(instruction)
    sb, bb = extract_blinds(instruction)
    feats["small_blind"] = sb
    feats["big_blind"] = bb

    # Hero hand
    c1, c2 = extract_hero_cards(instruction)
    feats["hero_card1_rank"] = c1[0] if c1 else None
    feats["hero_card1_suit"] = c1[1] if c1 else None
    feats["hero_card2_rank"] = c2[0] if c2 else None
    feats["hero_card2_suit"] = c2[1] if c2 else None
    if c1 and c2:
        feats["hero_is_suited"] = int(c1[1] == c2[1])
        feats["hero_is_pocket_pair"] = int(c1[0] == c2[0])
        feats["hero_high_rank"] = max(c1[0], c2[0])
        feats["hero_low_rank"] = min(c1[0], c2[0])
        feats["hero_gap"] = abs(c1[0] - c2[0])
    else:
        feats["hero_is_suited"] = None
        feats["hero_is_pocket_pair"] = None
        feats["hero_high_rank"] = None
        feats["hero_low_rank"] = None
        feats["hero_gap"] = None

    # Board
    board = extract_board_cards(instruction)
    feats["board_card_count"] = len(board)
    for i, (rk, su) in enumerate(board[:5]):
        slot = ["flop1", "flop2", "flop3", "turn", "river"][i]
        feats[f"{slot}_rank"] = rk
        feats[f"{slot}_suit"] = su
    if len(board) >= 3:
        ranks = [c[0] for c in board]
        suits = [c[1] for c in board]
        feats["board_paired"] = int(len(set(ranks)) < len(ranks))
        suit_counts = {s: suits.count(s) for s in set(suits)}
        feats["board_max_suit_count"] = max(suit_counts.values())
        feats["board_monotone"] = int(feats["board_max_suit_count"] == len(board))
        feats["board_high_rank"] = max(ranks)
        feats["board_low_rank"] = min(ranks)
    else:
        feats["board_paired"] = None
        feats["board_max_suit_count"] = None
        feats["board_monotone"] = None
        feats["board_high_rank"] = None
        feats["board_low_rank"] = None

    # Action
    pre_chunk = extract_preflop_action_chunk(instruction)
    feats["preflop_aggressor"] = extract_preflop_aggressor(pre_chunk)
    feats["num_preflop_raises"] = extract_num_preflop_raises(pre_chunk)
    feats["preflop_was_hero_aggressor"] = (
        int(feats["preflop_aggressor"] == feats["hero_position"])
        if feats["preflop_aggressor"] and feats["hero_position"] else None
    )
    feats["postflop_action_count"] = extract_postflop_action_count(instruction)

    # Pot / stakes
    feats["pot_size_chips"] = extract_pot_size(instruction)
    if feats["pot_size_chips"] is not None and feats["big_blind"]:
        feats["pot_size_bb"] = feats["pot_size_chips"] / feats["big_blind"]
    else:
        feats["pot_size_bb"] = None

    return feats
