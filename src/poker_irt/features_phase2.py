"""Phase 2 item features: hero-board interactions and stack metrics.

Combines hero hand with board to produce hand-strength, draw, and SPR features
used as covariates in factor interpretation regressions. Heuristic
approximations rather than exact poker evaluations.
"""
from __future__ import annotations

from collections import Counter


def _hero_ranks(hero1, hero2):
    if hero1 is None or hero2 is None:
        return None
    return {hero1[0], hero2[0]}


def _board_ranks(board):
    return [c[0] for c in board]


def hero_paired_with_board(hero1, hero2, board: list) -> int | None:
    if hero1 is None or hero2 is None:
        return None
    if hero1[0] == hero2[0]:
        return 1
    if not board:
        return 0
    return int(bool(_hero_ranks(hero1, hero2) & set(_board_ranks(board))))


def hero_has_top_pair(hero1, hero2, board: list) -> int | None:
    if not board or hero1 is None or hero2 is None:
        return None if not board else 0
    top = max(_board_ranks(board))
    return int(top in _hero_ranks(hero1, hero2))


def hero_has_overpair(hero1, hero2, board: list) -> int | None:
    if not board or hero1 is None or hero2 is None:
        return None if not board else 0
    if hero1[0] != hero2[0]:
        return 0
    top_board = max(_board_ranks(board))
    return int(hero1[0] > top_board)


def hero_has_overcard(hero1, hero2, board: list) -> int | None:
    if not board or hero1 is None or hero2 is None:
        return None if not board else 0
    top_board = max(_board_ranks(board))
    return int(any(r > top_board for r in _hero_ranks(hero1, hero2)))


def hero_has_flush_draw(hero1, hero2, board: list) -> int | None:
    """Suited hero with two of that suit on board (flop or turn only)."""
    if hero1 is None or hero2 is None:
        return None
    if hero1[1] != hero2[1]:
        return 0
    if not board or len(board) > 4:
        return None if not board else 0
    suit = hero1[1]
    return int(sum(1 for c in board if c[1] == suit) == 2)


def hero_flush_made(hero1, hero2, board: list) -> int | None:
    if hero1 is None or hero2 is None or not board:
        return None
    all_suits = [hero1[1], hero2[1]] + [c[1] for c in board]
    suit_count = Counter(all_suits)
    return int(max(suit_count.values()) >= 5)


def _has_straight(rank_set: set[int]) -> bool:
    if not rank_set:
        return False
    if {14, 2, 3, 4, 5}.issubset(rank_set):  # ace-low straight
        return True
    for low in range(2, 11):
        if all(r in rank_set for r in range(low, low + 5)):
            return True
    return False


def hero_straight_draw(hero1, hero2, board: list) -> int | None:
    """OESD or gutshot: any single rank that would complete a 5-card straight."""
    if hero1 is None or hero2 is None or not board:
        return None
    cards = [hero1[0], hero2[0]] + [c[0] for c in board]
    cs = set(cards)
    for cand in range(2, 15):
        if cand in cs:
            continue
        if _has_straight(cs | {cand}):
            return 1
    return 0


def hand_strength_class(hero1, hero2, board: list) -> int:
    """0=high card, 1=pair, 2=two pair, 3=trips/set, 4=straight, 5=flush, 6=quads or full house."""
    if hero1 is None or hero2 is None:
        return 0
    cards = [hero1, hero2] + list(board or [])
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    rank_count = Counter(ranks)
    suit_count = Counter(suits)
    if 4 in rank_count.values():
        return 6
    if 3 in rank_count.values() and any(v >= 2 for k, v in rank_count.items() if v != 3):
        return 6
    if max(suit_count.values()) >= 5:
        return 5
    if _has_straight(set(ranks)):
        return 4
    if 3 in rank_count.values():
        return 3
    if list(rank_count.values()).count(2) >= 2:
        return 2
    if 2 in rank_count.values():
        return 1
    return 0


def board_straight_possible(board: list) -> int | None:
    if not board or len(board) < 3:
        return None
    ranks = sorted(set(c[0] for c in board))
    if len(ranks) < 3:
        return 0
    for i in range(len(ranks) - 2):
        if ranks[i + 2] - ranks[i] <= 4:
            return 1
    return 0


def effective_stack_bb(starting_stack, big_blind, pot_size_chips) -> float | None:
    if starting_stack is None or big_blind is None or pot_size_chips is None:
        return None
    if big_blind <= 0:
        return None
    contribution = pot_size_chips / 2.0
    return max(starting_stack - contribution, 0.0) / big_blind


def spr_value(starting_stack, big_blind, pot_size_chips) -> float | None:
    eff = effective_stack_bb(starting_stack, big_blind, pot_size_chips)
    if eff is None or pot_size_chips is None or big_blind is None or pot_size_chips <= 0:
        return None
    pot_bb = pot_size_chips / big_blind
    return eff / pot_bb if pot_bb > 0 else None


def spr_bucket(starting_stack, big_blind, pot_size_chips) -> int | None:
    s = spr_value(starting_stack, big_blind, pot_size_chips)
    if s is None:
        return None
    if s < 2:    return 0
    if s < 5:    return 1
    if s < 15:   return 2
    return 3


def features_phase2_for_item(
    hero_card1, hero_card2, board: list,
    starting_stack, big_blind, pot_size_chips,
) -> dict:
    """Return Phase 2 features given a parsed hand state."""
    return {
        "hero_paired_with_board": hero_paired_with_board(hero_card1, hero_card2, board),
        "hero_has_top_pair":      hero_has_top_pair(hero_card1, hero_card2, board),
        "hero_has_overpair":      hero_has_overpair(hero_card1, hero_card2, board),
        "hero_has_overcard":      hero_has_overcard(hero_card1, hero_card2, board),
        "hero_has_flush_draw":    hero_has_flush_draw(hero_card1, hero_card2, board),
        "hero_flush_made":        hero_flush_made(hero_card1, hero_card2, board),
        "hero_straight_draw":     hero_straight_draw(hero_card1, hero_card2, board),
        "hand_strength_class":    hand_strength_class(hero_card1, hero_card2, board),
        "board_straight_possible": board_straight_possible(board),
        "effective_stack_bb":     effective_stack_bb(starting_stack, big_blind, pot_size_chips),
        "spr_value":              spr_value(starting_stack, big_blind, pot_size_chips),
        "spr_bucket":             spr_bucket(starting_stack, big_blind, pot_size_chips),
    }
