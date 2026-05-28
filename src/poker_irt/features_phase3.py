"""Phase 3 features: hand equity vs a uniformly random villain via ``treys``."""
from __future__ import annotations

from functools import lru_cache

from treys import Card, Evaluator, Deck

_eval = Evaluator()


_TREYS_RANK = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
               9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
_TREYS_SUIT = {"H": "h", "D": "d", "S": "s", "C": "c"}


def _to_treys(card):
    if card is None:
        return None
    rank, suit = card
    return Card.new(_TREYS_RANK[rank] + _TREYS_SUIT[suit])


def hero_equity_vs_random(hero1, hero2, board, n_samples: int | None = None) -> float | None:
    """Equity of hero vs a uniformly random villain hand. Ties count 0.5."""
    if hero1 is None or hero2 is None:
        return None
    h1, h2 = _to_treys(hero1), _to_treys(hero2)
    b = [_to_treys(c) for c in (board or [])]
    used = {h1, h2, *b}
    all_cards = []
    for r in range(2, 15):
        for s in "HDSC":
            c = Card.new(_TREYS_RANK[r] + _TREYS_SUIT[s])
            if c not in used:
                all_cards.append(c)

    if not b or len(b) < 3:
        # Preflop: must also deal the full board; sample n_samples runouts.
        if n_samples is None:
            n_samples = 200
        import random
        rng = random.Random(0)
        wins = ties = total = 0
        for _ in range(n_samples):
            sample = rng.sample(all_cards, 5 + 2)  # board + villain hand
            board_s = sample[:5]
            v1, v2 = sample[5], sample[6]
            hero_rank  = _eval.evaluate(board_s, [h1, h2])
            vill_rank  = _eval.evaluate(board_s, [v1, v2])
            if hero_rank < vill_rank:  wins  += 1   # lower rank = stronger hand in treys
            elif hero_rank == vill_rank: ties  += 1
            total += 1
        if total == 0:
            return None
        return (wins + 0.5 * ties) / total

    # Postflop branches: sample remaining streets and villain hand.
    if len(b) == 5:
        if n_samples is None:
            n_samples = 200
        import random
        rng = random.Random(0)
        hero_rank = _eval.evaluate(b, [h1, h2])
        wins = ties = total = 0
        for _ in range(n_samples):
            v1, v2 = rng.sample(all_cards, 2)
            vill_rank = _eval.evaluate(b, [v1, v2])
            if hero_rank < vill_rank: wins  += 1
            elif hero_rank == vill_rank: ties  += 1
            total += 1
        return (wins + 0.5 * ties) / total
    elif len(b) == 4:
        if n_samples is None:
            n_samples = 100
        import random
        rng = random.Random(0)
        wins = ties = total = 0
        for _ in range(n_samples):
            sample = rng.sample(all_cards, 1 + 2)
            board_done = b + [sample[0]]
            v1, v2 = sample[1], sample[2]
            hero_rank  = _eval.evaluate(board_done, [h1, h2])
            vill_rank  = _eval.evaluate(board_done, [v1, v2])
            if hero_rank < vill_rank: wins  += 1
            elif hero_rank == vill_rank: ties  += 1
            total += 1
        return (wins + 0.5 * ties) / total
    elif len(b) == 3:
        if n_samples is None:
            n_samples = 80
        import random
        rng = random.Random(0)
        wins = ties = total = 0
        for _ in range(n_samples):
            sample = rng.sample(all_cards, 2 + 2)
            board_done = b + sample[:2]
            v1, v2 = sample[2], sample[3]
            hero_rank  = _eval.evaluate(board_done, [h1, h2])
            vill_rank  = _eval.evaluate(board_done, [v1, v2])
            if hero_rank < vill_rank: wins  += 1
            elif hero_rank == vill_rank: ties  += 1
            total += 1
        return (wins + 0.5 * ties) / total
    return None


def hero_made_hand_rank(hero1, hero2, board) -> int | None:
    """``treys`` rank int (1 = royal flush, 7462 = high card); requires >= 3 board cards."""
    if hero1 is None or hero2 is None or not board or len(board) < 3:
        return None
    h1, h2 = _to_treys(hero1), _to_treys(hero2)
    b = [_to_treys(c) for c in board]
    return int(_eval.evaluate(b, [h1, h2]))


def features_phase3_for_item(hero1, hero2, board, n_samples: int = 80) -> dict:
    return {
        "hero_equity_vs_random":  hero_equity_vs_random(hero1, hero2, board, n_samples=n_samples),
        "hero_made_hand_rank":    hero_made_hand_rank(hero1, hero2, board),
    }
