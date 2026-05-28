"""Action Accuracy (AA) and Exact Match (EM) scoring."""
from __future__ import annotations

SIZING_TOLERANCE = 0.5  # chips


def score(
    parsed: tuple[str, float | None] | None,
    gto: tuple[str, float | None],
) -> tuple[int, int]:
    """Return ``(AA, EM)``. AA matches action class; EM also matches sizing within tolerance."""
    if parsed is None:
        return (0, 0)
    aa = int(parsed[0] == gto[0])
    if not aa:
        return (0, 0)
    gto_size = gto[1]
    parsed_size = parsed[1]
    if gto_size is None:
        em = int(parsed_size is None)
    else:
        em = int(parsed_size is not None and abs(parsed_size - gto_size) < SIZING_TOLERANCE)
    return (aa, em)
