"""Token estimation with an optional tiktoken fast path.

The package is useful without network access or API keys. If tiktoken is
installed, estimates use the requested model encoding. Otherwise a conservative
regex heuristic is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Optional


_PIECE_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)


@dataclass(frozen=True)
class TokenEstimate:
    tokens: int
    chars: int
    backend: str

    def cost(self, price_per_million: float) -> float:
        return (self.tokens / 1_000_000) * price_per_million


def estimate_tokens(text: str, model: Optional[str] = None) -> TokenEstimate:
    """Estimate token count for text.

    Parameters
    ----------
    text:
        Text to estimate.
    model:
        Optional model name used only when tiktoken is available.
    """

    if not text:
        return TokenEstimate(tokens=0, chars=0, backend="empty")

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        return TokenEstimate(tokens=len(encoding.encode(text)), chars=len(text), backend="tiktoken")
    except Exception:
        return TokenEstimate(tokens=_fallback_count(text), chars=len(text), backend="heuristic")


def _fallback_count(text: str) -> int:
    pieces = _PIECE_RE.findall(text)
    total = 0
    for piece in pieces:
        if not piece:
            continue
        if piece.isascii() and (piece[0].isalnum() or piece[0] == "_"):
            total += max(1, math.ceil(len(piece) / 4))
        elif len(piece) == 1:
            total += 1
        else:
            total += max(1, math.ceil(len(piece) / 2))
    return total


def token_savings(before: int, after: int) -> tuple[int, float]:
    saved = max(0, before - after)
    percent = (saved / before * 100) if before else 0.0
    return saved, percent
