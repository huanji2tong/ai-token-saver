"""Deterministic prompt and log compaction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .selection import squeeze_text
from .tokenizer import estimate_tokens, token_savings


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class CompactResult:
    text: str
    original_tokens: int
    compact_tokens: int
    saved_tokens: int
    saved_percent: float
    notes: tuple[str, ...]
    backend: str


def compact_text(
    text: str,
    *,
    head_lines: int = 80,
    tail_lines: int = 80,
    model: str | None = None,
    budget_tokens: int | None = None,
    query: str = "",
    strategy: str = "clean",
) -> CompactResult:
    """Compact common prompt waste while keeping content readable.

    The transform is intentionally deterministic: normalize whitespace, remove
    ANSI terminal noise, collapse repeated consecutive lines, then summarize
    oversized logs by keeping the head and tail.
    """

    original = estimate_tokens(text, model=model)
    notes: list[str] = []

    cleaned = _ANSI_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in cleaned.split("\n")]
    if cleaned != text:
        notes.append("normalized terminal/control noise")

    lines, blank_removed = _collapse_blank_lines(lines, max_blank_run=1)
    if blank_removed:
        notes.append(f"collapsed {blank_removed} extra blank lines")

    lines, repeat_removed = _collapse_repeated_lines(lines)
    if repeat_removed:
        notes.append(f"collapsed {repeat_removed} repeated lines")

    if len(lines) > head_lines + tail_lines + 20:
        omitted = len(lines) - head_lines - tail_lines
        lines = (
            lines[:head_lines]
            + [f"[... omitted {omitted} middle lines to fit context ...]"]
            + lines[-tail_lines:]
        )
        notes.append(f"kept first {head_lines} and last {tail_lines} lines")

    compacted_text = "\n".join(lines).strip() + ("\n" if lines else "")
    if strategy == "extractive" and budget_tokens is not None:
        squeezed = squeeze_text(
            compacted_text,
            budget_tokens=budget_tokens,
            query=query,
            model=model,
            mode="aggressive",
        )
        if squeezed:
            compacted_text = squeezed
            if estimate_tokens(compacted_text, model=model).tokens > budget_tokens:
                compacted_text = truncate_to_token_budget(compacted_text, budget_tokens, model=model)
                notes.append("tightened extractive output to final budget")
            notes.append(f"extractive squeeze to {budget_tokens} token budget")

    compacted = estimate_tokens(compacted_text, model=model)
    saved, pct = token_savings(original.tokens, compacted.tokens)

    return CompactResult(
        text=compacted_text,
        original_tokens=original.tokens,
        compact_tokens=compacted.tokens,
        saved_tokens=saved,
        saved_percent=pct,
        notes=tuple(notes),
        backend=compacted.backend,
    )


def truncate_to_token_budget(text: str, budget_tokens: int, *, model: str | None = None) -> str:
    """Return a readable head/tail truncation that fits the approximate budget."""

    if estimate_tokens(text, model=model).tokens <= budget_tokens:
        return text
    if budget_tokens <= 24:
        return "[... omitted: token budget too small ...]\n"

    approx_chars = max(80, budget_tokens * 4)
    marker = "\n[... truncated to fit token budget ...]\n"
    keep = max(40, (approx_chars - len(marker)) // 2)
    truncated = text[:keep].rstrip() + marker + text[-keep:].lstrip()

    # The fallback estimate is approximate, so tighten in a few passes.
    for _ in range(8):
        if estimate_tokens(truncated, model=model).tokens <= budget_tokens:
            return truncated
        keep = max(20, int(keep * 0.82))
        truncated = text[:keep].rstrip() + marker + text[-keep:].lstrip()
    return truncated


def _collapse_blank_lines(lines: list[str], *, max_blank_run: int) -> tuple[list[str], int]:
    output: list[str] = []
    blanks = 0
    removed = 0
    for line in lines:
        if line.strip():
            blanks = 0
            output.append(line)
            continue
        blanks += 1
        if blanks <= max_blank_run:
            output.append("")
        else:
            removed += 1
    return output, removed


def _collapse_repeated_lines(lines: list[str]) -> tuple[list[str], int]:
    output: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        run = 1
        while index + run < len(lines) and lines[index + run] == line:
            run += 1
        if line.strip() and run >= 4:
            output.append(line)
            output.append(f"[... repeated {run - 1} more times ...]")
            removed += run - 2
        else:
            output.extend(lines[index : index + run])
        index += run
    return output, removed
