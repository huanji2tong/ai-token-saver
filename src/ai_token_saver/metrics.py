"""Loss and retention metrics for context compression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import re

from .files import iter_text_files
from .packer import build_context_pack
from .selection import build_chunks, extract_terms, make_code_skeleton, select_chunks
from .tokenizer import estimate_tokens


_SYMBOL_NAME_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|const|let|var|struct|enum|fn)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_SYMBOL_MAP_RE = re.compile(r"^- `[^`]+`:\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class LossReport:
    source_tokens: int
    packed_tokens: int
    saved_tokens: int
    saved_percent: float
    token_growth_tokens: int
    token_growth_percent: float
    token_retention_percent: float
    token_removal_percent: float
    query_term_recall_percent: float
    symbol_recall_percent: float
    file_coverage_percent: float
    chunk_coverage_percent: float
    critical_retention_percent: float
    estimated_loss_percent: float
    source_files: int
    selected_files: int
    source_chunks: int
    selected_chunks: int
    skipped_paths: int
    skip_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, float | int]:
        return {
            "eligible_source_tokens": self.source_tokens,
            "source_tokens": self.source_tokens,
            "packed_tokens": self.packed_tokens,
            "saved_tokens": self.saved_tokens,
            "saved_percent": self.saved_percent,
            "token_growth_tokens": self.token_growth_tokens,
            "token_growth_percent": self.token_growth_percent,
            "token_retention_percent": self.token_retention_percent,
            "token_removal_percent": self.token_removal_percent,
            "query_term_recall_percent": self.query_term_recall_percent,
            "symbol_recall_percent": self.symbol_recall_percent,
            "file_coverage_percent": self.file_coverage_percent,
            "chunk_coverage_percent": self.chunk_coverage_percent,
            "critical_retention_percent": self.critical_retention_percent,
            "estimated_loss_percent": self.estimated_loss_percent,
            "source_files": self.source_files,
            "selected_files": self.selected_files,
            "source_chunks": self.source_chunks,
            "selected_chunks": self.selected_chunks,
            "skipped_paths": self.skipped_paths,
            "skip_counts": dict(self.skip_counts),
        }


def analyze_loss(
    paths: list[str],
    *,
    root: Path,
    budget_tokens: int,
    query: str = "",
    mode: str = "balanced",
    model: str | None = None,
    max_file_bytes: int = 200_000,
) -> LossReport:
    files = list(iter_text_files(paths, root=root, max_file_bytes=max_file_bytes))
    pack = build_context_pack(
        paths,
        root=root,
        budget_tokens=budget_tokens,
        query=query,
        mode=mode,
        model=model,
        max_file_bytes=max_file_bytes,
    )

    max_chunk_tokens = 220 if mode == "aggressive" else 380
    chunks = []
    source_text_parts = []
    for text_file in files:
        source_text_parts.append(text_file.text)
        chunks.extend(build_chunks(text_file, model=model, max_chunk_tokens=max_chunk_tokens))
        skeleton = make_code_skeleton(text_file, model=model)
        if skeleton:
            chunks.append(skeleton)

    body_budget = max(120, budget_tokens - min(1000, max(300, budget_tokens // 5)))
    selected = select_chunks(chunks, budget_tokens=body_budget, query=query, mode=mode)
    source_text = "\n".join(source_text_parts)
    packed_text = pack.body_markdown

    query_recall = _query_recall(source_text, packed_text, query)
    symbol_recall = _set_recall(_symbols(source_text), _symbols(packed_text))
    source_file_paths = {text_file.rel_path for text_file in files}
    selected_file_paths = {
        item.path for item in pack.files if item.status in {"selected", "truncated"}
    }
    file_coverage = _set_recall(source_file_paths, selected_file_paths)
    selected_chunk_count = len(selected) if selected else sum(
        max(1, item.chunks) for item in pack.files if item.status == "truncated"
    )
    chunk_coverage = selected_chunk_count / len(chunks) if chunks else 1.0

    # Weighted toward task evidence, then code navigation safety, then breadth.
    critical_retention = (
        query_recall * 0.45
        + symbol_recall * 0.30
        + file_coverage * 0.15
        + min(1.0, chunk_coverage * 3.0) * 0.10
    )
    estimated_loss = max(0.0, 1.0 - critical_retention)
    if pack.source_tokens:
        token_retention = min(1.0, pack.packed_tokens / pack.source_tokens)
        token_growth_tokens = max(0, pack.packed_tokens - pack.source_tokens)
        token_growth_percent = (token_growth_tokens / pack.source_tokens) * 100 if token_growth_tokens else 0.0
    elif pack.packed_tokens:
        token_retention = 0.0
        token_growth_tokens = pack.packed_tokens
        token_growth_percent = 0.0
    else:
        token_retention = 0.0
        token_growth_tokens = 0
        token_growth_percent = 0.0
    token_removal_percent = 0.0 if not pack.source_tokens else max(0.0, (1 - token_retention) * 100)

    return LossReport(
        source_tokens=pack.source_tokens,
        packed_tokens=pack.packed_tokens,
        saved_tokens=pack.saved_tokens,
        saved_percent=pack.saved_percent,
        token_growth_tokens=token_growth_tokens,
        token_growth_percent=token_growth_percent,
        token_retention_percent=token_retention * 100,
        token_removal_percent=token_removal_percent,
        query_term_recall_percent=query_recall * 100,
        symbol_recall_percent=symbol_recall * 100,
        file_coverage_percent=file_coverage * 100,
        chunk_coverage_percent=chunk_coverage * 100,
        critical_retention_percent=critical_retention * 100,
        estimated_loss_percent=estimated_loss * 100,
        source_files=len(source_file_paths),
        selected_files=len(selected_file_paths),
        source_chunks=len(chunks),
        selected_chunks=selected_chunk_count,
        skipped_paths=len(pack.skipped_paths),
        skip_counts=_skip_counts(pack.skipped_paths),
    )


def _query_recall(source_text: str, packed_text: str, query: str) -> float:
    query_terms = set(extract_terms(query))
    if not query_terms:
        return 1.0
    source_terms = set(extract_terms(source_text))
    packed_terms = set(extract_terms(packed_text))
    relevant_terms = query_terms & source_terms
    if not relevant_terms:
        return 1.0
    return len(relevant_terms & packed_terms) / len(relevant_terms)


def _symbols(text: str) -> set[str]:
    names = {match.group(1) for match in _SYMBOL_NAME_RE.finditer(text)}
    for match in _SYMBOL_MAP_RE.finditer(text):
        for item in match.group(1).split(","):
            name = item.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                names.add(name)
    return names


def _set_recall(source: set[str], packed: set[str]) -> float:
    if not source:
        return 1.0
    return len(source & packed) / len(source)


def _skip_counts(skipped_paths: tuple[object, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped_paths:
        reason = str(getattr(item, "reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts
