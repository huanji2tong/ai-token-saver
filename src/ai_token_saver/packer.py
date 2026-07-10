"""Build token-budgeted context packs from local files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from .files import SkippedPath, scan_text_files
from .optimizer import compact_text, truncate_to_token_budget
from .selection import ContextChunk, build_chunks, extract_terms, make_code_skeleton, select_chunks
from .tokenizer import estimate_tokens, token_savings


_SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|type|const|let|var|struct|enum|fn)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PackedFile:
    path: str
    original_tokens: int
    packed_tokens: int
    status: str
    notes: tuple[str, ...]
    chunks: int = 0


@dataclass(frozen=True)
class PackResult:
    markdown: str
    files: tuple[PackedFile, ...]
    skipped_paths: tuple[SkippedPath, ...]
    source_tokens: int
    packed_tokens: int
    saved_tokens: int
    saved_percent: float
    backend: str


def build_context_pack(
    paths: list[str],
    *,
    root: Path,
    budget_tokens: int,
    model: str | None = None,
    max_file_bytes: int = 200_000,
    query: str = "",
    mode: str = "balanced",
) -> PackResult:
    scan = scan_text_files(paths, root=root, max_file_bytes=max_file_bytes)
    files = list(scan.files)
    packed_files: list[PackedFile] = []
    source_tokens = 0
    backend = "heuristic"
    summary_reserve = min(1000, max(300, budget_tokens // 5))
    body_budget = max(120, budget_tokens - summary_reserve)

    preamble = (
        "# AI Token Saver Context Pack\n\n"
        "This pack was generated locally. It removes duplicate/log noise and keeps "
        "content inside the requested token budget. Sensitive files are skipped "
        "by default and secret-like values are redacted.\n\n"
    )
    raw_by_path: dict[str, int] = {}
    all_chunks: list[ContextChunk] = []
    for text_file in files:
        original = estimate_tokens(text_file.text, model=model)
        source_tokens += original.tokens
        raw_by_path[text_file.rel_path] = original.tokens
        backend = original.backend
        max_chunk_tokens = 220 if mode == "aggressive" else 380
        all_chunks.extend(build_chunks(text_file, model=model, max_chunk_tokens=max_chunk_tokens))
        skeleton = make_code_skeleton(text_file, model=model)
        if skeleton:
            all_chunks.append(skeleton)

    selected = select_chunks(all_chunks, budget_tokens=body_budget, query=query, mode=mode)
    symbol_map = _format_symbol_map(all_chunks)
    symbol_map_tokens = estimate_tokens(symbol_map, model=model).tokens
    if symbol_map_tokens > body_budget // 3:
        symbol_map = truncate_to_token_budget(symbol_map, max(80, body_budget // 3), model=model)
    body = symbol_map + _format_selected_chunks(selected)

    selected_by_path: dict[str, list[ContextChunk]] = defaultdict(list)
    for chunk in selected:
        selected_by_path[chunk.path].append(chunk)

    for text_file in files:
        path_chunks = selected_by_path.get(text_file.rel_path, [])
        if path_chunks:
            notes = []
            if any(chunk.kind == "code-skeleton" for chunk in path_chunks):
                notes.append("repo-map skeleton")
            if query:
                notes.append("evidence-ranked")
            packed_tokens = sum(estimate_tokens(chunk.text, model=model).tokens for chunk in path_chunks)
            packed_files.append(
                PackedFile(
                    path=text_file.rel_path,
                    original_tokens=raw_by_path[text_file.rel_path],
                    packed_tokens=packed_tokens,
                    status="selected",
                    notes=tuple(notes),
                    chunks=len(path_chunks),
                )
            )
        else:
            packed_files.append(
                PackedFile(
                    path=text_file.rel_path,
                    original_tokens=raw_by_path[text_file.rel_path],
                    packed_tokens=0,
                    status="skipped",
                    notes=("lower evidence score",),
                    chunks=0,
                )
            )

    if not selected and files:
        # Fallback for very small budgets: preserve the old deterministic head/tail behavior.
        first = files[0]
        compacted = compact_text(first.text, model=model)
        body = _format_plain_block(first.rel_path, first.path.suffix.lstrip(".") or "text", compacted.text)
        packed_files[0] = PackedFile(
            path=first.rel_path,
            original_tokens=raw_by_path[first.rel_path],
            packed_tokens=estimate_tokens(body, model=model).tokens,
            status="truncated",
            notes=("fallback head/tail block",),
            chunks=1,
        )

    markdown, final_tokens, saved, pct = _assemble_markdown(
        body=body,
        budget_tokens=budget_tokens,
        source_tokens=source_tokens,
        files=packed_files,
        skipped_paths=scan.skipped,
        backend=backend,
        query=query,
        mode=mode,
        model=model,
    )

    return PackResult(
        markdown=markdown,
        files=tuple(packed_files),
        skipped_paths=scan.skipped,
        source_tokens=source_tokens,
        packed_tokens=final_tokens,
        saved_tokens=saved,
        saved_percent=pct,
        backend=backend,
    )


def _assemble_markdown(
    *,
    body: str,
    budget_tokens: int,
    source_tokens: int,
    files: list[PackedFile],
    skipped_paths: tuple[SkippedPath, ...],
    backend: str,
    query: str,
    mode: str,
    model: str | None,
) -> tuple[str, int, int, float]:
    body_for_output = body
    marker = ""
    final_tokens = 0
    saved = 0
    pct = 0.0

    for _ in range(8):
        saved, pct = token_savings(source_tokens, final_tokens)
        summary = _summary(
            budget_tokens=budget_tokens,
            source_tokens=source_tokens,
            packed_tokens=final_tokens,
            saved_tokens=saved,
            saved_percent=pct,
            files=files,
            skipped_paths=skipped_paths,
            backend=backend,
            query=query,
            mode=mode,
            compact=False,
        )
        marker = ""
        summary = _fit_summary(
            summary,
            budget_tokens=budget_tokens,
            source_tokens=source_tokens,
            packed_tokens=final_tokens,
            saved_tokens=saved,
            saved_percent=pct,
            files=files,
            skipped_paths=skipped_paths,
            backend=backend,
            query=query,
            mode=mode,
            model=model,
        )
        summary_tokens = estimate_tokens(summary, model=model).tokens
        if summary_tokens >= budget_tokens:
            final_markdown = truncate_to_token_budget(summary, budget_tokens, model=model)
            final_tokens = estimate_tokens(final_markdown, model=model).tokens
            saved, pct = token_savings(source_tokens, final_tokens)
            return final_markdown, final_tokens, saved, pct

        markdown = summary + marker + body_for_output
        measured_tokens = estimate_tokens(markdown, model=model).tokens
        if measured_tokens == final_tokens and measured_tokens <= budget_tokens:
            return markdown, measured_tokens, saved, pct

        final_tokens = measured_tokens
        if final_tokens <= budget_tokens:
            continue

        marker = "\n\n[... context pack body tightened to fit final token budget ...]\n"
        summary_tokens = estimate_tokens(summary + marker, model=model).tokens
        body_tokens = max(0, budget_tokens - summary_tokens)
        body_for_output = truncate_to_token_budget(body, body_tokens, model=model) if body_tokens else ""

    saved, pct = token_savings(source_tokens, final_tokens)
    summary = _summary(
        budget_tokens=budget_tokens,
        source_tokens=source_tokens,
        packed_tokens=final_tokens,
        saved_tokens=saved,
        saved_percent=pct,
        files=files,
        skipped_paths=skipped_paths,
        backend=backend,
        query=query,
        mode=mode,
        compact=False,
    )
    summary = _fit_summary(
        summary,
        budget_tokens=budget_tokens,
        source_tokens=source_tokens,
        packed_tokens=final_tokens,
        saved_tokens=saved,
        saved_percent=pct,
        files=files,
        skipped_paths=skipped_paths,
        backend=backend,
        query=query,
        mode=mode,
        model=model,
    )
    markdown = summary + marker + body_for_output
    final_markdown = truncate_to_token_budget(markdown, budget_tokens, model=model)
    final_tokens = estimate_tokens(final_markdown, model=model).tokens
    saved, pct = token_savings(source_tokens, final_tokens)
    return final_markdown, final_tokens, saved, pct


def _fit_summary(
    summary: str,
    *,
    budget_tokens: int,
    source_tokens: int,
    packed_tokens: int,
    saved_tokens: int,
    saved_percent: float,
    files: list[PackedFile],
    skipped_paths: tuple[SkippedPath, ...],
    backend: str,
    query: str,
    mode: str,
    model: str | None,
) -> str:
    if estimate_tokens(summary, model=model).tokens <= budget_tokens:
        return summary
    compact_summary = _summary(
        budget_tokens=budget_tokens,
        source_tokens=source_tokens,
        packed_tokens=packed_tokens,
        saved_tokens=saved_tokens,
        saved_percent=saved_percent,
        files=files,
        skipped_paths=skipped_paths,
        backend=backend,
        query=query,
        mode=mode,
        compact=True,
    )
    return compact_summary


def _summary(
    *,
    budget_tokens: int,
    source_tokens: int,
    packed_tokens: int,
    saved_tokens: int,
    saved_percent: float,
    files: list[PackedFile],
    skipped_paths: tuple[SkippedPath, ...],
    backend: str,
    query: str,
    mode: str,
    compact: bool,
) -> str:
    selected = sum(1 for item in files if item.status == "selected")
    truncated = sum(1 for item in files if item.status == "truncated")
    skipped = sum(1 for item in files if item.status == "skipped")
    query_terms = len(set(extract_terms(query)))
    rows = [
        "# AI Token Saver Context Pack",
        "",
        f"- Budget: {budget_tokens:,} tokens",
        f"- Eligible source estimate: {source_tokens:,} tokens",
        f"- Packed estimate: {packed_tokens:,} tokens",
        f"- Estimated savings: {saved_tokens:,} tokens ({saved_percent:.1f}%)",
        f"- Counter: {backend}",
        f"- Strategy: evidence-first/{mode}",
        (
            f"- Query-guided scoring: yes ({query_terms} extracted term(s))"
            if query_terms
            else "- Query-guided scoring: no (structure-only scoring)"
        ),
        f"- Files: {selected} selected, {truncated} truncated, {skipped} skipped",
    ]
    if skipped_paths:
        counts = Counter(item.reason for item in skipped_paths)
        reasons = ", ".join(f"{reason}={counts[reason]}" for reason in sorted(counts))
        rows.append(f"- Skipped before scoring: {len(skipped_paths)} path(s) ({reasons})")
    if compact:
        return "\n".join(rows) + "\n"
    rows.extend(
        [
            "",
            "## File Index",
            "",
            "| file | status | chunks | eligible tokens | packed tokens | notes |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in files:
        notes = "; ".join(item.notes) if item.notes else ""
        rows.append(
            f"| `{item.path}` | {item.status} | {item.chunks:,} | {item.original_tokens:,} | "
            f"{item.packed_tokens:,} | {notes} |"
        )
    return "\n".join(rows) + "\n"


def _format_selected_chunks(chunks: list[ContextChunk]) -> str:
    parts: list[str] = []
    by_path: dict[str, list[ContextChunk]] = defaultdict(list)
    for chunk in chunks:
        by_path[chunk.path].append(chunk)

    for path in sorted(by_path):
        parts.append(f"## {path}\n\n")
        for chunk in sorted(by_path[path], key=lambda item: (item.start_line, item.kind)):
            ext = chunk.suffix.lstrip(".") or "text"
            header = f"### {chunk.kind} lines {chunk.start_line}-{chunk.end_line}"
            parts.append(header + "\n\n")
            parts.append(_format_plain_block(path, ext, chunk.text))
    return "".join(parts)


def _format_symbol_map(chunks: list[ContextChunk]) -> str:
    rows: list[str] = []
    for chunk in chunks:
        if chunk.kind != "code-skeleton":
            continue
        names = _SYMBOL_RE.findall(chunk.text)
        if names:
            rows.append(f"- `{chunk.path}`: " + ", ".join(names[:40]))
    if not rows:
        return ""
    return "## Repository Symbol Map\n\n" + "\n".join(sorted(rows)) + "\n\n"


def _format_plain_block(path: str, ext: str, text: str) -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{ext}\n{text.rstrip()}\n{fence}\n\n"
