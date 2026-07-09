"""Evidence-first context selection.

This module implements a local, deterministic alternative to blind prompt
packing. It does not try to imitate neural prompt compressors. Instead it uses
the same operating principle: remove low-evidence context before it reaches the
LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from pathlib import Path

from .files import TextFile
from .tokenizer import estimate_tokens


_TERM_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_PY_SYMBOL_RE = re.compile(r"^\s*(async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*")
_JS_SYMBOL_RE = re.compile(
    r"^\s*(export\s+)?(async\s+)?(function|class|interface|type|const|let|var)\s+[A-Za-z_$][A-Za-z0-9_$]*"
)
_GENERIC_SYMBOL_RE = re.compile(
    r"^\s*(public|private|protected|static|func|fn|struct|enum|impl|class|interface)\b"
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "that",
    "this",
    "into",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "not",
    "but",
    "can",
    "will",
    "about",
    "under",
    "using",
    "use",
    "文件",
    "代码",
    "项目",
    "这个",
    "一个",
    "需要",
}

CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".ts",
    ".tsx",
}


@dataclass(frozen=True)
class ContextChunk:
    path: str
    suffix: str
    start_line: int
    end_line: int
    text: str
    kind: str
    tokens: int
    score: float = 0.0
    fingerprint: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.path}:{self.start_line}"
        return f"{self.path}:{self.start_line}-{self.end_line}"


def build_chunks(
    text_file: TextFile,
    *,
    model: str | None = None,
    max_chunk_tokens: int = 380,
) -> list[ContextChunk]:
    """Split a file into structure-aware chunks."""

    suffix = text_file.path.suffix.lower()
    lines = text_file.text.splitlines()
    if not lines:
        return []

    if suffix == ".md":
        spans = _heading_spans(lines)
    elif suffix in CODE_SUFFIXES:
        spans = _code_spans(lines, suffix)
    else:
        spans = _paragraph_spans(lines)

    chunks: list[ContextChunk] = []
    for start, end, kind in spans:
        chunk_lines = lines[start - 1 : end]
        chunks.extend(
            _split_oversize_span(
                path=text_file.rel_path,
                suffix=suffix,
                lines=chunk_lines,
                start_line=start,
                kind=kind,
                model=model,
                max_chunk_tokens=max_chunk_tokens,
            )
        )
    return chunks


def make_code_skeleton(text_file: TextFile, *, model: str | None = None) -> ContextChunk | None:
    """Create a compact repo-map style skeleton for code files."""

    suffix = text_file.path.suffix.lower()
    if suffix not in CODE_SUFFIXES:
        return None

    lines = text_file.text.splitlines()
    output: list[str] = []
    skipped = 0
    last_kept = 0
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        keep = False
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ", "package ", "use ", "#include")):
            keep = True
        elif stripped.startswith(("@", "export ", "type ", "interface ")):
            keep = True
        elif _is_symbol_line(line, suffix):
            keep = True
        elif re.match(r"^[A-Z][A-Z0-9_]+\s*=", stripped):
            keep = True

        if keep:
            if number - last_kept > 4 and skipped:
                output.append(f"    # ... {skipped} implementation lines omitted ...")
                skipped = 0
            output.append(line.rstrip())
            last_kept = number
        else:
            skipped += 1

    if not output:
        return None

    skeleton = "\n".join(output[:180])
    if len(output) > 180:
        skeleton += f"\n# ... {len(output) - 180} skeleton lines omitted ..."
    tokens = estimate_tokens(skeleton, model=model).tokens
    return ContextChunk(
        path=text_file.rel_path,
        suffix=suffix,
        start_line=1,
        end_line=len(lines),
        text=skeleton + "\n",
        kind="code-skeleton",
        tokens=tokens,
        fingerprint=frozenset(_terms(skeleton)),
    )


def score_chunks(
    chunks: list[ContextChunk],
    *,
    query: str = "",
    mode: str = "balanced",
) -> list[ContextChunk]:
    query_terms = set(_terms(query))
    scored: list[ContextChunk] = []
    for chunk in chunks:
        terms = set(_terms(chunk.text))
        path_terms = set(_terms(Path(chunk.path).stem.replace("_", " ")))
        query_hits = len(query_terms & (terms | path_terms))
        query_score = query_hits / max(1, len(query_terms)) if query_terms else 0.0

        structural = 0.0
        if chunk.kind == "code-skeleton":
            structural += 2.2
        if chunk.kind in {"code-symbol", "heading"}:
            structural += 1.2
        if chunk.start_line <= 20:
            structural += 0.6
        if any(name in chunk.path.lower() for name in ("readme", "pyproject", "package", "config")):
            structural += 0.8
        if any(name in chunk.path.lower() for name in ("test", "spec")):
            structural += 0.3
        if any(word in terms for word in ("error", "exception", "failed", "失败", "错误")):
            structural += 0.5

        density = min(2.0, len(terms) / max(1, math.sqrt(max(chunk.tokens, 1))) / 6)
        length_penalty = math.log(max(chunk.tokens, 20), 120)
        if mode == "aggressive":
            length_penalty *= 1.35

        score = 1.0 + query_score * 9.0 + structural + density - max(0.0, length_penalty - 1.0)
        scored.append(replace(chunk, score=max(0.05, score), fingerprint=frozenset(terms)))
    return scored


def select_chunks(
    chunks: list[ContextChunk],
    *,
    budget_tokens: int,
    query: str = "",
    mode: str = "balanced",
) -> list[ContextChunk]:
    """Select diverse chunks under a budget using score density."""

    if budget_tokens <= 0:
        return []
    scored = score_chunks(chunks, query=query, mode=mode)
    scored.sort(key=lambda item: _density(item, mode), reverse=True)

    selected: list[ContextChunk] = []
    used = 0
    seen: list[frozenset[str]] = []
    min_tokens = 16 if mode == "aggressive" else 24

    for chunk in scored:
        if chunk.tokens < min_tokens:
            continue
        if used + chunk.tokens > budget_tokens:
            continue
        if _is_near_duplicate(chunk.fingerprint, seen):
            continue
        selected.append(chunk)
        seen.append(chunk.fingerprint)
        used += chunk.tokens

    selected.sort(key=lambda item: (item.path, item.start_line, item.kind))
    return selected


def squeeze_text(
    text: str,
    *,
    budget_tokens: int,
    query: str = "",
    model: str | None = None,
    mode: str = "balanced",
) -> str:
    """Extract the highest-value lines/paragraphs from free-form text."""

    fake_file = TextFile(path=Path("stdin.txt"), rel_path="stdin.txt", text=text, bytes_read=len(text.encode("utf-8")))
    chunks = build_chunks(fake_file, model=model, max_chunk_tokens=180 if mode == "aggressive" else 280)
    selected = select_chunks(chunks, budget_tokens=budget_tokens, query=query, mode=mode)
    if not selected:
        return ""
    parts = []
    for chunk in selected:
        parts.append(f"[lines {chunk.start_line}-{chunk.end_line}]\n{chunk.text.strip()}")
    return "\n\n".join(parts).strip() + "\n"


def extract_terms(text: str) -> list[str]:
    return _terms(text)


def _density(chunk: ContextChunk, mode: str) -> float:
    divisor = max(chunk.tokens, 1)
    if mode == "aggressive":
        divisor = divisor**1.15
    else:
        divisor = math.sqrt(divisor)
    return chunk.score / divisor


def _heading_spans(lines: list[str]) -> list[tuple[int, int, str]]:
    starts = [index + 1 for index, line in enumerate(lines) if _MD_HEADING_RE.match(line)]
    if not starts or starts[0] != 1:
        starts.insert(0, 1)
    spans: list[tuple[int, int, str]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] - 1 if pos + 1 < len(starts) else len(lines)
        spans.append((start, end, "heading"))
    return spans


def _code_spans(lines: list[str], suffix: str) -> list[tuple[int, int, str]]:
    starts = [index + 1 for index, line in enumerate(lines) if _is_symbol_line(line, suffix)]
    if not starts or starts[0] != 1:
        starts.insert(0, 1)
    spans: list[tuple[int, int, str]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] - 1 if pos + 1 < len(starts) else len(lines)
        spans.append((start, end, "code-symbol" if _is_symbol_line(lines[start - 1], suffix) else "code-preamble"))
    return spans


def _paragraph_spans(lines: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    for index, line in enumerate(lines, start=1):
        if line.strip() and start is None:
            start = index
        elif not line.strip() and start is not None:
            spans.append((start, index - 1, "paragraph"))
            start = None
    if start is not None:
        spans.append((start, len(lines), "paragraph"))
    return spans or [(1, len(lines), "paragraph")]


def _split_oversize_span(
    *,
    path: str,
    suffix: str,
    lines: list[str],
    start_line: int,
    kind: str,
    model: str | None,
    max_chunk_tokens: int,
) -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    buffer: list[str] = []
    buffer_start = start_line

    for offset, line in enumerate(lines):
        candidate = buffer + [line]
        if buffer and estimate_tokens("\n".join(candidate), model=model).tokens > max_chunk_tokens:
            chunks.append(_make_chunk(path, suffix, buffer_start, buffer, kind, model))
            buffer = [line]
            buffer_start = start_line + offset
        else:
            buffer = candidate

    if buffer:
        chunks.append(_make_chunk(path, suffix, buffer_start, buffer, kind, model))
    return chunks


def _make_chunk(
    path: str,
    suffix: str,
    start_line: int,
    lines: list[str],
    kind: str,
    model: str | None,
) -> ContextChunk:
    text = "\n".join(lines).strip()
    if text:
        text += "\n"
    return ContextChunk(
        path=path,
        suffix=suffix,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        text=text,
        kind=kind,
        tokens=estimate_tokens(text, model=model).tokens,
        fingerprint=frozenset(_terms(text)),
    )


def _is_symbol_line(line: str, suffix: str) -> bool:
    if suffix == ".py":
        return bool(_PY_SYMBOL_RE.match(line))
    if suffix in {".js", ".jsx", ".mjs", ".ts", ".tsx"}:
        return bool(_JS_SYMBOL_RE.match(line))
    return bool(_GENERIC_SYMBOL_RE.match(line))


def _terms(text: str) -> list[str]:
    terms = []
    for match in _TERM_RE.findall(text.lower()):
        if match in STOPWORDS or len(match) > 48:
            continue
        terms.append(match)
    return terms


def _is_near_duplicate(candidate: frozenset[str], seen: list[frozenset[str]]) -> bool:
    if len(candidate) < 8:
        return False
    for prior in seen:
        if not prior:
            continue
        overlap = len(candidate & prior)
        union = len(candidate | prior)
        if union and overlap / union > 0.82:
            return True
    return False
