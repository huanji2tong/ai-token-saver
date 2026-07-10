"""File discovery and safe text reading for context packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .privacy import is_sensitive_path, redact_path_text, redact_sensitive_text


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".nuxt",
    ".cache",
    "context-pack.md",
    "shotpack",
}

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".css",
    ".csv",
    ".env.example",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class TextFile:
    path: Path
    rel_path: str
    text: str
    bytes_read: int
    redactions: int = 0


def iter_text_files(
    paths: Iterable[str],
    *,
    root: Path,
    max_file_bytes: int = 200_000,
) -> Iterator[TextFile]:
    root = root.resolve()
    for raw_path in paths:
        path = (root / raw_path).resolve() if raw_path != "-" else Path("-")
        if raw_path == "-":
            continue
        if path.is_dir():
            yield from _walk_dir(path, root=root, max_file_bytes=max_file_bytes)
        elif path.is_file():
            text_file = read_text_file(path, root=root, max_file_bytes=max_file_bytes)
            if text_file:
                yield text_file


def read_text_file(path: Path, *, root: Path, max_file_bytes: int) -> TextFile | None:
    if _should_ignore(path):
        return None
    if path.stat().st_size > max_file_bytes:
        return None
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return None

    redaction = redact_sensitive_text(text)
    rel_path_report = redact_path_text(_safe_relative(path, root))
    return TextFile(
        path=path,
        rel_path=rel_path_report.text,
        text=redaction.text,
        bytes_read=len(raw),
        redactions=redaction.replacements + rel_path_report.replacements,
    )


def _walk_dir(path: Path, *, root: Path, max_file_bytes: int) -> Iterator[TextFile]:
    for item in sorted(path.rglob("*")):
        if not item.is_file() or _should_ignore(item):
            continue
        if item.suffix and item.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text_file = read_text_file(item, root=root, max_file_bytes=max_file_bytes)
        if text_file:
            yield text_file


def _should_ignore(path: Path) -> bool:
    if any(part in DEFAULT_IGNORES or part.endswith(".egg-info") for part in path.parts):
        return True
    return is_sensitive_path(path)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
