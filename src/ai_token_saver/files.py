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


@dataclass(frozen=True)
class SkippedPath:
    path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    files: tuple[TextFile, ...]
    skipped: tuple[SkippedPath, ...]


def iter_text_files(
    paths: Iterable[str],
    *,
    root: Path,
    max_file_bytes: int = 200_000,
) -> Iterator[TextFile]:
    yield from scan_text_files(paths, root=root, max_file_bytes=max_file_bytes).files


def scan_text_files(
    paths: Iterable[str],
    *,
    root: Path,
    max_file_bytes: int = 200_000,
) -> ScanResult:
    root = root.resolve()
    files: list[TextFile] = []
    skipped: list[SkippedPath] = []
    for raw_path in paths:
        path = (root / raw_path).resolve() if raw_path != "-" else Path("-")
        if raw_path == "-":
            continue
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if not item.is_file():
                    continue
                text_file, skipped_path = _scan_candidate(
                    item,
                    root=root,
                    max_file_bytes=max_file_bytes,
                    require_known_suffix=True,
                )
                if text_file:
                    files.append(text_file)
                elif skipped_path:
                    skipped.append(skipped_path)
        elif path.is_file():
            text_file, skipped_path = _scan_candidate(
                path,
                root=root,
                max_file_bytes=max_file_bytes,
                require_known_suffix=False,
            )
            if text_file:
                files.append(text_file)
            elif skipped_path:
                skipped.append(skipped_path)
    return ScanResult(files=tuple(files), skipped=tuple(skipped))


def read_text_file(path: Path, *, root: Path, max_file_bytes: int) -> TextFile | None:
    text_file, _ = _scan_candidate(
        path,
        root=root,
        max_file_bytes=max_file_bytes,
        require_known_suffix=False,
    )
    return text_file


def _should_ignore(path: Path) -> bool:
    return _ignore_reason(path) is not None


def _scan_candidate(
    path: Path,
    *,
    root: Path,
    max_file_bytes: int,
    require_known_suffix: bool,
) -> tuple[TextFile | None, SkippedPath | None]:
    ignore_reason = _ignore_reason(path)
    if ignore_reason:
        return None, _make_skipped(path, root=root, reason=ignore_reason)
    if require_known_suffix and path.suffix and path.suffix.lower() not in TEXT_SUFFIXES:
        return None, _make_skipped(path, root=root, reason="unsupported file type")
    if path.stat().st_size > max_file_bytes:
        return None, _make_skipped(path, root=root, reason="over max file bytes")

    raw = path.read_bytes()
    if b"\x00" in raw:
        return None, _make_skipped(path, root=root, reason="binary file")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return None, _make_skipped(path, root=root, reason="unreadable text encoding")

    redaction = redact_sensitive_text(text)
    rel_path_report = redact_path_text(_safe_relative(path, root))
    return (
        TextFile(
            path=path,
            rel_path=rel_path_report.text,
            text=redaction.text,
            bytes_read=len(raw),
            redactions=redaction.replacements + rel_path_report.replacements,
        ),
        None,
    )


def _ignore_reason(path: Path) -> str | None:
    if any(part in DEFAULT_IGNORES or part.endswith(".egg-info") for part in path.parts):
        return "ignore list"
    if is_sensitive_path(path):
        return "sensitive path"
    return None


def _make_skipped(path: Path, *, root: Path, reason: str) -> SkippedPath:
    rel_path_report = redact_path_text(_safe_relative(path, root))
    return SkippedPath(path=rel_path_report.text, reason=reason)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
