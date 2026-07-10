"""Privacy guards for context packing.

The goal is conservative defaults: skip obviously sensitive files and redact
secret-like values from text that is still eligible for packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SAFE_HIDDEN_PARTS = {
    ".github",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
    ".env.example",
    ".env.sample",
    ".env.template",
    ".flake8",
    ".npmignore",
    ".prettierignore",
    ".prettierrc",
    ".ruff.toml",
}

SENSITIVE_EXACT_NAMES = {
    ".env",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
}

SENSITIVE_SUFFIXES = {
    ".asc",
    ".cer",
    ".crt",
    ".env",
    ".key",
    ".kdbx",
    ".p12",
    ".pem",
    ".pfx",
    ".p7b",
    ".p8",
}

SENSITIVE_NAME_TOKENS = {
    "apikey",
    "appsecret",
    "credential",
    "credentials",
    "passwd",
    "password",
    "private",
    "secret",
    "secrets",
    "token",
    "tokens",
}

SAFE_EXAMPLE_MARKERS = {"example", "sample", "template", "demo"}

SECRET_FIELD_PATTERN = (
    r"api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"aws[_-]?secret[_-]?access[_-]?key|aws[_-]?session[_-]?token|"
    r"password|passwd|secret|token"
)

_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_PRIVATE_KEY_BLOCK_RE, "<REDACTED:PRIVATE_KEY_BLOCK>"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"), "<REDACTED:OPENAI_KEY>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<REDACTED:GITHUB_TOKEN>"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "<REDACTED:ANTHROPIC_KEY>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "<REDACTED:GOOGLE_API_KEY>"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "<REDACTED:SLACK_TOKEN>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<REDACTED:AWS_ACCESS_KEY>"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}"), r"\1<REDACTED:BEARER_TOKEN>"),
    (
        re.compile(
            rf"(?im)^(\s*[{{\[]?\s*[\"']?(?:{SECRET_FIELD_PATTERN})[\"']?\s*[:=]\s*)([\"']?)"
            r"((?!<REDACTED:)[^\"'\n,#;\}\]]{6,})\2(\s*(?:[,#;].*)?)$"
        ),
        r"\1\2<REDACTED:SECRET_VALUE>\2\4",
    ),
    (
        re.compile(
            rf"(?i)([\"'](?:{SECRET_FIELD_PATTERN})[\"']\s*:\s*)([\"'])((?!<REDACTED:)[^\"'\n]{{6,}})\2"
        ),
        r"\1\2<REDACTED:SECRET_VALUE>\2",
    ),
    (
        re.compile(r"(https?://[^/\s:@]+:)([^@\s/]+)(@)"),
        r"\1<REDACTED:URL_PASSWORD>\3",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "<REDACTED:EMAIL>",
    ),
    (
        re.compile(r"(?<!\w)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\w)"),
        "<REDACTED:CN_PHONE>",
    ),
    (
        re.compile(r"(?<!\w)(?:\+?1[- ]?)?(?:\(\d{3}\)|\d{3})[- ]?\d{3}[- ]?\d{4}(?!\w)"),
        "<REDACTED:US_PHONE>",
    ),
]


@dataclass(frozen=True)
class RedactionReport:
    text: str
    replacements: int


def is_sensitive_path(path: Path) -> bool:
    parts = [part for part in path.parts if part not in {".", "..", "/"}]
    for part in parts:
        lowered = part.lower()
        if lowered in {".git", ".hg", ".svn"}:
            continue
        if lowered.startswith(".") and lowered not in SAFE_HIDDEN_PARTS:
            return True

    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in SENSITIVE_EXACT_NAMES or suffix in SENSITIVE_SUFFIXES:
        return True
    if name.startswith(".env") and name not in SAFE_HIDDEN_PARTS:
        return True

    tokens = [token for token in re.split(r"[^a-z0-9]+", path.stem.lower()) if token]
    sensitive_tokens = any(token in SENSITIVE_NAME_TOKENS for token in tokens)
    if any(token in SAFE_EXAMPLE_MARKERS for token in tokens) and not sensitive_tokens:
        return False
    return sensitive_tokens


def redact_sensitive_text(text: str) -> RedactionReport:
    redacted = text
    replacements = 0
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        replacements += count
    return RedactionReport(text=redacted, replacements=replacements)


def redact_path_text(path_text: str) -> RedactionReport:
    parts = []
    replacements = 0
    for segment in path_text.split("/"):
        segment_path = Path(segment)
        if segment_path.suffix:
            base = segment_path.stem
            suffix = segment_path.suffix
            report = redact_sensitive_text(base)
            if report.replacements:
                parts.append(report.text + suffix)
                replacements += report.replacements
                continue
        report = redact_sensitive_text(segment)
        parts.append(report.text)
        replacements += report.replacements
    return RedactionReport(text="/".join(parts), replacements=replacements)
