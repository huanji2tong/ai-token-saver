"""Render context packs into dense PNG pages for multimodal LLMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import textwrap

from .packer import build_context_pack


@dataclass(frozen=True)
class ImagePage:
    path: str
    lines: int
    width: int
    height: int


@dataclass(frozen=True)
class ShotpackResult:
    output_dir: Path
    markdown_path: Path
    manifest_path: Path
    pages: tuple[ImagePage, ...]
    source_tokens: int
    text_pack_tokens: int
    text_saved_percent: float


def build_shotpack(
    paths: list[str],
    *,
    root: Path,
    output_dir: Path,
    budget_tokens: int,
    query: str = "",
    mode: str = "aggressive",
    model: str | None = None,
    max_file_bytes: int = 200_000,
    stem: str = "context",
    width: int = 1800,
    height: int = 2400,
    font_size: int = 20,
    columns: int = 2,
) -> ShotpackResult:
    pack = build_context_pack(
        paths,
        root=root,
        budget_tokens=budget_tokens,
        query=query,
        mode=mode,
        model=model,
        max_file_bytes=max_file_bytes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(pack.markdown, encoding="utf-8")

    pages = render_text_pages(
        pack.markdown,
        output_dir=output_dir,
        stem=stem,
        width=width,
        height=height,
        font_size=font_size,
        columns=columns,
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "strategy": "evidence-first + optical screenshot pack",
        "source_tokens": pack.source_tokens,
        "text_pack_tokens": pack.packed_tokens,
        "text_saved_percent": pack.saved_percent,
        "query": query,
        "mode": mode,
        "loss_note": (
            "PNG pages are for visual/context reading. Keep the markdown pack for exact "
            "identifiers, code edits, numbers, secrets, and copy/paste-sensitive text."
        ),
        "pages": [page.__dict__ for page in pages],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ShotpackResult(
        output_dir=output_dir,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        pages=tuple(pages),
        source_tokens=pack.source_tokens,
        text_pack_tokens=pack.packed_tokens,
        text_saved_percent=pack.saved_percent,
    )


def render_text_pages(
    text: str,
    *,
    output_dir: Path,
    stem: str,
    width: int,
    height: int,
    font_size: int,
    columns: int,
) -> tuple[ImagePage, ...]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "shotpack requires Pillow. Install with: python -m pip install -e '.[image]'"
        ) from exc

    font = _load_font(ImageFont, font_size)
    margin = max(28, font_size * 2)
    gutter = max(24, font_size * 2)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
    bbox = draw_probe.textbbox((0, 0), "M", font=font)
    char_width = max(1, bbox[2] - bbox[0])
    line_height = max(font_size + 5, bbox[3] - bbox[1] + 6)
    column_width = (width - margin * 2 - gutter * (columns - 1)) // columns
    wrap_width = max(24, column_width // char_width)
    lines = _wrap_text(text, width=wrap_width)
    lines_per_column = max(1, (height - margin * 2) // line_height)
    lines_per_page = max(1, lines_per_column * columns)

    pages: list[ImagePage] = []
    for page_index, start in enumerate(range(0, len(lines), lines_per_page), start=1):
        page_lines = lines[start : start + lines_per_page]
        image = Image.new("RGB", (width, height), "#fbfaf8")
        draw = ImageDraw.Draw(image)
        for local_index, line in enumerate(page_lines):
            col = local_index // lines_per_column
            row = local_index % lines_per_column
            x = margin + col * (column_width + gutter)
            y = margin + row * line_height
            draw.text((x, y), line, fill="#111827", font=font)
        page_path = output_dir / f"{stem}-{page_index:03d}.png"
        image.save(page_path, optimize=True)
        pages.append(ImagePage(path=page_path.name, lines=len(page_lines), width=width, height=height))
    return tuple(pages)


def _wrap_text(text: str, *, width: int) -> list[str]:
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped.append("")
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        prefix = " " * min(indent, 8)
        pieces = textwrap.wrap(
            raw_line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
            subsequent_indent=prefix,
        )
        wrapped.extend(pieces or [""])
    return wrapped


def _load_font(ImageFont, font_size: int):
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)
    return ImageFont.load_default()
