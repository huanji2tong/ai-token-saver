"""Command line interface for ai-token-saver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .files import scan_text_files
from .metrics import analyze_loss
from .optimizer import compact_text
from .packer import build_context_pack
from .privacy import is_sensitive_path, redact_path_text, redact_sensitive_text
from .render import build_shotpack
from .tokenizer import estimate_tokens, token_savings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-token-saver",
        description="Audit, trim, and pack AI context to reduce token spend.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Estimate privacy-filtered token hotspots in files or directories.")
    audit.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan.")
    audit.add_argument("--root", default=".", help="Root directory used for relative paths.")
    audit.add_argument("--model", default=None, help="Optional tokenizer model name if tiktoken is installed.")
    audit.add_argument("--budget", type=int, default=8000, help="Reference token budget.")
    audit.add_argument("--max-file-bytes", type=int, default=200_000, help="Skip files larger than this.")
    audit.add_argument("--price-per-million", type=float, default=1.0, help="Input token price for cost estimate.")
    audit.add_argument("--show-skipped", action="store_true", help="Show skipped paths and skip reasons.")
    audit.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    audit.set_defaults(func=_cmd_audit)

    trim = subparsers.add_parser("trim", help="Compact one text file or stdin.")
    trim.add_argument("input", nargs="?", default="-", help="Input file, or '-' for stdin.")
    trim.add_argument("-o", "--output", help="Write compacted text to a file.")
    trim.add_argument("--model", default=None, help="Optional tokenizer model name if tiktoken is installed.")
    trim.add_argument("--head-lines", type=int, default=80, help="Lines kept from the beginning of long input.")
    trim.add_argument("--tail-lines", type=int, default=80, help="Lines kept from the end of long input.")
    trim.add_argument("--strategy", choices=["clean", "extractive"], default="clean", help="Compaction strategy.")
    trim.add_argument("--budget", type=int, help="Target token budget for extractive compaction.")
    trim.add_argument("--query", default="", help="Task/query used to keep relevant evidence.")
    trim.add_argument("--no-redact", action="store_true", help="Disable built-in secret and PII redaction.")
    trim.add_argument(
        "--allow-sensitive-path",
        action="store_true",
        help="Allow direct reads from sensitive paths such as credentials or private key files.",
    )
    trim.add_argument("--json", action="store_true", help="Print report as JSON to stderr.")
    trim.set_defaults(func=_cmd_trim)

    pack = subparsers.add_parser("pack", help="Create a markdown context pack under a token budget.")
    pack.add_argument("paths", nargs="*", default=["."], help="Files or directories to include.")
    pack.add_argument("--root", default=".", help="Root directory used for relative paths.")
    pack.add_argument("-b", "--budget", type=int, default=8000, help="Target token budget.")
    pack.add_argument("-o", "--output", default="context-pack.md", help="Output markdown file.")
    pack.add_argument("--model", default=None, help="Optional tokenizer model name if tiktoken is installed.")
    pack.add_argument("--max-file-bytes", type=int, default=200_000, help="Skip files larger than this.")
    pack.add_argument("--query", default="", help="Task/query used for evidence scoring.")
    pack.add_argument("--mode", choices=["balanced", "aggressive"], default="balanced", help="Packing aggressiveness. Default: balanced.")
    pack.set_defaults(func=_cmd_pack)

    measure = subparsers.add_parser("measure", help="Report compression savings and loss proxy metrics on eligible input.")
    measure.add_argument("paths", nargs="*", default=["."], help="Files or directories to measure.")
    measure.add_argument("--root", default=".", help="Root directory used for relative paths.")
    measure.add_argument("-b", "--budget", type=int, default=8000, help="Target token budget.")
    measure.add_argument("--query", default="", help="Task/query used for evidence scoring.")
    measure.add_argument("--mode", choices=["balanced", "aggressive"], default="balanced", help="Packing aggressiveness.")
    measure.add_argument("--model", default=None, help="Optional tokenizer model name if tiktoken is installed.")
    measure.add_argument("--max-file-bytes", type=int, default=200_000, help="Skip files larger than this.")
    measure.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    measure.set_defaults(func=_cmd_measure)

    shotpack = subparsers.add_parser("shotpack", help="Create a text pack plus dense PNG screenshot pages.")
    shotpack.add_argument("paths", nargs="*", default=["."], help="Files or directories to include.")
    shotpack.add_argument("--root", default=".", help="Root directory used for relative paths.")
    shotpack.add_argument("-b", "--budget", type=int, default=8000, help="Text pack token budget before rendering.")
    shotpack.add_argument("-o", "--output-dir", default="shotpack", help="Output directory for markdown, PNGs, and manifest.")
    shotpack.add_argument("--stem", default="context", help="Output filename stem.")
    shotpack.add_argument("--query", default="", help="Task/query used for evidence scoring.")
    shotpack.add_argument(
        "--mode",
        choices=["balanced", "aggressive"],
        default="aggressive",
        help="Packing aggressiveness. Default: aggressive for denser screenshot pages.",
    )
    shotpack.add_argument("--model", default=None, help="Optional tokenizer model name if tiktoken is installed.")
    shotpack.add_argument("--max-file-bytes", type=int, default=200_000, help="Skip files larger than this.")
    shotpack.add_argument("--image-width", type=int, default=1800, help="PNG page width.")
    shotpack.add_argument("--image-height", type=int, default=2400, help="PNG page height.")
    shotpack.add_argument("--font-size", type=int, default=20, help="Rendered monospace font size.")
    shotpack.add_argument("--columns", type=int, default=2, help="Rendered text columns per page.")
    shotpack.set_defaults(func=_cmd_shotpack)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.root)
    scan = scan_text_files(args.paths, root=root, max_file_bytes=args.max_file_bytes)
    rows = []
    total_raw = 0
    total_compact = 0
    total_redactions = 0

    for text_file in scan.files:
        raw = estimate_tokens(text_file.text, model=args.model)
        compacted = compact_text(text_file.text, model=args.model)
        saved, pct = token_savings(raw.tokens, compacted.compact_tokens)
        rows.append(
            {
                "path": text_file.rel_path,
                "eligible_tokens": raw.tokens,
                "raw_tokens": raw.tokens,
                "compact_tokens": compacted.compact_tokens,
                "saved_tokens": saved,
                "saved_percent": pct,
                "bytes": text_file.bytes_read,
                "backend": raw.backend,
                "redactions": text_file.redactions,
            }
        )
        total_raw += raw.tokens
        total_compact += compacted.compact_tokens
        total_redactions += text_file.redactions

    rows.sort(key=lambda item: item["raw_tokens"], reverse=True)
    saved, pct = token_savings(total_raw, total_compact)
    report = {
        "budget": args.budget,
        "price_per_million": args.price_per_million,
        "eligible_tokens": total_raw,
        "raw_tokens": total_raw,
        "compact_tokens": total_compact,
        "saved_tokens": saved,
        "saved_percent": pct,
        "estimated_raw_cost": total_raw / 1_000_000 * args.price_per_million,
        "estimated_compact_cost": total_compact / 1_000_000 * args.price_per_million,
        "redactions": total_redactions,
        "skip_counts": _skip_counts(scan.skipped),
        "skipped_paths": [item.__dict__ for item in scan.skipped],
        "files": rows,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Scanned {len(rows)} text files")
    print(f"Eligible source: {total_raw:,} tokens | Compact: {total_compact:,} tokens | Saved: {saved:,} ({pct:.1f}%)")
    print(
        "Estimated input cost: "
        f"${report['estimated_raw_cost']:.4f} -> ${report['estimated_compact_cost']:.4f} "
        f"at ${args.price_per_million:g}/1M tokens"
    )
    if scan.skipped:
        print(f"Skipped before scoring: {len(scan.skipped)} path(s) [{_format_skip_counts(scan.skipped)}]")
    if total_redactions:
        print(f"Privacy filter redacted {total_redactions} secret/PII match(es).")
    print()
    print(_table(rows[:30]))
    if len(rows) > 30:
        print(f"\nShowing top 30 of {len(rows)} files by eligible token count.")
    if args.show_skipped and scan.skipped:
        print()
        print(_skip_table(list(scan.skipped)[:30]))
        if len(scan.skipped) > 30:
            print(f"\nShowing top 30 of {len(scan.skipped)} skipped paths.")
    return 0


def _cmd_trim(args: argparse.Namespace) -> int:
    if args.input == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if is_sensitive_path(input_path) and not args.allow_sensitive_path:
            safe_path = redact_path_text(str(input_path)).text
            print(
                f"Refusing to trim sensitive path by default: {safe_path}. "
                "Use --allow-sensitive-path only when you intentionally want to process it.",
                file=sys.stderr,
            )
            return 2
        text = input_path.read_text(encoding="utf-8")
    redactions = 0
    if not args.no_redact:
        redaction = redact_sensitive_text(text)
        text = redaction.text
        redactions = redaction.replacements
    if args.strategy == "extractive" and args.budget is None:
        print("Extractive mode needs --budget; falling back to clean compaction.", file=sys.stderr)
    result = compact_text(
        text,
        head_lines=args.head_lines,
        tail_lines=args.tail_lines,
        model=args.model,
        strategy=args.strategy,
        budget_tokens=args.budget,
        query=args.query,
    )

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
    else:
        print(result.text, end="")

    report = {
        "eligible_tokens": result.original_tokens,
        "raw_tokens": result.original_tokens,
        "compact_tokens": result.compact_tokens,
        "saved_tokens": result.saved_tokens,
        "saved_percent": result.saved_percent,
        "notes": list(result.notes),
        "backend": result.backend,
        "redactions": redactions,
    }
    if args.json:
        print(json.dumps(report, indent=2), file=sys.stderr)
    else:
        print(
            f"\nSaved {result.saved_tokens:,} tokens ({result.saved_percent:.1f}%) "
            f"using {result.backend}.",
            file=sys.stderr,
        )
        if redactions:
            print(f"Privacy filter redacted {redactions} secret/PII match(es).", file=sys.stderr)
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    result = build_context_pack(
        list(args.paths),
        root=Path(args.root),
        budget_tokens=args.budget,
        model=args.model,
        max_file_bytes=args.max_file_bytes,
        query=args.query,
        mode=args.mode,
    )
    output = Path(args.output)
    output.write_text(result.markdown, encoding="utf-8")
    print(f"Wrote {output}")
    print(
        f"Eligible source: {result.source_tokens:,} tokens | Pack: {result.packed_tokens:,} tokens | "
        f"Saved: {result.saved_tokens:,} ({result.saved_percent:.1f}%)"
    )
    print(f"Files: {len(result.files)} | Counter: {result.backend}")
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    report = analyze_loss(
        list(args.paths),
        root=Path(args.root),
        budget_tokens=args.budget,
        query=args.query,
        mode=args.mode,
        model=args.model,
        max_file_bytes=args.max_file_bytes,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    print(f"Eligible source: {report.source_tokens:,} tokens")
    print(f"Pack: {report.packed_tokens:,} tokens")
    print(f"Saved: {report.saved_tokens:,} tokens ({report.saved_percent:.1f}%)")
    print(f"Token removal: {report.token_removal_percent:.1f}%")
    print(f"Token retention: {report.token_retention_percent:.1f}%")
    print(f"Query-term recall: {report.query_term_recall_percent:.1f}%")
    print(f"Code-symbol recall: {report.symbol_recall_percent:.1f}%")
    print(f"File coverage: {report.selected_files}/{report.source_files} ({report.file_coverage_percent:.1f}%)")
    print(f"Chunk coverage: {report.selected_chunks}/{report.source_chunks} ({report.chunk_coverage_percent:.1f}%)")
    print(f"Critical retention proxy: {report.critical_retention_percent:.1f}%")
    print(f"Estimated loss proxy: {report.estimated_loss_percent:.1f}%")
    return 0


def _cmd_shotpack(args: argparse.Namespace) -> int:
    try:
        result = build_shotpack(
            list(args.paths),
            root=Path(args.root),
            output_dir=Path(args.output_dir),
            budget_tokens=args.budget,
            query=args.query,
            mode=args.mode,
            model=args.model,
            max_file_bytes=args.max_file_bytes,
            stem=args.stem,
            width=args.image_width,
            height=args.image_height,
            font_size=args.font_size,
            columns=args.columns,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Wrote {result.markdown_path}")
    print(f"Wrote {result.manifest_path}")
    print(f"Rendered {len(result.pages)} PNG page(s) in {result.output_dir}")
    print(
        f"Eligible source stage: {result.source_tokens:,} -> {result.text_pack_tokens:,} tokens "
        f"({result.text_saved_percent:.1f}% saved before screenshot rendering)"
    )
    print("Use PNG pages for visual bulk context; keep the markdown for exact strings/code.")
    return 0


def _table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No text files found."
    headers = ("eligible", "compact", "saved", "file")
    rendered = [headers]
    for row in rows:
        rendered.append(
            (
                f"{int(row['raw_tokens']):,}",
                f"{int(row['compact_tokens']):,}",
                f"{float(row['saved_percent']):.1f}%",
                str(row["path"]),
            )
        )
    widths = [max(len(item[index]) for item in rendered) for index in range(len(headers))]
    lines = []
    for index, item in enumerate(rendered):
        line = "  ".join(value.rjust(widths[pos]) if pos < 3 else value for pos, value in enumerate(item))
        lines.append(line)
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _skip_counts(skipped: tuple[object, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped:
        reason = str(getattr(item, "reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _format_skip_counts(skipped: tuple[object, ...]) -> str:
    counts = _skip_counts(skipped)
    return ", ".join(f"{reason}={counts[reason]}" for reason in sorted(counts))


def _skip_table(rows: list[object]) -> str:
    headers = ("reason", "file")
    rendered = [headers]
    for row in rows:
        rendered.append((str(getattr(row, "reason")), str(getattr(row, "path"))))
    widths = [max(len(item[index]) for item in rendered) for index in range(len(headers))]
    lines = []
    for index, item in enumerate(rendered):
        line = "  ".join(value.ljust(widths[pos]) for pos, value in enumerate(item))
        lines.append(line)
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
