import io
from pathlib import Path
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ai_token_saver.cli import main
from ai_token_saver.optimizer import compact_text
from ai_token_saver.packer import build_context_pack
from ai_token_saver.metrics import analyze_loss
from ai_token_saver.privacy import is_sensitive_path, redact_sensitive_text
from ai_token_saver.render import render_text_pages
from ai_token_saver.selection import ContextChunk, make_code_skeleton, select_chunks
from ai_token_saver.files import TextFile, iter_text_files, scan_text_files
from ai_token_saver.tokenizer import estimate_tokens


class TokenSaverTests(unittest.TestCase):
    def test_estimator_counts_text(self):
        estimate = estimate_tokens("Fix this function and explain the regression.")
        self.assertGreater(estimate.tokens, 0)
        self.assertGreater(estimate.chars, 0)

    def test_compact_text_collapses_repeated_lines(self):
        text = "\n".join(["ERROR database timeout"] * 10)
        result = compact_text(text)
        self.assertIn("repeated", result.text)
        self.assertLess(result.compact_tokens, result.original_tokens)

    def test_context_pack_contains_file_index(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        sample = tmp_path / "sample.py"
        sample.write_text("print('hello')\n" * 40, encoding="utf-8")

        result = build_context_pack(["."], root=tmp_path, budget_tokens=500)

        self.assertIn("AI Token Saver Context Pack", result.markdown)
        self.assertIn("sample.py", result.markdown)
        self.assertGreater(result.source_tokens, 0)
        self.assertGreater(result.packed_tokens, 0)
        self.assertLessEqual(result.packed_tokens, 500)

    def test_evidence_pack_prefers_query_relevant_chunks(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / "noise.txt").write_text(
            ("heartbeat ok retrying idle worker shard=17\n" * 2500),
            encoding="utf-8",
        )
        (tmp_path / "design.md").write_text(
            "# Token Budget Design\n\n"
            "The new token compression strategy ranks evidence chunks by query overlap, "
            "structure value, and token density before packing.\n\n"
            "# Unrelated Notes\n\n"
            + ("calendar reminder\n" * 120),
            encoding="utf-8",
        )

        result = build_context_pack(
            ["."],
            root=tmp_path,
            budget_tokens=650,
            query="token compression evidence budget",
            mode="aggressive",
        )

        self.assertIn("Token Budget Design", result.markdown)
        self.assertIn("evidence chunks", result.markdown)
        self.assertLess(result.packed_tokens, result.source_tokens)
        self.assertGreater(result.saved_percent, 70)

    def test_code_skeleton_keeps_signatures_not_full_body(self):
        text = (
            "import os\n\n"
            "class PromptBudget:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "        self.cache = {}\n"
            "        self.total = 0\n"
            "    def add(self, value):\n"
            + "        self.items.append(value)\n" * 80
        )
        text_file = TextFile(
            path=Path("budget.py"),
            rel_path="budget.py",
            text=text,
            bytes_read=len(text.encode("utf-8")),
        )

        skeleton = make_code_skeleton(text_file)

        self.assertIsNotNone(skeleton)
        assert skeleton is not None
        self.assertIn("class PromptBudget", skeleton.text)
        self.assertIn("def add", skeleton.text)
        self.assertNotIn("self.items.append(value)\n        self.items.append(value)", skeleton.text)

    def test_loss_report_keeps_query_and_symbols(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / "engine.py").write_text(
            "class TokenEngine:\n"
            "    def evidence_budget(self):\n"
            "        return 'token compression budget evidence'\n",
            encoding="utf-8",
        )
        (tmp_path / "notes.txt").write_text("unrelated chatter\n" * 120, encoding="utf-8")

        report = analyze_loss(
            ["."],
            root=tmp_path,
            budget_tokens=500,
            query="token compression evidence budget",
            mode="aggressive",
        )

        self.assertEqual(report.query_term_recall_percent, 100.0)
        self.assertEqual(report.symbol_recall_percent, 100.0)
        self.assertLess(report.estimated_loss_percent, 20.0)

    def test_render_text_pages_when_pillow_available(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow is not installed")

        tmp_path = Path(self.enterContext(TempDirectory()))
        pages = render_text_pages(
            "# Context\n\n" + "token compression evidence budget\n" * 80,
            output_dir=tmp_path,
            stem="page",
            width=800,
            height=600,
            font_size=16,
            columns=2,
        )

        self.assertGreaterEqual(len(pages), 1)
        self.assertTrue((tmp_path / pages[0].path).exists())

    def test_privacy_redacts_tokens_and_contact_info(self):
        report = redact_sensitive_text(
            "api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n"
            "email=person@example.test\n"
            "phone=13800138000\n"
            "gh=gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n"
        )

        self.assertGreaterEqual(report.replacements, 4)
        self.assertNotIn("sk-proj-", report.text)
        self.assertNotIn("gmail.com", report.text)
        self.assertNotIn("13800138000", report.text)
        self.assertIn("<REDACTED:OPENAI_KEY>", report.text)

    def test_sensitive_files_are_skipped_by_default(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("contact person@example.test\n", encoding="utf-8")

        files = list(iter_text_files(["."], root=tmp_path))
        rel_paths = {item.rel_path for item in files}

        self.assertNotIn(".env", rel_paths)
        self.assertIn("notes.txt", rel_paths)
        note = next(item for item in files if item.rel_path == "notes.txt")
        self.assertIn("<REDACTED:EMAIL>", note.text)
        self.assertGreater(note.redactions, 0)

    def test_sensitive_path_detector_keeps_examples(self):
        self.assertTrue(is_sensitive_path(Path(".env")))
        self.assertTrue(is_sensitive_path(Path("credentials.json")))
        self.assertTrue(is_sensitive_path(Path("prod-secrets-sample.json")))
        self.assertFalse(is_sensitive_path(Path(".env.example")))
        self.assertFalse(is_sensitive_path(Path("README.md")))

    def test_query_matched_short_symbol_chunk_can_be_selected(self):
        chunk = ContextChunk(
            path="src/ai_token_saver/privacy.py",
            suffix=".py",
            start_line=10,
            end_line=12,
            text="def redact_sensitive_text(text):\n    return text\n",
            kind="code-symbol",
            tokens=15,
            score=0.0,
            fingerprint=frozenset({"redact_sensitive_text", "text"}),
        )

        selected = select_chunks([chunk], budget_tokens=40, query="privacy redaction")

        self.assertEqual([item.path for item in selected], ["src/ai_token_saver/privacy.py"])

    def test_privacy_redacts_structured_secret_fields(self):
        report = redact_sensitive_text(
            '{\n'
            '  "password": "hunter2",\n'
            '  "aws_secret_access_key": "abcd1234efgh5678ijkl9012mnop3456"\n'
            '}\n'
        )

        self.assertGreaterEqual(report.replacements, 2)
        self.assertNotIn("hunter2", report.text)
        self.assertNotIn("abcd1234efgh5678ijkl9012mnop3456", report.text)
        self.assertIn("<REDACTED:SECRET_VALUE>", report.text)

    def test_trim_refuses_sensitive_path_by_default(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        sensitive = tmp_path / ".netrc"
        sensitive.write_text("machine example.com login demo password hunter2\n", encoding="utf-8")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["trim", str(sensitive)])

        self.assertEqual(code, 2)
        self.assertIn("Refusing to trim sensitive path by default", stderr.getvalue())

    def test_rel_path_is_redacted_in_outputs(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        target = tmp_path / "alice@example.test.txt"
        target.write_text("hello\n", encoding="utf-8")

        files = list(iter_text_files([target.name], root=tmp_path))

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].rel_path, "<REDACTED:EMAIL>.txt")
        self.assertGreater(files[0].redactions, 0)

    def test_scan_text_files_reports_skip_reasons(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (tmp_path / "big.txt").write_text("x" * 32, encoding="utf-8")

        scan = scan_text_files(["."], root=tmp_path, max_file_bytes=16)
        reasons = {(item.path, item.reason) for item in scan.skipped}

        self.assertIn((".env", "sensitive path"), reasons)
        self.assertIn(("image.png", "unsupported file type"), reasons)
        self.assertIn(("big.txt", "over max file bytes"), reasons)

    def test_audit_show_skipped_prints_skip_table(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["audit", ".", "--root", str(tmp_path), "--show-skipped"])

        self.assertEqual(code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Skipped before scoring:", rendered)
        self.assertIn("sensitive path", rendered)
        self.assertIn(".env", rendered)

    def test_context_pack_summary_includes_skip_reason_counts(self):
        tmp_path = Path(self.enterContext(TempDirectory()))
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("hello token budget\n", encoding="utf-8")

        result = build_context_pack(["."], root=tmp_path, budget_tokens=300)

        self.assertIn("Skipped before scoring: 1 path(s) (sensitive path=1)", result.markdown)


class TempDirectory:
    def __enter__(self) -> str:
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()
        return self._tempdir.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._tempdir.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()
