from pathlib import Path
import unittest

from ai_token_saver.optimizer import compact_text
from ai_token_saver.packer import build_context_pack
from ai_token_saver.metrics import analyze_loss
from ai_token_saver.render import render_text_pages
from ai_token_saver.selection import make_code_skeleton
from ai_token_saver.files import TextFile
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


class TempDirectory:
    def __enter__(self) -> str:
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()
        return self._tempdir.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._tempdir.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()
