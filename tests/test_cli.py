# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The command line: what each subcommand accepts and calls."""
import contextlib
import io
import unittest

from acbench import __main__ as cli
from acbench import recall
from tests.fixtures import Patch


def run(argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            cli.main(argv)
            return 0, err.getvalue()
        except SystemExit as e:
            return e.code, err.getvalue()


class Embedder(unittest.TestCase):
    def test_alias_or_model_name(self):
        self.assertEqual(cli.embedder_alias("qwen"), "qwen")
        self.assertEqual(cli.embedder_alias("qwen3-embedding:4b"), "qwen")
        self.assertEqual(cli.embedder_alias("nomic-embed-text"), "nomic")
        self.assertEqual(cli.embedder_alias("mxbai-embed-large"), "mxbai")
        with self.assertRaises(Exception):
            cli.embedder_alias("gpt-embed")


class Recall(unittest.TestCase):
    def test_recall_passes_its_arguments_through(self):
        calls = []
        with Patch((recall, "run_recall", lambda *a: calls.append(a))):
            self.assertEqual(run(["recall", "--bench", "locomo", "--format", "plain"])[0], 0)
            self.assertEqual(run(["recall", "--bench", "lme", "--format", "chats", "--embedder", "qwen3-embedding:4b",
                                  "--set", "prelim", "--limit", "3"])[0], 0)
            self.assertEqual(run(["recall", "--bench", "lme", "--format", "plainwin", "--embedder", "nomic"])[0], 0)
        self.assertEqual(calls[0], ("plain", "locomo", "full", None, "qwen"))
        self.assertEqual(calls[1], ("chats", "lme", "prelim", 3, "qwen"))
        self.assertEqual(calls[2], ("plainwin", "lme", "full", None, "nomic"))

    def test_embedder_flag_is_refused_where_it_does_not_apply(self):
        for fmt in ("notes", "pgvector"):
            code, err = run(["recall", "--bench", "locomo", "--format", fmt, "--embedder", "nomic"])
            self.assertEqual(code, "--embedder is for --format plain and chats")

    def test_bad_values_are_refused(self):
        for argv in (["recall", "--bench", "x", "--format", "plain"],
                     ["recall", "--bench", "lme", "--format", "house"],
                     ["recall", "--bench", "lme", "--format", "plain", "--set", "nope"],
                     ["recall", "--format", "plain"],
                     []):
            self.assertEqual(run(argv)[0], 2)

    def test_a_voided_stage_exits_one(self):
        def void(*a):
            raise recall.StageVoid("embed cache failed")
        with Patch((recall, "run_recall", void)):
            code, err = run(["recall", "--bench", "lme", "--format", "plain"])
        self.assertEqual(code, 1)
        self.assertIn("STAGE VOID: embed cache failed", err)

    def test_run_dir_redirects_the_outputs(self):
        old = recall.RUN
        try:
            with Patch((recall, "run_recall", lambda *a: None)):
                run(["recall", "--bench", "lme", "--format", "plain", "--run-dir", "/tmp/acbench-cli-run"])
            self.assertEqual(recall.RUN, "/tmp/acbench-cli-run")
        finally:
            recall.RUN = old


class Others(unittest.TestCase):
    def test_verdict_and_score_and_convert(self):
        seen = []
        with Patch((recall, "run_recall_verdict", lambda *a: seen.append(("verdict",) + a)),
                   (recall, "run_score", lambda *a, **k: seen.append(("score",) + a + (k,))),
                   (cli.convert, "run_cli", lambda a: seen.append(("convert", a.bench, a.fmt, a.limit, a.prelim)))):
            run(["verdict", "--bench", "lme"])
            run(["verdict", "--bench", "locomo", "--set", "prelim"])
            run(["score", "--bench", "lme", "--format", "plain", "--set", "full", "--split"])
            run(["convert", "--bench", "lme", "--format", "chats", "--limit", "2", "--prelim"])
        self.assertEqual(seen[0], ("verdict", "lme", "full"))
        self.assertEqual(seen[1], ("verdict", "locomo", "prelim"))
        self.assertEqual(seen[2], ("score", "plain", "lme", "full", None, "qwen", {"split": True}))
        self.assertEqual(seen[3], ("convert", "lme", "chats", 2, True))


if __name__ == "__main__":
    unittest.main()
