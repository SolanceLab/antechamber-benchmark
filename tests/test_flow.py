# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""A whole recall stage end to end with the embedder, the cache, the engine and the lock replaced by fakes.

The real thing needs Ollama and the engine; these tests prove the plumbing around them: which questions are asked,
what is written where, what the lock sees, and what is refused.
"""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest

from acbench import convert, lock, recall, sets
from tests.fixtures import Patch, data_env, read_json, read_text


def fake_vector(text, dims=16):
    """A repeatable stand-in embedding: a hashed bag of words. Texts that share words are close."""
    vec = [0.0] * dims
    for w in recall.words(text):
        vec[int(hashlib.sha1(w.encode()).hexdigest(), 16) % dims] += 1.0
    return vec


def fake_cached(texts, model, dims):
    return [fake_vector(t) for t in texts]


class FakeLog:
    closed = False

    def close(self):
        self.closed = True


class FakeProc:
    pid = 4242

    def poll(self):
        return None


def _write_chats(bench):
    convert.convert(bench, "chats")


class PlainFlow(unittest.TestCase):
    def test_plain_and_plainwin_write_a_recall_file_and_a_manifest(self):
        with data_env() as tmp:
            _write_chats("locomo")
            _write_chats("lme")
            seen_scopes = []
            with Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): seen_scopes.append((scope, embedders, derived))),
                       (recall, "ensure_cache", lambda: (None, None)),
                       (recall, "embed_cached", fake_cached)):
                for bench in ("locomo", "lme"):
                    doc = recall.run_recall("plain", bench, "full", None)
                    self.assertEqual(doc["format"], "plain")
                    self.assertEqual(doc["embedder"], "qwen3-embedding:4b")
                    self.assertEqual(doc["dimensions"], 2000)
                    self.assertEqual(doc["limit"], 8)
                    self.assertEqual(doc["version"], lock.VERSION)
                    self.assertEqual(len(doc["questions"]), len(sets.select(bench, "full", None)))
                    for q in doc["questions"]:
                        self.assertEqual(q["raw"]["failed"], [])
                        self.assertNotIn("served_by", q["raw"])
                        self.assertLessEqual(len(q["raw"]["winners"]), 8)
                        for w in q["raw"]["winners"]:
                            self.assertEqual(set(w), {"lines", "score"})
                            self.assertEqual(set(w["lines"][0]), {"id", "sender", "message"})
                    path = os.path.join(recall.RUN, "recall", "plain", "qwen", "full", bench + ".json")
                    self.assertTrue(os.path.isfile(path))
                    win = recall.run_recall("plainwin", bench, "full", 2)
                    self.assertEqual(win["format"], "plainwin")
                    self.assertEqual(len(win["questions"]), 2)
                    self.assertTrue(os.path.isfile(os.path.join(
                        recall.RUN, "recall", "plainwin", "qwen", "full-limit2", bench + ".json")))
            manifest = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            self.assertIn("recall/plain/qwen/full/locomo.json", manifest)
            self.assertIn("recall/plainwin/qwen/full-limit2/lme.json", manifest)
            self.assertTrue(seen_scopes)
            self.assertTrue(all(s[:2] == ("plain", ("qwen",)) for s in seen_scopes))   # plain needs no engine
            self.assertFalse(os.path.exists(os.path.join(tmp, "runs")))            # outputs went to the run dir only

    def test_embedder_alias_changes_model_and_dims(self):
        with data_env():
            _write_chats("locomo")
            seen = []

            def spy(texts, model, dims):
                seen.append((model, dims))
                return [fake_vector(t) for t in texts]
            with Patch((lock, "require", lambda *a, **k: None), (recall, "ensure_cache", lambda: (None, None)),
                       (recall, "embed_cached", spy)):
                doc = recall.run_recall("plain", "locomo", "full", 1, "nomic")
            self.assertEqual((doc["embedder"], doc["dimensions"], doc["embedder_flag"]), ("nomic-embed-text", 768, "nomic"))
            self.assertIn(("nomic-embed-text", 768), seen)
            self.assertTrue(os.path.isfile(os.path.join(
                recall.RUN, "recall", "plain", "nomic", "full-limit1", "locomo.json")))

    def test_a_failed_lock_stops_the_stage_before_any_question(self):
        with data_env():
            _write_chats("locomo")
            asked = []

            def refuse(*a, **k):
                raise SystemExit(1)
            with Patch((lock, "require", refuse), (recall, "ensure_cache", lambda: asked.append("cache") or (None, None)),
                       (recall, "embed_cached", fake_cached)):
                with self.assertRaises(SystemExit):
                    recall.run_recall("plain", "locomo", "full", None)
            self.assertEqual(asked, [])

    def test_lock_is_rechecked_every_25_questions(self):
        calls = []
        with Patch((lock, "require", lambda *a, **k: calls.append(a))):
            for asked in range(0, 51):
                recall._tick(asked)
        self.assertEqual(len(calls), 2)   # at 25 and 50

    def test_cache_failure_voids_the_stage(self):
        with data_env():
            _write_chats("locomo")

            def broken(texts, model, dims):
                raise recall.StageVoid("embed cache failed: down")
            with Patch((lock, "require", lambda *a, **k: None), (recall, "ensure_cache", lambda: (None, None)),
                       (recall, "embed_cached", broken),
                       (recall, "_embedder_refuses_as_too_long", lambda *a: False)):
                with self.assertRaises(recall.StageVoid):
                    recall.run_recall("plain", "locomo", "full", None)


class EngineFlow(unittest.TestCase):
    def _engine(self, tmp):
        folder = os.path.join(tmp, "antechamber")
        os.makedirs(os.path.join(folder, "engine"))
        with open(os.path.join(folder, "engine", "recall_service.py"), "w", encoding="utf-8") as f:
            f.write("# stand-in\n")
        return folder

    def test_chats_run_starts_the_engine_per_unit_and_scores_what_it_returns(self):
        with data_env() as tmp:
            os.environ["ANTECHAMBER_DIR"] = self._engine(tmp)
            try:
                _write_chats("locomo")
                started, answered = [], []

                def fake_start(script, env, log_path):
                    started.append((script, dict(env), log_path))
                    return FakeProc(), FakeLog()

                def fake_http(port, question, timeout=180):
                    answered.append((port, question))
                    # the engine "finds" the evidence line for the pottery question only
                    if "make for her sister" in question:
                        return {"winners": [{"lines": [{"message": "Mira: Yes, a blue bowl with a chipped rim that she kept anyway."}]}],
                                "failed": []}, None
                    return {"winners": [{"lines": [{"message": "unrelated words"}]}], "failed": []}, None

                scopes = []
                with Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): scopes.append(scope)),
                           (recall, "ensure_cache", lambda: (None, None)), (recall, "start_proc", fake_start),
                           (recall, "wait_health", lambda *a, **k: {"ok": True}), (recall, "stop_proc", lambda p: None),
                           (recall, "pick_port", lambda wanted: 8791), (recall, "http_recall", fake_http),
                           (recall, "guard_question_misses", lambda rows: None)):
                    doc = recall.run_recall("chats", "locomo", "full", None)
                self.assertEqual([s[0] for s in started], [os.path.join(os.environ["ANTECHAMBER_DIR"], "engine", "recall_service.py")] * 2)
                self.assertEqual(len(answered), 6)        # every question of the invented set
                self.assertTrue(all(p == 8791 for p, _ in answered))
                env = started[0][1]
                self.assertEqual(env["ANTECHAMBER_WIDE"], "on")
                self.assertEqual(env["ANTECHAMBER_DATES"], "on")
                self.assertEqual(env["ANTECHAMBER_CHRONO"], "on")
                self.assertTrue(os.path.isfile(env["ANTECHAMBER_CONFIG"]))
                self.assertEqual(set(env) - {"PATH", "HOME"}, {"ANTECHAMBER_CONFIG", "ANTECHAMBER_WIDE", "ANTECHAMBER_DATES",
                                                                 "ANTECHAMBER_CHRONO", "MEMORY_ENTRY_USE_DOOR"})
                self.assertEqual(doc["format"], "chats")
                self.assertIn("engine_commit", doc)
                self.assertEqual({"engine"}, set(scopes))
                scored = recall.score_rows("locomo", doc["questions"])
                self.assertEqual((scored["n"], scored["found"]), (5, 1))
                self.assertTrue(os.path.isfile(os.path.join(recall.RUN, "recall", "chats", "qwen", "full", "locomo.json")))
            finally:
                os.environ.pop("ANTECHAMBER_DIR", None)

    def test_two_benchmarks_share_one_run_dir_and_the_manifest_still_verifies(self):
        with data_env() as tmp:
            os.environ["ANTECHAMBER_DIR"] = self._engine(tmp)
            try:
                _write_chats("locomo")
                _write_chats("lme")

                def fake_start(script, env, log_path):
                    with open(env["ANTECHAMBER_CONFIG"], encoding="utf-8") as f:
                        text = f.read()
                    index_dir = json.loads(text.split("[index]", 1)[1].split("dir = ", 1)[1].splitlines()[0])
                    os.makedirs(index_dir, exist_ok=True)
                    with open(os.path.join(index_dir, "index.db"), "w", encoding="utf-8") as f:
                        f.write(index_dir)
                    return FakeProc(), FakeLog()

                def fake_http(port, question, timeout=180):
                    return {"winners": [{"lines": [{"message": "unrelated words"}]}], "failed": []}, None

                with Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): None),
                           (recall, "ensure_cache", lambda: (None, None)), (recall, "embed_cached", fake_cached),
                           (recall, "start_proc", fake_start),
                           (recall, "wait_health", lambda *a, **k: {"ok": True}), (recall, "stop_proc", lambda p: None),
                           (recall, "pick_port", lambda wanted: 8791), (recall, "http_recall", fake_http),
                           (recall, "guard_question_misses", lambda rows: None)):
                    recall.run_recall("plain", "locomo", "full", None)
                    recall.run_recall("chats", "locomo", "full", None)
                    recall.run_recall("plain", "lme", "full", None)
                    recall.run_recall("chats", "lme", "full", None)
                    # same benchmark, different limit, same run dir: the full run's indexes must survive
                    recall.run_recall("chats", "lme", "full", 1)
                    recall.run_recall("chats", "lme", "full", 2)
                manifest = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
                for bench in ("locomo", "lme"):
                    self.assertIn("recall/chats/qwen/full/%s.json" % bench, manifest)
                    self.assertIn("index/chats/qwen/%s/full/" % bench, manifest)
                    self.assertIsNone(recall._manifest_refusal(
                        os.path.join(recall.RUN, "recall", "chats", "qwen", "full", bench + ".json")))
                self.assertIn("index/chats/qwen/lme/full-limit1/", manifest)
                self.assertIn("index/chats/qwen/lme/full-limit2/", manifest)
            finally:
                os.environ.pop("ANTECHAMBER_DIR", None)

    def test_engine_formats_need_antechamber_dir(self):
        with data_env():
            os.environ.pop("ANTECHAMBER_DIR", None)
            _write_chats("locomo")
            with Patch((lock, "require", lambda *a, **k: None), (recall, "ensure_cache", lambda: (None, None))):
                with self.assertRaises(SystemExit) as caught:
                    recall.run_recall("chats", "locomo", "full", None)
            self.assertIn("ANTECHAMBER_DIR", str(caught.exception))

    def test_missing_export_says_how_to_make_it(self):
        with data_env():
            with self.assertRaises(SystemExit) as caught:
                recall.public_config("chats", "locomo", "0", "/tmp/x", 9999)
            self.assertIn("python3 -m acbench convert", str(caught.exception))

    def test_variants_are_chats_only(self):
        with self.assertRaises(SystemExit):
            recall.run_recall("notes", "locomo", "full_spread1", None)


class Verdict(unittest.TestCase):
    def _doc(self, fmt, bench, rows, embedder="qwen"):
        spec = recall.named_embedder(embedder)
        doc = {"version": lock.VERSION, "stage": "recall", "format": fmt, "bench": bench, "set": "full", "limit": 8,
               "embedder": spec["model"], "dimensions": spec["dimensions"], "questions": rows}
        flag = embedder if fmt in ("plain", "plainwin", "chats") else None
        recall._write(recall.out_path("recall", fmt, bench, flag, "full"), doc)
        recall.write_manifest()

    def _rows(self, bench, found_ids):
        rows = []
        for r in sets.select(bench, "full", None):
            kind, texts = recall.evidence_of(bench, r["id"])
            hit = r["id"] in found_ids and kind == "score"
            msg = "\n".join(texts) if hit else "zzzz unrelated padding words"
            rows.append({"id": r["id"], "unit": r["unit"], "question": r["question"], "error": None,
                         "raw": {"winners": [{"lines": [{"message": msg}]}], "failed": []}})
        return rows

    def test_verdict_pairs_chats_with_plain_and_formats_with_chats(self):
        with data_env():
            self._doc("chats", "locomo", self._rows("locomo", {"0:0", "0:1", "0:3", "1:0", "1:1"}))
            self._doc("plain", "locomo", self._rows("locomo", {"0:0"}))
            self._doc("notes", "locomo", self._rows("locomo", {"0:0", "0:1", "0:3", "1:0", "1:1"}))
            with contextlib.redirect_stdout(io.StringIO()), Patch((lock, "require", lambda *a, **k: None)):
                recall.run_recall_verdict("locomo", "full")
            out = read_json(os.path.join(recall.RUN, "score", "recall-verdict", "full", "locomo.json"))
            qwen = out["C1_C2"]["qwen"]
            self.assertEqual(qwen["n"], 5)
            self.assertEqual(qwen["gain"], 80.0)
            self.assertAlmostEqual(qwen["p"], 0.125)           # 4 vs 0 discordant pairs: exact p = 2 / 16
            self.assertFalse(qwen["counts"])                    # 5 questions cannot reach p < 0.05
            self.assertEqual(out["C1_C2"]["nomic"], "not run")
            self.assertEqual(out["C3"]["notes"]["diff"], 0.0)
            self.assertEqual(out["C3"]["pgvector"], "not run")
            manifest = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            self.assertIn("score/recall-verdict/full/locomo.json", manifest)   # the verdict itself is listed

    def test_partial_runs_never_pass(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("chats", "locomo", rows[:3])
            self._doc("plain", "locomo", rows)
            with contextlib.redirect_stdout(io.StringIO()), Patch((lock, "require", lambda *a, **k: None)):
                recall.run_recall_verdict("locomo", "full")
            out = read_json(os.path.join(recall.RUN, "score", "recall-verdict", "full", "locomo.json"))
            self.assertEqual(out["C1_C2"]["qwen"], "not run")   # the chats file does not cover every selected question

    def test_a_run_from_another_version_is_not_used(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("chats", "locomo", rows)
            self._doc("plain", "locomo", rows)
            path = recall.out_path("recall", "plain", "locomo", "qwen", "full")
            doc = read_json(path)
            doc["version"] = "v0.0.0"
            recall._write(path, doc)
            self.assertIsNone(recall.recall_found("plain", "locomo", "full", "qwen"))


class Split(unittest.TestCase):
    def test_prelim_vs_rest_and_the_lead(self):
        with data_env():
            real = {"locomo": ["0:0"], "lme": ["fxq1"]}
            fake_prelim = lambda: {"locomo": [{"conv": 0, "qa_index": 0, "category": 1}],
                                   "lme": [{"question_id": "fxq1", "type": "single-session-user"}]}
            with Patch((sets, "prelim", fake_prelim), (recall, "prelim", fake_prelim),
                       (lock, "require", lambda *a, **k: None)):
                for bench in ("locomo", "lme"):
                    def rows(found):
                        out = []
                        for r in sets.select(bench, "full", None):
                            kind, texts = recall.evidence_of(bench, r["id"])
                            msg = "\n".join(texts) if (r["id"] in found and kind == "score") else "zzzz padding words only"
                            out.append({"id": r["id"], "unit": r["unit"], "question": "q", "error": None,
                                        "raw": {"winners": [{"lines": [{"message": msg}]}], "failed": []}})
                        return out
                    everything = {r["id"] for r in sets.select(bench, "full", None)}
                    spec = recall.named_embedder("qwen")
                    for fmt, found in (("chats", everything), ("plain", set(real[bench]))):
                        recall._write(recall.out_path("recall", fmt, bench, "qwen", "full"),
                                      {"version": lock.VERSION, "stage": "recall", "format": fmt, "bench": bench,
                                       "set": "full", "embedder": spec["model"], "dimensions": spec["dimensions"],
                                       "questions": rows(found)})
                    recall.write_manifest()
                    doc = recall.run_split_score("qwen", bench=bench)
                    self.assertEqual(doc["prelim_ids"], 1)
                    self.assertEqual(doc["prelim"]["ids"], real[bench])
                    self.assertAlmostEqual(doc["prelim"]["lead"], 0.0)   # both found the one tuning question
                    self.assertGreater(doc["rest"]["lead"], 0.0)          # chats found the rest, plain did not
                    self.assertTrue(os.path.isfile(os.path.join(recall.RUN, "score", "split", "qwen", "full", bench + ".json")))
            manifest = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            self.assertIn("score/split/qwen/full/lme.json", manifest)

    def test_split_refuses_two_equally_incomplete_runs(self):
        with data_env():
            with Patch((lock, "require", lambda *a, **k: None)):
                spec = recall.named_embedder("qwen")
                every = [r["id"] for r in sets.select("lme", "full", None)]
                self.assertGreater(len(every), 1)
                partial = [{"id": i, "unit": i, "question": "q", "error": None,
                            "raw": {"winners": [{"lines": [{"message": "zzzz padding"}]}], "failed": []}}
                           for i in every[:-1]]
                for fmt in ("chats", "plain"):
                    recall._write(recall.out_path("recall", fmt, "lme", "qwen", "full"),
                                  {"version": lock.VERSION, "stage": "recall", "format": fmt, "bench": "lme",
                                   "set": "full", "embedder": spec["model"], "dimensions": spec["dimensions"],
                                   "questions": partial})
                recall.write_manifest()
                with self.assertRaises(SystemExit) as e:
                    recall.run_split_score("qwen", bench="lme")
                self.assertIn("every question", str(e.exception))

    def test_split_needs_full_and_matching_question_sets(self):
        with data_env():
            with Patch((lock, "require", lambda *a, **k: None)):
                with self.assertRaises(SystemExit) as a:
                    recall.run_split_score("qwen", which="prelim")
                self.assertIn("--set full", str(a.exception))
                with self.assertRaises(SystemExit) as b:
                    recall.run_split_score("qwen", which="full", limit=2)
                self.assertIn("no --limit", str(b.exception))
                with self.assertRaises(SystemExit) as c:
                    recall.run_score("notes", "lme", "full", None, "qwen", split=True)
                self.assertIn("plain and chats", str(c.exception))

    def test_score_command_writes_a_score_file(self):
        with data_env():
            spec = recall.named_embedder("qwen")
            rows = []
            for r in sets.select("lme", "full", None):
                rows.append({"id": r["id"], "unit": r["unit"], "question": "q", "error": None,
                             "raw": {"winners": [{"lines": [{"message": "zzzz nothing"}]}], "failed": []}})
            recall._write(recall.out_path("recall", "plain", "lme", "qwen", "full"),
                          {"version": lock.VERSION, "stage": "recall", "format": "plain", "bench": "lme", "set": "full",
                           "embedder": spec["model"], "dimensions": spec["dimensions"], "questions": rows})
            recall.write_manifest()
            with Patch((lock, "require", lambda *a, **k: None)):
                recall.run_score("plain", "lme", "full", None, "qwen")
            score = read_json(os.path.join(recall.RUN, "score", "plain", "qwen", "full", "lme.json"))
            self.assertEqual((score["stage"], score["format"], score["n"], score["found"]), ("score", "plain", 2, 0))


class Paths(unittest.TestCase):
    def test_paths_keep_set_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = recall.RUN
            recall.set_run_dir(tmp)
            try:
                self.assertEqual(recall.out_path("recall", "chats", "locomo", "qwen", "prelim", None),
                                 os.path.join(tmp, "recall", "chats", "qwen", "prelim", "locomo.json"))
                self.assertEqual(recall.out_path("recall", "chats", "locomo", "qwen", "prelim", 2),
                                 os.path.join(tmp, "recall", "chats", "qwen", "prelim-limit2", "locomo.json"))
                self.assertEqual(recall.out_path("recall", "notes", "lme", None, "full"),
                                 os.path.join(tmp, "recall", "notes", "full", "lme.json"))
                self.assertEqual(recall.out_path("recall", "plain", "lme", None, "full"),
                                 os.path.join(tmp, "recall", "plain", "qwen", "full", "lme.json"))
                with self.assertRaises(SystemExit):
                    recall.out_path("recall", "chats", "locomo", "qwen", "no-such-set")
                with self.assertRaises(SystemExit):
                    recall.out_path("recall", "chats", "locomo", "qwen", "prelim", 0)
            finally:
                recall.RUN = old


if __name__ == "__main__":
    unittest.main()


class VerdictIntegrity(Verdict):
    """The verdict is bound to the lock and to unedited, same-version, same-embedder recall files."""

    def test_verdict_checks_the_lock_with_the_converted_data(self):
        with data_env():
            self._doc("chats", "locomo", self._rows("locomo", {"0:0"}))
            self._doc("plain", "locomo", self._rows("locomo", {"0:0"}))
            calls = []
            with contextlib.redirect_stdout(io.StringIO()), \
                    Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): calls.append((scope, embedders, derived)))):
                recall.run_recall_verdict("locomo", "full")
            self.assertTrue(calls)
            # chats results are an engine result: the verdict takes the engine scope, not the plain one
            self.assertEqual(calls[0], ("engine", ("qwen",), (("chats", "locomo"),)))

    def test_a_refused_lock_stops_the_verdict(self):
        def refuse(*a, **k):
            raise SystemExit(1)
        with data_env():
            self._doc("chats", "locomo", self._rows("locomo", {"0:0"}))
            self._doc("plain", "locomo", self._rows("locomo", {"0:0"}))
            with Patch((lock, "require", refuse)):
                with self.assertRaises(SystemExit):
                    recall.run_recall_verdict("locomo", "full")
            self.assertFalse(os.path.exists(os.path.join(recall.RUN, "score", "recall-verdict")))

    def test_an_edited_recall_file_is_refused(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("chats", "locomo", rows)
            self._doc("plain", "locomo", rows)
            path = recall.out_path("recall", "plain", "locomo", "qwen", "full")
            doc = read_json(path)
            doc["questions"][0]["raw"]["winners"] = [{"lines": [{"message": "edited"}]}]
            recall._write(path, doc)   # edited after the manifest was written
            with Patch((lock, "require", lambda *a, **k: None)):
                with self.assertRaises(SystemExit) as caught:
                    recall.run_recall_verdict("locomo", "full")
                self.assertIn("changed since MANIFEST", str(caught.exception))
                with self.assertRaises(SystemExit):
                    recall._load_recall("plain", "locomo", "full", None, "qwen")

    def test_stale_stage_or_embedder_artifacts_are_refused(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("plain", "locomo", rows)
            path = recall.out_path("recall", "plain", "locomo", "qwen", "full")
            for field, value, word in (("stage", "score", "stage"), ("embedder", "other-model", "embedder"),
                                       ("dimensions", 768, "embedder"), ("version", "v0.0.0", "version")):
                doc = read_json(path)
                doc[field] = value
                recall._write(path, doc)
                recall.write_manifest()   # even a re-blessed manifest cannot make it valid
                with self.assertRaises(SystemExit) as caught:
                    recall._load_recall("plain", "locomo", "full", None, "qwen")
                self.assertIn(word, str(caught.exception))
                self._doc("plain", "locomo", rows)

    def test_a_file_missing_from_the_manifest_is_refused(self):
        with data_env():
            self._doc("plain", "locomo", self._rows("locomo", {"0:0"}))
            os.remove(os.path.join(recall.RUN, "MANIFEST.sha256"))
            with self.assertRaises(SystemExit) as caught:
                recall._load_recall("plain", "locomo", "full", None, "qwen")
            self.assertIn("MANIFEST", str(caught.exception))


class ManifestAndVerdictScope(Verdict):
    def _scopes(self, fmts):
        with data_env():
            for fmt in fmts:
                self._doc(fmt, "locomo", self._rows("locomo", {"0:0"}))
            calls = []
            with contextlib.redirect_stdout(io.StringIO()), \
                    Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): calls.append((scope, derived)))):
                recall.run_recall_verdict("locomo", "full")
            return calls[0]

    def test_verdict_lock_scope_covers_every_result_it_reads(self):
        self.assertEqual(self._scopes(["plain"]), ("plain", (("chats", "locomo"),)))
        self.assertEqual(self._scopes(["chats", "plain", "notes"]),
                         ("engine", (("chats", "locomo"), ("notes", "locomo"))))
        self.assertEqual(self._scopes(["chats", "plain", "notes", "pgvector"]),
                         ("pgvector", (("chats", "locomo"), ("notes", "locomo"), ("pgvector", "locomo"))))

    def test_a_tampered_listed_artifact_is_not_re_signed_by_a_later_stage(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("plain", "locomo", rows)
            path = recall.out_path("recall", "plain", "locomo", "qwen", "full")
            with open(path, "a", encoding="utf-8") as f:   # a raw edit, not through the harness writer
                f.write(" ")
            before = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            with self.assertRaises(SystemExit) as caught:
                recall.write_manifest()
            self.assertIn("changed since MANIFEST", str(caught.exception))
            self.assertEqual(read_text(os.path.join(recall.RUN, "MANIFEST.sha256")), before)
            os.remove(path)
            with self.assertRaises(SystemExit):
                recall.write_manifest()   # a listed artifact may not vanish either

    def test_new_artifacts_are_added_and_old_entries_kept(self):
        with data_env():
            rows = self._rows("locomo", {"0:0"})
            self._doc("plain", "locomo", rows)
            first = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            self._doc("chats", "locomo", rows)
            second = read_text(os.path.join(recall.RUN, "MANIFEST.sha256"))
            for line in first.splitlines():
                self.assertIn(line, second.splitlines())
            self.assertIn("recall/chats/qwen/full/locomo.json", second)

    def test_split_score_takes_the_engine_scope(self):
        calls = []
        with data_env():
            with Patch((lock, "require", lambda scope="plain", embedders=("qwen",), derived=(): calls.append(scope))):
                try:
                    recall.run_split_score("qwen", bench="locomo")
                except SystemExit:
                    pass   # no recall files here; only the lock scope it asked for matters
        self.assertEqual(calls[0], "engine")
