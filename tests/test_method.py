# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The ruler and the scoring: the parts of the method that decide a result. Offline, on an invented dataset."""
import json
import os
import unittest

from acbench import recall, sets
from acbench.render import render
from tests.fixtures import Patch, data_env


def _ok_row(qid="0:0", text="zzzz unrelated words"):
    return {"id": qid, "error": None, "raw": {"winners": [{"lines": [{"message": text}]}], "failed": []}}


def _miss_row(qid="0:0", reason="timeout: timed out"):
    return {"id": qid, "error": reason, "raw": None}


class ShingleRule(unittest.TestCase):
    def test_sixty_percent_of_four_word_phrases(self):
        blob = "Caroline went to the support group with her friends on Monday"
        self.assertTrue(recall.covered("went to the support group with her friends", blob))
        self.assertFalse(recall.covered("a completely different set of words about nothing", blob))
        # 8 words make 5 four-word phrases; 3 of 5 is exactly 60%
        turn = "one two three four five six seven eight"
        self.assertTrue(recall.covered(turn, "one two three four five six"))
        self.assertFalse(recall.covered(turn, "one two three four five"))
        self.assertFalse(recall.covered("", blob))

    def test_words_and_shingles(self):
        self.assertEqual(recall.words("It's a Test, ok?"), ["its", "a", "test", "ok"])
        self.assertEqual(recall.shingles(["a", "b", "c"]), ["a b c"])
        self.assertEqual(recall.shingles(["a", "b", "c", "d", "e"]), ["a b c d", "b c d e"])
        self.assertEqual(recall.shingles([]), [])

    def test_scores_every_winner_the_service_returned(self):
        winners = [{"lines": [{"message": "w%02d one two three four" % i}]} for i in range(16)]
        text = recall.human_text({"winners": winners, "limit": 8})
        self.assertEqual(len(text.split("\n\n")), 16)
        for i in range(16):
            self.assertIn("w%02d one two three four" % i, text)

    def test_only_the_text_the_reader_is_shown_is_scored(self):
        payload = {"winners": [{"title": "T", "summary": "S", "snippet": "N", "lines": [{"sender": "A", "message": "M"}]},
                               "junk"]}
        self.assertEqual(recall.human_text(payload), render(payload))
        self.assertEqual(recall.human_text(payload).split("\n"), ["A: M"])

    def test_evidence_only_in_a_hidden_title_does_not_count(self):
        ev = "the quick brown fox jumps over the lazy dog today"
        hidden = {"id": "0:0", "error": None, "raw": {"failed": [], "winners": [
            {"title": ev, "lines": [{"sender": "A", "message": "zzzz unrelated words"}]}]}}
        shown = {"id": "0:0", "error": None, "raw": {"failed": [], "winners": [
            {"title": "x", "lines": [{"sender": "A", "message": ev}]}]}}
        with data_env():
            with Patch((recall, "evidence_of", lambda bench, qid: ("score", [ev]))):
                self.assertEqual(recall.score_rows("locomo", [hidden])["found"], 0)
                self.assertEqual(recall.score_rows("locomo", [shown])["found"], 1)


class FailureRules(unittest.TestCase):
    def test_empty_shelves_are_not_a_miss(self):
        row = {"raw": {"served_by": "local", "winners": [{"lines": [{"message": "hi"}]}],
                       "failed": ["jrn", "jkw", "ppl_stale", "rul", "doc", "dkw", "vault", "mem"]}}
        self.assertFalse(recall.recall_failed(row))
        row["raw"]["failed"] = ["kw"]
        self.assertTrue(recall.recall_failed(row))
        row["raw"]["failed"] = ["msg_stale"]
        self.assertTrue(recall.recall_failed(row))
        public = {"raw": {"winners": [{"lines": [{"message": "hi"}]}], "failed": []}}
        self.assertFalse(recall.recall_failed(public))
        public["raw"]["failed"] = ["chats-locomo-0"]
        self.assertTrue(recall.recall_failed(public))
        public["raw"]["failed"] = []
        public["raw"]["degraded"] = ["chats-locomo-0"]
        self.assertTrue(recall.recall_failed(public))

    def test_infra_voids_the_stage(self):
        samples = (
            {"raw": {"served_by": "cloud", "reason": "local_stale_or_unsynced", "winners": []}},
            {"raw": {"served_by": "local", "reason": "local_error:down", "winners": []}},
        )
        for row in samples:
            with self.assertRaises(recall.StageVoid):
                recall.recall_failed(row)
        # a per-question HTTP error, timeout, or all-shelves-failed is a miss, not a void
        misses = (
            {"raw": {"winners": [], "error": "AllShelvesFailed: msg,kw"}, "error": None},
            {"error": "http 500: AllShelvesFailed: msg", "raw": None},
            {"error": "timeout: timed out", "raw": None},
        )
        for row in misses:
            self.assertTrue(recall.question_miss_reason(row))
            self.assertTrue(recall.recall_failed(row))

    def test_question_errors_stay_in_the_denominator_until_five_percent(self):
        one = [_miss_row()] + [_ok_row() for _ in range(19)]
        recall.guard_question_misses(one)   # 1/20 is 5%, not over
        self.assertFalse(recall.errors_exceed_rate(1, 20))
        self.assertTrue(recall.errors_exceed_rate(1, 19))
        self.assertTrue(recall.errors_exceed_rate(2, 20))
        with self.assertRaises(recall.StageVoid) as over:
            recall.guard_question_misses([_miss_row()] + [_ok_row() for _ in range(18)])
        self.assertIn("exceed 5%", str(over.exception))
        with data_env():
            scored = recall.score_rows("locomo", one)
        self.assertEqual(scored["n"], 20)
        self.assertEqual(scored["recall_errors"], 1)
        self.assertEqual(scored["failures"], 1)
        self.assertFalse(scored["questions"][0]["found"])
        self.assertEqual(scored["questions"][0]["reason"], "timeout: timed out")
        self.assertIn("recall_errors=1", recall.summary_line("chats", "locomo", scored))

    def test_ask_voids_when_misses_pass_five_percent(self):
        calls = {"n": 0}

        def fake_http(port, question, timeout=180):
            calls["n"] += 1
            if calls["n"] <= 2:
                return None, "http 500: AllShelvesFailed: msg"
            return {"winners": [{"lines": [{"message": "ok"}]}], "failed": []}, None

        batch = [{"id": "q%d" % i, "unit": "u", "question": "q"} for i in range(20)]
        old = recall.http_recall
        recall.http_recall = fake_http
        try:
            with self.assertRaises(recall.StageVoid):
                recall.ask(9, batch, 0, None, 20, [])
        finally:
            recall.http_recall = old
        self.assertEqual(calls["n"], 2)   # 2/20 exceeds 5%; the rest are not asked

    def test_replica_answer_voids_scoring(self):
        cloud = {"id": "0:0", "raw": {"served_by": "cloud", "reason": "local_stale_or_unsynced", "winners": []}}
        with data_env():
            with self.assertRaises(recall.StageVoid):
                recall.score_rows("locomo", [cloud])


class EvidenceRules(unittest.TestCase):
    def test_parse_evidence_ids(self):
        self.assertEqual(recall.parse_evidence_ids("D1:3; D2:8, D:11:26\tD3:1"), ["D1:3", "D2:8", "D11:26", "D3:1"])
        self.assertEqual(recall.parse_evidence_ids(["D1:3", "D:11:26"]), ["D1:3", "D11:26"])
        self.assertEqual(recall.parse_evidence_ids(None), [])

    def test_unmapped_evidence_is_excluded(self):
        dia = {"D1:3": "alpha beta gamma delta", "D11:26": "epsilon zeta eta theta"}
        kind, texts = recall.locomo_evidence({"category": 1, "evidence": "D1:3; D:11:26"}, dia)
        self.assertEqual((kind, texts), ("score", ["alpha beta gamma delta", "epsilon zeta eta theta"]))
        missing, _ = recall.locomo_evidence({"category": 1, "evidence": ["D1:3", "D:11:26", "NOPE"]}, dia)
        self.assertEqual(missing, "no_evidence")
        self.assertEqual(recall.locomo_evidence({"category": 5, "evidence": ["D1:3"]}, dia)[0], "no_evidence")
        self.assertEqual(recall.locomo_evidence({"category": 1}, dia)[0], "no_evidence")

    def test_evidence_of_each_bench(self):
        with data_env():
            kind, texts = recall.evidence_of("locomo", "0:0")
            self.assertEqual(kind, "score")
            self.assertTrue(texts[0].startswith("Yes, a blue bowl"))
            self.assertEqual(recall.evidence_of("locomo", "0:2")[0], "no_evidence")   # D9:9 exists nowhere
            self.assertEqual(recall.evidence_of("lme", "fxq3_abs")[0], "abstention")
            kind, texts = recall.evidence_of("lme", "fxq2")
            self.assertEqual(kind, "score")
            self.assertEqual(len(texts), 1)

    def test_score_counts_excluded_questions_and_chars(self):
        with data_env():
            rows = []
            for r in sets.select("locomo", "full", None):
                _kind, texts = recall.evidence_of("locomo", r["id"])
                rows.append({"id": r["id"], "error": None,
                             "raw": {"winners": [{"lines": [{"message": "\n".join(texts) or "none"}]}], "failed": []}})
            scored = recall.score_rows("locomo", rows)
            self.assertEqual((scored["n"], scored["found"], scored["no_evidence"]), (5, 5, 1))
            self.assertEqual(scored["no_evidence_ids"], ["0:2"])
            self.assertIn("no_evidence_ids=0:2", recall.summary_line("chats", "locomo", scored))
            lme = [{"id": r["id"], "error": None, "raw": {"winners": [{"lines": [{"message": "zzzz nothing"}]}],
                                                       "failed": []}} for r in sets.select("lme", "full", None)]
            s = recall.score_rows("lme", lme)
            self.assertEqual((s["n"], s["found"], s["abstentions"]), (2, 0, 1))
            self.assertEqual(s["recall"], 0.0)

    def test_chars_returned(self):
        long = "x" * 3000
        ante = recall.chars_returned({"raw": {"winners": [{"lines": [{"message": long}], "summary": "sum"}]}})
        self.assertEqual(ante, len(long) + 1 + len("sum"))
        self.assertEqual(recall.chars_returned({"raw": None}), 0)


class PlainSearch(unittest.TestCase):
    def test_cosine_order_ignores_position(self):
        msgs = [{"id": "m%d" % i, "sender": "human", "text": "message %d" % i} for i in range(10)]
        vecs = [[0.0, 1.0] for _ in msgs]
        vecs[9] = [1.0, 0.0]
        vecs[8] = [0.6, 0.8]
        winners = recall.rank_plain(msgs, vecs, [1.0, 0.0], limit=8)
        self.assertEqual(len(winners), 8)
        self.assertEqual(winners[0]["lines"][0]["id"], "m9")
        self.assertEqual(winners[1]["lines"][0]["id"], "m8")
        self.assertNotIn("date", winners[0])
        scores = [w["score"] for w in winners]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # ties keep export order: m1 before m2, both equally far from the query
        tied = [w["lines"][0]["id"] for w in winners if w["lines"][0]["id"] in ("m1", "m2")]
        self.assertEqual(tied, ["m1", "m2"])

    def test_vector_mismatch_voids(self):
        msgs = [{"id": "a", "sender": "h", "text": "a"}]
        with self.assertRaises(recall.StageVoid):
            recall.rank_plain(msgs, [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
        with self.assertRaises(recall.StageVoid):
            recall.rank_plain(msgs, [[1.0, 0.0, 0.0]], [1.0, 0.0])

    def test_plain_returns_whole_uncapped_messages(self):
        long = ("word " * 500) + ("W" * 2500)
        self.assertGreater(len(long), 2000)
        with data_env() as tmp:
            path = os.path.join(tmp, "chats", "locomo", "0", "conversations.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"uuid": "t", "chat_messages": [
                    {"uuid": "m-long", "sender": "human", "text": long},
                    {"uuid": "m-short", "sender": "assistant", "text": "short note"},
                    {"uuid": "m-empty", "sender": "assistant", "text": ""}]}], f)
            msgs = recall.load_chat_messages(path)
            self.assertEqual([m["text"] for m in msgs], [long, "short note"])
            winners = recall.rank_plain(msgs, [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], limit=8)
            self.assertEqual(len(winners), 2)
            self.assertEqual(winners[0]["lines"][0]["message"], long)
            rec = recall._record({"id": "0:0", "unit": "0", "question": "q"}, {"winners": winners, "failed": []},
                                 None, "sha")
            self.assertEqual(rec["chars"], len(long) + 1 + len("short note"))
            with self.assertRaises(recall.StageVoid):
                recall.load_chat_messages(os.path.join(tmp, "nowhere.json"))

    def test_too_long_is_unsearchable_not_cut(self):
        def fake_cached(texts, model, dims):
            if any(t == "LONG" for t in texts):
                raise recall.StageVoid("embed cache failed: HTTP Error 502")
            return [[1.0, 0.0] for _ in texts]
        old_c, old_r = recall.embed_cached, recall._embedder_refuses_as_too_long
        recall.embed_cached = fake_cached
        recall._embedder_refuses_as_too_long = lambda t, m, d: t == "LONG"
        try:
            vecs = recall.embed_plain(["a", "LONG", "b"], "m", 2)
        finally:
            recall.embed_cached, recall._embedder_refuses_as_too_long = old_c, old_r
        self.assertEqual(len(vecs), 3)
        self.assertIsNone(vecs[1])
        msgs = [{"id": x, "sender": "user", "text": x, "thread": "t"} for x in ("a", "LONG", "b")]
        ids = [w["lines"][0]["id"] for w in recall.rank_plain(msgs, vecs, [1.0, 0.0], 8)]
        self.assertNotIn("LONG", ids)

    def test_other_embed_errors_still_void(self):
        def fake_cached(texts, model, dims):
            raise recall.StageVoid("embed cache failed: HTTP Error 502")
        old_c, old_r = recall.embed_cached, recall._embedder_refuses_as_too_long
        recall.embed_cached = fake_cached
        recall._embedder_refuses_as_too_long = lambda t, m, d: False
        try:
            with self.assertRaises(recall.StageVoid):
                recall.embed_plain(["a"], "m", 2)
        finally:
            recall.embed_cached, recall._embedder_refuses_as_too_long = old_c, old_r

    def test_window_brings_neighbours_same_thread_only(self):
        msgs = [{"id": str(i), "sender": "user", "text": "x%d" % i, "thread": "A" if i < 6 else "B"}
                for i in range(10)]
        vecs = [[0.0, 1.0] for _ in range(10)]
        vecs[5] = [1.0, 0.0]
        out = recall.rank_plain_window(msgs, vecs, [1.0, 0.0], limit=1)
        ids = [line["id"] for line in out[0]["lines"]]
        self.assertEqual(ids, ["3", "4", "5"])   # 2 before, none after: 6.. are another conversation
        self.assertTrue(out[0]["lines"][2]["hit"])


class RefusalReason(unittest.TestCase):
    def test_probe_sends_a_post_body_with_the_text(self):
        import urllib.request
        seen = {}

        def ok(req, timeout=0):
            seen["data"], seen["method"] = req.data, req.get_method()

            class R:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self): return b"{}"
            return R()
        old = urllib.request.urlopen
        urllib.request.urlopen = ok
        try:
            self.assertFalse(recall._embedder_refuses_as_too_long("hello", "m", 2))
        finally:
            urllib.request.urlopen = old
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(json.loads(seen["data"])["input"], ["hello"])

    def test_ollama_refusal_text_is_read_from_ollama_itself(self):
        import io
        import urllib.error
        import urllib.request

        def refuse(req, timeout=0):
            self.assertTrue(req.full_url.endswith("/api/embed"))
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                         io.BytesIO(b'{"error":"the input length exceeds the context length"}'))
        old = urllib.request.urlopen
        urllib.request.urlopen = refuse
        try:
            self.assertTrue(recall._embedder_refuses_as_too_long("t", "m", 2))
        finally:
            urllib.request.urlopen = old

        def other(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 500, "bad", {}, io.BytesIO(b'{"error":"boom"}'))
        urllib.request.urlopen = other
        try:
            self.assertFalse(recall._embedder_refuses_as_too_long("t", "m", 2))
        finally:
            urllib.request.urlopen = old


if __name__ == "__main__":
    unittest.main()
