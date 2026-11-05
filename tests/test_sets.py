# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Which questions each set asks, and the frozen question lists."""
import json
import os
import re
import unittest

from acbench import sets
from tests.fixtures import Patch, data_env, read_json

FAKE_PRELIM = {"locomo": [{"conv": 0, "qa_index": 0, "category": 1}],
               "lme": [{"question_id": "fxq1", "type": "single-session-user"}]}


class FrozenLists(unittest.TestCase):
    def test_counts_match_what_the_readme_says(self):
        pre = read_json(sets.PRELIM_SET)
        self.assertEqual((len(pre["locomo"]), len(pre["lme"])), (39, 60))
        self.assertEqual(len({(p["conv"], p["qa_index"]) for p in pre["locomo"]}), 39)
        self.assertEqual(len({p["question_id"] for p in pre["lme"]}), 60)
        self.assertEqual(len(sets.answer_set_lme()), 12)
        self.assertTrue(set(sets.answer_set_lme()) <= {p["question_id"] for p in pre["lme"]})

    def test_question_files_hold_ids_only(self):
        ident = re.compile(r"^[A-Za-z0-9_:\-]{1,40}$")
        for name in ("PRELIM_SET.json", "SETS_V2.json", "ANSWER_SET_LME12.json"):
            doc = read_json(os.path.join(sets.QUESTIONS, name))

            def walk(node, key=""):
                if isinstance(node, dict):
                    for k, v in node.items():
                        walk(v, k)
                elif isinstance(node, list):
                    for v in node:
                        walk(v, key)
                elif isinstance(node, str) and key not in ("rule", "protocol", "type"):
                    self.assertRegex(node, ident, "%s holds text under %s" % (name, key))
            walk(doc)

    def test_v2_roles_are_disjoint_and_counted(self):
        doc = read_json(sets.SETS_V2)
        seen = {}
        for role, v in doc["sets"].items():
            for bench in ("locomo", "lme"):
                self.assertEqual(doc["counts"][role][bench], len(v.get(bench, [])))
                for q in v.get(bench, []):
                    if (bench, q) in seen:
                        self.assertTrue(role.startswith("nm_") and seen[(bench, q)].startswith("nm_"))
                    seen[(bench, q)] = role
        self.assertEqual(len(doc["eligible_pools"]["locomo_answering"]), 1428)
        self.assertEqual(sorted(doc["sets"]), sorted(sets.V2_ROLES))

    @unittest.skipUnless(os.environ.get("ACBENCH_REAL_DATA_CHECK"),
                         "set ACBENCH_REAL_DATA_CHECK=1 with the real datasets in ACBENCH_DATA to rebuild SETS_V2.json")
    def test_sets_v2_matches_a_rebuild_from_the_real_datasets(self):
        sets.reset_caches()
        with open(sets.SETS_V2, encoding="utf-8") as f:
            self.assertEqual(sets.sets_v2_text(sets.build_sets_v2()), f.read())


class Select(unittest.TestCase):
    def test_sets_on_the_invented_dataset(self):
        with data_env(), Patch((sets, "prelim", lambda: FAKE_PRELIM)):
            ids = lambda bench, which, limit=None: [r["id"] for r in sets.select(bench, which, limit)]
            self.assertEqual(ids("locomo", "prelim"), ["0:0"])
            self.assertEqual(ids("locomo", "full"), ["0:0", "0:1", "0:2", "0:3", "1:0", "1:1"])   # categories 1-4 only
            self.assertEqual(ids("locomo", "unseen"), ["0:1", "0:2", "0:3", "1:0", "1:1"])
            self.assertEqual(ids("locomo", "answer"), ids("locomo", "prelim"))
            self.assertEqual(ids("locomo", "full", 2), ["0:0", "0:1"])
            self.assertEqual(ids("lme", "prelim"), ["fxq1"])
            self.assertEqual(ids("lme", "full"), ["fxq1", "fxq2", "fxq3_abs"])
            with self.assertRaises(SystemExit):
                sets.select("lme", "unseen", None)
            row = sets.select("lme", "prelim", None)[0]
            self.assertEqual((row["unit"], row["question"]), ("fxq1", "Which road did I cycle along on Sunday?"))
            row = sets.select("locomo", "full", None)[0]
            self.assertEqual((row["unit"], row["question"]), ("0", "What did Mira make for her sister?"))

    def test_unseen_is_full_minus_prelim(self):
        with data_env(), Patch((sets, "prelim", lambda: FAKE_PRELIM)):
            full = {r["id"] for r in sets.select("locomo", "full", None)}
            seen = {r["id"] for r in sets.select("locomo", "prelim", None)}
            unseen = {r["id"] for r in sets.select("locomo", "unseen", None)}
            self.assertEqual(unseen | seen, full)
            self.assertFalse(unseen & seen)

    def test_fixed_v2_roles_read_their_ids(self):
        with data_env():
            ids = sets.v2_ids("short_memory", "lme")
            self.assertEqual(len(ids), 12)
            with self.assertRaises(KeyError):
                sets.select("lme", "validation", None)   # the invented dataset does not hold those questions

    def test_group_keeps_first_seen_order(self):
        rows = [{"id": "a", "unit": "2"}, {"id": "b", "unit": "1"}, {"id": "c", "unit": "2"}]
        self.assertEqual([(u, [r["id"] for r in rs]) for u, rs in sets.group(rows)], [("2", ["a", "c"]), ("1", ["b"])])

    def test_missing_dataset_says_how_to_get_it(self):
        with data_env() as tmp:
            os.remove(os.path.join(tmp, "data", "locomo10.json"))
            sets.reset_caches()
            with self.assertRaises(SystemExit) as caught:
                sets.select("locomo", "full", None)
            self.assertIn("download.sh", str(caught.exception))


class Segments(unittest.TestCase):
    def test_segment_names(self):
        self.assertEqual(sets.set_segment("prelim"), "prelim")
        self.assertEqual(sets.set_segment("full", 10), "full-limit10")
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(SystemExit):
                sets.set_segment("full", bad)
        with self.assertRaises(SystemExit):
            sets.set_segment("everything")

    def test_split_variant(self):
        self.assertEqual(sets.split_variant("full_spread1"), ("full", "spread1"))
        self.assertEqual(sets.split_variant("full"), ("full", None))


if __name__ == "__main__":
    unittest.main()
