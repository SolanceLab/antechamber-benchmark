# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Paired statistics: exact McNemar and the resampled intervals."""
import unittest

from acbench import stats


class McNemar(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(stats.mcnemar_exact(10, 2), 0.0386, places=3)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(stats.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 10), 2 / 1024)
        self.assertEqual(stats.mcnemar_exact(3, 9), stats.mcnemar_exact(9, 3))


class Intervals(unittest.TestCase):
    def test_same_seed_same_interval(self):
        diff = {"q%d" % i: (1 if i % 3 else 0) for i in range(60)}
        a = stats.paired_ci(diff, "lme", n=500)
        b = stats.paired_ci(diff, "lme", n=500)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])

    def test_interval_around_a_clear_gain_is_above_zero(self):
        diff = {"%d:%d" % (c, i): 1 for c in range(5) for i in range(10)}
        lo, hi = stats.paired_ci(diff, "locomo", n=300)
        self.assertEqual((lo, hi), (1.0, 1.0))
        none = {"%d:%d" % (c, i): 0 for c in range(5) for i in range(10)}
        self.assertEqual(stats.paired_ci(none, "locomo", n=300), (0.0, 0.0))

    def test_locomo_resamples_whole_conversations(self):
        # conversation 0 always gains, conversation 1 never does: a resample is all-0, all-1 or a mix by question count
        diff = {"0:%d" % i: 1 for i in range(4)}
        diff.update({"1:%d" % i: 0 for i in range(4)})
        lo, hi = stats.clustered_ci(diff, lambda q: q.split(":")[0], n=400)
        self.assertEqual((lo, hi), (0.0, 1.0))
        # question-level resampling of the same data would sit much closer to 0.5
        qlo, qhi = stats.paired_ci(diff, "lme", n=400)
        self.assertGreater(qlo, 0.0)
        self.assertLess(qhi, 1.0)


if __name__ == "__main__":
    unittest.main()
