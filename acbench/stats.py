# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Paired statistics for the recall comparison. Standard library only.

mcnemar_exact: two-sided exact McNemar p on the questions where the two runs disagree.
clustered_ci / paired_ci: 95% percentile interval of the paired difference. LoCoMo resamples whole conversations
(questions inside one conversation are not independent); LongMemEval resamples questions.
"""
import math


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p on discordant pairs b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def clustered_ci(diff_by_q, conv_of, seed=20261018, n=10000):
    """95% percentile interval of the paired difference, resampling conversations with replacement; in each resample
    the difference is pooled over ALL questions of the drawn conversations (question-weighted)."""
    import random
    convs = {}
    for q, d in diff_by_q.items():
        convs.setdefault(conv_of(q), []).append(d)
    keys = sorted(convs)
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        vals = [d for k in (rnd.choice(keys) for _ in keys) for d in convs[k]]
        out.append(sum(vals) / len(vals))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def paired_ci(diff_by_q, bench, seed=20261018, n=10000):
    if bench == "locomo":
        return clustered_ci(diff_by_q, lambda q: q.split(":")[0], seed, n)
    import random
    vals = list(diff_by_q.values())
    rnd = random.Random(seed)
    out = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]
