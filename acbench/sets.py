# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Which questions a run asks: the sets, in run order.

  prelim   the 99 questions (39 LoCoMo, 60 LongMemEval) that were looked at while Antechamber was tuned
  full     every answerable question (LoCoMo categories 1-4, every LongMemEval question)
  unseen   LoCoMo only: full minus prelim, the questions tuning never saw
  answer   the 39 prelim LoCoMo questions and 12 fixed LongMemEval ids
  short_memory ... nm_confirmation   fixed splits listed in questions/SETS_V2.json

Question ids only live in questions/*.json; the question and evidence text come from the downloaded datasets.
build_sets_v2() regenerates SETS_V2.json from the datasets; it refuses to overwrite a different file.
"""
import hashlib
import json
import os
import sys

from . import settings

QUESTIONS = os.path.join(settings.PACKAGE, "questions")
_LOCOMO = None
PRELIM_SET = os.path.join(QUESTIONS, "PRELIM_SET.json")
SETS_V2 = os.path.join(QUESTIONS, "SETS_V2.json")
_LME = None
ANSWER_SET = os.path.join(QUESTIONS, "ANSWER_SET_LME12.json")


V2_ROLES = ("short_memory", "validation", "confirmation", "reserve", "nm_short", "nm_validation", "nm_confirmation")

SETS = ("prelim", "full", "answer", "unseen") + V2_ROLES   # V2_ROLES come from SETS_V2.json


def reset_caches():
    """Forget the datasets read so far (tests point ACBENCH_DATA at a fixture)."""
    global _LME, _LOCOMO
    _LME = _LOCOMO = None


VARIANTS = {"spread1": {"ANTECHAMBER_SPREAD": "1"}, "spread2": {"ANTECHAMBER_SPREAD": "2"}}   # engine knobs; not reachable from the CLI sets


VARIANT_LIMIT = {}   # no variant widens the request


def split_variant(which):
    for v in VARIANTS:
        if which.endswith("_" + v):
            return which[:-len(v) - 1], v
    return which, None


def _dataset(name):
    path = os.path.join(settings.data_dir(), name)
    if not os.path.isfile(path):
        raise SystemExit("missing %s. Run data/download.sh first (ACBENCH_DATA is %s)." % (path, settings.data_dir()))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def locomo_data():
    global _LOCOMO
    if _LOCOMO is None:
        _LOCOMO = _dataset("locomo10.json")
    return _LOCOMO


def lme_index():
    global _LME
    if _LME is None:
        _LME = {x["question_id"]: x for x in _dataset("longmemeval_s_cleaned.json")}
    return _LME


def prelim():
    with open(PRELIM_SET, encoding="utf-8") as f:
        return json.load(f)


def answer_set_lme():
    """The 12 LongMemEval ids fixed by rule (2 non-abstention per type, PRELIM_SET order)."""
    with open(ANSWER_SET, encoding="utf-8") as f:
        return [i for ids in json.load(f)["lme"].values() for i in ids]


def _dia_map(item):
    out = {}
    conv = item["conversation"]
    n = 1
    while "session_%d" % n in conv:
        for t in conv["session_%d" % n]:
            out[str(t.get("dia_id"))] = t.get("text") or ""
        n += 1
    return out


def v2_ids(which, bench):
    with open(SETS_V2, encoding="utf-8") as f:
        return json.load(f)["sets"][which][bench]


def select(bench, which, limit):
    """Questions in run order. Each is {id, unit, question}. LME keeps its row."""
    rows = []
    if which in V2_ROLES:
        ids = v2_ids(which, bench)
        if bench == "locomo":
            data = locomo_data()
            for qid in ids:
                ci, i = (int(x) for x in qid.split(":"))
                rows.append({"id": qid, "unit": str(ci), "question": data[ci]["qa"][i]["question"]})
        else:
            idx = lme_index()
            rows = [{"id": q, "unit": q, "question": idx[q]["question"], "_row": idx[q]} for q in ids]
        return rows[:limit] if limit is not None else rows
    if bench == "locomo":
        data = locomo_data()
        seen = [(p["conv"], p["qa_index"]) for p in prelim()["locomo"]]
        if which in ("prelim", "answer"):   # the answer set's LoCoMo part IS the prelim 39
            picks = seen
        else:
            picks = [(ci, i) for ci, item in enumerate(data) for i, qa in enumerate(item["qa"])
                     if qa.get("category") in (1, 2, 3, 4)]
            if which == "unseen":
                picks = [p for p in picks if p not in set(seen)]
        for ci, i in picks:
            qa = data[ci]["qa"][i]
            rows.append({"id": "%d:%d" % (ci, i), "unit": str(ci), "question": qa["question"]})
    else:
        idx = lme_index()
        if which == "prelim":
            ids = [p["question_id"] for p in prelim()["lme"]]
        elif which == "answer":
            ids = answer_set_lme()
        elif which == "unseen":
            raise SystemExit("unseen is LoCoMo only; for LongMemEval use `score --split` on a full run")
        else:
            ids = list(idx)
        for qid in ids:
            q = idx[qid]
            rows.append({"id": qid, "unit": qid, "question": q["question"], "_row": q})
    if limit is not None:
        rows = rows[:limit]
    return rows


def group(rows):
    order, by = [], {}
    for row in rows:
        if row["unit"] not in by:
            order.append(row["unit"])
            by[row["unit"]] = []
        by[row["unit"]].append(row)
    return [(u, by[u]) for u in order]


def set_segment(which, limit=None):
    """Directory piece for a set, plus --limit when one was given. prelim and prelim-limit2 never share a file."""
    if which not in SETS:
        raise SystemExit("set must be one of %s" % ", ".join(SETS))
    if limit is None:
        return which
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise SystemExit("--limit must be a positive integer")
    return "%s-limit%d" % (which, limit)


# --- building SETS_V2.json -------------------------------------------------------------------------------------
# Hash = SHA-1 of the UTF-8 bytes of the canonical id, ascending. Canonical ids: LoCoMo "conv_index:qa_index",
# LongMemEval question_id. Run `python3 -m acbench.sets` to check the committed file against the datasets.

def _h(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_sets_v2():
    """The set manifest as a dict, built from the datasets and the fixed prelim / answer lists."""
    loc = locomo_data()
    lme = list(lme_index().values())
    pre = prelim()
    seen_loc = {"%d:%d" % (p["conv"], p["qa_index"]) for p in pre["locomo"]}
    seen_lme = [p["question_id"] for p in pre["lme"]]
    lme12 = answer_set_lme()

    # CHEAP short set for the memory loop: the 2 lowest-hash LoCoMo conversations, 10 lowest-hash questions per
    # category 1-4 in each, + 5 lowest-hash category-5 per conversation (no extra sessions);
    # LongMemEval: the 12 fixed ids (two LongMemEval questions cannot show a direction: 50-point steps).
    short_convs = sorted(range(len(loc)), key=lambda ci: _h(loc[ci]["sample_id"]))[:2]
    short_loc, short_nm = [], []
    for ci in short_convs:
        for cat in (1, 2, 3, 4):
            ids = sorted(("%d:%d" % (ci, i) for i, q in enumerate(loc[ci]["qa"]) if q.get("category") == cat), key=_h)
            short_loc += ids[:10]
        short_nm += sorted(("%d:%d" % (ci, i) for i, q in enumerate(loc[ci]["qa"]) if q.get("category") == 5),
                           key=_h)[:5]
    short_lme = list(lme12)
    taken = set(short_loc) | set(short_nm)
    # LoCoMo answering pools (never the 39 pilot questions, never the short set)
    cat14 = ["%d:%d" % (ci, i) for ci, c in enumerate(loc) for i, q in enumerate(c["qa"])
             if q.get("category") in (1, 2, 3, 4)]
    pool_loc = sorted((x for x in cat14 if x not in seen_loc and x not in taken), key=_h)
    cat5 = sorted(("%d:%d" % (ci, i) for ci, c in enumerate(loc) for i, q in enumerate(c["qa"])
                   if q.get("category") == 5 and "%d:%d" % (ci, i) not in taken), key=_h)
    # LongMemEval answering pools
    types = sorted({q["question_type"] for q in lme})
    pool_lme = {t: sorted((q["question_id"] for q in lme if q["question_type"] == t and q["question_id"] not in seen_lme
                           and not q["question_id"].endswith("_abs")), key=_h) for t in types}
    abs_other = sorted((q["question_id"] for q in lme if q["question_id"].endswith("_abs")
                        and q["question_id"] not in seen_lme), key=_h)
    abs_seen = [i for i in seen_lme if i.endswith("_abs")]

    sets = {
        "short_memory": {"locomo": short_loc, "lme": short_lme, "conversations": short_convs},
        "validation": {"locomo": pool_loc[:600], "lme": [i for t in types for i in pool_lme[t][:5]]},
        "confirmation": {"locomo": pool_loc[600:1200], "lme": [i for t in types for i in pool_lme[t][5:10]]},
        "reserve": {"locomo": pool_loc[1200:], "lme": [i for t in types for i in pool_lme[t][10:]]},
        "nm_short": {"locomo": short_nm, "lme": []},
        "nm_validation": {"locomo": cat5[:20], "lme": abs_other[:14]},
        "nm_confirmation": {"locomo": cat5[20:40], "lme": abs_other[14:28]},
    }
    rule = ("SHA-1 of UTF-8 canonical id ascending. SHORT (cheap loop): the 2 lowest-hash LoCoMo conversations "
            "(by sample_id), 10 lowest per category 1-4 each + 5 lowest category-5 each; LongMemEval the lowest "
            "12 (ANSWER_SET_LME12.json). LoCoMo pool = categories 1-4 not in PRELIM 39 (1,501): validation "
            "= first 600, confirmation = next 600, reserve = rest. LongMemEval pool = not in PRELIM 60, not abstention, "
            "per type: validation = 5 lowest, confirmation = next 5, reserve = all the rest (a failed confirmation "
            "takes the next 5 per type from the reserve). Not-mentioned: LoCoMo category 5 "
            "10 / 20 / 20; LongMemEval abstention: the 2 in the 60 / 14 / 14 of the other 28.")
    pools = {"locomo_answering": pool_loc, "locomo_not_mentioned": cat5,
             "lme_answering_by_type": pool_lme, "lme_abstention_unseen": abs_other, "lme_abstention_seen": abs_seen}
    doc = {"rule": rule, "protocol": "PROTOCOL-v2 r4", "sets": sets, "eligible_pools": pools,
           "counts": {k: {b: len(v.get(b, [])) for b in ("locomo", "lme")} for k, v in sets.items()}}
    # disjointness: no question in two roles
    seen = {}
    for role, v in sets.items():
        for b in ("locomo", "lme"):
            for q in v.get(b, []):
                if (b, q) in seen and not (role.startswith("nm_") and seen[(b, q)].startswith("nm_")):
                    sys.exit("REFUSED: %s %s in both %s and %s" % (b, q, seen[(b, q)], role))
                seen[(b, q)] = role
    return doc


def sets_v2_text(doc):
    return json.dumps(doc, indent=1, sort_keys=True)


def main():
    """Check the committed SETS_V2.json against the datasets. Never overwrites: a set change is a new version."""
    doc = build_sets_v2()
    text = sets_v2_text(doc)
    with open(SETS_V2, encoding="utf-8") as f:
        have = f.read()
    if have != text:
        sys.exit("REFUSED: SETS_V2.json differs from what the datasets give; a set change is a new version")
    print(json.dumps(doc["counts"]), "sha256", hashlib.sha256(text.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
