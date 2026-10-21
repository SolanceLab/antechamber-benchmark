# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""python3 -m acbench <recall|score|verdict|convert> ...

  recall   ask every question of a set through one format and save what came back
  score    score a saved recall file (--split: tuning-seen questions vs never-seen, chats minus plain)
  verdict  the paired statistics between formats and embedders, from saved recall files
  convert  turn the downloaded datasets into chat exports, note folders or pgvector tables
"""
import argparse
import sys

from . import convert, recall
from .convert import EMBEDDERS
from .sets import SETS

FORMATS = ("plain", "plainwin", "chats", "notes", "pgvector")


def embedder_alias(value):
    """--embedder takes the short name (qwen, nomic, mxbai) or the model name (qwen3-embedding:4b)."""
    if value in EMBEDDERS:
        return value
    for alias, spec in EMBEDDERS.items():
        if spec["model"] == value:
            return alias
    raise argparse.ArgumentTypeError("embedder must be one of %s" % ", ".join(
        sorted(EMBEDDERS) + sorted(s["model"] for s in EMBEDDERS.values())))


def build_parser():
    p = argparse.ArgumentParser(prog="python3 -m acbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("recall", "score"):
        sp = sub.add_parser(name)
        sp.add_argument("--bench", required=True, choices=("locomo", "lme"))
        sp.add_argument("--format", required=True, choices=FORMATS)
        sp.add_argument("--set", default="full", choices=SETS, help="default full")
        sp.add_argument("--limit", type=int, default=None, help="first N questions of the set")
        sp.add_argument("--run-dir", default=None, help="write this stage here instead of ACBENCH_RUNS/<version>")
        sp.add_argument("--embedder", default=None, type=embedder_alias,
                        help="plain, plainwin and chats only; default qwen (qwen3-embedding:4b). "
                             "The pair always uses the same model.")
        if name == "score":
            sp.add_argument("--split", action="store_true",
                            help="prelim (tuning-seen) questions vs the rest, from the same full result files, "
                                 "plus the chats-minus-plain lead")
    sp = sub.add_parser("verdict")
    sp.add_argument("--bench", required=True, choices=("locomo", "lme"))
    sp.add_argument("--set", default="full", choices=SETS, help="default full")
    sp.add_argument("--run-dir", default=None)
    sp = sub.add_parser("convert")
    convert.add_arguments(sp)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "run_dir", None):
        recall.set_run_dir(a.run_dir)
    try:
        if a.cmd == "convert":
            convert.run_cli(a)
        elif a.cmd == "verdict":
            recall.run_recall_verdict(a.bench, a.set)
        else:
            embedder = recall.require_embedder_flag(a.format, a.embedder)
            if a.cmd == "recall":
                recall.run_recall(a.format, a.bench, a.set, a.limit, embedder)
            else:
                recall.run_score(a.format, a.bench, a.set, a.limit, embedder, split=a.split)
    except recall.StageVoid as e:
        print("STAGE VOID: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
