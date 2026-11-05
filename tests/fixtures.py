# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""A tiny invented dataset in the exact shape of the two real ones. No real dataset text is ever used in a test.

Every test that needs data points ACBENCH_DATA at a temporary folder holding these two files.
"""
import contextlib
import json
import os
import tempfile

from acbench import recall, sets

LOCOMO = [
    {
        "sample_id": "fx-0",
        "conversation": {
            "speaker_a": "Mira", "speaker_b": "Tobias",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Mira", "dia_id": "D1:1", "text": "I started pottery classes last spring and the kiln scared me at first."},
                {"speaker": "Tobias", "dia_id": "D1:2", "text": "That sounds lovely, did you make a bowl for your sister's birthday?"},
                {"speaker": "Mira", "dia_id": "D1:3", "text": "Yes, a blue bowl with a chipped rim that she kept anyway."},
            ],
            "session_2_date_time": "9:10 am on 14 June, 2023",
            "session_2": [
                {"speaker": "Tobias", "dia_id": "D2:1", "text": "We drove to the coast for a long weekend and the ferry was cancelled twice."},
                {"speaker": "Mira", "dia_id": "D2:2", "text": "Ok"},
            ],
        },
        "qa": [
            {"question": "What did Mira make for her sister?", "answer": "a bowl", "evidence": ["D1:3"], "category": 1},
            {"question": "When did Tobias go to the coast?", "answer": "June", "evidence": ["D2:1"], "category": 2},
            {"question": "Where does the ferry go?", "answer": "n/a", "evidence": ["D9:9"], "category": 3},
            {"question": "What scared Mira at first?", "answer": "the kiln", "evidence": ["D1:1"], "category": 4},
            {"question": "What is Tobias's job?", "answer": "Not mentioned", "category": 5},
        ],
    },
    {
        "sample_id": "fx-1",
        "conversation": {
            "speaker_a": "Ines", "speaker_b": "Jonas",
            "session_1_date_time": "6:30 pm on 2 February, 2023",
            "session_1": [
                {"speaker": "Ines", "dia_id": "D1:1", "text": "The allotment gave us forty courgettes this summer and nobody wants any more."},
                {"speaker": "Jonas", "dia_id": "D1:2", "text": "I will take a few and turn them into a soup for the neighbours."},
            ],
        },
        "qa": [
            {"question": "How many courgettes did the allotment give?", "answer": "forty", "evidence": ["D1:1"], "category": 1},
            {"question": "Who makes the soup?", "answer": "Jonas", "evidence": ["D1:2"], "category": 4},
        ],
    },
]

LME = [
    {
        "question_id": "fxq1", "question_type": "single-session-user",
        "question": "Which road did I cycle along on Sunday?",
        "answer": "the canal path",
        "haystack_session_ids": ["s-a", "s-b"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:05"],
        "haystack_sessions": [
            [{"role": "user", "content": "Planning my week, nothing much going on besides laundry and a dentist visit."},
             {"role": "assistant", "content": "A quiet week sounds restful. Anything you want help scheduling?"}],
            [{"role": "user", "content": "I cycled along the canal path on Sunday and saw two herons near the lock.",
              "has_answer": True},
             {"role": "assistant", "content": "Herons near locks are common in spring."}],
        ],
    },
    {
        "question_id": "fxq2", "question_type": "multi-session",
        "question": "How many plants did I buy in total?",
        "answer": "seven",
        "haystack_session_ids": ["s-c"],
        "haystack_dates": ["2023/06/01 (Thu) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought four ferns at the market today and one cactus.", "has_answer": True},
             {"role": "assistant", "content": "Nice haul. Ferns like humidity."}],
        ],
    },
    {
        "question_id": "fxq3_abs", "question_type": "single-session-user",
        "question": "What colour is my bicycle?",
        "answer": "not mentioned",
        "haystack_session_ids": ["s-d"],
        "haystack_dates": ["2023/07/04 (Tue) 18:45"],
        "haystack_sessions": [
            [{"role": "user", "content": "Remind me to renew the library card before the end of the month."},
             {"role": "assistant", "content": "Noted: renew the library card."}],
        ],
    },
]


def write_dataset(folder):
    with open(os.path.join(folder, "locomo10.json"), "w", encoding="utf-8") as f:
        json.dump(LOCOMO, f)
    with open(os.path.join(folder, "longmemeval_s_cleaned.json"), "w", encoding="utf-8") as f:
        json.dump(LME, f)


@contextlib.contextmanager
def data_env(extra_env=None):
    """A temp folder with the invented dataset, ACBENCH_DATA pointed at it, and recall's RUN sent to a sibling folder.
    Everything is put back on exit."""
    keys = ["ACBENCH_DATA", "ACBENCH_CACHE", "ACBENCH_RUNS"] + list(extra_env or {})
    saved = {k: os.environ.get(k) for k in keys}
    old_run = recall.RUN
    old_pid, old_token = recall.CACHE_PID, recall.CACHE_TOKEN
    with tempfile.TemporaryDirectory(prefix="acbench-test-") as tmp:
        data = os.path.join(tmp, "data")
        os.makedirs(data)
        write_dataset(data)
        os.environ["ACBENCH_DATA"] = data
        os.environ["ACBENCH_CACHE"] = os.path.join(tmp, "cache")
        os.environ["ACBENCH_RUNS"] = os.path.join(tmp, "runs")
        for k, v in (extra_env or {}).items():
            os.environ[k] = v
        sets.reset_caches()
        recall.set_run_dir(os.path.join(tmp, "run"))
        recall.CACHE_PID = os.path.join(tmp, "cache", "embed-cache.pid")
        recall.CACHE_TOKEN = os.path.join(tmp, "cache", "embed-cache.token")
        try:
            yield tmp
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            sets.reset_caches()
            recall.RUN = old_run
            recall.CACHE_PID, recall.CACHE_TOKEN = old_pid, old_token


class Patch:
    """Several attribute patches at once, undone together. `with Patch((obj, "name", value), ...):`"""

    def __init__(self, *triples):
        self.triples = triples
        self.old = []

    def __enter__(self):
        for obj, name, value in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.old):
            setattr(obj, name, value)
        return False


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()
