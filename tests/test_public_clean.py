# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Generic publication checks that need no secret list. Patterns are assembled in pieces so this file does not match itself."""
import json
import os
import re
import subprocess
import unittest

from acbench import lock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "runs", "cache", "__pycache__", ".venv", "data"}

PATHS = [
    ("a home path", re.compile("/" + "Users/[^/\\s]+/")),
    ("a home path", re.compile("/" + "home/[^/\\s]+/")),
    ("a home path", re.compile("[A-Za-z]:" + "\\\\" + "Users" + "\\\\")),
]
SECRETS = [
    ("an API key", re.compile("\\bsk" + "-[A-Za-z0-9_-]{16,}")),
    ("a GitHub token", re.compile("\\bghp" + "_[A-Za-z0-9]{16,}")),
    ("a private key", re.compile("BEGIN [A-Z ]*PRIVATE" + " KEY")),
    ("a hard-coded key", re.compile("api_key" + "\\s*=\\s*[\"'][^\"'\\s]+[\"']", re.I)),
]


def files():
    for dirpath, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith((".pyc", ".sqlite", ".DS_Store")):
                continue
            path = os.path.join(dirpath, name)
            yield os.path.relpath(path, ROOT), path


class PublicTree(unittest.TestCase):
    def test_no_home_paths_or_secrets(self):
        found = []
        for rel, path in files():
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue
            for label, pat in PATHS + SECRETS:
                if pat.search(text):
                    found.append("%s holds %s" % (rel, label))
        self.assertEqual(found, [])

    def test_no_dataset_text_is_committed(self):
        # the datasets are git-ignored, and nothing tracked under data/ except the download script may exist
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as f:
            ignore = f.read().split()
        for entry in ("data/*", "!data/download.sh", "runs/", "cache/"):
            self.assertIn(entry, ignore)
        out = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0:
            tracked = [p for p in out.stdout.split("\n") if p and p != "data/download.sh"]
            self.assertEqual(tracked, [])

    def test_the_committed_lock_matches_the_code_version(self):
        # a stale LOCK.json must fail here: refreeze (make lock) after any version bump
        with open(os.path.join(ROOT, "LOCK.json"), encoding="utf-8") as f:
            recorded = json.load(f)["version"]
        self.assertEqual(recorded, lock.VERSION)


if __name__ == "__main__":
    unittest.main()
