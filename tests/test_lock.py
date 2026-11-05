# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The lock refuses a run when anything that can move a number has changed."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from acbench import lock
from tests.fixtures import Patch, data_env, read_json

TAGS = {"models": [{"name": "qwen3-embedding:4b", "digest": "aaa"},
                   {"name": "nomic-embed-text:latest", "digest": "bbb"},
                   {"name": "mxbai-embed-large:latest", "digest": "ccc"}]}


class Rig:
    """A package folder, an engine folder, a LOCK path and fake Ollama / psql, all in one temp folder."""

    def __init__(self, tmp):
        self.pkg = os.path.join(tmp, "pkg")
        self.engine_root = os.path.join(tmp, "engine-checkout")
        os.makedirs(self.pkg)
        os.makedirs(os.path.join(self.engine_root, "engine"))
        self.write(os.path.join(self.pkg, "a.py"), "print('a')\n")
        self.write(os.path.join(self.engine_root, "engine", "recall_service.py"), "# engine\n")
        self.lock_path = os.path.join(tmp, "LOCK.json")
        self.tags = dict(TAGS)
        self.pg = {"pgvector.locomo_0": "h1"}
        os.environ["ANTECHAMBER_DIR"] = self.engine_root
        self.patch = Patch((lock, "PACKAGE", self.pkg), (lock, "LOCK", self.lock_path),
                           (lock, "ollama_tags", lambda: self.tags), (lock, "ollama_version", lambda: "0.9.9"),
                           (lock, "_pgvector_hashes", lambda: dict(self.pg)),
                           (lock, "_cmd", lambda args: "psql (PostgreSQL) 16.1"))

    @staticmethod
    def write(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def freeze(self):
        rec = lock.compute("all")
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        return rec

    def __enter__(self):
        self.patch.__enter__()
        return self

    def __exit__(self, *exc):
        os.environ.pop("ANTECHAMBER_DIR", None)
        return self.patch.__exit__(*exc)


@contextlib.contextmanager
def rig():
    with data_env() as tmp:
        data = os.environ["ACBENCH_DATA"]
        for fmt in ("chats", "notes", "pgvector"):
            for bench in ("locomo", "lme"):
                os.makedirs(os.path.join(data, fmt, bench))
                Rig.write(os.path.join(data, fmt, bench, "x.txt"), "converted %s %s\n" % (fmt, bench))
        with Rig(tmp) as r:
            yield r


class Scopes(unittest.TestCase):
    def test_each_scope_fingerprints_only_what_it_uses(self):
        with rig():
            plain = lock.compute("plain")["fingerprints"]
            self.assertEqual(sorted(plain), ["code", "data.lme", "data.locomo", "model.qwen_digest",
                                             "version.ollama", "version.python"])
            engine = lock.compute("engine")
            self.assertIn("engine.dir", engine["fingerprints"])
            self.assertNotIn("version.psql", engine["fingerprints"])
            self.assertIn("main_commit", engine)
            pg = lock.compute("pgvector")["fingerprints"]
            self.assertIn("pgvector.locomo_0", pg)
            self.assertIn("version.psql", pg)
            allfp = lock.compute("all")["fingerprints"]
            self.assertTrue({"model.qwen_digest", "model.nomic_digest", "model.mxbai_digest"} <= set(allfp))
            self.assertEqual(lock.scope_for("plainwin"), "plain")
            self.assertEqual(lock.scope_for("notes"), "engine")
            self.assertEqual(lock.scope_for("pgvector"), "pgvector")
            with self.assertRaises(ValueError):
                lock.compute("everything")

    def test_python_is_fingerprinted_by_minor_version_and_nothing_names_this_machine(self):
        with rig() as r:
            rec = r.freeze()
            self.assertRegex(rec["fingerprints"]["version.python"], r"^\d+\.\d+$")
            text = json.dumps(rec)
            self.assertNotIn("/" + "Users/", text)
            self.assertNotIn(r.pkg, text)


class Refusal(unittest.TestCase):
    def test_no_lock_file_refuses(self):
        with rig():
            diffs = lock.check("plain")
            self.assertEqual(len(diffs), 1)
            self.assertIn("LOCK.json missing", diffs[0])

    def test_unchanged_inputs_pass_every_scope(self):
        with rig() as r:
            r.freeze()
            for scope in ("plain", "engine", "pgvector"):
                self.assertEqual(lock.check(scope), [])

    def test_a_planted_file_in_the_code_folder_is_refused(self):
        with rig() as r:
            r.freeze()
            r.write(os.path.join(r.pkg, "_plant"), "planted\n")
            diffs = lock.check("plain")
            self.assertTrue(any(d.startswith("code:") for d in diffs))
            os.remove(os.path.join(r.pkg, "_plant"))
            self.assertEqual(lock.check("plain"), [])

    def test_cache_and_pycache_do_not_count(self):
        with rig() as r:
            r.freeze()
            os.makedirs(os.path.join(r.pkg, "__pycache__"))
            r.write(os.path.join(r.pkg, "__pycache__", "a.cpython.pyc"), "x")
            r.write(os.path.join(r.pkg, ".DS_Store"), "x")
            r.write(os.path.join(r.pkg, "run.log"), "x")
            self.assertEqual(lock.check("plain"), [])

    def test_changed_dataset_is_refused(self):
        with rig() as r:
            r.freeze()
            path = os.path.join(os.environ["ACBENCH_DATA"], "locomo10.json")
            with open(path, "a", encoding="utf-8") as f:
                f.write(" ")
            self.assertTrue(any(d.startswith("data.locomo:") for d in lock.check("plain")))

    def test_engine_change_matters_only_to_engine_formats(self):
        with rig() as r:
            r.freeze()
            r.write(os.path.join(r.engine_root, "engine", "recall_service.py"), "# edited\n")
            self.assertEqual(lock.check("plain"), [])
            self.assertTrue(any(d.startswith("engine.dir:") for d in lock.check("engine")))
            self.assertTrue(any(d.startswith("engine.dir:") for d in lock.check("pgvector")))

    def test_engine_commit_mismatch_is_refused_when_both_are_known(self):
        with rig() as r:
            with Patch((lock, "engine_commit", lambda: "c" * 40)):
                r.freeze()
                self.assertEqual(lock.check("engine"), [])
            with Patch((lock, "engine_commit", lambda: "d" * 40)):
                self.assertTrue(any(d.startswith("main_commit:") for d in lock.check("engine")))
            with Patch((lock, "engine_commit", lambda: lock.UNKNOWN)):
                self.assertEqual(lock.check("engine"), [])   # not a git checkout: the folder fingerprint decides

    def test_embedder_digest_is_checked_for_the_embedder_in_use(self):
        with rig() as r:
            r.freeze()
            r.tags = {"models": [{"name": "qwen3-embedding:4b", "digest": "aaa"}]}   # nomic and mxbai not pulled here
            self.assertEqual(lock.check("plain", ("qwen",)), [])
            self.assertTrue(lock.check("plain", ("nomic",)))                 # needed and missing: refused
            r.tags = {"models": [{"name": "qwen3-embedding:4b", "digest": "zzz"}]}   # weights replaced
            self.assertTrue(any(d.startswith("model.qwen_digest:") for d in lock.check("plain")))

    def test_missing_input_is_refused_not_ignored(self):
        with rig() as r:
            r.freeze()
            r.tags = None   # Ollama not answering
            diffs = lock.check("plain")
            self.assertIn("an input is MISSING", diffs)

    def test_pgvector_tables_are_hashed(self):
        with rig() as r:
            r.freeze()
            r.pg = {"pgvector.locomo_0": "changed"}
            self.assertTrue(any(d.startswith("pgvector.locomo_0:") for d in lock.check("pgvector")))
            self.assertEqual(lock.check("engine"), [])

    def test_version_or_settings_change_is_refused(self):
        with rig() as r:
            r.freeze()
            with Patch((lock, "VERSION", "v9.9.9")):
                self.assertTrue(any(d.startswith("version:") for d in lock.check("plain")))
            with Patch((lock, "SETTINGS", dict(lock.SETTINGS, limit=9))):
                self.assertTrue(any(d.startswith("settings:") for d in lock.check("plain")))

    def test_require_exits_one_and_names_each_difference(self):
        with rig() as r:
            r.freeze()
            r.write(os.path.join(r.pkg, "b.py"), "new\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as caught:
                    lock.require("plain")
            self.assertEqual(caught.exception.code, 1)
            self.assertIn("LOCK REFUSED", err.getvalue())
            self.assertIn("code:", err.getvalue())
            os.remove(os.path.join(r.pkg, "b.py"))
            lock.require("plain")   # back to the locked state: no exit


class Freeze(unittest.TestCase):
    def test_freeze_writes_the_lock_and_refuses_to_overwrite_the_same_version(self):
        with rig() as r:
            lock.main(["freeze"])
            first = read_json(r.lock_path)
            self.assertEqual(first["version"], lock.VERSION)
            lock.main(["freeze"])   # identical inputs: allowed, same content
            r.write(os.path.join(r.pkg, "c.py"), "changed\n")
            with self.assertRaises(SystemExit) as caught:
                lock.main(["freeze"])
            self.assertIn("Bump VERSION", str(caught.exception))
            self.assertEqual(read_json(r.lock_path), first)

    def test_freeze_refuses_when_an_input_is_missing(self):
        with rig() as r:
            r.tags = {"models": []}
            with self.assertRaises(SystemExit) as caught:
                lock.main(["freeze"])
            self.assertIn("MISSING", str(caught.exception))
            self.assertFalse(os.path.exists(r.lock_path))

    def test_a_narrow_freeze_covers_only_its_scope(self):
        with rig() as r:
            r.tags = {"models": [{"name": "qwen3-embedding:4b", "digest": "aaa"}]}   # only qwen is pulled
            os.environ.pop("ANTECHAMBER_DIR")
            lock.main(["freeze", "plain"])
            self.assertEqual(lock.check("plain"), [])
            self.assertTrue(lock.check("engine"))          # engine inputs were never locked: refused
            with self.assertRaises(SystemExit):
                lock.main(["freeze", "nonsense"])

    def test_narrow_freeze_records_the_converted_data_its_runs_require(self):
        with rig() as r:
            r.tags = {"models": [{"name": "qwen3-embedding:4b", "digest": "aaa"}]}
            os.environ.pop("ANTECHAMBER_DIR")
            lock.main(["freeze", "plain"])
            fp = read_json(r.lock_path)["fingerprints"]
            self.assertIn("derived.chats.locomo", fp)
            self.assertIn("derived.chats.lme", fp)
            self.assertNotIn("derived.notes.locomo", fp)
            # a plain recall's own requirement is met by the narrow freeze
            lock.require("plain", ("qwen",), lock.derived_for("plain", "locomo"))
            Rig.write(os.path.join(os.environ["ACBENCH_DATA"], "chats", "locomo", "x.txt"), "changed\n")
            self.assertTrue(lock.check("plain", ("qwen",), lock.derived_for("plain", "locomo")))

    def test_freeze_can_be_limited_to_one_bench(self):
        with rig() as r:
            r.tags = {"models": [{"name": "qwen3-embedding:4b", "digest": "aaa"}]}
            os.environ.pop("ANTECHAMBER_DIR")
            lock.main(["freeze", "plain", "locomo"])
            fp = read_json(r.lock_path)["fingerprints"]
            self.assertIn("derived.chats.locomo", fp)
            self.assertNotIn("derived.chats.lme", fp)
            engine_fp = lock.compute("engine", derived=lock.scope_derived("engine", "lme"))["fingerprints"]
            self.assertIn("derived.notes.lme", engine_fp)

    def test_same_version_block(self):
        rec = {"version": "v9", "fingerprints": {"a": "1"}, "settings": {}, "main_commit": "def"}
        self.assertIsNone(lock.same_version_block(dict(rec), rec))
        self.assertEqual(lock.same_version_block(dict(rec, fingerprints={"a": "2"}), rec), "fingerprints")
        self.assertEqual(lock.same_version_block(dict(rec, main_commit="x"), rec), "main_commit")
        self.assertEqual(lock.same_version_block(dict(rec, settings={"k": 1}), rec), "settings")
        self.assertIsNone(lock.same_version_block(rec, dict(rec, version="v10")))
        self.assertIsNone(lock.same_version_block(None, rec))

    def test_check_command_exit_codes(self):
        with rig() as r:
            r.freeze()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as ok:
                    lock.main(["check", "plain"])
            self.assertEqual(ok.exception.code, 0)
            self.assertIn("LOCK OK", out.getvalue())
            r.write(os.path.join(r.pkg, "d.py"), "x\n")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as bad:
                    lock.main(["check", "plain"])
            self.assertEqual(bad.exception.code, 1)


class MakeLockOrder(unittest.TestCase):
    """`make lock` runs precheck, then convert-all, then freeze. A locked version whose inputs differ must be
    refused by precheck before conversion rewrites any converted folder or table."""

    def _snapshot(self):
        out = {}
        data = os.environ["ACBENCH_DATA"]
        for fmt in ("chats", "notes", "pgvector"):
            for bench in ("locomo", "lme"):
                folder = os.path.join(data, fmt, bench)
                out[(fmt, bench)] = {n: open(os.path.join(folder, n), encoding="utf-8").read()
                                     for n in sorted(os.listdir(folder))}
        return out

    def _make_lock(self, psql_calls):
        """The make lock order, with conversion's database and embedder faked. Returns what conversion did."""
        from acbench import convert
        with contextlib.redirect_stdout(io.StringIO()):
            lock.main(["precheck"])
            with Patch((convert, "_psql", lambda db, sql: psql_calls.append(sql) or ""),
                       (convert, "ensure_db", lambda: None),
                       (convert, "embed_texts", lambda texts, model, dims: [[0.5] * dims for _ in texts])):
                for bench in ("locomo", "lme"):
                    for fmt in ("chats", "notes", "pgvector"):
                        convert.convert(bench, fmt)

    def _assert_refused_before_conversion(self, r, mutate):
        r.freeze()
        mutate()
        before, psql_calls = self._snapshot(), []
        with self.assertRaises(SystemExit) as caught:
            self._make_lock(psql_calls)
        self.assertIn("nothing was converted", str(caught.exception))
        self.assertEqual(psql_calls, [])                  # no table was dropped or rewritten
        self.assertEqual(self._snapshot(), before)        # no converted folder was rewritten
        self.assertFalse(any(os.path.exists(os.path.join(os.environ["ACBENCH_DATA"], f, b, "0"))
                             for f in ("chats", "notes", "pgvector") for b in ("locomo", "lme")))

    def test_an_unchanged_locked_version_may_convert(self):
        with rig() as r:
            r.freeze()
            psql_calls = []
            self._make_lock(psql_calls)
            self.assertTrue(psql_calls)                   # control: the faked conversion does run when allowed

    def test_a_changed_input_is_refused_before_anything_is_converted(self):
        with rig() as r:
            def mutate():
                with open(os.path.join(os.environ["ACBENCH_DATA"], "locomo10.json"), "a", encoding="utf-8") as f:
                    f.write(" ")
            self._assert_refused_before_conversion(r, mutate)

    def test_a_changed_converted_folder_is_refused_before_anything_is_converted(self):
        with rig() as r:
            self._assert_refused_before_conversion(
                r, lambda: r.write(os.path.join(os.environ["ACBENCH_DATA"], "notes", "lme", "x.txt"), "edited\n"))

    def test_a_changed_pgvector_table_is_refused_before_anything_is_converted(self):
        with rig() as r:
            self._assert_refused_before_conversion(r, lambda: r.pg.update({"pgvector.locomo_0": "edited"}))


class Digests(unittest.TestCase):
    def test_model_digest_prefers_exact_then_latest_then_family(self):
        tags = {"models": [{"name": "m:7b", "digest": "fam"}, {"name": "m:latest", "digest": "lat"},
                           {"name": "m", "digest": "exact"}]}
        self.assertEqual(lock.model_digest("m", tags), "exact")
        tags["models"].pop()
        self.assertEqual(lock.model_digest("m", tags), "lat")
        tags["models"].pop(1)
        self.assertEqual(lock.model_digest("m", tags), "fam")
        self.assertEqual(lock.model_digest("other", tags), "MISSING")
        self.assertEqual(lock.model_digest("m", {}), "MISSING")


if __name__ == "__main__":
    unittest.main()


class CommittedLock(unittest.TestCase):
    def test_code_fingerprint_matches_when_the_version_is_locked(self):
        # Editing acbench/ changes the code fingerprint; the edit must come with a new VERSION and a refreeze.
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LOCK.json")
        if not os.path.exists(path):
            self.skipTest("no LOCK.json")
        recorded = read_json(path)
        if recorded.get("version") != lock.VERSION:
            return   # a new VERSION awaiting its refreeze
        self.assertEqual(recorded["fingerprints"]["code"], lock._dir_sha(lock.PACKAGE),
                         "acbench/ changed under a locked VERSION: bump VERSION in lock.py and refreeze")


class Derived(unittest.TestCase):
    def test_converted_data_is_fingerprinted_and_an_edit_is_refused(self):
        with rig() as r:
            r.freeze()
            lock.require("plain", ("qwen",), (("chats", "locomo"),))
            Rig.write(os.path.join(os.environ["ACBENCH_DATA"], "chats", "locomo", "x.txt"), "edited\n")
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    lock.require("plain", ("qwen",), (("chats", "locomo"),))
            lock.require("plain", ("qwen",), (("chats", "lme"),))   # other bench untouched

    def test_an_unconverted_folder_is_missing(self):
        with rig():
            import shutil
            shutil.rmtree(os.path.join(os.environ["ACBENCH_DATA"], "notes", "lme"))
            fp = lock.compute("engine", derived=(("notes", "lme"),))["fingerprints"]
            self.assertEqual(fp["derived.notes.lme"], "MISSING")
            self.assertEqual(lock.derived_for("plainwin", "lme"), (("chats", "lme"),))
            self.assertEqual(lock.derived_for("pgvector", "lme"), (("pgvector", "lme"),))
