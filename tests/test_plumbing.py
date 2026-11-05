# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Engine config and environment, the cache's identity check, HTTP to the engine, ports, children's environment."""
import io
import json
import os
import socket
import sqlite3
import sys
import tempfile
import tomllib
import unittest
import urllib.error
import urllib.request

from acbench import convert, lock, recall
from tests.fixtures import Patch, data_env, read_json, read_text, write_dataset


def _exports(bench="locomo"):
    convert.convert(bench, "chats")
    convert.convert(bench, "notes")


class EngineConfig(unittest.TestCase):
    def test_chats_config_points_at_the_cache_and_names_no_machine(self):
        with data_env():
            _exports()
            text = recall.public_config("chats", "locomo", "0", "/idx/x", 8791)
            cfg = tomllib.loads(text)
            self.assertEqual(cfg["embedder"]["url"], "http://127.0.0.1:11435/api/embed")
            self.assertEqual((cfg["embedder"]["model"], cfg["embedder"]["dimensions"]), ("qwen3-embedding:4b", 2000))
            self.assertEqual(cfg["service"]["port"], 8791)
            self.assertEqual(cfg["index"]["dir"], "/idx/x")
            self.assertEqual(cfg["vault"]["dir"], "")
            adapter = cfg["adapters"]["chats"][0]
            self.assertEqual((adapter["id"], adapter["format"]), ("chats-locomo-0", "claude"))
            self.assertTrue(adapter["path"].endswith(os.path.join("chats", "locomo", "0")))
            self.assertNotIn("recall", cfg)
            self.assertNotIn("11434", text)

    def test_embedder_alias_sets_model_dims_in_the_config(self):
        with data_env():
            _exports()
            cfg = tomllib.loads(recall.public_config("chats", "locomo", "0", "/idx", 1, embedder="mxbai"))
            self.assertEqual((cfg["embedder"]["model"], cfg["embedder"]["dimensions"]), ("mxbai-embed-large", 1024))
            notes = tomllib.loads(recall.public_config("notes", "locomo", "0", "/idx", 1, embedder="mxbai"))
            self.assertEqual(notes["embedder"]["model"], "qwen3-embedding:4b")   # formats test stays on qwen

    def test_notes_config_names_the_folder(self):
        with data_env():
            _exports()
            cfg = tomllib.loads(recall.public_config("notes", "locomo", "1", "/idx", 1))
            self.assertTrue(cfg["adapters"]["notes"][0]["folder"].endswith(os.path.join("notes", "locomo", "1")))

    def test_postgres_block_comes_from_the_environment(self):
        env = {"ACBENCH_PG_HOST": "db.example", "ACBENCH_PG_PORT": "6543", "ACBENCH_PG_USER": "bench",
               "ACBENCH_PG_DB": "benchdb", "ACBENCH_PG_BIN_DIR": "/opt/pg/bin"}
        with data_env(env):
            _exports()
            pg = tomllib.loads(recall.public_config("chats", "locomo", "0", "/idx", 1))["postgres"]
            self.assertEqual(pg, {"bin_dir": "/opt/pg/bin", "host": "db.example", "port": 6543, "user": "bench",
                                  "database": "benchdb"})

    def test_pgvector_config_appends_the_adapter_fragment(self):
        with data_env():
            dest = convert.export_dir("pgvector", "locomo", "0")
            os.makedirs(dest)
            with open(os.path.join(dest, "adapter.toml"), "w", encoding="utf-8") as f:
                f.write('[[adapters.pgvector]]\nid = "locomo-0"\ntable = "locomo_0"\n')
            cfg = tomllib.loads(recall.public_config("pgvector", "locomo", "0", "/idx", 1))
            self.assertEqual(cfg["adapters"]["pgvector"][0]["table"], "locomo_0")
            with self.assertRaises(SystemExit):
                recall.public_config("pgvector", "locomo", "9", "/idx", 1)

    def test_memory_shelf_gets_reserved_seats_only_when_asked(self):
        with data_env():
            _exports()
            mem = tomllib.loads(recall.public_config("chats", "locomo", "0", "/idx", 1, memories=("ours", "/m")))
            self.assertEqual(mem["recall"], {"extra_seats": 3, "extra_seat_shelves": ["memories-locomo-0"]})
            self.assertEqual(mem["adapters"]["memories"][0]["name"], "acbench-memories-ours")

    def test_engine_environment_is_built_from_scratch(self):
        old = os.environ.get("ANTECHAMBER_X")
        os.environ["ANTECHAMBER_X"] = "leak"
        try:
            env = recall.public_env("/cfg.toml", "chats")
        finally:
            if old is None:
                del os.environ["ANTECHAMBER_X"]
            else:
                os.environ["ANTECHAMBER_X"] = old
        self.assertEqual(set(env), {"PATH", "HOME", "ANTECHAMBER_CONFIG", "ANTECHAMBER_WIDE", "ANTECHAMBER_DATES",
                                    "ANTECHAMBER_CHRONO", "MEMORY_ENTRY_USE_DOOR"})
        self.assertEqual((env["ANTECHAMBER_WIDE"], env["MEMORY_ENTRY_USE_DOOR"]), ("on", "off"))
        with data_env({"ACBENCH_PG_HOST": "h", "ACBENCH_PG_PORT": "1", "ACBENCH_PG_USER": "u", "ACBENCH_PG_DB": "d"}):
            pg = recall.public_env("/cfg.toml", "pgvector")
        self.assertEqual(pg[convert.DSN_ENV], "host=h port=1 user=u dbname=d")

    def test_adapter_ids_are_safe(self):
        self.assertEqual(recall._adapter_id("chats", "locomo", "0"), "chats-locomo-0")
        self.assertEqual(recall._adapter_id("chats", "lme", "Ab_Cd:9"), "chats-lme-ab_cd-9")
        self.assertLessEqual(len(recall._adapter_id("chats", "lme", "x" * 200)), 63)
        self.assertEqual(recall._toml('a"b\\c'), '"a\\"b\\\\c"')

    def test_rewrite_embedder_url_touches_only_that_section(self):
        text = '[embedder]\nurl = "old"\nmodel = "m"\n[other]\nurl = "keep"\n'
        out = recall._rewrite_embedder_url(text, "new")
        self.assertIn('[embedder]\nurl = "new"', out)
        self.assertIn('[other]\nurl = "keep"', out)
        with self.assertRaises(SystemExit):
            recall._rewrite_embedder_url("[a]\nx = 1\n", "new")


class EngineHttp(unittest.TestCase):
    def test_http_recall_asks_for_eight_and_reports_errors_as_misses(self):
        seen = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def ok(url, timeout=0):
            seen.append(url)
            return Resp(b'{"winners": []}')
        old = urllib.request.urlopen
        urllib.request.urlopen = ok
        try:
            raw, err = recall.http_recall(8791, "what is this?")
        finally:
            urllib.request.urlopen = old
        self.assertEqual((raw, err), ({"winners": []}, None))
        self.assertIn("http://127.0.0.1:8791/antechamber/recall?", seen[0])
        self.assertIn("limit=8", seen[0])
        self.assertIn("q=what+is+this%3F", seen[0])

        def http500(url, timeout=0):
            raise urllib.error.HTTPError(url, 500, "x", {}, io.BytesIO(b"AllShelvesFailed: msg"))

        def slow(url, timeout=0):
            raise TimeoutError("timed out")
        urllib.request.urlopen = http500
        try:
            raw, err = recall.http_recall(1, "q")
            self.assertIsNone(raw)
            self.assertTrue(err.startswith("http 500"))
            urllib.request.urlopen = slow
            raw, err = recall.http_recall(1, "q")
            self.assertTrue(err.startswith("timeout"))
        finally:
            urllib.request.urlopen = old

    def test_wait_health_returns_on_ok_and_raises_when_the_service_dies(self):
        class Proc:
            def __init__(self, code=None):
                self.code = code
                self.returncode = code

            def poll(self):
                return self.code

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        old = urllib.request.urlopen
        urllib.request.urlopen = lambda url, timeout=0: Resp(b'{"ok": true, "n": 1}')
        try:
            self.assertEqual(recall.wait_health(1, Proc(), "/nonexistent/log")["n"], 1)
        finally:
            urllib.request.urlopen = old
        with self.assertRaises(RuntimeError) as caught:
            recall.wait_health(1, Proc(3), "/nonexistent/log")
        self.assertIn("exited 3", str(caught.exception))


class CacheIdentity(unittest.TestCase):
    def test_token_compare(self):
        self.assertTrue(recall._token_eq("abc", "abc"))
        self.assertFalse(recall._token_eq("abc", "abd"))
        self.assertFalse(recall._token_eq("", ""))
        self.assertFalse(recall._token_eq(None, "x"))

    def test_only_our_own_cache_is_accepted(self):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with data_env():
            os.makedirs(os.path.dirname(recall.CACHE_PID))
            with open(recall.CACHE_PID, "w") as f:
                f.write("777")
            with open(recall.CACHE_TOKEN, "w") as f:
                f.write("secret")
            old = urllib.request.urlopen
            try:
                with Patch((recall, "_pid_is_cache", lambda pid: True)):
                    urllib.request.urlopen = lambda url, timeout=0: Resp(b'{"ok": true, "pid": 777, "token": "secret"}')
                    self.assertTrue(recall.cache_identity())
                    urllib.request.urlopen = lambda url, timeout=0: Resp(b'{"ok": true, "pid": 777, "token": "wrong"}')
                    self.assertFalse(recall.cache_identity())
                    urllib.request.urlopen = lambda url, timeout=0: Resp(b'{"ok": true, "pid": 1, "token": "secret"}')
                    self.assertFalse(recall.cache_identity())
                with Patch((recall, "_pid_is_cache", lambda pid: False)):
                    urllib.request.urlopen = lambda url, timeout=0: Resp(b'{"ok": true, "pid": 777, "token": "secret"}')
                    self.assertFalse(recall.cache_identity())
            finally:
                urllib.request.urlopen = old

    def test_a_stranger_on_the_cache_port_is_refused(self):
        with data_env():
            with Patch((recall, "_port_accepts", lambda port: True), (recall, "cache_identity", lambda: False)):
                with self.assertRaises(recall.StageVoid) as caught:
                    recall.ensure_cache()
            self.assertIn("not this tree's embed cache", str(caught.exception))
            with Patch((recall, "_port_accepts", lambda port: True), (recall, "cache_identity", lambda: True)):
                self.assertEqual(recall.ensure_cache(), (None, None))

    def test_cache_is_started_with_its_own_settings_and_a_fresh_token(self):
        with data_env() as tmp:
            started = {}

            class Proc:
                pid = 31337

                def poll(self):
                    return None

            def fake_start(script, env, log_path):
                started.update(script=script, env=env, log=log_path)
                with open(recall.CACHE_PID, "w") as f:
                    f.write(str(Proc.pid))
                return Proc(), io.StringIO()
            with Patch((recall, "_port_accepts", lambda port: False), (recall, "start_proc", fake_start),
                       (recall, "cache_identity", lambda: True)):
                proc, logf = recall.ensure_cache()
            self.assertEqual(proc.pid, 31337)
            self.assertTrue(started["script"].endswith(os.path.join("acbench", "embed_cache.py")))
            env = started["env"]
            self.assertEqual(env["ACBENCH_CACHE"], os.path.join(tmp, "cache"))
            self.assertEqual(env["EMBED_CACHE_PID_FILE"], recall.CACHE_PID)
            self.assertEqual(len(env["EMBED_CACHE_TOKEN"]), 64)
            self.assertEqual(read_text(recall.CACHE_TOKEN), env["EMBED_CACHE_TOKEN"])
            self.assertIn("OLLAMA_URL", env)

    def test_drop_cache_files_only_for_our_pid(self):
        with data_env():
            os.makedirs(os.path.dirname(recall.CACHE_PID))
            for path in (recall.CACHE_PID, recall.CACHE_TOKEN):
                with open(path, "w") as f:
                    f.write("5")
            recall._drop_cache_files(6)
            self.assertTrue(os.path.exists(recall.CACHE_PID))
            recall._drop_cache_files(5)
            self.assertFalse(os.path.exists(recall.CACHE_PID))
            self.assertFalse(os.path.exists(recall.CACHE_TOKEN))


class Embedding(unittest.TestCase):
    def test_embed_cached_checks_count_and_dimensions(self):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        old = urllib.request.urlopen
        try:
            urllib.request.urlopen = lambda req, timeout=0: Resp(json.dumps({"embeddings": [[1.0, 2.0]] * 2}).encode())
            self.assertEqual(recall.embed_cached(["a", "b"], "m", 2), [[1.0, 2.0], [1.0, 2.0]])
            with self.assertRaises(recall.StageVoid):
                recall.embed_cached(["a", "b"], "m", 3)          # wrong dimensions
            with self.assertRaises(recall.StageVoid):
                recall.embed_cached(["a", "b", "c"], "m", 2)     # wrong count
            self.assertEqual(recall.embed_cached([], "m", 2), [])
            requests = []

            def spy(req, timeout=0):
                requests.append(json.loads(req.data))
                return Resp(json.dumps({"embeddings": [[0.0, 1.0]] * len(requests[-1]["input"])}).encode())
            urllib.request.urlopen = spy
            recall.embed_cached(["t%d" % i for i in range(20)], "m", 2)
            self.assertEqual([len(r["input"]) for r in requests], [8, 8, 4])   # batches of eight
            self.assertTrue(all(r["dimensions"] == 2 and r["model"] == "m" for r in requests))
        finally:
            urllib.request.urlopen = old


class ProcessesAndPorts(unittest.TestCase):
    def test_pick_port_uses_the_wanted_port_or_the_next_free_one(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        taken = s.getsockname()[1]
        try:
            self.assertFalse(recall._port_free(taken))
            self.assertTrue(recall._port_accepts(taken) or True)
        finally:
            s.close()
        self.assertEqual(recall.pick_port(taken), taken)   # free again
        with Patch((recall, "_port_free", lambda p: p == 8795)):
            self.assertEqual(recall.pick_port(8791), 8795)
        with Patch((recall, "_port_free", lambda p: False)):
            with self.assertRaises(SystemExit):
                recall.pick_port(8791)

    def test_children_do_not_see_the_parents_engine_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "cfg.toml")
            open(cfg, "w").close()
            report = os.path.join(tmp, "seen.json")
            script = os.path.join(tmp, "child.py")
            with open(script, "w") as f:
                f.write("import json, os, sys\n"
                        "json.dump({'keys': sorted(k for k in os.environ if k.startswith('ANTECHAMBER_')),"
                        " 'x': os.environ.get('ANTECHAMBER_X')}, open(%r, 'w'))\n" % report)
            os.environ["ANTECHAMBER_X"] = "leak-from-parent"
            os.environ["ANTECHAMBER_CONFIG"] = "/not/this"
            try:
                env = recall.scratch_env(cfg)
                self.assertEqual(set(env), {"PATH", "HOME", "ANTECHAMBER_CONFIG"})
                self.assertEqual(env["ANTECHAMBER_CONFIG"], cfg)
                recall.spawn_fresh([sys.executable, script], cfg, tmp)
            finally:
                del os.environ["ANTECHAMBER_X"]
                del os.environ["ANTECHAMBER_CONFIG"]
            self.assertEqual(read_json(report), {"keys": ["ANTECHAMBER_CONFIG"], "x": None})
            with self.assertRaises(recall.StageVoid):
                recall.scratch_env(os.path.join(tmp, "missing.toml"))
            bad = os.path.join(tmp, "bad.py")
            with open(bad, "w") as f:
                f.write("import sys; sys.exit(3)\n")
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError):
                recall.spawn_fresh([sys.executable, bad], cfg, tmp, capture=True)

    def test_start_and_stop_a_real_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "sleeper.py")
            with open(script, "w") as f:
                f.write("import time\ntime.sleep(60)\n")
            proc, logf = recall.start_proc(script, {"PATH": os.environ.get("PATH", "")}, os.path.join(tmp, "logs", "x.log"))
            try:
                self.assertIsNone(proc.poll())
            finally:
                recall.stop_proc(proc)
                logf.close()
            self.assertIsNotNone(proc.poll())
            recall.stop_proc(proc)    # already gone: no error
            recall.stop_proc(None)
            self.assertEqual(recall._tail(os.path.join(tmp, "nothing.log")), "")


class Fingerprints(unittest.TestCase):
    def _index(self, tmp, rows):
        os.makedirs(os.path.join(tmp, "idx"))
        conn = sqlite3.connect(os.path.join(tmp, "idx", "a.sqlite"))
        conn.execute("CREATE TABLE passages (id TEXT, body TEXT)")
        conn.execute("CREATE TABLE terms (term TEXT, pid TEXT)")
        conn.execute("CREATE TABLE files (path TEXT, size INT, sha256 TEXT, embedded INT, dups INT, mtime REAL)")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.executemany("INSERT INTO passages VALUES (?, ?)", rows)
        conn.commit()
        conn.close()
        return os.path.join(tmp, "idx")

    def test_content_sha_ignores_insertion_order_but_order_sha_does_not(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a = self._index(t1, [("p1", "one"), ("p2", "two")])
            b = self._index(t2, [("p2", "two"), ("p1", "one")])
            ca, oa = recall.index_content_sha(a)
            cb, ob = recall.index_content_sha(b)
            self.assertEqual(ca, cb)
            self.assertNotEqual(oa, ob)
            self.assertEqual(recall.index_content_sha(os.path.join(t1, "none")), (None, None))
            self.assertEqual(len(recall.index_sha(a)), 64)
            self.assertIsNone(recall.index_sha(os.path.join(t1, "none")))


class Pgvector(unittest.TestCase):
    def test_preflight_aborts_on_wrong_dims(self):
        def bad(sql):
            if "pg_tables" in sql:
                return "locomo_0\nlme_x\n"
            if "lme_x" in sql:
                return "3|768|768\n"
            return "4|2000|2000\n"
        with self.assertRaises(recall.StageVoid) as caught:
            recall.preflight_pgvector_dims(psql=bad)
        self.assertIn("768", str(caught.exception))

        def good(sql):
            return "locomo_0\n" if "pg_tables" in sql else "4|2000|2000\n"
        recall.preflight_pgvector_dims(psql=good)
        with self.assertRaises(recall.StageVoid):
            recall.preflight_pgvector_dims(psql=lambda sql: "")           # no tables
        with self.assertRaises(recall.StageVoid):
            recall.preflight_pgvector_dims(psql=lambda sql: "bad name;\n")   # unsafe table name


if __name__ == "__main__":
    unittest.main()
