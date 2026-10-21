# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The converters: dataset -> chat export, Markdown notes, vector-database tables. Offline; Postgres and embedding mocked."""
import datetime
import json
import os
import re
import subprocess
import tomllib
import unittest

from acbench import convert
from tests.fixtures import LME, LOCOMO, Patch, data_env, read_json, read_text


class Dates(unittest.TestCase):
    def test_locomo_and_lme_dates(self):
        self.assertEqual(convert.parse_locomo_when("1:56 pm on 8 May, 2023"), datetime.datetime(2023, 5, 8, 13, 56))
        self.assertEqual(convert.parse_lme_when("2023/05/20 (Sat) 02:21"), datetime.datetime(2023, 5, 20, 2, 21))
        with self.assertRaises(ValueError):
            convert.parse_lme_when("last Tuesday")
        self.assertEqual(convert.iso(datetime.datetime(2023, 5, 8, 13, 56)), "2023-05-08T13:56:00Z")


class Chats(unittest.TestCase):
    def test_locomo_chat_export_keeps_every_turn_in_order_with_the_speaker_in_the_text(self):
        with data_env():
            self.assertEqual(convert.convert("locomo", "chats"), 2)
            convs = read_json(os.path.join(convert.export_dir("chats", "locomo", "0"), "conversations.json"))
            self.assertEqual(len(convs), 2)                       # one conversation per session
            first = convs[0]["chat_messages"]
            self.assertEqual([m["uuid"] for m in first], ["D1:1", "D1:2", "D1:3"])
            self.assertEqual([m["sender"] for m in first], ["human", "assistant", "human"])
            self.assertTrue(first[0]["text"].startswith("Mira: I started pottery classes"))
            self.assertEqual(convs[0]["created_at"], "2023-05-08T13:56:00Z")
            self.assertEqual(first[1]["created_at"], "2023-05-08T13:56:01Z")
            self.assertEqual(convs[0]["name"], "1:56 pm on 8 May, 2023")
            self.assertEqual(convs[1]["uuid"], "locomo-0-s2")

    def test_lme_export_uses_roles_and_has_no_speaker_prefix(self):
        with data_env():
            self.assertEqual(convert.convert("lme", "chats"), 3)
            convs = read_json(os.path.join(convert.export_dir("chats", "lme", "fxq1"), "conversations.json"))
            msgs = [m for c in convs for m in c["chat_messages"]]
            self.assertEqual([m["sender"] for m in msgs], ["user", "assistant", "user", "assistant"])
            self.assertEqual(msgs[2]["uuid"], "s-b:0")
            self.assertTrue(msgs[2]["text"].startswith("I cycled along the canal path"))

    def test_limit_stops_early(self):
        with data_env():
            self.assertEqual(convert.convert("locomo", "chats", limit=1), 1)
            self.assertFalse(os.path.exists(convert.export_dir("chats", "locomo", "1")))

    def test_only_filters_lme_haystacks(self):
        with data_env():
            self.assertEqual(convert.convert("lme", "chats", only={"fxq2"}), 1)
            self.assertTrue(os.path.isdir(convert.export_dir("chats", "lme", "fxq2")))
            self.assertFalse(os.path.isdir(convert.export_dir("chats", "lme", "fxq1")))

    def test_limit_counts_after_the_prelim_filter(self):
        with data_env():
            # fxq1 comes first in the dataset but is not in the filter: the limit must not spend itself on it
            self.assertEqual(convert.convert("lme", "chats", limit=1, only={"fxq2"}), 1)
            self.assertTrue(os.path.isdir(convert.export_dir("chats", "lme", "fxq2")))

    def test_reconverting_replaces_the_folder(self):
        with data_env():
            convert.convert("locomo", "chats", limit=1)
            stale = os.path.join(convert.export_dir("chats", "locomo", "0"), "stale.txt")
            with open(stale, "w") as f:
                f.write("x")
            convert.convert("locomo", "chats", limit=1)
            self.assertFalse(os.path.exists(stale))

    def test_a_limited_conversion_leaves_no_stale_unit_or_table_to_be_locked(self):
        tables = {"locomo_0", "locomo_1", "lme_fxq1"}       # left by a prior full run; lme_ is another benchmark's

        def fake_psql(db, sql):
            if sql.startswith("SELECT tablename"):
                return "\n".join(sorted(t for t in tables if t.startswith("locomo_")))
            for m in re.finditer(r'DROP TABLE IF EXISTS "acbench"\."(\w+)"', sql):
                tables.discard(m.group(1))
            for m in re.finditer(r'CREATE TABLE "acbench"\."(\w+)"', sql):
                tables.add(m.group(1))
            return ""

        with data_env():
            with Patch((convert, "_psql", fake_psql), (convert, "ensure_db", lambda: None),
                       (convert, "embed_texts", lambda texts, model, dims: [[0.5] * dims for _ in texts])):
                for fmt in ("chats", "notes", "pgvector"):
                    self.assertEqual(convert.convert("locomo", fmt), 2)       # the prior full run
                tables.add("locomo_1")
                for fmt in ("chats", "notes", "pgvector"):
                    self.assertEqual(convert.convert("locomo", fmt, limit=1), 1)
            for fmt in ("chats", "notes", "pgvector"):
                self.assertEqual(os.listdir(os.path.join(os.environ["ACBENCH_DATA"], fmt, "locomo")), ["0"])
            self.assertEqual(tables, {"locomo_0", "lme_fxq1"})   # stale locomo_1 dropped; other benchmark untouched

    def test_a_full_conversion_also_clears_a_stale_unit_and_table(self):
        tables = {"locomo_stale", "lme_fxq1"}

        def fake_psql(db, sql):
            if sql.startswith("SELECT tablename"):
                return "\n".join(sorted(t for t in tables if t.startswith("locomo_")))
            for m in re.finditer(r'DROP TABLE IF EXISTS "acbench"\."(\w+)"', sql):
                tables.discard(m.group(1))
            for m in re.finditer(r'CREATE TABLE "acbench"\."(\w+)"', sql):
                tables.add(m.group(1))
            return ""

        with data_env():
            for fmt in ("chats", "notes", "pgvector"):
                stale = os.path.join(os.environ["ACBENCH_DATA"], fmt, "locomo", "stale")
                os.makedirs(stale)
            with Patch((convert, "_psql", fake_psql), (convert, "ensure_db", lambda: None),
                       (convert, "embed_texts", lambda texts, model, dims: [[0.5] * dims for _ in texts])):
                for fmt in ("chats", "notes", "pgvector"):
                    self.assertEqual(convert.convert("locomo", fmt), 2)
            for fmt in ("chats", "notes", "pgvector"):
                self.assertEqual(sorted(os.listdir(os.path.join(os.environ["ACBENCH_DATA"], fmt, "locomo"))), ["0", "1"])
            self.assertNotIn("locomo_stale", tables)
            self.assertIn("lme_fxq1", tables)

    def test_pgvector_cleanup_never_touches_an_unrelated_public_table(self):
        tables = {("public", "locomo_archive"), ("public", "lme_notes"), ("acbench", "locomo_stale")}

        def fake_psql(db, sql):
            if sql.startswith("SELECT tablename"):
                m = re.search(r"schemaname = '(\w+)'", sql)
                return "\n".join(sorted(n for s, n in tables if s == m.group(1) and n.startswith("locomo_")))
            for m in re.finditer(r'DROP TABLE IF EXISTS "(\w+)"\."(\w+)"', sql):
                tables.discard((m.group(1), m.group(2)))
            for m in re.finditer(r'CREATE TABLE "(\w+)"\."(\w+)"', sql):
                tables.add((m.group(1), m.group(2)))
            return ""

        with data_env():
            with Patch((convert, "_psql", fake_psql), (convert, "ensure_db", lambda: None),
                       (convert, "embed_texts", lambda texts, model, dims: [[0.5] * dims for _ in texts])):
                self.assertEqual(convert.convert("locomo", "pgvector"), 2)
            self.assertIn(("public", "locomo_archive"), tables)
            self.assertIn(("public", "lme_notes"), tables)
            self.assertNotIn(("acbench", "locomo_stale"), tables)
            self.assertEqual({t for t in tables if t[0] == "acbench"}, {("acbench", "locomo_0"), ("acbench", "locomo_1")})

    def test_missing_dataset_says_how_to_get_it(self):
        with data_env() as tmp:
            os.remove(os.path.join(tmp, "data", "longmemeval_s_cleaned.json"))
            with self.assertRaises(SystemExit) as caught:
                convert.convert("lme", "chats")
            self.assertIn("download.sh", str(caught.exception))


class Notes(unittest.TestCase):
    def test_one_markdown_file_per_session_with_a_date_front_matter(self):
        with data_env():
            convert.convert("locomo", "notes", limit=1)
            folder = convert.export_dir("notes", "locomo", "0")
            self.assertEqual(sorted(os.listdir(folder)), ["session_01.md", "session_02.md"])
            text = read_text(os.path.join(folder, "session_01.md"))
            self.assertTrue(text.startswith("---\ndate: 2023-05-08T13:56:00Z\n---\n# 1:56 pm on 8 May, 2023\n\n"))
            self.assertIn("Mira: I started pottery classes", text)
            self.assertIn("Tobias: We drove to the coast", read_text(os.path.join(folder, "session_02.md")))

    def test_a_block_under_twenty_characters_is_never_alone(self):
        paras = convert.paragraphs(["long enough line number one here", "ok", "another long enough line here too"])
        self.assertTrue(all(len(p) >= convert.MIN_NOTE for p in paras))
        self.assertEqual("\n".join(paras).count("ok"), 1)
        self.assertEqual(convert.paragraphs(["short"]), ["short"])

    def test_lme_notes_use_the_role_as_the_speaker(self):
        with data_env():
            convert.convert("lme", "notes", only={"fxq1"})
            folder = convert.export_dir("notes", "lme", "fxq1")
            self.assertEqual(sorted(os.listdir(folder)), ["0000-s-a.md", "0001-s-b.md"])
            self.assertIn("user: I cycled along the canal path", read_text(os.path.join(folder, "0001-s-b.md")))


class Pgvector(unittest.TestCase):
    def test_tables_adapter_and_sql_for_one_unit(self):
        sql_seen, embedded = [], []

        def fake_psql(db, sql):
            sql_seen.append((db, sql))
            return ""

        def fake_embed(texts, model, dims):
            embedded.append((len(texts), model, dims))
            return [[0.5] * dims for _ in texts]
        with data_env({"ACBENCH_PG_DB": "benchdb"}):
            with Patch((convert, "_psql", fake_psql), (convert, "embed_texts", fake_embed),
                       (convert, "ensure_db", lambda: None)):
                convert.convert("locomo", "pgvector", limit=1)
            self.assertEqual(embedded, [(5, "qwen3-embedding:4b", 2000)])
            db, sql = sql_seen[-1]
            self.assertEqual(db, "benchdb")
            self.assertIn('"acbench"."locomo_0"', sql)
            self.assertIn("vector(2000)", sql)
            self.assertIn("'D1:1'", sql)
            self.assertIn("2023-05-08T13:56:00Z", sql)
            adapter = tomllib.loads(read_text(os.path.join(convert.export_dir("pgvector", "locomo", "0"), "adapter.toml")))
            row = adapter["adapters"]["pgvector"][0]
            self.assertEqual((row["id"], row["table"], row["dsn_env"]), ("locomo-0", "locomo_0", convert.DSN_ENV))
            self.assertEqual((row["id_col"], row["text_col"], row["vec_col"]), ("id", "body", "embedding"))

    def test_psql_touches_only_the_benchmark_database(self):
        env = {"ACBENCH_PG_DB": "benchdb", "ACBENCH_PSQL": "/opt/pg/bin/psql", "ACBENCH_PG_HOST": "h",
               "ACBENCH_PG_PORT": "7", "ACBENCH_PG_USER": "u"}
        with data_env(env):
            calls = []

            def fake_run(cmd, **kw):
                calls.append((cmd, kw))
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            with Patch((convert.subprocess, "run", fake_run)):
                self.assertEqual(convert._psql("benchdb", "SELECT 1"), "ok")
                cmd = calls[-1][0]
                self.assertEqual(cmd[:9], ["/opt/pg/bin/psql", "-h", "h", "-p", "7", "-U", "u", "-d", "benchdb"])
                self.assertEqual(calls[-1][1]["input"], "SELECT 1")
                self.assertEqual(calls[-1][1]["env"]["PGCLIENTENCODING"], "UTF-8")
                convert._psql("postgres", 'CREATE DATABASE "benchdb"')   # allowed: making the benchmark database
                with self.assertRaises(RuntimeError):
                    convert._psql("postgres", "SELECT 1; anything else")
                with self.assertRaises(RuntimeError):
                    convert._psql("someone_elses_db", "SELECT 1")

    def test_psql_failure_surfaces_the_error(self):
        with data_env():
            with Patch((convert.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such table"))):
                with self.assertRaises(RuntimeError) as caught:
                    convert._psql(convert.settings.pg_db(), "SELECT 1")
            self.assertIn("no such table", str(caught.exception))

    def test_identifiers_are_checked(self):
        self.assertEqual(convert.table_name("lme", "e47becba"), "lme_e47becba")
        self.assertEqual(convert.table_name("lme", "9abc"), "lme_9abc")
        self.assertEqual(convert.adapter_instance_id("lme", "Ab_C"), "lme-ab_c")
        with self.assertRaises(ValueError):
            convert._quote_ident("x; y")
        self.assertEqual(convert._sql_lit("it's"), "'it''s'")

    def test_embedder_answers_are_checked(self):
        class Resp:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(self.payload).encode()
        with Patch((convert.urllib.request, "urlopen", lambda req, timeout=0: Resp({"embeddings": [[1.0, 2.0]]}))):
            self.assertEqual(convert.embed_texts(["a"], "m", 2), [[1.0, 2.0]])
            with self.assertRaises(RuntimeError):
                convert.embed_texts(["a"], "m", 3)
            with self.assertRaises(RuntimeError):
                convert.embed_texts(["a", "b"], "m", 2)


class Plumbing(unittest.TestCase):
    def test_embedder_table_and_defaults(self):
        self.assertEqual(convert.EMBEDDERS["qwen"], {"model": "qwen3-embedding:4b", "dimensions": 2000})
        self.assertEqual(convert.DEFAULT_EMBEDDER, "qwen")
        for fmt in ("chats", "notes", "pgvector"):
            self.assertEqual(convert.EMBED[fmt], convert.EMBEDDERS["qwen"])   # the format test uses one embedder

    def test_dataset_fixture_is_in_the_real_shape(self):
        self.assertEqual(len(LOCOMO), 2)
        self.assertEqual(len(LME), 3)

    def test_prelim_ids_come_from_the_question_list(self):
        self.assertEqual(len(convert.prelim_lme_ids()), 60)


if __name__ == "__main__":
    unittest.main()
