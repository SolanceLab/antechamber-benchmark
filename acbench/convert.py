# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Turn LoCoMo and LongMemEval into Antechamber chat exports, note folders, or pgvector tables.

usage: python3 -m acbench convert --bench locomo|lme --format chats|notes|pgvector [--limit N] [--prelim]

Reads the two datasets from ACBENCH_DATA (data/download.sh puts them there) and writes the converted copies under
ACBENCH_DATA/<format>/<bench>/<unit>/. Nothing here is ever committed.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

from . import settings

MIN_NOTE = 20  # notes.passages drops a shorter block
PACKAGE = os.path.dirname(os.path.abspath(__file__))

# Antechamber's pgvector adapter (engine/adapters/pgvector.py): plain identifiers only,
# one cosine-search table, date optional. The FORMAT test (chats / notes / vector DB) uses
# one embedder for all three: qwen3-embedding:4b at 2000. nomic and mxbai are recall-only
# flags on chats and plain search, not a second pgvector default.
DSN_ENV = "ACBENCH_PGVECTOR_DSN"
# The embedding cache (acbench/embed_cache.py) sits in front of Ollama: byte-identical vectors, warmed vectors reused.
EMBED_URL = "http://localhost:%d/api/embed" % settings.CACHE_PORT
# Aliases for `recall --embedder`. Format conversion always uses qwen.
ADAPTER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")  # adapters/index.py ID_RE
EMBEDDERS = {
    "qwen": {"model": "qwen3-embedding:4b", "dimensions": 2000},
    "nomic": {"model": "nomic-embed-text", "dimensions": 768},
    "mxbai": {"model": "mxbai-embed-large", "dimensions": 1024},
}
PG_SCHEMA = "acbench"   # the benchmark owns this schema; nothing in "public" is ever read or changed
DEFAULT_EMBEDDER = "qwen"
PG_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")  # adapters/pgvector.py IDENT
EMBED = {fmt: dict(EMBEDDERS["qwen"]) for fmt in ("chats", "notes", "pgvector")}
_db_ready = False


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_locomo_when(s):
    # "1:56 pm on 8 May, 2023"
    clock, day = s.split(" on ", 1)
    return datetime.datetime.strptime(day.strip() + " " + clock.strip(), "%d %B, %Y %I:%M %p")


def parse_lme_when(s):
    # "2023/05/20 (Sat) 02:21"
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", s)
    if not m:
        raise ValueError("bad haystack date: %r" % s)
    y, mo, d, h, mi = map(int, m.groups())
    return datetime.datetime(y, mo, d, h, mi)


def paragraphs(lines):
    """Blank line between blocks; a block under 20 characters would be dropped by the notes reader,
    so a short turn stays on its own line inside a neighbour's block."""
    groups, cur = [], []

    def weight(g):
        return sum(len(x) for x in g) + max(0, len(g) - 1)

    for line in lines:
        if cur and weight(cur) >= MIN_NOTE and len(line) >= MIN_NOTE:
            groups.append(cur)
            cur = []
        cur.append(line)
    if cur:
        groups.append(cur)
    if len(groups) >= 2 and weight(groups[-1]) < MIN_NOTE:
        groups[-2].extend(groups[-1])
        groups.pop()
    return ["\n".join(g) for g in groups]


def locomo_sessions(item):
    conv = item["conversation"]
    a, b = conv["speaker_a"], conv["speaker_b"]
    n = 1
    while "session_%d" % n in conv:
        when = parse_locomo_when(conv["session_%d_date_time" % n])
        label = conv["session_%d_date_time" % n]
        turns = []
        for t in conv["session_%d" % n]:
            src = (t.get("text") or "").strip()
            if not src:
                continue
            name = t["speaker"]
            sender = "human" if name == a else "assistant" if name == b else "human"
            turns.append({"id": str(t["dia_id"]), "sender": sender, "name": name, "text": src})
        yield {"n": n, "file": "session_%02d.md" % n, "when": when, "label": label, "turns": turns}
        n += 1


def lme_sessions(q):
    rows = zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])
    for n, (sid, date, sess) in enumerate(rows):
        sid = str(sid)
        turns = []
        for i, t in enumerate(sess):
            src = (t.get("content") or "").strip()
            if not src:
                continue
            role = t.get("role") if t.get("role") in ("user", "assistant") else "user"
            turns.append({"id": "%s:%d" % (sid, i), "sender": role, "name": role, "text": src})
        yield {"n": n, "file": "%04d-%s.md" % (n, sid), "when": parse_lme_when(date),
               "label": date, "turns": turns}


def chat_text(bench, turn):
    if bench == "locomo":
        return "%s: %s" % (turn["name"], turn["text"])
    return turn["text"]


def table_name(bench, unit):
    """One safe table per unit. Bench is included so locomo and lme cannot collide in one database."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", "%s_%s" % (bench, unit)).lower()
    if not safe or safe[0].isdigit():
        safe = "u_" + safe
    safe = safe[:63]
    if not PG_IDENT.match(safe):
        raise ValueError("no safe table name for %s %s" % (bench, unit))
    return safe


def adapter_instance_id(bench, unit):
    safe = re.sub(r"[^a-z0-9_-]", "-", ("%s-%s" % (bench, unit)).lower()).strip("-")
    if not safe or not safe[0].isalnum():
        safe = "u-" + safe
    safe = safe[:63]
    if not ADAPTER_ID.match(safe):
        raise ValueError("no safe adapter id for %s %s" % (bench, unit))
    return safe


def _sql_lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def _quote_ident(name):
    if not PG_IDENT.match(name):
        raise ValueError("refusing identifier %r" % name)
    return '"' + name + '"'


def _psql_cmd(db, *extra):
    return [settings.psql(), "-h", settings.pg_host(), "-p", settings.pg_port(), "-U", settings.pg_user(),
            "-d", db, *extra]


def _psql(db, sql):
    """Run SQL. The server's maintenance database is used only to create the benchmark database; every other
    statement stays in the benchmark database."""
    pg_db = settings.pg_db()
    if db == "postgres":
        allowed = (
            "SELECT 1 FROM pg_database WHERE datname = '%s'" % pg_db,
            "CREATE DATABASE %s" % _quote_ident(pg_db),
        )
        if sql not in allowed:
            raise RuntimeError("refusing to touch database postgres")
    elif db != pg_db:
        raise RuntimeError("refusing to touch database %s" % db)
    env = os.environ.copy()
    env["PGCLIENTENCODING"] = "UTF-8"
    # stdin, not -c: one unit is thousands of 2000-d vectors and will not fit in argv.
    cmd = _psql_cmd(db, "-v", "ON_ERROR_STOP=1", "-tA", "--no-psqlrc")
    r = subprocess.run(cmd, input=sql, capture_output=True, text=True, encoding="utf-8", env=env)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:800] or "psql failed")
    return r.stdout


def ensure_db():
    global _db_ready
    if _db_ready:
        return
    pg_db = settings.pg_db()
    probe = subprocess.run(
        _psql_cmd(pg_db, "-tA", "--no-psqlrc", "-c", "SELECT 1"),
        capture_output=True, text=True, encoding="utf-8")
    if probe.returncode:
        if "does not exist" not in (probe.stderr or ""):
            raise RuntimeError(probe.stderr.strip()[:800] or "cannot connect to %s" % pg_db)
        _psql("postgres", "CREATE DATABASE %s" % _quote_ident(pg_db))
    _psql(pg_db, "CREATE EXTENSION IF NOT EXISTS vector")
    _db_ready = True


def embed_texts(texts, model, dims):
    """Local Ollama only. Returns one vector per text, in order, at `dims`."""
    out = []
    step = 16
    for i in range(0, len(texts), step):
        chunk = texts[i:i + step]
        body = json.dumps({"model": model, "input": chunk,
                           "dimensions": dims, "keep_alive": -1}).encode()
        req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError("ollama embed failed: %s" % detail) from e
        vecs = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vecs, list) or len(vecs) != len(chunk):
            raise RuntimeError("embedder returned %s vectors for %d texts" % (
                len(vecs) if isinstance(vecs, list) else "none", len(chunk)))
        for v in vecs:
            if not isinstance(v, list) or len(v) != dims:
                raise RuntimeError("expected %d dimensions, got %s" % (
                    dims, len(v) if isinstance(v, list) else type(v).__name__))
        out.extend(vecs)
        print("embedded %d/%d" % (len(out), len(texts)), file=sys.stderr, flush=True)
    return out


def _vec_sql(v):
    return "'[" + ",".join(repr(float(x)) for x in v) + "]'::vector"


def replace_table(table, rows, vecs, dims):
    ident = _quote_ident(PG_SCHEMA) + "." + _quote_ident(table)
    parts = [
        "BEGIN",
        "CREATE SCHEMA IF NOT EXISTS %s" % _quote_ident(PG_SCHEMA),
        "DROP TABLE IF EXISTS %s" % ident,
        "CREATE TABLE %s ("
        "seq bigint PRIMARY KEY, "
        "id text NOT NULL, "
        "body text NOT NULL, "
        "embedding vector(%d) NOT NULL, "
        "created_at timestamptz NOT NULL)" % (ident, dims),
    ]
    for i in range(0, len(rows), 20):
        vals = []
        for j, (row, vec) in enumerate(zip(rows[i:i + 20], vecs[i:i + 20])):
            vals.append("(%d, %s, %s, %s, %s::timestamptz)" % (
                i + j + 1, _sql_lit(row["id"]), _sql_lit(row["text"]),
                _vec_sql(vec), _sql_lit(row["date"])))
        parts.append("INSERT INTO %s (seq, id, body, embedding, created_at) VALUES %s" % (
            ident, ", ".join(vals)))
    parts.append("COMMIT")
    _psql(settings.pg_db(), ";\n".join(parts) + ";")


def _toml_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def export_dir(fmt, bench, unit):
    return os.path.join(settings.data_dir(), fmt, bench, unit)


def write_pgvector(bench, unit, sessions):
    """One table and one [[adapters.pgvector]] snippet. Text matches chats; date is the session's."""
    ensure_db()
    table = table_name(bench, unit)
    iid = adapter_instance_id(bench, unit)
    spec = EMBED["pgvector"]
    rows = []
    for s in sessions:
        when = iso(s["when"])
        for t in s["turns"]:
            rows.append({"id": t["id"], "text": chat_text(bench, t), "date": when})
    vecs = embed_texts([r["text"] for r in rows], spec["model"], spec["dimensions"])
    replace_table(table, rows, vecs, spec["dimensions"])
    dest = export_dir("pgvector", bench, unit)
    reset(dest)
    name = "%s %s" % (bench, unit)
    snippet = "\n".join([
        "[[adapters.pgvector]]",
        "id = %s" % _toml_str(iid),
        "name = %s" % _toml_str(name),
        "dsn_env = %s" % _toml_str(DSN_ENV),
        "schema = %s" % _toml_str(PG_SCHEMA),
        "table = %s" % _toml_str(table),
        "id_col = \"id\"",
        "text_col = \"body\"",
        "vec_col = \"embedding\"",
        "date_col = \"created_at\"",
        "limit = 20",
        "statement_timeout_ms = 3000",
        "",
    ])
    with open(os.path.join(dest, "adapter.toml"), "w", encoding="utf-8") as f:
        f.write(snippet)


def note_line(bench, turn):
    name = turn["name"] if bench == "locomo" else turn["sender"]
    return "%s: %s" % (name, norm(turn["text"]))


def reset(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def write_chats(bench, unit, sessions):
    dest = export_dir("chats", bench, unit)
    reset(dest)
    convs = []
    for s in sessions:
        msgs = []
        for i, t in enumerate(s["turns"]):
            msgs.append({"uuid": t["id"], "sender": t["sender"],
                         "created_at": iso(s["when"] + datetime.timedelta(seconds=i)),
                         "text": chat_text(bench, t)})
        convs.append({"uuid": "%s-%s-s%d" % (bench, unit, s["n"]), "name": s["label"],
                      "created_at": iso(s["when"]), "chat_messages": msgs})
    with open(os.path.join(dest, "conversations.json"), "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False)


def write_notes(bench, unit, sessions):
    dest = export_dir("notes", bench, unit)
    reset(dest)
    for s in sessions:
        lines = [note_line(bench, t) for t in s["turns"]]
        paras = paragraphs(lines)
        if len(paras) == 1 and len(paras[0]) < MIN_NOTE:
            paras[0] = paras[0] + "\n" + s["label"]
        body = "\n\n".join(paras)
        text = "---\ndate: %s\n---\n# %s\n\n%s\n" % (iso(s["when"]), s["label"], body)
        with open(os.path.join(dest, s["file"]), "w", encoding="utf-8") as f:
            f.write(text)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_path(bench):
    name = "locomo10.json" if bench == "locomo" else "longmemeval_s_cleaned.json"
    path = os.path.join(settings.data_dir(), name)
    if not os.path.isfile(path):
        raise SystemExit("missing %s. Run data/download.sh first (ACBENCH_DATA is %s)." % (path, settings.data_dir()))
    return path


def units(bench, limit, only=None):
    data = load_json(dataset_path(bench))
    if bench == "locomo":
        for idx, item in enumerate(data):
            if limit is not None and idx >= limit:
                break
            yield str(idx), locomo_sessions(item)
    else:
        kept = 0
        for q in data:
            if only is not None and q["question_id"] not in only:
                continue
            if limit is not None and kept >= limit:
                break
            kept += 1
            yield str(q["question_id"]), lme_sessions(q)


def clear_bench_output(bench, fmt):
    """Empty the benchmark-level output (and, for pgvector, this benchmark's tables) so a limited conversion
    leaves no stale unit from an earlier run to be fingerprinted into the lock."""
    reset(os.path.join(settings.data_dir(), fmt, bench))
    if fmt == "pgvector":
        ensure_db()
        names = _psql(settings.pg_db(), "SELECT tablename FROM pg_tables WHERE schemaname = '%s' "
                                        "AND tablename LIKE '%s\\_%%' ORDER BY 1" % (PG_SCHEMA, bench)).split()
        for name in names:
            _psql(settings.pg_db(), "DROP TABLE IF EXISTS %s.%s" % (_quote_ident(PG_SCHEMA), _quote_ident(name)))


def convert(bench, fmt, limit=None, only=None):
    n = 0
    clear_bench_output(bench, fmt)   # every conversion, full or limited: no stale unit or table survives
    writer = {"chats": write_chats, "notes": write_notes, "pgvector": write_pgvector}[fmt]
    for unit, sessions in units(bench, limit, only):
        writer(bench, unit, list(sessions))
        n += 1
    return n


def prelim_lme_ids():
    return {q["question_id"] for q in load_json(os.path.join(PACKAGE, "questions", "PRELIM_SET.json"))["lme"]}


def add_arguments(p):
    p.add_argument("--bench", required=True, choices=("locomo", "lme"))
    p.add_argument("--format", dest="fmt", required=True, choices=("chats", "notes", "pgvector"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prelim", action="store_true", help="LongMemEval: only the 60 PRELIM_SET haystacks")


def run_cli(a):
    only = prelim_lme_ids() if a.prelim and a.bench == "lme" else None
    if a.fmt == "pgvector":
        # the vectors come through the embedding cache, so it must be up while the tables are built
        from . import recall
        proc, logf = recall.ensure_cache()
        try:
            n = convert(a.bench, a.fmt, a.limit, only)
        finally:
            if proc is not None:
                recall.stop_proc(proc)
                recall._drop_cache_files(proc.pid)
            if logf is not None:
                logf.close()
    else:
        n = convert(a.bench, a.fmt, a.limit, only)
    print("wrote %d %s %s" % (n, a.bench, a.fmt))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m acbench convert",
        usage="python3 -m acbench convert --bench locomo|lme --format chats|notes|pgvector [--limit N]")
    add_arguments(p)
    run_cli(p.parse_args(argv))


if __name__ == "__main__":
    main()
