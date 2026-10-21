# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The lock: every input that can move a number is fingerprinted. Trust lives here, not in whoever runs it.

  python3 -m acbench.lock freeze [plain|engine|pgvector|all] [locomo|lme]
                                   -> writes LOCK.json (commit it on its own; that commit IS the version).
                                      Default all: every embedder, the engine and the pgvector tables. A narrower
                                      scope freezes only what that kind of run uses, including the converted
                                      data it reads (plain: chats; engine: + notes; pgvector: + pgvector). A bench
                                      name limits that to one benchmark's converted folders.
  python3 -m acbench.lock check    -> exit 0 if every fingerprint matches LOCK.json, else prints each difference, exit 1

The runner calls require() at the start of a stage, every 25 questions, and at the end. There is no override.
freeze refuses to overwrite a LOCK.json of the same VERSION whose fingerprints differ. Bump VERSION first.

What is fingerprinted: the two dataset files, this package's code and question lists, the embedding models'
weights (Ollama's full digest), Python's major.minor, Ollama's version, and, for the formats that use them, the
Antechamber engine folder and its git commit, and the pgvector tables. A run only checks what its format uses:

  plain      data, code, settings, versions, the embedder's digest
  engine     plain + the engine folder (ANTECHAMBER_DIR/engine) and its commit      (chats, notes)
  pgvector   engine + psql's version and a hash of every pgvector table             (pgvector)
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

from . import settings

LOCK = os.path.join(settings.ROOT, "LOCK.json")
PACKAGE = settings.PACKAGE
VERSION = "v4.4.3"   # the recall path of the v4.4.x harness; a change to any fingerprinted input is a new version
SETTINGS = {"limit": 8,
            "embedder": "qwen3-embedding:4b", "embedder_dims": 2000,
            "pgvector_embedder": "qwen3-embedding:4b", "pgvector_dims": 2000,
            "nomic_embedder": "nomic-embed-text", "nomic_dims": 768,
            "mxbai_embedder": "mxbai-embed-large", "mxbai_dims": 1024}
EMBEDDER_MODELS = {"qwen": SETTINGS["embedder"], "nomic": SETTINGS["nomic_embedder"], "mxbai": SETTINGS["mxbai_embedder"]}
SCOPES = ("plain", "engine", "pgvector", "all")
UNKNOWN = "unknown"   # the engine folder is not a git checkout


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _dir_sha(d):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        # caches and run outputs are derived, and they grow during a run; hashing them would void a stage at the
        # 25-question re-check.
        dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", "runs", "cache", "node_modules", ".pytest_cache",
                                                      ".git"))
        for f in sorted(files):
            p = os.path.join(root, f)
            # .DS_Store is Finder metadata. It is not an input, and hashing it voids a stage
            # whenever a directory is opened in Finder.
            if f.endswith(".pyc") or f.endswith(".log") or f == ".DS_Store":
                continue
            h.update(os.path.relpath(p, d).encode() + b"\0" + _sha(p).encode() + b"\n")
    return h.hexdigest()


def _cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True)
    except OSError:
        return "MISSING"
    text = (r.stdout or r.stderr or "").strip()
    return text.splitlines()[0] if text else "MISSING"


def engine_commit():
    """The engine checkout's git commit, or UNKNOWN when it is not a git checkout (or ANTECHAMBER_DIR is unset)."""
    path = settings.antechamber_dir()
    if not path:
        return UNKNOWN
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return UNKNOWN
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else UNKNOWN


def _get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def ollama_tags():
    """Full model records from the local Ollama. Digests come from /api/tags, not the short `ollama list` id."""
    return _get_json(settings.ollama_tags_url())


def ollama_version():
    got = _get_json(settings.ollama_url() + "/api/version")
    return str(got.get("version")) if isinstance(got, dict) and got.get("version") else "MISSING"


def model_digest(name, tags=None):
    """The full digest Ollama reports for `name` (exact, else name:latest, else the same model family)."""
    if tags is None:
        tags = ollama_tags()
    if not tags:
        return "MISSING"
    exact = latest = prefix = None
    for m in tags.get("models") or []:
        got = m.get("name") or ""
        digest = m.get("digest") or ""
        if not digest:
            continue
        if got == name:
            exact = digest
        elif got == name + ":latest":
            latest = digest
        elif got.split(":")[0] == name and prefix is None:
            prefix = digest
    return exact or latest or prefix or "MISSING"


def _psql(sql):
    cmd = [settings.psql(), "-h", settings.pg_host(), "-p", settings.pg_port(), "-U", settings.pg_user(),
           "-d", settings.pg_db(), "-v", "ON_ERROR_STOP=1", "-q", "-tA", "--no-psqlrc"]
    return subprocess.run(cmd, input=sql, capture_output=True, text=True)


def _pgvector_hashes():
    """sha256 over id, body, created_at, embedding ordered by id, one entry per table."""
    try:
        probe = _psql("SELECT tablename FROM pg_tables WHERE schemaname = 'acbench' ORDER BY 1")
    except OSError:
        return {"pgvector.database": "MISSING"}
    if probe.returncode:
        return {"pgvector.database": "MISSING"}
    names = [t for t in probe.stdout.split() if t]
    out = {}
    for name in names:
        if not name.replace("_", "").isalnum() or not name[0].isalpha():
            return {"pgvector." + name: "MISSING"}
        sql = (
            "COPY (SELECT id, body, "
            "to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'), "
            "embedding::text FROM acbench.\"%s\" ORDER BY id) TO STDOUT" % name
        )
        r = _psql(sql)
        if r.returncode:
            out["pgvector." + name] = "MISSING"
            continue
        out["pgvector." + name] = hashlib.sha256(r.stdout.encode()).hexdigest()
    if not out:
        out["pgvector.tables"] = hashlib.sha256(b"").hexdigest()
    return out


def _files():
    data = settings.data_dir()
    return {"data.locomo": os.path.join(data, "locomo10.json"),
            "data.lme": os.path.join(data, "longmemeval_s_cleaned.json")}


BENCHES = ("locomo", "lme")
DERIVED_ALL = tuple((f, b) for f in ("chats", "notes", "pgvector") for b in BENCHES)


def derived_for(fmt, bench=None):
    """The converted inputs (data/<format>/<bench>) a recall format reads. plain reads the chat export."""
    benches = (bench,) if bench else BENCHES
    f = fmt if fmt in ("notes", "pgvector") else "chats"
    return tuple((f, b) for b in benches)


_SCOPE_FORMATS = {"plain": ("chats",), "engine": ("chats", "notes"), "pgvector": ("chats", "notes", "pgvector"),
                  "all": ("chats", "notes", "pgvector")}


def scope_derived(scope, bench=None):
    """The converted folders a freeze of `scope` must record, so a run that needs them is not refused at once.
    plain reads the chat export; engine adds notes; pgvector adds the vector tables' export. bench narrows to one."""
    benches = (bench,) if bench else BENCHES
    return tuple((f, b) for f in _SCOPE_FORMATS[scope] for b in benches)


def compute(scope="plain", embedders=("qwen",), derived=None):
    """Fingerprints for `scope` (see the module note). embedders names the aliases whose model digest is checked.
    derived lists (format, bench) pairs of converted data (data/<format>/<bench>) to fingerprint as well; an
    unconverted or empty folder is MISSING. scope all fingerprints every converted folder."""
    if scope not in SCOPES:
        raise ValueError("scope must be one of %s" % ", ".join(SCOPES))
    if scope == "all":
        embedders = tuple(EMBEDDER_MODELS)
    wants_engine = scope in ("engine", "pgvector", "all")
    wants_pg = scope in ("pgvector", "all")
    tags = ollama_tags()
    fp = {k: (_sha(p) if os.path.isfile(p) else "MISSING") for k, p in _files().items()}
    fp["code"] = _dir_sha(PACKAGE)
    if scope == "all":
        derived = DERIVED_ALL
    for fmt, bench in sorted(derived or ()):
        d = os.path.join(settings.data_dir(), fmt, bench)
        fp["derived.%s.%s" % (fmt, bench)] = _dir_sha(d) if os.path.isdir(d) and os.listdir(d) else "MISSING"
    if wants_engine:
        eng = settings.engine_dir()
        fp["engine.dir"] = _dir_sha(eng) if eng and os.path.isdir(eng) else "MISSING"
    wanted = list(embedders)
    if wants_pg and "qwen" not in wanted:
        wanted.append("qwen")
    for alias in sorted(set(wanted)):
        fp["model.%s_digest" % alias] = model_digest(EMBEDDER_MODELS[alias], tags)
    fp["version.python"] = "%d.%d" % sys.version_info[:2]
    fp["version.ollama"] = ollama_version()
    if wants_pg:
        fp["version.psql"] = _cmd([settings.psql(), "--version"])
        fp.update(_pgvector_hashes())
    rec = {"version": VERSION, "settings": SETTINGS, "fingerprints": fp}
    if wants_engine:
        rec["main_commit"] = engine_commit()
    return rec


def _diff(want, have):
    diffs = [f"{k}: locked {want['fingerprints'].get(k)} != now {v}" for k, v in have["fingerprints"].items()
             if want["fingerprints"].get(k) != v]
    for k in ("version", "settings"):
        if want.get(k) != have.get(k):
            diffs.append(f"{k}: locked {want.get(k)} != now {have.get(k)}")
    if "main_commit" in have:
        locked, now = want.get("main_commit"), have["main_commit"]
        if UNKNOWN not in (locked, now) and locked != now:
            diffs.append(f"main_commit: locked {locked} != now {now}")
    if any("MISSING" in str(v) for v in have["fingerprints"].values()):
        diffs.append("an input is MISSING")
    return diffs


def same_version_block(existing, rec):
    """None when freeze may write. A reason string when this VERSION is already locked to different inputs."""
    if not existing or existing.get("version") != rec.get("version"):
        return None
    if existing.get("fingerprints") != rec.get("fingerprints"):
        return "fingerprints"
    for k in ("settings", "main_commit"):
        if existing.get(k) != rec.get(k):
            return k
    return None


def refreeze_block(scope="all", bench=None):
    """The same-version refreeze check, run BEFORE any conversion. It compares exactly what freeze compares
    (inputs, converted folders and pgvector tables) through same_version_block, so a locked version whose inputs
    differ in any way is refused before conversion rewrites a folder or table. None when it may proceed."""
    if not os.path.exists(LOCK):
        return None
    with open(LOCK, encoding="utf-8") as f:
        existing = json.load(f)
    return same_version_block(existing, compute(scope, derived=scope_derived(scope, bench)))


def check(scope="plain", embedders=("qwen",), derived=None):
    """Returns the list of differences (empty = locked inputs unchanged)."""
    if not os.path.exists(LOCK):
        return ["LOCK.json missing. Run `python3 -m acbench.lock freeze` and keep the result."]
    with open(LOCK, encoding="utf-8") as f:
        want = json.load(f)
    return _diff(want, compute(scope, embedders, derived))


def require(scope="plain", embedders=("qwen",), derived=None):
    d = check(scope, embedders, derived)
    if d:
        print("LOCK REFUSED. Inputs differ from LOCK.json:", *("  " + x for x in d), sep="\n", file=sys.stderr)
        sys.exit(1)


def scope_for(fmt):
    """The lock scope a recall format needs."""
    if fmt in ("plain", "plainwin"):
        return "plain"
    return "pgvector" if fmt == "pgvector" else "engine"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else None
    if cmd == "freeze":
        scope = argv[1] if len(argv) > 1 else "all"
        bench = argv[2] if len(argv) > 2 else None
        if scope not in SCOPES:
            sys.exit("scope must be one of %s" % ", ".join(SCOPES))
        if bench is not None and bench not in BENCHES:
            sys.exit("bench must be one of %s" % ", ".join(BENCHES))
        rec = compute(scope, derived=scope_derived(scope, bench))
        if any("MISSING" in str(v) for v in rec["fingerprints"].values()):
            sys.exit("refusing to freeze: an input is MISSING\n" + json.dumps(rec["fingerprints"], indent=1))
        if os.path.exists(LOCK):
            with open(LOCK, encoding="utf-8") as f:
                existing = json.load(f)
            why = same_version_block(existing, rec)
            if why:
                sys.exit("refusing to freeze %s: LOCK.json already records this VERSION with different %s. "
                         "Bump VERSION in lock.py, then freeze that new version." % (VERSION, why))
        with open(LOCK, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        print(f"frozen {VERSION} -> {os.path.basename(LOCK)}")
    elif cmd == "precheck":
        why = refreeze_block("all")
        if why:
            sys.exit("refusing to convert for a freeze: LOCK.json already records this VERSION with different %s. "
                     "Bump VERSION in lock.py first; nothing was converted." % why)
        print("refreeze check clear")
    elif cmd == "check":
        scope = argv[1] if len(argv) > 1 else "all"
        bench = argv[2] if len(argv) > 2 else None
        d = check(scope, tuple(EMBEDDER_MODELS) if scope == "all" else ("qwen",), scope_derived(scope, bench))
        print("LOCK OK" if not d else "LOCK REFUSED\n" + "\n".join(d))
        sys.exit(1 if d else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
