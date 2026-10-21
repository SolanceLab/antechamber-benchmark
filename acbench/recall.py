# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The recall stage: ask every question of a set through a retrieval format and score what came back.

Formats (--format):
  plain      standard search: each chat message embedded whole, the 8 closest by cosine. No service, no engine.
  plainwin   plain + the surrounding lines the chats adapter brings (2 before, 3 after): the "smart or generous" check
  chats      Antechamber reading the chat export          (needs ANTECHAMBER_DIR)
  notes      Antechamber reading Markdown notes           (needs ANTECHAMBER_DIR)
  pgvector   Antechamber reading a vector database        (needs ANTECHAMBER_DIR and Postgres with pgvector)

The lock is checked at the start, every 25 questions, and at the end of a stage (lock.py).
The embedding cache (embed_cache.py) sits between everything and Ollama; this tree starts it and refuses any
other listener on its port.
"""
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import convert, lock, settings
from .render import render
from .convert import DEFAULT_EMBEDDER, EMBED, EMBEDDERS
from .sets import (SETS, VARIANT_LIMIT, VARIANTS, _dia_map, group, lme_index, locomo_data, prelim, select,
                   set_segment, split_variant)
from .stats import clustered_ci, mcnemar_exact, paired_ci

_VARIANT = None
MEMORY_EXTRA_SEATS = 3   # reserved seats for the memory shelf (memory rungs only; recall-only runs never use them)
CACHE_PORT = settings.CACHE_PORT
CACHE_PID = os.path.join(settings.cache_dir(), "embed-cache.pid")
RUNS_ROOT = settings.runs_root()
CACHE_TOKEN = os.path.join(settings.cache_dir(), "embed-cache.token")
PACKAGE = settings.PACKAGE
VERSION = lock.VERSION
CACHE_EMBED_URL = "http://localhost:%d/api/embed" % CACHE_PORT
RECALL_LIMIT = settings.RECALL_LIMIT
RUN = os.path.join(RUNS_ROOT, VERSION)
_STAGE = {"scope": "plain", "embedders": (DEFAULT_EMBEDDER,), "derived": ()}


def set_run_dir(path):
    """Send this stage's outputs somewhere other than ACBENCH_RUNS/<version>. Tests use a temp directory."""
    global RUN
    RUN = os.path.abspath(path)
    _FRESH.clear()


def _lock_stage(fmt, embedder=None, bench=None, embedders=None):
    """Remember which inputs this stage depends on, so every later lock check looks at the right ones.
    That includes the converted data this format reads (data/<format>/<bench>)."""
    _STAGE["scope"] = lock.scope_for(fmt)
    _STAGE["derived"] = lock.derived_for(fmt, bench) if bench else ()
    if embedders:
        _STAGE["embedders"] = tuple(embedders)
        return
    _STAGE["embedders"] = (DEFAULT_EMBEDDER if fmt == "pgvector" else (embedder or DEFAULT_EMBEDDER),)


def _require():
    lock.require(_STAGE["scope"], _STAGE["embedders"], _STAGE["derived"])


_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def words(s):
    return _WORD.sub("", (s or "").lower()).split()


def shingles(ws, n=4):
    if not ws:
        return []
    if len(ws) < n:
        return [" ".join(ws)]
    return [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def covered(turn, blob):
    """True when at least 60% of the turn's 4-word shingles sit in blob."""
    sh = shingles(words(turn))
    if not sh:
        return False
    hay = " " + " ".join(words(blob)) + " "
    hit = sum(1 for g in sh if (" " + g + " ") in hay)
    return hit * 100 >= len(sh) * 60


def human_text(raw):
    """The text the reader is shown for one recall payload: exactly render() from render.py, nothing else.

    Scoring counts only what the reader sees (decided 21 Oct 2026). Every winner the service returned is
    rendered, not trimmed back to 8. Fields the reader is never shown (a hidden title, section, rule) do not count.
    """
    return render(raw if isinstance(raw, dict) else {})


_CHAR_KEYS = ("title", "summary", "snippet", "excerpt", "text", "section", "topic", "rule", "applies_when")


def _returned_text(raw):
    """Every string the service returned, for the chars_returned size statistic ONLY. Never used for scoring."""
    parts = []
    for w in (raw or {}).get("winners") or []:
        if not isinstance(w, dict):
            continue
        for key in _CHAR_KEYS:
            val = w.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
        for ln in w.get("lines") or []:
            if isinstance(ln, dict) and isinstance(ln.get("message"), str) and ln["message"]:
                parts.append(ln["message"])
    return "\n".join(parts)


def parse_evidence_ids(raw):
    """LoCoMo evidence tokens. Split on ';', ',', and whitespace. 'D:11:26' becomes 'D11:26'."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            out.extend(parse_evidence_ids(item))
        return out
    out = []
    for tok in re.split(r"[;,\s]+", str(raw).strip()):
        if not tok:
            continue
        folded = re.fullmatch(r"[Dd]:(\d+):(\d+)", tok)
        if folded:
            tok = "D%s:%s" % (folded.group(1), folded.group(2))
        out.append(tok)
    return out


def locomo_evidence(qa, dia):
    """('score', texts) or ('no_evidence', []). One unmapped id excludes the question."""
    ids = parse_evidence_ids(qa.get("evidence"))
    if qa.get("category") not in (1, 2, 3, 4) or not ids:
        return "no_evidence", []
    if any(e not in dia for e in ids):
        return "no_evidence", []
    return "score", [dia[e] for e in ids]


def evidence_of(bench, qid):
    """('score', [turn texts]) or ('abstention'| 'no_evidence', [])."""
    if bench == "locomo":
        ci, i = qid.split(":")
        item = locomo_data()[int(ci)]
        qa = item["qa"][int(i)]
        return locomo_evidence(qa, _dia_map(item))
    if str(qid).endswith("_abs"):
        return "abstention", []
    q = lme_index()[qid]
    texts = [t.get("content") or ""
             for sess in q["haystack_sessions"] for t in sess if t.get("has_answer") is True]
    if not texts:
        return "no_evidence", []
    return "score", texts


class StageVoid(RuntimeError):
    """Infra failure. The stage is not a score; the process exits non-zero."""


_DATA_SHELVES = {"msg", "kw"}


def _failed_names(raw):
    failed = raw.get("failed") or []
    if isinstance(failed, str):
        failed = [part for part in failed.split(",") if part]
    names = []
    for item in failed:
        name = str(item)
        if name.endswith("_stale"):
            name = name[:-6]
        if name:
            names.append(name)
    return names


def infra_reason(row):
    """Replica gate or served_by other than local. None when the row can be scored.

    A per-question HTTP error, timeout, or all-shelves-failed is not infra. Those stay
    in the denominator as not-found. The stage voids only when they exceed 5%.
    """
    raw = row.get("raw")
    if not isinstance(raw, dict):
        if row.get("error"):
            return None
        return "recall returned no JSON object"
    served = raw.get("served_by")
    # The engine's public answer omits served_by; `local` is also fine. Anything else came from a fallback service.
    if served not in (None, "local"):
        return "served_by=%s" % served
    if raw.get("reason"):
        return "reason=%s" % raw.get("reason")
    return None


def question_miss_reason(row):
    """HTTP error, timeout, or all-shelves-failed (http_recall's err, including HTTP 500).

    None when this row is not that kind of miss. The reason is stored on the question.
    """
    err = row.get("error")
    if err:
        return str(err)
    raw = row.get("raw")
    if isinstance(raw, dict) and raw.get("error"):
        return "error=%s" % raw.get("error")
    return None


def errors_exceed_rate(n_errors, n_questions, percent=5):
    """True when errors are strictly more than `percent` percent of the stage's questions."""
    if n_questions <= 0:
        return False
    return n_errors * 100 > n_questions * percent


def guard_question_misses(rows):
    """Void the stage when HTTP/timeout/all-shelves-failed exceeds 5% of its questions."""
    misses = [(r.get("id"), question_miss_reason(r)) for r in rows if question_miss_reason(r)]
    if errors_exceed_rate(len(misses), len(rows)):
        qid, reason = misses[0]
        raise StageVoid(
            "recall errors on %d/%d questions exceed 5%% (first %s: %s)"
            % (len(misses), len(rows), qid, reason))


def recall_failed(row):
    """True when this question is not-found because recall failed or came back degraded.

    A shelf that is empty by design does not count. served_by other than local, or a
    replica-gate reason, voids the stage. A per-question HTTP error, timeout, or
    all-shelves-failed is a miss (True) and does not void on its own.
    """
    reason = infra_reason(row)
    if reason:
        raise StageVoid(reason)
    if question_miss_reason(row):
        return True
    raw = row.get("raw") or {}
    if raw.get("degraded"):
        return True
    names = _failed_names(raw)
    if raw.get("served_by") == "local":
        return any(name in _DATA_SHELVES for name in names)
    return bool(names)


def chars_returned(row):
    """Characters of human-readable text the recall returned for one question."""
    raw = row.get("raw") if isinstance(row, dict) else None
    if not isinstance(raw, dict):
        return 0
    return len(_returned_text(raw))


def score_rows(bench, rows):
    found = failures = abstentions = no_evidence = recall_errors = 0
    chars_sum = 0
    no_evidence_ids = []
    detail = []
    for row in rows:
        # Replica gate voids even on a question that would otherwise be excluded.
        why = infra_reason(row)
        if why:
            raise StageVoid("question %s: %s" % (row.get("id"), why))
        nchars = chars_returned(row)
        chars_sum += nchars
        kind, texts = evidence_of(bench, row["id"])
        if kind == "abstention":
            abstentions += 1
            detail.append({"id": row["id"], "excluded": "abstention", "found": False,
                           "failure": False, "chars": nchars})
            continue
        if kind != "score":
            no_evidence += 1
            no_evidence_ids.append(row["id"])
            detail.append({"id": row["id"], "excluded": "no_evidence", "found": False,
                           "failure": False, "chars": nchars})
            continue
        miss = question_miss_reason(row)
        if miss:
            recall_errors += 1
            failures += 1
            detail.append({"id": row["id"], "excluded": None, "found": False, "failure": True,
                           "reason": miss, "evidence": len(texts), "chars": nchars})
            continue
        failed = recall_failed(row)
        if failed:
            failures += 1
            ok = False
        else:
            blob = human_text(row.get("raw"))
            ok = all(covered(t, blob) for t in texts)
        if ok:
            found += 1
        detail.append({"id": row["id"], "excluded": None, "found": ok, "failure": failed,
                       "evidence": len(texts), "chars": nchars})
    n = sum(1 for d in detail if d["excluded"] is None)
    mean_chars = (chars_sum / len(rows)) if rows else 0.0
    return {"n": n, "found": found, "failures": failures, "abstentions": abstentions,
            "no_evidence": no_evidence, "no_evidence_ids": no_evidence_ids,
            "recall_errors": recall_errors, "mean_chars": mean_chars,
            "recall": (found / n) if n else 0.0, "questions": detail}


def summary_line(fmt, bench, scored):
    pct = 100.0 * scored["recall"]
    ids = scored.get("no_evidence_ids") or []
    listed = (" no_evidence_ids=%s" % ",".join(str(i) for i in ids)) if ids else ""
    return ("recall %s/%s %.1f%% (n=%d) failures=%d recall_errors=%d no_evidence=%d%s mean_chars=%.1f"
            % (fmt, bench, pct, scored["n"], scored["failures"], scored.get("recall_errors", 0),
               scored.get("no_evidence", 0), listed, scored.get("mean_chars") or 0.0))


def _toml(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _adapter_id(fmt, bench, unit):
    safe = re.sub(r"[^a-z0-9_-]", "-", ("%s-%s-%s" % (fmt, bench, unit)).lower()).strip("-")
    if not safe or not safe[0].isalnum():
        safe = "u-" + safe
    return safe[:63]


def _port_free(port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(wanted):
    if _port_free(wanted):
        return wanted
    for port in range(8794, 8820):
        if _port_free(port):
            _say("port %d is taken by a live service; bench will use %d" % (wanted, port))
            return port
    raise SystemExit("no free localhost port for the bench service")


def named_embedder(alias):
    """qwen (2000), nomic (768), mxbai (1024). Alias is the --embedder flag."""
    spec = EMBEDDERS.get(alias)
    if spec is None:
        raise SystemExit("embedder must be qwen, nomic, or mxbai")
    return spec


def require_embedder_flag(fmt, alias):
    """nomic/mxbai are the embedder test on chats + plain. Format test stays on qwen."""
    if alias is None:
        return DEFAULT_EMBEDDER
    if fmt not in ("plain", "plainwin", "chats"):
        raise SystemExit("--embedder is for --format plain and chats")
    if alias not in EMBEDDERS:
        raise SystemExit("embedder must be qwen, nomic, or mxbai")
    return alias


def public_config(fmt, bench, unit, index_dir, port, notes_folder=None, chats_root=None,
                  embedder=DEFAULT_EMBEDDER, memories=None):
    spec = named_embedder(embedder) if fmt == "chats" else EMBED[fmt]
    if fmt == "chats" and chats_root:
        data = os.path.join(os.path.abspath(chats_root), bench, unit)
    else:
        data = convert.export_dir(fmt, bench, unit)
    if fmt == "pgvector":
        frag_path = os.path.join(data, "adapter.toml")
        if not os.path.isfile(frag_path):
            raise SystemExit("no pgvector adapter.toml for %s/%s. Run `python3 -m acbench convert` first" % (bench, unit))
    elif fmt == "chats" and not os.path.isfile(os.path.join(data, "conversations.json")):
        raise SystemExit("no chats export for %s/%s. Run `python3 -m acbench convert` first" % (bench, unit))
    elif fmt == "notes" and not os.path.isdir(data):
        raise SystemExit("no notes folder for %s/%s. Run `python3 -m acbench convert` first" % (bench, unit))
    lines = [
        "[postgres]",
        "bin_dir = %s" % _toml(settings.pg_bin_dir()),
        "host = %s" % _toml(settings.pg_host()),
        "port = %s" % settings.pg_port(),
        "user = %s" % _toml(settings.pg_user()),
        "database = %s" % _toml(settings.pg_db()),
        "",
        "[embedder]",
        'url = "http://127.0.0.1:%d/api/embed"' % CACHE_PORT,
        "model = %s" % _toml(spec["model"]),
        "dimensions = %d" % spec["dimensions"],
        "dims = %d" % spec["dimensions"],
        "send_dimensions = true",
        "timeout = 180",
        "batch_size = 8",
        "",
        "[vault]",
        'dir = ""',
        "",
        "[service]",
        "port = %d" % port,
        "extensions = []",
        "",
        "[shelves]",
        'import_root = ".."',
        "adapters = []",
        "",
        "[index]",
        "dir = %s" % _toml(index_dir),
        "refresh_minutes = 1440",
        "",
    ]
    if fmt == "chats":
        lines += [
            "[[adapters.chats]]",
            "id = %s" % _toml(_adapter_id(fmt, bench, unit)),
            "name = %s" % _toml("%s %s" % (bench, unit)),
            "path = %s" % _toml(data),
            'format = "claude"',
            "",
        ]
    elif fmt == "notes":
        lines += [
            "[[adapters.notes]]",
            "id = %s" % _toml(_adapter_id(fmt, bench, unit)),
            "name = %s" % _toml("%s %s" % (bench, unit)),
            "folder = %s" % _toml(data),
            "",
        ]
    else:
        with open(os.path.join(data, "adapter.toml"), encoding="utf-8") as f:
            lines.append(f.read().rstrip() + "\n")
    if notes_folder:
        raise SystemExit("memories enter through [[adapters.memories]], not a notes shelf")
    if memories:
        # A second shelf beside the chats: the writer's JSON files, read by Antechamber's memories adapter. Each
        # winning memory brings its source lines from the same export. redundant_with names the real chats id; the
        # engine declares it only (memories compete for the regular seats).
        kind, folder = memories
        lines += [
            "[[adapters.memories]]",
            "id = %s" % _toml(_adapter_id("memories", bench, unit)),
            "name = %s" % _toml(memory_shelf_name(kind)),
            "folder = %s" % _toml(os.path.abspath(folder)),
            "export = %s" % _toml(os.path.join(data, "conversations.json")),
            "redundant_with = [%s]" % _toml(_adapter_id(fmt, bench, unit)),
            "",
            # Memories ride in MEMORY_EXTRA_SEATS reserved seats ON TOP of the regular 8, so they cannot push
            # the original lines out. The engine's own [recall] switch; conversation-only is unchanged. Recall-only
            # runs never write this block.
            "[recall]",
            "extra_seats = %d" % MEMORY_EXTRA_SEATS,
            "extra_seat_shelves = [%s]" % _toml(_adapter_id("memories", bench, unit)),
            "",
        ]
    return "\n".join(lines) + "\n"


def memory_shelf_name(kind):
    """The shelf name winners carry as their source, so memory winners can be counted."""
    return "acbench-memories-%s" % kind


def _rewrite_embedder_url(text, url):
    """Replace only the [embedder] url line. Other sections keep their own url keys."""
    section = None
    out = []
    found = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            section = stripped.strip("[]").strip()
        key = ""
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
        if section == "embedder" and key == "url":
            line = 'url = "%s"\n' % url
            found = True
        out.append(line)
    if not found:
        raise SystemExit("config has no [embedder] url")
    return "".join(out)


def scratch_env(config_path):
    """PATH, HOME, and ANTECHAMBER_CONFIG. The caller shell is not copied."""
    if not config_path or not os.path.isfile(config_path):
        raise StageVoid("engine config is missing: %s" % config_path)
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "ANTECHAMBER_CONFIG": os.path.abspath(config_path),
    }


def spawn_fresh(argv, config_path, cwd, capture=False):
    kw = {"cwd": cwd, "env": scratch_env(config_path)}
    if capture:
        kw["capture_output"] = True
        kw["text"] = True
        r = subprocess.run(argv, **kw)
        if r.returncode:
            raise subprocess.CalledProcessError(r.returncode, argv, output=r.stdout, stderr=r.stderr)
        return r
    subprocess.check_call(argv, **kw)
    return None


def public_env(config_path, fmt):
    # The engine turns each switch on only when the variable equals "on".
    # WIDE may return 16 winners for a list question; that response is what we score.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "ANTECHAMBER_CONFIG": config_path,
        "ANTECHAMBER_WIDE": "on",
        "ANTECHAMBER_DATES": "on",
        "ANTECHAMBER_CHRONO": "on",
        # Contract switch: a benchmark never feeds its own use back into ranking. An engine without use-door
        # code ignores it, so it is recorded, not enforced.
        "MEMORY_ENTRY_USE_DOOR": "off",
    }
    if fmt == "pgvector":
        env[convert.DSN_ENV] = settings.pg_dsn()
        if os.environ.get("PGPASSWORD"):
            env["PGPASSWORD"] = os.environ["PGPASSWORD"]
    if _VARIANT:
        env.update(VARIANTS[_VARIANT])
    return env


def _tail(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4000))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def start_proc(script, env, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, script], env=env, cwd=os.path.dirname(script),
        stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, logf


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def wait_health(port, proc, log_path, timeout=3600):
    url = "http://127.0.0.1:%d/health" % port
    deadline = time.time() + timeout
    last = ""
    reported = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("recall service exited %s\n%s" % (proc.returncode, _tail(log_path)))
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = json.loads(resp.read().decode())
            if body.get("ok") is True:
                return body
            last = json.dumps(body)[:300]
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
        if time.time() - reported > 30:
            _say("  waiting for /health: %s" % last[:200])
            reported = time.time()
        time.sleep(1)
    raise RuntimeError("health was not ok within %ds (%s)\n%s" % (timeout, last, _tail(log_path)))


def http_recall(port, question, timeout=180):
    url = "http://127.0.0.1:%d/antechamber/recall?%s" % (
        port, urllib.parse.urlencode({"q": question, "limit": str(VARIANT_LIMIT.get(_VARIANT, RECALL_LIMIT))}))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode("utf-8", "replace")
        return None, "http %s: %s" % (e.code, detail)
    except Exception as e:
        kind = "timeout" if isinstance(e, TimeoutError) or "timed out" in str(e).lower() else type(e).__name__
        return None, "%s: %s" % (kind, e)[:300]


def _token_eq(a, b):
    a, b = str(a or ""), str(b or "")
    if not a or len(a) != len(b):
        return False
    return hmac.compare_digest(a, b)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _pid_is_cache(pid):
    if not _pid_alive(pid):
        return False
    r = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return "embed_cache.py" in (r.stdout or "")


def _read_small(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def cache_identity():
    """True only when port 11435 is the cache this tree started (pid file + health token)."""
    pid = _read_small(CACHE_PID)
    token = _read_small(CACHE_TOKEN)
    if not pid.isdigit() or not token or not _pid_is_cache(pid):
        return False
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % CACHE_PORT, timeout=1) as resp:
            body = json.loads(resp.read().decode())
    except Exception:
        return False
    return (body.get("ok") is True and str(body.get("pid")) == pid
            and _token_eq(body.get("token"), token))


def _drop_cache_files(pid=None):
    current = _read_small(CACHE_PID)
    if pid is not None and current != str(pid):
        return
    for path in (CACHE_PID, CACHE_TOKEN):
        try:
            os.remove(path)
        except OSError:
            pass


def _port_accepts(port):
    """True when a process is accepting connections. A socket in TIME_WAIT is not."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def ensure_cache():
    """Start our cache, or keep the one we already started. Anything else on 11435 is refused."""
    if _port_accepts(CACHE_PORT):
        if cache_identity():
            return None, None
        raise StageVoid(
            "port %d is in use and it is not this tree's embed cache "
            "(pid file + health token). Refusing to use it." % CACHE_PORT)
    os.makedirs(os.path.dirname(CACHE_PID), exist_ok=True)
    token = secrets.token_hex(32)
    with open(CACHE_TOKEN, "w", encoding="utf-8") as f:
        f.write(token)
    log_path = os.path.join(RUN, "logs", "embed-cache.log")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "EMBED_CACHE_TOKEN": token,
        "EMBED_CACHE_PID_FILE": CACHE_PID,
        "ACBENCH_CACHE": settings.cache_dir(),
        "OLLAMA_URL": settings.ollama_url(),
    }
    proc, logf = start_proc(os.path.join(PACKAGE, "embed_cache.py"), env, log_path)
    for _ in range(50):
        if cache_identity() and _read_small(CACHE_PID) == str(proc.pid):
            return proc, logf
        if proc.poll() is not None:
            logf.close()
            _drop_cache_files()
            raise StageVoid("embed cache exited\n%s" % _tail(log_path))
        time.sleep(0.1)
    stop_proc(proc)
    logf.close()
    _drop_cache_files(proc.pid)
    raise StageVoid("embed cache did not come up\n%s" % _tail(log_path))


def index_content_sha(index_dir):
    """(content, order) fingerprints of an index, independent of SQLite's byte layout (v4.2.0: the byte sha differs
    between identical builds). content = every passage row incl. its vector, every (term, pid), files without mtime,
    meta — all sorted; order = passage ids in insertion (rowid) order, to see whether builds are also order-identical."""
    if not os.path.isdir(index_dir):
        return None, None
    hc, ho = hashlib.sha256(), hashlib.sha256()
    for root, dirs, files in os.walk(index_dir):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            if not name.endswith(".sqlite"):
                continue
            conn = sqlite3.connect(os.path.join(root, name))
            try:
                hc.update(name.encode() + b"\0")
                for table, sql in (("passages", "SELECT * FROM passages ORDER BY id"),
                                   ("terms", "SELECT term, pid FROM terms ORDER BY term, pid"),
                                   ("files", "SELECT path, size, sha256, embedded, dups FROM files ORDER BY path"),
                                   ("meta", "SELECT key, value FROM meta ORDER BY key")):
                    hc.update(table.encode())
                    for row in conn.execute(sql):
                        hc.update(repr(row).encode())
                ho.update(name.encode() + b"\0")
                for (pid,) in conn.execute("SELECT id FROM passages ORDER BY rowid"):
                    ho.update(pid.encode() + b"\n")
            finally:
                conn.close()
    return hc.hexdigest(), ho.hexdigest()


def index_sha(index_dir):
    if not os.path.isdir(index_dir):
        return None
    paths = []
    for root, dirs, files in os.walk(index_dir):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            if name.endswith(".sqlite"):
                paths.append(os.path.join(root, name))
    if not paths:
        return None
    for path in paths:
        try:
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except sqlite3.Error:
            pass
    h = hashlib.sha256()
    for path in paths:
        rel = os.path.relpath(path, index_dir).encode()
        hh = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                hh.update(block)
        h.update(rel + b"\0" + hh.hexdigest().encode() + b"\n")
    return h.hexdigest()


_FRESH = set()   # files this process wrote since the manifest was last written; only these may be re-hashed


def _write(path, obj):
    _FRESH.add(os.path.abspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest():
    """Add this stage's new artifacts to MANIFEST.sha256. An entry already listed is VERIFIED and kept, never
    re-hashed: a recall file edited after it was listed makes this refuse instead of re-signing the edit. Only a
    file this process just wrote (`_write`) may change its hash, and a listed file may not disappear."""
    man = os.path.join(RUN, "MANIFEST.sha256")
    listed = {}
    if os.path.isfile(man):
        with open(man, encoding="utf-8") as f:
            for line in f:
                if "  " in line:
                    h, name = line.rstrip("\n").split("  ", 1)
                    listed[name] = h
    fresh = {os.path.relpath(p, RUN) for p in _FRESH if p.startswith(os.path.abspath(RUN) + os.sep)}
    rows = {}
    for dirpath, dirs, files in os.walk(RUN):
        dirs[:] = sorted(d for d in dirs if d != "logs")
        for name in sorted(files):
            if name == "MANIFEST.sha256" or name.endswith(".log"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, RUN)
            now = _sha_file(path)
            if rel in listed and rel not in fresh and listed[rel] != now:
                raise SystemExit("REFUSED manifest update: %s changed since MANIFEST.sha256 listed it" % rel)
            rows[rel] = now
    for rel in listed:
        if rel not in rows:
            raise SystemExit("REFUSED manifest update: %s is listed in MANIFEST.sha256 but is gone" % rel)
    _FRESH.clear()
    with open(man, "w", encoding="utf-8") as f:
        f.writelines("%s  %s\n" % (h, rel) for rel, h in sorted(rows.items()))


def out_path(stage, fmt, bench, embedder=None, which="full", limit=None):
    """plain and chats nest the embedder. Every path nests the set and any --limit.

    recall/<fmt>/<embedder>/<set>[-limitN]/<bench>.json
    A prelim run cannot overwrite a full run, and --limit 2 cannot overwrite --limit 10.
    """
    segment = set_segment(which, limit)
    if fmt in ("plain", "plainwin", "chats"):
        return os.path.join(RUN, stage, fmt, embedder or DEFAULT_EMBEDDER, segment, "%s.json" % bench)
    return os.path.join(RUN, stage, fmt, segment, "%s.json" % bench)


def _tick(asked):
    """Re-check the lock every 25 finished questions, before the next one."""
    if asked and asked % 25 == 0:
        _require()


def _record(row, raw, err, index_sha256):
    rec = {"id": row["id"], "unit": row["unit"], "question": row["question"],
           "raw": raw, "error": err, "index_sha256": index_sha256}
    rec["chars"] = chars_returned(rec)
    return rec


def ask(port, rows, asked, index_sha256, stage_total=None, misses=None):
    """Ask each question. HTTP/timeout/all-shelves-failed is a miss.

    The stage voids as soon as those misses exceed 5% of `stage_total`
    (the whole stage, not this batch). Replica-gate responses void at once.
    """
    out = []
    if misses is None:
        misses = []
    total = len(rows) if stage_total is None else stage_total
    for row in rows:
        _tick(asked)
        raw, err = http_recall(port, row["question"])
        rec = _record(row, raw, err, index_sha256)
        try:
            recall_failed(rec)
        except StageVoid as e:
            raise StageVoid("question %s: %s" % (row["id"], e)) from e
        out.append(rec)
        asked += 1
        reason = question_miss_reason(rec)
        if reason:
            misses.append((row["id"], reason))
            _say("  %s not-found (%s)" % (row["id"], reason))
            if errors_exceed_rate(len(misses), total):
                raise StageVoid(
                    "recall errors on %d/%d questions exceed 5%% (first %s: %s)"
                    % (len(misses), total, misses[0][0], misses[0][1]))
        else:
            _say("  %s ok" % row["id"])
    return out, asked


def _clear_indexes(*scope):
    """Clear only this run's own index subfolder (fmt[/embedder]/bench/set[-limitN]). Indexes of other benchmarks, sets and limits share the run
    dir and are listed in MANIFEST.sha256, so they must survive: a listed file may not disappear."""
    index_root = os.path.join(RUN, "index", *scope)
    if os.path.isdir(index_root):
        shutil.rmtree(index_root)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _l2(a):
    return math.sqrt(_dot(a, a))


def chat_export_file(bench, unit, chats_root=None):
    root = chats_root or os.path.join(settings.data_dir(), "chats")
    return os.path.join(root, bench, unit, "conversations.json")


def load_chat_messages(path):
    """Every message of the chat export, in file order, whole text unchanged.

    Plain does not cap lines and does not cut a message into parts.
    """
    if not os.path.isfile(path):
        raise StageVoid("no chats export at %s" % path)
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    if not isinstance(convs, list):
        raise StageVoid("chat export is not a list: %s" % path)
    msgs = []
    for conv in convs:
        if not isinstance(conv, dict):
            continue
        for m in conv.get("chat_messages") or []:
            if not isinstance(m, dict):
                continue
            text = m.get("text")
            if not isinstance(text, str) or text == "":
                continue
            msgs.append({"id": str(m.get("uuid") or ""), "sender": str(m.get("sender") or ""), "text": text,
                         "thread": str(conv.get("uuid") or conv.get("name") or "")})
    return msgs


def embed_cached(texts, model, dims):
    """Vectors from the embed cache. Key includes the model's full digest (embed_cache.py)."""
    if not texts:
        return []
    out = []
    step = 8
    for i in range(0, len(texts), step):
        chunk = texts[i:i + step]
        payload = {"model": model, "input": list(chunk), "dimensions": dims}
        req = urllib.request.Request(
            CACHE_EMBED_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            raise StageVoid("embed cache failed: %s" % e) from e
        vecs = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vecs, list) or len(vecs) != len(chunk):
            raise StageVoid("embed cache returned the wrong number of vectors")
        for v in vecs:
            if not isinstance(v, list) or len(v) != dims:
                raise StageVoid("embed cache returned %s dims, want %s" % (
                    len(v) if isinstance(v, list) else type(v).__name__, dims))
        out.extend(vecs)
        _say("  embedded %d/%d" % (len(out), len(texts)))
    return out




_TOO_LONG = "exceeds the context length"


def _embedder_refuses_as_too_long(text, model, dims):
    """Ask Ollama directly (no cache) whether THIS text is refused for length. A message the standard embedder
    cannot index is plain search's real loss: counted, never cut to fit, never a crash."""
    payload = {"model": model, "input": [text], "dimensions": dims}
    # Ollama itself, used ONLY to read a refusal reason
    req = urllib.request.Request(settings.ollama_embed_url(), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
        return False
    except urllib.error.HTTPError as e:
        return _TOO_LONG in (e.read() or b"").decode("utf-8", "replace")
    except Exception:
        return False


def embed_plain(texts, model, dims):
    """embed_cached, except a text the embedder refuses AS TOO LONG becomes None (unsearchable). Any other failure
    still voids the stage."""
    out = []
    step = 8
    for i in range(0, len(texts), step):
        chunk = texts[i:i + step]
        try:
            out.extend(embed_cached(chunk, model, dims))
            continue
        except StageVoid:
            pass
        for t in chunk:
            try:
                out.extend(embed_cached([t], model, dims))
            except StageVoid:
                if _embedder_refuses_as_too_long(t, model, dims):
                    out.append(None)
                else:
                    raise
    return out


def rank_plain_window(msgs, vecs, qv, limit=RECALL_LIMIT, before=2, after=3):
    """DIAGNOSTIC 'plainwin' (PLAN r2 item 4), never the baseline: rank exactly like rank_plain, then each winner
    brings the window the engine's chats adapter brings (2 before, 3 after, same conversation, no line twice)."""
    winners = rank_plain(msgs, vecs, qv, limit)
    pos = {m["id"]: i for i, m in enumerate(msgs)}
    shown, out = set(), []
    for w in winners:
        hit = w["lines"][0]["id"]
        if hit in shown:
            continue
        i = pos[hit]; t = msgs[i].get("thread")
        lo = hi = i
        while lo - 1 >= 0 and i - (lo - 1) <= before and msgs[lo - 1].get("thread") == t:
            lo -= 1
        while hi + 1 < len(msgs) and (hi + 1) - i <= after and msgs[hi + 1].get("thread") == t:
            hi += 1
        lines = [{"id": m["id"], "sender": m["sender"], "message": m["text"], "hit": m["id"] == hit}
                 for m in msgs[lo:hi + 1] if m["id"] not in shown]
        shown.update(l["id"] for l in lines)
        out.append({"lines": lines, "score": w["score"]})
    return out


def rank_plain(msgs, vecs, qv, limit=RECALL_LIMIT):
    """Top `limit` whole messages by cosine. Ties keep export order.

    Plain search is the standard baseline. Each message is embedded and returned
    as-is: no line cap and no split into parts. Antechamber's own 600/1200 caps
    stay inside its engine. No keyword list, no neighbour window, no date field,
    no widening past `limit`, and no chronological reordering.
    """
    if len(msgs) != len(vecs):
        raise StageVoid("plain embed count does not match the export")
    some = next((v for v in vecs if v is not None), None)
    if some is not None and len(qv) != len(some):
        raise StageVoid("plain vector length mismatch")
    qn = _l2(qv)
    scored = []
    for i, vec in enumerate(vecs):
        if vec is None:          # refused by the embedder as too long: unsearchable (standard search's real loss)
            continue
        if len(vec) != len(qv):
            raise StageVoid("plain vector length mismatch")
        denom = _l2(vec) * qn
        sim = (_dot(vec, qv) / denom) if denom else 0.0
        if not math.isfinite(sim):
            sim = 0.0
        scored.append((sim, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    winners = []
    for sim, i in scored[:limit]:
        msg = msgs[i]
        winners.append({
            "lines": [{"id": msg["id"], "sender": msg["sender"], "message": msg["text"]}],
            "score": sim,
        })
    return winners


def run_plain_recall(bench, which, limit, embedder=DEFAULT_EMBEDDER, window=False):
    """Same chat export, same embedder as the chats pair, top 8 by cosine."""
    _lock_stage("plain", embedder, bench)
    _require()
    fmt_name = "plainwin" if window else "plain"
    _clear_indexes(fmt_name, embedder, bench, set_segment(which, limit))
    spec = named_embedder(embedder)
    model, dims = spec["model"], spec["dimensions"]
    rows = select(bench, which, limit)
    cache_proc = cache_log = None
    got = []
    asked = 0
    unindexable = 0
    try:
        cache_proc, cache_log = ensure_cache()
        for unit, qs in group(rows):
            path = chat_export_file(bench, unit)
            msgs = load_chat_messages(path)
            if not msgs:
                raise StageVoid("plain: chat export has no messages for %s/%s" % (bench, unit))
            _say("plain %s/%s/%s unit %s (%d messages, %d questions)" % (
                embedder, bench, which, unit, len(msgs), len(qs)))
            vecs = embed_plain([m["text"] for m in msgs], model, dims)
            unindexable += sum(1 for v in vecs if v is None)
            export_sha = _sha_file(path)
            for row in qs:
                _tick(asked)
                qv = embed_cached([row["question"]], model, dims)[0]
                ranker = rank_plain_window if window else rank_plain
                raw = {"winners": ranker(msgs, vecs, qv, RECALL_LIMIT), "failed": []}
                rec = _record(row, raw, None, export_sha)
                try:
                    recall_failed(rec)
                except StageVoid as e:
                    raise StageVoid("question %s: %s" % (row["id"], e)) from e
                got.append(rec)
                asked += 1
                _say("  %s ok" % row["id"])
    finally:
        if cache_proc is not None:
            stop_proc(cache_proc)
            _drop_cache_files(cache_proc.pid)
        if cache_log is not None:
            cache_log.close()
    _require()
    guard_question_misses(got)
    doc = {"version": VERSION, "stage": "recall", "format": fmt_name, "bench": bench, "set": which,
           "limit": RECALL_LIMIT, "embedder": model, "embedder_flag": embedder,
           "dimensions": dims, "unindexable_messages": unindexable, "questions": got}
    _write(out_path("recall", fmt_name, bench, embedder, which, limit), doc)
    write_manifest()
    scored = score_rows(bench, got)
    print(summary_line(fmt_name, bench, scored) + " unindexable_messages=%d" % unindexable)
    return doc


def preflight_pgvector_dims(psql=None):
    """Abort before any question unless every embedding in the pgvector database is 2000-d (qwen).

    An empty table has no vector to be the wrong size. A table with any other
    dimension, or a table we cannot measure, voids the stage.
    """
    run_sql = psql or (lambda sql: convert._psql(settings.pg_db(), sql))
    try:
        probe = run_sql("SELECT tablename FROM pg_tables WHERE schemaname = 'acbench' ORDER BY 1")
    except Exception as e:
        raise StageVoid("pgvector preflight: cannot read the pgvector database (%s)" % e) from e
    names = [t for t in (probe or "").split() if t]
    if not names:
        raise StageVoid("pgvector preflight: the pgvector database has no tables")
    bad = []
    for name in names:
        if not name.replace("_", "").isalnum() or not name[0].isalpha():
            raise StageVoid("pgvector preflight: unsafe table name %s" % name)
        sql = (
            "SELECT COUNT(*), COALESCE(MIN(vector_dims(embedding)), -1), "
            "COALESCE(MAX(vector_dims(embedding)), -1) FROM acbench.\"%s\"" % name
        )
        try:
            row = (run_sql(sql) or "").strip()
        except Exception as e:
            raise StageVoid("pgvector preflight: %s (%s)" % (name, e)) from e
        parts = [p.strip() for p in row.split("|")]
        if len(parts) != 3:
            raise StageVoid("pgvector preflight: bad dims row for %s: %r" % (name, row))
        try:
            n, dmin, dmax = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as e:
            raise StageVoid("pgvector preflight: bad dims row for %s: %r" % (name, row)) from e
        if n == 0:
            continue
        if dmin != 2000 or dmax != 2000:
            bad.append("%s dims %s..%s (n=%d)" % (name, dmin, dmax, n))
    if bad:
        raise StageVoid(
            "pgvector preflight: every embedding must be 2000 (qwen); %s" % "; ".join(bad))


def run_recall(fmt, bench, which, limit, embedder=DEFAULT_EMBEDDER):
    global _VARIANT
    _VARIANT = split_variant(which)[1]
    if _VARIANT and fmt != "chats":
        raise SystemExit("REFUSED: variants run on chats only")
    _lock_stage(fmt, embedder, bench)
    if fmt in ("plain", "plainwin"):
        return run_plain_recall(bench, which, limit, embedder, window=(fmt == "plainwin"))
    _require()
    if fmt == "pgvector":
        preflight_pgvector_dims()
    segment = set_segment(which, limit)
    if fmt == "chats":
        _clear_indexes(fmt, embedder, bench, segment)
    else:
        _clear_indexes(fmt, bench, segment)
    spec = named_embedder(embedder) if fmt == "chats" else EMBED.get(fmt) or named_embedder(DEFAULT_EMBEDDER)
    model, dims = spec["model"], spec["dimensions"]
    rows = select(bench, which, limit)
    script = settings.engine_script()
    port = pick_port(settings.first_port())
    cache_proc = cache_log = None
    got = []
    asked = 0
    misses = []
    try:
        cache_proc, cache_log = ensure_cache()
        for unit, qs in group(rows):
            _say("%s %s/%s/%s unit %s (%d questions)" % (
                fmt, embedder if fmt == "chats" else "qwen", bench, which, unit, len(qs)))
            if fmt == "chats":
                index_dir = os.path.join(RUN, "index", fmt, embedder, bench, segment, unit)
                cfg = os.path.join(RUN, "config", fmt, embedder, bench, segment, "%s.toml" % unit)
            else:
                index_dir = os.path.join(RUN, "index", fmt, bench, segment, unit)
                cfg = os.path.join(RUN, "config", fmt, bench, segment, "%s.toml" % unit)
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            with open(cfg, "w", encoding="utf-8") as f:
                f.write(public_config(fmt, bench, unit, index_dir, port, embedder=embedder))
            log_path = os.path.join(RUN, "logs", "%s-%s-%s-%s.log" % (fmt, embedder, bench, unit))
            proc, logf = start_proc(script, public_env(cfg, fmt), log_path)
            try:
                wait_health(port, proc, log_path)
                part, asked = ask(port, qs, asked, None, len(rows), misses)
            finally:
                stop_proc(proc)
                logf.close()
            sha = index_sha(index_dir)
            csha, osha = index_content_sha(index_dir)
            for rec in part:
                rec["index_sha256"] = sha
                rec["index_content_sha256"], rec["index_order_sha256"] = csha, osha
            got.extend(part)
    finally:
        if cache_proc is not None:
            stop_proc(cache_proc)
            _drop_cache_files(cache_proc.pid)
        if cache_log is not None:
            cache_log.close()
    _require()
    guard_question_misses(got)
    flag = embedder if fmt in ("plain", "plainwin", "chats") else DEFAULT_EMBEDDER
    doc = {"version": VERSION, "stage": "recall", "format": fmt, "bench": bench, "set": which,
           "limit": VARIANT_LIMIT.get(_VARIANT, RECALL_LIMIT), "embedder": model, "embedder_flag": flag,
           "dimensions": dims, "engine_commit": lock.engine_commit(),
           "variant": {"name": _VARIANT, "env": VARIANTS.get(_VARIANT)}, "questions": got}
    _write(out_path("recall", fmt, bench, flag if fmt in ("plain", "plainwin", "chats") else None, which, limit), doc)
    write_manifest()
    scored = score_rows(bench, got)
    print(summary_line(fmt, bench, scored))
    return doc


def _recall_path(fmt, bench, embedder=None, which="full", limit=None):
    flag = embedder if fmt in ("plain", "plainwin", "chats") else None
    return out_path("recall", fmt, bench, flag, which, limit)


def _expected_embedder(fmt, embedder):
    """(model, dimensions) a recall file of this format must record."""
    spec = named_embedder(embedder or DEFAULT_EMBEDDER) if fmt in ("plain", "plainwin", "chats") \
        else EMBED.get(fmt) or named_embedder(DEFAULT_EMBEDDER)
    return spec["model"], spec["dimensions"]


def _manifest_refusal(path):
    """None when MANIFEST.sha256 beside the run lists this file with its current hash. Else the reason.
    Looks upward from the file for the run folder that holds the manifest (the --run-dir or ACBENCH_RUNS/<version>)."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        man = os.path.join(d, "MANIFEST.sha256")
        if os.path.isfile(man):
            break
        parent = os.path.dirname(d)
        if parent == d:
            return "no MANIFEST.sha256 covers %s" % path
        d = parent
    rel = os.path.relpath(os.path.abspath(path), d)
    listed = {}
    with open(man, encoding="utf-8") as f:
        for line in f:
            if "  " in line:
                h, name = line.rstrip("\n").split("  ", 1)
                listed[name] = h
    if rel not in listed:
        return "%s is not in %s" % (rel, man)
    if listed[rel] != _sha_file(path):
        return "%s changed since MANIFEST.sha256 was written" % rel
    return None


def _recall_doc_refusal(doc, path, fmt, bench, which, embedder=None):
    """None when a saved recall file is this version's, this stage's, from this embedder, and unedited. Else why not."""
    if not isinstance(doc, dict) or not isinstance(doc.get("questions"), list):
        return "not a recall file"
    if doc.get("version") != VERSION:
        return "version %s, this harness is %s" % (doc.get("version"), VERSION)
    if doc.get("stage") != "recall":
        return "stage is %s, not recall" % doc.get("stage")
    if doc.get("format") != fmt or doc.get("bench") != bench or doc.get("set") != which:
        return "it is %s %s %s, not %s %s %s" % (doc.get("format"), doc.get("bench"), doc.get("set"), fmt, bench, which)
    model, dims = _expected_embedder(fmt, embedder)
    if doc.get("embedder") != model or doc.get("dimensions") != dims:
        return "embedder %s/%s, expected %s/%s" % (doc.get("embedder"), doc.get("dimensions"), model, dims)
    return _manifest_refusal(path)


def _load_recall(fmt, bench, which, limit, embedder=None):
    path = _recall_path(fmt, bench, embedder, which, limit)
    if not os.path.isfile(path):
        raise SystemExit("no recall file at %s — run recall first" % path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    why = _recall_doc_refusal(doc, path, fmt, bench, which, embedder)
    if why:
        raise SystemExit("REFUSED recall file %s: %s" % (path, why))
    want = [r["id"] for r in select(bench, which, limit)]
    by = {q["id"]: q for q in doc["questions"]}
    missing = [i for i in want if i not in by]
    if missing:
        raise SystemExit("recall file is missing %d questions (first %s)" % (len(missing), missing[0]))
    return doc, [by[i] for i in want]


def read_recall_doc(fmt, bench, embedder=None, which="full", limit=None):
    """Load a recall file as written. --split uses the full set, so 60 vs the rest come from one file."""
    path = _recall_path(fmt, bench, embedder, which, limit)
    if not os.path.isfile(path):
        raise SystemExit("no recall file at %s — run recall first" % path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    why = _recall_doc_refusal(doc, path, fmt, bench, which, embedder)
    if why:
        raise SystemExit("REFUSED recall file %s: %s" % (path, why))
    return doc


def _write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = __import__("tempfile").mkstemp(prefix=".writing-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def recall_found(fmt, bench, which, embedder):
    """Found/not per question — only when the run covers EVERY selected question (two identically partial
    runs must not pass). Returns None (= not run / incomplete) otherwise."""
    path = _recall_path(fmt, bench, embedder, which, None)
    doc = _read_json(path)
    if not doc or doc.get("version") != VERSION:
        return None
    why = _recall_doc_refusal(doc, path, fmt, bench, which, embedder)
    if why:
        raise SystemExit("REFUSED recall file %s: %s" % (path, why))
    expected = {r["id"] for r in select(bench, which, None)}
    if {q["id"] for q in doc["questions"]} != expected:
        return None
    return {q["id"]: q["found"] for q in score_rows(bench, doc["questions"])["questions"] if q["excluded"] is None}


def run_recall_verdict(bench, which):
    """Claims C1-C3 from this version's saved recall files. C1/C2: Antechamber (chats) vs standard search
    on the SAME embedder: gain >= 3 pts AND exact McNemar p < 0.05 AND the 95% paired interval (LoCoMo clustered by
    conversation) above 0. C3: notes and vector DB vs chats (qwen): the 95% paired interval inside +-3 points."""
    out = {"version": VERSION, "bench": bench, "set": which, "C1_C2": {}, "C3": {}}
    # The verdict is only valid for the locked setup: same lock check as a recall stage, over every embedder
    # whose recall files it reads and the converted data those runs used.
    embs = tuple(e for e in ("qwen", "nomic", "mxbai")
                 if os.path.isfile(_recall_path("chats", bench, e, which)) or os.path.isfile(_recall_path("plain", bench, e, which)))
    # The verdict also consumes the chats (engine), notes (engine) and pgvector (engine + Postgres) results, so it
    # takes the widest lock scope of the files it reads, and the converted data behind each.
    present = [f for f in ("chats", "notes", "pgvector")
               if os.path.isfile(_recall_path(f, bench, "qwen" if f == "chats" else None, which))]
    scope = "pgvector" if "pgvector" in present else "engine" if present else "plain"
    derived = tuple(sorted(set(lock.derived_for("plain", bench)) | {(f, bench) for f in present if f != "chats"}))
    _STAGE["scope"], _STAGE["derived"] = scope, derived
    _STAGE["embedders"] = embs or (DEFAULT_EMBEDDER,)
    _require()
    for emb in ("qwen", "nomic", "mxbai"):
        a, p = recall_found("chats", bench, which, emb), recall_found("plain", bench, which, emb)
        if not a or not p:
            out["C1_C2"][emb] = "not run"
            continue
        if set(a) != set(p):
            out["C1_C2"][emb] = "INCOMPLETE: the two runs do not cover the same questions (%d vs %d)" % (len(a), len(p))
            continue
        ids = sorted(a)
        b = sum(1 for i in ids if a[i] and not p[i]); c = sum(1 for i in ids if p[i] and not a[i])
        gain = 100.0 * (sum(a[i] for i in ids) - sum(p[i] for i in ids)) / len(ids)
        lo, hi = paired_ci({i: int(a[i]) - int(p[i]) for i in ids}, bench)
        out["C1_C2"][emb] = {"n": len(ids), "gain": round(gain, 1), "p": mcnemar_exact(b, c),
                             "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                             "counts": gain >= 3 and mcnemar_exact(b, c) < 0.05 and lo > 0}
    base = recall_found("chats", bench, which, "qwen")
    for fmt in ("notes", "pgvector"):
        f = recall_found(fmt, bench, which, None)
        if not f or not base:
            out["C3"][fmt] = "not run"
            continue
        if set(f) != set(base):
            out["C3"][fmt] = "INCOMPLETE: %d vs %d questions" % (len(f), len(base))
            continue
        ids = sorted(f)
        lo, hi = paired_ci({i: int(f[i]) - int(base[i]) for i in ids}, bench)
        out["C3"][fmt] = {"n": len(ids), "diff": round(100.0 * sum(int(f[i]) - int(base[i]) for i in ids) / len(ids), 1),
                          "ci95": [round(100 * lo, 1), round(100 * hi, 1)], "same": lo > -0.03 and hi < 0.03}
    _require()
    vpath = os.path.join(RUN, "score", "recall-verdict", which, "%s.json" % bench)
    _write_atomic(vpath, out)
    _FRESH.add(os.path.abspath(vpath))
    write_manifest()
    print(json.dumps(out, indent=1))


def _score_fields(scored):
    return {"n": scored["n"], "found": scored["found"], "recall": scored["recall"],
            "failures": scored["failures"], "abstentions": scored["abstentions"],
            "no_evidence": scored["no_evidence"],
            "no_evidence_ids": scored.get("no_evidence_ids") or [],
            "recall_errors": scored.get("recall_errors", 0),
            "mean_chars": scored.get("mean_chars", 0.0)}


def lme_prelim_ids():
    return {p["question_id"] for p in prelim()["lme"]}


def tuned_ids(bench):
    """The questions looked at while Antechamber was tuned: 60 LongMemEval ids, 39 LoCoMo "conv:index" ids."""
    if bench == "lme":
        return lme_prelim_ids()
    return {"%d:%d" % (p["conv"], p["qa_index"]) for p in prelim()["locomo"]}


def partition_lme(rows, prelim_ids=None, bench="lme"):
    ids = prelim_ids if prelim_ids is not None else tuned_ids(bench)
    pre, rest = [], []
    for row in rows:
        (pre if row["id"] in ids else rest).append(row)
    return pre, rest


def split_pair(chats_rows, plain_rows, prelim_ids=None, bench="lme"):
    """Prelim ids vs the rest, from the same result files, plus chats-minus-plain lead."""
    ids = prelim_ids if prelim_ids is not None else tuned_ids(bench)
    c_pre, c_rest = partition_lme(chats_rows, ids, bench)
    p_pre, p_rest = partition_lme(plain_rows, ids, bench)
    chats_pre = score_rows(bench, c_pre)
    chats_rest = score_rows(bench, c_rest)
    plain_pre = score_rows(bench, p_pre)
    plain_rest = score_rows(bench, p_rest)
    return {
        "prelim_ids": len(ids),
        "prelim": {
            "plain": _score_fields(plain_pre),
            "chats": _score_fields(chats_pre),
            "lead": chats_pre["recall"] - plain_pre["recall"],
            "ids": [r["id"] for r in c_pre],
        },
        "rest": {
            "plain": _score_fields(plain_rest),
            "chats": _score_fields(chats_rest),
            "lead": chats_rest["recall"] - plain_rest["recall"],
            "ids": [r["id"] for r in c_rest],
        },
    }


def _split_line(pile, name, n_ids, bench="lme"):
    plain, chats = pile["plain"], pile["chats"]
    return "split %s %s (%d ids, plain n=%d chats n=%d) plain %.1f%% chats %.1f%% lead %+.1f" % (
        bench, name, n_ids, plain["n"], chats["n"],
        100.0 * plain["recall"], 100.0 * chats["recall"], 100.0 * pile["lead"])


def _id_set(rows):
    ids = [q.get("id") for q in rows]
    if len(ids) != len(set(ids)):
        return None
    return set(ids)


def run_split_score(embedder=DEFAULT_EMBEDDER, which="full", limit=None, bench="lme"):
    """Overfitting check: the prelim questions vs all the others, same full files, chats minus plain.
    "rest" is the never-seen headline."""
    if which != "full" or limit is not None:
        raise SystemExit("--split requires --set full and no --limit")
    _lock_stage("chats", embedder, bench)   # also consumes the chats (engine) recall file
    _require()
    chats_doc = read_recall_doc("chats", bench, embedder, "full")
    plain_doc = read_recall_doc("plain", bench, embedder, "full")
    if chats_doc.get("set") != "full" or plain_doc.get("set") != "full":
        raise SystemExit("--split requires set=full on both recall files")
    if chats_doc.get("embedder") != plain_doc.get("embedder"):
        raise SystemExit("split needs the same embedder on plain and chats; got %s vs %s" % (
            plain_doc.get("embedder"), chats_doc.get("embedder")))
    chats_rows = chats_doc.get("questions") or []
    plain_rows = plain_doc.get("questions") or []
    if _id_set(chats_rows) is None or _id_set(chats_rows) != _id_set(plain_rows):
        raise SystemExit("--split: chats and plain question-id sets differ")
    expected = {r["id"] for r in select(bench, "full", None)}
    if _id_set(chats_rows) != expected:
        raise SystemExit("--split: the runs do not cover every question of the full set (%d of %d)"
                         % (len(_id_set(chats_rows) & expected), len(expected)))
    guard_question_misses(chats_rows)
    guard_question_misses(plain_rows)
    report = split_pair(chats_rows, plain_rows, None, bench)
    spec = named_embedder(embedder)
    doc = {"version": VERSION, "stage": "score", "split": True, "bench": bench, "set": "full",
           "embedder": spec["model"], "embedder_flag": embedder,
           "dimensions": spec["dimensions"],
           "prelim_ids": report["prelim_ids"],
           "prelim": report["prelim"], "rest": report["rest"]}
    _require()
    _write(os.path.join(RUN, "score", "split", embedder, "full", "%s.json" % bench), doc)
    write_manifest()
    print(_split_line(report["prelim"], "prelim", report["prelim_ids"], bench))
    print(_split_line(report["rest"], "rest",
                      len(chats_doc.get("questions") or []) - len(report["prelim"]["ids"]), bench))
    return doc


def run_score(fmt, bench, which, limit, embedder=DEFAULT_EMBEDDER, split=False):
    if split:
        if fmt not in ("plain", "plainwin", "chats"):
            raise SystemExit("--split is for --format plain and chats")
        return run_split_score(embedder, which, limit, bench)
    _lock_stage(fmt, embedder, bench)
    _require()
    _doc, rows = _load_recall(fmt, bench, which, limit, embedder)
    guard_question_misses(rows)
    scored = score_rows(bench, rows)
    spec = named_embedder(embedder) if fmt in ("plain", "plainwin", "chats") else EMBED.get(fmt) or named_embedder(DEFAULT_EMBEDDER)
    doc = {"version": VERSION, "stage": "score", "format": fmt, "bench": bench, "set": which,
           "embedder": spec["model"], "embedder_flag": embedder if fmt in ("plain", "plainwin", "chats") else DEFAULT_EMBEDDER,
           "dimensions": spec["dimensions"],
           "n": scored["n"], "found": scored["found"], "recall": scored["recall"],
           "failures": scored["failures"], "abstentions": scored["abstentions"],
           "no_evidence": scored["no_evidence"], "no_evidence_ids": scored["no_evidence_ids"],
           "recall_errors": scored["recall_errors"], "mean_chars": scored["mean_chars"],
           "questions": scored["questions"]}
    _require()
    _write(out_path("score", fmt, bench, embedder if fmt in ("plain", "plainwin", "chats") else None, which, limit), doc)
    write_manifest()
    print(summary_line(fmt, bench, scored))
