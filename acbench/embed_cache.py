#!/usr/bin/env python3
# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Local cache in front of Ollama's POST /api/embed.

Listens on 127.0.0.1:11435. A miss is forwarded to $OLLAMA_URL/api/embed (default http://localhost:11434).
Hits are stored in $ACBENCH_CACHE/embed.sqlite (default <repo>/cache) keyed by
sha256(model|digest|dimensions|text), where digest is the model's full id from Ollama /api/tags.
A weight change is a new key. Each returned vector is the exact JSON array Ollama sent, spliced back in order.

The runner uses this port only when the listener is this process: EMBED_CACHE_TOKEN is
returned by /health and EMBED_CACHE_PID_FILE holds our pid.

This file runs as its own process and imports nothing from the package.
"""
import hashlib
import json
import os
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 11435

_lock = threading.Lock()
_db = None
_TOKEN = ""


def ollama_url():
    return (os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")


def upstream():
    return ollama_url() + "/api/embed"


def tags_url():
    return ollama_url() + "/api/tags"


def cache_dir():
    value = os.environ.get("ACBENCH_CACHE")
    if value:
        return os.path.abspath(os.path.expanduser(value))
    return os.path.join(os.path.dirname(HERE), "cache")


def db_path():
    return os.path.join(cache_dir(), "embed.sqlite")


def _conn():
    global _db
    if _db is None:
        path = db_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _db = sqlite3.connect(path, check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, raw TEXT NOT NULL)")
        _db.commit()
    return _db


def model_digest(model):
    """Full digest from Ollama /api/tags, read on every batch so a weight change cannot reuse a key."""
    try:
        with urllib.request.urlopen(tags_url(), timeout=30) as resp:
            tags = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError("ollama /api/tags failed: %s" % e) from e
    exact = latest = prefix = None
    for m in tags.get("models") or []:
        got = m.get("name") or ""
        digest = m.get("digest") or ""
        if not digest:
            continue
        if got == model:
            exact = digest
        elif got == model + ":latest":
            latest = digest
        elif got.split(":")[0] == model and prefix is None:
            prefix = digest
    found = exact or latest or prefix
    if not found:
        raise RuntimeError("ollama has no digest for %s" % model)
    return found


def cache_key(model, dimensions, text, digest=None):
    """Key includes the weight digest so a replaced model cannot reuse an old vector."""
    if digest is None:
        digest = model_digest(model)
    dim = "" if dimensions is None else str(int(dimensions))
    return hashlib.sha256(("%s|%s|%s|%s" % (model, digest, dim, text)).encode()).hexdigest()


def _embeddings_slice(text):
    i = text.find('"embeddings"')
    if i < 0:
        raise RuntimeError("ollama response has no embeddings")
    j = text.find("[", i + len('"embeddings"'))
    if j < 0:
        raise RuntimeError("ollama embeddings is not an array")
    value, end = json.JSONDecoder().raw_decode(text, j)
    if not isinstance(value, list):
        raise RuntimeError("ollama embeddings is not a list")
    return text[j:end], value


def _split_array(array_text):
    """Top-level elements of a JSON array, each element kept as Ollama wrote it."""
    blobs = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(array_text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "]":
            if depth == 2 and start is not None:
                blobs.append(array_text[start:i + 1])
                start = None
            depth -= 1
    return blobs


def fetch_upstream(model, texts, dimensions):
    """Forward a miss. Returns one raw JSON array per text, byte-identical to Ollama."""
    body = {"model": model, "input": list(texts), "keep_alive": -1}
    if dimensions is not None:
        body["dimensions"] = dimensions
    req = urllib.request.Request(
        upstream(), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        raise RuntimeError("ollama embed failed: %s" % detail) from e
    array_text, parsed = _embeddings_slice(raw.decode("utf-8"))
    blobs = _split_array(array_text)
    if len(blobs) != len(parsed) or len(parsed) != len(texts):
        raise RuntimeError("ollama returned %d vectors for %d texts" % (len(parsed), len(texts)))
    for blob, value in zip(blobs, parsed):
        if json.loads(blob) != value:
            raise RuntimeError("could not keep Ollama's vector bytes")
    return blobs


def embed(model, texts, dimensions):
    digest = model_digest(model)
    keys = [cache_key(model, dimensions, t, digest) for t in texts]
    with _lock:
        db = _conn()
        have = {}
        for k in keys:
            row = db.execute("SELECT raw FROM vec WHERE k = ?", (k,)).fetchone()
            if row:
                have[k] = row[0]
    missing_at = [i for i, k in enumerate(keys) if k not in have]
    if missing_at:
        got = fetch_upstream(model, [texts[i] for i in missing_at], dimensions)
        pairs = []
        for i, blob in zip(missing_at, got):
            have[keys[i]] = blob
            pairs.append((keys[i], blob))
        with _lock:
            db = _conn()
            db.executemany("INSERT OR REPLACE INTO vec (k, raw) VALUES (?, ?)", pairs)
            db.commit()
    return [have[k] for k in keys]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            body = {"ok": True, "pid": os.getpid()}
            if _TOKEN:
                body["token"] = _TOKEN
            return self._send(200, json.dumps(body))
        return self._send(404, '{"error":"not found"}')

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/embed":
            return self._send(404, '{"error":"not found"}')
        try:
            n = int(self.headers.get("Content-Length") or "0")
            data = json.loads(self.rfile.read(n) or b"{}")
            model = data.get("model")
            if not model:
                return self._send(400, '{"error":"model required"}')
            incoming = data.get("input")
            if isinstance(incoming, str):
                texts = [incoming]
            elif isinstance(incoming, list) and all(isinstance(t, str) for t in incoming):
                texts = incoming
            else:
                return self._send(400, '{"error":"input must be a string or a list of strings"}')
            dimensions = data.get("dimensions")
            if dimensions is not None:
                dimensions = int(dimensions)
            blobs = embed(model, texts, dimensions)
        except Exception as e:
            return self._send(502, json.dumps({"error": str(e)[:300]}))
        return self._send(200, '{"embeddings":[%s]}' % ",".join(blobs))


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True


def serve(port=PORT):
    global _TOKEN
    _TOKEN = os.environ.get("EMBED_CACHE_TOKEN", "")
    pid_path = os.environ.get("EMBED_CACHE_PID_FILE")
    if pid_path:
        os.makedirs(os.path.dirname(pid_path) or ".", exist_ok=True)
        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    httpd = _Server(("127.0.0.1", port), Handler)
    print("embed cache on 127.0.0.1:%d" % port, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else PORT)
