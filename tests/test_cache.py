# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The embedding cache: same bytes as Ollama, one upstream call per new text, a new key when weights change."""
import http.server
import json
import os
import tempfile
import threading
import unittest
import urllib.request

from acbench import embed_cache
from tests.fixtures import Patch


class FakeOllama:
    """Answers /api/tags and /api/embed. A vector is derived from the text so a test can tell which text it was."""

    def __init__(self):
        self.digest = "d1"
        self.embed_calls = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_GET(self):
                body = json.dumps({"models": [{"name": "fake-embed:latest", "digest": outer.digest}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.embed_calls.append(data)
                vecs = ",".join("[%s,0.1000000000000001,%d]" % (len(t) / 7.0, len(t)) for t in data["input"])
                body = ('{"model":"%s","embeddings":[%s],"total_duration":1}' % (data["model"], vecs)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class CacheBehaviour(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="acbench-cache-")
        self.fake = FakeOllama()
        self.saved = {k: os.environ.get(k) for k in ("OLLAMA_URL", "ACBENCH_CACHE")}
        os.environ["OLLAMA_URL"] = self.fake.url
        os.environ["ACBENCH_CACHE"] = self.tmp.name
        self.old_db = embed_cache._db
        embed_cache._db = None

    def tearDown(self):
        if embed_cache._db is not None:
            embed_cache._db.close()
        embed_cache._db = self.old_db
        self.fake.close()
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_vectors_keep_ollamas_exact_bytes_and_a_second_ask_is_a_hit(self):
        first = embed_cache.embed("fake-embed", ["alpha", "beta beta"], 3)
        self.assertEqual(first[0], "[0.7142857142857143,0.1000000000000001,5]")   # not re-rounded
        self.assertEqual(len(self.fake.embed_calls), 1)
        again = embed_cache.embed("fake-embed", ["beta beta", "alpha"], 3)
        self.assertEqual(again, [first[1], first[0]])
        self.assertEqual(len(self.fake.embed_calls), 1)           # nothing went upstream
        mixed = embed_cache.embed("fake-embed", ["alpha", "gamma"], 3)
        self.assertEqual(self.fake.embed_calls[-1]["input"], ["gamma"])   # only the new text is forwarded
        self.assertEqual(mixed[0], first[0])
        self.assertEqual(self.fake.embed_calls[-1]["dimensions"], 3)

    def test_a_weight_change_is_a_new_key(self):
        embed_cache.embed("fake-embed", ["alpha"], 3)
        self.fake.digest = "d2"
        embed_cache.embed("fake-embed", ["alpha"], 3)
        self.assertEqual(len(self.fake.embed_calls), 2)
        self.assertNotEqual(embed_cache.cache_key("m", 3, "t", "d1"), embed_cache.cache_key("m", 3, "t", "d2"))
        self.assertNotEqual(embed_cache.cache_key("m", 3, "t", "d1"), embed_cache.cache_key("m", 4, "t", "d1"))
        self.assertEqual(len(embed_cache.cache_key("m", 3, "t", "d1")), 64)

    def test_cache_lives_in_acbench_cache(self):
        embed_cache.embed("fake-embed", ["alpha"], 3)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "embed.sqlite")))

    def test_http_front_door(self):
        server = embed_cache._Server(("127.0.0.1", 0), embed_cache.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = "http://127.0.0.1:%d" % server.server_address[1]
        try:
            with Patch((embed_cache, "_TOKEN", "tok")):
                with urllib.request.urlopen(base + "/health", timeout=5) as resp:
                    body = json.loads(resp.read())
                self.assertEqual((body["ok"], body["token"], body["pid"]), (True, "tok", os.getpid()))
            req = urllib.request.Request(base + "/api/embed", data=json.dumps(
                {"model": "fake-embed", "input": ["alpha"], "dimensions": 3}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                got = json.loads(resp.read())
            self.assertEqual(got["embeddings"][0][2], 5)
            bad = urllib.request.Request(base + "/api/embed", data=b'{"input": ["x"]}')
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad, timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            server.shutdown()
            server.server_close()


class Parsing(unittest.TestCase):
    def test_split_array_keeps_each_element_as_written(self):
        text = '[[0.1, 2e-05,3],[ 4 ,5,6]]'
        self.assertEqual(embed_cache._split_array(text), ["[0.1, 2e-05,3]", "[ 4 ,5,6]"])

    def test_embeddings_slice(self):
        raw = '{"model":"m","embeddings":[[1,2],[3,4]],"total_duration":7}'
        array_text, value = embed_cache._embeddings_slice(raw)
        self.assertEqual(array_text, "[[1,2],[3,4]]")
        self.assertEqual(value, [[1, 2], [3, 4]])
        with self.assertRaises(RuntimeError):
            embed_cache._embeddings_slice('{"nothing": 1}')

    def test_digest_missing_model_raises(self):
        fake = FakeOllama()
        old = os.environ.get("OLLAMA_URL")
        os.environ["OLLAMA_URL"] = fake.url
        try:
            self.assertEqual(embed_cache.model_digest("fake-embed"), "d1")
            with self.assertRaises(RuntimeError):
                embed_cache.model_digest("not-pulled")
        finally:
            fake.close()
            if old is None:
                os.environ.pop("OLLAMA_URL", None)
            else:
                os.environ["OLLAMA_URL"] = old


if __name__ == "__main__":
    unittest.main()
