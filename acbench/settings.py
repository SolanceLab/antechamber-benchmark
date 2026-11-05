# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Every setting the benchmark reads from the environment, with its default. Nothing else is configurable.

  ANTECHAMBER_DIR   a checkout of the Antechamber engine; <dir>/engine/recall_service.py is started for the
                    chats / notes / pgvector formats. No default. Plain search does not need it.
  OLLAMA_URL        the local Ollama that embeds text            default http://localhost:11434
  ACBENCH_DATA      the two downloaded datasets and the converted  default <repo>/data
                    chat / note / pgvector exports
  ACBENCH_RUNS      every run's output (git-ignored)              default <repo>/runs
  ACBENCH_CACHE     the embedding cache (git-ignored)             default <repo>/cache

Only the pgvector format needs a Postgres with the pgvector extension:

  ACBENCH_PSQL      the psql program                              default psql
  ACBENCH_PG_HOST   default localhost      ACBENCH_PG_PORT   default 5432
  ACBENCH_PG_USER   default postgres       ACBENCH_PG_DB     default acbench_pgvector
  ACBENCH_PG_BIN_DIR  the Postgres bin directory written into the engine config   default: the folder of psql
  ACBENCH_PORT      first port tried for the engine service       default 8791

Relative values are taken from the folder the command is run in. The functions read the environment when called,
so a test can change a setting and see it.
"""
import os
import shutil

PACKAGE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE)

CACHE_PORT = 11435
RECALL_LIMIT = 8


def _path(name, default):
    value = os.environ.get(name)
    if value:
        return os.path.abspath(os.path.expanduser(value))
    return default


def antechamber_dir():
    """The engine checkout, or None when ANTECHAMBER_DIR is not set."""
    value = os.environ.get("ANTECHAMBER_DIR")
    return os.path.abspath(os.path.expanduser(value)) if value else None


def require_antechamber_dir():
    path = antechamber_dir()
    if not path:
        raise SystemExit("ANTECHAMBER_DIR is not set. Point it at a checkout of the Antechamber engine "
                         "(the folder that holds engine/recall_service.py).")
    return path


def engine_dir():
    path = antechamber_dir()
    return os.path.join(path, "engine") if path else None


def engine_script():
    return os.path.join(require_antechamber_dir(), "engine", "recall_service.py")


def ollama_url():
    return (os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")


def ollama_embed_url():
    return ollama_url() + "/api/embed"


def ollama_tags_url():
    return ollama_url() + "/api/tags"


def data_dir():
    return _path("ACBENCH_DATA", os.path.join(ROOT, "data"))


def runs_root():
    return _path("ACBENCH_RUNS", os.path.join(ROOT, "runs"))


def cache_dir():
    return _path("ACBENCH_CACHE", os.path.join(ROOT, "cache"))


def first_port():
    try:
        return int(os.environ.get("ACBENCH_PORT") or 8791)
    except ValueError:
        raise SystemExit("ACBENCH_PORT must be a number")


def psql():
    return os.environ.get("ACBENCH_PSQL") or "psql"


def pg_host():
    return os.environ.get("ACBENCH_PG_HOST") or "localhost"


def pg_port():
    return str(os.environ.get("ACBENCH_PG_PORT") or "5432")


def pg_user():
    return os.environ.get("ACBENCH_PG_USER") or "postgres"


def pg_db():
    return os.environ.get("ACBENCH_PG_DB") or "acbench_pgvector"


def pg_bin_dir():
    value = os.environ.get("ACBENCH_PG_BIN_DIR")
    if value:
        return os.path.abspath(os.path.expanduser(value))
    found = shutil.which(psql())
    return os.path.dirname(os.path.realpath(found)) if found else ""


def pg_dsn():
    return "host=%s port=%s user=%s dbname=%s" % (pg_host(), pg_port(), pg_user(), pg_db())
