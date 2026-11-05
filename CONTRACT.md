# Antechamber Benchmark contract

## Doors it offers

None. It is a command-line test: no service, no MCP server, no network port it keeps open. While a recall run is in
progress it starts the Antechamber engine you point it at (`ANTECHAMBER_DIR`) on a local port and stops it when the
run ends.

| Door | What it answers |
|---|---|
| — | — |

## Data it owns

Only its own working files, all git-ignored, all on your machine.

| Table / bucket / file | What is in it |
|---|---|
| `data/` (`ACBENCH_DATA`) | the two downloaded datasets and the exports converted from them |
| `runs/` (`ACBENCH_RUNS`) | every run's results, scores and `MANIFEST.sha256` |
| `cache/` (`ACBENCH_CACHE`) | the embedding cache (`embed.sqlite`) |
| `LOCK.json` | the fingerprints of every input the published numbers depend on (committed) |
| Postgres database `acbench_pgvector` (`ACBENCH_PG_DB`) | the vector-database export, only if you run that format |

## Settings and secrets it needs

Named here, never valued. It needs no API key: embeddings run on your own Ollama, and no language model or judge is
called.

| Name | What it is |
|---|---|
| `ANTECHAMBER_DIR` | path to an Antechamber engine checkout (only for the Antechamber formats) |
| `OLLAMA_URL` | the Ollama that embeds text |
| `ACBENCH_DATA`, `ACBENCH_RUNS`, `ACBENCH_CACHE` | where its working files go |
| `ACBENCH_PSQL`, `ACBENCH_PG_HOST`, `ACBENCH_PG_PORT`, `ACBENCH_PG_USER`, `ACBENCH_PG_DB`, `ACBENCH_PG_BIN_DIR`, `ACBENCH_PGVECTOR_DSN` | Postgres connection, only for the vector-database format |

It sets `ANTECHAMBER_CONFIG`, `ANTECHAMBER_WIDE`, `ANTECHAMBER_DATES`, `ANTECHAMBER_CHRONO` and `ANTECHAMBER_SPREAD` itself
for the engine it starts, so every run uses the published engine settings; they are not yours to set.
| `ACBENCH_PORT` | the first local port tried for the engine during a run |
