# Antechamber Benchmark

Made by House of Solance · free for noncommercial use · credit required

## What it is

The open, locked test behind every number we publish for **Antechamber** (our recall engine) and **Memory entry**
(our memory writer). Anyone can re-run it and get the same numbers.

## Results

Produced by the code in this repository (benchmark v4.4.3; LOCK.json records every input) against the published
Antechamber engine 1.0.0, embedder qwen3-embedding 4B, 21 Oct 2026. Recall is scored only on the text the reader is
shown.

| Recall (evidence found) | Standard search | Antechamber | Lead |
|---|---|---|---|
| **LoCoMo, never seen (1,493 questions)** | 56.7% | **78.9%** | **+22.2** |
| **LongMemEval-S, never seen (412 questions)** | 62.6% | **77.7%** | **+15.0** |
| LoCoMo, all answerable incl. the questions used for tuning (1,532) | 56.5% | 78.6% | +22.1 (p < 10⁻¹⁵) |
| LongMemEval-S, all answerable incl. the questions used for tuning (470) | 63.8% | 77.7% | +13.8 (p = 1.5×10⁻⁸, 95% interval 9.1 to 18.5) |

"Never seen" excludes every question looked at while Antechamber was tuned (39 LoCoMo, 58 LongMemEval); it is the
headline. On those tuning questions the lead was +20.5 (LoCoMo) and +5.2 (LongMemEval).

**Smart or generous?** Antechamber returns each hit with its surrounding lines, so it hands the reader more text.
Giving standard search the same surrounding lines closes about two thirds of the gap:

| All answerable | Standard search | Standard search + the same surrounding lines | Antechamber |
|---|---|---|---|
| LoCoMo | 56.5% (1,024 characters) | 71.7% (4,703 characters) | **78.6%** (8,253 characters) |
| LongMemEval-S | 63.8% (2,576 characters) | 72.6% (25,746 characters) | **77.7%** (24,964 characters) |

Given the same surrounding lines, standard search still trails: Antechamber leads by +6.9 on LoCoMo (p = 2.7×10⁻¹⁴,
95% interval 4.5 to 9.4), where standard search was handed less text (about 4,700 characters against 8,300), so part
of that lead is still more text; and by +5.1 on LongMemEval-S at equal text (about 25,000 characters each; p = 0.037,
95% interval 0.6 to 9.8). Most of the lead is returning the surrounding conversation; the rest is choosing better.

- **Correction (21 Oct 2026):** the pre-release scorer also counted text fields the reader is never shown. Scored on
  reader-visible text only, LongMemEval's Antechamber result is 77.7%, not the 80.3% (never seen) / 80.6% (all) we
  first reported; LoCoMo and standard search did not change. This repository scores exactly what the reader sees.
- **Claim 3 (any storage format) is not re-measured in this version.** In the pre-release run the notes and
  vector-database adapters returned less surrounding context than the chat adapter and did not hold; it will be
  re-measured, not re-described.
- **Answering** (a model answers from what recall returned, a fixed judge marks it) was measured separately on 600 +
  600 LoCoMo and 30 + 30 LongMemEval questions; its code is not in this repository yet. In words: on a small local
  model (Qwen 3.5 9B) Antechamber plus our memory writer and People cards beat a generic memory writer by about 6
  points on both LoCoMo sets; on large models the two tied.

## What you need

- Python 3.11 or newer (standard library only, nothing to install).
- [Ollama](https://ollama.com) with the embedding model every table uses: `ollama pull qwen3-embedding:4b`.
- To test Antechamber itself: a checkout of the [Antechamber](https://github.com/SolanceLab/antechamber) engine.
  Standard search needs no engine.
- Only for the vector-database format: Postgres with pgvector.

## Install

```sh
git clone https://github.com/SolanceLab/antechamber-benchmark && cd antechamber-benchmark
ollama pull qwen3-embedding:4b           # the embedder every table uses (2000 dimensions)

make test                                # 1. offline checks; no Ollama, no engine, no datasets needed
make data                                # 2. downloads LoCoMo and LongMemEval-S, checks each file's fingerprint
export ANTECHAMBER_DIR=/path/to/antechamber   # 3. a checkout of the engine (the folder holding engine/recall_service.py)
make lock                                # 4. only if the repository has no LOCK.json (see "The lock" under How it works)
make recall                              # 5. standard search and Antechamber, both datasets, every answerable question
make score                               # 6. found / asked per run, plus the never-seen split
make verdict                             # 7. the paired statistics (gain, exact McNemar p, 95% interval)
```

`data/download.sh` fetches LoCoMo and LongMemEval-S from their owners and checks each file against the exact
fingerprint our runs used. The datasets belong to their authors and keep their own licences; they are not
redistributed here.

## Settings

Where things go, all git-ignored:

| Setting | Default | Holds |
|---|---|---|
| `ANTECHAMBER_DIR` | none | the engine checkout; `engine/recall_service.py` is started once per conversation |
| `OLLAMA_URL` | `http://localhost:11434` | the Ollama that embeds text |
| `ACBENCH_DATA` | `./data` | the two datasets and the converted exports |
| `ACBENCH_RUNS` | `./runs` | every run's output, in `runs/<version>/recall/<format>/<embedder>/<set>/<bench>.json` |
| `ACBENCH_CACHE` | `./cache` | the embedding cache (same text, same model weights, same vector: it saves time, never changes a result) |

Only the vector-database format reads these Postgres settings: `ACBENCH_PSQL` (default `psql`), `ACBENCH_PG_HOST`
(`localhost`), `ACBENCH_PG_PORT` (`5432`), `ACBENCH_PG_USER` (`postgres`), `ACBENCH_PG_DB` (`acbench_pgvector`, created
for you), `ACBENCH_PG_BIN_DIR` (the folder of `psql`). The engine port starts at `ACBENCH_PORT` (`8791`) and moves up
if it is taken.

## Tools

Standard search needs no engine, so `python3 -m acbench recall --bench lme --format plain` runs with only Ollama.
Each command is one line you can run on its own:

```sh
python3 -m acbench convert --bench locomo|lme --format chats|notes|pgvector [--limit N] [--prelim]
python3 -m acbench recall  --bench locomo|lme --format plain|plainwin|chats|notes|pgvector
                           [--set prelim|full|unseen] [--embedder qwen3-embedding:4b] [--limit N]
python3 -m acbench score   --bench locomo|lme --format plain|chats [--split]
python3 -m acbench verdict --bench locomo|lme [--set full]
```

| `--format` | What answers the question | Needs |
|---|---|---|
| `plain` | standard search: each message embedded whole, the 8 closest by meaning | Ollama |
| `plainwin` | standard search plus the same surrounding lines Antechamber returns (the "smart or generous" check) | Ollama |
| `chats` | Antechamber reading the conversations as a chat export | Ollama, `ANTECHAMBER_DIR` |
| `notes` | Antechamber reading the same conversations as Markdown notes | Ollama, `ANTECHAMBER_DIR` |
| `pgvector` | Antechamber reading the same conversations from a vector database | Ollama, `ANTECHAMBER_DIR`, Postgres with pgvector |

`--set full` is every answerable question (1,532 LoCoMo, 470 LongMemEval-S). `--set prelim` is the 99 questions
used while tuning, which is also the check set for `plainwin`. `--set unseen` (LoCoMo) is everything else.
`--embedder nomic` (or `mxbai`) repeats the pair with a smaller embedder for claim 2; `make recall-embedder` does it.
`make recall-check` runs `plainwin`, `make recall-formats` runs claim 3.

`score --split` reads the saved `plain` and `chats` runs of the full set and reports the tuning-seen questions and the
never-seen questions separately, with the Antechamber lead on each. The never-seen line is the headline number.

## Limits

- **No leaderboard position.** Published LoCoMo and LongMemEval tables mostly used GPT-4o-mini to answer and GPT-4o or
  GPT-4o-mini to judge; those models are no longer available on OpenAI's API (owner's note, 21 Oct 2026), so those
  setups cannot be rerun by anyone. Answer scores also move with the answering model and the judge (an independent
  audit found GPT-4o-mini accepting 62.8% of deliberately wrong LoCoMo answers). We lead with recall against plain
  search on never-seen questions, which anyone can rerun from this repository with any embedder.
- One locked setup per table: code, data, settings and model versions are fingerprinted; the runner refuses to start
  if anything changed. Any change is a new version and a full re-run; numbers from different versions never share a
  table.
- Every version's numbers stay in the record.
- No comparisons with other memory systems. Their published numbers use different rulers (whole sessions vs exact
  lines, top 5 vs top 20, their own judges), so a side-by-side would look precise and mean nothing. Run your system
  through this test instead.
- **Not here yet.** The answering ladder (a reader answers from what recall returned, a fixed judge marks it); its
  results are described under Results. Claim 3 (any storage format) is not re-measured in this version.

## How it works

**Recall** — when a question is asked, does the engine hand back the conversation lines that hold the answer?
A line counts as found when at least 60% of its four-word phrases appear in the text the engine returned. Only text a
reader actually sees counts. Errors and time-outs count as not found and never leave the total.

Three claims, each with its own test, so one failing does not sink the others:

| Claim | Test |
|---|---|
| 1. Antechamber beats standard search | the same questions through standard search and through Antechamber |
| 2. …whatever embedding model you use | claim 1 repeated with a second, smaller embedder |
| 3. …whatever format you store memory in | the same conversations as a chat export, Markdown notes and a vector database |

Two checks guard the claims:

- **Smart or generous?** Standard search is also given the same surrounding lines Antechamber returns, so a lead
  that comes only from returning more text is visible.
- **Overfitting.** Part of LongMemEval was looked at while Antechamber was tuned. Those questions are scored
  separately from the ones it never saw.

**Answering** is measured separately (a reader model answers from what recall returned, a fixed judge marks the
answer); see Results.

**The baseline.** Standard search is what you get by doing nothing special: each message embedded whole, the 8 closest messages by
meaning, nothing else. That matches common chat-memory practice (we checked the LlamaIndex and LangChain
chat-memory documentation). It is never adjusted to make anything look better. A message the embedder refuses as too
long is standard search's real loss: counted and reported, not cut to fit.

**The lock.** Every input that can move a number is fingerprinted in `LOCK.json`: both dataset files, the converted data
the runs read (`data/chats`, `data/notes`, `data/pgvector`), this code and
its question lists, each embedding model's weights, the Ollama and Python versions, and (for Antechamber) the engine
folder and its git commit. Every run checks it at the start, every 25 questions and at the end, and refuses to
continue if anything differs; there is no override. `verdict` and `score` check it too, and refuse a recall file
that is from another version, stage or embedder, or that changed after its `MANIFEST.sha256` was written. If your setup differs from the published one, `make lock` converts everything,
then freezes yours (all three embedders, the engine and Postgres must be present; `python3 -m acbench.lock freeze plain` freezes
only what standard search uses): your numbers are then yours, and they are not the published table. A change to the code or the setup is a new
version, with a full re-run.

## Next: LongMemEval-M (planned, not yet run)

LongMemEval-S gives each question about 48 past conversations (~115k tokens). That is small enough that handing a
model a large slice of whole conversations can score well without much real finding. LongMemEval-M gives each
question about 500 conversations (~1.5M tokens), where no slice is safe and only finding works.

The plan: recall only, no language model, local embeddings. Three contenders on the same questions: plain vector
search, whole conversations picked by search up to ~24k tokens, and Antechamber. Scored by whether every evidence
line was handed over, by whether any was, and by how much text each handed over. The same questions also run on
LongMemEval-S with the same code, so the only difference is the size of the history. The held-out questions are not
used. It runs slowly on one Mac (about 390,000 conversation lines to embed), so results will be added here when done.

## Credit and use

The Antechamber Benchmark is made by House of Solance. You're welcome to use it, change it and build on it for any
noncommercial purpose, as long as our credit travels with it.

**Keep the credit.** Every copy, changed or not, must carry the licence and this line:
`Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)`
Removing or replacing it breaks the licence.

**Commercial use needs our permission.** Ask us first.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md). See [NOTICE.md](NOTICE.md).

