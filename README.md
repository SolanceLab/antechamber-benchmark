# Antechamber Benchmark

Made by House of Solance · free for noncommercial use · credit required

## What it is

The open, locked test behind every number we publish for **Antechamber** (our recall engine) and **Memory entry**
(our memory writer). Anyone can re-run it and get the same numbers.

## What it measures

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

**Answering** (coming next) — a reader model answers from what recall returned, a fixed judge marks the answer.
It is a ladder: conversation alone, then plus a standard memory, then plus Memory entry. Each rung must beat the one
below it, with the same reader, judge and questions.

## The baseline

Standard search is what you get by doing nothing special: each message embedded whole, the 8 closest messages by
meaning, nothing else. That matches common chat-memory practice (we checked the LlamaIndex and LangChain
chat-memory documentation). It is never adjusted to make anything look better. A message the embedder refuses as too
long is standard search's real loss: counted and reported, not cut to fit.

## Results

> These numbers were produced by the pre-release version of this code (v3.4.0). Before this repository goes public
> they will be re-run with the code published here (v3.5.0), so every number comes from code you can read.

| Recall (evidence found), embedder qwen3-embedding 4B | Standard search | Antechamber | Lead |
|---|---|---|---|
| LoCoMo, never seen (1,493 questions) | 56.7% | 78.9% | +22.2 |
| LongMemEval-S, never seen (412 questions) | 62.6% | 80.3% | +17.7 |
| LoCoMo, all answerable incl. the questions used for tuning (1,532) | 56.5% | 78.6% | +22.1 |
| LongMemEval-S, all answerable incl. the questions used for tuning (470) | 63.8% | 80.6% | +16.8 |

- **Overfitting check (LongMemEval):** lead +10.3 on the 58 scored questions seen during tuning, +17.7 on the 412 never seen.
  "Never seen" excludes every question used while tuning (39 LoCoMo, 58 LongMemEval); it is the headline.
- **Smart or generous (99-question check set):** giving standard search the same surrounding lines closes about half
  the gap; the other half is Antechamber choosing better.
- **Claim 3 does not hold yet:** the notes and vector-database adapters return less surrounding context than the chat
  adapter. The fix is product work; the result will be re-measured, not re-described.
- Every scored verdict was spot-checked by hand (20 of 20 on the check set, 10 of 10 on the full LongMemEval run).

## Next: LongMemEval-M (planned, not yet run)

LongMemEval-S gives each question about 48 past conversations (~115k tokens). That is small enough that handing a
model a large slice of whole conversations can score well without much real finding. LongMemEval-M gives each
question about 500 conversations (~1.5M tokens), where no slice is safe and only finding works.

The plan: recall only, no language model, local embeddings. Three contenders on the same questions: plain vector
search, whole conversations picked by search up to ~24k tokens, and Antechamber. Scored by whether every evidence
line was handed over, by whether any was, and by how much text each handed over. The same questions also run on
LongMemEval-S with the same code, so the only difference is the size of the history. The held-out questions are not
used. It runs slowly on one Mac (about 390,000 conversation lines to embed), so results will be added here when done.

## Rules we hold ourselves to

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

## Data

`data/download.sh` fetches LoCoMo and LongMemEval-S from their owners and checks each file against the exact
fingerprint our runs used. The datasets belong to their authors and keep their own licences; they are not
redistributed here.

## Credit and use

The Antechamber Benchmark is made by House of Solance. You're welcome to use it, change it and build on it for any
noncommercial purpose, as long as our credit travels with it.

**Keep the credit.** Every copy, changed or not, must carry the licence and this line:
`Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)`
Removing or replacing it breaks the licence.

**Commercial use needs our permission.** Ask us first.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md). See [NOTICE.md](NOTICE.md).
