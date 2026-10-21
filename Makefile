# Antechamber Benchmark. Run from this folder. Settings are environment variables; see acbench/settings.py.
#
#   make test      offline checks, no Ollama and no engine needed
#   make data      download LoCoMo and LongMemEval-S and check their fingerprints (data/download.sh)
#   make lock      freeze the locked setup into LOCK.json (the published LOCK.json is already frozen)
#   make recall    standard search and Antechamber on both datasets, qwen3-embedding:4b
#                  (needs Ollama running and ANTECHAMBER_DIR set)
#   make verdict   the paired statistics from the saved recall files
#
# Optional extras:  make recall-check   the "smart or generous" check (standard search + the same surrounding lines)
#                   make recall-embedder   claim 2: the same pair with a second, smaller embedder
#                   make recall-formats   claim 3: notes and vector database (the vector database needs Postgres)
PY ?= python3
BENCHES ?= locomo lme
EMBEDDER ?= nomic

.PHONY: test data lock convert-all convert recall recall-check recall-embedder recall-formats verdict score clean-runs

test:
	$(PY) -m unittest discover -s tests -t . -v

data:
	sh data/download.sh

# The all-scope freeze fingerprints every converted folder and the pgvector tables, so it runs after conversion.
lock:
	$(PY) -m acbench.lock precheck
	$(MAKE) convert-all
	$(PY) -m acbench.lock freeze

convert-all:
	for b in $(BENCHES); do \
	  for f in chats notes pgvector; do $(PY) -m acbench convert --bench $$b --format $$f || exit 1; done; \
	done

convert:
	for b in $(BENCHES); do $(PY) -m acbench convert --bench $$b --format chats || exit 1; done

recall: convert
	for b in $(BENCHES); do \
	  $(PY) -m acbench recall --bench $$b --format plain --set full || exit 1; \
	  $(PY) -m acbench recall --bench $$b --format chats --set full || exit 1; \
	done

score:
	for b in $(BENCHES); do \
	  for f in plain chats; do $(PY) -m acbench score --bench $$b --format $$f --set full || exit 1; done; \
	  $(PY) -m acbench score --bench $$b --format chats --set full --split || exit 1; \
	done

recall-check: convert
	for b in $(BENCHES); do $(PY) -m acbench recall --bench $$b --format plainwin --set prelim || exit 1; done

recall-embedder: convert
	for b in $(BENCHES); do \
	  $(PY) -m acbench recall --bench $$b --format plain --set full --embedder $(EMBEDDER) || exit 1; \
	  $(PY) -m acbench recall --bench $$b --format chats --set full --embedder $(EMBEDDER) || exit 1; \
	done

recall-formats:
	for b in $(BENCHES); do \
	  $(PY) -m acbench convert --bench $$b --format notes || exit 1; \
	  $(PY) -m acbench recall --bench $$b --format notes --set full || exit 1; \
	done
	for b in $(BENCHES); do \
	  $(PY) -m acbench convert --bench $$b --format pgvector || exit 1; \
	  $(PY) -m acbench recall --bench $$b --format pgvector --set full || exit 1; \
	done

verdict:
	for b in $(BENCHES); do $(PY) -m acbench verdict --bench $$b --set full || exit 1; done

clean-runs:
	rm -rf runs
