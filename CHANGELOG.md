# Changelog

## 1.0.0

First release. Every result in the README is produced by this repository's own code (harness v4.4.3).

- The recall harness: `python3 -m acbench recall|score|verdict|convert`, the dataset converters, the question sets,
  scoring, the paired statistics, the lock, and offline tests (`make test`).
- Settings by environment variables; no path to any machine is written into the code.
- Results (21 Oct 2026) rerun with this code (v4.4.3) against Antechamber 1.0.0 (59197ac): recall
  scored on reader-visible text only; LongMemEval corrected to 77.7% (was 80.3% never seen); smart-or-generous check
  rerun on all answerable questions.
