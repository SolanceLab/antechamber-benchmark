#!/bin/sh
# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
# Fetch the two public benchmark datasets and verify they are byte-identical to the files every published run used.
set -e
cd "$(dirname "$0")"
fetch() { # url file sha256
  [ -f "$2" ] || curl -fsSL "$1" -o "$2"
  got=$(shasum -a 256 "$2" | cut -d' ' -f1)
  [ "$got" = "$3" ] || { echo "MISMATCH $2: $got (want $3)"; exit 1; }
  echo "ok $2"
}
fetch https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
      locomo10.json 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
fetch https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
      longmemeval_s_cleaned.json d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
