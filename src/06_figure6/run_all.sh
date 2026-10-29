#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/06_figure6 \
  logs

echo "========================================"
echo " Figure 6 analysis"
echo "========================================"

echo
echo "[1/2] wav2vec2 reason generalization"

python \
  src/06_figure6/02_reason_generalization.py \
  > logs/figure5_reason_generalization.log 2>&1

tail -50 \
  logs/figure5_reason_generalization.log


echo
echo "[2/2] wav2vec2 dataset identity"

python \
  src/06_figure6/03_dataset_identity.py \
  > logs/figure5_dataset_identity.log 2>&1

tail -40 \
  logs/figure5_dataset_identity.log


echo
echo "========================================"
echo " Figure 6 analysis completed"
echo "========================================"
