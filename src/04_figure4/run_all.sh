#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/04_figure4 \
  logs

echo "========================================"
echo " Figure 4 analysis"
echo "========================================"

echo
echo "[1/3] Dataset identity"

python \
  src/04_figure4/01_dataset_identity.py \
  > logs/figure3_dataset_identity.log 2>&1

tail -30 \
  logs/figure3_dataset_identity.log


echo
echo "[2/3] Feature-family identity"

python \
  src/04_figure4/02_feature_family_identity.py \
  > logs/figure3_feature_family.log 2>&1

tail -50 \
  logs/figure3_feature_family.log


echo
echo "[3/3] Reason vs dataset comparison"

python \
  src/04_figure4/03_reason_vs_dataset.py \
  > logs/figure3_reason_vs_dataset.log 2>&1

cat \
  logs/figure3_reason_vs_dataset.log


echo
echo "========================================"
echo " Figure 4 analysis completed"
echo "========================================"
