#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/05_figure5 \
  logs

echo "========================================"
echo " Figure 5 analysis"
echo "========================================"

echo
echo "[1/2] Feature-level importance"

python \
  src/05_figure5/01_feature_importance.py \
  > logs/figure4_feature_importance.log 2>&1

tail -40 \
  logs/figure4_feature_importance.log


echo
echo "[2/2] Representative feature distributions"

python \
  src/05_figure5/02_feature_distributions.py \
  > logs/figure4_feature_distributions.log 2>&1

cat \
  logs/figure4_feature_distributions.log


echo
echo "========================================"
echo " Figure 5 analysis completed"
echo "========================================"
