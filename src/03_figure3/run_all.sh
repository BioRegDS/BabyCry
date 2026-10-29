#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/03_figure3 \
  results/03_figure3/predictions \
  results/03_figure3/infant_auc \
  logs

echo "========================================"
echo " Figure 3 analysis"
echo "========================================"

echo
echo "[1/2] Cross-dataset reason classification"

python \
  src/03_figure3/01_cross_dataset_reason.py \
  > logs/figure2_cross_dataset_reason.log 2>&1

cat \
  logs/figure2_cross_dataset_reason.log


echo
echo "[2/2] Cluster-aware inference"

python \
  src/03_figure3/02_cluster_inference.py \
  > logs/figure2_cluster_inference.log 2>&1

cat \
  logs/figure2_cluster_inference.log


echo
echo "========================================"
echo " Figure 3 analysis completed"
echo "========================================"
