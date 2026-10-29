#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/02_figure2 \
  figures/main \
  logs

echo "========================================"
echo " Figure 2"
echo "========================================"

echo
echo "[1/3] Reason generalization hierarchy"

python \
  src/02_figure2/01_generalization_hierarchy.py \
  > logs/figure1_reason_generalization.log 2>&1

tail -40 \
  logs/figure1_reason_generalization.log


echo
echo "[2/3] Infant identity"

python \
  src/02_figure2/02_infant_identity.py \
  > logs/figure1_infant_identity.log 2>&1

tail -30 \
  logs/figure1_infant_identity.log


echo
echo "[3/3] Generate Figure 2"

python \
  src/02_figure2/03_make_figure2.py \
  > logs/figure1_make.log 2>&1

cat \
  logs/figure1_make.log


echo
echo "========================================"
echo " Figure 2 completed"
echo "========================================"
