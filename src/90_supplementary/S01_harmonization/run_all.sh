#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$PROJECT"

mkdir -p \
  results/supplementary/S01_harmonization \
  figures/supplementary \
  logs

echo "========================================"
echo " Supplementary Figure S1"
echo "========================================"

echo
echo "[1/2] Harmonization sensitivity"

python \
  src/90_supplementary/S01_harmonization/01_run_sensitivity.py \
  > logs/figureS1_harmonization.log 2>&1

cat \
  logs/figureS1_harmonization.log


echo
echo "[2/2] Generate Figure S1"

python \
  src/90_supplementary/S01_harmonization/02_make_figureS1.py \
  > logs/figureS1_make.log 2>&1

cat \
  logs/figureS1_make.log


echo
echo "========================================"
echo " Figure S1 completed"
echo "========================================"
