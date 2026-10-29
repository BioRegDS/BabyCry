#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT"

mkdir -p logs

echo "============================================"
echo " BabyCry: 00_dataset pipeline"
echo "============================================"
echo

echo "[1/6] Prepare Donate-a-Cry and Dunstan"
python src/00_dataset/01_prepare_datasets.py \
    --force \
    > logs/01_prepare_datasets.log 2>&1

tail -20 logs/01_prepare_datasets.log

echo
echo "[2/6] QC Enes raw audio"
python src/00_dataset/02_qc_enes_audio.py \
    > logs/02_qc_enes_audio.log 2>&1

tail -20 logs/02_qc_enes_audio.log

echo
echo "[3/6] Prepare Enes high-confidence labels"
python src/00_dataset/03_prepare_enes_labels.py \
    > logs/03_prepare_enes_labels.log 2>&1

tail -30 logs/03_prepare_enes_labels.log

echo
echo "[4/6] Extract variable-duration acoustic features"
python src/00_dataset/04_extract_acoustic_features.py \
    > logs/04_extract_acoustic_features.log 2>&1

tail -30 logs/04_extract_acoustic_features.log

echo
echo "[5/6] Prepare fixed 0.25-s features"
python src/00_dataset/05_prepare_fixed025.py \
    > logs/05_prepare_fixed025.log 2>&1

tail -30 logs/05_prepare_fixed025.log

echo
echo "[6/6] Final dataset validation"
python src/00_dataset/06_validate_dataset.py \
    > logs/06_validate_dataset.log 2>&1

cat logs/06_validate_dataset.log

echo
echo "============================================"
echo " 00_dataset pipeline completed successfully"
echo "============================================"
