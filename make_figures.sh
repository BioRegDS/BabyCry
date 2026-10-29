#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

mkdir -p figures/main figures/supplementary

echo "=== Main figures ==="

python src/01_figure1/01_make_figure1.py
python src/02_figure2/03_make_figure2.py
python src/03_figure3/03_make_figure3.py
python src/04_figure4/04_make_figure4_qc.py
python src/05_figure5/03_make_figure5_group.py
python src/06_figure6/04_make_figure6_qc.py

echo "=== Supplementary figure ==="

python src/90_supplementary/S01_harmonization/02_make_figureS1_qc.py

echo "=== Supplementary tables ==="

python src/90_supplementary/tables/01_make_tableS1.py
python src/90_supplementary/tables/03_make_tableS3.py
python src/90_supplementary/tables/04_make_tableS4_qc.py

echo "All figures and tables generated."
