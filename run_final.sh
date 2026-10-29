#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

mkdir -p logs

run() {
    local label="$1"
    shift
    echo "Running: $label"
    "$@" > "logs/${label}.log" 2>&1 || {
        tail -60 "logs/${label}.log"
        return 1
    }
    tail -5 "logs/${label}.log"
}

echo "=== Figure 2 ==="
run fig1_reason python src/02_figure2/01_generalization_hierarchy.py
run fig1_identity python src/02_figure2/02_infant_identity.py

echo "=== Figure 3 ==="
run fig2_transfer python src/03_figure3/01_cross_dataset_reason.py
run fig2_inference python src/03_figure3/02_cluster_inference.py

echo "=== Figure 4: group-aware ==="
run fig3_identity python src/04_figure4/01_dataset_identity_groupcv.py --repeats 200
run fig3_families python src/04_figure4/02_feature_family_groupcv_all.py --repeats 200
run fig3_comparison python src/04_figure4/03_reason_vs_dataset_qc.py

echo "=== Figure 5: group-aware ==="
run fig4_importance python src/05_figure5/01_feature_importance_group.py
run fig4_distributions python src/05_figure5/02_feature_distributions_matched.py

echo "=== Figure 6: single-audio wav2vec2 ==="
run fig5_embeddings python src/06_figure6/05_extract_wav2vec2_single.py --dataset both
run fig5_reason python src/06_figure6/single/02_reason_generalization.py
run fig5_identity python src/06_figure6/single/04_dataset_identity_groupcv.py

echo "=== Supplementary Figure S1 ==="
run supp_harmonization python src/90_supplementary/S01_harmonization/01_run_sensitivity_qc.py

echo "All final analyses completed."
