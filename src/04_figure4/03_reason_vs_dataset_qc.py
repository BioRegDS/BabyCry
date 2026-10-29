#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


PROJECT = Path(
    str(Path(__file__).resolve().parents[2])
)

OUTDIR = (
    PROJECT
    / "results"
    / "04_figure4"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Cry-reason transfer
# ============================================================

reason = pd.read_csv(
    PROJECT
    / "results"
    / "03_figure3"
    / "cluster_inference_summary.csv"
)

# Main comparison:
# CMVN + Linear SVC
reason_main = reason[
    (
        reason[
            "condition"
        ]
        == "cmvn"
    )
    &
    (
        reason[
            "model"
        ]
        == "linear_svc"
    )
].copy()


# ============================================================
# Dataset identity
# ============================================================

identity = pd.read_csv(
    OUTDIR
    / "groupcv_qc"
    / "dataset_identity_groupcv_summary.csv"
)

identity_main = identity[
    (
        identity[
            "condition"
        ]
        == "cmvn"
    )
    &
    (
        identity[
            "model"
        ]
        == "linear_svc"
    )
].copy()


rows = []

for _, row in (
    reason_main.iterrows()
):

    if (
        row[
            "direction"
        ]
        == "donate_to_enes"
    ):

        label = (
            "Reason transfer\nDonate → Enes"
        )

    else:

        label = (
            "Reason transfer\nEnes → Donate"
        )

    rows.append(
        {
            "analysis":
                label,

            "auc":
                row[
                    "clip_level_auc"
                ],

            "ci95_low":
                row[
                    "ci95_low"
                ],

            "ci95_high":
                row[
                    "ci95_high"
                ],
        }
    )


row = identity_main.iloc[0]

rows.append(
    {
        "analysis":
            "Dataset identity\nDonate vs Enes",

        "auc":
            row[
                "mean_auc"
            ],

        "ci95_low": row["mean_auc"] - row["sd_auc"],

        "ci95_high": row["mean_auc"] + row["sd_auc"],
    }
)


comparison = pd.DataFrame(
    rows
)

comparison.to_csv(
    OUTDIR
    / "reason_vs_dataset_summary_qc.csv",
    index=False,
)

print(
    comparison.to_string(
        index=False
    )
)
