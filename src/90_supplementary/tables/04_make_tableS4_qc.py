#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[3]

RESULTS2 = (
    PROJECT
    / "results"
    / "03_figure3"
)

RESULTS3 = (
    PROJECT
    / "results"
    / "04_figure4"
)

RESULTS5 = (
    PROJECT
    / "results"
    / "06_figure6"
)

RESULTS_S1 = (
    PROJECT
    / "results"
    / "supplementary"
    / "S01_harmonization_qc"
)

OUTDIR = (
    PROJECT
    / "results"
    / "supplementary"
    / "tables"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def fmt3(x):

    if pd.isna(x):
        return ""

    return f"{float(x):.3f}"


def fmt_ci(low, high):

    if pd.isna(low) or pd.isna(high):
        return ""

    return (
        f"{float(low):.3f}"
        f"–"
        f"{float(high):.3f}"
    )


def fmt_p(x):

    if pd.isna(x):
        return ""

    x = float(x)

    if x < 0.001:
        return "<0.001"

    return f"{x:.3f}"


def pretty_direction(x):

    mapping = {
        "donate_to_enes":
            "Donate-a-Cry → Enes",

        "enes_to_donate":
            "Enes → Donate-a-Cry",
    }

    return mapping.get(
        x,
        x,
    )


def pretty_model(x):

    mapping = {
        "linear_svc":
            "Linear SVC",

        "logreg":
            "Logistic regression",
    }

    return mapping.get(
        x,
        x,
    )


def pretty_condition(x):

    mapping = {
        "original":
            "Fixed 0.25 s",

        "cmvn":
            "Fixed 0.25 s + CMVN",

        "variable":
            "Variable duration",

        "fixed_original":
            "Fixed 0.25 s",

        "fixed_cmvn":
            "Fixed 0.25 s + CMVN",
    }

    return mapping.get(
        x,
        x,
    )


# ============================================================
# TABLE S4A
# Primary cross-dataset transfer
# ============================================================

cluster = pd.read_csv(
    RESULTS2
    / "cluster_inference_summary.csv"
)


required_cluster = {
    "condition",
    "direction",
    "model",
    "clip_level_auc",
    "ci95_low",
    "ci95_high",
    "permutation_p_two_sided",
    "session_level_auc",
    "group_level_auc",
    "n_infants_auc_defined",
    "median_infant_auc",
}

missing = (
    required_cluster
    - set(cluster.columns)
)

if missing:

    raise RuntimeError(
        "Missing columns in "
        "cluster_inference_summary.csv: "
        + ", ".join(
            sorted(missing)
        )
    )


condition_order = {
    "original": 0,
    "cmvn": 1,
}

direction_order = {
    "donate_to_enes": 0,
    "enes_to_donate": 1,
}

model_order = {
    "linear_svc": 0,
    "logreg": 1,
}


cluster["_condition_order"] = (
    cluster["condition"]
    .map(condition_order)
)

cluster["_direction_order"] = (
    cluster["direction"]
    .map(direction_order)
)

cluster["_model_order"] = (
    cluster["model"]
    .map(model_order)
)

cluster = (
    cluster
    .sort_values(
        [
            "_condition_order",
            "_direction_order",
            "_model_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


rows = []

for _, row in cluster.iterrows():

    direction = row[
        "direction"
    ]

    if direction == "donate_to_enes":

        aggregate_auc = row[
            "session_level_auc"
        ]

        aggregate_label = (
            "Session-level ROC-AUC"
        )

        infant_median = row[
            "median_infant_auc"
        ]

        n_infants = row[
            "n_infants_auc_defined"
        ]

    else:

        aggregate_auc = row[
            "group_level_auc"
        ]

        aggregate_label = (
            "Group-level ROC-AUC"
        )

        infant_median = np.nan
        n_infants = np.nan

    rows.append({
        "Preprocessing":
            pretty_condition(
                row["condition"]
            ),

        "Direction":
            pretty_direction(
                direction
            ),

        "Classifier":
            pretty_model(
                row["model"]
            ),

        "Clip-level ROC-AUC":
            fmt3(
                row[
                    "clip_level_auc"
                ]
            ),

        "95% cluster-bootstrap CI":
            fmt_ci(
                row[
                    "ci95_low"
                ],
                row[
                    "ci95_high"
                ],
            ),

        "Aggregate metric":
            aggregate_label,

        "Aggregate ROC-AUC":
            fmt3(
                aggregate_auc
            ),

        "Permutation P":
            fmt_p(
                row[
                    "permutation_p_two_sided"
                ]
            ),

        "Median infant-specific ROC-AUC":
            fmt3(
                infant_median
            ),

        "Infants with defined ROC-AUC":
            (
                ""
                if pd.isna(
                    n_infants
                )
                else str(
                    int(
                        n_infants
                    )
                )
            ),
    })


table_s4a = pd.DataFrame(
    rows
)


table_s4a.to_csv(
    OUTDIR
    / "TableS4A_cross_dataset_transfer.csv",
    index=False,
)


# ============================================================
# TABLE S4B
# Harmonization / preprocessing sensitivity
# ============================================================

harm = pd.read_csv(
    RESULTS_S1
    / "harmonization_sensitivity.csv"
)


required_harm = {
    "condition",
    "model",
    "direction",
    "auc",
    "ci95_low",
    "ci95_high",
    "n_features",
}

missing = (
    required_harm
    - set(harm.columns)
)

if missing:

    raise RuntimeError(
        "Missing columns in "
        "harmonization_sensitivity.csv: "
        + ", ".join(
            sorted(missing)
        )
    )


harm_condition_order = {
    "variable": 0,
    "fixed_original": 1,
    "fixed_cmvn": 2,
}

harm["_condition_order"] = (
    harm[
        "condition"
    ].map(
        harm_condition_order
    )
)

harm["_direction_order"] = (
    harm[
        "direction"
    ].map(
        direction_order
    )
)

harm["_model_order"] = (
    harm[
        "model"
    ].map(
        model_order
    )
)

harm = (
    harm
    .sort_values(
        [
            "_condition_order",
            "_direction_order",
            "_model_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


rows = []

for _, row in harm.iterrows():

    rows.append({
        "Preprocessing":
            pretty_condition(
                row[
                    "condition"
                ]
            ),

        "Direction":
            pretty_direction(
                row[
                    "direction"
                ]
            ),

        "Classifier":
            pretty_model(
                row[
                    "model"
                ]
            ),

        "No. features":
            int(
                row[
                    "n_features"
                ]
            ),

        "ROC-AUC":
            fmt3(
                row[
                    "auc"
                ]
            ),

        "95% cluster-bootstrap CI":
            fmt_ci(
                row[
                    "ci95_low"
                ],
                row[
                    "ci95_high"
                ],
            ),
    })


table_s4b = pd.DataFrame(
    rows
)


table_s4b.to_csv(
    OUTDIR
    / "TableS4B_harmonization_sensitivity.csv",
    index=False,
)


# ============================================================
# TABLE S4C
# Representation robustness
# ============================================================

w2v_reason = pd.read_csv(
    RESULTS5
    / "single"
    / "wav2vec2_reason_generalization.csv"
)

w2v_identity = pd.read_csv(
    RESULTS5
    / "single"
    / "groupcv"
    / "dataset_identity_groupcv_summary.csv"
)

hc_identity = pd.read_csv(
    RESULTS3
    / "groupcv_qc"
    / "dataset_identity_groupcv_summary.csv"
)


# ------------------------------------------------------------
# C1: wav2vec2 within-Enes generalization
# ------------------------------------------------------------

eval_order = {
    "random": 0,
    "session": 1,
    "infant": 2,
    "loio": 3,
}

eval_names = {
    "random":
        "Random clips",

    "session":
        "New sessions",

    "infant":
        "New infants",

    "loio":
        "Leave-one-infant-out (LOIO)",
}


w2v_reason["_eval_order"] = (
    w2v_reason[
        "evaluation"
    ].map(
        eval_order
    )
)

w2v_reason["_model_order"] = (
    w2v_reason[
        "model"
    ].map(
        model_order
    )
)

w2v_reason = (
    w2v_reason
    .sort_values(
        [
            "_model_order",
            "_eval_order",
        ]
    )
)


reason_rows = []

for _, row in w2v_reason.iterrows():

    reason_rows.append({
        "Analysis":
            "Within-Enes cry-reason classification",

        "Representation":
            "wav2vec2",

        "Evaluation":
            eval_names[
                row[
                    "evaluation"
                ]
            ],

        "Classifier":
            pretty_model(
                row[
                    "model"
                ]
            ),

        "ROC-AUC":
            fmt3(
                row[
                    "auc"
                ]
            ),

        "95% CI":
            fmt_ci(
                row[
                    "ci95_low"
                ],
                row[
                    "ci95_high"
                ],
            ),
    })


# ------------------------------------------------------------
# C2: Dataset identity
# ------------------------------------------------------------

identity_rows = []


# Handcrafted + CMVN
hc_subset = hc_identity[
    hc_identity[
        "condition"
    ]
    == "cmvn"
].copy()

hc_subset["_model_order"] = (
    hc_subset[
        "model"
    ].map(
        model_order
    )
)

hc_subset = hc_subset.sort_values(
    "_model_order"
)

for _, row in hc_subset.iterrows():

    identity_rows.append({
        "Analysis":
            "Dataset identity",

        "Representation":
            "Handcrafted + CMVN",

        "Evaluation":
            "Donate-a-Cry vs Enes",

        "Classifier":
            pretty_model(
                row[
                    "model"
                ]
            ),

        "ROC-AUC":
            fmt3(
                row[
                    "mean_auc"
                ]
            ),

        "95% CI": "",

        "SD across repeats": fmt3(row["sd_auc"]),
    })


# wav2vec2
w2v_identity["_model_order"] = (
    w2v_identity[
        "model"
    ].map(
        model_order
    )
)

w2v_identity = (
    w2v_identity
    .sort_values(
        "_model_order"
    )
)

for _, row in w2v_identity.iterrows():

    identity_rows.append({
        "Analysis":
            "Dataset identity",

        "Representation":
            "wav2vec2",

        "Evaluation":
            "Donate-a-Cry vs Enes",

        "Classifier":
            pretty_model(
                row[
                    "model"
                ]
            ),

        "ROC-AUC":
            fmt3(
                row[
                    "mean_auc"
                ]
            ),

        "95% CI": "",

        "SD across repeats": fmt3(row["sd_auc"]),
    })


table_s4c = pd.DataFrame(
    reason_rows
    + identity_rows
)


table_s4c.to_csv(
    OUTDIR
    / "TableS4C_representation_robustness.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print(
    "\n========================================"
)

print(
    " TABLE S4A — CROSS-DATASET TRANSFER"
)

print(
    "========================================\n"
)

print(
    table_s4a.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    " TABLE S4B — HARMONIZATION SENSITIVITY"
)

print(
    "========================================\n"
)

print(
    table_s4b.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    " TABLE S4C — REPRESENTATION ROBUSTNESS"
)

print(
    "========================================\n"
)

print(
    table_s4c.to_string(
        index=False
    )
)


print(
    "\nSaved to:",
    OUTDIR
)
