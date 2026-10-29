#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[3]

RESULTS = (
    PROJECT
    / "results"
    / "02_figure2"
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
# Load primary Figure 2 results
# ============================================================

reason = pd.read_csv(
    RESULTS
    / "reason_generalization_summary.csv"
)

identity = pd.read_csv(
    RESULTS
    / "infant_identity_summary.csv"
)


# ============================================================
# Strict validation
# ============================================================

required_reason = {
    "evaluation",
    "auc",
    "balanced_accuracy",
    "f1",
    "bootstrap_mean",
    "ci95_low",
    "ci95_high",
    "n_boot_valid",
}

missing = (
    required_reason
    - set(reason.columns)
)

if missing:
    raise RuntimeError(
        "Missing reason columns: "
        + ", ".join(sorted(missing))
    )


required_identity = {
    "evaluation",
    "accuracy",
    "macro_f1",
    "chance_accuracy",
    "ci95_low",
    "ci95_high",
}

missing = (
    required_identity
    - set(identity.columns)
)

if missing:
    raise RuntimeError(
        "Missing infant-identity columns: "
        + ", ".join(sorted(missing))
    )


expected_reason = {
    "random",
    "session",
    "infant",
    "loio",
}

if set(
    reason["evaluation"]
) != expected_reason:

    raise RuntimeError(
        "Unexpected cry-reason evaluations: "
        f"{reason['evaluation'].tolist()}"
    )


expected_identity = {
    "random",
    "session",
}

if set(
    identity["evaluation"]
) != expected_identity:

    raise RuntimeError(
        "Unexpected infant-identity evaluations: "
        f"{identity['evaluation'].tolist()}"
    )


# ============================================================
# Display names
# ============================================================

evaluation_names = {
    "random":
        "Random clips",

    "session":
        "New sessions",

    "infant":
        "New infants",

    "loio":
        "Leave-one-infant-out (LOIO)",
}


# ============================================================
# Section A
# Cry-reason classification
# ============================================================

reason_order = [
    "random",
    "session",
    "infant",
    "loio",
]

reason = (
    reason
    .set_index("evaluation")
    .loc[reason_order]
    .reset_index()
)


reason_table = pd.DataFrame({
    "Task":
        "Cry reason (Hunger vs Discomfort)",

    "Evaluation":
        reason[
            "evaluation"
        ].map(
            evaluation_names
        ),

    "ROC-AUC":
        reason[
            "auc"
        ],

    "95% CI lower":
        reason[
            "ci95_low"
        ],

    "95% CI upper":
        reason[
            "ci95_high"
        ],

    "Balanced accuracy":
        reason[
            "balanced_accuracy"
        ],

    "F1":
        reason[
            "f1"
        ],

    "Bootstrap replicates":
        reason[
            "n_boot_valid"
        ].astype(int),
})


# ============================================================
# Section B
# Infant-identity classification
# ============================================================

identity_order = [
    "random",
    "session",
]

identity = (
    identity
    .set_index("evaluation")
    .loc[identity_order]
    .reset_index()
)


identity_table = pd.DataFrame({
    "Task":
        "Infant identity (24 classes)",

    "Evaluation":
        identity[
            "evaluation"
        ].map(
            evaluation_names
        ),

    "Accuracy":
        identity[
            "accuracy"
        ],

    "95% CI lower":
        identity[
            "ci95_low"
        ],

    "95% CI upper":
        identity[
            "ci95_high"
        ],

    "Macro-F1":
        identity[
            "macro_f1"
        ],

    "Chance accuracy":
        identity[
            "chance_accuracy"
        ],
})


# ============================================================
# Publication-friendly tables
# ============================================================

reason_publication = reason_table.copy()

reason_publication[
    "ROC-AUC"
] = reason_publication[
    "ROC-AUC"
].map(
    lambda x: f"{x:.3f}"
)

reason_publication[
    "95% CI"
] = [
    f"{low:.3f}–{high:.3f}"
    for low, high in zip(
        reason_table[
            "95% CI lower"
        ],
        reason_table[
            "95% CI upper"
        ],
    )
]

reason_publication[
    "Balanced accuracy"
] = reason_publication[
    "Balanced accuracy"
].map(
    lambda x: f"{x:.3f}"
)

reason_publication[
    "F1"
] = reason_publication[
    "F1"
].map(
    lambda x: f"{x:.3f}"
)

reason_publication = reason_publication[
    [
        "Task",
        "Evaluation",
        "ROC-AUC",
        "95% CI",
        "Balanced accuracy",
        "F1",
        "Bootstrap replicates",
    ]
]


identity_publication = identity_table.copy()

identity_publication[
    "Accuracy"
] = identity_publication[
    "Accuracy"
].map(
    lambda x: f"{x:.3f}"
)

identity_publication[
    "95% CI"
] = [
    f"{low:.3f}–{high:.3f}"
    for low, high in zip(
        identity_table[
            "95% CI lower"
        ],
        identity_table[
            "95% CI upper"
        ],
    )
]

identity_publication[
    "Macro-F1"
] = identity_publication[
    "Macro-F1"
].map(
    lambda x: f"{x:.3f}"
)

identity_publication[
    "Chance accuracy"
] = identity_publication[
    "Chance accuracy"
].map(
    lambda x: f"{x:.3f}"
)

identity_publication = identity_publication[
    [
        "Task",
        "Evaluation",
        "Accuracy",
        "95% CI",
        "Macro-F1",
        "Chance accuracy",
    ]
]


# ============================================================
# Save raw numerical versions
# ============================================================

reason_table.to_csv(
    OUTDIR
    / "TableS3A_within_enes_reason_raw.csv",
    index=False,
)

identity_table.to_csv(
    OUTDIR
    / "TableS3B_infant_identity_raw.csv",
    index=False,
)


# ============================================================
# Save publication-ready versions
# ============================================================

reason_publication.to_csv(
    OUTDIR
    / "TableS3A_within_enes_reason.csv",
    index=False,
)

identity_publication.to_csv(
    OUTDIR
    / "TableS3B_infant_identity.csv",
    index=False,
)


# ============================================================
# Console output
# ============================================================

print(
    "\n========================================"
)

print(
    " TABLE S3A — CRY-REASON CLASSIFICATION"
)

print(
    "========================================\n"
)

print(
    reason_publication.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    " TABLE S3B — INFANT-IDENTITY CLASSIFICATION"
)

print(
    "========================================\n"
)

print(
    identity_publication.to_string(
        index=False
    )
)


print(
    "\nSaved to:",
    OUTDIR
)
