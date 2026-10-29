#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]

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
# Load metadata
# ============================================================

donate = pd.read_csv(
    PROJECT
    / "data"
    / "processed"
    / "donate"
    / "metadata.csv"
)

enes_all = pd.read_csv(
    PROJECT
    / "data"
    / "raw"
    / "enes"
    / "dataset_44605_short.csv"
)

enes_hc = pd.read_csv(
    PROJECT
    / "data"
    / "processed"
    / "enes"
    / "high_confidence_metadata.csv"
)

enes_hd = pd.read_csv(
    PROJECT
    / "data"
    / "processed"
    / "enes"
    / "hunger_discomfort_metadata.csv"
)


# ============================================================
# Detect Donate label column
# ============================================================

candidate_label_columns = [
    "label",
    "class",
    "cry_reason",
]

donate_label_col = None

for col in candidate_label_columns:

    if col in donate.columns:
        donate_label_col = col
        break

if donate_label_col is None:

    raise RuntimeError(
        "Could not identify Donate label column. "
        f"Columns: {donate.columns.tolist()}"
    )


# ============================================================
# Table S1 — label definitions
# ============================================================

rows = []


for label in [
    "hunger",
    "discomfort",
]:

    n = int(
        (
            donate[
                donate_label_col
            ]
            .astype(str)
            .str.lower()
            == label
        ).sum()
    )

    rows.append({
        "Dataset":
            "Donate-a-Cry",

        "Original label / information":
            label.capitalize(),

        "Analysis label":
            label.capitalize(),

        "Inclusion criterion":
            (
                "Original dataset label "
                f"= {label.capitalize()}"
            ),

        "n":
            n,
    })


for label in [
    "hunger",
    "discomfort",
]:

    subset = enes_hd[
        enes_hd[
            "cause_parents_engl"
        ].astype(str).str.lower()
        == label
    ]

    # Strict validation:
    # high-confidence label requires agreement
    if not (
        subset[
            "cause_stop_engl"
        ].astype(str).str.lower()
        == label
    ).all():

        raise RuntimeError(
            f"Enes high-confidence validation failed: {label}"
        )

    rows.append({
        "Dataset":
            "Enes",

        "Original label / information":
            (
                f"cause_parents_engl = {label}; "
                f"cause_stop_engl = {label}"
            ),

        "Analysis label":
            label.capitalize(),

        "Inclusion criterion":
            (
                "Parent-reported cause and "
                "stopping action agreed"
            ),

        "n":
            len(subset),
    })


table = pd.DataFrame(
    rows
)


# ============================================================
# Validation
# ============================================================

expected = {
    (
        "Donate-a-Cry",
        "Hunger",
    ): 382,

    (
        "Donate-a-Cry",
        "Discomfort",
    ): 27,

    (
        "Enes",
        "Hunger",
    ): 9891,

    (
        "Enes",
        "Discomfort",
    ): 9246,
}


for _, row in table.iterrows():

    key = (
        row["Dataset"],
        row["Analysis label"],
    )

    if key in expected:

        observed = int(
            row["n"]
        )

        if observed != expected[key]:

            raise RuntimeError(
                f"{key}: expected "
                f"{expected[key]}, "
                f"observed {observed}"
            )


# ============================================================
# Enes selection summary
# ============================================================

agreement = (
    enes_all[
        "cause_parents_engl"
    ]
    .astype(str)
    .str.lower()
    ==
    enes_all[
        "cause_stop_engl"
    ]
    .astype(str)
    .str.lower()
)

n_both_labels = int(
    (
        enes_all[
            "cause_parents_engl"
        ].notna()
        &
        enes_all[
            "cause_stop_engl"
        ].notna()
    ).sum()
)

agreement_rate = float(
    agreement.mean()
)


enes_summary = pd.DataFrame(
    [
        {
            "Characteristic":
                "Metadata clips",

            "Value":
                len(enes_all),
        },
        {
            "Characteristic":
                "Parent/stop label agreement (%)",

            "Value":
                agreement_rate * 100,
        },
        {
            "Characteristic":
                "High-confidence clips",

            "Value":
                len(enes_hc),
        },
        {
            "Characteristic":
                "High-confidence Hunger/Discomfort clips",

            "Value":
                len(enes_hd),
        },
        {
            "Characteristic":
                "Infants in Hunger/Discomfort analysis",

            "Value":
                enes_hd[
                    "baby"
                ].nunique(),
        },
        {
            "Characteristic":
                "Sessions in Hunger/Discomfort analysis",

            "Value":
                enes_hd[
                    "file_seq_S"
                ].nunique(),
        },
    ]
)


# Infants containing both classes
infant_classes = (
    enes_hd
    .groupby(
        "baby"
    )[
        "cause_parents_engl"
    ]
    .nunique()
)

n_both = int(
    (
        infant_classes
        >= 2
    ).sum()
)

enes_summary.loc[
    len(enes_summary)
] = {
    "Characteristic":
        "Infants with both Hunger and Discomfort",

    "Value":
        n_both,
}


# ============================================================
# Save
# ============================================================

table_path = (
    OUTDIR
    / "TableS1_label_definitions.csv"
)

summary_path = (
    OUTDIR
    / "TableS1_enes_selection_summary.csv"
)


table.to_csv(
    table_path,
    index=False,
)

enes_summary.to_csv(
    summary_path,
    index=False,
)


print(
    "\n===== TABLE S1 ====="
)

print(
    table.to_string(
        index=False
    )
)

print(
    "\n===== ENES SELECTION SUMMARY ====="
)

print(
    enes_summary.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    table_path
)

print(
    "Saved:",
    summary_path
)
