#!/usr/bin/env python3
"""
Prepare manuscript analysis labels from the official Enes metadata.

High-confidence label definition
--------------------------------
A clip is considered high-confidence when:

    parent-reported cause == stop-action cause

using the English label columns in the official metadata.

Primary manuscript subset
-------------------------
High-confidence:
    hunger
    discomfort

This script also validates infant/session structure and expected counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Expected manuscript dataset structure
# ============================================================

EXPECTED = {
    "n_rows": 44605,
    "n_unique_files": 44605,
    "n_sessions": 674,
    "n_infants": 24,
    "n_inconsistent_sessions": 0,

    "n_high_conf_clips": 33779,

    "high_conf_counts": {
        "hunger": 9891,
        "discomfort": 9246,
        "loneliness": 9623,
        "dk": 4853,
        "pain": 95,
        "other": 71,
    },

    "n_hunger_discomfort": 19137,
    "n_hd_infants": 24,
    "n_hd_infants_with_both": 22,
}


REQUIRED_COLUMNS = [
    "file",
    "file_seq_S",
    "baby",
    "sex",
    "age_month",
    "cause_parents",
    "cause_parents_engl",
    "cause_stop",
    "cause_stop_engl",
]


# ============================================================
# Utilities
# ============================================================

def normalize_text(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def count_table(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:

    return (
        df
        .groupby(
            label_col,
            dropna=False,
        )
        .agg(
            n_clips=(
                "file",
                "size",
            ),

            n_sessions=(
                "file_seq_S",
                "nunique",
            ),

            n_infants=(
                "baby",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "n_clips",
            ascending=False,
        )
    )


def validate_equal(
    name,
    observed,
    expected,
    errors,
):

    status = (
        "OK"
        if observed == expected
        else "ERROR"
    )

    print(
        f"{name:40s} "
        f"observed={observed} "
        f"expected={expected} "
        f"{status}"
    )

    if observed != expected:

        errors.append(
            f"{name}: "
            f"observed={observed}, "
            f"expected={expected}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        formatter_class=
        argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            str(Path(__file__).resolve().parents[2])
        ),
    )

    args = parser.parse_args()

    project = (
        args.project
        .resolve()
    )

    metadata_path = (
        project
        / "data"
        / "raw"
        / "enes"
        / "dataset_44605_short.csv"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            "Enes metadata not found:\n"
            f"{metadata_path}"
        )

    outdir = (
        project
        / "data"
        / "processed"
        / "enes"
    )

    qcdir = (
        project
        / "data"
        / "metadata"
        / "enes_labels"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qcdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read metadata
    # ========================================================

    df = pd.read_csv(
        metadata_path
    )

    print(
        "Metadata:",
        metadata_path
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required metadata columns:\n"
            + "\n".join(
                missing
            )
        )

    # ========================================================
    # Normalize metadata
    # ========================================================

    for col in [
        "baby",
        "sex",
        "cause_parents",
        "cause_parents_engl",
        "cause_stop",
        "cause_stop_engl",
    ]:

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    df[
        "parent_label"
    ] = normalize_text(
        df[
            "cause_parents_engl"
        ]
    )

    df[
        "stop_label"
    ] = normalize_text(
        df[
            "cause_stop_engl"
        ]
    )

    df[
        "label_agree"
    ] = (
        df[
            "parent_label"
        ].notna()
        &
        df[
            "stop_label"
        ].notna()
        &
        (
            df[
                "parent_label"
            ]
            ==
            df[
                "stop_label"
            ]
        )
    )

    df[
        "high_conf_label"
    ] = np.where(
        df[
            "label_agree"
        ],
        df[
            "parent_label"
        ],
        pd.NA,
    )

    # ========================================================
    # Metadata integrity
    # ========================================================

    duplicate_file_rows = df[
        df.duplicated(
            "file",
            keep=False,
        )
    ].copy()

    duplicate_file_rows.to_csv(
        qcdir
        / "duplicate_metadata_file_rows.csv",
        index=False,
    )

    # Session integrity
    session_structure = (
        df
        .groupby(
            "file_seq_S"
        )
        .agg(
            n_clips=(
                "file",
                "size",
            ),

            n_infants=(
                "baby",
                "nunique",
            ),

            n_parent_labels=(
                "parent_label",
                "nunique",
            ),

            n_stop_labels=(
                "stop_label",
                "nunique",
            ),

            infant=(
                "baby",
                "first",
            ),

            parent_label=(
                "parent_label",
                "first",
            ),

            stop_label=(
                "stop_label",
                "first",
            ),
        )
        .reset_index()
    )

    session_structure.to_csv(
        qcdir
        / "session_structure.csv",
        index=False,
    )

    inconsistent_sessions = (
        session_structure[
            (
                session_structure[
                    "n_infants"
                ]
                > 1
            )
            |
            (
                session_structure[
                    "n_parent_labels"
                ]
                > 1
            )
            |
            (
                session_structure[
                    "n_stop_labels"
                ]
                > 1
            )
        ]
        .copy()
    )

    inconsistent_sessions.to_csv(
        qcdir
        / "session_inconsistencies.csv",
        index=False,
    )

    # ========================================================
    # Label tables
    # ========================================================

    parent_counts = count_table(
        df,
        "parent_label",
    )

    stop_counts = count_table(
        df,
        "stop_label",
    )

    high_conf = df[
        df[
            "label_agree"
        ]
    ].copy()

    high_conf_counts = count_table(
        high_conf,
        "high_conf_label",
    )

    parent_counts.to_csv(
        qcdir
        / "counts_parent_label.csv",
        index=False,
    )

    stop_counts.to_csv(
        qcdir
        / "counts_stop_label.csv",
        index=False,
    )

    high_conf_counts.to_csv(
        qcdir
        / "counts_high_conf_label.csv",
        index=False,
    )

    # Agreement table
    agreement = pd.crosstab(
        df[
            "parent_label"
        ],
        df[
            "stop_label"
        ],
        margins=True,
        dropna=False,
    )

    agreement.to_csv(
        qcdir
        / "parent_vs_stop_crosstab.csv"
    )

    # ========================================================
    # Infant-level structure
    # ========================================================

    infant_summary = (
        df
        .groupby(
            "baby"
        )
        .agg(
            sex=(
                "sex",
                "first",
            ),

            age_month_min=(
                "age_month",
                "min",
            ),

            age_month_max=(
                "age_month",
                "max",
            ),

            n_clips=(
                "file",
                "size",
            ),

            n_sessions=(
                "file_seq_S",
                "nunique",
            ),

            n_high_conf_clips=(
                "label_agree",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "n_clips",
            ascending=False,
        )
    )

    infant_summary.to_csv(
        qcdir
        / "infant_summary.csv",
        index=False,
    )

    # ========================================================
    # Manuscript Hunger vs Discomfort subset
    # ========================================================

    hd = high_conf[
        high_conf[
            "high_conf_label"
        ].isin(
            [
                "hunger",
                "discomfort",
            ]
        )
    ].copy()

    hd = hd.reset_index(
        drop=True
    )

    hd[
        "label"
    ] = hd[
        "high_conf_label"
    ]

    # Infant x label presence
    hd_by_infant = pd.crosstab(
        hd[
            "baby"
        ],
        hd[
            "label"
        ],
    )

    for label in [
        "hunger",
        "discomfort",
    ]:

        if label not in hd_by_infant.columns:

            hd_by_infant[
                label
            ] = 0

    hd_by_infant[
        "has_both"
    ] = (
        (
            hd_by_infant[
                "hunger"
            ]
            > 0
        )
        &
        (
            hd_by_infant[
                "discomfort"
            ]
            > 0
        )
    )

    hd_by_infant.to_csv(
        qcdir
        / "hunger_discomfort_by_infant.csv"
    )

    # Session-level table
    hd_sessions = (
        hd
        .groupby(
            [
                "file_seq_S",
                "baby",
                "label",
            ]
        )
        .agg(
            n_clips=(
                "file",
                "size",
            ),

            age_month=(
                "age_month",
                "median",
            ),

            sex=(
                "sex",
                "first",
            ),
        )
        .reset_index()
    )

    hd_sessions.to_csv(
        qcdir
        / "hunger_discomfort_sessions.csv",
        index=False,
    )

    # ========================================================
    # Save analysis-ready metadata
    # ========================================================

    high_conf.to_csv(
        outdir
        / "high_confidence_metadata.csv",
        index=False,
    )

    hd.to_csv(
        outdir
        / "hunger_discomfort_metadata.csv",
        index=False,
    )

    # ========================================================
    # Strict validation
    # ========================================================

    errors = []

    print(
        "\n===== STRICT ENES LABEL VALIDATION ====="
    )

    validate_equal(
        "Total metadata rows",
        len(df),
        EXPECTED[
            "n_rows"
        ],
        errors,
    )

    validate_equal(
        "Unique files",
        df[
            "file"
        ].nunique(),
        EXPECTED[
            "n_unique_files"
        ],
        errors,
    )

    validate_equal(
        "Sessions",
        df[
            "file_seq_S"
        ].nunique(),
        EXPECTED[
            "n_sessions"
        ],
        errors,
    )

    validate_equal(
        "Infants",
        df[
            "baby"
        ].nunique(),
        EXPECTED[
            "n_infants"
        ],
        errors,
    )

    validate_equal(
        "Inconsistent sessions",
        len(
            inconsistent_sessions
        ),
        EXPECTED[
            "n_inconsistent_sessions"
        ],
        errors,
    )

    validate_equal(
        "High-confidence clips",
        len(
            high_conf
        ),
        EXPECTED[
            "n_high_conf_clips"
        ],
        errors,
    )

    observed_hc_counts = (
        high_conf[
            "high_conf_label"
        ]
        .value_counts()
        .to_dict()
    )

    for (
        label,
        expected_n,
    ) in EXPECTED[
        "high_conf_counts"
    ].items():

        validate_equal(
            f"High-confidence {label}",
            int(
                observed_hc_counts.get(
                    label,
                    0,
                )
            ),
            expected_n,
            errors,
        )

    validate_equal(
        "Hunger/Discomfort clips",
        len(
            hd
        ),
        EXPECTED[
            "n_hunger_discomfort"
        ],
        errors,
    )

    validate_equal(
        "H/D infants",
        hd[
            "baby"
        ].nunique(),
        EXPECTED[
            "n_hd_infants"
        ],
        errors,
    )

    validate_equal(
        "H/D infants with both labels",
        int(
            hd_by_infant[
                "has_both"
            ].sum()
        ),
        EXPECTED[
            "n_hd_infants_with_both"
        ],
        errors,
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    both_labels = df[
        df[
            "parent_label"
        ].notna()
        &
        df[
            "stop_label"
        ].notna()
    ]

    agreement_rate = (
        float(
            (
                both_labels[
                    "parent_label"
                ]
                ==
                both_labels[
                    "stop_label"
                ]
            ).mean()
        )
        if len(
            both_labels
        )
        else np.nan
    )

    summary = {
        "metadata_path":
            str(
                metadata_path.relative_to(
                    project
                )
            ),

        "n_rows":
            int(
                len(df)
            ),

        "n_unique_files":
            int(
                df[
                    "file"
                ].nunique()
            ),

        "n_duplicate_file_rows":
            int(
                len(
                    duplicate_file_rows
                )
            ),

        "n_sessions":
            int(
                df[
                    "file_seq_S"
                ].nunique()
            ),

        "n_infants":
            int(
                df[
                    "baby"
                ].nunique()
            ),

        "n_inconsistent_sessions":
            int(
                len(
                    inconsistent_sessions
                )
            ),

        "parent_stop_exact_agreement_rate":
            agreement_rate,

        "n_high_conf_clips":
            int(
                len(
                    high_conf
                )
            ),

        "n_hunger_discomfort_high_conf_clips":
            int(
                len(
                    hd
                )
            ),

        "n_hd_infants":
            int(
                hd[
                    "baby"
                ].nunique()
            ),

        "n_hd_infants_with_both":
            int(
                hd_by_infant[
                    "has_both"
                ].sum()
            ),

        "strict_validation_passed":
            len(
                errors
            )
            == 0,

        "validation_errors":
            errors,
    }

    with (
        qcdir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Console report
    # ========================================================

    print(
        "\n===== HIGH-CONFIDENCE LABEL COUNTS ====="
    )

    print(
        high_conf_counts.to_string(
            index=False
        )
    )

    print(
        "\n===== HUNGER vs DISCOMFORT ====="
    )

    print(
        hd[
            "label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nInfants:",
        hd[
            "baby"
        ].nunique(),
    )

    print(
        "Infants with both labels:",
        int(
            hd_by_infant[
                "has_both"
            ].sum()
        ),
    )

    print(
        "Sessions:",
        hd[
            "file_seq_S"
        ].nunique(),
    )

    if errors:

        print(
            "\n===== VALIDATION FAILED ====="
        )

        for error in errors:

            print(
                "-",
                error
            )

        raise RuntimeError(
            "Enes label preparation failed strict validation."
        )

    print(
        "\n03_prepare_enes_labels.py "
        "completed successfully."
    )

    print(
        "\nSaved analysis metadata:"
    )

    print(
        outdir
        / "high_confidence_metadata.csv"
    )

    print(
        outdir
        / "hunger_discomfort_metadata.csv"
    )


if __name__ == "__main__":
    main()
