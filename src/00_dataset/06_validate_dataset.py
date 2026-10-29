#!/usr/bin/env python3
"""
Final integrity validation for all datasets generated in 00_dataset.

This script performs NO model training.

It verifies that the manuscript-analysis datasets are internally
consistent and safe to pass to downstream Figure analyses.

Checks
------
Raw/prepared datasets
- Donate-a-Cry counts
- Dunstan counts
- Enes label structure
- exact duplicate QC
- Enes raw-audio QC

Variable-duration handcrafted features
- expected row counts
- expected feature count = 618
- Donate / Enes feature columns identical
- no forbidden target/metadata leakage
- no +/-Inf
- no completely missing feature

Fixed 0.25-s handcrafted features
- expected row counts
- expected feature count = 617
- Donate / Enes feature columns identical
- Original / CMVN row alignment
- exact analysis duration = 0.25 s
- no forbidden target/metadata leakage
- no +/-Inf
- no completely missing feature

Output
------
data/metadata/final_validation/
    validation_summary.json
    feature_missingness.csv
    validation_checks.csv
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

EXPECTED_DONATE_3CLASS = {
    "hunger": 382,
    "discomfort": 27,
    "tired": 24,
}

EXPECTED_DUNSTAN_3CLASS = {
    "hunger": 72,
    "discomfort": 57,
    "tired": 76,
}

EXPECTED_DONATE_HD = {
    "hunger": 382,
    "discomfort": 27,
}

EXPECTED_ENES_HD = {
    "hunger": 9891,
    "discomfort": 9246,
}

EXPECTED_ENES_ALL_ROWS = 44605
EXPECTED_ENES_HIGH_CONF = 33779

EXPECTED_ENES_INFANTS = 24
EXPECTED_ENES_HD_SESSIONS = 378
EXPECTED_ENES_BOTH_LABEL_INFANTS = 22

EXPECTED_VARIABLE_FEATURES = 618
EXPECTED_FIXED025_FEATURES = 617

FIXED_DURATION_SEC = 0.25


# ============================================================
# Metadata columns that must never be interpreted as features
# ============================================================

METADATA_COLUMNS = {
    "dataset",
    "file",
    "filename",
    "group_id",
    "file_seq_S",
    "baby",
    "sex",
    "age_month",
    "label",
    "high_conf_label",
    "parent_label",
    "stop_label",
    "label_agree",
    "original_label",
    "original_sr",
    "sampling_rate_hz",
    "channels",
    "frames",
    "duration_sec",
    "original_duration_sec",
    "analysis_duration_sec",
    "audio_relpath",
    "source_zip_member",
    "sha256",
    "format",
    "subtype",
    "read_error",
}


FORBIDDEN_FEATURE_NAMES = {
    "y",
    "target",
    "target_label",
    "class",
    "class_id",
    "dataset_id",
    "dataset_label",
    "label",
    "high_conf_label",
    "parent_label",
    "stop_label",
    "baby",
    "infant",
    "infant_id",
    "group_id",
    "file_seq_S",
    "session",
    "session_id",
    "original_sr",
    "sampling_rate_hz",
    "channels",
    "duration_sec",
    "original_duration_sec",
    "analysis_duration_sec",
}


# ============================================================
# Validation recorder
# ============================================================

class Validator:

    def __init__(self):

        self.rows = []

    def check(
        self,
        name: str,
        passed: bool,
        observed=None,
        expected=None,
        detail: str = "",
    ):

        row = {
            "check": name,
            "passed": bool(passed),
            "observed": (
                ""
                if observed is None
                else str(observed)
            ),
            "expected": (
                ""
                if expected is None
                else str(expected)
            ),
            "detail": detail,
        }

        self.rows.append(
            row
        )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{status}] {name}"
        )

        if (
            observed is not None
            or expected is not None
        ):

            print(
                f"       observed={observed} "
                f"expected={expected}"
            )

        if detail:

            print(
                f"       {detail}"
            )

    @property
    def errors(self):

        return [
            x
            for x in self.rows
            if not x[
                "passed"
            ]
        ]


# ============================================================
# Helpers
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    return [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS
    ]


def validate_label_counts(
    validator: Validator,
    name: str,
    df: pd.DataFrame,
    expected: dict,
):

    observed = (
        df[
            "label"
        ]
        .value_counts()
        .to_dict()
    )

    for label, expected_n in (
        expected.items()
    ):

        observed_n = int(
            observed.get(
                label,
                0,
            )
        )

        validator.check(
            f"{name}: {label} count",
            observed_n
            == expected_n,
            observed_n,
            expected_n,
        )


def check_feature_schema(
    validator: Validator,
    dataset_name: str,
    df: pd.DataFrame,
    expected_n_features: int,
):

    features = feature_columns(
        df
    )

    validator.check(
        f"{dataset_name}: feature count",
        len(features)
        == expected_n_features,
        len(features),
        expected_n_features,
    )

    leaked = sorted(
        set(features)
        & FORBIDDEN_FEATURE_NAMES
    )

    validator.check(
        f"{dataset_name}: no forbidden feature columns",
        len(leaked) == 0,
        leaked,
        [],
    )

    return features


def finite_feature_qc(
    dataset_name: str,
    df: pd.DataFrame,
    features: list[str],
    validator: Validator,
):

    missing_rows = []

    inf_features = []
    all_missing = []

    for feature in features:

        x = pd.to_numeric(
            df[
                feature
            ],
            errors="coerce",
        )

        arr = x.to_numpy(
            dtype=float
        )

        n_missing = int(
            np.isnan(
                arr
            ).sum()
        )

        n_inf = int(
            np.isinf(
                arr
            ).sum()
        )

        n_finite = int(
            np.isfinite(
                arr
            ).sum()
        )

        missing_rows.append(
            {
                "dataset":
                    dataset_name,

                "feature":
                    feature,

                "n_rows":
                    len(df),

                "n_missing":
                    n_missing,

                "missing_fraction":
                    (
                        n_missing
                        / len(df)
                        if len(df)
                        else np.nan
                    ),

                "n_inf":
                    n_inf,

                "n_finite":
                    n_finite,
            }
        )

        if n_inf > 0:

            inf_features.append(
                (
                    feature,
                    n_inf,
                )
            )

        if n_finite == 0:

            all_missing.append(
                feature
            )

    validator.check(
        f"{dataset_name}: no infinite feature values",
        len(
            inf_features
        )
        == 0,
        inf_features[:10],
        [],
    )

    validator.check(
        f"{dataset_name}: no completely missing features",
        len(
            all_missing
        )
        == 0,
        all_missing[:10],
        [],
    )

    return missing_rows


def validate_same_feature_set(
    validator: Validator,
    name: str,
    features_a,
    features_b,
):

    a = set(
        features_a
    )

    b = set(
        features_b
    )

    only_a = sorted(
        a - b
    )

    only_b = sorted(
        b - a
    )

    validator.check(
        name,
        a == b,
        (
            f"only_A={only_a[:10]}, "
            f"only_B={only_b[:10]}"
        ),
        "identical feature sets",
    )


def validate_row_alignment(
    validator: Validator,
    name: str,
    original: pd.DataFrame,
    cmvn: pd.DataFrame,
    key_columns: list[str],
):

    validator.check(
        f"{name}: Original/CMVN row count",
        len(original)
        == len(cmvn),
        (
            len(original),
            len(cmvn),
        ),
        "identical",
    )

    if len(original) != len(cmvn):

        return

    existing_keys = [
        c
        for c in key_columns
        if (
            c in original.columns
            and c in cmvn.columns
        )
    ]

    if not existing_keys:

        validator.check(
            f"{name}: row alignment key available",
            False,
            [],
            key_columns,
        )

        return

    left = (
        original[
            existing_keys
        ]
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    right = (
        cmvn[
            existing_keys
        ]
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    same = left.equals(
        right
    )

    validator.check(
        f"{name}: Original/CMVN row alignment",
        same,
        existing_keys,
        "same values in same order",
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

    outdir = (
        project
        / "data"
        / "metadata"
        / "final_validation"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validator = Validator()

    missingness_rows = []

    print(
        "========================================"
    )

    print(
        " BabyCry final dataset validation"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. Donate / Dunstan
    # ========================================================

    print(
        "\n===== 1. DONATE / DUNSTAN ====="
    )

    donate_meta = load_csv(
        project
        / "data"
        / "processed"
        / "donate"
        / "metadata.csv"
    )

    dunstan_meta = load_csv(
        project
        / "data"
        / "processed"
        / "dunstan"
        / "metadata.csv"
    )

    validator.check(
        "Donate total rows",
        len(
            donate_meta
        )
        == 433,
        len(
            donate_meta
        ),
        433,
    )

    validator.check(
        "Dunstan total rows",
        len(
            dunstan_meta
        )
        == 205,
        len(
            dunstan_meta
        ),
        205,
    )

    validate_label_counts(
        validator,
        "Donate",
        donate_meta,
        EXPECTED_DONATE_3CLASS,
    )

    validate_label_counts(
        validator,
        "Dunstan",
        dunstan_meta,
        EXPECTED_DUNSTAN_3CLASS,
    )

    validator.check(
        "Donate unique SHA-256",
        donate_meta[
            "sha256"
        ].nunique()
        == len(
            donate_meta
        ),
        donate_meta[
            "sha256"
        ].nunique(),
        len(
            donate_meta
        ),
    )

    validator.check(
        "Dunstan unique SHA-256",
        dunstan_meta[
            "sha256"
        ].nunique()
        == len(
            dunstan_meta
        ),
        dunstan_meta[
            "sha256"
        ].nunique(),
        len(
            dunstan_meta
        ),
    )

    cross_dup = load_csv(
        project
        / "data"
        / "metadata"
        / "exact_duplicates_cross_dataset.csv"
    )

    validator.check(
        "Donate/Dunstan cross-dataset exact duplicates",
        len(
            cross_dup
        )
        == 0,
        len(
            cross_dup
        ),
        0,
    )

    # ========================================================
    # 2. Enes raw-audio QC
    # ========================================================

    print(
        "\n===== 2. ENES RAW AUDIO ====="
    )

    enes_audio_summary = load_json(
        project
        / "data"
        / "metadata"
        / "enes_audio_qc"
        / "summary.json"
    )

    validator.check(
        "Enes raw-audio QC passed",
        bool(
            enes_audio_summary.get(
                "strict_validation_passed",
                False,
            )
        ),
        enes_audio_summary.get(
            "strict_validation_passed"
        ),
        True,
    )

    validator.check(
        "Enes raw audio files",
        int(
            enes_audio_summary[
                "n_audio_recognized"
            ]
        )
        == 78094,
        enes_audio_summary[
            "n_audio_recognized"
        ],
        78094,
    )

    validator.check(
        "Enes raw audio read errors",
        int(
            enes_audio_summary[
                "n_audio_read_error"
            ]
        )
        == 0,
        enes_audio_summary[
            "n_audio_read_error"
        ],
        0,
    )

    validator.check(
        "Enes raw audio duplicate excess",
        int(
            enes_audio_summary[
                "n_duplicate_files_excess"
            ]
        )
        == 0,
        enes_audio_summary[
            "n_duplicate_files_excess"
        ],
        0,
    )

    # ========================================================
    # 3. Enes labels
    # ========================================================

    print(
        "\n===== 3. ENES LABELS ====="
    )

    enes_label_summary = load_json(
        project
        / "data"
        / "metadata"
        / "enes_labels"
        / "summary.json"
    )

    validator.check(
        "Enes label QC passed",
        bool(
            enes_label_summary.get(
                "strict_validation_passed",
                False,
            )
        ),
        enes_label_summary.get(
            "strict_validation_passed"
        ),
        True,
    )

    validator.check(
        "Enes metadata rows",
        int(
            enes_label_summary[
                "n_rows"
            ]
        )
        == EXPECTED_ENES_ALL_ROWS,
        enes_label_summary[
            "n_rows"
        ],
        EXPECTED_ENES_ALL_ROWS,
    )

    validator.check(
        "Enes high-confidence rows",
        int(
            enes_label_summary[
                "n_high_conf_clips"
            ]
        )
        == EXPECTED_ENES_HIGH_CONF,
        enes_label_summary[
            "n_high_conf_clips"
        ],
        EXPECTED_ENES_HIGH_CONF,
    )

    enes_hd_meta = load_csv(
        project
        / "data"
        / "processed"
        / "enes"
        / "hunger_discomfort_metadata.csv"
    )

    validate_label_counts(
        validator,
        "Enes H/D",
        enes_hd_meta,
        EXPECTED_ENES_HD,
    )

    validator.check(
        "Enes H/D infants",
        enes_hd_meta[
            "baby"
        ].nunique()
        == EXPECTED_ENES_INFANTS,
        enes_hd_meta[
            "baby"
        ].nunique(),
        EXPECTED_ENES_INFANTS,
    )

    validator.check(
        "Enes H/D sessions",
        enes_hd_meta[
            "file_seq_S"
        ].nunique()
        == EXPECTED_ENES_HD_SESSIONS,
        enes_hd_meta[
            "file_seq_S"
        ].nunique(),
        EXPECTED_ENES_HD_SESSIONS,
    )

    hd_presence = pd.crosstab(
        enes_hd_meta[
            "baby"
        ],
        enes_hd_meta[
            "label"
        ],
    )

    for col in [
        "hunger",
        "discomfort",
    ]:

        if col not in hd_presence.columns:

            hd_presence[
                col
            ] = 0

    n_both = int(
        (
            (
                hd_presence[
                    "hunger"
                ]
                > 0
            )
            &
            (
                hd_presence[
                    "discomfort"
                ]
                > 0
            )
        ).sum()
    )

    validator.check(
        "Enes infants with both H/D labels",
        n_both
        == EXPECTED_ENES_BOTH_LABEL_INFANTS,
        n_both,
        EXPECTED_ENES_BOTH_LABEL_INFANTS,
    )

    validator.check(
        "Enes H/D duplicate file rows",
        int(
            enes_hd_meta[
                "file"
            ]
            .duplicated()
            .sum()
        )
        == 0,
        int(
            enes_hd_meta[
                "file"
            ]
            .duplicated()
            .sum()
        ),
        0,
    )

    # ========================================================
    # 4. Variable-duration features
    # ========================================================

    print(
        "\n===== 4. VARIABLE-DURATION FEATURES ====="
    )

    var_donate = load_csv(
        project
        / "data"
        / "processed"
        / "features_variable"
        / "donate_hunger_discomfort.csv"
    )

    var_enes = load_csv(
        project
        / "data"
        / "processed"
        / "features_variable"
        / "enes_hunger_discomfort.csv"
    )

    validator.check(
        "Variable Donate rows",
        len(
            var_donate
        )
        == 409,
        len(
            var_donate
        ),
        409,
    )

    validator.check(
        "Variable Enes rows",
        len(
            var_enes
        )
        == 19137,
        len(
            var_enes
        ),
        19137,
    )

    validate_label_counts(
        validator,
        "Variable Donate",
        var_donate,
        EXPECTED_DONATE_HD,
    )

    validate_label_counts(
        validator,
        "Variable Enes",
        var_enes,
        EXPECTED_ENES_HD,
    )

    var_d_features = (
        check_feature_schema(
            validator,
            "Variable Donate",
            var_donate,
            EXPECTED_VARIABLE_FEATURES,
        )
    )

    var_e_features = (
        check_feature_schema(
            validator,
            "Variable Enes",
            var_enes,
            EXPECTED_VARIABLE_FEATURES,
        )
    )

    validate_same_feature_set(
        validator,
        "Variable Donate/Enes identical feature set",
        var_d_features,
        var_e_features,
    )

    missingness_rows += (
        finite_feature_qc(
            "variable_donate",
            var_donate,
            var_d_features,
            validator,
        )
    )

    missingness_rows += (
        finite_feature_qc(
            "variable_enes",
            var_enes,
            var_e_features,
            validator,
        )
    )

    # ========================================================
    # 5. Fixed 0.25 s features
    # ========================================================

    print(
        "\n===== 5. FIXED 0.25 s FEATURES ====="
    )

    fixed_dir = (
        project
        / "data"
        / "processed"
        / "fixed025"
    )

    fixed_d_orig = load_csv(
        fixed_dir
        / "donate_original.csv"
    )

    fixed_d_cmvn = load_csv(
        fixed_dir
        / "donate_cmvn.csv"
    )

    fixed_e_orig = load_csv(
        fixed_dir
        / "enes_original.csv"
    )

    fixed_e_cmvn = load_csv(
        fixed_dir
        / "enes_cmvn.csv"
    )

    for name, df, expected_n in [
        (
            "Fixed Donate Original",
            fixed_d_orig,
            409,
        ),
        (
            "Fixed Donate CMVN",
            fixed_d_cmvn,
            409,
        ),
        (
            "Fixed Enes Original",
            fixed_e_orig,
            19137,
        ),
        (
            "Fixed Enes CMVN",
            fixed_e_cmvn,
            19137,
        ),
    ]:

        validator.check(
            f"{name}: row count",
            len(df)
            == expected_n,
            len(df),
            expected_n,
        )

        duration = pd.to_numeric(
            df[
                "analysis_duration_sec"
            ],
            errors="coerce",
        )

        duration_ok = (
            np.isclose(
                duration.to_numpy(),
                FIXED_DURATION_SEC,
                atol=1e-12,
                rtol=0,
            )
            .all()
        )

        validator.check(
            f"{name}: exact 0.25 s duration",
            bool(
                duration_ok
            ),
            (
                duration.min(),
                duration.max(),
            ),
            (
                FIXED_DURATION_SEC,
                FIXED_DURATION_SEC,
            ),
        )

    fixed_d_orig_features = (
        check_feature_schema(
            validator,
            "Fixed Donate Original",
            fixed_d_orig,
            EXPECTED_FIXED025_FEATURES,
        )
    )

    fixed_d_cmvn_features = (
        check_feature_schema(
            validator,
            "Fixed Donate CMVN",
            fixed_d_cmvn,
            EXPECTED_FIXED025_FEATURES,
        )
    )

    fixed_e_orig_features = (
        check_feature_schema(
            validator,
            "Fixed Enes Original",
            fixed_e_orig,
            EXPECTED_FIXED025_FEATURES,
        )
    )

    fixed_e_cmvn_features = (
        check_feature_schema(
            validator,
            "Fixed Enes CMVN",
            fixed_e_cmvn,
            EXPECTED_FIXED025_FEATURES,
        )
    )

    validate_same_feature_set(
        validator,
        "Fixed Donate Original/CMVN identical feature set",
        fixed_d_orig_features,
        fixed_d_cmvn_features,
    )

    validate_same_feature_set(
        validator,
        "Fixed Enes Original/CMVN identical feature set",
        fixed_e_orig_features,
        fixed_e_cmvn_features,
    )

    validate_same_feature_set(
        validator,
        "Fixed Donate/Enes identical feature set",
        fixed_d_orig_features,
        fixed_e_orig_features,
    )

    validate_row_alignment(
        validator,
        "Fixed Donate",
        fixed_d_orig,
        fixed_d_cmvn,
        [
            "file",
            "group_id",
            "label",
        ],
    )

    validate_row_alignment(
        validator,
        "Fixed Enes",
        fixed_e_orig,
        fixed_e_cmvn,
        [
            "file",
            "file_seq_S",
            "baby",
            "label",
        ],
    )

    for name, frame, features in [
        (
            "fixed_donate_original",
            fixed_d_orig,
            fixed_d_orig_features,
        ),
        (
            "fixed_donate_cmvn",
            fixed_d_cmvn,
            fixed_d_cmvn_features,
        ),
        (
            "fixed_enes_original",
            fixed_e_orig,
            fixed_e_orig_features,
        ),
        (
            "fixed_enes_cmvn",
            fixed_e_cmvn,
            fixed_e_cmvn_features,
        ),
    ]:

        missingness_rows += (
            finite_feature_qc(
                name,
                frame,
                features,
                validator,
            )
        )

    # ========================================================
    # 6. CMVN sanity check
    # ========================================================

    print(
        "\n===== 6. CMVN SANITY ====="
    )

    cmvn_mean_columns = [
        f"mfcc_{i}_mean"
        for i in range(
            1,
            21,
        )
    ]

    for name, frame in [
        (
            "Donate CMVN",
            fixed_d_cmvn,
        ),
        (
            "Enes CMVN",
            fixed_e_cmvn,
        ),
    ]:

        available = [
            c
            for c
            in cmvn_mean_columns
            if c in frame.columns
        ]

        max_abs_mean = float(
            frame[
                available
            ]
            .abs()
            .to_numpy()
            .max()
        )

        # Floating-point residuals are expected,
        # but the CMVN mean should be effectively zero.
        validator.check(
            f"{name}: MFCC means approximately zero",
            max_abs_mean
            < 1e-4,
            max_abs_mean,
            "< 1e-4",
        )

    # ========================================================
    # 7. Save QC reports
    # ========================================================

    checks = pd.DataFrame(
        validator.rows
    )

    checks.to_csv(
        outdir
        / "validation_checks.csv",
        index=False,
    )

    missingness = pd.DataFrame(
        missingness_rows
    )

    missingness.to_csv(
        outdir
        / "feature_missingness.csv",
        index=False,
    )

    failed_checks = (
        checks[
            ~checks[
                "passed"
            ]
        ]
    )

    summary = {
        "n_checks":
            int(
                len(
                    checks
                )
            ),

        "n_passed":
            int(
                checks[
                    "passed"
                ].sum()
            ),

        "n_failed":
            int(
                len(
                    failed_checks
                )
            ),

        "overall_validation_passed":
            len(
                failed_checks
            )
            == 0,

        "variable_feature_count":
            int(
                len(
                    var_d_features
                )
            ),

        "fixed025_feature_count":
            int(
                len(
                    fixed_d_orig_features
                )
            ),

        "donate_total":
            int(
                len(
                    donate_meta
                )
            ),

        "dunstan_total":
            int(
                len(
                    dunstan_meta
                )
            ),

        "enes_hd_total":
            int(
                len(
                    enes_hd_meta
                )
            ),

        "enes_hd_infants":
            int(
                enes_hd_meta[
                    "baby"
                ].nunique()
            ),

        "enes_hd_sessions":
            int(
                enes_hd_meta[
                    "file_seq_S"
                ].nunique()
            ),

        "failed_checks":
            failed_checks[
                "check"
            ].tolist(),
    }

    with (
        outdir
        / "validation_summary.json"
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
    # Final report
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " FINAL VALIDATION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if len(
        missingness
    ):

        print(
            "\n===== HIGHEST FEATURE MISSINGNESS ====="
        )

        print(
            missingness
            .sort_values(
                "missing_fraction",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    if validator.errors:

        print(
            "\n*** DATASET VALIDATION FAILED ***"
        )

        print(
            "\nFailed checks:"
        )

        for row in (
            validator.errors
        ):

            print(
                "-",
                row[
                    "check"
                ],
            )

        raise RuntimeError(
            "Final BabyCry dataset "
            "validation failed."
        )

    print(
        "\n========================================"
    )

    print(
        " 00_DATASET QC: PASS"
    )

    print(
        "========================================"
    )

    print(
        "\nThe dataset-generation pipeline is "
        "ready for downstream manuscript analyses."
    )


if __name__ == "__main__":
    main()
