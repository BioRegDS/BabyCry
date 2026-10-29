#!/usr/bin/env python3
"""
Figure 4A:
Feature-level importance for dataset identity classification.

Condition
---------
Fixed 0.25 s + CMVN

Task
----
Donate-a-Cry vs Enes

Sampling
--------
Reason-matched and class-balanced:
27 hunger + 27 discomfort from each dataset per repeat.

Outputs
-------
results/05_figure5/
    feature_importance_repeat.csv
    feature_importance_summary.csv
    top_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
N_REPEATS = 100
N_PERM_REPEATS = 20
TEST_SIZE = 0.30
TOP_N = 20

META_COLUMNS = {
    "dataset",
    "file",
    "filename",
    "group_id",
    "file_seq_S",
    "baby",
    "sex",
    "age_month",
    "label",
    "original_sr",
    "original_duration_sec",
    "analysis_duration_sec",
}


def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]


def family_name(feature):

    if feature.startswith("ddmfcc_"):
        return "DeltaDeltaMFCC"

    if feature.startswith("dmfcc_"):
        return "DeltaMFCC"

    if feature.startswith("mfcc_"):
        return "MFCC"

    if feature.startswith("chroma_"):
        return "Chroma"

    if (
        feature.startswith("f0_")
        or feature.startswith("voiced_")
    ):
        return "Pitch_Voicing"

    if feature.startswith("rms_"):
        return "Energy"

    if feature.startswith("onset_"):
        return "Temporal_Onset"

    if (
        feature.startswith("spectral_")
        or feature.startswith("zcr_")
    ):
        return "Spectral"

    return "Other"


def reason_matched_sample(
    donate,
    enes,
    rng,
):

    n_each = min(
        (donate["label"] == "hunger").sum(),
        (donate["label"] == "discomfort").sum(),
        (enes["label"] == "hunger").sum(),
        (enes["label"] == "discomfort").sum(),
    )

    frames = []

    for dataset_name, df in [
        ("donate", donate),
        ("enes", enes),
    ]:

        for label in [
            "hunger",
            "discomfort",
        ]:

            subset = df[
                df["label"] == label
            ]

            idx = rng.choice(
                subset.index.to_numpy(),
                size=n_each,
                replace=False,
            )

            part = subset.loc[
                idx
            ].copy()

            part[
                "dataset_target"
            ] = (
                0
                if dataset_name == "donate"
                else 1
            )

            frames.append(part)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def make_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            str(Path(__file__).resolve().parents[2])
        ),
    )

    args = parser.parse_args()

    project = args.project.resolve()

    input_dir = (
        project
        / "data"
        / "processed"
        / "fixed025"
    )

    outdir = (
        project
        / "results"
        / "05_figure5"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    donate = pd.read_csv(
        input_dir
        / "donate_cmvn.csv"
    )

    enes = pd.read_csv(
        input_dir
        / "enes_cmvn.csv"
    )

    features = sorted(
        set(
            feature_columns(
                donate
            )
        )
        &
        set(
            feature_columns(
                enes
            )
        )
    )

    repeat_rows = []

    for repeat in range(
        N_REPEATS
    ):

        print(
            f"Repeat "
            f"{repeat + 1}/"
            f"{N_REPEATS}",
            flush=True,
        )

        rng = np.random.default_rng(
            SEED + repeat
        )

        matched = (
            reason_matched_sample(
                donate,
                enes,
                rng,
            )
        )

        y = matched[
            "dataset_target"
        ].to_numpy()

        train_idx, test_idx = train_test_split(
            np.arange(
                len(matched)
            ),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=
                SEED
                + 1000
                + repeat,
        )

        X_train = matched.iloc[
            train_idx
        ][features]

        y_train = y[
            train_idx
        ]

        X_test = matched.iloc[
            test_idx
        ][features]

        y_test = y[
            test_idx
        ]

        model = make_model()

        model.fit(
            X_train,
            y_train,
        )

        score = model.decision_function(
            X_test
        )

        auc = float(
            roc_auc_score(
                y_test,
                score,
            )
        )

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=N_PERM_REPEATS,
            random_state=
                SEED
                + 2000
                + repeat,
            n_jobs=-1,
        )

        # standardized logistic regression coefficients
        coef = (
            model.named_steps[
                "model"
            ]
            .coef_[0]
        )

        for feature, perm_imp, coef_value in zip(
            features,
            perm.importances_mean,
            coef,
        ):

            repeat_rows.append(
                {
                    "repeat":
                        repeat,

                    "auc":
                        auc,

                    "feature":
                        feature,

                    "family":
                        family_name(
                            feature
                        ),

                    "permutation_importance":
                        float(
                            perm_imp
                        ),

                    "abs_logreg_coef":
                        float(
                            abs(
                                coef_value
                            )
                        ),

                    "logreg_coef":
                        float(
                            coef_value
                        ),
                }
            )

    repeat_df = pd.DataFrame(
        repeat_rows
    )

    repeat_df.to_csv(
        outdir
        / "feature_importance_repeat.csv",
        index=False,
    )

    summary = (
        repeat_df
        .groupby(
            [
                "feature",
                "family",
            ],
            as_index=False,
        )
        .agg(
            mean_permutation_importance=(
                "permutation_importance",
                "mean",
            ),
            sd_permutation_importance=(
                "permutation_importance",
                "std",
            ),
            mean_abs_logreg_coef=(
                "abs_logreg_coef",
                "mean",
            ),
            mean_logreg_coef=(
                "logreg_coef",
                "mean",
            ),
        )
    )

    # rank each feature using both measures
    summary[
        "rank_perm"
    ] = summary[
        "mean_permutation_importance"
    ].rank(
        ascending=False,
        method="average",
    )

    summary[
        "rank_coef"
    ] = summary[
        "mean_abs_logreg_coef"
    ].rank(
        ascending=False,
        method="average",
    )

    summary[
        "mean_rank"
    ] = (
        summary[
            [
                "rank_perm",
                "rank_coef",
            ]
        ]
        .mean(
            axis=1
        )
    )

    summary = (
        summary
        .sort_values(
            "mean_rank"
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        outdir
        / "feature_importance_summary.csv",
        index=False,
    )

    top = summary.head(
        TOP_N
    ).copy()

    top.to_csv(
        outdir
        / "top_features.csv",
        index=False,
    )

    print(
        "\n===== TOP FEATURES ====="
    )

    print(
        top[
            [
                "feature",
                "family",
                "mean_permutation_importance",
                "mean_abs_logreg_coef",
                "mean_logreg_coef",
                "mean_rank",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        outdir
    )


if __name__ == "__main__":
    main()
