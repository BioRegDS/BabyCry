#!/usr/bin/env python3
"""
Figure 3A:
Classify dataset identity (Donate-a-Cry vs Enes)
using fixed 0.25-s acoustic features.

Conditions
----------
1. original
2. cmvn

Models
------
1. Linear SVC
2. Logistic regression

Sampling
--------
Reason-matched and class-balanced:
for each repeat, sample equal numbers of
hunger/discomfort from each dataset.

This prevents dataset identity from being
explained merely by label imbalance.

Outputs
-------
results/04_figure4/
    dataset_identity_summary.csv
    dataset_identity_repeat_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
N_REPEATS = 200
N_SPLITS = 5

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

EXPECTED_FEATURES = 617


def feature_columns(df):

    return [
        c for c in df.columns
        if c not in META_COLUMNS
    ]


def make_model(model_name, seed):

    if model_name == "linear_svc":

        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=seed,
        )

    elif model_name == "logreg":

        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )

    else:

        raise ValueError(model_name)

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
                clf,
            ),
        ]
    )


def reason_matched_sample(
    donate,
    enes,
    rng,
):
    """
    Build equal-sized matched subsets.

    Donate has:
        hunger 382
        discomfort 27

    Therefore use 27 per reason per dataset.
    """

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

            sampled_idx = rng.choice(
                subset.index.to_numpy(),
                size=n_each,
                replace=False,
            )

            part = subset.loc[
                sampled_idx
            ].copy()

            part[
                "dataset_target"
            ] = (
                0
                if dataset_name == "donate"
                else 1
            )

            frames.append(part)

    matched = pd.concat(
        frames,
        ignore_index=True,
    )

    return matched, int(n_each)


def crossvalidated_auc(
    df,
    features,
    model_name,
    seed,
):

    y = df[
        "dataset_target"
    ].to_numpy()

    scores = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            y,
        ),
        start=1,
    ):

        model = make_model(
            model_name,
            seed + fold,
        )

        model.fit(
            df.iloc[
                train_idx
            ][features],
            y[
                train_idx
            ],
        )

        score = model.decision_function(
            df.iloc[
                test_idx
            ][features]
        )

        scores[
            test_idx
        ] = score

    if np.isnan(scores).any():
        raise RuntimeError(
            "Missing OOF scores."
        )

    return float(
        roc_auc_score(
            y,
            scores,
        )
    )


def summarize(
    repeat_results,
):

    rows = []

    for (
        condition,
        model,
    ), g in repeat_results.groupby(
        [
            "condition",
            "model",
        ]
    ):

        values = g[
            "auc"
        ].to_numpy()

        rows.append(
            {
                "condition":
                    condition,

                "model":
                    model,

                "mean_auc":
                    float(
                        values.mean()
                    ),

                "sd_auc":
                    float(
                        values.std(
                            ddof=1
                        )
                    ),

                "median_auc":
                    float(
                        np.median(
                            values
                        )
                    ),

                "p2_5":
                    float(
                        np.percentile(
                            values,
                            2.5,
                        )
                    ),

                "p97_5":
                    float(
                        np.percentile(
                            values,
                            97.5,
                        )
                    ),

                "min_auc":
                    float(
                        values.min()
                    ),

                "max_auc":
                    float(
                        values.max()
                    ),

                "repeats":
                    len(
                        values
                    ),
            }
        )

    return pd.DataFrame(
        rows
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

    project = (
        args.project.resolve()
    )

    input_dir = (
        project
        / "data"
        / "processed"
        / "fixed025"
    )

    outdir = (
        project
        / "results"
        / "04_figure4"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for condition in [
        "original",
        "cmvn",
    ]:

        donate = pd.read_csv(
            input_dir
            / f"donate_{condition}.csv"
        )

        enes = pd.read_csv(
            input_dir
            / f"enes_{condition}.csv"
        )

        donate_features = (
            feature_columns(
                donate
            )
        )

        enes_features = (
            feature_columns(
                enes
            )
        )

        if set(
            donate_features
        ) != set(
            enes_features
        ):

            raise RuntimeError(
                f"{condition}: "
                "feature sets differ."
            )

        features = sorted(
            donate_features
        )

        if len(features) != EXPECTED_FEATURES:

            raise RuntimeError(
                f"{condition}: "
                f"expected {EXPECTED_FEATURES} features, "
                f"found {len(features)}"
            )

        for model_name in [
            "linear_svc",
            "logreg",
        ]:

            print(
                condition,
                model_name,
                flush=True,
            )

            for repeat in range(
                N_REPEATS
            ):

                rng = (
                    np.random.default_rng(
                        SEED
                        + repeat
                    )
                )

                matched, n_each = (
                    reason_matched_sample(
                        donate,
                        enes,
                        rng,
                    )
                )

                auc = crossvalidated_auc(
                    matched,
                    features,
                    model_name,
                    SEED
                    + 1000
                    + repeat,
                )

                rows.append(
                    {
                        "condition":
                            condition,

                        "model":
                            model_name,

                        "repeat":
                            repeat,

                        "n_per_reason_per_dataset":
                            n_each,

                        "n_total":
                            len(
                                matched
                            ),

                        "n_features":
                            len(
                                features
                            ),

                        "auc":
                            auc,
                    }
                )

    repeat_results = pd.DataFrame(
        rows
    )

    summary = summarize(
        repeat_results
    )

    repeat_results.to_csv(
        outdir
        / "dataset_identity_repeat_results.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "dataset_identity_summary.csv",
        index=False,
    )

    print(
        "\n===== DATASET IDENTITY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
