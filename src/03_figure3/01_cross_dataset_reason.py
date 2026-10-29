#!/usr/bin/env python3
"""
Figure 3: cross-dataset cry-reason classification.

Datasets
--------
Donate-a-Cry
Enes

Task
----
Hunger vs Discomfort

Acoustic conditions
-------------------
1. original
2. CMVN

Models
------
1. Linear SVC
2. Logistic regression

Directions
----------
1. Donate -> Enes
2. Enes -> Donate

Input
-----
data/processed/fixed025/

Outputs
-------
results/03_figure3/
    cross_dataset_summary.csv
    predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# Constants
# ============================================================

SEED = 42

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


# ============================================================
# Utilities
# ============================================================

def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]


def encode_label(df):

    # Positive class = discomfort
    return (
        df["label"]
        == "discomfort"
    ).astype(int).to_numpy()


def make_model(
    model_name,
):

    if model_name == "linear_svc":

        classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED,
        )

    elif model_name == "logreg":

        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )

    else:

        raise ValueError(
            model_name
        )

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
                classifier,
            ),
        ]
    )


def get_score(
    model,
    X,
):

    if hasattr(
        model,
        "decision_function",
    ):

        return model.decision_function(
            X
        )

    return model.predict_proba(
        X
    )[:, 1]


def calculate_metrics(
    y,
    score,
):

    pred = (
        score >= 0
    ).astype(int)

    return {
        "auc":
            float(
                roc_auc_score(
                    y,
                    score,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),
    }


# ============================================================
# One transfer experiment
# ============================================================

def run_transfer(
    train_df,
    test_df,
    features,
    condition,
    direction,
    model_name,
):

    X_train = train_df[
        features
    ]

    X_test = test_df[
        features
    ]

    y_train = encode_label(
        train_df
    )

    y_test = encode_label(
        test_df
    )

    model = make_model(
        model_name
    )

    model.fit(
        X_train,
        y_train,
    )

    score = get_score(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        score,
    )

    prediction_columns = [
        c
        for c in [
            "dataset",
            "file",
            "group_id",
            "file_seq_S",
            "baby",
            "sex",
            "age_month",
            "label",
        ]
        if c in test_df.columns
    ]

    pred = (
        test_df[
            prediction_columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    pred["y"] = y_test

    pred["score"] = score

    pred[
        "condition"
    ] = condition

    pred[
        "direction"
    ] = direction

    pred[
        "model"
    ] = model_name

    return (
        metrics,
        pred,
    )


# ============================================================
# Main
# ============================================================

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
        args.project
        .resolve()
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
        / "03_figure3"
    )

    pred_dir = (
        outdir
        / "predictions"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load all four fixed025 datasets
    # --------------------------------------------------------

    data = {
        "original": {
            "donate":
                pd.read_csv(
                    input_dir
                    / "donate_original.csv"
                ),

            "enes":
                pd.read_csv(
                    input_dir
                    / "enes_original.csv"
                ),
        },

        "cmvn": {
            "donate":
                pd.read_csv(
                    input_dir
                    / "donate_cmvn.csv"
                ),

            "enes":
                pd.read_csv(
                    input_dir
                    / "enes_cmvn.csv"
                ),
        },
    }

    rows = []

    for condition in [
        "original",
        "cmvn",
    ]:

        donate = data[
            condition
        ][
            "donate"
        ]

        enes = data[
            condition
        ][
            "enes"
        ]

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
                "Donate and Enes feature sets differ."
            )

        features = sorted(
            donate_features
        )

        if len(
            features
        ) != EXPECTED_FEATURES:

            raise RuntimeError(
                f"{condition}: "
                f"{len(features)} features; "
                f"expected {EXPECTED_FEATURES}"
            )

        experiments = [
            (
                "donate_to_enes",
                donate,
                enes,
            ),
            (
                "enes_to_donate",
                enes,
                donate,
            ),
        ]

        for (
            direction,
            train_df,
            test_df,
        ) in experiments:

            for model_name in [
                "linear_svc",
                "logreg",
            ]:

                print(
                    condition,
                    model_name,
                    direction,
                    flush=True,
                )

                metrics, pred = (
                    run_transfer(
                        train_df,
                        test_df,
                        features,
                        condition,
                        direction,
                        model_name,
                    )
                )

                rows.append(
                    {
                        "condition":
                            condition,

                        "direction":
                            direction,

                        "model":
                            model_name,

                        "n_train":
                            len(
                                train_df
                            ),

                        "n_test":
                            len(
                                test_df
                            ),

                        "n_features":
                            len(
                                features
                            ),

                        **metrics,
                    }
                )

                output_name = (
                    f"{condition}__"
                    f"{direction}__"
                    f"{model_name}"
                    "__predictions.csv"
                )

                pred.to_csv(
                    pred_dir
                    / output_name,
                    index=False,
                )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        outdir
        / "cross_dataset_summary.csv",
        index=False,
    )

    print(
        "\n===== FIXED 0.25 s CROSS-DATASET REASON ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        outdir
        / "cross_dataset_summary.csv"
    )


if __name__ == "__main__":
    main()
