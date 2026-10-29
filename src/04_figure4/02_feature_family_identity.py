#!/usr/bin/env python3
"""
Figure 3B:
Dataset identity classification using individual acoustic feature families.

Primary condition:
    fixed 0.25 s + CMVN

Dataset identity:
    Donate-a-Cry vs Enes

Evaluation:
    reason-matched, class-balanced repeated 5-fold CV

Outputs
-------
results/04_figure4/
    feature_family_repeat_results.csv
    feature_family_summary.csv
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


def family_name(feature):

    if feature.startswith(
        "ddmfcc_"
    ):
        return "DeltaDeltaMFCC"

    if feature.startswith(
        "dmfcc_"
    ):
        return "DeltaMFCC"

    if feature.startswith(
        "mfcc_"
    ):
        return "MFCC"

    if feature.startswith(
        "chroma_"
    ):
        return "Chroma"

    if (
        feature.startswith(
            "f0_"
        )
        or feature.startswith(
            "voiced_"
        )
    ):
        return "Pitch_Voicing"

    if feature.startswith(
        "rms_"
    ):
        return "Energy"

    if (
        feature.startswith(
            "onset_"
        )
    ):
        return "Temporal_Onset"

    if (
        feature.startswith(
            "spectral_"
        )
        or feature.startswith(
            "zcr_"
        )
    ):
        return "Spectral"

    return "Other"


def feature_columns(df):

    return [
        c for c in df.columns
        if c not in META_COLUMNS
    ]


def make_model(
    model_name,
    seed,
):

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
                clf,
            ),
        ]
    )


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
                df["label"]
                == label
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

            frames.append(
                part
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


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
            seed
            + fold,
        )

        model.fit(
            df.iloc[
                train_idx
            ][features],
            y[
                train_idx
            ],
        )

        scores[
            test_idx
        ] = (
            model
            .decision_function(
                df.iloc[
                    test_idx
                ][features]
            )
        )

    return float(
        roc_auc_score(
            y,
            scores,
        )
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

    donate = pd.read_csv(
        input_dir
        / "donate_cmvn.csv"
    )

    enes = pd.read_csv(
        input_dir
        / "enes_cmvn.csv"
    )

    all_features = sorted(
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

    families = {}

    for feature in all_features:

        family = family_name(
            feature
        )

        families.setdefault(
            family,
            []
        ).append(
            feature
        )

    families[
        "ALL_FEATURES"
    ] = all_features

    rows = []

    for family, features in (
        families.items()
    ):

        print(
            family,
            len(features),
            flush=True,
        )

        for model_name in [
            "linear_svc",
            "logreg",
        ]:

            for repeat in range(
                N_REPEATS
            ):

                rng = np.random.default_rng(
                    SEED
                    + repeat
                )

                matched = (
                    reason_matched_sample(
                        donate,
                        enes,
                        rng,
                    )
                )

                auc = (
                    crossvalidated_auc(
                        matched,
                        features,
                        model_name,
                        SEED
                        + 1000
                        + repeat,
                    )
                )

                rows.append(
                    {
                        "family":
                            family,

                        "n_features":
                            len(
                                features
                            ),

                        "model":
                            model_name,

                        "repeat":
                            repeat,

                        "auc":
                            auc,
                    }
                )

    repeats = pd.DataFrame(
        rows
    )

    summary_rows = []

    for (
        family,
        n_features,
        model,
    ), g in repeats.groupby(
        [
            "family",
            "n_features",
            "model",
        ]
    ):

        values = g[
            "auc"
        ].to_numpy()

        summary_rows.append(
            {
                "family":
                    family,

                "n_features":
                    int(
                        n_features
                    ),

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
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    repeats.to_csv(
        outdir
        / "feature_family_repeat_results.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "feature_family_summary.csv",
        index=False,
    )

    print(
        "\n===== FEATURE FAMILY IDENTITY ====="
    )

    print(
        summary
        .sort_values(
            [
                "model",
                "mean_auc",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
