#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
N_REPEATS = 200
N_SPLITS = 5


def embedding_columns(
    df,
):

    cols = [
        c
        for c in df.columns
        if c.startswith(
            "w2v_"
        )
    ]

    if len(cols) != 768:

        raise RuntimeError(
            f"Expected 768 wav2vec2 features, "
            f"found {len(cols)}"
        )

    return cols


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
                df[
                    "label"
                ]
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
                if dataset_name
                == "donate"
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

        scores[
            test_idx
        ] = model.decision_function(
            df.iloc[
                test_idx
            ][features]
        )

    if np.isnan(
        scores
    ).any():

        raise RuntimeError(
            "Missing dataset-identity predictions."
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
        / "wav2vec2_variable"
    )

    outdir = (
        project
        / "results"
        / "06_figure6"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    donate = pd.read_csv(
        input_dir
        / "donate_embeddings.csv"
    )

    enes = pd.read_csv(
        input_dir
        / "enes_embeddings.csv"
    )

    donate_features = embedding_columns(
        donate
    )

    enes_features = embedding_columns(
        enes
    )

    if set(
        donate_features
    ) != set(
        enes_features
    ):

        raise RuntimeError(
            "Donate and Enes wav2vec2 dimensions differ."
        )

    features = sorted(
        donate_features
    )

    rows = []

    for model_name in [
        "linear_svc",
        "logreg",
    ]:

        print(
            model_name,
            flush=True,
        )

        for repeat in range(
            N_REPEATS
        ):

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
                    "representation":
                        "wav2vec2",

                    "model":
                        model_name,

                    "repeat":
                        repeat,

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

    repeats = pd.DataFrame(
        rows
    )

    summary_rows = []

    for model_name, g in repeats.groupby(
        "model"
    ):

        values = g[
            "auc"
        ].to_numpy()

        summary_rows.append(
            {
                "representation":
                    "wav2vec2",

                "model":
                    model_name,

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
        / "wav2vec2_dataset_identity_repeats.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "wav2vec2_dataset_identity_summary.csv",
        index=False,
    )

    print(
        "\n===== WAV2VEC2 DATASET IDENTITY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
