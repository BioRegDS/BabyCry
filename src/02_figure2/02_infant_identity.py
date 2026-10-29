#!/usr/bin/env python3
"""
Figure 2: infant-identity classification within Enes.

Question
--------
How strongly does the acoustic representation encode infant identity?

Evaluation
----------
1. Random clip split
2. Session-independent split

Chance accuracy = 1 / 24.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)


META_COLUMNS = {
    "dataset",
    "file",
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

SEED = 42
N_BOOT = 2000


def feature_columns(
    df,
):

    return [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]


def make_model(
    seed,
):

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
                    max_iter=3000,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def predict_identity(
    df,
    features,
    evaluation,
):

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df[
            "baby"
        ].astype(str)
    )

    pred = np.full(
        len(df),
        -1,
        dtype=int,
    )

    if evaluation == "random":

        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )

        splits = splitter.split(
            df,
            y,
        )

    elif evaluation == "session":

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )

        splits = splitter.split(
            df,
            y,
            groups=df[
                "file_seq_S"
            ],
        )

    else:

        raise ValueError(
            evaluation
        )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        model = make_model(
            SEED
            + fold
        )

        model.fit(
            df.iloc[
                train_idx
            ][features],
            y[
                train_idx
            ],
        )

        pred[
            test_idx
        ] = model.predict(
            df.iloc[
                test_idx
            ][features]
        )

    if (
        pred < 0
    ).any():

        raise RuntimeError(
            "Missing identity OOF predictions."
        )

    out = df[
        [
            "file",
            "file_seq_S",
            "baby",
        ]
    ].copy()

    out[
        "y"
    ] = y

    out[
        "pred"
    ] = pred

    out[
        "evaluation"
    ] = evaluation

    return out


def bootstrap_accuracy(
    pred,
    cluster_col,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    values = []

    if cluster_col is None:

        n = len(
            pred
        )

        for _ in range(
            N_BOOT
        ):

            idx = rng.choice(
                n,
                n,
                replace=True,
            )

            values.append(
                accuracy_score(
                    pred[
                        "y"
                    ].to_numpy()[
                        idx
                    ],
                    pred[
                        "pred"
                    ].to_numpy()[
                        idx
                    ],
                )
            )

    else:

        groups = (
            pred[
                cluster_col
            ]
            .astype(str)
            .unique()
        )

        lookup = {
            g:
                pred[
                    pred[
                        cluster_col
                    ].astype(str)
                    == g
                ]
            for g in groups
        }

        for _ in range(
            N_BOOT
        ):

            sampled = rng.choice(
                groups,
                len(groups),
                replace=True,
            )

            boot = pd.concat(
                [
                    lookup[
                        g
                    ]
                    for g
                    in sampled
                ],
                ignore_index=True,
            )

            values.append(
                accuracy_score(
                    boot[
                        "y"
                    ],
                    boot[
                        "pred"
                    ],
                )
            )

    values = np.asarray(
        values
    )

    return {
        "ci95_low":
            float(
                np.percentile(
                    values,
                    2.5,
                )
            ),

        "ci95_high":
            float(
                np.percentile(
                    values,
                    97.5,
                )
            ),
    }


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

    df = pd.read_csv(
        project
        / "data"
        / "processed"
        / "features_variable"
        / "enes_hunger_discomfort.csv"
    )

    outdir = (
        project
        / "results"
        / "02_figure2"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = feature_columns(
        df
    )

    n_infants = (
        df[
            "baby"
        ].nunique()
    )

    chance = (
        1.0
        / n_infants
    )

    rows = []
    predictions = []

    for evaluation, cluster in [
        (
            "random",
            None,
        ),
        (
            "session",
            "file_seq_S",
        ),
    ]:

        print(
            "Running identity:",
            evaluation,
            flush=True,
        )

        pred = predict_identity(
            df,
            features,
            evaluation,
        )

        accuracy = float(
            accuracy_score(
                pred[
                    "y"
                ],
                pred[
                    "pred"
                ],
            )
        )

        macro_f1 = float(
            f1_score(
                pred[
                    "y"
                ],
                pred[
                    "pred"
                ],
                average="macro",
                zero_division=0,
            )
        )

        ci = bootstrap_accuracy(
            pred,
            cluster,
            SEED
            + (
                100
                if evaluation
                == "random"
                else 200
            ),
        )

        rows.append(
            {
                "evaluation":
                    evaluation,

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "chance_accuracy":
                    chance,

                **ci,
            }
        )

        predictions.append(
            pred
        )

    summary = pd.DataFrame(
        rows
    )

    all_pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    summary.to_csv(
        outdir
        / "infant_identity_summary.csv",
        index=False,
    )

    all_pred.to_csv(
        outdir
        / "infant_identity_predictions.csv",
        index=False,
    )

    print(
        "\n===== INFANT IDENTITY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
