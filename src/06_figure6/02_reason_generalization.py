#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
N_BOOT = 2000


def embedding_columns(df):

    cols = [
        c
        for c in df.columns
        if c.startswith("w2v_")
    ]

    if len(cols) != 768:

        raise RuntimeError(
            f"Expected 768 wav2vec2 features, found {len(cols)}"
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


def oof_predict(
    df,
    features,
    evaluation,
    model_name,
):

    y = (
        df["label"]
        == "discomfort"
    ).astype(int).to_numpy()

    scores = np.full(
        len(df),
        np.nan,
        dtype=float,
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

    elif evaluation == "infant":

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )

        splits = splitter.split(
            df,
            y,
            groups=df[
                "baby"
            ],
        )

    elif evaluation == "loio":

        splitter = LeaveOneGroupOut()

        splits = splitter.split(
            df,
            y,
            groups=df[
                "baby"
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
            model_name,
            SEED + fold,
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
            f"Missing predictions for {evaluation}/{model_name}"
        )

    out = df[
        [
            "file",
            "file_seq_S",
            "baby",
            "label",
        ]
    ].copy()

    out[
        "y"
    ] = y

    out[
        "score"
    ] = scores

    out[
        "evaluation"
    ] = evaluation

    out[
        "model"
    ] = model_name

    return out


def bootstrap_auc(
    pred,
    cluster_col,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    values = []

    if cluster_col is None:

        n = len(pred)

        for _ in range(
            N_BOOT
        ):

            idx = rng.choice(
                n,
                n,
                replace=True,
            )

            y = pred[
                "y"
            ].to_numpy()[
                idx
            ]

            score = pred[
                "score"
            ].to_numpy()[
                idx
            ]

            if len(
                np.unique(y)
            ) < 2:
                continue

            values.append(
                roc_auc_score(
                    y,
                    score,
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
                    lookup[g]
                    for g in sampled
                ],
                ignore_index=True,
            )

            if (
                boot[
                    "y"
                ].nunique()
                < 2
            ):
                continue

            values.append(
                roc_auc_score(
                    boot[
                        "y"
                    ],
                    boot[
                        "score"
                    ],
                )
            )

    values = np.asarray(
        values
    )

    return {
        "bootstrap_mean":
            float(
                values.mean()
            ),

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


def infant_auc_table(
    pred,
):

    rows = []

    for baby, g in pred.groupby(
        "baby"
    ):

        if g[
            "y"
        ].nunique() < 2:

            auc = np.nan

        else:

            auc = float(
                roc_auc_score(
                    g[
                        "y"
                    ],
                    g[
                        "score"
                    ],
                )
            )

        rows.append(
            {
                "baby":
                    baby,

                "n":
                    len(g),

                "auc":
                    auc,
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

    input_path = (
        project
        / "data"
        / "processed"
        / "wav2vec2_variable"
        / "enes_embeddings.csv"
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

    df = pd.read_csv(
        input_path
    )

    features = embedding_columns(
        df
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Features:",
        len(features)
    )

    print(
        "Infants:",
        df[
            "baby"
        ].nunique()
    )

    print(
        "Sessions:",
        df[
            "file_seq_S"
        ].nunique()
    )

    summary_rows = []
    predictions = []

    evaluations = [
        (
            "random",
            None,
        ),
        (
            "session",
            "file_seq_S",
        ),
        (
            "infant",
            "baby",
        ),
        (
            "loio",
            "baby",
        ),
    ]

    for model_name in [
        "linear_svc",
        "logreg",
    ]:

        for i, (
            evaluation,
            cluster_col,
        ) in enumerate(
            evaluations
        ):

            print(
                model_name,
                evaluation,
                flush=True,
            )

            pred = oof_predict(
                df,
                features,
                evaluation,
                model_name,
            )

            auc = float(
                roc_auc_score(
                    pred[
                        "y"
                    ],
                    pred[
                        "score"
                    ],
                )
            )

            ci = bootstrap_auc(
                pred,
                cluster_col,
                SEED
                + i
                + (
                    100
                    if model_name
                    == "logreg"
                    else 0
                ),
            )

            summary_rows.append(
                {
                    "representation":
                        "wav2vec2",

                    "model":
                        model_name,

                    "evaluation":
                        evaluation,

                    "auc":
                        auc,

                    **ci,
                }
            )

            predictions.append(
                pred
            )

            if evaluation == "loio":

                infant = infant_auc_table(
                    pred
                )

                infant.to_csv(
                    outdir
                    / (
                        f"wav2vec2_loio_infant_auc_"
                        f"{model_name}.csv"
                    ),
                    index=False,
                )

    summary = pd.DataFrame(
        summary_rows
    )

    all_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    summary.to_csv(
        outdir
        / "wav2vec2_reason_generalization.csv",
        index=False,
    )

    all_predictions.to_csv(
        outdir
        / "wav2vec2_reason_predictions.csv",
        index=False,
    )

    print(
        "\n===== WAV2VEC2 REASON GENERALIZATION ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
