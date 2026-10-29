#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT = Path(__file__).resolve().parents[3]

OUTDIR = (
    PROJECT
    / "results"
    / "supplementary"
    / "S01_harmonization"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_BOOT = 5000
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
    "duration_sec",
    "audio_source",
}


def get_features(
    train_df,
    test_df,
):

    common = (
        set(train_df.columns)
        &
        set(test_df.columns)
    )

    features = []

    for col in sorted(common):

        if col in META_COLUMNS:
            continue

        if (
            pd.api.types.is_numeric_dtype(
                train_df[col]
            )
            and pd.api.types.is_numeric_dtype(
                test_df[col]
            )
        ):
            features.append(col)

    return features


def make_model(
    model_name,
):

    if model_name == "linear_svc":

        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED,
        )

    elif model_name == "logreg":

        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
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


def encode_label(
    df,
):

    return (
        df["label"]
        == "discomfort"
    ).astype(int).to_numpy()


def cluster_bootstrap_auc(
    df,
    cluster_col,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    groups = (
        df[cluster_col]
        .astype(str)
        .unique()
    )

    lookup = {
        g:
            df[
                df[cluster_col].astype(str)
                == g
            ]
        for g in groups
    }

    values = []

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
            boot["y"]
            .nunique()
            < 2
        ):
            continue

        values.append(
            roc_auc_score(
                boot["y"],
                boot["score"],
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

        "n_boot_valid":
            len(values),
    }


def run_transfer(
    train_df,
    test_df,
    model_name,
    cluster_col,
    seed,
):

    features = get_features(
        train_df,
        test_df,
    )

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
        train_df[features],
        y_train,
    )

    score = model.decision_function(
        test_df[features]
    )

    auc = float(
        roc_auc_score(
            y_test,
            score,
        )
    )

    pred = pd.DataFrame(
        {
            "y": y_test,
            "score": score,
        }
    )

    pred[
        cluster_col
    ] = (
        test_df[
            cluster_col
        ]
        .reset_index(
            drop=True
        )
    )

    ci = cluster_bootstrap_auc(
        pred,
        cluster_col,
        seed,
    )

    return {
        "auc":
            auc,

        "n_features":
            len(features),

        **ci,
    }


# ============================================================
# Load variable-duration features
# ============================================================

variable_donate = pd.read_csv(
    PROJECT
    / "data"
    / "processed"
    / "features_variable"
    / "donate_hunger_discomfort.csv"
)

variable_enes = pd.read_csv(
    PROJECT
    / "data"
    / "processed"
    / "features_variable"
    / "enes_hunger_discomfort.csv"
)


# ============================================================
# Load fixed-duration features
# ============================================================

fixed = {}

for condition in [
    "original",
    "cmvn",
]:

    fixed[condition] = {
        "donate":
            pd.read_csv(
                PROJECT
                / "data"
                / "processed"
                / "fixed025"
                / f"donate_{condition}.csv"
            ),

        "enes":
            pd.read_csv(
                PROJECT
                / "data"
                / "processed"
                / "fixed025"
                / f"enes_{condition}.csv"
            ),
    }


# ============================================================
# Run analyses
# ============================================================

rows = []

conditions = [
    (
        "variable",
        "Variable duration",
        variable_donate,
        variable_enes,
    ),
    (
        "fixed_original",
        "Fixed 0.25 s",
        fixed["original"]["donate"],
        fixed["original"]["enes"],
    ),
    (
        "fixed_cmvn",
        "Fixed 0.25 s + CMVN",
        fixed["cmvn"]["donate"],
        fixed["cmvn"]["enes"],
    ),
]


counter = 0

for (
    condition,
    condition_label,
    donate,
    enes,
) in conditions:

    for model_name in [
        "linear_svc",
        "logreg",
    ]:

        for direction in [
            "donate_to_enes",
            "enes_to_donate",
        ]:

            counter += 1

            print(
                condition,
                model_name,
                direction,
                flush=True,
            )

            if direction == "donate_to_enes":

                result = run_transfer(
                    donate,
                    enes,
                    model_name,
                    cluster_col="baby",
                    seed=
                        SEED
                        + counter,
                )

            else:

                result = run_transfer(
                    enes,
                    donate,
                    model_name,
                    cluster_col="group_id",
                    seed=
                        SEED
                        + counter,
                )

            rows.append(
                {
                    "condition":
                        condition,

                    "condition_label":
                        condition_label,

                    "model":
                        model_name,

                    "direction":
                        direction,

                    **result,
                }
            )


summary = pd.DataFrame(
    rows
)

summary.to_csv(
    OUTDIR
    / "harmonization_sensitivity.csv",
    index=False,
)


print(
    "\n===== HARMONIZATION SENSITIVITY ====="
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    OUTDIR
    / "harmonization_sensitivity.csv"
)
