#!/usr/bin/env python3
"""
Figure 3: cluster-aware inference for cross-dataset transfer.

For Donate -> Enes:
    - infant-cluster bootstrap
    - session-level AUC
    - infant-specific AUC

For Enes -> Donate:
    - Donate group_id-cluster bootstrap
    - group-level AUC

The bootstrap respects the dominant dependence structure in the
external test dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# Constants
# ============================================================

N_BOOT = 5000
SEED = 42


# ============================================================
# Cluster bootstrap
# ============================================================

def cluster_bootstrap_auc(
    df,
    cluster_col,
    n_boot,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    cluster_values = (
        df[
            cluster_col
        ]
        .astype(str)
        .unique()
    )

    lookup = {
        cluster:
            df[
                df[
                    cluster_col
                ].astype(str)
                == cluster
            ]
        for cluster
        in cluster_values
    }

    values = []

    for _ in range(
        n_boot
    ):

        sampled = rng.choice(
            cluster_values,
            size=len(
                cluster_values
            ),
            replace=True,
        )

        boot = pd.concat(
            [
                lookup[
                    cluster
                ]
                for cluster
                in sampled
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
        values,
        dtype=float,
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
            int(
                len(values)
            ),
    }


# ============================================================
# Aggregate scores
# ============================================================

def aggregate_auc(
    df,
    group_cols,
):

    agg = (
        df
        .groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            y=(
                "y",
                "first",
            ),

            score=(
                "score",
                "mean",
            ),

            n_clips=(
                "score",
                "size",
            ),
        )
        .reset_index()
    )

    if (
        agg[
            "y"
        ].nunique()
        < 2
    ):

        return (
            np.nan,
            agg,
        )

    auc = float(
        roc_auc_score(
            agg[
                "y"
            ],
            agg[
                "score"
            ],
        )
    )

    return (
        auc,
        agg,
    )


# ============================================================
# Infant-specific Enes AUC
# ============================================================

def infant_auc_table(
    df,
):

    rows = []

    for baby, g in (
        df.groupby(
            "baby"
        )
    ):

        if (
            g[
                "y"
            ].nunique()
            < 2
        ):

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

                "n_clips":
                    len(g),

                "n_hunger":
                    int(
                        (
                            g[
                                "y"
                            ]
                            == 0
                        ).sum()
                    ),

                "n_discomfort":
                    int(
                        (
                            g[
                                "y"
                            ]
                            == 1
                        ).sum()
                    ),

                "auc":
                    auc,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Session-label permutation for Enes
# ============================================================

def enes_session_permutation(
    df,
    n_perm,
    seed,
):
    """
    Aggregate clips to session first.

    Labels are permuted among sessions within each infant.
    This preserves infant-specific structure while disrupting the
    association between session reason and model score.
    """

    _, sessions = aggregate_auc(
        df,
        [
            "baby",
            "file_seq_S",
        ],
    )

    observed = float(
        roc_auc_score(
            sessions[
                "y"
            ],
            sessions[
                "score"
            ],
        )
    )

    rng = np.random.default_rng(
        seed
    )

    perm_values = []

    babies = (
        sessions[
            "baby"
        ].unique()
    )

    for _ in range(
        n_perm
    ):

        perm = sessions.copy()

        for baby in babies:

            idx = (
                perm[
                    "baby"
                ]
                == baby
            )

            labels = (
                perm.loc[
                    idx,
                    "y"
                ]
                .to_numpy()
                .copy()
            )

            rng.shuffle(
                labels
            )

            perm.loc[
                idx,
                "y"
            ] = labels

        if (
            perm[
                "y"
            ].nunique()
            < 2
        ):

            continue

        perm_values.append(
            roc_auc_score(
                perm[
                    "y"
                ],
                perm[
                    "score"
                ],
            )
        )

    perm_values = np.asarray(
        perm_values
    )

    # Two-sided around 0.5
    observed_distance = abs(
        observed
        - 0.5
    )

    perm_distance = np.abs(
        perm_values
        - 0.5
    )

    p = (
        1
        + np.sum(
            perm_distance
            >= observed_distance
        )
    ) / (
        1
        + len(
            perm_values
        )
    )

    return {
        "session_level_auc":
            observed,

        "permutation_p_two_sided":
            float(
                p
            ),

        "n_session_units":
            int(
                len(
                    sessions
                )
            ),

        "n_permutations_valid":
            int(
                len(
                    perm_values
                )
            ),
    }


# ============================================================
# Donate group permutation
# ============================================================

def donate_group_permutation(
    df,
    n_perm,
    seed,
):
    """
    Group-level inference for Enes -> Donate.

    Donate group_id is treated as the clustering unit for bootstrap
    inference. However, some group_ids contain clips from both hunger
    and discomfort classes.

    For group-level AUC and label permutation, only group_ids with a
    single unique cry-reason label are retained, because a unique
    group-level target cannot otherwise be defined.
    """

    # --------------------------------------------------------
    # Identify single-label group_ids
    # --------------------------------------------------------

    group_label_counts = (
        df
        .groupby(
            "group_id"
        )[
            "y"
        ]
        .nunique()
    )

    valid_groups = (
        group_label_counts[
            group_label_counts
            == 1
        ]
        .index
    )

    excluded_groups = (
        group_label_counts[
            group_label_counts
            > 1
        ]
        .index
    )

    print(
        "Donate group-level inference:"
    )

    print(
        "  total group_ids:",
        len(
            group_label_counts
        )
    )

    print(
        "  single-label group_ids:",
        len(
            valid_groups
        )
    )

    print(
        "  multi-label group_ids excluded:",
        len(
            excluded_groups
        )
    )

    # --------------------------------------------------------
    # Restrict to groups with a unique target label
    # --------------------------------------------------------

    valid_df = (
        df[
            df[
                "group_id"
            ].isin(
                valid_groups
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Aggregate prediction score within each group_id
    # --------------------------------------------------------

    groups = (
        valid_df
        .groupby(
            "group_id"
        )
        .agg(
            y=(
                "y",
                "first",
            ),

            score=(
                "score",
                "mean",
            ),

            n_clips=(
                "score",
                "size",
            ),
        )
        .reset_index()
    )

    if (
        groups[
            "y"
        ].nunique()
        < 2
    ):

        raise RuntimeError(
            "Group-level Donate data "
            "contain fewer than two classes."
        )

    observed = float(
        roc_auc_score(
            groups[
                "y"
            ],
            groups[
                "score"
            ],
        )
    )

    # --------------------------------------------------------
    # Group-label permutation
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    original_labels = (
        groups[
            "y"
        ]
        .to_numpy()
    )

    values = []

    for _ in range(
        n_perm
    ):

        shuffled = (
            original_labels
            .copy()
        )

        rng.shuffle(
            shuffled
        )

        if len(
            np.unique(
                shuffled
            )
        ) < 2:

            continue

        values.append(
            roc_auc_score(
                shuffled,
                groups[
                    "score"
                ],
            )
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    # Two-sided test around chance AUC = 0.5
    observed_distance = abs(
        observed
        - 0.5
    )

    permutation_distance = np.abs(
        values
        - 0.5
    )

    p = (
        1
        + np.sum(
            permutation_distance
            >= observed_distance
        )
    ) / (
        1
        + len(
            values
        )
    )

    return {
        "group_level_auc":
            observed,

        "permutation_p_two_sided":
            float(
                p
            ),

        "n_group_units":
            int(
                len(
                    groups
                )
            ),

        "n_multilabel_groups_excluded":
            int(
                len(
                    excluded_groups
                )
            ),

        "n_permutations_valid":
            int(
                len(
                    values
                )
            ),
    }


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

    pred_dir = (
        project
        / "results"
        / "03_figure3"
        / "predictions"
    )

    outdir = (
        project
        / "results"
        / "03_figure3"
    )

    infant_dir = (
        outdir
        / "infant_auc"
    )

    infant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_files = sorted(
        pred_dir.glob(
            "*__predictions.csv"
        )
    )

    if not prediction_files:

        raise FileNotFoundError(
            "No prediction files found."
        )

    rows = []

    for i, path in enumerate(
        prediction_files
    ):

        print(
            "\nProcessing:",
            path.name,
            flush=True,
        )

        df = pd.read_csv(
            path
        )

        condition = (
            df[
                "condition"
            ]
            .iloc[0]
        )

        direction = (
            df[
                "direction"
            ]
            .iloc[0]
        )

        model = (
            df[
                "model"
            ]
            .iloc[0]
        )

        clip_auc = float(
            roc_auc_score(
                df[
                    "y"
                ],
                df[
                    "score"
                ],
            )
        )

        row = {
            "condition":
                condition,

            "direction":
                direction,

            "model":
                model,

            "clip_level_auc":
                clip_auc,
        }

        # ----------------------------------------------------
        # Donate -> Enes
        # ----------------------------------------------------

        if (
            direction
            == "donate_to_enes"
        ):

            boot = cluster_bootstrap_auc(
                df,
                cluster_col="baby",
                n_boot=N_BOOT,
                seed=SEED + i,
            )

            inference = (
                enes_session_permutation(
                    df,
                    n_perm=N_BOOT,
                    seed=SEED
                    + 100
                    + i,
                )
            )

            infant_auc = (
                infant_auc_table(
                    df
                )
            )

            infant_auc.to_csv(
                infant_dir
                / (
                    f"{condition}__"
                    f"{model}__"
                    "donate_to_enes.csv"
                ),
                index=False,
            )

            valid_auc = (
                infant_auc[
                    "auc"
                ]
                .dropna()
            )

            row.update(
                {
                    **boot,
                    **inference,

                    "cluster_column":
                        "baby",

                    "n_clusters":
                        int(
                            df[
                                "baby"
                            ].nunique()
                        ),

                    "n_infants_auc_defined":
                        int(
                            len(
                                valid_auc
                            )
                        ),

                    "median_infant_auc":
                        float(
                            valid_auc.median()
                        ),

                    "mean_infant_auc":
                        float(
                            valid_auc.mean()
                        ),

                    "group_level_auc":
                        np.nan,

                    "n_group_units":
                        np.nan,
                }
            )

        # ----------------------------------------------------
        # Enes -> Donate
        # ----------------------------------------------------

        elif (
            direction
            == "enes_to_donate"
        ):

            boot = cluster_bootstrap_auc(
                df,
                cluster_col="group_id",
                n_boot=N_BOOT,
                seed=SEED + i,
            )

            inference = (
                donate_group_permutation(
                    df,
                    n_perm=N_BOOT,
                    seed=SEED
                    + 200
                    + i,
                )
            )

            row.update(
                {
                    **boot,
                    **inference,

                    "cluster_column":
                        "group_id",

                    "n_clusters":
                        int(
                            df[
                                "group_id"
                            ].nunique()
                        ),

                    "session_level_auc":
                        np.nan,

                    "n_session_units":
                        np.nan,

                    "n_infants_auc_defined":
                        np.nan,

                    "median_infant_auc":
                        np.nan,

                    "mean_infant_auc":
                        np.nan,
                }
            )

        else:

            raise ValueError(
                direction
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    preferred_order = [
        "condition",
        "direction",
        "model",
        "clip_level_auc",
        "bootstrap_mean",
        "ci95_low",
        "ci95_high",
        "n_boot_valid",
        "cluster_column",
        "n_clusters",
        "session_level_auc",
        "group_level_auc",
        "permutation_p_two_sided",
        "n_session_units",
        "n_group_units",
        "n_permutations_valid",
        "n_infants_auc_defined",
        "median_infant_auc",
        "mean_infant_auc",
    ]

    summary = summary[
        [
            c
            for c in preferred_order
            if c in summary.columns
        ]
    ]

    summary.to_csv(
        outdir
        / "cluster_inference_summary.csv",
        index=False,
    )

    print(
        "\n===== FIGURE 2 CLUSTER-AWARE INFERENCE ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        outdir
        / "cluster_inference_summary.csv"
    )


if __name__ == "__main__":
    main()
