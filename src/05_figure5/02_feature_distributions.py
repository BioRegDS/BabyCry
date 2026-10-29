#!/usr/bin/env python3
"""
Figure 4B:
Prepare distributions for top dataset-identity acoustic features.

Condition
---------
Fixed 0.25 s + CMVN

Outputs
-------
results/05_figure5/
    representative_features.csv
    representative_feature_data.csv
    representative_feature_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)


N_REPRESENTATIVE = 6


def cliffs_delta(
    x,
    y,
):
    """
    Cliff's delta:
    positive = y tends to exceed x
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    y = y[
        np.isfinite(y)
    ]

    if (
        len(x) == 0
        or len(y) == 0
    ):

        return np.nan

    # Efficient rank-based implementation
    combined = np.concatenate(
        [
            x,
            y,
        ]
    )

    ranks = pd.Series(
        combined
    ).rank(
        method="average"
    ).to_numpy()

    rx = ranks[
        :len(x)
    ].sum()

    u_x = (
        rx
        - len(x)
        * (
            len(x) + 1
        )
        / 2
    )

    # P(y > x) - P(y < x)
    delta = (
        1
        - 2
        * u_x
        / (
            len(x)
            * len(y)
        )
    )

    return float(
        delta
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

    donate = pd.read_csv(
        input_dir
        / "donate_cmvn.csv"
    )

    enes = pd.read_csv(
        input_dir
        / "enes_cmvn.csv"
    )

    importance = pd.read_csv(
        outdir
        / "feature_importance_summary.csv"
    )

    # --------------------------------------------------------
    # Choose representative top features
    #
    # Avoid showing multiple nearly identical statistics from
    # exactly the same coefficient if possible.
    # --------------------------------------------------------

    selected = []

    used_base = set()

    for _, row in importance.iterrows():

        feature = row[
            "feature"
        ]

        parts = feature.split(
            "_"
        )

        # e.g. ddmfcc_1_mean -> ddmfcc_1
        if len(parts) >= 3:

            base = "_".join(
                parts[:-1]
            )

        else:

            base = feature

        if base in used_base:
            continue

        selected.append(
            feature
        )

        used_base.add(
            base
        )

        if len(
            selected
        ) >= N_REPRESENTATIVE:

            break

    representative = importance[
        importance[
            "feature"
        ].isin(
            selected
        )
    ].copy()

    representative[
        "selection_order"
    ] = representative[
        "feature"
    ].map(
        {
            feature:
                i
            for i, feature
            in enumerate(
                selected
            )
        }
    )

    representative = (
        representative
        .sort_values(
            "selection_order"
        )
    )

    representative.to_csv(
        outdir
        / "representative_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Long-format data
    # --------------------------------------------------------

    long_rows = []

    stats_rows = []

    for feature in selected:

        x = pd.to_numeric(
            donate[
                feature
            ],
            errors="coerce",
        )

        y = pd.to_numeric(
            enes[
                feature
            ],
            errors="coerce",
        )

        x_valid = (
            x[
                np.isfinite(
                    x
                )
            ]
            .to_numpy()
        )

        y_valid = (
            y[
                np.isfinite(
                    y
                )
            ]
            .to_numpy()
        )

        for value in x_valid:

            long_rows.append(
                {
                    "feature":
                        feature,

                    "dataset":
                        "Donate-a-Cry",

                    "value":
                        float(
                            value
                        ),
                }
            )

        for value in y_valid:

            long_rows.append(
                {
                    "feature":
                        feature,

                    "dataset":
                        "Enes",

                    "value":
                        float(
                            value
                        ),
                }
            )

        pooled_sd = np.std(
            np.concatenate(
                [
                    x_valid,
                    y_valid,
                ]
            ),
            ddof=1,
        )

        wasserstein_std = (
            wasserstein_distance(
                x_valid,
                y_valid,
            )
            / pooled_sd
            if pooled_sd > 0
            else np.nan
        )

        ks = ks_2samp(
            x_valid,
            y_valid,
        )

        stats_rows.append(
            {
                "feature":
                    feature,

                "n_donate":
                    len(
                        x_valid
                    ),

                "n_enes":
                    len(
                        y_valid
                    ),

                "donate_median":
                    float(
                        np.median(
                            x_valid
                        )
                    ),

                "enes_median":
                    float(
                        np.median(
                            y_valid
                        )
                    ),

                "wasserstein_std":
                    float(
                        wasserstein_std
                    ),

                "ks_stat":
                    float(
                        ks.statistic
                    ),

                "cliffs_delta_enes_minus_donate":
                    cliffs_delta(
                        x_valid,
                        y_valid,
                    ),
            }
        )

    long_df = pd.DataFrame(
        long_rows
    )

    stats = pd.DataFrame(
        stats_rows
    )

    long_df.to_csv(
        outdir
        / "representative_feature_data.csv",
        index=False,
    )

    stats.to_csv(
        outdir
        / "representative_feature_stats.csv",
        index=False,
    )

    print(
        "\n===== REPRESENTATIVE FEATURES ====="
    )

    print(
        representative[
            [
                "feature",
                "family",
                "mean_permutation_importance",
                "mean_abs_logreg_coef",
                "mean_rank",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n===== DISTRIBUTION STATISTICS ====="
    )

    print(
        stats.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
