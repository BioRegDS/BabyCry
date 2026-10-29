#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[3]

RESULTS = (
    PROJECT
    / "results"
    / "05_figure5"
)

FIGURES = (
    PROJECT
    / "figures"
    / "supplementary"
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
})


# ============================================================
# Load
# ============================================================

importance = pd.read_csv(
    RESULTS
    / "feature_importance_summary.csv"
)

rep_features = pd.read_csv(
    RESULTS
    / "representative_features.csv"
)

rep_data = pd.read_csv(
    RESULTS
    / "representative_feature_data.csv"
)


# ============================================================
# Helpers
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.14,
    y=1.14,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        width=1.1,
        length=5,
    )


def pretty_feature_name(feature):

    replacements = {
        "spectral_contrast_1_p25":
            "Spectral contrast 1\n25th percentile",

        "spectral_contrast_1_median":
            "Spectral contrast 1\nmedian",

        "spectral_contrast_1_p75":
            "Spectral contrast 1\n75th percentile",

        "spectral_contrast_1_mean":
            "Spectral contrast 1\nmean",

        "spectral_contrast_1_std":
            "Spectral contrast 1\nSD",

        "spectral_contrast_1_max":
            "Spectral contrast 1\nmaximum",
    }

    if feature in replacements:
        return replacements[feature]

    if feature.startswith("ddmfcc_"):

        parts = feature.split("_")

        coef = parts[1]
        stat = "_".join(
            parts[2:]
        )

        stat_map = {
            "mean": "mean",
            "median": "median",
            "p25": "25th percentile",
            "p75": "75th percentile",
            "std": "SD",
            "min": "minimum",
            "max": "maximum",
        }

        stat = stat_map.get(
            stat,
            stat,
        )

        return (
            f"ΔΔ-MFCC {coef}\n{stat}"
        )

    if feature.startswith("dmfcc_"):

        parts = feature.split("_")

        return (
            f"Δ-MFCC {parts[1]}\n"
            + "_".join(parts[2:])
        )

    if feature.startswith("mfcc_"):

        parts = feature.split("_")

        return (
            f"MFCC {parts[1]}\n"
            + "_".join(parts[2:])
        )

    return feature


# ============================================================
# Panel A data
# ============================================================

TOP_N = 12

top = (
    importance
    .sort_values(
        "mean_rank"
    )
    .head(
        TOP_N
    )
    .copy()
)

top = (
    top
    .sort_values(
        "mean_abs_logreg_coef",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Panel B data
# ============================================================

selected_features = (
    rep_features
    .sort_values(
        "selection_order"
    )[
        "feature"
    ]
    .head(4)
    .tolist()
)


# ============================================================
# Figure
# ============================================================

fig = plt.figure(
    figsize=(13.2, 8.6)
)

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[
        0.90,
        1.50,
    ],
    wspace=0.38,
)


# ============================================================
# A — Feature importance
# ============================================================

ax = fig.add_subplot(
    gs[0, 0]
)

add_panel_label(
    ax,
    "A",
    x=-0.20,
    y=1.08,
)

ypos = np.arange(
    len(top)
)

values = top[
    "mean_abs_logreg_coef"
].to_numpy()

labels = [
    pretty_feature_name(
        x
    )
    for x in top[
        "feature"
    ]
]


ax.barh(
    ypos,
    values,
    height=0.62,
    fill=False,
    linewidth=1.5,
)


ax.set_yticks(
    ypos
)

ax.set_yticklabels(
    labels
)

ax.set_xlabel(
    "Mean absolute standardized coefficient"
    #    "Mean absolute standardized\n"
#    "logistic regression coefficient"
)


max_value = values.max()

for yi, value in zip(
    ypos,
    values,
):

    ax.text(
        value
        + max_value
        * 0.025,
        yi,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=12,
    )


ax.set_xlim(
    0,
    max_value
    * 1.22,
)

clean_axes(
    ax
)


# ============================================================
# B — Representative distributions
# ============================================================

subgs = gs[
    0,
    1
].subgridspec(
    nrows=2,
    ncols=2,
    hspace=0.46,
    wspace=0.55,
)


for i, feature in enumerate(
    selected_features
):

    ax = fig.add_subplot(
        subgs[
            i // 2,
            i % 2
        ]
    )

    if i == 0:

        add_panel_label(
            ax,
            "B",
            x=-0.24,
            y=1.18,
        )

    temp = rep_data[
        rep_data[
            "feature"
        ]
        == feature
    ].copy()

    donate = (
        temp[
            temp[
                "dataset"
            ]
            == "Donate-a-Cry"
        ][
            "value"
        ]
        .dropna()
        .to_numpy()
    )

    enes = (
        temp[
            temp[
                "dataset"
            ]
            == "Enes"
        ][
            "value"
        ]
        .dropna()
        .to_numpy()
    )

    positions = [
        0,
        1,
    ]


    violin = ax.violinplot(
        [
            donate,
            enes,
        ],
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )


    for body in violin[
        "bodies"
    ]:

        body.set_alpha(
            0.35
        )


    for pos, values_array in zip(
        positions,
        [
            donate,
            enes,
        ],
    ):

        q25 = np.percentile(
            values_array,
            25,
        )

        median = np.median(
            values_array
        )

        q75 = np.percentile(
            values_array,
            75,
        )

        ax.plot(
            [
                pos,
                pos,
            ],
            [
                q25,
                q75,
            ],
            linewidth=4,
            solid_capstyle="butt",
        )

        ax.scatter(
            [
                pos
            ],
            [
                median
            ],
            s=35,
            zorder=4,
        )


    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [
            "Donate-\na-Cry",
            "Enes",
        ]
    )


    # Only left column gets y-axis label
    if i % 2 == 0:

        ax.set_ylabel(
            "Feature value"
        )

    else:

        ax.set_ylabel("")


    ax.text(
        0.03,
        0.88,
        pretty_feature_name(
            feature
        ).replace(
            "\n",
            " "
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )


    clean_axes(
        ax
    )


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.16,
    right=0.98,
    top=0.94,
    bottom=0.10,
)


# ============================================================
# Save
# ============================================================

for suffix in [
    "png",
    "pdf",
    "svg",
]:

    output = (
        FIGURES
        / f"FigureS2_feature_signatures.{suffix}"
    )

    if suffix == "png":

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

    else:

        fig.savefig(
            output,
            bbox_inches="tight",
        )

    print(
        "Saved:",
        output
    )


plt.close(fig)
