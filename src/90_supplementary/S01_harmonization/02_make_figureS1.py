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
    / "supplementary"
    / "S01_harmonization"
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
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.linewidth": 1.3,
    "legend.fontsize": 12.5,
})


# ============================================================
# Load
# ============================================================

df = pd.read_csv(
    RESULTS
    / "harmonization_sensitivity.csv"
)


# ============================================================
# Helpers
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.15,
    y=1.16,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=25,
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
        width=1.2,
        length=5,
    )


def get_row(
    condition,
    model,
    direction,
):

    subset = df[
        (
            df["condition"]
            == condition
        )
        &
        (
            df["model"]
            == model
        )
        &
        (
            df["direction"]
            == direction
        )
    ]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one row for "
            f"{condition} / {model} / {direction}, "
            f"found {len(subset)}"
        )

    return subset.iloc[0]


def plot_panel(
    ax,
    conditions,
    labels,
    show_legend=False,
):

    base_x = np.arange(
        len(conditions)
    )

    # Separation between transfer directions
    direction_offset = {
        "donate_to_enes": -0.12,
        "enes_to_donate": +0.12,
    }

    # Smaller separation between classifiers
    model_offset = {
        "linear_svc": -0.035,
        "logreg": +0.035,
    }

    series = [
        (
            "donate_to_enes",
            "linear_svc",
            "o",
            "Donate → Enes, Linear SVC",
        ),
        (
            "enes_to_donate",
            "linear_svc",
            "o",
            "Enes → Donate, Linear SVC",
        ),
        (
            "donate_to_enes",
            "logreg",
            "s",
            "Donate → Enes, LogReg",
        ),
        (
            "enes_to_donate",
            "logreg",
            "s",
            "Enes → Donate, LogReg",
        ),
    ]

    for (
        direction,
        model,
        marker,
        legend_label,
    ) in series:

        values = []
        lows = []
        highs = []

        for condition in conditions:

            row = get_row(
                condition,
                model,
                direction,
            )

            values.append(
                float(
                    row["auc"]
                )
            )

            lows.append(
                float(
                    row["ci95_low"]
                )
            )

            highs.append(
                float(
                    row["ci95_high"]
                )
            )

        values = np.asarray(
            values
        )

        lows = np.asarray(
            lows
        )

        highs = np.asarray(
            highs
        )

        xpos = (
            base_x
            + direction_offset[
                direction
            ]
            + model_offset[
                model
            ]
        )

        ax.errorbar(
            xpos,
            values,
            yerr=np.vstack(
                [
                    values - lows,
                    highs - values,
                ]
            ),
            fmt=marker,
            markersize=8,
            linewidth=1.5,
            elinewidth=1.5,
            capsize=5,
            capthick=1.3,
            label=legend_label,
            zorder=3,
        )

    # Chance level
    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.3,
        zorder=1,
    )

    ax.set_xticks(
        base_x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "ROC-AUC"
    )

    # Same scale for both panels
    ax.set_ylim(
        0.30,
        0.70,
    )

    ax.set_yticks(
        np.arange(
            0.30,
            0.701,
            0.05,
        )
    )

    ax.set_xlim(
        -0.35,
        len(conditions) - 0.65,
    )

    clean_axes(
        ax
    )

    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.70, 1.10),
            fontsize=11.5,
            handlelength=1.6,
            labelspacing=0.35,
            borderaxespad=0.0,
            ncol=1,
        )


# ============================================================
# Figure
# ============================================================

fig = plt.figure(
    figsize=(11.8, 5.8)
)

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    wspace=0.30,
)


# ============================================================
# A
# Variable-duration vs fixed-duration
# ============================================================

ax_a = fig.add_subplot(
    gs[0, 0]
)

add_panel_label(
    ax_a,
    "A",
    x=-0.15,
    y=1.16,
)

plot_panel(
    ax_a,
    conditions=[
        "variable",
        "fixed_original",
    ],
    labels=[
        "Variable\nduration",
        "Fixed\n0.25 s",
    ],
    show_legend=True,
)


# ============================================================
# B
# Fixed 0.25 s before/after CMVN
# ============================================================

ax_b = fig.add_subplot(
    gs[0, 1]
)

add_panel_label(
    ax_b,
    "B",
    x=-0.15,
    y=1.16,
)

plot_panel(
    ax_b,
    conditions=[
        "fixed_original",
        "fixed_cmvn",
    ],
    labels=[
        "Fixed\n0.25 s",
        "Fixed 0.25 s\n+ CMVN",
    ],
    show_legend=False,
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.90,
    bottom=0.20,
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
        / f"FigureS1_harmonization_sensitivity.{suffix}"
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
