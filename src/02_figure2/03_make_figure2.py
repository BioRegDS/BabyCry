#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

RESULTS = (
    PROJECT
    / "results"
    / "02_figure2"
)

FIGURES = (
    PROJECT
    / "figures"
    / "main"
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
})


# ============================================================
# Load data
# ============================================================

reason = pd.read_csv(
    RESULTS
    / "reason_generalization_summary.csv"
)

identity = pd.read_csv(
    RESULTS
    / "infant_identity_summary.csv"
)

infant_auc = pd.read_csv(
    RESULTS
    / "loio_infant_auc.csv"
)


# ============================================================
# Helper
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.07,
    y=1.12,
):

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def clean_axes(
    ax,
):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        width=1.1,
        length=5,
    )


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(10.5, 8.5)
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[
        1.10,
        1.00,
    ],
    hspace=0.43,
    wspace=0.34,
)


# ============================================================
# A
# Cry-reason classification
# ============================================================

ax = fig.add_subplot(
    gs[0, :]
)

add_panel_label(
    ax,
    "A",
    x=-0.055,
    y=1.12,
)

order = [
    "random",
    "session",
    "infant",
    "loio",
]

display_labels = [
    "Random\nclips",
    "New\nsessions",
    "New\ninfants",
    "LOIO",
]

rp = (
    reason
    .set_index("evaluation")
    .loc[order]
)

x = np.arange(
    len(order)
)

y = rp[
    "auc"
].to_numpy()

ci_low = rp[
    "ci95_low"
].to_numpy()

ci_high = rp[
    "ci95_high"
].to_numpy()

yerr = np.vstack(
    [
        y - ci_low,
        ci_high - y,
    ]
)

ax.errorbar(
    x,
    y,
    yerr=yerr,
    marker="o",
    markersize=8,
    linewidth=1.8,
    capsize=5,
    capthick=1.4,
    zorder=3,
)

# Chance line
ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

# Value labels
value_offsets = [
    0.024,
    0.026,
    0.026,
    0.028,
]

for xi, yi, offset in zip(
    x,
    y,
    value_offsets,
):

    ax.text(
        xi + 5 * offset,
        yi + offset,
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=15,
    )

# Chance annotation:
# move to far right and above line
#ax.text(
#    3.28,
#    0.512,
#    "Chance",
#    ha="right",
#    va="bottom",
#    fontsize=14,
#)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    display_labels
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0.38,
    0.80,
)

ax.set_xlim(
    -0.35,
    3.35,
)

clean_axes(
    ax
)


# ============================================================
# B
# Infant identity classification
# ============================================================

ax = fig.add_subplot(
    gs[1, 0]
)

add_panel_label(
    ax,
    "B",
    x=-0.16,
    y=1.12,
)

identity_order = [
    "random",
    "session",
]

identity_labels = [
    "Random\nclips",
    "New\nsessions",
]

ip = (
    identity
    .set_index("evaluation")
    .loc[identity_order]
)

x = np.arange(
    len(identity_order)
)

y = ip[
    "accuracy"
].to_numpy()

ci_low = ip[
    "ci95_low"
].to_numpy()

ci_high = ip[
    "ci95_high"
].to_numpy()

yerr = np.vstack(
    [
        y - ci_low,
        ci_high - y,
    ]
)

ax.bar(
    x,
    y,
    width=0.56,
    fill=False,
    linewidth=1.6,
)

ax.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="none",
    capsize=5,
    linewidth=1.4,
    capthick=1.3,
)

chance = float(
    ip[
        "chance_accuracy"
    ].iloc[0]
)

ax.axhline(
    chance,
    linestyle="--",
    linewidth=1.2,
)

# Bar value labels
for xi, yi in zip(
    x,
    y,
):

    ax.text(
        xi,
        yi + 0.028,
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=15,
    )

# Chance label:
# right side and clearly above line
#ax.text(
#    1.34,
#    chance + 0.020,
#    "Chance = 1/24",
#    ha="right",
#    va="bottom",
#    fontsize=14,
#)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    identity_labels
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_ylim(
    0,
    0.80,
)

ax.set_xlim(
    -0.45,
    1.45,
)

clean_axes(
    ax
)


# ============================================================
# C
# Infant-wise LOIO
# ============================================================

ax = fig.add_subplot(
    gs[1, 1]
)

add_panel_label(
    ax,
    "C",
    x=-0.16,
    y=1.12,
)

valid = (
    infant_auc
    .dropna(
        subset=["auc"]
    )
    .sort_values("auc")
    .reset_index(drop=True)
)

x = np.arange(
    1,
    len(valid) + 1
)

y = valid[
    "auc"
].to_numpy()

ax.scatter(
    x,
    y,
    s=42,
    zorder=3,
)

# Chance line
ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

median_auc = float(
    valid[
        "auc"
    ].median()
)

# Median line
ax.axhline(
    median_auc,
    linewidth=1.6,
    zorder=2,
)

# Median label:
# place above centre-left of plot, away from points
ax.text(
    len(valid) * 0.46,
    median_auc + 0.055,
    f"Median = {median_auc:.2f}",
    ha="center",
    va="bottom",
    fontsize=14,
)

ax.set_xlabel(
    "Infants"
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0,
    1.0,
)

ax.set_xlim(
    0.3,
    len(valid) + 0.7,
)

# Sorted infants; individual IDs omitted
ax.set_xticks([])

clean_axes(
    ax
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.97,
    top=0.95,
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
        / f"Figure2_generalization_hierarchy.{suffix}"
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


plt.close(
    fig
)
