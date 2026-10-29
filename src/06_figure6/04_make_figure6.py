#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

RESULTS5 = (
    PROJECT
    / "results"
    / "06_figure6"
)

RESULTS1 = (
    PROJECT
    / "results"
    / "02_figure2"
)

RESULTS3 = (
    PROJECT
    / "results"
    / "04_figure4"
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
# Style
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
})


# ============================================================
# Load data
# ============================================================

# Handcrafted reason generalization
hc_reason = pd.read_csv(
    RESULTS1
    / "reason_generalization_summary.csv"
)

# wav2vec2 reason generalization
w2v_reason = pd.read_csv(
    RESULTS5
    / "wav2vec2_reason_generalization.csv"
)

# Handcrafted dataset identity
hc_identity = pd.read_csv(
    RESULTS3
    / "dataset_identity_summary.csv"
)

# wav2vec2 dataset identity
w2v_identity = pd.read_csv(
    RESULTS5
    / "wav2vec2_dataset_identity_summary.csv"
)


# ============================================================
# Helpers
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.12,
    y=1.15,
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


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(11.5, 5.6)
)

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[
        1.25,
        0.85,
    ],
    wspace=0.34,
)


# ============================================================
# A
# Reason generalization hierarchy
# ============================================================

ax = fig.add_subplot(
    gs[0, 0]
)

add_panel_label(
    ax,
    "A",
    x=-0.13,
    y=1.14,
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


# ------------------------------------------------------------
# Handcrafted
# ------------------------------------------------------------

hc = (
    hc_reason
    .set_index("evaluation")
    .loc[order]
)

x = np.arange(
    len(order)
)

hc_y = hc[
    "auc"
].to_numpy()

hc_low = hc[
    "ci95_low"
].to_numpy()

hc_high = hc[
    "ci95_high"
].to_numpy()

hc_yerr = np.vstack(
    [
        hc_y - hc_low,
        hc_high - hc_y,
    ]
)


ax.errorbar(
    x,
    hc_y,
    yerr=hc_yerr,
    marker="o",
    markersize=8,
    linewidth=1.8,
    capsize=5,
    capthick=1.3,
    label="Handcrafted",
)


# ------------------------------------------------------------
# wav2vec2
# Use Linear SVC for direct classifier consistency
# ------------------------------------------------------------

w2v = (
    w2v_reason[
        w2v_reason[
            "model"
        ]
        == "linear_svc"
    ]
    .set_index("evaluation")
    .loc[order]
)

w2v_y = w2v[
    "auc"
].to_numpy()

w2v_low = w2v[
    "ci95_low"
].to_numpy()

w2v_high = w2v[
    "ci95_high"
].to_numpy()

w2v_yerr = np.vstack(
    [
        w2v_y - w2v_low,
        w2v_high - w2v_y,
    ]
)


ax.errorbar(
    x,
    w2v_y,
    yerr=w2v_yerr,
    marker="s",
    markersize=7,
    linewidth=1.8,
    capsize=5,
    capthick=1.3,
    label="wav2vec2",
)


# Chance line
ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)


# Numeric labels
for xi, yi in zip(
    x,
    hc_y,
):
    ax.text(
        xi - 0.08,
        yi + 0.026,
        f"{yi:.2f}",
        ha="right",
        va="bottom",
        fontsize=13,
    )


for xi, yi in zip(
    x,
    w2v_y,
):
    ax.text(
        xi + 0.08,
        yi - 0.030,
        f"{yi:.2f}",
        ha="left",
        va="top",
        fontsize=13,
    )


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
    0.40,
    0.80,
)

ax.set_xlim(
    -0.35,
    3.35,
)

ax.legend(
    frameon=False,
    fontsize=14,
    loc="upper right",
)

clean_axes(
    ax
)


# ============================================================
# B
# Dataset identity
# ============================================================

ax = fig.add_subplot(
    gs[0, 1]
)

add_panel_label(
    ax,
    "B",
    x=-0.18,
    y=1.14,
)


# ------------------------------------------------------------
# Handcrafted
# Use CMVN + Linear SVC
# ------------------------------------------------------------

hc_row = hc_identity[
    (
        hc_identity[
            "condition"
        ]
        == "cmvn"
    )
    &
    (
        hc_identity[
            "model"
        ]
        == "linear_svc"
    )
].iloc[0]


# ------------------------------------------------------------
# wav2vec2
# Use Linear SVC
# ------------------------------------------------------------

w2v_row = w2v_identity[
    w2v_identity[
        "model"
    ]
    == "linear_svc"
].iloc[0]


labels = [
    "Handcrafted\n+ CMVN",
    "wav2vec2",
]

values = np.array(
    [
        float(
            hc_row[
                "mean_auc"
            ]
        ),
        float(
            w2v_row[
                "mean_auc"
            ]
        ),
    ]
)

low = np.array(
    [
        float(
            hc_row[
                "p2_5"
            ]
        ),
        float(
            w2v_row[
                "p2_5"
            ]
        ),
    ]
)

high = np.array(
    [
        float(
            hc_row[
                "p97_5"
            ]
        ),
        float(
            w2v_row[
                "p97_5"
            ]
        ),
    ]
)

yerr = np.vstack(
    [
        values - low,
        high - values,
    ]
)

x = np.arange(
    len(labels)
)


ax.errorbar(
    x,
    values,
    yerr=yerr,
    fmt="o",
    markersize=9,
    linewidth=1.7,
    capsize=6,
    capthick=1.4,
)


ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)


# Numeric labels
for xi, yi in zip(
    x,
    values,
):

    ax.text(
        xi,
        yi - 0.055,
        f"{yi:.3f}",
        ha="center",
        va="top",
        fontsize=15,
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Dataset identity ROC-AUC"
)

ax.set_ylim(
    0.45,
    1.04,
)

ax.set_xlim(
    -0.45,
    1.45,
)

clean_axes(
    ax
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.97,
    top=0.92,
    bottom=0.17,
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
        / f"Figure6_representation_robustness.{suffix}"
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
