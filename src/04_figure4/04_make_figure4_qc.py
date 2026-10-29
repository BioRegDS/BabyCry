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
# Load
# ============================================================

identity = pd.read_csv(
    RESULTS
    / "groupcv_qc"
    / "dataset_identity_groupcv_summary.csv"
)

family = pd.read_csv(
    RESULTS
    / "feature_family_groupcv_all"
    / "feature_family_groupcv_summary.csv"
)

comparison = pd.read_csv(
    RESULTS
    / "reason_vs_dataset_summary_qc.csv"
)


# ============================================================
# Helpers
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.12,
    y=1.16,
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
    figsize=(11.5, 8.8)
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[
        1.0,
        1.08,
    ],
    hspace=0.46,
    wspace=0.48,
)


# ============================================================
# A
# Dataset identity: Original vs CMVN
# ============================================================

ax = fig.add_subplot(
    gs[0, 0]
)

add_panel_label(
    ax,
    "A",
    x=-0.16,
    y=1.16,
)

plot_order = [
    ("original", "linear_svc", "Original\nLinear SVC"),
    ("original", "logreg", "Original\nLogReg"),
    ("cmvn", "linear_svc", "CMVN\nLinear SVC"),
    ("cmvn", "logreg", "CMVN\nLogReg"),
]

rows = []

for condition, model, label in plot_order:

    row = identity[
        (identity["condition"] == condition)
        &
        (identity["model"] == model)
    ].iloc[0]

    rows.append(
        {
            "label": label,
            "auc": row["mean_auc"],
            "low": (row["mean_auc"] - row["sd_auc"]),
            "high": (row["mean_auc"] + row["sd_auc"]),
        }
    )

df = pd.DataFrame(rows)

x = np.arange(
    len(df)
)

y = df[
    "auc"
].to_numpy()

low = df[
    "low"
].to_numpy()

high = df[
    "high"
].to_numpy()

yerr = np.vstack(
    [
        y - low,
        high - y,
    ]
)

ax.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    markersize=8,
    linewidth=1.6,
    capsize=5,
    capthick=1.3,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)

for xi, yi in zip(
    x,
    y,
):

    ax.text(
        xi,
        yi + 0.012,
        f"{yi:.3f}",
        ha="center",
        va="bottom",
        fontsize=13,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    df["label"]
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0.45,
    1.04,
)

ax.set_xlim(
    -0.45,
    len(df) - 0.55,
)

clean_axes(ax)


# ============================================================
# B
# Feature-family-specific dataset identity
# CMVN + Linear SVC
# ============================================================

ax = fig.add_subplot(
    gs[0, 1]
)

add_panel_label(
    ax,
    "B",
    x=-0.16,
    y=1.16,
)

family_plot = family[
    (
        family["model"]
        == "linear_svc"
    )
    &
    (
        family["family"]
        != "ALL_FEATURES"
    )
].copy()

family_order = [
    "DeltaDeltaMFCC",
    "Spectral",
    "MFCC",
    "DeltaMFCC",
    "Pitch_Voicing",
    "Temporal_Onset",
    "Chroma",
    "Energy",
]

family_labels = {
    "DeltaDeltaMFCC": "ΔΔ-MFCC",
    "Spectral": "Spectral",
    "MFCC": "MFCC",
    "DeltaMFCC": "Δ-MFCC",
    "Pitch_Voicing": "Pitch/voicing",
    "Temporal_Onset": "Temporal/onset",
    "Chroma": "Chroma",
    "Energy": "Energy",
}

family_plot = (
    family_plot
    .set_index("family")
    .loc[family_order]
    .reset_index()
)

y_pos = np.arange(
    len(family_plot)
)

auc = family_plot[
    "mean_auc"
].to_numpy()

low = (family_plot["mean_auc"] - family_plot["sd_auc"]).to_numpy()

high = (family_plot["mean_auc"] + family_plot["sd_auc"]).to_numpy()

xerr = np.vstack(
    [
        auc - low,
        high - auc,
    ]
)

ax.errorbar(
    auc,
    y_pos,
    xerr=xerr,
    fmt="o",
    markersize=7,
    linewidth=1.5,
    capsize=4,
    capthick=1.2,
)

ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)

ax.set_yticks(
    y_pos
)

ax.set_yticklabels(
    [
        family_labels[x]
        for x in family_plot["family"]
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Dataset identity ROC-AUC"
)

ax.set_xlim(
    0.35,
    1.03,
)

# value labels
for yi, value in zip(
    y_pos,
    auc,
):

    if value > 0.92:
        text_x = value - 0.035
        ha = "right"
    else:
        text_x = value + 0.025
        ha = "left"

    ax.text(
        text_x,
        yi - 0.28,
        f"{value:.2f}",
        ha=ha,
        va="center",
        fontsize=13,
    )

clean_axes(ax)


# ============================================================
# C
# Reason transfer vs dataset identity
# ============================================================

ax = fig.add_subplot(
    gs[1, :]
)

add_panel_label(
    ax,
    "C",
    x=-0.065,
    y=1.16,
)

comparison_order = [
    "Reason transfer\nDonate → Enes",
    "Reason transfer\nEnes → Donate",
    "Dataset identity\nDonate vs Enes",
]

cp = (
    comparison
    .set_index("analysis")
    .loc[
        comparison_order
    ]
    .reset_index()
)

x = np.arange(
    len(cp)
)

y = cp[
    "auc"
].to_numpy()

low = cp[
    "ci95_low"
].to_numpy()

high = np.minimum(
    cp["ci95_high"].to_numpy(),
    1.0,
)

yerr = np.vstack(
    [
        y - low,
        high - y,
    ]
)

ax.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    markersize=9,
    linewidth=1.7,
    capsize=6,
    capthick=1.4,
    zorder=3,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

# Numeric labels:
# place above upper CI except near 1.0
for xi, yi, upper in zip(
    x,
    y,
    high,
):

    if yi > 0.9:

        ax.text(
            xi,
            yi - 0.065,
            f"{yi:.3f}",
            ha="center",
            va="top",
            fontsize=15,
        )

    else:

        ax.text(
            xi,
            min(
                0.96,
                upper + 0.055,
            ),
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

ax.set_xticks(x)

ax.set_xticklabels(
    comparison_order
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0.25,
    1.05,
)

ax.set_xlim(
    -0.55,
    len(cp) - 0.45,
)

clean_axes(ax)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.97,
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
        / f"Figure4_dataset_identity_qc.{suffix}"
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
