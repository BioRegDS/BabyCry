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
    / "03_figure3"
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
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
})


# ============================================================
# Load results
# ============================================================

summary = pd.read_csv(
    RESULTS
    / "cluster_inference_summary.csv"
)


# ============================================================
# Helpers
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.14,
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


def prepare_panel(direction):

    df = (
        summary[
            summary["direction"]
            == direction
        ]
        .copy()
    )

    order = pd.DataFrame(
        [
            {
                "condition": "original",
                "model": "linear_svc",
                "label": "Original\nLinear SVC",
            },
            {
                "condition": "original",
                "model": "logreg",
                "label": "Original\nLogReg",
            },
            {
                "condition": "cmvn",
                "model": "linear_svc",
                "label": "CMVN\nLinear SVC",
            },
            {
                "condition": "cmvn",
                "model": "logreg",
                "label": "CMVN\nLogReg",
            },
        ]
    )

    df = order.merge(
        df,
        on=[
            "condition",
            "model",
        ],
        how="left",
        validate="one_to_one",
    )

    if df["clip_level_auc"].isna().any():
        raise RuntimeError(
            f"Missing Figure 3 rows for {direction}"
        )

    return df


# ============================================================
# Prepare A / B
# ============================================================

donate_to_enes = prepare_panel(
    "donate_to_enes"
)

enes_to_donate = prepare_panel(
    "enes_to_donate"
)


# ============================================================
# Prepare C
# ============================================================

infant_tables = []

for condition, model, display in [
    (
        "original",
        "linear_svc",
        "Original\nLinear SVC",
    ),
    (
        "original",
        "logreg",
        "Original\nLogReg",
    ),
    (
        "cmvn",
        "linear_svc",
        "CMVN\nLinear SVC",
    ),
    (
        "cmvn",
        "logreg",
        "CMVN\nLogReg",
    ),
]:

    path = (
        RESULTS
        / "infant_auc"
        / (
            f"{condition}__"
            f"{model}__"
            "donate_to_enes.csv"
        )
    )

    df = pd.read_csv(path)

    df = (
        df
        .dropna(
            subset=["auc"]
        )
        .copy()
    )

    df["condition_model"] = display

    infant_tables.append(df)


infant_all = pd.concat(
    infant_tables,
    ignore_index=True,
)


# ============================================================
# Figure
# ============================================================

fig = plt.figure(
    figsize=(11.5, 8.8)
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[
        1.0,
        1.05,
    ],
    hspace=0.46,
    wspace=0.34,
)


# ============================================================
# A — Donate -> Enes
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

df = donate_to_enes

x = np.arange(
    len(df)
)

y = df[
    "clip_level_auc"
].to_numpy()

ci_low = df[
    "ci95_low"
].to_numpy()

ci_high = df[
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
    fmt="o",
    markersize=8,
    linewidth=1.6,
    capsize=5,
    capthick=1.3,
    zorder=3,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

# Value labels:
# put above the upper CI, not above point estimate
for xi, yi, upper in zip(
    x,
    y,
    ci_high,
):
    ax.text(
        xi,
        upper + 0.008,
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    df["label"]
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0.40,
    0.60,
)

ax.set_xlim(
    -0.45,
    len(df) - 0.55,
)

clean_axes(ax)


# ============================================================
# B — Enes -> Donate
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

df = enes_to_donate

x = np.arange(
    len(df)
)

y = df[
    "clip_level_auc"
].to_numpy()

ci_low = df[
    "ci95_low"
].to_numpy()

ci_high = df[
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
    fmt="o",
    markersize=8,
    linewidth=1.6,
    capsize=5,
    capthick=1.3,
    zorder=3,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

# Put text above CI
for xi, yi, upper in zip(
    x,
    y,
    ci_high,
):
    ax.text(
        xi,
        upper + 0.012,
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    df["label"]
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_ylim(
    0.35,
    0.70,
)

ax.set_xlim(
    -0.45,
    len(df) - 0.55,
)

clean_axes(ax)


# ============================================================
# C — Infant-specific AUC
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

group_order = [
    "Original\nLinear SVC",
    "Original\nLogReg",
    "CMVN\nLinear SVC",
    "CMVN\nLogReg",
]

group_positions = np.arange(
    len(group_order)
)

rng = np.random.default_rng(42)

for group_index, group_name in enumerate(
    group_order
):

    values = (
        infant_all[
            infant_all[
                "condition_model"
            ]
            == group_name
        ]["auc"]
        .to_numpy()
    )

    # Slightly wider jitter
    jitter = rng.uniform(
        -0.13,
        0.13,
        size=len(values),
    )

    xvals = (
        np.full(
            len(values),
            group_index,
            dtype=float,
        )
        + jitter
    )

    ax.scatter(
        xvals,
        values,
        s=42,
        alpha=0.90,
        zorder=3,
    )

    median_value = float(
        np.median(values)
    )

    # Median segment
    ax.plot(
        [
            group_index - 0.20,
            group_index + 0.20,
        ],
        [
            median_value,
            median_value,
        ],
        linewidth=2.2,
        zorder=4,
    )

    # Put median label above the highest point in the group
    text_y = min(
        0.92,
        np.max(values) + 0.07,
    )

    ax.text(
        group_index,
        text_y,
        f"{median_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
    )


ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    zorder=1,
)

ax.set_xticks(
    group_positions
)

ax.set_xticklabels(
    group_order
)

ax.set_ylabel(
    "Infant-specific ROC-AUC"
)

ax.set_ylim(
    0,
    1,
)

ax.set_xlim(
    -0.45,
    len(group_order) - 0.55,
)

clean_axes(ax)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.10,
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
        / f"Figure3_cross_dataset_transfer.{suffix}"
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
