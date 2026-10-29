#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

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
    "axes.linewidth": 1.2,
})


# ============================================================
# Helpers
# ============================================================

def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fontsize=13,
    linewidth=1.6,
):

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=linewidth,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    linewidth=1.5,
):

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=linewidth,
    )

    ax.add_patch(arrow)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(12.5, 7.2)
)

ax.axis("off")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# ============================================================
# Top: input
# ============================================================

add_box(
    ax,
    0.37,
    0.83,
    0.26,
    0.10,
    "Infant-cry audio",
    fontsize=15,
)


# ============================================================
# Representation
# ============================================================

add_arrow(
    ax,
    0.50,
    0.83,
    0.50,
    0.75,
)

add_box(
    ax,
    0.13,
    0.60,
    0.30,
    0.12,
    "Handcrafted\nacoustic features",
    fontsize=14,
)

add_box(
    ax,
    0.57,
    0.60,
    0.30,
    0.12,
    "wav2vec2\nembeddings",
    fontsize=14,
)

add_arrow(
    ax,
    0.50,
    0.75,
    0.28,
    0.72,
)

add_arrow(
    ax,
    0.50,
    0.75,
    0.72,
    0.72,
)


# ============================================================
# Central question
# ============================================================

add_box(
    ax,
    0.31,
    0.42,
    0.38,
    0.10,
    "What does the model learn?",
    fontsize=15,
    linewidth=1.8,
)

add_arrow(
    ax,
    0.28,
    0.60,
    0.43,
    0.52,
)

add_arrow(
    ax,
    0.72,
    0.60,
    0.57,
    0.52,
)


# ============================================================
# Prediction targets
# ============================================================

add_arrow(
    ax,
    0.50,
    0.42,
    0.22,
    0.33,
)

add_arrow(
    ax,
    0.50,
    0.42,
    0.50,
    0.33,
)

add_arrow(
    ax,
    0.50,
    0.42,
    0.78,
    0.33,
)


add_box(
    ax,
    0.07,
    0.20,
    0.30,
    0.11,
    "Cry reason\nHunger vs Discomfort",
    fontsize=13,
)

add_box(
    ax,
    0.35,
    0.20,
    0.30,
    0.11,
    "Infant identity",
    fontsize=13,
)

add_box(
    ax,
    0.63,
    0.20,
    0.30,
    0.11,
    "Dataset identity",
    fontsize=13,
)


# ============================================================
# Generalization
# ============================================================

add_arrow(
    ax,
    0.22,
    0.20,
    0.22,
    0.12,
)

add_arrow(
    ax,
    0.50,
    0.20,
    0.50,
    0.12,
)

add_arrow(
    ax,
    0.78,
    0.20,
    0.78,
    0.12,
)


ax.text(
    0.22,
    0.07,
    "Across sessions",
    ha="center",
    va="center",
    fontsize=12.5,
)

ax.text(
    0.50,
    0.07,
    "Across infants",
    ha="center",
    va="center",
    fontsize=12.5,
)

ax.text(
    0.78,
    0.07,
    "Across datasets",
    ha="center",
    va="center",
    fontsize=12.5,
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
        / f"Figure1_workflow.{suffix}"
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

    print("Saved:", output)


plt.close(fig)
