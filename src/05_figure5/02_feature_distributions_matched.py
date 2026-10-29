#!/usr/bin/env python3
"""Reason-matched acoustic feature distributions for Figure 4B."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

SEED = 42
N_REPEATS = 100
N_REPRESENTATIVE = 6


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_features(importance):
    importance = importance.sort_values(
        "mean_permutation_importance",
        ascending=False,
    )

    selected = []
    used_bases = set()

    for feature in importance["feature"]:
        parts = feature.split("_")
        base = "_".join(parts[:-1]) if len(parts) >= 3 else feature

        if base in used_bases:
            continue

        selected.append(feature)
        used_bases.add(base)

        if len(selected) == N_REPRESENTATIVE:
            break

    if len(selected) != N_REPRESENTATIVE:
        raise RuntimeError("Insufficient representative features")

    return selected


def calculate_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        raise RuntimeError("Insufficient valid feature values")

    pooled = np.concatenate([x, y])
    sd = np.std(pooled, ddof=1)

    return {
        "n_donate": len(x),
        "n_enes": len(y),
        "donate_median": float(np.median(x)),
        "enes_median": float(np.median(y)),
        "wasserstein_std": (
            float(wasserstein_distance(x, y) / sd)
            if sd > 0 else np.nan
        ),
        "ks_stat": float(ks_2samp(x, y).statistic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    project = args.project.resolve()
    input_dir = project / "data/processed/fixed025"
    outdir = project / "results/05_figure5/matched"
    outdir.mkdir(parents=True, exist_ok=True)

    original = load_module(
        project / "src/05_figure5/01_feature_importance.py",
        "figure4_importance",
    )

    distribution_original = load_module(
        project / "src/05_figure5/02_feature_distributions.py",
        "figure4_distribution",
    )

    donate = pd.read_csv(input_dir / "donate_cmvn.csv")
    enes = pd.read_csv(input_dir / "enes_cmvn.csv")

    importance = pd.read_csv(
        project
        / "results/05_figure5/groupcv"
        / "feature_importance_summary.csv"
    )

    selected = select_features(importance)

    representative = importance.set_index("feature").loc[
        selected
    ].reset_index()

    representative["selection_order"] = np.arange(len(selected))

    representative.to_csv(
        outdir / "representative_features.csv",
        index=False,
    )

    repeat_rows = []
    plot_rows = []

    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(SEED + repeat)

        matched = original.reason_matched_sample(
            donate, enes, rng
        )

        if len(matched) != 108:
            raise RuntimeError("Unexpected matched sample size")

        counts = matched.groupby(
            ["dataset_target", "label"]
        ).size()

        if len(counts) != 4 or not (counts == 27).all():
            raise RuntimeError(
                f"Reason matching failed: {counts.to_dict()}"
            )

        for feature in selected:
            x = pd.to_numeric(
                matched.loc[
                    matched["dataset_target"] == 0, feature
                ],
                errors="coerce",
            ).to_numpy()

            y = pd.to_numeric(
                matched.loc[
                    matched["dataset_target"] == 1, feature
                ],
                errors="coerce",
            ).to_numpy()

            stats = calculate_stats(x, y)

            stats["cliffs_delta_enes_minus_donate"] = (
                distribution_original.cliffs_delta(x, y)
            )

            repeat_rows.append({
                "repeat": repeat,
                "feature": feature,
                **stats,
            })

            if repeat == 0:
                for dataset_target, dataset_name in [
                    (0, "Donate-a-Cry"),
                    (1, "Enes"),
                ]:
                    values = pd.to_numeric(
                        matched.loc[
                            matched["dataset_target"] == dataset_target,
                            feature,
                        ],
                        errors="coerce",
                    ).dropna()

                    for value in values:
                        plot_rows.append({
                            "feature": feature,
                            "dataset": dataset_name,
                            "value": float(value),
                        })

        if (repeat + 1) % 20 == 0:
            print(
                f"Completed {repeat + 1}/{N_REPEATS}",
                flush=True,
            )

    repeats = pd.DataFrame(repeat_rows)
    plot_data = pd.DataFrame(plot_rows)

    metric_cols = [
        "wasserstein_std",
        "ks_stat",
        "cliffs_delta_enes_minus_donate",
    ]

    summary_rows = []

    for feature, sub in repeats.groupby("feature"):
        row = {"feature": feature}

        for metric in metric_cols:
            values = sub[metric].dropna().to_numpy()

            row[f"{metric}_median"] = float(
                np.median(values)
            )
            row[f"{metric}_p2_5"] = float(
                np.percentile(values, 2.5)
            )
            row[f"{metric}_p97_5"] = float(
                np.percentile(values, 97.5)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary = representative[
        ["feature", "selection_order"]
    ].merge(
        summary,
        on="feature",
        validate="one_to_one",
    ).sort_values("selection_order")

    repeats.to_csv(
        outdir / "representative_feature_stats_repeats.csv",
        index=False,
    )

    summary.to_csv(
        outdir / "representative_feature_stats_summary.csv",
        index=False,
    )

    plot_data.to_csv(
        outdir / "representative_feature_data.csv",
        index=False,
    )

    print("\n===== REPRESENTATIVE FEATURES =====")
    print(
        representative[[
            "feature",
            "family",
            "mean_permutation_importance",
            "selection_order",
        ]].to_string(index=False)
    )

    print("\n===== MATCHED DISTRIBUTION SUMMARY =====")
    print(
        summary[[
            "feature",
            "wasserstein_std_median",
            "ks_stat_median",
            "cliffs_delta_enes_minus_donate_median",
        ]].to_string(index=False)
    )

    print("\n===== QC =====")
    print("Repeats:", repeats["repeat"].nunique())
    print("Features:", repeats["feature"].nunique())
    print("Statistic rows:", len(repeats))
    print("Plot rows:", len(plot_data))
    print("Missing statistic values:",
          repeats[metric_cols].isna().sum().to_dict())

    print("\nSaved:", outdir)


if __name__ == "__main__":
    main()
