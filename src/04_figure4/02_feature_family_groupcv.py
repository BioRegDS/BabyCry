#!/usr/bin/env python3
"""Group-aware CV for selected acoustic feature families."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    project = args.project.resolve()
    src = project / "src/04_figure4"

    original = load_module(
        src / "01_dataset_identity.py", "dataset_identity_original"
    )
    family_mod = load_module(
        src / "02_feature_family_identity.py", "feature_family_original"
    )
    group_mod = load_module(
        src / "01_dataset_identity_groupcv.py", "dataset_identity_groupcv"
    )

    outdir = project / "results/04_figure4/feature_family_groupcv"
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    fold_results = []

    for condition in ["cmvn"]:
        indir = project / "data/processed/fixed025"
        donate = pd.read_csv(indir / f"donate_{condition}.csv")
        enes = pd.read_csv(indir / f"enes_{condition}.csv")

        common_features = sorted(
            set(original.feature_columns(donate))
            & set(original.feature_columns(enes))
        )

        for family in ["DeltaDeltaMFCC", "Spectral"]:
            features = [
                f for f in common_features
                if family_mod.family_name(f) == family
            ]

            expected = {
                "DeltaDeltaMFCC": 140,
                "Spectral": 84,
            }[family]

            if len(features) != expected:
                raise ValueError(
                    f"{family}: expected {expected}, found {len(features)}"
                )

            print(f"\n=== {family}: {len(features)} features ===", flush=True)

            for repeat in range(args.repeats):
                rng = np.random.default_rng(original.SEED + repeat)

                matched, _ = original.reason_matched_sample(
                    donate, enes, rng
                )

                for model_name in ["linear_svc", "logreg"]:
                    seed = original.SEED + 1000 + repeat

                    auc, folds = group_mod.group_cv_auc(
                        matched, features, model_name, seed, original
                    )

                    if not np.isfinite(auc):
                        raise ValueError(
                            f"Non-finite AUC: {family}, {model_name}, {repeat}"
                        )

                    results.append({
                        "condition": condition,
                        "family": family,
                        "n_features": len(features),
                        "model": model_name,
                        "repeat": repeat,
                        "auc": auc,
                    })

                    for fold in folds:
                        fold_results.append({
                            "condition": condition,
                            "family": family,
                            "model": model_name,
                            "repeat": repeat,
                            **fold,
                        })

                if (repeat + 1) % 20 == 0:
                    print(
                        f"{family}: {repeat + 1}/{args.repeats}",
                        flush=True
                    )

    detail = pd.DataFrame(results)
    fold_detail = pd.DataFrame(fold_results)

    summary = (
        detail.groupby(["condition", "family", "n_features", "model"])
        .agg(
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
            n_valid=("auc", "count"),
        )
        .reset_index()
    )

    detail.to_csv(
        outdir / "feature_family_groupcv_repeat_results.csv",
        index=False
    )
    fold_detail.to_csv(
        outdir / "feature_family_groupcv_fold_qc.csv",
        index=False
    )
    summary.to_csv(
        outdir / "feature_family_groupcv_summary.csv",
        index=False
    )

    print("\n=== GROUP-AWARE FEATURE FAMILY SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved to:", outdir)


if __name__ == "__main__":
    main()
