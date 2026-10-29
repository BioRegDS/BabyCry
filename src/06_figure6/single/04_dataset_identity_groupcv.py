#!/usr/bin/env python3
"""Group-aware dataset identity CV for single-audio wav2vec2 embeddings."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
N_REPEATS = 200
N_SPLITS = 5


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    project = args.project.resolve()

    original = load_module(
        project / "src/06_figure6/03_dataset_identity.py",
        "wav2vec2_identity",
    )

    input_dir = project / "data/processed/wav2vec2_single"
    outdir = project / "results/06_figure6/single/groupcv"
    outdir.mkdir(parents=True, exist_ok=True)

    donate = pd.read_csv(input_dir / "donate_embeddings.csv")
    enes = pd.read_csv(input_dir / "enes_embeddings.csv")

    features = sorted(original.embedding_columns(donate))

    if set(features) != set(original.embedding_columns(enes)):
        raise RuntimeError("Embedding feature columns differ")

    if "group_id" not in donate.columns:
        raise RuntimeError("Donate group_id is missing")
    if "baby" not in enes.columns:
        raise RuntimeError("Enes baby is missing")

    if donate["group_id"].isna().any():
        raise RuntimeError("Missing Donate group IDs")
    if enes["baby"].isna().any():
        raise RuntimeError("Missing Enes infant IDs")

    # Prefix group IDs to prevent accidental collisions.
    donate = donate.copy()
    enes = enes.copy()

    donate["cv_group"] = "donate:" + donate["group_id"].astype(str)
    enes["cv_group"] = "enes:" + enes["baby"].astype(str)

    rows = []
    fold_rows = []

    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(SEED + repeat)

        matched = original.reason_matched_sample(
            donate, enes, rng
        )

        y = matched["dataset_target"].to_numpy()
        groups = matched["cv_group"].to_numpy()

        if len(matched) != 108:
            raise RuntimeError(
                f"Unexpected sample size: {len(matched)}"
            )

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED + 1000 + repeat,
        )

        splits = list(splitter.split(matched, y, groups))

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            overlap = train_groups & test_groups

            if overlap:
                raise RuntimeError(
                    f"Group leakage: repeat={repeat}, fold={fold}"
                )

            if len(np.unique(y[train_idx])) < 2:
                raise RuntimeError(
                    f"Single-class training fold: {repeat}/{fold}"
                )

            fold_rows.append({
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "group_overlap": len(overlap),
                "n_test_donate": int(np.sum(y[test_idx] == 0)),
                "n_test_enes": int(np.sum(y[test_idx] == 1)),
            })

        for model_name in ["linear_svc", "logreg"]:
            scores = np.full(len(matched), np.nan)

            for fold, (train_idx, test_idx) in enumerate(splits, 1):
                model = original.make_model(
                    model_name, SEED + repeat * 10 + fold
                )

                model.fit(
                    matched.iloc[train_idx][features],
                    y[train_idx],
                )

                scores[test_idx] = model.decision_function(
                    matched.iloc[test_idx][features]
                )

            if not np.isfinite(scores).all():
                raise RuntimeError("Missing or non-finite predictions")

            auc = roc_auc_score(y, scores)

            rows.append({
                "representation": "wav2vec2_single",
                "model": model_name,
                "repeat": repeat,
                "n_total": len(matched),
                "n_features": len(features),
                "n_groups": len(np.unique(groups)),
                "auc": float(auc),
            })

        if (repeat + 1) % 20 == 0:
            print(f"Completed {repeat + 1}/{N_REPEATS}", flush=True)

    repeats = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)

    summary = (
        repeats.groupby("model")["auc"]
        .agg(
            mean_auc="mean",
            sd_auc="std",
            median_auc="median",
            n_repeats="count",
        )
        .reset_index()
    )

    for model_name in summary["model"]:
        values = repeats.loc[
            repeats["model"] == model_name, "auc"
        ].to_numpy()

        idx = summary["model"] == model_name
        summary.loc[idx, "p2_5"] = np.percentile(values, 2.5)
        summary.loc[idx, "p97_5"] = np.percentile(values, 97.5)

    repeats.to_csv(
        outdir / "dataset_identity_groupcv_repeats.csv",
        index=False,
    )

    folds.to_csv(
        outdir / "dataset_identity_groupcv_fold_qc.csv",
        index=False,
    )

    summary.to_csv(
        outdir / "dataset_identity_groupcv_summary.csv",
        index=False,
    )

    print("\n===== GROUP-AWARE DATASET IDENTITY =====")
    print(summary.to_string(index=False))

    print("\n===== FOLD QC =====")
    print("Folds:", len(folds))
    print("Group overlap:", folds["group_overlap"].sum())
    print(
        "Test folds containing both classes:",
        int(
            (
                (folds["n_test_donate"] > 0)
                & (folds["n_test_enes"] > 0)
            ).sum()
        ),
        "/", len(folds),
    )


if __name__ == "__main__":
    main()
