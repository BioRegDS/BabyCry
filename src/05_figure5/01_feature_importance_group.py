#!/usr/bin/env python3
"""Group-aware permutation importance for Figure 4A."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
N_REPEATS = 100
N_SPLITS = 5
N_PERM_REPEATS = 20
TOP_N = 20


def load_original(path):
    spec = importlib.util.spec_from_file_location(
        "figure4_original", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    project = args.project.resolve()

    original = load_original(
        project / "src/05_figure5/01_feature_importance.py"
    )

    input_dir = project / "data/processed/fixed025"
    outdir = project / "results/05_figure5/groupcv"
    outdir.mkdir(parents=True, exist_ok=True)

    donate = pd.read_csv(input_dir / "donate_cmvn.csv")
    enes = pd.read_csv(input_dir / "enes_cmvn.csv")

    features = sorted(
        set(original.feature_columns(donate))
        & set(original.feature_columns(enes))
    )

    if len(features) != 617:
        raise RuntimeError(
            f"Expected 617 features, found {len(features)}"
        )

    donate = donate.copy()
    enes = enes.copy()

    donate["cv_group"] = (
        "donate:" + donate["group_id"].astype(str)
    )
    enes["cv_group"] = (
        "enes:" + enes["baby"].astype(str)
    )

    # cv_group must not enter the model.
    # Select the 617 validated acoustic features explicitly.
    acoustic_features = [
        c for c in features
        if c.startswith((
            "mfcc_", "dmfcc_", "ddmfcc_",
            "chroma_", "f0_", "voiced_",
            "rms_", "onset_", "spectral_", "zcr_",
        ))
    ]

    if len(acoustic_features) != 617:
        raise RuntimeError(
            "Feature selection includes unexpected columns"
        )

    features = acoustic_features

    repeat_rows = []
    fold_rows = []

    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(SEED + repeat)

        matched = original.reason_matched_sample(
            donate, enes, rng
        )

        y = matched["dataset_target"].to_numpy()
        groups = matched["cv_group"].to_numpy()
        X = matched[features]

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED + 1000 + repeat,
        )

        weighted_importance = np.zeros(len(features))
        weighted_abs_coef = np.zeros(len(features))
        weighted_coef = np.zeros(len(features))

        total_test = 0
        oof_scores = np.full(len(matched), np.nan)

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            overlap = len(train_groups & test_groups)

            if overlap:
                raise RuntimeError(
                    f"Group leakage at repeat={repeat}, fold={fold}"
                )

            if len(np.unique(y[train_idx])) != 2:
                raise RuntimeError(
                    f"Single-class train fold: {repeat}/{fold}"
                )

            if len(np.unique(y[test_idx])) != 2:
                raise RuntimeError(
                    f"Single-class test fold: {repeat}/{fold}"
                )

            model = original.make_model()
            model.fit(X.iloc[train_idx], y[train_idx])

            scores = model.decision_function(
                X.iloc[test_idx]
            )
            oof_scores[test_idx] = scores

            fold_auc = roc_auc_score(y[test_idx], scores)

            perm = permutation_importance(
                model,
                X.iloc[test_idx],
                y[test_idx],
                scoring="roc_auc",
                n_repeats=N_PERM_REPEATS,
                random_state=SEED + 2000 + repeat * 10 + fold,
                n_jobs=-1,
            )

            coef = model.named_steps["model"].coef_[0]

            n_test = len(test_idx)
            weighted_importance += perm.importances_mean * n_test
            weighted_abs_coef += np.abs(coef) * n_test
            weighted_coef += coef * n_test
            total_test += n_test

            fold_rows.append({
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": n_test,
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "group_overlap": overlap,
                "n_test_donate": int(np.sum(y[test_idx] == 0)),
                "n_test_enes": int(np.sum(y[test_idx] == 1)),
                "fold_auc": fold_auc,
            })

        if not np.isfinite(oof_scores).all():
            raise RuntimeError("Non-finite OOF scores")

        auc = roc_auc_score(y, oof_scores)

        weighted_importance /= total_test
        weighted_abs_coef /= total_test
        weighted_coef /= total_test

        for i, feature in enumerate(features):
            repeat_rows.append({
                "repeat": repeat,
                "auc": auc,
                "feature": feature,
                "family": original.family_name(feature),
                "permutation_importance": weighted_importance[i],
                "abs_logreg_coef": weighted_abs_coef[i],
                "logreg_coef": weighted_coef[i],
            })

        if (repeat + 1) % 10 == 0:
            print(
                f"Completed {repeat + 1}/{N_REPEATS}, "
                f"OOF AUC={auc:.4f}",
                flush=True,
            )

    repeats = pd.DataFrame(repeat_rows)
    folds = pd.DataFrame(fold_rows)

    summary = (
        repeats.groupby(["feature", "family"], as_index=False)
        .agg(
            mean_permutation_importance=(
                "permutation_importance", "mean"
            ),
            sd_permutation_importance=(
                "permutation_importance", "std"
            ),
            mean_abs_logreg_coef=(
                "abs_logreg_coef", "mean"
            ),
            mean_logreg_coef=(
                "logreg_coef", "mean"
            ),
        )
    )

    summary["rank_perm"] = summary[
        "mean_permutation_importance"
    ].rank(ascending=False, method="average")

    summary["rank_coef"] = summary[
        "mean_abs_logreg_coef"
    ].rank(ascending=False, method="average")

    summary["mean_rank"] = (
        summary[["rank_perm", "rank_coef"]].mean(axis=1)
    )

    # Permutation importance is the primary ranking criterion.
    summary = summary.sort_values(
        "mean_permutation_importance",
        ascending=False,
    ).reset_index(drop=True)

    repeats.to_csv(
        outdir / "feature_importance_repeat.csv",
        index=False,
    )
    folds.to_csv(
        outdir / "feature_importance_fold_qc.csv",
        index=False,
    )
    summary.to_csv(
        outdir / "feature_importance_summary.csv",
        index=False,
    )

    top = summary.head(TOP_N)
    top.to_csv(outdir / "top_features.csv", index=False)

    print("\n===== GROUP-AWARE FEATURE IMPORTANCE =====")
    print(
        top[[
            "feature",
            "family",
            "mean_permutation_importance",
            "mean_abs_logreg_coef",
        ]].to_string(index=False)
    )

    print("\n===== FOLD QC =====")
    print("Folds:", len(folds))
    print("Group overlap:", folds["group_overlap"].sum())
    print(
        "Both classes in test:",
        int((
            (folds["n_test_donate"] > 0)
            & (folds["n_test_enes"] > 0)
        ).sum()),
        "/", len(folds),
    )

    print("\nSaved:", outdir)


if __name__ == "__main__":
    main()
