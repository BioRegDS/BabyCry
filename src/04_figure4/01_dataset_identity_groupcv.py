#!/usr/bin/env python3
"""Group-disjoint dataset identity QC; imports original script without modifying it."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def load_original(path):
    spec = importlib.util.spec_from_file_location('babycry_dataset_identity_original', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_cv_auc(df, features, model_name, seed, original):
    y = df["dataset_target"].to_numpy()

    # Check group identifiers before constructing groups.
    if df.loc[df["dataset_target"] == 0, "group_id"].isna().any():
        raise ValueError("Missing Donate group_id")

    if df.loc[df["dataset_target"] == 1, "baby"].isna().any():
        raise ValueError("Missing Enes baby")

    # Dataset-specific, group-disjoint identifiers.
    groups = np.empty(len(df), dtype=object)

    donate_mask = y == 0
    enes_mask = y == 1

    groups[donate_mask] = (
        "donate:"
        + df.loc[donate_mask, "group_id"].astype(str)
    ).to_numpy()
    
    groups[enes_mask] = (
        "enes:"
        + df.loc[enes_mask, "baby"].astype(str)
    ).to_numpy()
    
    # Verify that every group identifier is valid.
    assert all(isinstance(g, str) for g in groups)
    assert np.isin(y, [0, 1]).all()
    
    scores = np.full(len(df), np.nan)
    fold_rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_groups = set(groups[train_idx]); test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups), 'Group leakage!'
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            raise ValueError(f'Fold {fold} has only one dataset class')
        model = original.make_model(model_name, seed + fold)
        model.fit(df.iloc[train_idx][features], y[train_idx])
        scores[test_idx] = model.decision_function(df.iloc[test_idx][features])
        fold_rows.append({'fold': fold, 'n_train': len(train_idx), 'n_test': len(test_idx),
                          'n_train_groups': len(train_groups), 'n_test_groups': len(test_groups),
                          'n_test_donate': int((y[test_idx] == 0).sum()),
                          'n_test_enes': int((y[test_idx] == 1).sum()),
                          'n_test_hunger': int((df.iloc[test_idx]['label'] == 'hunger').sum()),
                          'n_test_discomfort': int((df.iloc[test_idx]['label'] == 'discomfort').sum())})
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite OOF scores')
    return float(roc_auc_score(y, scores)), fold_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    project = args.project.expanduser().resolve()
    original = load_original(project / 'src/04_figure4/01_dataset_identity.py')
    indir = project / 'data/processed/fixed025'
    outdir = project / 'results/04_figure4/groupcv_qc'
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []; fold_rows = []
    for condition in ('original', 'cmvn'):
        donate = pd.read_csv(indir / f'donate_{condition}.csv')
        enes = pd.read_csv(indir / f'enes_{condition}.csv')
        assert set(original.feature_columns(donate)) == set(original.feature_columns(enes))
        features = sorted(original.feature_columns(donate))
        assert len(features) == original.EXPECTED_FEATURES, len(features)
        for repeat in range(args.repeats):
            rng = np.random.default_rng(original.SEED + repeat)
            matched, n_each = original.reason_matched_sample(donate, enes, rng)
            for model_name in ('linear_svc', 'logreg'):
                seed = original.SEED + 1000 + repeat
                try:
                    auc, folds = group_cv_auc(matched, features, model_name, seed, original)
                    status = 'ok'; error = ''
                except ValueError as exc:
                    auc = np.nan; folds = []; status = 'invalid_fold'; error = str(exc)
                rows.append({'condition': condition, 'model': model_name, 'repeat': repeat,
                             'n_per_reason_per_dataset': n_each, 'n_total': len(matched),
                             'n_features': len(features), 'auc': auc, 'status': status, 'error': error})
                for f in folds:
                    fold_rows.append({'condition': condition, 'model': model_name,
                                      'repeat': repeat, **f})
            if (repeat + 1) % 10 == 0:
                print(f'{condition}: {repeat + 1}/{args.repeats}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(outdir / 'dataset_identity_groupcv_repeat_results.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / 'dataset_identity_groupcv_fold_qc.csv', index=False)
    summary = (results.groupby(['condition', 'model'], as_index=False)
               .agg(mean_auc=('auc', 'mean'), sd_auc=('auc', 'std'),
                    n_valid=('auc', 'count'), n_attempted=('repeat', 'size')))
    summary.to_csv(outdir / 'dataset_identity_groupcv_summary.csv', index=False)
    print('\n=== GROUP CV SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== INVALID SPLITS ===')
    print(results[results.status != 'ok'][['condition', 'model', 'repeat', 'error']].to_string(index=False))
    print(f'\nSaved to: {outdir}')


if __name__ == '__main__':
    main()
