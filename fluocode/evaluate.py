import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

log = logging.getLogger(__name__)

TARGETS = ["ex_max", "em_max", "ext_coeff", "qy"]


def fit_scalers(tgt_df):
    return {col: StandardScaler().fit(tgt_df[col].dropna().values.reshape(-1, 1))
            for col in tgt_df.columns}


def scale_targets(tgt_df, scalers):
    out = np.full(tgt_df.shape, np.nan, dtype=np.float64)
    for i, col in enumerate(tgt_df.columns):
        mask = tgt_df[col].notna()
        if mask.any():
            out[mask.values, i] = scalers[col].transform(
                tgt_df.loc[mask, col].values.reshape(-1, 1)
            ).ravel()
    return out


def train_xgb_regressors(X_tr, X_val, X_te,
                          y_tr, y_val, y_te,
                          scalers, targets, xgb_params,
                          checkpoint_dir=None):
    metrics = {}
    for i, target in enumerate(targets):
        trm  = ~np.isnan(y_tr[:, i])
        valm = ~np.isnan(y_val[:, i])
        tem  = ~np.isnan(y_te[:, i])

        if trm.sum() < 10:
            log.info(f"  {target}: skipped ({trm.sum()} labels)")
            metrics[target] = None
            continue

        log.info(f"  {target}: {trm.sum()} train / {valm.sum()} val / {tem.sum()} test")
        bst = xgb.XGBRegressor(**xgb_params)
        bst.fit(X_tr[trm], y_tr[trm, i],
                eval_set=[(X_val[valm], y_val[valm, i])],
                verbose=False)

        if tem.sum() < 5:
            metrics[target] = {"r2": float("nan"), "spearman": float("nan"),
                               "n": int(tem.sum())}
            continue

        preds  = bst.predict(X_te[tem])
        p_orig = scalers[target].inverse_transform(preds.reshape(-1, 1)).ravel()
        t_orig = scalers[target].inverse_transform(y_te[tem, i].reshape(-1, 1)).ravel()

        r2  = r2_score(t_orig, p_orig)
        spr = stats.spearmanr(t_orig, p_orig).statistic
        metrics[target] = {"r2": round(float(r2), 4),
                           "spearman": round(float(spr), 4),
                           "n": int(tem.sum())}
        log.info(f"    R²={r2:.4f}  Spearman={spr:.4f}")

        if checkpoint_dir:
            bst.save_model(str(Path(checkpoint_dir) / f"xgb_{target}.json"))

    return metrics


def aggregate_cv_results(fold_results, targets):
    summary = {}
    for target in targets:
        vals = [(r[target]["spearman"], r[target]["r2"], r[target]["n"])
                for r in fold_results
                if r.get(target) and not np.isnan(r[target].get("spearman", float("nan")))]
        if vals:
            sp, r2, ns = zip(*vals)
            summary[target] = {
                "spearman_mean": round(float(np.mean(sp)), 4),
                "spearman_std":  round(float(np.std(sp)),  4),
                "r2_mean":       round(float(np.mean(r2)), 4),
                "r2_std":        round(float(np.std(r2)),  4),
                "n_folds":       len(vals),
                "n_total":       sum(ns),
            }
        else:
            summary[target] = None
    return summary


def print_cv_summary(summary, targets, model_name=""):
    label = f"CV results — {model_name}" if model_name else "CV results"
    log.info(f"\n{label}")
    log.info("-" * 60)
    for target in targets:
        m = summary.get(target)
        if m:
            log.info(f"  {target:12s}  "
                     f"Spearman {m['spearman_mean']:.4f} ± {m['spearman_std']:.4f}  "
                     f"R² {m['r2_mean']:.4f} ± {m['r2_std']:.4f}  "
                     f"({m['n_folds']} folds, n={m['n_total']})")
        else:
            log.info(f"  {target:12s}  skipped")
