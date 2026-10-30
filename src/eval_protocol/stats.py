# -*- coding: utf-8 -*-
"""
eval_protocol/stats.py
======================
Summaries for honest evaluation:

* pooled out-of-fold R2 / MAE / Spearman (not the mean of per-fold R2, which is
  unstable for tiny test groups),
* **cluster bootstrap** CIs (resample substrates or catalysts, not rows) - with
  only ~5 LOSO clusters a fold SD alone is not an interval,
* paired seed comparison (Wilcoxon signed-rank + paired t),
* decomposition  Delta_total = Delta_structure + Delta_label  with CIs.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats as sps
from sklearn.metrics import mean_absolute_error, r2_score


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(sps.spearmanr(a, b).statistic)


def summarize_oof(y: np.ndarray, oof: np.ndarray) -> Dict[str, float]:
    """``oof`` is (n_seeds, n) or (n,). Metrics averaged over seeds, SD across seeds."""
    oof = np.atleast_2d(oof)
    ok = ~np.isnan(oof[0])
    r2 = np.array([r2_score(y[ok], o[ok]) for o in oof])
    mae = np.array([mean_absolute_error(y[ok], o[ok]) for o in oof])
    sp = np.array([spearman(y[ok], o[ok]) for o in oof])
    return {
        "r2_mean": float(r2.mean()), "r2_seed_sd": float(r2.std(ddof=1)) if len(r2) > 1 else np.nan,
        "mae_mean": float(mae.mean()), "spearman_mean": float(np.nanmean(sp)),
        "n_seeds": int(len(oof)), "n_eval": int(ok.sum()),
    }


def _boot_clusters(clusters: np.ndarray, rng: np.random.RandomState):
    uniq = np.unique(clusters)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return pick


def _r2_seedmean(y, oof, idx):
    return float(np.mean([r2_score(y[idx], o[idx]) for o in oof]))


def cluster_bootstrap_ci(
    y: np.ndarray,
    oof: np.ndarray,
    clusters: Sequence,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    metric=_r2_seedmean,
) -> Dict[str, float]:
    """Percentile CI of seed-mean pooled OOF R2, resampling whole clusters."""
    y = np.asarray(y, dtype=float)
    oof = np.atleast_2d(oof)
    cl = np.asarray(clusters).astype(str)
    ok = ~np.isnan(oof[0])
    members = {c: np.where((cl == c) & ok)[0] for c in np.unique(cl[ok])}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = _boot_clusters(np.array(list(members)), rng)
        idx = np.concatenate([members[c] for c in pick])
        if len(idx) > 2 and np.ptp(y[idx]) > 0:
            vals.append(metric(y, oof, idx))
    vals = np.array(vals)
    full = metric(y, oof, np.where(ok)[0])
    return {"point": full, "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "n_clusters": len(members), "n_boot_used": int(len(vals))}


def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of per-seed scores (same seeds, same splits)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("need paired arrays with >= 2 seeds")
    d = a - b
    out = {"mean_diff": float(d.mean()), "n": int(len(d))}
    out["t_p"] = float(sps.ttest_rel(a, b).pvalue) if np.ptp(d) > 0 else float("nan")
    try:
        out["wilcoxon_p"] = float(sps.wilcoxon(d).pvalue)
    except ValueError:  # all differences zero
        out["wilcoxon_p"] = float("nan")
    return out


def decompose_delta(
    y: np.ndarray,
    oof_by_scope: Dict[str, np.ndarray],
    clusters: Optional[Sequence] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Delta_structure / Delta_label / Delta_total with (optionally cluster-bootstrap) CIs.

    ``oof_by_scope`` must contain keys honest / transductive_x / pretrain_frozen.
    The same cluster resample is applied to all three scopes in each replicate,
    so the intervals respect the pairing.
    """
    need = ("honest", "transductive_x", "pretrain_frozen")
    for k in need:
        if k not in oof_by_scope:
            raise KeyError(f"missing scope {k!r}")
    y = np.asarray(y, float)
    oofs = {k: np.atleast_2d(oof_by_scope[k]) for k in need}
    ok = ~np.isnan(oofs["honest"][0])

    def deltas(idx):
        r = {k: _r2_seedmean(y, oofs[k], idx) for k in need}
        return {"r2_honest": r["honest"], "r2_transductive_x": r["transductive_x"],
                "r2_pretrain_frozen": r["pretrain_frozen"],
                "delta_structure": r["transductive_x"] - r["honest"],
                "delta_label": r["pretrain_frozen"] - r["transductive_x"],
                "delta_total": r["pretrain_frozen"] - r["honest"]}

    point = deltas(np.where(ok)[0])
    out = {k: {"point": v} for k, v in point.items()}
    if clusters is not None and n_boot > 0:
        cl = np.asarray(clusters).astype(str)
        members = {c: np.where((cl == c) & ok)[0] for c in np.unique(cl[ok])}
        rng = np.random.RandomState(seed)
        reps = {k: [] for k in point}
        for _ in range(n_boot):
            pick = _boot_clusters(np.array(list(members)), rng)
            idx = np.concatenate([members[c] for c in pick])
            if len(idx) > 2 and np.ptp(y[idx]) > 0:
                for k, v in deltas(idx).items():
                    reps[k].append(v)
        for k in point:
            arr = np.array(reps[k])
            out[k]["lo"] = float(np.quantile(arr, 0.025))
            out[k]["hi"] = float(np.quantile(arr, 0.975))
    return out
