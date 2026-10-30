# -*- coding: utf-8 -*-
"""
eval_protocol/controls.py
=========================
Negative / dose-response controls that turn "the honest and frozen R2 differ"
into a mechanistic statement.

1. ``label_permutation_control``
   The encoder is given a *shuffled* label vector (``y_enc``) while the
   regressor still trains on the true labels of the training rows.  If the
   frozen-vs-transductive gap (Delta_label) is really caused by the encoder
   having memorised the test labels, it must vanish when the labels the encoder
   sees carry no information about the rows.  The result is compared with the
   real-label gap through an empirical p-value.

2. ``lambda_dose_response``
   Delta_label as a function of the property-loss weight lambda.  lambda = 0 is
   an exact zero (the label never enters the encoder loss); a monotone increase
   with lambda is the dose-response expected from label leakage.

3. ``permutation_verdict``
   The feature-permutation (SHAP) verdict shared by
   ``permuted_feature_ablation*.py`` (previously copy-pasted three times).

Near-duplicate auditing lives in ``audit.py``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .harness import evaluate


def _r2_mean(res: Dict) -> float:
    return float(res["per_seed"]["r2_oof"].mean())


def _delta_label(common: Dict, enc_f, reg_f, y_enc: Optional[np.ndarray]) -> Dict[str, float]:
    """Delta_label = R2(pretrain_frozen) - R2(transductive_x) for a given encoder label vector."""
    tx = evaluate(fit_scope="transductive_x", encoder_factory=enc_f,
                  regressor_factory=reg_f, y_enc=y_enc, **common)
    fr = evaluate(fit_scope="pretrain_frozen", encoder_factory=enc_f,
                  regressor_factory=reg_f, y_enc=y_enc, **common)
    r_tx, r_fr = _r2_mean(tx), _r2_mean(fr)
    return {"r2_transductive_x": r_tx, "r2_pretrain_frozen": r_fr, "delta_label": r_fr - r_tx}


def label_permutation_control(
    X_enc: np.ndarray,
    X_aux: Optional[np.ndarray],
    y: np.ndarray,
    splits: Sequence,
    encoder_factory: Callable[[int], object],
    regressor_factory: Callable[[int], object],
    seeds: Sequence[int] = (0,),
    n_perm: int = 5,
    rng_seed: int = 0,
    inner_groups: Optional[np.ndarray] = None,
) -> Dict:
    """Real-label vs shuffled-label Delta_label.

    Returns ``{"real": {...}, "perm": DataFrame, "p_value": float, ...}``.
    ``p_value`` is the share of permutations whose Delta_label is >= the real
    one (add-one smoothed), i.e. how often a label-free encoder produces a gap
    this large by chance.
    """
    y = np.asarray(y, dtype=np.float32)
    common = dict(X_enc=X_enc, X_aux=X_aux, y=y, splits=splits, seeds=seeds,
                  inner_groups=inner_groups)
    real = _delta_label(common, encoder_factory, regressor_factory, None)
    rng = np.random.RandomState(rng_seed)
    rows = []
    for b in range(int(n_perm)):
        y_perm = y[rng.permutation(len(y))]
        r = _delta_label(common, encoder_factory, regressor_factory, y_perm)
        r["perm"] = b
        rows.append(r)
    perm = pd.DataFrame(rows)
    d_perm = perm["delta_label"].to_numpy()
    p = float((1 + np.sum(d_perm >= real["delta_label"])) / (1 + len(d_perm)))
    return {
        "real": real,
        "perm": perm,
        "perm_mean": float(d_perm.mean()) if len(d_perm) else float("nan"),
        "perm_sd": float(d_perm.std(ddof=1)) if len(d_perm) > 1 else float("nan"),
        "p_value": p,
        "n_perm": int(n_perm),
        "interpretation": (
            "Delta_label(real) >> Delta_label(shuffled)  => the gap is carried by the labels the "
            "encoder saw (label leakage).  Comparable values => the gap is not a label effect."
        ),
    }


def _spearman_exact_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation p of Spearman's rho (enumerates all n! pairings; n <= 8)."""
    from itertools import permutations

    ra, rb = sps.rankdata(a), sps.rankdata(b)
    obs = abs(float(np.corrcoef(ra, rb)[0, 1]))
    hits = total = 0
    for perm in permutations(range(len(rb))):
        total += 1
        if abs(float(np.corrcoef(ra, rb[list(perm)])[0, 1])) >= obs - 1e-12:
            hits += 1
    return hits / total


def lambda_dose_response(
    X_enc: np.ndarray,
    X_aux: Optional[np.ndarray],
    y: np.ndarray,
    splits: Sequence,
    encoder_factory: Callable[[float, int], object],      # (lambda, seed) -> encoder
    regressor_factory: Callable[[int], object],
    lambdas: Sequence[float] = (0.0, 10.0, 50.0, 150.0, 500.0),
    seeds: Sequence[int] = (0,),
    inner_groups: Optional[np.ndarray] = None,
) -> Dict:
    """Delta_label vs lambda (+ honest R2 for reference) and a Spearman trend test."""
    y = np.asarray(y, dtype=np.float32)
    common = dict(X_enc=X_enc, X_aux=X_aux, y=y, splits=splits, seeds=seeds,
                  inner_groups=inner_groups)
    rows = []
    for lam in lambdas:
        ef = (lambda seed, _l=float(lam): encoder_factory(_l, seed))
        hon = evaluate(fit_scope="honest", encoder_factory=ef,
                       regressor_factory=regressor_factory, **common)
        r = _delta_label(common, ef, regressor_factory, None)
        r.update({"lambda": float(lam), "r2_honest": _r2_mean(hon)})
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    pos = tab[tab["lambda"] > 0]
    if len(pos) >= 3 and pos["delta_label"].nunique() > 1:
        rho, p = sps.spearmanr(pos["lambda"], pos["delta_label"])
        if len(pos) <= 8:
            # scipy's t-approximation gives p = 0 at |rho| = 1 for n = 4 (the default
            # grid); use the exact permutation p instead (2 / n! for a perfect trend).
            p = _spearman_exact_p(pos["lambda"].to_numpy(), pos["delta_label"].to_numpy())
        trend = {"spearman_rho": float(rho), "p_value": float(p), "n_points": int(len(pos))}
    else:
        trend = {"spearman_rho": float("nan"), "p_value": float("nan")}
    zero = tab[tab["lambda"] == 0]
    return {
        "table": tab,
        "trend": trend,
        "delta_label_at_lambda0": float(zero["delta_label"].iloc[0]) if len(zero) else float("nan"),
    }


def permutation_verdict(original: float, perm_values: Sequence[float],
                        alpha: float = 0.05) -> Dict[str, object]:
    """Genuine / ambiguous call for a permutation-null of a scalar statistic.

    ``genuine``  iff the original lies outside the 95 % permutation interval AND
    the two-sided empirical p < ``alpha``.  Identical to the previous inline
    logic of ``permuted_feature_ablation{,_v2,_v3_controls}.py``.
    """
    pm = np.asarray(perm_values, dtype=np.float64)
    mean = float(pm.mean())
    sd = float(pm.std(ddof=1)) if len(pm) > 1 else float("nan")
    lo, hi = float(np.percentile(pm, 2.5)), float(np.percentile(pm, 97.5))
    p = float((np.abs(pm) >= abs(original)).mean())
    in_ci = bool(lo <= original <= hi)
    z = (original - mean) / (sd + 1e-12)
    return {
        "permuted_mean": mean, "permuted_std": sd, "ci_low": lo, "ci_high": hi,
        "p_value": p, "z_score": float(z), "original_in_95ci": in_ci,
        "verdict": "genuine" if (not in_ci and p < alpha) else "ambiguous",
    }
