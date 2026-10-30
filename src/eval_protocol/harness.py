# -*- coding: utf-8 -*-
"""
eval_protocol/harness.py
========================
One code path, three fit-scopes.

    honest           every fitted object (feature scaler, encoder, regressor)
                     sees ONLY the training rows of the current fold.
    transductive_x   scaler + encoder are fitted on ALL rows' features X but
                     no test label is used (encoder label loss is masked to the
                     training rows).      -> structure / unsupervised leakage
    pretrain_frozen  scaler + encoder are fitted on ALL rows' X AND y, then
                     frozen (the pre-split PCL-AE protocol).
                                           -> structure + label leakage

The regressor is always trained on training rows only.

    Delta_structure = R2(transductive_x) - R2(honest)
    Delta_label     = R2(pretrain_frozen) - R2(transductive_x)
    Delta_total     = R2(pretrain_frozen) - R2(honest) = sum of the two

For encoders that do not use y (identity / PCA / Standard-AE) transductive_x and
pretrain_frozen coincide by construction, so Delta_label == 0 exactly.

Feature layout
--------------
``X_enc``  block passed through the encoder (e.g. DRFP bits).
``X_aux``  block appended after encoding (xTB, conditions, one-hot ...), may be
           ``None``.
The concatenated ``[Z | aux]`` matrix is standardised with a scaler fitted on the
scope-dependent row set, so the Conv1d branch of ``DualBranchRegressor`` gets
the first ``Z.shape[1]`` columns.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .stats import spearman

FIT_SCOPES = ("honest", "transductive_x", "pretrain_frozen")


@dataclass
class FoldFit:
    """Fitted artefacts of one fold (kept so leak tests can inspect them)."""
    fit_scope: str
    pred: np.ndarray
    scaler: StandardScaler
    encoder: object
    regressor: object
    train_idx: np.ndarray
    test_idx: np.ndarray


def _fit_regressor(reg, X, y, groups):
    sig = inspect.signature(reg.fit)
    if groups is not None and "groups" in sig.parameters:
        return reg.fit(X, y, groups=groups)
    return reg.fit(X, y)


def fit_fold(
    X_enc: np.ndarray,
    X_aux: Optional[np.ndarray],
    y: np.ndarray,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    encoder,
    regressor,
    fit_scope: str = "honest",
    inner_groups: Optional[np.ndarray] = None,
    y_enc: Optional[np.ndarray] = None,
) -> FoldFit:
    """Fit one fold under ``fit_scope`` and predict the test rows.

    ``encoder`` and ``regressor`` are cloned; the originals are untouched.

    ``y_enc`` (default: ``y``) is the label vector the *encoder* sees.  The
    regressor always trains on the true ``y`` of the training rows.  Passing a
    permuted ``y_enc`` is the label-permutation control (see ``controls.py``).
    """
    if fit_scope not in FIT_SCOPES:
        raise ValueError(f"fit_scope must be one of {FIT_SCOPES}, got {fit_scope!r}")
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if np.intersect1d(tr, te).size:
        raise ValueError("train/test overlap")
    X_enc = np.asarray(X_enc)
    n = len(X_enc)
    y = np.asarray(y, dtype=np.float32)
    ye = y if y_enc is None else np.asarray(y_enc, dtype=np.float32)
    if ye.shape != y.shape:
        raise ValueError("y_enc must have the same shape as y")
    aux = None if X_aux is None else np.asarray(X_aux, dtype=np.float32)

    enc = clone(encoder)
    reg = clone(regressor)

    # -- 1. encoder ---------------------------------------------------------
    if fit_scope == "honest":
        enc.fit(X_enc[tr], ye[tr])
    else:
        mask = np.zeros(n, dtype=bool)
        if fit_scope == "transductive_x":
            mask[tr] = True              # test labels hidden from the label loss
        else:                            # pretrain_frozen
            mask[:] = True               # every label visible
        enc.fit(X_enc, np.where(mask, ye, 0.0).astype(np.float32), y_mask=mask)

    Z = np.asarray(enc.transform(X_enc), dtype=np.float32)
    F = Z if aux is None else np.concatenate([Z, aux], axis=1)

    # -- 2. downstream scaler ----------------------------------------------
    scaler = StandardScaler()
    scaler.fit(F[tr] if fit_scope == "honest" else F)
    Fs = scaler.transform(F).astype(np.float32)

    # -- 3. regressor (train rows only) ------------------------------------
    if "n_first" in reg.get_params() and reg.get_params()["n_first"] is None:
        reg.set_params(n_first=int(Z.shape[1]))
    g_tr = None if inner_groups is None else np.asarray(inner_groups)[tr]
    _fit_regressor(reg, Fs[tr], y[tr], g_tr)
    pred = np.asarray(reg.predict(Fs[te]), dtype=np.float64)
    return FoldFit(fit_scope, pred, scaler, enc, reg, tr, te)


# ---------------------------------------------------------------------------
def evaluate(
    X_enc: np.ndarray,
    X_aux: Optional[np.ndarray],
    y: np.ndarray,
    splits: Sequence,                       # SplitIndices-like: .name .fold .train_idx .test_idx
    encoder_factory: Callable[[int], object],
    regressor_factory: Callable[[int], object],
    fit_scope: str = "honest",
    seeds: Sequence[int] = (0,),
    inner_groups: Optional[np.ndarray] = None,
    clip: Optional[tuple] = (0.0, 1.0),
    y_enc: Optional[np.ndarray] = None,
) -> Dict:
    """Run every split for every seed. Returns per-fold table + OOF predictions.

    ``encoder_factory(seed)`` / ``regressor_factory(seed)`` build fresh
    estimators whose RNG depends on ``seed`` - repeats are genuinely independent.
    ``oof`` has shape ``(n_seeds, n)`` with NaN where a row is in no test fold.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    oof = np.full((len(seeds), n), np.nan)
    recs = []
    for si, seed in enumerate(seeds):
        for sp in splits:
            ff = fit_fold(X_enc, X_aux, y, sp.train_idx, sp.test_idx,
                          encoder_factory(seed), regressor_factory(seed),
                          fit_scope, inner_groups, y_enc=y_enc)
            p = ff.pred if clip is None else np.clip(ff.pred, *clip)
            oof[si, ff.test_idx] = p
            yt = y[ff.test_idx]
            recs.append({
                "seed": seed, "split": sp.name, "fold": sp.fold,
                "n_test": len(yt), "fit_scope": fit_scope,
                "r2": float(r2_score(yt, p)) if len(yt) > 1 else np.nan,
                "mae": float(mean_absolute_error(yt, p)),
            })
    folds = pd.DataFrame(recs)
    covered = ~np.isnan(oof[0])
    per_seed = [
        {"seed": s,
         "r2_oof": float(r2_score(y[covered], oof[i, covered])),
         "mae_oof": float(mean_absolute_error(y[covered], oof[i, covered])),
         "spearman_oof": spearman(y[covered], oof[i, covered])}
        for i, s in enumerate(seeds)
    ]
    return {"folds": folds, "oof": oof, "covered": covered,
            "per_seed": pd.DataFrame(per_seed), "fit_scope": fit_scope}


def evaluate_all_scopes(*args, scopes: Sequence[str] = FIT_SCOPES, **kwargs) -> Dict[str, Dict]:
    """Convenience: run the same evaluation under every fit-scope."""
    return {s: evaluate(*args, fit_scope=s, **kwargs) for s in scopes}
