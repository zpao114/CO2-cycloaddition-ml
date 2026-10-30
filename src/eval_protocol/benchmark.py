# -*- coding: utf-8 -*-
"""
eval_protocol/benchmark.py
==========================
Fold-aware drop-in evaluators for the benchmark scripts (201 / 301 / 801).

The legacy scripts first turned the *whole* DRFP table into PCA / AE / PCL-AE
features and only then cut the folds ("pre-split").  Here the DRFP reduction is
a first-class, fold-aware object, :class:`FoldReducer`:

    scope="honest"          encoder is fitted on the training rows of each fold
    scope="transductive_x"  encoder sees every row's X; labels of the test rows are
                            masked out of the property loss
    scope="pretrain_frozen" encoder sees every row's X and y  (legacy PCL-AE protocol)

Only the *encoder* scope changes; the downstream scaler is fitted on the
training rows in every scope, so scope differences isolate the encoder leak.
Because the reduction does not depend on the feature suffix or on the regressor,
``FoldReducer`` caches the per-fold embedding and every (suffix x model)
combination reuses it.

Notes on statistics
-------------------
* The manifest folds are fixed, so "repeats" only re-seed the regressors (and
  optionally the encoder).  ``r2_std`` is the SD over fold x repeat evaluations -
  descriptive, **not** an SE of the mean (training sets overlap across folds).
* Tree models receive ``seed = seed0 + 1000*rep + fold`` - the legacy
  ``random_state=42`` made every "repeat" identical for RF / XGB / LGBM.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .encoders import IdentityEncoder, PCAEncoder, PCLAEEncoder, StandardAEEncoder
from .models import fit_predict_dual, make_tree_model

Fold = Tuple[int, Sequence[int], Sequence[int]]
SCOPES = ("honest", "transductive_x", "pretrain_frozen")


def make_encoder(method: str, seed: int = 0, lam: Optional[float] = None,
                 device: str = "cpu", ae_epochs: int = 100,
                 pcl_epochs: Optional[int] = None,
                 standardize_input: bool = True):
    """``raw`` | ``pca<k>`` | ``ae<k>`` | ``pcl<k>`` -> unfitted encoder.

    ``standardize_input=True`` reproduces the legacy pipeline in which the AE /
    PCL-AE consumed a z-scored DRFP; here that scaler is fitted *inside* the
    encoder so it obeys the same fit-scope (legacy: fitted on the whole table).
    """
    if method == "raw":
        return IdentityEncoder()
    m = re.fullmatch(r"(pca|ae|pcl)(\d+)", method)
    if not m:
        raise ValueError(f"unknown reduction method {method!r}")
    kind, k = m.group(1), int(m.group(2))
    if kind == "pca":
        return PCAEncoder(n_components=k, random_state=seed)
    if kind == "ae":
        return StandardAEEncoder(latent_dim=k, epochs=ae_epochs, random_state=seed,
                                 device=device, standardize_input=standardize_input)
    if lam is None:
        raise ValueError("PCL-AE requires an explicit lambda (lam=...)")
    if pcl_epochs is None:
        try:
            from config import PCL_AE_EPOCHS as pcl_epochs  # type: ignore
        except ImportError:  # pragma: no cover
            from src.config import PCL_AE_EPOCHS as pcl_epochs  # type: ignore
    return PCLAEEncoder(latent_dim=k, lambda_prop=float(lam), epochs=pcl_epochs,
                        random_state=seed, device=device,
                        standardize_input=standardize_input)


class FoldReducer:
    """Per-fold DRFP embedding under a chosen encoder fit-scope (cached)."""

    def __init__(self, X_drfp: np.ndarray, y: np.ndarray, folds: Sequence[Fold],
                 method: str = "raw", scope: str = "honest", enc_seed: int = 0,
                 lam: Optional[float] = None, device: str = "cpu"):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        self.X = np.asarray(X_drfp, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.folds = list(folds)
        self.method, self.scope = method, scope
        self.enc_seed, self.lam, self.device = enc_seed, lam, device
        self._cache: Dict[int, np.ndarray] = {}
        self._global: Optional[np.ndarray] = None
        self.n_features_out = None
        # probe whether the encoder uses y (only then do the scopes differ)
        self._uses_y = bool(getattr(make_encoder(method, enc_seed, lam or 0.0), "requires_y", False))

    def _enc(self):
        return make_encoder(self.method, self.enc_seed, self.lam, self.device)

    def features(self, k: int) -> np.ndarray:
        """Embedding of ALL rows for fold index ``k`` (use only [tr] / [te])."""
        n = len(self.X)
        _, tr, te = self.folds[k]
        tr = np.asarray(tr)
        if self.method == "raw":
            return self.X
        scope_is_global = (self.scope == "pretrain_frozen") or \
                          (self.scope == "transductive_x" and not self._uses_y)
        if scope_is_global:
            if self._global is None:
                enc = self._enc()
                enc.fit(self.X, self.y if self._uses_y else None)
                self._global = np.asarray(enc.transform(self.X), dtype=np.float32)
            return self._global
        if k in self._cache:
            return self._cache[k]
        enc = self._enc()
        if self.scope == "honest":
            enc.fit(self.X[tr], self.y[tr] if self._uses_y else None)
        else:  # transductive_x with a supervised encoder: test labels masked
            mask = np.zeros(n, dtype=bool); mask[tr] = True
            enc.fit(self.X, np.where(mask, self.y, 0.0).astype(np.float32), y_mask=mask)
        Z = np.asarray(enc.transform(self.X), dtype=np.float32)
        self._cache[k] = Z
        return Z

    def dim(self) -> int:
        return int(self.features(0).shape[1])


# ---------------------------------------------------------------------------
def _metrics(yt, p) -> Dict[str, float]:
    try:
        pr = float(pearsonr(yt, p)[0])
    except Exception:
        pr = np.nan
    return {"r2": float(r2_score(yt, p)), "mae": float(mean_absolute_error(yt, p)),
            "rmse": float(np.sqrt(mean_squared_error(yt, p))), "pearson": pr}


def _summ(recs: List[Dict]) -> Dict[str, float]:
    df = pd.DataFrame(recs)
    return {"r2_mean": float(df["r2"].mean()), "r2_std": float(df["r2"].std()),
            "mae_mean": float(df["mae"].mean()), "rmse_mean": float(df["rmse"].mean()),
            "pearson_mean": float(df["pearson"].mean()), "n_evals": int(len(df))}


def _features(red: FoldReducer, k: int, X_aux: Optional[np.ndarray]) -> np.ndarray:
    Z = red.features(k)
    return Z if X_aux is None else np.hstack([Z, X_aux]).astype(np.float32)


def cv_tree(model_name: str, red: FoldReducer, X_aux: Optional[np.ndarray],
            y: np.ndarray, n_repeats: int = 3, seed0: int = 0) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    recs = []
    for rep in range(n_repeats):
        for k, (fold_id, tr, te) in enumerate(red.folds):
            F = _features(red, k, X_aux)
            sc = StandardScaler().fit(F[tr])
            m = make_tree_model(model_name, seed=seed0 + 1000 * rep + int(fold_id))
            m.fit(sc.transform(F[tr]), y[tr])
            p = np.clip(m.predict(sc.transform(F[te])), 0, 1)
            recs.append({"rep": rep, "fold": fold_id, **_metrics(y[te], p)})
    return _summ(recs)


def _dual_predict(red, k, X_aux, y, tr, te, seed, **hp) -> np.ndarray:
    Z = red.features(k)
    sd, so = StandardScaler().fit(Z[tr]), StandardScaler().fit(X_aux[tr])
    return fit_predict_dual(sd.transform(Z[tr]), so.transform(X_aux[tr]), y[tr],
                            sd.transform(Z[te]), so.transform(X_aux[te]), seed=seed, **hp)


def cv_dual(red: FoldReducer, X_aux: np.ndarray, y: np.ndarray, n_repeats: int = 2,
            seed0: int = 0, **hp) -> Dict[str, float]:
    """DualBranch ANN; early stopping on an inner split of the training fold."""
    y = np.asarray(y, dtype=np.float64)
    recs = []
    for rep in range(n_repeats):
        for k, (fold_id, tr, te) in enumerate(red.folds):
            p = np.clip(_dual_predict(red, k, X_aux, y, tr, te,
                                      seed0 + 1000 * rep + int(fold_id), **hp), 0, 1)
            recs.append({"rep": rep, "fold": fold_id, **_metrics(y[te], p)})
    return _summ(recs)


def cv_ensemble(red: FoldReducer, X_aux: np.ndarray, y: np.ndarray, n_repeats: int = 1,
                seed0: int = 0, **dual_hp) -> Dict[str, float]:
    """XGB + LGBM + DualANN equal-weight ensemble; pooled OOF R2 per repeat."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    recs = []
    for rep in range(n_repeats):
        oof = np.zeros(n); cnt = np.zeros(n)
        for k, (fold_id, tr, te) in enumerate(red.folds):
            seed = seed0 + 1000 * rep + int(fold_id)
            F = _features(red, k, X_aux)
            sc = StandardScaler().fit(F[tr])
            Ftr, Fte = sc.transform(F[tr]), sc.transform(F[te])
            ps = []
            for name in ("XGB", "LGBM"):
                m = make_tree_model(name, seed=seed); m.fit(Ftr, y[tr])
                ps.append(np.clip(m.predict(Fte), 0, 1))
            ps.append(np.clip(_dual_predict(red, k, X_aux, y, tr, te, seed, **dual_hp), 0, 1))
            oof[te] += np.mean(ps, axis=0); cnt[te] += 1
        mk = cnt > 0
        recs.append({"rep": rep, **_metrics(y[mk], oof[mk] / cnt[mk])})
    return _summ(recs)


DUAL_HP_DEFAULT = dict(hidden=128, lr=5e-4, weight_decay=1e-3, batch_size=32,
                       max_epochs=250, patience=40, lr_patience=20)


def part_a_rows(method: str, label: str, X_drfp, X_xtb, X_xtb_cond, X_xtb_cond_inter,
                y, folds: Sequence[Fold], scope: str, lam: float, device: str = "cpu",
                trees: Sequence[str] = ("RF", "XGB", "LGBM"), log=print,
                dual_hp: Optional[Dict] = None):
    """Benchmark "Part A" for one DRFP reduction method with fold-internal reduction.

    Returns ``[(result_row, tag), ...]``.  The per-fold embedding is cached in one
    :class:`FoldReducer` and reused by every (feature-suffix x model) combination.
    """
    import time
    red = FoldReducer(X_drfp, y, folds, method=method, scope=scope, lam=lam, device=device)
    d = red.dim()
    aux_sets = {
        f"drfp_{label}": None,
        f"drfp_{label}_xtb": X_xtb,
        f"drfp_{label}_xtbc": X_xtb_cond,
        f"drfp_{label}_full": X_xtb_cond_inter,
    }
    out = []
    for fsname, aux in aux_sets.items():
        dim_total = d + (0 if aux is None else aux.shape[1])
        log(f"\n[{fsname}]  dim={dim_total}  protocol={scope}")
        runs = [(m, (lambda m=m, aux=aux: cv_tree(m, red, aux, y, n_repeats=3, seed0=0)))
                for m in trees]
        if (dim_total >= 100 and method in ("raw", "pca128", "pca256", "pcl128", "pcl256")
                and fsname.endswith(("_xtbc", "_full"))):
            runs.append(("DualANN", (lambda aux=aux: cv_dual(
                red, aux, y, n_repeats=2, seed0=0, device=device,
                **(dual_hp or DUAL_HP_DEFAULT)))))
        for mname, fn in runs:
            t1 = time.time()
            m = fn()
            el = time.time() - t1
            log(f"  {mname:7s} R2={m['r2_mean']:.4f}+-{m['r2_std']:.4f}  "
                f"MAE={m['mae_mean']:.4f}  ({el:.0f}s)")
            out.append(({
                "stage": "A_DRFP_reduce", "drfp_method": label, "feature_set": fsname,
                "model": mname,
                **{k: m[k] for k in ["r2_mean", "r2_std", "mae_mean", "rmse_mean", "pearson_mean"]},
                "feature_dim": dim_total, "time_s": el,
            }, f"{label} | {fsname} | {mname}"))
    return out
