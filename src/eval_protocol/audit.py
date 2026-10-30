# -*- coding: utf-8 -*-
"""
eval_protocol/audit.py
======================
Near-duplicate / redundancy audit of a set of splits.

A random K-fold on reaction-level data puts many test rows next to a training
row that shares the same substrate *and* the same catalyst system (the same
reaction reported under slightly different conditions).  A model that merely
looks the key up then scores well.  This module quantifies that:

* ``dup_share``      - share of test rows whose key also occurs in the training rows
* ``lookup_r2``      - R2 of the trivial predictor "mean training yield of the same key"
                       (rows without a match fall back to the training mean)

Keys (increasing strictness):  substrate | catalyst_set | substrate+catalyst_set |
substrate+catalyst_set+solvent_set.  Catalyst / solvent *sets* are order-free.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

SUBSTRATE_COL = "reactant_name"
CATALYST_COLS = [f"catalyst_{i}_name" for i in range(1, 5)]
SOLVENT_COLS = [f"solvent_{i}" for i in range(1, 5)]


def _set_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    def clean(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none") else s

    arr = df[cols].to_numpy(dtype=object)
    keys = ["+".join(sorted(x for x in (clean(v) for v in row) if x)) for row in arr]
    return pd.Series(keys, index=df.index)


def build_keys(df: pd.DataFrame, substrate_col: str = SUBSTRATE_COL) -> Dict[str, np.ndarray]:
    sub = df[substrate_col].astype(str)
    cat = _set_key(df, CATALYST_COLS)
    sol = _set_key(df, SOLVENT_COLS)
    return {
        "substrate": sub.to_numpy(),
        "catalyst_set": cat.to_numpy(),
        "substrate+catalyst": (sub + "||" + cat).to_numpy(),
        "substrate+catalyst+solvent": (sub + "||" + cat + "||" + sol).to_numpy(),
    }


def near_duplicate_audit(
    df: pd.DataFrame,
    y: Sequence[float],
    splits: Sequence,                       # .name .fold .train_idx .test_idx
    substrate_col: str = SUBSTRATE_COL,
) -> Dict[str, pd.DataFrame]:
    """Per-split and pooled duplicate shares / lookup-baseline R2."""
    y = np.asarray(y, dtype=np.float64)
    keys = build_keys(df, substrate_col)
    rows: List[Dict] = []
    for sp in splits:
        tr = np.asarray(sp.train_idx); te = np.asarray(sp.test_idx)
        for kname, k in keys.items():
            trk = pd.Series(y[tr]).groupby(k[tr]).mean()
            hit = np.isin(k[te], trk.index.to_numpy())
            pred = np.full(len(te), y[tr].mean())
            if hit.any():
                pred[hit] = trk.reindex(k[te][hit]).to_numpy()
            r2 = float(r2_score(y[te], pred)) if len(te) > 1 and np.ptp(y[te]) > 0 else float("nan")
            rows.append({"split": sp.name, "fold": sp.fold, "key": kname, "n_test": len(te),
                         "n_dup": int(hit.sum()), "dup_share": float(hit.mean()),
                         "lookup_r2": r2})
    per = pd.DataFrame(rows)
    pooled = (per.groupby("key")
                 .apply(lambda g: pd.Series({
                     "n_test": int(g["n_test"].sum()),
                     "dup_share": float(g["n_dup"].sum() / g["n_test"].sum()),
                     "lookup_r2_mean": float(np.nanmean(g["lookup_r2"])),
                 }), include_groups=False)
                 .reset_index())
    return {"per_split": per, "pooled": pooled}
