# -*- coding: utf-8 -*-
"""
eval_protocol/features.py
=========================
Explicit numeric feature list + fold-internal missing-value handling.

Why
---
* Several scripts selected the numeric block with
  ``df.columns if dtype in (float64, ...)``.  Any new numeric column silently
  became a feature, and all-empty text columns (``reagent_1_name``,
  ``reagent_2_name``) were read as float NaN and became two constant-zero
  features.  ``NUMERIC_FEATURE_COLS`` is now the single explicit list.
* ``fillna(0)`` on xTB HOMO/LUMO/gap (eV) is physically meaningless, and doing
  it on the whole table before splitting hides how many values were missing.
  ``FoldImputer`` imputes with *training-fold* medians instead.  Catalyst
  loading columns are different: NaN means "this catalyst slot is unused", so
  0 is the correct value there and is kept (``ZERO_FILL_COLS``).
* ``add_indicators=True`` appends one missing-indicator per block (row-wise,
  needs no fitting) so the model can see *that* a value was imputed.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Explicit feature list (51 columns of co2_drfp_xtb_extended.csv)
# ---------------------------------------------------------------------------
CONDITION_COLS: List[str] = ["temperature (\u2103)", "pressure (MPa)", "time (h)"]
LOADING_COLS: List[str] = [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]
XTB_SUB_COLS: List[str] = ["sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D"]
XTB_CO2_COLS: List[str] = ["co2_homo_eV", "co2_lumo_eV", "co2_gap_eV"]
XTB_CAT_COLS: List[str] = [
    "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
    "delta_E_hl_cat_sub", "global_hardness", "nucleophilicity_index",
    "cat_homo_eV_min", "cat_lumo_eV_max", "cat_gap_eV_min", "activation_proxy",
    "sub_cat_orbital_match", "gap_ratio", "reaction_polarity", "co2_activation_proxy",
    "cat_cation_homo_eV", "cat_cation_lumo_eV", "cat_cation_gap_eV",
    "cat_cation_dipole_D", "hardness_ratio", "nucleophilicity_cat",
    "electrophilicity_cat", "electrodonating_cat",
]
XTB_ANION_COLS: List[str] = [
    "cat_anion_homo_eV", "cat_anion_lumo_eV", "cat_anion_gap_eV", "cat_anion_dipole_D",
    "ion_pair_interaction", "charge_transfer_potential",
]
XTB_SOLV_COLS: List[str] = [
    "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV", "solv_dipole_D",
    "solv_cat_interaction", "solv_sub_interaction", "dielectric_proxy",
]
XTB_MISC_COLS: List[str] = ["total_polarity_index"]

NUMERIC_FEATURE_COLS: List[str] = (
    CONDITION_COLS + LOADING_COLS + XTB_SUB_COLS + XTB_CO2_COLS + XTB_CAT_COLS
    + XTB_SOLV_COLS + XTB_ANION_COLS + XTB_MISC_COLS
)

# NaN == "slot not used"  ->  0 is the correct value, no imputation.
ZERO_FILL_COLS: List[str] = list(LOADING_COLS)

# Numeric-looking columns that must never be features (target, identifier,
# bibliographic metadata).
NON_FEATURE_NUMERIC: tuple = ("yield (%)", "yield", "row_id", "publication_year")
# Text slot columns that pandas reads as float64 when a whole column is empty
# (e.g. no row uses a 4th catalyst); they are never features.
KNOWN_EMPTY_COLS: tuple = (
    "reagent_1_name", "reagent_2_name",
    "catalyst_4_name", "catalyst_4_smiles",
    "solvent_3", "solvent_4", "solvent_3_smiles", "solvent_4_smiles",
)

MISSING_BLOCKS: Dict[str, List[str]] = {
    "cond": CONDITION_COLS,
    "cat_xtb": XTB_CAT_COLS,
    "anion_xtb": XTB_ANION_COLS,
    "solv_xtb": XTB_SOLV_COLS,
}


def select_numeric_features(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
                            warn_extra: bool = True) -> pd.DataFrame:
    """Return ``df[cols]`` as float64 (NaN preserved).

    Raises ``KeyError`` if an expected column is absent (schema drift is an
    error, not something to paper over) and warns about numeric columns that
    exist in ``df`` but are *not* in the explicit list, so that a new column is
    a conscious decision.
    """
    cols = list(NUMERIC_FEATURE_COLS if cols is None else cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing from frame: {missing}")
    if warn_extra:
        skip = set(cols) | set(NON_FEATURE_NUMERIC) | set(KNOWN_EMPTY_COLS)
        extra = [c for c in df.columns
                 if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        if extra:
            warnings.warn(f"numeric columns NOT used as features: {extra}", stacklevel=2)
    return df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)


def missing_report(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fraction missing per feature column (documentation / SI table)."""
    X = select_numeric_features(df, cols, warn_extra=False)
    rep = X.isna().mean().rename("frac_missing").to_frame()
    rep["policy"] = ["zero (unused slot)" if c in ZERO_FILL_COLS else "train-fold median"
                     for c in rep.index]
    return rep


# ---------------------------------------------------------------------------
# Fold-internal imputer
# ---------------------------------------------------------------------------
class FoldImputer(BaseEstimator, TransformerMixin):
    """Median-impute with statistics learnt on the *training fold only*.

    Parameters
    ----------
    columns : names of the columns of ``X`` (same order).
    zero_fill : names imputed with 0 instead of the median.
    add_indicators : append one 0/1 column per entry of ``blocks`` that is 1 when
        any column of that block was missing in the row.
    """

    def __init__(self, columns: Sequence[str], zero_fill: Iterable[str] = tuple(ZERO_FILL_COLS),
                 add_indicators: bool = False,
                 blocks: Optional[Dict[str, List[str]]] = None):
        self.columns = list(columns)
        self.zero_fill = tuple(zero_fill)
        self.add_indicators = add_indicators
        self.blocks = blocks

    def _block_map(self):
        blocks = MISSING_BLOCKS if self.blocks is None else self.blocks
        return {k: [self.columns.index(c) for c in v if c in self.columns]
                for k, v in blocks.items()}

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.columns):
            raise ValueError(f"X has {X.shape[1]} columns, expected {len(self.columns)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice -> NaN
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)            # column empty in this fold
        for j, c in enumerate(self.columns):
            if c in self.zero_fill:
                med[j] = 0.0
        self.fill_ = med
        self.n_missing_fit_ = int(np.isnan(X).sum())
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        nan = np.isnan(X)
        out = np.where(nan, self.fill_[None, :], X)
        if self.add_indicators:
            ind = [nan[:, idx].any(axis=1).astype(np.float64)[:, None]
                   for idx in self._block_map().values() if idx]
            if ind:
                out = np.hstack([out] + ind)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.columns)
        if self.add_indicators:
            names += [f"missing_{k}" for k, idx in self._block_map().items() if idx]
        return np.asarray(names)


class ImputeScaleModel:
    """Fold-safe wrapper: [impute first ``n_num`` cols] -> StandardScaler -> model.

    Drop-in for the ``ScaledModel`` copies (``fit(X, y)`` / ``predict(X)``).
    Columns after ``n_num`` (e.g. an already-encoded DRFP latent) are not
    imputed.  Everything is fitted on the training rows passed to ``fit``.
    """

    def __init__(self, model, n_num: int = 0, columns: Optional[Sequence[str]] = None,
                 add_indicators: bool = False):
        self.model = model
        self.n_num = int(n_num)
        self.columns = list(NUMERIC_FEATURE_COLS if columns is None else columns)
        self.add_indicators = add_indicators

    def _prep(self, X, fit: bool):
        X = np.asarray(X, dtype=np.float64)
        if self.n_num == 0:
            return X
        head, tail = X[:, :self.n_num], X[:, self.n_num:]
        if fit:
            cols = self.columns if len(self.columns) == self.n_num else \
                [f"f{i}" for i in range(self.n_num)]
            self.imp = FoldImputer(cols, add_indicators=self.add_indicators).fit(head)
        return np.hstack([self.imp.transform(head), tail])

    def fit(self, X_tr, y_tr):
        Xp = self._prep(X_tr, fit=True)
        self.sc = StandardScaler()
        self.model.fit(self.sc.fit_transform(Xp).astype(np.float64), y_tr)
        return self

    def predict(self, X_te):
        Xp = self._prep(X_te, fit=False)
        return self.model.predict(self.sc.transform(Xp).astype(np.float64))


# ---------------------------------------------------------------------------
# Fold-level DRFP standardisation
# ---------------------------------------------------------------------------
def fold_standardize(X_tr, X_te=None):
    """Fit ``StandardScaler`` on the training rows; apply to train (and test).

    Use this instead of ``StandardScaler().fit_transform(X_all)`` before
    slicing folds (which lets test-row statistics shape the training input).
    Returns ``(X_tr_s, X_te_s)`` or just ``X_tr_s`` when ``X_te`` is None.
    """
    sc = StandardScaler().fit(X_tr)
    a = sc.transform(X_tr).astype(np.float32)
    if X_te is None:
        return a
    return a, sc.transform(X_te).astype(np.float32)


def global_standardize(X):
    """Deliberately *transductive* full-table standardisation (the 'cached'/legacy arm).

    Named explicitly so every use of full-table scaling is greppable and
    flagged as non-honest in the output metadata.
    """
    return StandardScaler().fit_transform(X).astype(np.float32)
