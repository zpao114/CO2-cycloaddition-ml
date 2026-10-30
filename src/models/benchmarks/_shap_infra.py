# -*- coding: utf-8 -*-
"""
_shap_infra.py
=============
Shared infrastructure for the CHO contrast experiments (807-810).

Provides
--------
1. load_X_y_groups(): the 85% canonical training pool as a (PCL-AE latent 128-D
   + XTB 43-D + Cond 7-D + Inter 2-D) = 180-D matrix.  Column names come from
   results_best_pipeline/artifacts/feature_meta.json.

2. compute_group_shap(): train on X[~mask], return mean SHAP on X[mask].  XGBoost
   early stopping uses an inner split of the TRAINING rows only.

3. contrast_bootstrap(): retrain-bootstrap of D = mean(terminal SHAP) - CHO SHAP
   for sub_homo_eV and for the whole substrate-level descriptor block.

Caveat the analyses must respect: sub_homo_eV is constant inside each substrate
(five values in total), so it is collinear with the other substrate-level
descriptors and with the substrate identity itself.

Design contract
---------------
* Feature column order is FIXED and identical to the training pipeline:
      latent_0 .. latent_127, then XTB[0..42], then Cond[0..6], then Inter[0..1]
* Interaction columns (T_x_activation_proxy, P_x_total_polarity_index) are
  computed on-the-fly inside load_X_y_groups() because they are NOT written to
  the source CSV — they were constructed at training time by the pipeline.
* Latent vector alignment uses the row_id mapping stored in
  results_pcl_ae/row_id.csv, NOT positional indices, because the original
  master CSV had its rows shuffled before latent encoding.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

PROJECT_ROOT = Path(
    os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_EXTENDED = (
    PROJECT_ROOT / "results" / "results_cho_diagnostic" / "co2_drfp_xtb_extended.csv"
)
PCL_LATENT = PROJECT_ROOT / "results_pcl_ae" / "pcl_ae_latent.npy"
PCL_ROWID = PROJECT_ROOT / "results_pcl_ae" / "row_id.csv"
# Authoritative path is results_best_pipeline/artifacts at PROJECT_ROOT
# (NOT under results/), as defined in src.paths.RESULTS_BEST_PIPELINE.
ARTIFACTS = PROJECT_ROOT / "results_best_pipeline" / "artifacts"

# ── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("shap_infra")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ── Substrate constants ─────────────────────────────────────────────────────
SUBSTRATE_ORDER = [
    "Cyclohexene oxide",         # CHO — intra-ring epoxide (key reversal substrate)
    "Propylene oxide",           # PO  — terminal
    "Styrene oxide",             # SO  — terminal
    "Epichlorohydrin",           # ECH — terminal
    "Isopropyl glycidyl ether",  # IGE — terminal
]
CHO_NAME = "Cyclohexene oxide"
TERMINAL_NAMES = [s for s in SUBSTRATE_ORDER if s != CHO_NAME]

# Temperature column name as produced by 101/102 (U+2103 DEGREE CELSIUS).
# Single definition for the SHAP scripts; the live feature names normally come
# from feature_meta.json cond_cols, which holds the same string.
TEMP_COL = "temperature (\u2103)"

# Feature names of mechanistic interest (for SHAP bar plots)
KEY_FEATURES = [
    "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
    TEMP_COL, "pressure (MPa)", "time (h)",
    "activation_proxy", "charge_transfer_potential", "ion_pair_interaction",
    "electrophilicity_cat", "electrodonating_cat",
    "nucleophilicity_index", "gap_ratio", "reaction_polarity",
    "co2_activation_proxy",
]

# Interaction column name → (cond_short_prefix, xtb_col_name)
INTERACTION_NAME_MAP = {
    "T * activation_proxy":          "T_x_activation_proxy",
    "P * total_polarity_index":      "P_x_total_polarity_index",
}


# ============================================================================
# 1.  Meta loading
# ============================================================================
class FeatureMeta:
    """
    Authoritative feature schema loaded from feature_meta.json.

    Attributes
    ----------
    xtb_cols     : list[str]   length 43
    cond_cols    : list[str]   length 7
    interactions : list[dict]  each {name, cond_idx, xtb_name}  length 2
    feature_names_52 : list[str]  (xtb + cond + inter), length 52, in pipeline order
    """

    def __init__(self, meta: dict):
        self.xtb_cols: list[str] = list(meta.get("xtb_cols", []) or [])
        self.cond_cols: list[str] = list(meta.get("cond_cols", []) or [])
        self.interactions: list[dict] = list(meta.get("interaction_rules", []) or [])

        inter_names: list[str] = []
        for rule in self.interactions:
            rule_name = rule["name"]
            # map rule.name → canonical column name (matches training pipeline)
            if rule_name in INTERACTION_NAME_MAP:
                inter_names.append(INTERACTION_NAME_MAP[rule_name])
            else:
                # fallback: derive from cond_idx + xtb_name
                ci = rule["cond_idx"]
                cond_full = self.cond_cols[ci] if ci < len(self.cond_cols) else f"c{ci}"
                head = cond_full.split(" ")[0].lower()
                short_map = {"temperature": "T", "pressure": "P"}
                head = short_map.get(head, head)
                inter_names.append(f"{head}_x_{rule['xtb_name']}")

        self.feature_names_52: list[str] = self.xtb_cols + self.cond_cols + inter_names

    def validate_against_csv(self, csv_path: Path) -> "FeatureMeta":
        """Return a NEW FeatureMeta filtered to columns actually present in the CSV.
        Missing columns are dropped (with a logger.info note)."""
        header = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0).columns.tolist()
        present_xtb = [c for c in self.xtb_cols if c in header]
        present_cond = [c for c in self.cond_cols if c in header]
        missing_xtb = [c for c in self.xtb_cols if c not in header]
        missing_cond = [c for c in self.cond_cols if c not in header]
        if missing_xtb:
            logger.info("XTB cols missing from CSV (dropped): %s", missing_xtb)
        if missing_cond:
            logger.info("Cond cols missing from CSV (dropped): %s", missing_cond)

        new = FeatureMeta.__new__(FeatureMeta)
        new.xtb_cols = present_xtb
        new.cond_cols = present_cond
        # interactions always computed on the fly (they are derived columns)
        new.interactions = [
            r for r in self.interactions if r["xtb_name"] in present_xtb
            and r["cond_idx"] < len(present_cond)
        ]
        new.feature_names_52 = (
            present_xtb + present_cond + [_rule_to_colname(r, present_cond)
                                          for r in new.interactions]
        )
        return new


def _rule_to_colname(rule: dict, cond_cols: list[str]) -> str:
    rule_name = rule["name"]
    if rule_name in INTERACTION_NAME_MAP:
        return INTERACTION_NAME_MAP[rule_name]
    ci = rule["cond_idx"]
    cond_full = cond_cols[ci] if ci < len(cond_cols) else f"c{ci}"
    head = cond_full.split(" ")[0].lower()
    short_map = {"temperature": "T", "pressure": "P"}
    head = short_map.get(head, head)
    return f"{head}_x_{rule['xtb_name']}"


def load_feature_meta(path: Optional[Path] = None) -> FeatureMeta:
    """Load and validate feature_meta.json. Returns a FeatureMeta instance."""
    p = path or (ARTIFACTS / "feature_meta.json")
    if not p.exists():
        logger.warning("feature_meta.json not found at %s; using empty schema", p)
        return FeatureMeta({})
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    meta = FeatureMeta(raw)
    return meta.validate_against_csv(DATA_EXTENDED)


# ============================================================================
# 2.  Data loading
# ============================================================================
def _clean_master(csv_path: Path) -> pd.DataFrame:
    """Read + canonical-clean the master CSV (the same protocol used everywhere)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    return df


def _align_latent(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return the latent matrix sliced to match ``df`` row order, using
    results_pcl_ae/row_id.csv as the authoritative mapping.

    Returns None if latent files are missing.

    Mapping convention (verified empirically on 2026-08-20):
        latent[k]  ↔  master-CSV row whose row_id == row_id.csv[k]

    master-CSV row_id is the integer index assigned by the original
    stratified split.  ``df`` after reset_index() has ``df.index`` = 0..N-1,
    so we re-attach the original row_id via the CSV's ``row_id`` column.
    """
    if not (PCL_LATENT.exists() and PCL_ROWID.exists()):
        return None
    if "row_id" not in df.columns:
        logger.error(
            "Master CSV is missing the 'row_id' column; cannot align latent."
        )
        return None

    latent = np.load(PCL_LATENT)        # (n_full, 128)
    rid = pd.read_csv(PCL_ROWID)["row_id"].values  # (n_full,)

    # Build a position lookup: master row_id → latent row index
    pos_of = {int(r): k for k, r in enumerate(rid)}

    # For every df row, find its latent row.
    rows = df["row_id"].astype(int).values
    latent_idx = np.array([pos_of[r] for r in rows if r in pos_of], dtype=int)
    if len(latent_idx) != len(df):
        logger.warning(
            "Latent alignment lost %d rows (out of %d)",
            len(df) - len(latent_idx), len(df),
        )
    return latent[latent_idx]


def _build_interaction_block(
    df: pd.DataFrame,
    interactions: list[dict],
    cond_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute the interaction columns on-the-fly (NOT in CSV).

    Returns (matrix, names) where matrix is (n, len(interactions)) float32.
    """
    if not interactions:
        return np.zeros((len(df), 0), dtype=np.float32), []
    cols = []
    names = []
    for rule in interactions:
        ci = rule["cond_idx"]
        xtb_name = rule["xtb_name"]
        if ci >= len(cond_cols) or xtb_name not in df.columns:
            logger.warning(
                "Interaction rule %s missing inputs (cond_idx=%d, xtb=%s); skipped",
                rule, ci, xtb_name,
            )
            cols.append(np.zeros(len(df), dtype=np.float32))
            names.append(f"missing_inter_{ci}_{xtb_name}")
            continue
        cond_vals = df[cond_cols[ci]].astype(np.float32).values
        xtb_vals = df[xtb_name].astype(np.float32).values
        prod = cond_vals * xtb_vals
        # replace NaN/Inf with 0
        prod = np.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(prod)
        names.append(_rule_to_colname(rule, cond_cols))
    block = np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(df), 0), dtype=np.float32)
    return block, names


def load_X_y_groups(
    filter_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    include_latent: bool = True,
    train_pool_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Load the training pool and build the (latent + XTB + Cond + Inter)
    feature matrix.

    Parameters
    ----------
    filter_fn   : optional callable(df -> bool array) for catalyst / condition
                  control experiments.  Applied BEFORE feature construction.
    include_latent : if False, omit the 128-D PCL-AE latent block (useful when
                  the filter has detached rows from the latent alignment).
    train_pool_only : if True (default), restrict to the 85% canonical training
                  pool defined by src.data_split.holdout_arrays.  If False,
                  use all valid rows of the master table.

    Returns
    -------
    X           : (n, 180) float32  when include_latent=True
                  (n, 52)  float32  when include_latent=False
    y           : (n,)    yield normalised to [0, 1]
    groups      : list[str] of reactant_name labels
    feat_names  : list[str] of feature column names
    df          : the underlying cleaned DataFrame (for debugging / extra cols)
    """
    logger.info("Loading data from %s", DATA_EXTENDED)
    df = _clean_master(DATA_EXTENDED)

    # ── Restrict to canonical train pool (85%) if requested ───────────────
    if train_pool_only:
        from src.data_split import holdout_arrays, load_manifest, verify_manifest  # local import
        _manifest = load_manifest()
        verify_manifest(df, _manifest)   # row count / row_id order / data hash
        train_idx, _, _ = holdout_arrays(_manifest)
        # NOTE: the manifest's indices refer to position-within-cleaned-CSV
        # (i.e. df.index after reset_index in _clean_master), NOT the
        # master-CSV 'row_id' column (which has gaps from earlier dropna).
        train_pos_set = set(int(i) for i in train_idx)
        # df has just been (possibly) reset by _clean_master; its index is
        # already 0..len(df)-1, matching the manifest's positional space.
        before = len(df)
        df = df[df.index.isin(train_pos_set)].reset_index(drop=True)
        logger.info(
            "  train_pool_only: kept %d / %d rows (canonical 85%% train pool)",
            len(df), before,
        )

    # ── Normalise y ─────────────────────────────────────────────────────────
    y = np.clip(df["yield (%)"].values.astype(np.float32) / 100.0, 0.0, 1.0)

    # ── Row filter ─────────────────────────────────────────────────────────
    if filter_fn is not None:
        mask = np.asarray(filter_fn(df), dtype=bool)
        df = df.loc[mask].reset_index(drop=True)
        y = y[mask]
        logger.info("  filter: kept %d / %d rows", int(mask.sum()), len(df))

    groups = df["reactant_name"].values.tolist()

    # ── Authoritative 52-D feature schema ──────────────────────────────────
    meta = load_feature_meta()
    xtb_block = df[meta.xtb_cols].values.astype(np.float32) if meta.xtb_cols else np.zeros((len(df), 0), dtype=np.float32)
    cond_block = df[meta.cond_cols].values.astype(np.float32) if meta.cond_cols else np.zeros((len(df), 0), dtype=np.float32)
    inter_block, inter_names = _build_interaction_block(df, meta.interactions, meta.cond_cols)
    X_52 = np.hstack([xtb_block, cond_block, inter_block]).astype(np.float32)
    feat_names_52 = meta.xtb_cols + meta.cond_cols + inter_names

    # ── Latent (128-D) ──────────────────────────────────────────────────────
    if include_latent:
        latent_slice = _align_latent(df)
        if latent_slice is None:
            logger.warning("PCL-AE latent not found; using XTB/Cond/Inter only (52-D)")
            X = X_52
            feat_names = feat_names_52
        else:
            X = np.hstack([latent_slice, X_52]).astype(np.float32)
            feat_names = [f"latent_{i}" for i in range(latent_slice.shape[1])] + feat_names_52
            logger.info(
                "  feature matrix: latent %d + XTB %d + Cond %d + Inter %d = %d",
                latent_slice.shape[1], len(meta.xtb_cols), len(meta.cond_cols),
                len(inter_names), X.shape[1],
            )
    else:
        X = X_52
        feat_names = feat_names_52

    # Replace NaN / Inf globally
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "  X shape: %s, y: %d rows, groups: %d unique",
        X.shape, len(y), len(set(groups)),
    )
    return X, y, np.asarray(groups), feat_names, df



# ============================================================================
# 3.  Group SHAP
# ============================================================================
def inner_validation_split(n: int, val_frac: float = 0.15, seed: int = 42):
    """Disjoint (fit_idx, val_idx) over ``range(n)`` used for early stopping."""
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(int(round(val_frac * n)), 8)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def compute_group_shap(
    X: np.ndarray,
    y: np.ndarray,
    group_mask: np.ndarray,
    feat_names: list[str],
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
    model_type: str = "xgb",
    rf_max_depth: int = 8,
    rf_min_samples_leaf: int = 3,
) -> dict:
    """
    Train on ``(X[~mask], y[~mask])`` and return the mean SHAP per feature on
    the held-out rows ``X[mask]``.

    ``model_type`` is ``"xgb"`` (native ``pred_contribs``) or ``"rf"``
    (``shap.TreeExplainer``; KernelExplainer is avoided because it fails when
    n_samples < n_features).  XGBoost early stopping uses an inner validation
    split of the TRAINING rows only, so the held-out group never influences
    model selection.

    Returns a dict with: shap_mean, shap_std, r2, n_test.
    """
    group_mask = np.asarray(group_mask)
    if group_mask.dtype != bool:
        raise TypeError("group_mask must be a boolean array, got %s" % group_mask.dtype)

    X_tr, y_tr = X[~group_mask], y[~group_mask]
    X_te, y_te = X[group_mask], y[group_mask]
    if len(X_tr) < 20:
        logger.warning("Train set small (%d rows); SHAP may be unreliable", len(X_tr))
    if len(X_te) < 5:
        logger.warning("Test set small (%d rows)", len(X_te))

    from sklearn.metrics import r2_score

    if model_type == "rf":
        import shap as _shap
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        ).fit(X_tr, y_tr)
        r2 = float(r2_score(y_te, model.predict(X_te)))
        shap_vals = np.asarray(_shap.TreeExplainer(model).shap_values(X_te), dtype=np.float32)
        if shap_vals.ndim == 3:        # very old shap versions return (n, n_feat, 1)
            shap_vals = shap_vals[..., 0]
    else:
        import xgboost as xgb

        fit_idx, val_idx = inner_validation_split(len(X_tr), seed=random_state)
        dfit = xgb.DMatrix(X_tr[fit_idx], label=y_tr[fit_idx], feature_names=feat_names)
        dval = xgb.DMatrix(X_tr[val_idx], label=y_tr[val_idx], feature_names=feat_names)
        dtest = xgb.DMatrix(X_te, label=y_te, feature_names=feat_names)
        params = dict(
            objective="reg:squarederror",
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
        model = xgb.train(
            params, dfit,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        best = (0, int(model.best_iteration) + 1)
        r2 = float(r2_score(y_te, model.predict(dtest, iteration_range=best)))
        contribs = model.predict(dtest, pred_contribs=True, validate_features=False,
                                 iteration_range=best)
        shap_vals = contribs[:, :-1].astype(np.float32)      # drop bias column

    return dict(
        shap_mean=shap_vals.mean(axis=0),
        shap_std=shap_vals.std(axis=0),
        r2=r2,
        n_test=int(len(X_te)),
    )


# ============================================================================
# 4.  CHO contrast with retrain-bootstrap
# ============================================================================
def substrate_level_cols(X: np.ndarray, groups: np.ndarray) -> list[int]:
    """Columns that are constant inside every substrate but differ across
    substrates (sub_homo_eV, sub_lumo_eV, ...).  With only five substrates
    these columns are mutually collinear, so SHAP is split between them
    arbitrarily; their sum is the identifiable quantity."""
    groups = np.asarray(groups)
    masks = [groups == s for s in sorted(set(groups))]
    cols = []
    for j in range(X.shape[1]):
        firsts = []
        for m in masks:
            c = X[m, j]
            if c.min() != c.max():
                break
            firsts.append(c[0])
        else:
            if len(set(firsts)) > 1:
                cols.append(j)
    return cols


def contrast_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_names: list[str],
    model_type: str = "xgb",
    n_boot: int = 200,
    n_estimators: int = 500,
    seed: int = 0,
    focus: str = "sub_homo_eV",
) -> pd.DataFrame:
    """
    Retrain-bootstrap for the CHO contrast.

    Each iteration resamples the TRAINING rows (with replacement) of every
    leave-one-substrate-out split, refits, and records the mean SHAP of
    ``focus`` and of the whole substrate-level descriptor block on the held-out
    substrate.  Per iteration:

        D = mean over terminal substrates of SHAP  -  SHAP of CHO

    so the interval reflects training-set and model variability (the held-out
    rows are never resampled: the feature is constant inside a substrate).

    Returns one row per iteration with: cho_homo, term_homo, D_homo,
    cho_block, term_block, D_block.
    """
    groups = np.asarray(groups)
    block = substrate_level_cols(X, groups)
    fi = feat_names.index(focus)
    rng = np.random.default_rng(seed)

    rows = []
    for b in range(n_boot):
        homo, blk = {}, {}
        for s in SUBSTRATE_ORDER:
            te = groups == s
            tr_idx = np.flatnonzero(~te)
            bi = rng.choice(tr_idx, size=len(tr_idx), replace=True)
            Xb = np.vstack([X[bi], X[te]])
            yb = np.concatenate([y[bi], y[te]])
            mask = np.r_[np.zeros(len(bi), bool), np.ones(int(te.sum()), bool)]
            sm = compute_group_shap(
                Xb, yb, mask, feat_names, n_estimators=n_estimators,
                random_state=seed * 10_000 + b, model_type=model_type,
            )["shap_mean"]
            homo[s], blk[s] = float(sm[fi]), float(sm[block].sum())
        th = float(np.mean([homo[s] for s in TERMINAL_NAMES]))
        tb = float(np.mean([blk[s] for s in TERMINAL_NAMES]))
        rows.append(dict(
            cho_homo=homo[CHO_NAME], term_homo=th, D_homo=th - homo[CHO_NAME],
            cho_block=blk[CHO_NAME], term_block=tb, D_block=tb - blk[CHO_NAME],
        ))
        if (b + 1) % 25 == 0:
            logger.info("    [%s] bootstrap %d/%d", model_type, b + 1, n_boot)
    return pd.DataFrame(rows)


# ============================================================================
# 5.  Shared helpers
# ============================================================================
MIN_N_RELIABLE = 30     # held-out groups smaller than this are not reported


def short_label(s: str) -> str:
    return s.replace(" oxide", "").replace("Isopropyl glycidyl ether", "IGE")
