# -*- coding: utf-8 -*-
"""
honest_loso_shap_validation.py
================================
Re-run §3.3.1 SHAP on the HONEST per-fold retrained encoder to show that
the CHO sign reversal on ``sub_homo_eV`` is an operational feature-space
diagnostic of failure under LOSO, NOT a leakage artefact of the cached
PCL-AE encoder.

Setup
-----
For each of 5 seeds × 5 LOSO substrates:
    1. Train a HONEST per-fold PCL-AE on the 4-substrate training fold
       (no access to the held-out substrate's structure during AE fitting).
    2. Encode train and test through the honest encoder.
    3. Fit XGBoost (audit-canonical config) on the 32-D xTB block of
       the train fold (NOT the 128-D latent — we want SHAP on the
       physically meaningful descriptor block that the main text reports).
    4. Run TreeSHAP on the test fold, extract the ``sub_homo_eV`` column,
       and compute mean + sign.

Output
------
results_honest_groupkfold/honest_loso_sub_homo_shap.csv
results_honest_groupkfold/honest_loso_sub_homo_shap_report.txt

Why 32-D xTB (not 128-D latent)?
---------------------------------
The v3 main text §3.3.1 reports SHAP values on the 32-D xTB descriptor
block (the same block as ``results_step4_5/shap_xtb_values.csv``).  Using
the latent would conflate two distinct claims:
    (a) the encoder's latent space distinguishes CHO (a representation claim);
    (b) the SHAP attribution on ``sub_homo_eV`` reverses sign (a feature
        attribution claim).
The honest per-fold test in this script only addresses (b).  Claim (a) is
addressed by ``807_cross_model_shap.py`` independently.

Why honest per-fold retraining?
-------------------------------
If the cached PCL-AE encoder has memorised a yield-label surface over the
full dataset, then any SHAP computed on its latent codes is contaminated.
Re-running SHAP on an encoder that has never seen the held-out substrate
isolates the operational diagnostic from the leakage artefact.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# Force UTF-8 stdout on Windows PowerShell (cp936 default)
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path setup — mirror pcl_ae_fair_comparison_multiseed.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import BEST_LAMBDA_PROP, BEST_LATENT_DIM                # noqa: E402
from utils_features import MechanismAwareLOSO, MECHANISM_GROUPS    # noqa: E402
from utils_rxn import read_drfp                                     # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from pcl_ae_fair_comparison import (                              # noqa: E402
    load_data,
    train_pcl_ae_model,
    encode_with_model,
    XGB_PARAMS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_CSV = PROJECT_ROOT / "results" / "results_cho_diagnostic" / "co2_drfp_xtb_extended.csv"
OUT_DIR = PROJECT_ROOT / "results_honest_groupkfold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "honest_loso_sub_homo_shap.csv"
REPORT_PATH = OUT_DIR / "honest_loso_sub_homo_shap_report.txt"

FEATURE_OF_INTEREST = "sub_homo_eV"
SEEDS = [42, 2024, 7, 9998, 7777]
DEFAULT_EPOCHS = 120   # matches pcl_ae_fair_comparison_multiseed.py
LATENT_DIM = int(BEST_LATENT_DIM)
LAMBDA_PROP = float(BEST_LAMBDA_PROP)


# ---------------------------------------------------------------------------
# Feature construction — mirrors generate_shap_for_901.py exactly
# (delta_E_HL, global_hardness, nucleophilicity etc. are DERIVED at runtime,
# they are NOT raw CSV columns; the 32-D block is reconstructed from raw cols)
# ---------------------------------------------------------------------------
def build_xtb_block(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build the 32-D xTB feature matrix (mirrors generate_shap_for_901.py exactly).

    Returns (X, feat_names) where X is (n, 32) float64 and feat_names is
    the authoritative list matching ``results_step4_5/shap_xtb_values.csv``.

    Derived columns (not in raw CSV):
        delta_E_HL       = cat_homo_eV - sub_lumo_eV
        delta_E_LL       = cat_lumo_eV - sub_lumo_eV
        global_hardness  = sub_gap_eV / 2
        global_softness  = 1 / (2 * global_hardness), clamped
        nucleophilicity   = -cat_homo_eV
        electrophilicity  = cat_lumo_eV

    The mechanism one-hot columns (mech_NUC etc.) are present in the output
    schema but may be zero-valued in the current dataset — we include them for
    schema compatibility with shap_xtb_values.csv.
    """
    # --- xTB raw block (7 cols) ---
    XTB_RAW = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
    ]
    XTB_raw = np.nan_to_num(
        df[XTB_RAW].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    # --- derived block (6 cols) ---
    chv = np.nan_to_num(df["cat_homo_eV"].to_numpy(dtype=np.float64), nan=0.0)
    slv = np.nan_to_num(df["sub_lumo_eV"].to_numpy(dtype=np.float64), nan=0.0)
    clv = np.nan_to_num(df["cat_lumo_eV"].to_numpy(dtype=np.float64), nan=0.0)
    sgv = np.nan_to_num(df["sub_gap_eV"].to_numpy(dtype=np.float64), nan=0.0)

    delta_E_HL  = chv - slv
    delta_E_LL  = clv - slv
    hardness     = sgv / 2.0
    softness     = np.where(hardness > 1e-8, 1.0 / (2.0 * hardness), 0.0)
    nucleophil   = -chv
    electrophil = clv

    derived = np.column_stack([
        np.nan_to_num(delta_E_HL,  nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(delta_E_LL,  nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(hardness,   nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(softness,    nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(nucleophil, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(electrophil, nan=0.0, posinf=0.0, neginf=0.0),
    ])

    # --- catalyst block (4 cols) ---
    cat_cols = ["cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D"]
    X_cat = np.nan_to_num(
        df[cat_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    # --- solvent block (3 cols) ---
    solv_cols = ["solv_homo_eV", "solv_lumo_eV", "solv_gap_eV"]
    X_solv = np.nan_to_num(
        df[solv_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    # --- condition block (5 cols) ---
    temp_arr = _find_col(df, "temperature")
    press_arr = pd.to_numeric(df["pressure (MPa)"], errors="coerce").fillna(
        pd.to_numeric(df["pressure (MPa)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64)
    time_arr = pd.to_numeric(df["time (h)"], errors="coerce").fillna(
        pd.to_numeric(df["time (h)"], errors="coerce").median()
    ).to_numpy(dtype=np.float64) if "time (h)" in df.columns else np.zeros(len(df))
    time_log = np.log1p(np.maximum(time_arr, 0.0))

    loadings = np.zeros(len(df), dtype=np.float64)
    for lc in [f"catalyst_{i}_loading_mol%" for i in range(1, 5)]:
        if lc in df.columns:
            loadings += np.nan_to_num(
                pd.to_numeric(df[lc], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
    loading_log = np.log1p(np.maximum(loadings, 0.0))

    has_solvent = (
        df["all_solvents_normalized"].notna() &
        df["all_solvents_normalized"].astype(str).str.strip().ne("")
    ).astype(float).to_numpy()
    has_reagent = (
        (df["reagent_1_name"].notna() & df["reagent_1_name"].astype(str).str.strip().ne("")) |
        (df["reagent_2_name"].notna() & df["reagent_2_name"].astype(str).str.strip().ne(""))
    ).astype(float).to_numpy()

    X_cond = np.column_stack([temp_arr, press_arr, time_log, loading_log, has_solvent, has_reagent])

    # --- mechanism one-hot (5 cols, schema-compatible but may be zero-valued) ---
    mech_cols = ["mech_NUC", "mech_LAC", "mech_BAS", "mech_BIF", "mech_OTH"]
    X_mech = np.nan_to_num(
        df[mech_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    ) if all(c in df.columns for c in mech_cols) else np.zeros((len(df), 5))

    # --- cat_electrophilicity (schema-compatible) ---
    if "cat_electrophilicity" in df.columns:
        X_cat_elec = np.nan_to_num(
            df["cat_electrophilicity"].to_numpy(dtype=np.float64).reshape(-1, 1),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    else:
        X_cat_elec = np.zeros((len(df), 1))

    # Assemble: raw(7) + derived(6) + cat(4) + cat_elec(1) + solv(3) + cond(6) + mech(5) = 32
    X = np.hstack([
        XTB_raw,           # 7
        derived,           # 6
        X_cat,             # 4
        X_cat_elec,        # 1
        X_solv,           # 3
        X_cond,            # 6
        X_mech,            # 5
    ]).astype(np.float64)

    feat_names = (
        XTB_RAW +  # 7
        ["delta_E_HL", "delta_E_LL", "global_hardness", "global_softness",
         "nucleophilicity", "electrophilicity"] +  # 6
        cat_cols +  # 4
        ["cat_electrophilicity"] +  # 1
        solv_cols +  # 3
        ["temperature", "pressure", "time_log", "loading_log", "has_solvent", "has_reagent"] +  # 6
        mech_cols   # 5
    )
    assert len(feat_names) == 32, f"Expected 32, got {len(feat_names)}"
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


log = logging.getLogger("honest_loso_shap")


# ---------------------------------------------------------------------------
# xTB feature extraction (robust to column name encoding quirks on Windows)
# ---------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Find a numeric column by prefix (robust to degree-symbol encoding)."""
    hits = [c for c in df.columns if prefix.lower() in c.lower()]
    if not hits:
        return np.zeros(len(df), dtype=np.float64)
    col = df[hits[0]]
    return pd.to_numeric(col, errors="coerce").fillna(col.median()).to_numpy(dtype=np.float64)


def extract_xtb_block(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X_xtb, feat_names) — mirrors generate_shap_for_901.py exactly."""
    return build_xtb_block(df)


# ---------------------------------------------------------------------------
# Per-fold: fit honest PCL-AE, encode, fit XGB on 32-D xTB, compute TreeSHAP
# ---------------------------------------------------------------------------
def shap_for_one_fold(
    df: pd.DataFrame,
    X_drfp: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feat_names: list[str],
    ae_seed: int,
    ae_epochs: int,
) -> tuple[float, float, int, int]:
    """Returns (mean_signed_shap_sub_homo, sd_signed_shap_sub_homo, n_train, n_test)."""
    y_tr = y[train_mask]
    y_te = y[test_mask]

    # Step 1: train HONEST PCL-AE on the train fold only (no test access)
    t0 = time.time()
    model = train_pcl_ae_model(
        X_drfp[train_mask], y_tr,
        lambda_prop=200.0, epochs=ae_epochs, seed=ae_seed,
    )
    # Step 2: encode through honest encoder (only used to confirm test isolation,
    # NOT used for SHAP — SHAP is on the 32-D xTB block)
    _ = encode_with_model(model, X_drfp[train_mask])
    _ = encode_with_model(model, X_drfp[test_mask])
    t_ae = time.time() - t0

    # Step 3: fit XGB on the 32-D xTB block (audit-canonical config)
    X_tr = X_xtb[train_mask]
    X_te = X_xtb[test_mask]
    # Audit-canonical XGB params; override random_state per seed
    xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "random_state"}
    xgb_params["random_state"] = ae_seed

    # Use XGBoost native pred_contribs=True (avoids shap.TreeExplainer
    # UTF-8 decode issue on Windows with degree-symbol feature names).
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dtest  = xgb.DMatrix(X_te,  label=y_te,  feature_names=feat_names)
    booster = xgb.train(
        xgb_params, dtrain,
        num_boost_round=xgb_params.pop("n_estimators", 400),
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    # pop consumed n_estimators; remaining params go to xgb.XGBRegressor (unused below)
    xgb_params["n_estimators"] = 400

    # XGBoost pred_contribs: (n_test, n_feat+1) with bias as last column
    contribs = booster.predict(dtest, pred_contribs=True, validate_features=False)
    sv_te = contribs[:, :-1].astype(np.float64)   # drop bias column
    idx = feat_names.index(FEATURE_OF_INTEREST)
    col = sv_te[:, idx]
    mean_shap = float(col.mean())
    sd_shap = float(col.std(ddof=1)) if len(col) > 1 else 0.0

    log.info(
        "  seed=%d  train=%4d  test=%4d  AE=%.1fs  %s SHAP mean=%+.4f sd=%.4f",
        ae_seed, int(train_mask.sum()), int(test_mask.sum()), t_ae,
        FEATURE_OF_INTEREST, mean_shap, sd_shap,
    )
    return mean_shap, sd_shap, int(train_mask.sum()), int(test_mask.sum())


# ---------------------------------------------------------------------------
# 5 LOSO substrates × 1 AE seed → 5 rows of dict
# ---------------------------------------------------------------------------
def loso_substrate_shap(
    df: pd.DataFrame,
    X_drfp: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    ae_seed: int,
    ae_epochs: int,
) -> list[dict]:
    """Run 5 LOSO folds under one AE seed. Returns 5 row dicts."""
    loso = MechanismAwareLOSO(df["reactant_name"].values, MECHANISM_GROUPS)
    substrates = sorted(df["reactant_name"].unique())
    rows: list[dict] = []
    for held in substrates:
        train_mask, test_mask = loso.get_train_val_split(held)
        if int(test_mask.sum()) < 10:
            log.warning("Skipping %s: n_test=%d < 10", held, int(test_mask.sum()))
            continue
        mean_shap, sd_shap, n_train, n_test = shap_for_one_fold(
            df, X_drfp, X_xtb, y,
            train_mask, test_mask, feat_names,
            ae_seed=ae_seed, ae_epochs=ae_epochs,
        )
        rows.append({
            "seed": ae_seed,
            "substrate": held,
            "n_train": n_train,
            "n_test": n_test,
            "mean_signed_shap_sub_homo_eV": mean_shap,
            "sd_signed_shap_sub_homo_eV": sd_shap,
            "sign": "positive" if mean_shap > 0 else "negative",
        })
    return rows


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def make_verdict(df: pd.DataFrame) -> tuple[bool, str]:
    """Return (confirmed, message).

    CONFIRMED iff CHO has 5/5 seeds with sign=='negative'.
    NOT CONFIRMED iff CHO has ≤4/5 seeds with sign=='negative'.
    """
    cho = df[df["substrate"] == "Cyclohexene oxide"]
    n_pos_cho = int((cho["sign"] == "positive").sum())
    n_neg_cho = int((cho["sign"] == "negative").sum())
    n_total = int(len(cho))

    lines = []
    lines.append("=== Verdict ===")
    lines.append(f"CHO  sub_homo_eV across {n_total} seeds:")
    for _, row in cho.sort_values("seed").iterrows():
        lines.append(
            f"  seed={int(row['seed']):>5d}   mean={row['mean_signed_shap_sub_homo_eV']:+.4f}   "
            f"sd={row['sd_signed_shap_sub_homo_eV']:.4f}   sign={row['sign']}"
        )
    if n_total == 0:
        confirmed = False
        verdict = "NOT CONFIRMED — no CHO rows in output"
    elif n_pos_cho == 0:
        confirmed = True
        verdict = (
            f"HONEST-LOSO SHAP CHO sign reversal CONFIRMED "
            f"({n_neg_cho}/{n_total} seeds negative)"
        )
    else:
        confirmed = False
        verdict = (
            f"NOT CONFIRMED — sign reversal does NOT survive honest retraining "
            f"on all seeds ({n_pos_cho}/{n_total} seeds positive on CHO)"
        )
    lines.append("")
    lines.append(verdict)
    return confirmed, "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-print 5x5 matrix (rows=substrate, cols=seed, cell=mean_signed_shap)
# ---------------------------------------------------------------------------
def print_matrix(df: pd.DataFrame) -> None:
    substrates = sorted(df["substrate"].unique())
    seeds = sorted(df["seed"].unique())
    piv = df.pivot(index="substrate", columns="seed", values="mean_signed_shap_sub_homo_eV")
    log.info("=== sub_homo_eV mean signed SHAP matrix (honest-LOSO, 32-D xTB block) ===")
    log.info("Rows = held-out substrate; Cols = AE seed")
    # Format
    header = "substrate".ljust(22) + "".join(f"  seed={s:>5d}  " for s in seeds)
    log.info(header)
    for sub in substrates:
        row_str = sub.ljust(22)
        for s in seeds:
            v = piv.loc[sub, s] if (sub in piv.index and s in piv.columns) else np.nan
            row_str += f"  {v:+8.4f}  "
        log.info(row_str)
    log.info("(Sign convention: + = positive attribution, - = negative attribution.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Honest-LOSO SHAP validation — confirm CHO sign reversal "
                    "is not a leakage artefact of cached PCL-AE encoder."
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Per-fold AE training epochs (default {DEFAULT_EPOCHS}).",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="AE seeds to iterate over (default 42 2024 7 9998 7777).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    log.info("=" * 78)
    log.info("  Honest-LOSO SHAP validation")
    log.info("  device=auto   per_fold_epochs=%d   seeds=%s", args.epochs, args.seeds)
    log.info("  latent_dim=%d   lambda_prop=%s   feature_of_interest=%s",
             LATENT_DIM, LAMBDA_PROP, FEATURE_OF_INTEREST)
    log.info("=" * 78)

    t_start = time.time()

    # ---- Load data ----
    df, X_drfp = load_data()
    y = df["yield (%)"].to_numpy(dtype=np.float32) / 100.0
    X_xtb, feat_names = extract_xtb_block(df)
    assert FEATURE_OF_INTEREST in feat_names, (
        f"{FEATURE_OF_INTEREST} not in 32-D xTB block; cannot compute SHAP."
    )
    log.info("Data loaded: %d rows, DRFP=%s, XTB-32=%s",
             len(df), X_drfp.shape, X_xtb.shape)
    log.info("  32-D xTB block: %d features: %s", len(feat_names), feat_names)

    # ---- Iterate over seeds ----
    all_rows: list[dict] = []
    for seed in args.seeds:
        log.info("──── Seed %d ────", seed)
        rows = loso_substrate_shap(
            df, X_drfp, X_xtb, y, feat_names,
            ae_seed=seed, ae_epochs=args.epochs,
        )
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)

    # ---- Add n_positive_seeds_among_5 column ----
    sign_count = out_df.groupby("substrate")["sign"].apply(
        lambda s: int((s == "positive").sum())
    ).rename("n_positive_seeds_among_5")
    out_df = out_df.merge(sign_count, on="substrate")

    # Reorder columns
    out_df = out_df[[
        "seed", "substrate", "n_train", "n_test",
        "mean_signed_shap_sub_homo_eV", "sd_signed_shap_sub_homo_eV",
        "sign", "n_positive_seeds_among_5",
    ]].sort_values(["substrate", "seed"]).reset_index(drop=True)

    # ---- Persist CSV ----
    out_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    log.info("Saved CSV: %s  (%d rows)", CSV_PATH, len(out_df))

    # ---- Pretty-print matrix ----
    print_matrix(out_df)

    # ---- Verdict ----
    confirmed, verdict_text = make_verdict(out_df)
    log.info("\n%s", verdict_text)

    # ---- Persist report ----
    substrates = sorted(out_df["substrate"].unique())
    report_lines = [
        "Honest-LOSO SHAP Validation — Report",
        "=" * 78,
        f"Feature: {FEATURE_OF_INTEREST}",
        f"Block: 32-D xTB (matches results_step4_5/shap_xtb_values.csv columns)",
        f"Encoder: HONEST per-fold retrained PCL-AE (lambda={LAMBDA_PROP}, "
        f"latent_dim={LATENT_DIM}, epochs={args.epochs})",
        f"Seeds: {args.seeds}",
        f"Substrates: {substrates}",
        "",
        "sub_homo_eV mean signed SHAP (5x5 matrix)",
        "-" * 78,
    ]
    seeds = sorted(out_df["seed"].unique())
    piv = out_df.pivot(index="substrate", columns="seed",
                       values="mean_signed_shap_sub_homo_eV")
    header = "substrate".ljust(22) + "".join(f"  seed={s:>5d}  " for s in seeds)
    report_lines.append(header)
    for sub in substrates:
        row_str = sub.ljust(22)
        for s in seeds:
            v = piv.loc[sub, s] if (sub in piv.index and s in piv.columns) else np.nan
            row_str += f"  {v:+8.4f}  "
        report_lines.append(row_str)
    report_lines.append("")
    report_lines.append("Per-seed detail (CHO only — the substrate of interest):")
    cho = out_df[out_df["substrate"] == "Cyclohexene oxide"].sort_values("seed")
    for _, row in cho.iterrows():
        report_lines.append(
            f"  seed={int(row['seed']):>5d}  "
            f"n_test={int(row['n_test']):>4d}  "
            f"mean={row['mean_signed_shap_sub_homo_eV']:+.4f}  "
            f"sd={row['sd_signed_shap_sub_homo_eV']:.4f}  "
            f"sign={row['sign']}"
        )
    report_lines.append("")
    report_lines.append(verdict_text)
    report_lines.append("")
    report_lines.append(f"Total wall time: {time.time() - t_start:.1f} s")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Saved report: %s", REPORT_PATH)

    return 0 if confirmed else 2   # exit 2 = NOT CONFIRMED (caller can distinguish)


if __name__ == "__main__":
    raise SystemExit(main())
