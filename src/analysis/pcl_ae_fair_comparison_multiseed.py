# -*- coding: utf-8 -*-
"""
pcl_ae_fair_comparison_multiseed.py
====================================
**Task C v3 — multi-seed (5 seeds) of the 2×2 fair comparison.**

The single-seed `pcl_ae_fair_comparison.py` gives a single point estimate
for each of the 4 cells; without multiple seeds we have no confidence
interval on the headline latent-leakage number (+0.877 in PCL-AE,
−0.004 honest gain C3−C4).

This wrapper runs the inner 2×2 grid at 5 seeds:
    * seed 42   (matches the production single-seed result; control)
    * seed 2024 (independent seed, no overlap with seed 42)
    * seed 7    (independent seed)
    * seed 9998 (new)
    * seed 7777 (new)

For each seed:
    - The cached PCL-AE latent is the same on-disk latent (it is the
      as-published full-data trained encoder and does not depend on
      XGBoost or AE seed); so Cell 1 and Cell 2 are NOT re-run across
      seeds.
    - Only the per-fold encoders (Cell 3 / Cell 4) are retrained at
      each new AE seed; the XGBoost XGBRegressor is also re-fit at the
      new XGB seed.

This produces:
    - per-fold × seed table for Cell 3 and Cell 4
    - aggregate mean ± SD across seeds for C3−C4 and C1−C3

The output is saved to ``results_pcl_ae_fair_comparison_multiseed/``.
"""
from __future__ import annotations
import io
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

PROJECT_ROOT = r"D:\machine-learning\CO2-cycloaddition"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import BEST_LAMBDA_PROP, BEST_LATENT_DIM
from utils_features import MechanismAwareLOSO, MECHANISM_GROUPS

# Reuse the inner pipeline functions from the single-seed script.
sys.path.insert(0, os.path.dirname(__file__))
from pcl_ae_fair_comparison import (
    load_data, load_cached_pcl_latent,
    train_pcl_ae_model, encode_with_model,
    XGB_PARAMS,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "results_pcl_ae_fair_comparison_multiseed")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 2024, 7, 9998, 7777]  # 5-seed: 2 new seeds added for expanded audit
EPOCHS = 120   # matches single-seed run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("fair_comparison_multiseed")


def xgb_predict(
    Z_tr: np.ndarray, y_tr: np.ndarray,
    Z_te: np.ndarray, y_te: np.ndarray,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Train XGB (max_depth=6, paper LOSO protocol) with a given seed; return (R², MAE, preds)."""
    params = {k: v for k, v in XGB_PARAMS.items() if k not in ("random_state", "n_estimators")}
    model = xgb.XGBRegressor(
        **params,
        random_state=seed,
        n_estimators=400,
    )
    model.fit(Z_tr, y_tr)
    pred = np.clip(model.predict(Z_te), 0.0, 1.0)
    from sklearn.metrics import mean_absolute_error, r2_score
    return (
        float(r2_score(y_te, pred)),
        float(mean_absolute_error(y_te, pred)),
        pred,
    )


def main():
    log.info("=== Task C v3: 5-seed multi-seed audit of 2x2 fair comparison ===")
    log.info("Seeds: %s; per-fold epochs: %d", SEEDS, EPOCHS)
    log.info("Cell 1 / Cell 2 use the cached latent (production pipeline); only")
    log.info("Cell 3 / Cell 4 are re-run across seeds (per-fold encoders + XGB).")

    t0 = time.time()

    # ---- Load data and cached latents once ----
    df, X_drfp = load_data()
    y = df["yield (%)"].to_numpy(dtype=np.float32) / 100.0

    z_pcl_cached = load_cached_pcl_latent(df)
    # Standard-AE cached: load if present, else train once
    std_cached_path = os.path.join(
        PROJECT_ROOT, "results_pcl_ae_fair_comparison",
        "std_ae_cached_latent.npy")
    std_cached_path = os.path.normpath(std_cached_path)
    if os.path.exists(std_cached_path):
        z_std_cached = np.load(std_cached_path).astype(np.float32)
        log.info("Loaded cached Standard-AE latent from %s", std_cached_path)
    else:
        log.error("Cached Standard-AE latent not found at %s — run "
                  "pcl_ae_fair_comparison.py first to produce it.", std_cached_path)
        sys.exit(1)

    # ---- Cell 1 / Cell 2: cached, seed-invariant ----
    loso = MechanismAwareLOSO(df["reactant_name"].values, MECHANISM_GROUPS)
    substrates = sorted(df["reactant_name"].unique())

    cached_rows = []
    for held in substrates:
        train_mask, test_mask = loso.get_train_val_split(held)
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_test < 10:
            continue
        y_tr = y[train_mask]
        y_te = y[test_mask]
        r2_1, mae_1, _ = xgb_predict(
            z_pcl_cached[train_mask], y_tr,
            z_pcl_cached[test_mask], y_te,
            seed=42)
        r2_2, mae_2, _ = xgb_predict(
            z_std_cached[train_mask], y_tr,
            z_std_cached[test_mask], y_te,
            seed=42)
        cached_rows.append({"substrate": held, "n_train": n_train, "n_test": n_test,
                            "r2_c1": r2_1, "mae_c1": mae_1,
                            "r2_c2": r2_2, "mae_c2": mae_2})

    # ---- Cell 3 / Cell 4 across seeds ----
    per_seed_rows = []
    for seed in SEEDS:
        log.info("──── Seed %d ────", seed)
        for held in substrates:
            train_mask, test_mask = loso.get_train_val_split(held)
            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())
            if n_test < 10:
                continue
            y_tr = y[train_mask]
            y_te = y[test_mask]

            t1 = time.time()
            m_pcl = train_pcl_ae_model(
                X_drfp[train_mask], y_tr,
                lambda_prop=200.0, epochs=EPOCHS, seed=seed)
            z_tr_pcl = encode_with_model(m_pcl, X_drfp[train_mask])
            z_te_pcl = encode_with_model(m_pcl, X_drfp[test_mask])
            r2_3, mae_3, _ = xgb_predict(
                z_tr_pcl, y_tr, z_te_pcl, y_te,
                seed=seed)

            # Cell 4: Standard-AE per-fold
            m_std = train_pcl_ae_model(
                X_drfp[train_mask], None,
                lambda_prop=0.0, epochs=EPOCHS, seed=seed)
            z_tr_std = encode_with_model(m_std, X_drfp[train_mask])
            z_te_std = encode_with_model(m_std, X_drfp[test_mask])
            r2_4, mae_4, _ = xgb_predict(
                z_tr_std, y_tr, z_te_std, y_te,
                seed=seed)

            log.info("  seed=%d %s: C3=%+.4f C4=%+.4f (Δ=%+.4f) t=%.1fs",
                     seed, held, r2_3, r2_4, r2_3 - r2_4,
                     time.time() - t1)

            per_seed_rows.append({
                "seed": seed, "substrate": held,
                "n_train": n_train, "n_test": n_test,
                "r2_c3": r2_3, "mae_c3": mae_3,
                "r2_c4": r2_4, "mae_c4": mae_4,
                "delta_c3_c4": r2_3 - r2_4,
            })

    # ---- Persist ----
    cached_df = pd.DataFrame(cached_rows)
    seed_df = pd.DataFrame(per_seed_rows)

    cached_df.to_csv(os.path.join(OUT_DIR, "cached_cells.csv"),
                     index=False, encoding="utf-8-sig")
    seed_df.to_csv(os.path.join(OUT_DIR, "perfold_per_seed.csv"),
                   index=False, encoding="utf-8-sig")

    # ---- Aggregate ----
    # Per-cell means, std across folds, and for C3/C4 also across seeds
    def _agg(group):
        return pd.Series({
            "mean_r2": group["r2_c3" if "c3" in group.columns[2] else
                              group.columns[2]].mean(),
            "std_r2":  group["r2_c3" if "c3" in group.columns[2] else
                              group.columns[2]].std(ddof=1),
            "n_folds": len(group),
        })

    # C1 / C2 means (no seed dependence)
    c1_mean = cached_df["r2_c1"].mean()
    c1_std = cached_df["r2_c1"].std(ddof=1)
    c2_mean = cached_df["r2_c2"].mean()
    c2_std = cached_df["r2_c2"].std(ddof=1)

    # C3 / C4 across seeds (group by seed, then mean)
    c3_per_seed = seed_df.groupby("seed")["r2_c3"].mean()
    c4_per_seed = seed_df.groupby("seed")["r2_c4"].mean()
    c3_mean = float(c3_per_seed.mean())
    c3_std = float(c3_per_seed.std(ddof=1))
    c4_mean = float(c4_per_seed.mean())
    c4_std = float(c4_per_seed.std(ddof=1))

    # Honest gain C3 - C4: per seed
    honest_per_seed = (c3_per_seed - c4_per_seed).values
    honest_mean = float(honest_per_seed.mean())
    honest_std = float(honest_per_seed.std(ddof=1))

    # C1 - C3 leakage: per seed (C1 is seed-invariant)
    leak_per_seed = c1_mean - c3_per_seed.values
    leak_mean = float(leak_per_seed.mean())
    leak_std = float(leak_per_seed.std(ddof=1))

    summary = {
        "config": {
            "seeds": SEEDS,
            "per_fold_epochs": EPOCHS,
            "xgb_n_estimators": 400,
        },
        "cells": {
            "Cell 1 PCL-AE_cached":   {"mean_r2": c1_mean, "std_r2": c1_std,
                                        "n_folds": len(cached_df)},
            "Cell 2 StdAE_cached":    {"mean_r2": c2_mean, "std_r2": c2_std,
                                        "n_folds": len(cached_df)},
            "Cell 3 PCL-AE_perfold":  {"mean_r2": c3_mean, "std_r2": c3_std,
                                        "per_seed": c3_per_seed.tolist()},
            "Cell 4 StdAE_perfold":   {"mean_r2": c4_mean, "std_r2": c4_std,
                                        "per_seed": c4_per_seed.tolist()},
        },
        "contrasts": {
            "C3_minus_C4_honest_gain": {
                "mean": honest_mean,
                "std": honest_std,
                "per_seed": honest_per_seed.tolist(),
            },
            "C1_minus_C3_PCL_AE_leakage": {
                "mean": leak_mean,
                "std": leak_std,
                "per_seed": leak_per_seed.tolist(),
            },
        },
    }
    with open(os.path.join(OUT_DIR, "multiseed_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("\nMulti-seed summary:")
    log.info("  Cell 1 mean=%+.4f std=%.4f", c1_mean, c1_std)
    log.info("  Cell 2 mean=%+.4f std=%.4f", c2_mean, c2_std)
    log.info("  Cell 3 per seed: %s  ->  mean=%+.4f std=%.4f",
             [round(x, 4) for x in c3_per_seed.values], c3_mean, c3_std)
    log.info("  Cell 4 per seed: %s  ->  mean=%+.4f std=%.4f",
             [round(x, 4) for x in c4_per_seed.values], c4_mean, c4_std)
    log.info("  Honest gain (C3 - C4) per seed: %s  ->  mean=%+.4f std=%.4f",
             [round(x, 4) for x in honest_per_seed], honest_mean, honest_std)
    log.info("  PCL-AE leakage (C1 - C3) per seed: %s  ->  mean=%+.4f std=%.4f",
             [round(x, 4) for x in leak_per_seed], leak_mean, leak_std)

    log.info("\nTotal wall time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
