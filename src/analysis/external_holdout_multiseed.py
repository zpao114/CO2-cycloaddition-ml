# -*- coding: utf-8 -*-
"""
external_holdout_multiseed.py
==============================

Task B — Multi-seed stability validation of the external holdout R² = 0.071
that the paper currently reports with a single seed (2026).

Background
----------
The reviewer flagged that the single-seed external-holdout number is
unstable. The reported value lives in
``src/models/persistence/405_external_validation.py`` and corresponds to the
``PCL-AE-128 + DualANN`` row of
``results_external_validation/external_validation_results.csv``
(``r2 = 0.07106518745422363, mae = 0.1436, n_test = 317, seed = 2026``).

Strategy
--------
Rather than reimplementing the full pipeline, this script:
  1. Imports the canonical helper functions from ``405_external_validation.py``.
  2. Monkey-patches the module-level ``SEED`` constant so that every code
     path inside the imported functions (data shuffle, DualANN init,
     KFold inner loops) picks up the new seed.
  3. Re-runs *only* the ``PCL-AE-128 + DualANN`` evaluation for the target
     seed (the model that produced R² = 0.071). Other tree models are
     skipped — they share the same train/test split, so the seed-controlled
     variability originates from the split itself.
  4. Reports R² mean ± SD across the three seeds, plus a per-substrate
     breakdown of which reactions fell into the holdout for each seed
     (different seeds produce different holdouts because the split is
     random within the 85% train pool).

Outputs
-------
``results_external_holdout/multiseed_summary.csv``
    columns: seed, r2, mae, n_external
``results_external_holdout/multiseed_per_substrate.csv``
    columns: seed, reactant_name, n_test_in_substrate, mean_yield
``results_external_holdout/multiseed_log.txt``
    human-readable run log

Usage
-----
    python src/analysis/external_holdout_multiseed.py
    python src/analysis/external_holdout_multiseed.py --seeds 42 2026 9999

Reproducibility
---------------
The script is fully deterministic w.r.t. (seed, torch, cuda availability).
Run from the project root with the same Python environment that produced
``external_validation_results.csv``.

References
----------
* Original external-validation script: ``src/models/persistence/405_external_validation.py``
* Original external-validation results: ``results_external_validation/external_validation_results.csv``
* Single-seed value (PCL-AE-128 + DualANN): ``r2 = 0.0711, mae = 0.1436, n = 317, seed = 2026``
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(os.environ.get("CO2_PROJECT_ROOT",
                                  r"D:\machine-learning\CO2-cycloaddition"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Import the canonical external-validation helpers. SEED is monkey-patched
# below before each call, so the inner KFold splits (SEED + 999,
# SEED + 1000 + fold_idx, SEED + 7 + fold_idx) all derive from the new seed.
# ---------------------------------------------------------------------------
import importlib.util

_EV_PATH = PROJECT_ROOT / "src" / "models" / "persistence" / "405_external_validation.py"
_spec = importlib.util.spec_from_file_location("ev_external_validation", _EV_PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)  # type: ignore[union-attr]

OUT_DIR = PROJECT_ROOT / "results_external_holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("external_holdout_multiseed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch RNGs globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_train_test_split(n: int, seed: int, holdout_ratio: float = 0.15):
    """Mirror the original split protocol: shuffle then take first n_test rows."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(n * holdout_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return sorted(train_idx.tolist()), sorted(test_idx.tolist())


def run_one_seed(seed: int) -> dict:
    """Train PCL-AE-128 + DualANN on the holdout split defined by ``seed``."""
    logger.info("=" * 72)
    logger.info("  [seed=%d] starting external-holdout evaluation", seed)
    logger.info("=" * 72)

    # Monkey-patch SEED inside the imported module so its inner KFold splits
    # (SEED + 999, SEED + 1000 + fold_idx, SEED + 7 + fold_idx) all derive
    # from this seed.
    ev.SEED = seed
    set_global_seed(seed)

    # Load data exactly as 405 does. Note: load_data() returns the 2116-row
    # training pool filtered from the master CSV (after dropping NaN/0
    # yields, applying extraction_status == valid). See data_split.py for
    # the canonical filter sequence.
    X_drfp, X_xtb, y, df_clean, xtb_cols = ev.load_data()

    n = len(y)
    logger.info("  [seed=%d] holdout pool size = %d", seed, n)
    train_idx, test_idx = make_train_test_split(n, seed)
    logger.info("  [seed=%d] n_train=%d, n_test=%d",
                seed, len(train_idx), len(test_idx))

    # Run the FULL evaluate_on_holdout so we get the PCL-AE-128+DualANN row
    # exactly as the canonical script reports it. This evaluates all 7
    # models, but we only keep the PCL-AE-128+DualANN row in the summary.
    results, records, df_pred = ev.evaluate_on_holdout(
        X_drfp, X_xtb, y, train_idx, test_idx, latent_dim=128,
    )
    pcl_dual = next(r for r in records if r["model"] == "PCL-AE-128+DualANN")

    # Per-substrate breakdown: which reactants fell into the test set for
    # THIS seed? (Different seeds produce different holdouts because the
    # 85/15 split is random.)
    df_test = df_clean.iloc[test_idx].copy()
    per_substrate = (
        df_test.groupby("reactant_name", as_index=False)
        .agg(n_test_in_substrate=("yield_pct", "size"),
             mean_yield=("yield_pct", "mean"))
    )

    summary = {
        "seed": seed,
        "r2": float(pcl_dual["r2"]),
        "mae": float(pcl_dual["mae"]),
        "n_external": int(len(test_idx)),
        "per_substrate": per_substrate,
    }
    logger.info("  [seed=%d] PCL-AE-128+DualANN R²=%.4f, MAE=%.4f, n_test=%d",
                seed, summary["r2"], summary["mae"], summary["n_external"])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed validation of external-holdout R² = 0.071 "
                    "(PCL-AE-128 + DualANN).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 9999],
                   help="Random seeds to evaluate (default: 42 2026 9999).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    seeds = args.seeds
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Output dir   : %s", OUT_DIR)
    logger.info("Seeds to run : %s", seeds)

    summaries: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        try:
            summaries.append(run_one_seed(seed))
        except Exception as ex:  # noqa: BLE001
            logger.exception("[seed=%d] FAILED: %s", seed, ex)
            raise
    elapsed = time.time() - t0

    # ── Build per-seed summary CSV ──────────────────────────────────────────
    summary_df = pd.DataFrame([
        {"seed": s["seed"], "r2": s["r2"], "mae": s["mae"],
         "n_external": s["n_external"]}
        for s in summaries
    ])
    summary_csv = OUT_DIR / "multiseed_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    r2_arr = np.array([s["r2"] for s in summaries], dtype=float)
    mae_arr = np.array([s["mae"] for s in summaries], dtype=float)
    r2_mean, r2_std = float(r2_arr.mean()), float(r2_arr.std(ddof=1))
    mae_mean, mae_std = float(mae_arr.mean()), float(mae_arr.std(ddof=1))

    # ── Build per-substrate CSV (one row per (seed, reactant)) ──────────────
    per_substrate_rows: list[dict] = []
    for s in summaries:
        for _, r in s["per_substrate"].iterrows():
            per_substrate_rows.append({
                "seed": s["seed"],
                "reactant_name": r["reactant_name"],
                "n_test_in_substrate": int(r["n_test_in_substrate"]),
                "mean_yield": float(r["mean_yield"]),
            })
    per_substrate_df = pd.DataFrame(per_substrate_rows)
    substrate_csv = OUT_DIR / "multiseed_per_substrate.csv"
    per_substrate_df.to_csv(substrate_csv, index=False, encoding="utf-8-sig")

    # ── Human-readable log ─────────────────────────────────────────────────
    log_lines = [
        "=" * 72,
        "  External Holdout — Multi-Seed Validation",
        "=" * 72,
        f"Model        : PCL-AE-128 + DualANN (the model that gave R²=0.071)",
        f"Seeds        : {seeds}",
        f"Train pool   : 2116 rows (filtered from 2490 by extraction_status "
        f"== valid, yield > 0, no NaN)",
        f"Holdout frac : 15%",
        f"Total time   : {elapsed/60:.1f} min",
        "",
        f"{'seed':>8}  {'r2':>10}  {'mae':>10}  {'n_test':>8}",
        "  " + "-" * 44,
    ]
    for s in summaries:
        log_lines.append(
            f"{s['seed']:>8}  {s['r2']:>10.4f}  {s['mae']:>10.4f}  "
            f"{s['n_external']:>8d}"
        )
    log_lines += [
        "  " + "-" * 44,
        f"R²  : mean = {r2_mean:.4f}, std (ddof=1) = {r2_std:.4f}",
        f"MAE : mean = {mae_mean:.4f}, std (ddof=1) = {mae_std:.4f}",
        "",
        f"Saved: {summary_csv}",
        f"Saved: {substrate_csv}",
        "=" * 72,
    ]
    log_text = "\n".join(log_lines)
    log_path = OUT_DIR / "multiseed_log.txt"
    log_path.write_text(log_text + "\n", encoding="utf-8")
    logger.info("\n%s", log_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
