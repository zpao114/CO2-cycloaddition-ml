# -*- coding: utf-8 -*-
"""
external_holdout_multiseed_v2.py
==================================

Task B (v2) — Multi-seed stability validation of the external holdout R² = 0.071
that the paper currently reports with a single seed (2026).

This is the corrected version of ``external_holdout_multiseed.py``.

Bug fixed
---------
The v1 script used ``np.random.default_rng(seed).shuffle(...)`` (new PCG64 RNG),
but the canonical ``405_external_validation.main()`` uses
``np.random.seed(SEED); np.random.shuffle(np.arange(n))`` (legacy MT19937 RNG).
These are DIFFERENT RNG APIs and produce DIFFERENT shuffles, even with the
same seed value. This v2 script uses the EXACT legacy protocol.

Verified protocol match
-----------------------
``405_external_validation.main()`` does, in order:

    random.seed(SEED)
    np.random.seed(SEED)             # legacy global RNG (MT19937)
    torch.manual_seed(SEED)
    X_drfp, X_xtb, y, _, _ = load_data()  # no random ops (deterministic filter)
    indices = np.arange(n)
    np.random.shuffle(indices)       # uses legacy global NumPy RNG
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

This v2 script mirrors this EXACT sequence in ``run_one_seed()`` and
``make_train_test_split()``.

Strategy
--------
  1. Imports the canonical helper functions from ``405_external_validation.py``.
  2. Monkey-patches the module-level ``SEED`` constant so that every code
     path inside the imported functions (DualANN init, KFold inner loops,
     PCA random_state, etc.) picks up the new seed.
  3. Seeds the global NumPy legacy RNG with ``np.random.seed(seed)`` BEFORE
     calling ``np.random.shuffle(np.arange(n))``.
  4. Re-runs *only* the ``PCL-AE-128 + DualANN`` evaluation for the target
     seed.
  5. Reports R² mean ± SD across the three seeds.

Outputs
-------
``results_external_holdout/multiseed_v2_summary.csv``
    columns: seed, r2, mae, n_external
``results_external_holdout/multiseed_v2_log.txt``
    human-readable run log

Usage
-----
    python src/analysis/external_holdout_multiseed_v2.py
    python src/analysis/external_holdout_multiseed_v2.py --seeds 2026
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
_spec = importlib.util.spec_from_file_location("ev_external_validation_v2", _EV_PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)  # type: ignore[union-attr]

OUT_DIR = PROJECT_ROOT / "results_external_holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("external_holdout_multiseed_v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch RNGs globally.

    Uses the LEGACY NumPy RNG (MT19937) — matches 405_external_validation.main().
    """
    random.seed(seed)
    np.random.seed(seed)  # legacy MT19937 — same API as 405.main()
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_train_test_split(n: int, seed: int, holdout_ratio: float = 0.15):
    """EXACT mirror of 405_external_validation.main() shuffle protocol.

    Caller MUST have already invoked ``np.random.seed(seed)`` (e.g. via
    ``set_global_seed(seed)``) so the legacy global NumPy RNG state matches
    what 405.main() produces.

    Mirrors these lines from 405_external_validation.main():
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_test = int(n * HOLDOUT_RATIO)
        test_idx  = indices[:n_test]
        train_idx = indices[n_test:]
    """
    indices = np.arange(n)
    np.random.shuffle(indices)  # uses legacy global NumPy RNG (MT19937)
    n_test = int(n * holdout_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return sorted(train_idx.tolist()), sorted(test_idx.tolist())


def run_one_seed(seed: int, debug_print_split: bool = False) -> dict:
    """Train PCL-AE-128 + DualANN on the holdout split defined by ``seed``."""
    logger.info("=" * 72)
    logger.info("  [seed=%d] starting external-holdout evaluation", seed)
    logger.info("=" * 72)

    # 1. Monkey-patch SEED inside the imported module so its inner KFold splits
    #    (SEED + 999, SEED + 1000 + fold_idx, SEED + 7 + fold_idx) all derive
    #    from this seed.
    ev.SEED = seed

    # 2. Seed global RNGs — MUST happen BEFORE load_data() and BEFORE the
    #    shuffle in make_train_test_split(), matching the order in 405.main().
    set_global_seed(seed)

    # 3. Load data exactly as 405 does. load_data() is deterministic (just
    #    reads CSV + filters via manifest), no random ops consumed.
    X_drfp, X_xtb, y, df_clean, xtb_cols = ev.load_data()

    n = len(y)
    logger.info("  [seed=%d] holdout pool size = %d", seed, n)
    train_idx, test_idx = make_train_test_split(n, seed)
    logger.info("  [seed=%d] n_train=%d, n_test=%d",
                seed, len(train_idx), len(test_idx))

    if debug_print_split:
        logger.info("  [seed=%d] train_idx[:10]=%s", seed, train_idx[:10])
        logger.info("  [seed=%d] test_idx[:10]=%s",  seed, test_idx[:10])

    # 4. Run the FULL evaluate_on_holdout so we get the PCL-AE-128+DualANN row
    #    exactly as the canonical script reports it.
    results, records, df_pred = ev.evaluate_on_holdout(
        X_drfp, X_xtb, y, train_idx, test_idx, latent_dim=128,
    )
    pcl_dual = next(r for r in records if r["model"] == "PCL-AE-128+DualANN")

    summary = {
        "seed": seed,
        "r2": float(pcl_dual["r2"]),
        "mae": float(pcl_dual["mae"]),
        "n_external": int(len(test_idx)),
        "train_idx": train_idx,
        "test_idx": test_idx,
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
                    "(PCL-AE-128 + DualANN) — v2 with correct legacy shuffle.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 9999],
                   help="Random seeds to evaluate (default: 42 2026 9999).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    p.add_argument("--debug-print-split", action="store_true",
                   help="Print first-10 train/test indices for each seed.")
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
    logger.info("Shuffle API  : legacy np.random.seed + np.random.shuffle "
                "(matches 405.main)")

    summaries: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        try:
            summaries.append(
                run_one_seed(seed, debug_print_split=args.debug_print_split)
            )
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
    summary_csv = OUT_DIR / "multiseed_v2_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    r2_arr = np.array([s["r2"] for s in summaries], dtype=float)
    mae_arr = np.array([s["mae"] for s in summaries], dtype=float)
    r2_mean, r2_std = float(r2_arr.mean()), float(r2_arr.std(ddof=1))
    mae_mean, mae_std = float(mae_arr.mean()), float(mae_arr.std(ddof=1))

    # ── Human-readable log ─────────────────────────────────────────────────
    log_lines = [
        "=" * 72,
        "  External Holdout — Multi-Seed Validation (v2, legacy-RNG shuffle)",
        "=" * 72,
        f"Model        : PCL-AE-128 + DualANN (the model that gave R²=0.071)",
        f"Seeds        : {seeds}",
        f"Train pool   : 2116 rows (filtered from 2490 by extraction_status "
        f"== valid, yield > 0, no NaN)",
        f"Holdout frac : 15%",
        f"Shuffle API  : np.random.seed(seed) + np.random.shuffle(np.arange(n))",
        f"              (matches 405_external_validation.main() EXACTLY)",
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
        "Reference (paper single-seed value):",
        "  seed=2026:  R²=0.0711, MAE=0.1436, n_test=317",
        "",
        f"Saved: {summary_csv}",
        "=" * 72,
    ]
    log_text = "\n".join(log_lines)
    log_path = OUT_DIR / "multiseed_v2_log.txt"
    log_path.write_text(log_text + "\n", encoding="utf-8")
    logger.info("\n%s", log_text)

    # ── Console-friendly one-liner per seed ─────────────────────────────────
    print()
    print(f"[v2] Multi-seed PCL-AE-128+DualANN R² (legacy-RNG shuffle):")
    for s in summaries:
        print(f"  seed={s['seed']:<6}  R²={s['r2']:.4f}  "
              f"MAE={s['mae']:.4f}  n={s['n_external']}")
    print(f"  mean±SD  : R²={r2_mean:.4f}±{r2_std:.4f}  "
          f"MAE={mae_mean:.4f}±{mae_std:.4f}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
