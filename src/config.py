# -*- coding: utf-8 -*-
"""
Project-level configuration. Single source of truth for tuned hyperparameters.

Workflow:
  201_ablation.py (DRFP variant ablation → λ sweep → full benchmark)
      ↓
  results_best_pipeline/drfp_ablation_meta.json    written
  results_best_pipeline/full_benchmark_results.csv written
  results_lambda_ablation/lambda_results.csv       written  ← BEST_LAMBDA_PROP source
  results_best_pipeline/drfp_ablation_meta.json    ← BEST_DRFP_VARIANT source

Hyperparameters below are AUTO-LOADED at import time from those artefacts,
NOT hard-coded. If 201_ablation.py has never been run, sensible fallbacks
are used (printed as warnings).
"""

import os
import warnings

# Project root resolution (matches utils_rxn.PROJECT_ROOT).
PROJECT_ROOT = os.environ.get(
    "CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)
_RESULTS = os.path.join(PROJECT_ROOT, "results_lambda_ablation",
                        "lambda_results.csv")
_DRFP_META = os.path.join(PROJECT_ROOT, "results_best_pipeline",
                          "drfp_ablation_meta.json")


def _load_best_lambda_prop():
    """
    Read BEST_LAMBDA_PROP from results_lambda_ablation/lambda_results.csv.

    Picks the λ whose DualANN R² is maximal (column ann_r2_mean).

    There is deliberately NO numeric fallback.  If the CSV is missing or unreadable the
    function returns ``(None, None)``: a made-up λ would silently run the whole B/G chain
    with the wrong property weight.  ``BEST_LAMBDA_PROP is None`` then makes every
    consumer fail loudly (``float(None)`` / ``require_best_lambda()``).  Import itself must
    not raise, because 201 stage 1 imports this module before stage 2 has produced the CSV.
    """
    if not os.path.isfile(_RESULTS):
        warnings.warn(
            f"[config] {_RESULTS} not found; BEST_LAMBDA_PROP=None.  "
            f"Run 201_ablation.py --stage 2 first.",
            stacklevel=2,
        )
        return None, None

    try:
        import csv
        with open(_RESULTS, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [
                r for r in reader
                if r.get("lambda") not in (None, "")
                and r.get("ann_r2_mean") not in (None, "")
            ]
        if not rows:
            warnings.warn(
                f"[config] {_RESULTS} has no parseable rows; BEST_LAMBDA_PROP=None.",
                stacklevel=2,
            )
            return None, None

        # Pick λ with the largest ann_r2_mean (DualANN is the primary model
        # in 201_ablation.py Stage 2; criterion matches line 530).
        best_row = max(rows, key=lambda r: float(r["ann_r2_mean"]))
        best_lam = float(best_row["lambda"])
        best_r2 = float(best_row["ann_r2_mean"])
        return best_lam, best_r2
    except Exception as e:
        warnings.warn(
            f"[config] failed to read {_RESULTS}: {e}; BEST_LAMBDA_PROP=None.",
            stacklevel=2,
        )
        return None, None


# Best λ for property-co-learning (auto-loaded from lambda_results.csv).
# (best_lambda, best_ann_r2); both are None when the sweep has not been run.
BEST_LAMBDA_PROP, BEST_LAMBDA_PROP_R2 = _load_best_lambda_prop()


def require_best_lambda() -> float:
    """Return BEST_LAMBDA_PROP or raise a clear error when the λ sweep has not been run."""
    if BEST_LAMBDA_PROP is None:
        raise RuntimeError(
            "BEST_LAMBDA_PROP is undefined: results_lambda_ablation/lambda_results.csv is "
            "missing or unreadable. Run `python src/data/201_ablation.py --stage 2` first.")
    return float(BEST_LAMBDA_PROP)


def _load_best_drfp_variant():
    """
    Read BEST_DRFP_VARIANT from results_best_pipeline/drfp_ablation_meta.json.
    Falls back to 'full' with a warning if the JSON is missing.
    """
    fallback = "full"
    if not os.path.isfile(_DRFP_META):
        warnings.warn(
            f"[config] {_DRFP_META} not found.  Using fallback "
            f"BEST_DRFP_VARIANT='{fallback}'.  Run 201_ablation.py to refresh.",
            stacklevel=2,
        )
        return fallback
    try:
        import json as _json
        with open(_DRFP_META, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)
        variant = meta.get("best_variant", fallback)
        if not isinstance(variant, str):
            raise ValueError(f"best_variant not a string: {variant!r}")
        return variant
    except Exception as e:
        warnings.warn(
            f"[config] failed to read {_DRFP_META}: {e}.  "
            f"Using fallback BEST_DRFP_VARIANT='{fallback}'.",
            stacklevel=2,
        )
        return fallback


# Latent dimension for PCL-AE (used by 05, 09, 10, 13).  This is a structural
# choice, not a tuned hyper-parameter, so it stays hard-coded.
BEST_LATENT_DIM = 128

# Training budget of every PCL-AE / label-free AE in the frozen-vs-retrain comparison.
# The frozen (cached, full-data) arm and the per-fold retrained arm must use the same
# value, otherwise Delta_protocol absorbs an optimisation-budget difference.
PCL_AE_EPOCHS = 150

# DRFP variant (auto-loaded from drfp_ablation_meta.json).
BEST_DRFP_VARIANT = _load_best_drfp_variant()


# ════════════════════════════════════════════════════════════════════════════════
# External I/O paths and rate limits (consumed by 102_smiles.py, 104b_run_xtb*.py)
# ════════════════════════════════════════════════════════════════════════════════

# Path to geckodriver.exe on Windows. Used by 102_smiles.py for Selenium-based
# PubChem fallback after Cactus. If file does not exist at runtime, 102 silently
# falls back to Cactus-only mode.
GECKODRIVER_PATH = r"C:\tools\geckodriver.exe"

# Cactus NCI-CADD name→SMILES resolver: minimum delay between HTTP requests.
# Official Cactus etiquette recommends ≥1 req/s to avoid temporary IP ban.
# Lower this only if you have a private API key / on-premise proxy.
CACTUS_REQUEST_DELAY_S = 1.0
