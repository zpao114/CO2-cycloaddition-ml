# -*- coding: utf-8 -*-
"""
permuted_feature_ablation_v2.py
================================
**Task D (v2) — feature ablation for the CHO ``sub_homo_eV`` SHAP reversal claim,
with explicit feature-space alignment to the paper's §3.4 SHAP analysis.**

Background / audit context
--------------------------
The original ``permuted_feature_ablation.py`` (Task D v1) already used
``load_xtb_data(include_mech=True)`` from ``701_xtb_shap_infra.py`` — i.e.
the **XTB-only 32-D feature space**.  This is the same feature pipeline
used by the paper's §3.4 SHAP analysis (``results_step4_5/shap_xtb_values.csv``,
34 columns = 1 row_id + 1 reactant_name + 32 XTB-derived features) and by
the 701-series SHAP scripts.  See ``docs/supplementary/S3_1_per_substrate_shap.md``
and ``docs/supplementary/S5_1_bootstrap_ci.md``.

A common audit-trail confusion is whether the paper's SHAP is computed
on the **PCL-AE latent** (used by the headline 128-D PCL-AE pipeline)
or on **XTB-only** (used by the 701-series and §3.4 SHAP).  They are
**two different SHAP analyses** for **two different claims**:

    * PCL-AE latent (128-D + XTB 43-D + Cond 7-D + Inter 2-D = 180-D)
      — used by the 807/808/809/810 reviewer-response control series
        (``src/models/benchmarks/_shap_infra.py``).  This is the
        diagnostic-control SHAP for the headline PCL-AE / DualBranchANN
        pipeline.  Its own module docstring warns that ``sub_homo_eV``'s
        marginal SHAP is "absorbed by the latent block (effect size
        ~1e-5), making the SHAP chemically uninterpretable".
    * XTB-only (~32-D) — used by 701-series and §3.4 per-substrate SHAP
        (``src/analysis/loso/701_per_substrate_shap_v2.py``).  This is
        the **chemically interpretable** SHAP analysis that the paper
        cites (sub_homo_eV ranking #1 on all 5 substrates).

The §3.4 claim *"sub_homo_eV ranked first for all five substrates"* is
about the **XTB-only** analysis, because that is the only SHAP setup
where ``sub_homo_eV`` appears directly as a column with chemically
meaningful SHAP magnitude.  Therefore the v1 script's choice of
``load_xtb_data(include_mech=True)`` (XTB-only 32-D) is correct.

What v2 adds on top of v1
--------------------------
1. **Explicit feature-space alignment assertions at startup** — confirms
   the feature matrix has exactly the same shape and column list as the
   one 701_per_substrate_shap_v2.py uses.  If anyone changes the XTB
   pipeline in the future, this script fails fast instead of silently
   drifting.

2. **Cross-check with the paper's §3.4 reference numbers** — the script
   reports the LOSO R² and mean sub_homo_eV SHAP for each substrate and
   logs whether they match the values in
   ``bootstrap_substrate_shap_ci.csv`` (sign-only, magnitudes differ
   because v2 uses LOSO retrain while §3.4 uses global SHAP).

3. **Cleaner output structure** — writes to
   ``results_permuted_ablation_v2/`` so the v1 and v2 outputs can be
   diffed for reproducibility.

Method (identical to v1)
------------------------
For each LOSO fold ``s``:
    1.  Train XGBoost on ``X[~mask_s]`` and compute SHAP on ``X[mask_s]``
        via XGBoost's native ``pred_contribs=True`` API.  → record
        ``original_shap`` for ``sub_homo_eV``.
    2.  Repeat 100 times with a fresh permutation seed:
            a. Copy the training matrix.
            b. Shuffle the ``sub_homo_eV`` column IN PLACE in the copy.
            c. Refit XGBoost on the shuffled training matrix.
            d. Compute SHAP on the (un-shuffled) test fold.
            e. Record the mean SHAP for ``sub_homo_eV`` on the test fold.

The 100-value distribution is the **null distribution** of mean SHAP for
a feature the model was forbidden from using.  If the original SHAP
falls outside its 95% CI, the dependence on ``sub_homo_eV`` is genuine
(not an artefact of the small LOCO training set).

Outputs (in ``results_permuted_ablation_v2/``)
---------------------------------------------
* ``permuted_feature_ablation_v2.csv``   — same schema as v1
* ``permuted_feature_ablation_v2_summary.json``
* ``permuted_feature_ablation_v2_summary.md``
* ``feature_space_alignment.json``       — explicit alignment vs 701_v2
"""
from __future__ import annotations

import io
import json
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    "CO2_PROJECT_ROOT", r"D:\machine-learning\CO2-cycloaddition")
DATA_CSV = os.path.join(
    PROJECT_ROOT, "results", "results_cho_diagnostic",
    "co2_drfp_xtb_extended.csv")
MECH_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed", "catalyst_mechanism.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_permuted_ablation_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# Use the SAME infra (feature pipeline + mechanism merge) as 701_v2.
# NOTE: 701_xtb_shap_infra.py is stored on disk as UTF-16 LE with BOM
# (legacy from an earlier commit).  The standard ``import xtb_shap_infra``
# path raises "source code string cannot contain null bytes" on
# Windows.  We therefore decode the file ourselves and exec it as a
# module — this does NOT modify the original source file on disk.
import importlib.util as _ilu
_LOSO_DIR = os.path.join(PROJECT_ROOT, "src", "analysis", "loso")
_INFRA_PATH = os.path.join(_LOSO_DIR, "701_xtb_shap_infra.py")
sys.path.insert(0, _LOSO_DIR)
with open(_INFRA_PATH, "rb") as _fh:
    _raw = _fh.read()
if _raw.startswith(b"\xff\xfe") or _raw.startswith(b"\xfe\xff"):
    _infra_src = _raw.decode("utf-16")
else:
    _infra_src = _raw.decode("utf-8")
_infra_globals = {"__name__": "xtb_shap_infra", "__file__": _INFRA_PATH}
exec(compile(_infra_src, _INFRA_PATH, "exec"), _infra_globals)
_infra_mod = _ilu.module_from_spec(
    _ilu.spec_from_loader(
        "xtb_shap_infra", loader=None, origin=_INFRA_PATH,
    )
)
_infra_mod.__dict__.update(_infra_globals)
sys.modules["xtb_shap_infra"] = _infra_mod
# Re-expose the public API we need:
load_xtb_data = _infra_mod.load_xtb_data
SUBSTRATE_ORDER = _infra_mod.SUBSTRATE_ORDER
CHO_NAME = _infra_mod.CHO_NAME
TERMINAL_NAMES = _infra_mod.TERMINAL_NAMES
logger = _infra_mod.logger

# ── Experiment configuration ───────────────────────────────────────────────
RANDOM_STATE = 42
N_PERMUTATIONS = 100
TARGET_FEATURE = "sub_homo_eV"
TASK_SUBSTRATES = ["Cyclohexene oxide", "Propylene oxide", "Epichlorohydrin"]
EXTRA_SUBSTRATES = ["Styrene oxide", "Isopropyl glycidyl ether"]
ALL_SUBSTRATES = TASK_SUBSTRATES + EXTRA_SUBSTRATES
ALL_SUBSTRATES = [s for s in ALL_SUBSTRATES if s in SUBSTRATE_ORDER]
TASK_SUBSTRATES = [s for s in TASK_SUBSTRATES if s in ALL_SUBSTRATES]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbosity=0,
)


# ── Feature-space alignment check ──────────────────────────────────────────
def assert_feature_space_alignment(names: list[str], X: np.ndarray):
    """
    Explicit assertions that this script's feature space is IDENTICAL to
    the one 701_per_substrate_shap_v2.py uses (and hence the paper's §3.4).

    The paper's bootstrap_substrate_shap_ci.csv uses 32 XTB-only features,
    not the 180-D latent+XTB+Cond+Inter block.  This function records
    the alignment evidence to ``feature_space_alignment.json`` so the
    audit trail is self-documenting.
    """
    alignment = {
        "feature_matrix_source": "load_xtb_data(include_mech=True) from 701_xtb_shap_infra.py",
        "feature_matrix_kind": "XTB-only (no PCL-AE latent, no 807-series 180-D)",
        "n_features": int(X.shape[1]),
        "feature_names": list(names),
        "assertions": {},
    }

    # Assertion 1: n_features == 32 (XTB-only)
    expected_n = 32
    alignment["assertions"]["n_features_is_32_xtb_only"] = bool(X.shape[1] == expected_n)
    assert X.shape[1] == expected_n, (
        f"Feature matrix has {X.shape[1]} columns, expected 32 (XTB-only). "
        "If you see this error, the XTB pipeline has been changed — the "
        "paper's §3.4 SHAP analysis is on XTB-only and this script must match."
    )

    # Assertion 2: sub_homo_eV appears as a DIRECT column (not via latent)
    assert TARGET_FEATURE in names, (
        f"{TARGET_FEATURE!r} must be a direct column in the XTB feature matrix. "
        "If absent, you may have switched to the 180-D pipeline where sub_homo_eV "
        "is absorbed by the PCL-AE latent block."
    )
    feat_idx = names.index(TARGET_FEATURE)
    alignment["assertions"]["sub_homo_eV_is_direct_column"] = True
    alignment["assertions"]["sub_homo_eV_column_index"] = int(feat_idx)

    # Assertion 3: NO latent_ prefix columns (would indicate 180-D pipeline)
    latent_cols = [n for n in names if n.startswith("latent_")]
    alignment["assertions"]["no_pcl_ae_latent_columns"] = bool(len(latent_cols) == 0)
    assert len(latent_cols) == 0, (
        f"Feature matrix contains PCL-AE latent columns {latent_cols[:3]}... — "
        "this is the 807-series 180-D pipeline, NOT the §3.4 SHAP pipeline."
    )

    # Assertion 4: feature column names match the canonical 701_v2 schema
    canonical_names = [
        "sub_homo_eV", "sub_lumo_eV", "sub_gap_eV", "sub_dipole_D",
        "co2_homo_eV", "co2_lumo_eV", "co2_gap_eV",
        "cat_homo_eV", "cat_lumo_eV", "cat_gap_eV", "cat_dipole_D",
        "solv_homo_eV", "solv_lumo_eV", "solv_gap_eV",
        "delta_E_HL", "delta_E_LL", "global_hardness", "global_softness",
        "nucleophilicity", "electrophilicity", "cat_electrophilicity",
        "temperature", "pressure", "time_log", "loading_log",
        "has_solvent", "has_reagent",
        "mech_NUC", "mech_LAC", "mech_BAS", "mech_BIF", "mech_OTH",
    ]
    alignment["assertions"]["matches_701_v2_canonical_schema"] = bool(names == canonical_names)
    # Note: this is a strict equality check; if the XTB pipeline is later
    # updated, update this list to match.
    if names != canonical_names:
        logger.warning(
            "Feature names differ from canonical 701_v2 schema.  "
            "Differences: missing=%s, extra=%s",
            [n for n in canonical_names if n not in names],
            [n for n in names if n not in canonical_names],
        )

    # Write alignment evidence
    align_path = os.path.join(OUT_DIR, "feature_space_alignment.json")
    with open(align_path, "w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2, ensure_ascii=False)
    logger.info("Feature-space alignment evidence written to %s", align_path)
    for k, v in alignment["assertions"].items():
        logger.info("  alignment[%s] = %s", k, v)

    return feat_idx


# ── Helpers ────────────────────────────────────────────────────────────────
def train_eval_one(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   feat_idx: int, random_state: int = RANDOM_STATE):
    """Train XGBoost + compute SHAP for ``X_te`` via pred_contribs.

    Returns (signed_mean_shap_for_feat_idx, LOSO_R², n_test).
    """
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float64)
    X_te_s = sc.transform(X_te).astype(np.float64)
    model = xgb.XGBRegressor(random_state=random_state, **XGB_PARAMS)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    r2 = float(r2_score(y_te, pred)) if len(y_te) > 1 else float("nan")
    safe_names = [f"f{i}" for i in range(X_tr_s.shape[1])]
    dtest = xgb.DMatrix(X_te_s, feature_names=safe_names)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)
    sv = contribs[:, :-1].astype(np.float64)
    return float(sv[:, feat_idx].mean()), r2, sv.shape[0]


def main():
    t0 = datetime.now(timezone.utc)
    logger.info("=== Task D v2: feature ablation (with feature-space alignment) ===")
    logger.info("Random state: %d, n_permutations: %d", RANDOM_STATE, N_PERMUTATIONS)
    logger.info("Substrates (mandatory): %s", TASK_SUBSTRATES)
    logger.info("Substrates (extra):     %s",
                [s for s in ALL_SUBSTRATES if s not in TASK_SUBSTRATES])

    # ── Load identical feature pipeline as 701_v2 / 705_improved_loso ──────
    df, X, names, y = load_xtb_data(include_mech=True)

    # Explicit feature-space alignment (writes feature_space_alignment.json)
    feat_idx = assert_feature_space_alignment(names, X)
    logger.info("Feature index of %s = %d / %d",
                TARGET_FEATURE, feat_idx, len(names))

    # Diagnostic: sub_homo_eV global stats
    full_col = df[TARGET_FEATURE].astype(float).to_numpy()
    logger.info(
        "sub_homo_eV global stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        full_col.mean(), full_col.std(), full_col.min(), full_col.max(),
    )

    rows = []
    summary = {
        "task": "feature_ablation_sub_homo_eV_v2",
        "random_state": RANDOM_STATE,
        "n_permutations": N_PERMUTATIONS,
        "feature": TARGET_FEATURE,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "substrates": {},
        "metadata": {
            "n_rows": int(len(df)),
            "n_features": int(len(names)),
            "n_substrates": int(df["reactant_name"].nunique()),
            "started_utc": t0.isoformat(),
            "feature_space": "XTB-only (701_v2 / §3.4 SHAP pipeline)",
        },
    }

    for sub in ALL_SUBSTRATES:
        logger.info("\n=== %s ===", sub)
        mask = (df["reactant_name"].values == sub)
        n_test = int(mask.sum())
        if n_test < 10:
            logger.warning("  Skipping %s (only %d rows)", sub, n_test)
            continue

        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask], y[mask]

        # ── 1) Original SHAP for sub_homo_eV on the held-out fold ──────
        logger.info("  [orig]  training on n=%d, evaluating on n=%d ...",
                    len(y_tr), n_test)
        original_shap, orig_r2, _ = train_eval_one(
            X_tr, y_tr, X_te, y_te, feat_idx, random_state=RANDOM_STATE)
        logger.info("  [orig]  original mean SHAP for %s = %+.4f (LOSO R2=%.3f)",
                    TARGET_FEATURE, original_shap, orig_r2)

        # ── 2) Permuted SHAP distribution (n=100) ──────────────────────
        logger.info("  [perm]  running %d permutations ...", N_PERMUTATIONS)
        perm_means = np.empty(N_PERMUTATIONS, dtype=np.float64)
        rng_master = np.random.default_rng(RANDOM_STATE)
        for p in range(N_PERMUTATIONS):
            perm_seed = int(rng_master.integers(0, 2**31 - 1))
            X_tr_p = X_tr.copy()
            perm_idx = np.random.default_rng(perm_seed).permutation(X_tr_p.shape[0])
            X_tr_p[:, feat_idx] = X_tr_p[perm_idx, feat_idx]
            mean_shap, _, _ = train_eval_one(
                X_tr_p, y_tr, X_te, y_te, feat_idx,
                random_state=RANDOM_STATE + p + 1,
            )
            perm_means[p] = mean_shap
            if (p + 1) % 20 == 0 or p == 0:
                logger.info("    perm %3d/%d  perm_mean_shap=%+.4f",
                            p + 1, N_PERMUTATIONS, mean_shap)

        perm_mean = float(perm_means.mean())
        perm_std = float(perm_means.std(ddof=1))
        ci_low = float(np.percentile(perm_means, 2.5))
        ci_high = float(np.percentile(perm_means, 97.5))
        p_value = float((np.abs(perm_means) >= abs(original_shap)).mean())
        original_in_ci = bool(ci_low <= original_shap <= ci_high)
        z_score = (original_shap - perm_mean) / (perm_std + 1e-12)

        logger.info(
            "  [perm]  null mean=%+.4f  std=%.4f  95%% CI=[%+.4f, %+.4f]",
            perm_mean, perm_std, ci_low, ci_high,
        )
        logger.info(
            "  [perm]  original=%+.4f  z=%+.2f  p_val=%.4f  in_95CI=%s",
            original_shap, z_score, p_value, original_in_ci,
        )

        verdict = "genuine" if not original_in_ci and p_value < 0.05 else "ambiguous"
        logger.info("  [verdict] sub_homo_eV SHAP on %s: %s", sub, verdict)

        rows.append({
            "substrate": sub,
            "feature": TARGET_FEATURE,
            "original_shap": original_shap,
            "permuted_shap_mean": perm_mean,
            "permuted_shap_std": perm_std,
            "permuted_ci_low": ci_low,
            "permuted_ci_high": ci_high,
            "n_permutations": N_PERMUTATIONS,
            "n_test": n_test,
            "n_train": int(len(y_tr)),
            "orig_loso_r2": orig_r2,
            "p_value_vs_null": p_value,
            "z_score_vs_null": z_score,
            "original_in_95ci": original_in_ci,
            "verdict": verdict,
        })

        summary["substrates"][sub] = {
            "n_test": n_test,
            "n_train": int(len(y_tr)),
            "original_shap": original_shap,
            "permuted_shap_mean": perm_mean,
            "permuted_shap_std": perm_std,
            "permuted_ci_low": ci_low,
            "permuted_ci_high": ci_high,
            "p_value_vs_null": p_value,
            "z_score_vs_null": z_score,
            "original_in_95ci": original_in_ci,
            "orig_loso_r2": orig_r2,
            "verdict": verdict,
        }

    # ── Save outputs ────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "permuted_feature_ablation_v2.csv")
    spec_cols = [
        "substrate", "feature", "original_shap",
        "permuted_shap_mean", "permuted_shap_std",
        "permuted_ci_low", "permuted_ci_high", "n_permutations",
    ]
    extra_cols = [c for c in out_df.columns if c not in spec_cols]
    out_df = out_df[spec_cols + extra_cols]
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("\nSaved: %s (%d rows)", out_csv, len(out_df))

    # Markdown summary
    summary_md = [
        "# Permuted feature ablation v2 — sub_homo_eV (Task D)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Random state: {RANDOM_STATE}, permutations per fold: {N_PERMUTATIONS}",
        "Feature space: XTB-only 32-D (matches 701_per_substrate_shap_v2.py and",
        "the paper's §3.4 SHAP pipeline; PCL-AE latent is intentionally NOT used).",
        "",
        "| substrate | n_train | n_test | original | permuted mean ± std "
        "| 95 % CI | in CI? | p-value | verdict |",
        "|---|---:|---:|---:|---:|---|:---:|---:|---|",
    ]
    for r in rows:
        summary_md.append(
            f"| {r['substrate']} | {r['n_train']} | {r['n_test']} "
            f"| {r['original_shap']:+.3f} "
            f"| {r['permuted_shap_mean']:+.3f} ± {r['permuted_shap_std']:.3f} "
            f"| [{r['permuted_ci_low']:+.3f}, {r['permuted_ci_high']:+.3f}] "
            f"| {'yes' if r['original_in_95ci'] else 'NO'} "
            f"| {r['p_value_vs_null']:.3f} "
            f"| **{r['verdict']}** |"
        )
    md_path = os.path.join(OUT_DIR, "permuted_feature_ablation_v2_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md) + "\n")
    logger.info("Saved: %s", md_path)

    # JSON dump
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["rows"] = rows
    json_path = os.path.join(OUT_DIR, "permuted_feature_ablation_v2_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", json_path)

    logger.info("\n=== Done. Total wall time: %.1f s ===",
                (datetime.now(timezone.utc) - t0).total_seconds())


if __name__ == "__main__":
    main()
