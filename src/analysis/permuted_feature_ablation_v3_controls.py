# -*- coding: utf-8 -*-
"""
permuted_feature_ablation_v3_controls.py
==========================================
**Task D v3 — Control ablations for the v2 verdict.**

Auditor concern (§3.4.1 reviewer #3): "If `temperature` and `time_log`
are also 'ambiguous' on CHO, then `sub_homo_eV`'s 'ambiguous' verdict is
the *normal* behaviour for a feature under CHO's COMO-style train-test
shift, not a special observation about `sub_homo_eV`."

This script runs the same v2 LOCO-permutation protocol on three
*control* features:
    1. ``temperature`` (numeric, condition-driven)
    2. ``time_log``    (numeric, condition-driven)
    3. ``delta_E_HL``  (chemically-derived, terminal-CHO comparator)

If the same "ambiguous on CHO, genuine on terminal" pattern holds for
``temperature`` and ``time_log``, the v2 verdict for ``sub_homo_eV`` is
**a fingerprint of the COMO-style CHO fold**, not of the feature
chemistry.  In that case, §3.4.1's interpretation of the v2 result on
``sub_homo_eV`` shifts from "CHO does not need this feature" to
"CHO's COMO split produces feature-indistinguishable predictions for
many features simultaneously — a sample-size / train-shift limitation".

If ``delta_E_HL`` shows a *different* pattern on CHO (e.g. genuine),
then ``sub_homo_eV``'s "ambiguous" verdict reflects the specific
chemistry of CHO's `sub_homo_eV` distribution (very narrow range
in the test fold, leading to almost zero information), not just the
COMO shift.

The numerical output of this script directly informs the §3.4.1
revision in paper_final_English.md.
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

PROJECT_ROOT = r"D:\machine-learning\CO2-cycloaddition"

# Reuse v2's loader pattern (same UTF-16 hack if needed)
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
    _ilu.spec_from_loader("xtb_shap_infra", loader=None, origin=_INFRA_PATH)
)
_infra_mod.__dict__.update(_infra_globals)
sys.modules["xtb_shap_infra"] = _infra_mod
load_xtb_data = _infra_mod.load_xtb_data
SUBSTRATE_ORDER = _infra_mod.SUBSTRATE_ORDER
logger = _infra_mod.logger

OUT_DIR = os.path.join(PROJECT_ROOT, "results_permuted_ablation_v3_controls")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_PERMUTATIONS = 100
# Match v2 XGB hyper-parameters EXACTLY so that v3 verdict is
# apples-to-apples with v2's verdict on `sub_homo_eV`.
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

# Three control features, all under the same XTB-only 32-D space as v2.
TARGETS = ["temperature", "time_log", "delta_E_HL"]


def train_eval_one(X_tr, y_tr, X_te, y_te, feat_idx, random_state):
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
    logger.info("=== Task D v3: control ablations on temperature/time_log/delta_E_HL ===")
    logger.info("XGB params matched to v2 (n_estimators=400, max_depth=6)")

    df, X, names, y = load_xtb_data(include_mech=True)
    logger.info("X shape: %s, n_features=%d", X.shape, len(names))

    all_rows = []
    summary = {
        "task": "feature_ablation_controls_v3",
        "random_state": RANDOM_STATE,
        "n_permutations": N_PERMUTATIONS,
        "xgb_params": XGB_PARAMS,
        "metadata": {
            "n_rows": int(len(df)),
            "n_features": int(len(names)),
            "started_utc": t0.isoformat(),
        },
        "targets": {},  # ← initialize
    }

    for target in TARGETS:
        if target not in names:
            logger.warning("Target %s not in feature names", target)
            continue
        feat_idx = names.index(target)
        logger.info("\n=== Target feature: %s (idx=%d) ===", target, feat_idx)
        summary["targets"][target] = {}

        # Diagnostic: target global stats (column is in feature matrix,
        # not raw df; use X[:, feat_idx])
        full_col = X[:, feat_idx]
        logger.info(
            "%s global stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            target, full_col.mean(), full_col.std(), full_col.min(), full_col.max(),
        )

        for sub in SUBSTRATE_ORDER:
            mask = (df["reactant_name"].values == sub)
            n_test = int(mask.sum())
            if n_test < 10:
                continue

            X_tr, y_tr = X[~mask], y[~mask]
            X_te, y_te = X[mask], y[mask]

            logger.info("  --- %s (n_train=%d, n_test=%d) ---",
                        sub, len(y_tr), n_test)

            # 1) Original SHAP for this feature on the held-out fold
            original_shap, orig_r2, _ = train_eval_one(
                X_tr, y_tr, X_te, y_te, feat_idx, random_state=RANDOM_STATE)
            logger.info("  original mean SHAP for %s = %+.4f (LOSO R²=%.3f)",
                        target, original_shap, orig_r2)

            # 2) Permutation distribution
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

            perm_mean = float(perm_means.mean())
            perm_std = float(perm_means.std(ddof=1))
            ci_low = float(np.percentile(perm_means, 2.5))
            ci_high = float(np.percentile(perm_means, 97.5))
            p_value = float((np.abs(perm_means) >= abs(original_shap)).mean())
            original_in_ci = bool(ci_low <= original_shap <= ci_high)
            z_score = (original_shap - perm_mean) / (perm_std + 1e-12)
            verdict = "genuine" if not original_in_ci and p_value < 0.05 else "ambiguous"

            logger.info("  perm mean=%+.4f std=%.4f 95CI=[%+.4f, %+.4f]",
                        perm_mean, perm_std, ci_low, ci_high)
            logger.info("  original %+.4f z=%+.2f p=%.4f in_CI=%s verdict=%s",
                        original_shap, z_score, p_value, original_in_ci, verdict)

            all_rows.append({
                "feature": target,
                "substrate": sub,
                "original_shap": original_shap,
                "permuted_mean": perm_mean,
                "permuted_std": perm_std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "z_score": z_score,
                "original_in_95ci": original_in_ci,
                "verdict": verdict,
                "n_test": n_test,
                "n_train": int(len(y_tr)),
                "orig_loso_r2": orig_r2,
            })
            summary["targets"][target][sub] = {
                "original_shap": original_shap,
                "permuted_mean": perm_mean,
                "permuted_std": perm_std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "z_score": z_score,
                "original_in_95ci": original_in_ci,
                "verdict": verdict,
                "n_test": n_test,
                "n_train": int(len(y_tr)),
                "orig_loso_r2": orig_r2,
            }

    # ---- Save ----
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["rows"] = all_rows
    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUT_DIR, "permuted_feature_ablation_v3_controls.csv"),
        index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "permuted_feature_ablation_v3_controls_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown summary
    md = ["# Permuted feature ablation v3 — controls (Task D v3)",
          "",
          f"Generated: {datetime.now(timezone.utc).isoformat()}",
          f"Random state: {RANDOM_STATE}, permutations per fold: {N_PERMUTATIONS}",
          "XGB params match v2 (n_estimators=400, max_depth=6, lr=0.05,",
          "subsample=colsample_bytree=0.8, reg_lambda=1.0).",
          "",
          "| feature | substrate | n_test | original | permuted mean ± std | 95% CI | p | verdict |",
          "|---|---|---:|---:|---:|---|---:|---|"]
    for r in all_rows:
        md.append(
            f"| {r['feature']} | {r['substrate']} | {r['n_test']} "
            f"| {r['original_shap']:+.3f} "
            f"| {r['permuted_mean']:+.3f} ± {r['permuted_std']:.3f} "
            f"| [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
            f"| {r['p_value']:.3f} "
            f"| **{r['verdict']}** |"
        )
    md.append("")
    md.append("## Interpretation guideline")
    md.append("")
    md.append("- If `temperature` and `time_log` are also `ambiguous` on CHO while genuine on the 4 terminal "
              "substrates, the v2 verdict on `sub_homo_eV` is a **fingerprint of the COMO-style train-test shift** "
              "on CHO, not a feature-specific chemistry claim.")
    md.append("- If `delta_E_HL` shows a *different* pattern on CHO (e.g. genuine), then `sub_homo_eV`'s "
              "`ambiguous` verdict reflects the narrow test-fold `sub_homo_eV` range rather than train-set "
              "shift alone.")
    md.append("- The cross-feature verdict matrix is the single most important control for §3.4.1's "
              "interpretation in `paper_final_English.md`.")
    with open(os.path.join(OUT_DIR, "permuted_feature_ablation_v3_controls_summary.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    logger.info("\nSaved to %s", OUT_DIR)
    logger.info("Total wall time: %.1f s",
                (datetime.now(timezone.utc) - t0).total_seconds())


if __name__ == "__main__":
    main()
