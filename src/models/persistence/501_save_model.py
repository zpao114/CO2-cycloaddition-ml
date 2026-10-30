# -*- coding: utf-8 -*-
"""
501_save_model.py
==================

Persist the stage-2 best pipeline (PCL-AE-128 DRFP reducer + DualBranchANN)
so that downstream scripts (e.g. 705_virtual_screening.py) can score
new reactions without re-training.

The property-co-learning weight lambda_prop is exposed as a CLI argument.
The 9-point ablation (lambda ∈ {0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0})
peaks at lambda = 10.0 (R^2 = 0.394 ± 0.060), the upper bound of the swept
range, suggesting values > 10 may yield further gains (future work).

Outputs (all under results_best_pipeline/artifacts/):
  - drfp_scaler.joblib            : StandardScaler fit on raw DRFP (2048 D)
  - pcl_ae_encoder.pt             : PCL-AE-128 encoder state_dict + meta
  - xtb_cond_inter_scaler.joblib  : StandardScaler fit on X_xtb_cond_inter
  - dual_branch_ann.pt            : DualBranchANN state_dict (drfp_in=128, other_in=52)
  - feature_meta.json             : column names, dims, dtype, splits
  - training_metrics.json         : R^2 / MAE / RMSE from 5-fold CV (sanity)
  - save_best_model_report.txt    : human-readable summary

Usage:
  python 501_save_model.py                  # use default lambda = 10.0
  python 501_save_model.py --lambda 0.5     # reproduce baseline (unsupervised)
  python 501_save_model.py --lambda 5.0     # reproduce v1 paper default
  python 501_save_model.py --lambda 10.0    # explicit optimal value (file-optimum)
"""

import os
import argparse
import io
import json
import os
import sys
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from eval_protocol.models import fit_predict_dual  # noqa: E402  (inner-val early stopping)
from utils_rxn import read_drfp, get_best_drfp_variant, XTB_COLS
from utils_features import require_cond_cols

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'data/processed/co2_drfp_xtb_extended.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_best_pipeline")
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults chosen by 201_ablation.py (Section 3.1.1 of the paper).
# BEST_LAMBDA_PROP is the single source of truth in config.py.
# To re-calibrate, run: python 201_ablation.py
# (it will overwrite config.py automatically).
from config import require_best_lambda  # noqa: E402  (raises if the λ sweep has not been run)
DEFAULT_LAMBDA_PROP = require_best_lambda()
LATENT_DIM = 128
EPOCHS_PCL = 150
EPOCHS_ANN = 200


# ----------------------------------------------------------------------
# Model architectures (mirrored from 08_benchmark.py)
# ----------------------------------------------------------------------

from eval_protocol.nets import PropertyCoLearningAE, train_pcl_ae_model  # noqa: E402,F401
from eval_protocol.models import DualBranchANN  # noqa: E402,F401


def train_pcl_ae(X, y, latent_dim=LATENT_DIM, epochs=EPOCHS_PCL,
                 batch_size=128, lr=1e-3, lambda_prop=DEFAULT_LAMBDA_PROP,
                 pos_weight=10.0):
    """Train PCL-AE on standardized DRFP (thin wrapper over eval_protocol.nets)."""
    print(f"[2/4] Training PCL-AE (lambda={lambda_prop}, latent_dim={latent_dim}, epochs={epochs}) ...")
    return train_pcl_ae_model(X, y, latent_dim, lambda_prop, epochs, batch_size, lr,
                              pos_weight, device=DEVICE)





# ----------------------------------------------------------------------
# Feature assembly (kept identical to 08_benchmark.load_data)
# ----------------------------------------------------------------------
# DRFP column is dynamically determined by the ablation experiment
# (see utils_rxn.get_best_drfp_variant())
#
# Canonical feature-column lists are imported from utils_rxn.XTB_COLS and
# utils_features.COND_COLS to keep training-time and inference-time input
# dimensions identical across the whole pipeline.


def parse_drfp_col(series):
    arrs = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


def load_data():
    """Load data, dynamically selecting the best DRFP variant from ablation meta."""
    meta = get_best_drfp_variant()
    best_col = meta['best_drfp_col']
    best_var = meta['best_variant']
    print(f"\n[Step 0] Loading {os.path.basename(DATA_EXTENDED)} ...")
    print(f"  DRFP variant: {best_var} ({best_col})")
    df = pd.read_csv(DATA_EXTENDED, encoding="utf-8-sig")
    df = df[df["extraction_status"] == "valid"].copy()
    df = df.dropna(subset=["yield (%)"])
    df = df[df["yield (%)"] > 0].reset_index(drop=True)
    print(f"  Valid samples: {len(df)}")

    X_drfp = parse_drfp_col(df[best_col]).astype(np.float32)
    print(f"  DRFP: {X_drfp.shape[1]}D")

    xtb_cols = [c for c in XTB_COLS if c in df.columns]
    X_xtb = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)
    print(f"  XTB: {X_xtb.shape[1]}D ({len(xtb_cols)} cols)")

    # Real column names in fixed order: temperature, pressure, time, loadings.
    # require_cond_cols raises if T/P/t is missing instead of silently shrinking Cond.
    cond_cols = require_cond_cols(df, include_loading=True)
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)
    print(f"  Cond: {X_cond.shape[1]}D ({cond_cols})")

    T = X_cond[:, 0:1]
    P = X_cond[:, 1:2]
    inter_parts = []
    if "activation_proxy" in xtb_cols:
        i = xtb_cols.index("activation_proxy")
        inter_parts.append(T * X_xtb[:, i:i + 1])
    if "total_polarity_index" in xtb_cols:
        i = xtb_cols.index("total_polarity_index")
        inter_parts.append(P * X_xtb[:, i:i + 1])
    X_inter = np.concatenate(inter_parts, axis=1).astype(np.float32) if inter_parts \
        else np.zeros((len(df), 0), np.float32)
    print(f"  Inter: {X_inter.shape[1]}D")

    y = df["yield (%)"].values.astype(np.float32) / 100.0
    return X_drfp, X_xtb, X_cond, X_inter, y, df, xtb_cols, cond_cols, best_var, best_col


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------



def train_dual_branch(Xd, Xo, y, hidden=128, epochs=EPOCHS_ANN,
                      batch_size=32, lr=5e-4, weight_decay=1e-3):
    print(f"[4/4] Training final DualBranchANN on full data (epochs={epochs}) ...")
    Xd_t = torch.tensor(Xd, dtype=torch.float32).to(DEVICE)
    Xo_t = torch.tensor(Xo, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    model = DualBranchANN(Xd.shape[1], Xo.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(Xd_t, Xo_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xd, xo, yy in dl:
            opt.zero_grad()
            nn.MSELoss()(model(xd, xo), yy).backward()
            opt.step()
    model.eval()
    return model


PERSIST_PROTOCOL = "pretrain_frozen"
PERSIST_PROTOCOL_NOTE = (
    "Encoder (scaler + PCL-AE) was fitted once on the whole pool and reused across folds; these CV numbers therefore carry the pretrain-frozen protocol inflation and are a sanity check of the exported artifact only. Honest performance: retrain-per-fold (eval_protocol)."
)


def eval_dual_branch_5fold(X_drfp, X_other, y, n_folds=5, hidden=128):
    """5-fold CV with early stopping (R^2-based)."""
    print(f"[3/4] 5-fold CV sanity (DualBranchANN on PCL-AE features) ...")
    fold_records = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(kf.split(X_drfp)):
        sd = StandardScaler(); so = StandardScaler()
        Xd_tr = sd.fit_transform(X_drfp[tr])
        Xd_te = sd.transform(X_drfp[te])
        Xo_tr = so.fit_transform(X_other[tr])
        Xo_te = so.transform(X_other[te])

        best_pred = np.clip(fit_predict_dual(
            Xd_tr, Xo_tr, y[tr], Xd_te, Xo_te, seed=int(fold),
            hidden=hidden, lr=5e-4, weight_decay=1e-3, batch_size=32,
            max_epochs=300, patience=40, lr_patience=20, device=str(DEVICE)), 0, 1)
        best_r2 = r2_score(y[te], best_pred)
        try:
            pr = float(pearsonr(y[te], best_pred)[0])
        except Exception:
            pr = np.nan
        fold_records.append({
            "fold": fold,
            "r2": float(best_r2),
            "mae": float(mean_absolute_error(y[te], best_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[te], best_pred))),
            "pearson": pr,
        })
    fm = pd.DataFrame(fold_records)
    return {
        "protocol": PERSIST_PROTOCOL,
        "protocol_note": PERSIST_PROTOCOL_NOTE,
        "r2_mean": float(fm["r2"].mean()),
        "r2_std": float(fm["r2"].std()),
        "mae_mean": float(fm["mae"].mean()),
        "rmse_mean": float(fm["rmse"].mean()),
        "pearson_mean": float(fm["pearson"].mean()),
        "per_fold": fold_records,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambda", dest="lambda_prop", type=float,
                   default=DEFAULT_LAMBDA_PROP,
                   help=f"PCL-AE property-co-learning weight (default {DEFAULT_LAMBDA_PROP})")
    p.add_argument("--latent-dim", dest="latent_dim", type=int,
                   default=LATENT_DIM,
                   help=f"Latent dimension for PCL-AE (default {LATENT_DIM})")
    return p.parse_args()


def main():
    args = parse_args()
    lambda_prop = args.lambda_prop
    latent_dim = args.latent_dim
    t0 = time.time()
    print("=" * 72)
    print(f"  Save Best Model  (PCL-AE-{latent_dim} DRFP + DualBranchANN, lambda = {lambda_prop})")
    print("=" * 72)
    print(f"  Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")

    X_drfp, X_xtb, X_cond, X_inter, y, df, xtb_cols, cond_cols, best_var, best_col = load_data()
    X_xtb_cond_inter = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)
    print(f"  Combined X_xtb_cond_inter: {X_xtb_cond_inter.shape}")

    print(f"\n[1/4] Fitting DRFP StandardScaler + training PCL-AE-{latent_dim} encoder ...")
    drfp_scaler = StandardScaler()
    Xd_s = drfp_scaler.fit_transform(X_drfp).astype(np.float32)

    pcl_model = train_pcl_ae(Xd_s, y, latent_dim=latent_dim, lambda_prop=lambda_prop)
    with torch.no_grad():
        Xd_reduced = pcl_model.encode(torch.FloatTensor(Xd_s).to(DEVICE)).cpu().numpy().astype(np.float32)
    print(f"  PCL-AE-{latent_dim} reduced DRFP: {Xd_reduced.shape}")

    print("\n[2/4] Fitting X_xtb_cond_inter StandardScaler ...")
    other_scaler = StandardScaler()
    Xo_s = other_scaler.fit_transform(X_xtb_cond_inter).astype(np.float32)

    metrics = eval_dual_branch_5fold(Xd_reduced, Xo_s, y, n_folds=5)
    print(f"  CV R^2 = {metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}  [protocol={metrics['protocol']}: sanity only, NOT an honest estimate]")
    print(f"  CV MAE = {metrics['mae_mean']:.4f}")
    print(f"  CV RMSE = {metrics['rmse_mean']:.4f}")

    final_ann = train_dual_branch(Xd_reduced, Xo_s, y, hidden=128, epochs=EPOCHS_ANN)

    print("\n[Save] Persisting artifacts ...")
    joblib.dump(drfp_scaler, os.path.join(ARTIFACT_DIR, "drfp_scaler.joblib"))
    torch.save({
        "state_dict": pcl_model.state_dict(),
        "input_dim": pcl_model.input_dim,
        "latent_dim": pcl_model.latent_dim,
        "lambda_prop": lambda_prop,
    }, os.path.join(ARTIFACT_DIR, "pcl_ae_encoder.pt"))
    joblib.dump(other_scaler, os.path.join(ARTIFACT_DIR, "xtb_cond_inter_scaler.joblib"))
    torch.save({
        "state_dict": final_ann.state_dict(),
        "drfp_dim": Xd_reduced.shape[1],
        "xtb_dim": Xo_s.shape[1],
        "hidden": 128,
        "lambda_prop": lambda_prop,
    }, os.path.join(ARTIFACT_DIR, "dual_branch_ann.pt"))

    meta = {
        "pipeline": f"PCL-AE-{latent_dim} + DualBranchANN (lambda={lambda_prop}, full data)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_samples": int(len(y)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "drfp_raw_dim": int(X_drfp.shape[1]),
        "drfp_reduced_dim": int(Xd_reduced.shape[1]),
        "drfp_variant": best_var,
        "drfp_col": best_col,
        "xtb_dim": int(X_xtb.shape[1]),
        "cond_dim": int(X_cond.shape[1]),
        "inter_dim": int(X_inter.shape[1]),
        "xtb_cond_inter_dim": int(X_xtb_cond_inter.shape[1]),
        "xtb_cols": xtb_cols,
        "cond_cols": cond_cols,
        "interaction_rules": [
            {"name": "T * activation_proxy", "cond_idx": 0, "xtb_name": "activation_proxy"},
            {"name": "P * total_polarity_index", "cond_idx": 1, "xtb_name": "total_polarity_index"},
        ],
        "scaler_notes": {
            "drfp_scaler": f"StandardScaler on raw DRFP ({best_var}, 2048-D)",
            "xtb_cond_inter_scaler": "StandardScaler on [X_xtb | X_cond | X_inter]",
        },
        "drfp_ablation_meta_source": os.path.join(os.path.dirname(ARTIFACT_DIR), "drfp_ablation_meta.json"),
        "pcl_ae_kwargs": {
            "latent_dim": LATENT_DIM, "lambda_prop": lambda_prop, "pos_weight": 10.0,
            "epochs": EPOCHS_PCL, "batch_size": 128, "lr": 1e-3,
        },
        "ann_kwargs": {
            "hidden": 128, "epochs": EPOCHS_ANN, "batch_size": 32, "lr": 5e-4,
            "weight_decay": 1e-3,
        },
        "cv_metrics": metrics,
    }
    with open(os.path.join(ARTIFACT_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ARTIFACT_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    report = [
        "=" * 72,
        "  Save Best Model - Report",
        "=" * 72,
        f"  Device              : {DEVICE}",
        f"  N training samples  : {len(y)}",
        f"  PCL-AE lambda_prop  : {lambda_prop}",
        f"  y range             : [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}",
        "",
        "  CV metrics (PCL-AE-128 + DualBranchANN, 5-fold, early stop)"
        f"  [protocol={metrics['protocol']}; sanity check only, not reported as performance]:",
        f"    R^2      = {metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}",
        f"    MAE      = {metrics['mae_mean']:.4f}",
        f"    RMSE     = {metrics['rmse_mean']:.4f}",
        f"    Pearson  = {metrics['pearson_mean']:.4f}",
        "",
        "  Per-fold:",
    ]
    for r in metrics["per_fold"]:
        report.append(
            f"    fold {r['fold']}: R^2={r['r2']:.4f}  MAE={r['mae']:.4f}  Pearson={r['pearson']:.4f}"
        )
    report.append("")
    report.append(f"  Artifacts saved under:")
    report.append(f"    {ARTIFACT_DIR}")
    for fname in sorted(os.listdir(ARTIFACT_DIR)):
        sz = os.path.getsize(os.path.join(ARTIFACT_DIR, fname))
        report.append(f"    {fname:38s} {sz:>10d} bytes")
    report.append("")
    report.append(f"  Elapsed: {elapsed:.1f} s")
    report.append("=" * 72)
    text = "\n".join(report)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "save_best_model_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()