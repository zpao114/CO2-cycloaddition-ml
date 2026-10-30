# -*- coding: utf-8 -*-
"""
utils_benchmark.py
==================
Shared benchmark utilities extracted from 201_ablation.py.

Exports:
    parse_drfp_col          parse a DataFrame column of DRFP bit-string cells
    load_data               load co2_drfp_xtb_extended.csv, split DRFP/XTB/Cond/Cat/Inter
    X_inter_base            legacy zero-column interaction builder (placeholder)
    train_standard_ae       StandardAE wrapper (X -> latent_dim)
    train_pcl_ae            PropertyCoLearningAE wrapper (X, y -> latent_dim)
    DualBranchANN           conv1d-on-DRFP + MLP-on-XTB fusion network
    get_tree_model          returns RF / XGB / LGBM with paper hyperparameters
    eval_sklearn_model      repeated KFold eval returning mean/std metrics
    eval_dual_branch        repeated KFold eval for DualBranchANN
    DEVICE                  torch.device('cuda' if available else 'cpu')
    OUTPUT_DIR              <PROJECT_ROOT>/results_best_pipeline

Downstream consumers:
    301_benchmark.py            (line 37)
    801_drfp_ablation_deep_analysis.py (line 37)

Notes:
    * This module is the de-facto 'tool subset' of 201_ablation.py.
      Every signature and body below was copied verbatim from that file on
      2026-08-12; the only difference is that 201's main()
      and main() are stripped.
    * Constants DATA_EXTENDED and the DRFP_VARIANTS dict remain here so that
      `load_data()` and `parse_drfp_col()` can resolve their inputs without
      importing from 201_ablation.
"""

import os
import sys
import io
import json
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold  # noqa: F401  (kept for legacy callers; not used in canonical 5-fold CV paths)
from src.data_split import (load_manifest, holdout_arrays, kfold_folds, split_iterator,
                            select_train_pool, verify_manifest)

import os as _os
_proj = _os.environ.get("CO2_PROJECT_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import sys as _sys
_sys.path.insert(0, _os.path.join(_proj, 'src'))
from config import BEST_LAMBDA_PROP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, PROJECT_ROOT)
from utils_rxn import read_drfp
from utils_features import require_cond_cols

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results_best_pipeline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRFP_VARIANTS = {
    'full':      ('drfp',          'DRFP full (epoxide+catalyst+solvent)'),
    'reactants': ('drfp React',    'DRFP reactants only'),
    'no_cats':   ('drfp wo cats',  'DRFP no catalyst'),
    'no_sols':   ('drfp wo sols',  'DRFP no solvent'),
}
DRFP_VARIANT_COLS = {k: v[0] for k, v in DRFP_VARIANTS.items()}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================
# DRFP column parser
# =============================================================
def parse_drfp_col(series, variant_key=None):
    """
    Convert a pandas Series of DRFP bit-string cells into a (n, 2048)
    float32 numpy array.  Missing / unparseable rows are filled with zeros.
    """
    arrs = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


# =============================================================
# Data loading (DRFP / XTB / Cond / Cat one-hot / Inter / y)
# =============================================================
def load_data(drfp_variant='full', use_holdout_train=True):
    """Load co2_drfp_xtb_extended.csv, split DRFP/XTB/Cond/Cat/Inter

    If ``use_holdout_train=True`` (default), restrict to the 85% holdout train
    pool from data_split.json so all downstream CV is on the SAME train pool.
    """
    print(f'\n[Step 0] Loading data {os.path.basename(DATA_EXTENDED)} ...')
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    df = df[df['extraction_status'] == 'valid'].copy()
    df = df.dropna(subset=['yield (%)'])
    df = df[df['yield (%)'] > 0].reset_index(drop=True)

    if use_holdout_train:
        # verifies row count / row_id order / SHA-256 data hash before indexing by position
        df = select_train_pool(df, load_manifest())
        print(f'  [load_data] filtered to holdout train pool: {len(df)} rows (manifest verified)')

    n = len(df)
    print(f'  Valid samples: {n}')

    col = DRFP_VARIANT_COLS.get(drfp_variant)
    if col is None or col not in df.columns:
        raise ValueError(f"DRFP variant '{drfp_variant}' col '{col}' not found in data file")
    X_drfp = parse_drfp_col(df[col], drfp_variant).astype(np.float32)
    drfp_label = DRFP_VARIANTS[drfp_variant][1]
    print(f'  DRFP [{drfp_variant}]: {X_drfp.shape[1]}D  ({drfp_label})')

    xtb_cols = [
        'sub_homo_eV', 'sub_lumo_eV', 'sub_gap_eV', 'sub_dipole_D',
        'co2_homo_eV', 'co2_lumo_eV', 'co2_gap_eV',
        'cat_homo_eV', 'cat_lumo_eV', 'cat_gap_eV', 'cat_dipole_D',
        'solv_homo_eV', 'solv_lumo_eV', 'solv_gap_eV',
        'delta_E_hl_cat_sub', 'global_hardness', 'nucleophilicity_index',
        'cat_homo_eV_min', 'cat_lumo_eV_max', 'cat_gap_eV_min',
        'cat_cation_homo_eV', 'cat_cation_lumo_eV', 'cat_cation_gap_eV',
        'cat_anion_homo_eV', 'cat_anion_lumo_eV', 'cat_anion_gap_eV',
        'cat_cation_dipole_D', 'cat_anion_dipole_D',
        'activation_proxy', 'charge_transfer_potential', 'ion_pair_interaction',
        'electrophilicity_cat', 'electrodonating_cat',
        'sub_cat_orbital_match', 'gap_ratio', 'hardness_ratio',
        'nucleophilicity_cat', 'reaction_polarity', 'co2_activation_proxy',
        'solv_cat_interaction', 'solv_sub_interaction',
        'total_polarity_index', 'dielectric_proxy',
    ]
    xtb_cols = [c for c in xtb_cols if c in df.columns]
    X_xtb_raw = df[xtb_cols].values.astype(np.float32)
    X_xtb = np.nan_to_num(X_xtb_raw, nan=0.0)
    print(f'  XTB full: {X_xtb.shape[1]}D  ({len(xtb_cols)} cols)')

    cond_cols = require_cond_cols(df, include_loading=True)
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)
    print(f'  Conditions: {X_cond.shape[1]}D')

    cat_parts = []
    if 'catalyst_system_type' in df.columns:
        for v in sorted(df['catalyst_system_type'].fillna('unknown').unique()):
            cat_parts.append(
                (df['catalyst_system_type'].fillna('unknown') == v).astype(np.float32).values
            )
    if 'reactant_name' in df.columns:
        for v in sorted(df['reactant_name'].fillna('unknown').unique()):
            cat_parts.append(
                (df['reactant_name'].fillna('unknown') == v).astype(np.float32).values
            )
    X_cat = np.stack(cat_parts, axis=1).astype(np.float32) if cat_parts else \
        np.zeros((n, 0), dtype=np.float32)
    print(f'  Category one-hot: {X_cat.shape[1]}D')

    def xtb_idx(name):
        return xtb_cols.index(name) if name in xtb_cols else None

    T = X_cond[:, 0:1]
    P = X_cond[:, 1:2]
    inter_parts = []
    ai = xtb_idx('activation_proxy')
    tpi = xtb_idx('total_polarity_index')
    if ai is not None:
        inter_parts.append(T * X_xtb[:, ai:ai+1])
    if tpi is not None:
        inter_parts.append(P * X_xtb[:, tpi:tpi+1])
    X_inter = np.concatenate(inter_parts, axis=1).astype(np.float32) \
        if inter_parts else np.zeros((n, 0), dtype=np.float32)
    print(f'  Interaction features: {X_inter.shape[1]}D')

    y = df['yield (%)'].values.astype(np.float32) / 100.0
    catalyst_groups = df['catalyst_1_name'].fillna('unknown').astype(str).values

    return X_drfp, X_xtb, X_cond, X_cat, X_inter, y, df, catalyst_groups, xtb_cols, drfp_variant


# =============================================================
# Legacy interaction-feature builder
# =============================================================
def X_inter_base(X_xtb, X_cond):
    """Build interaction features (placeholder, for backward compatibility).

    NOTE: this is a placeholder.  The real interaction features are built
    inside `load_data()`.  Kept here only because 301 / 801 historically
    imported it; both downstream scripts currently only re-export the name
    without actually calling it.
    """
    return np.zeros((X_xtb.shape[0], 0), dtype=np.float32)


# =============================================================
# Standard auto-encoder
# =============================================================
# --- canonical definitions live in eval_protocol (de-duplicated) ---
from eval_protocol.nets import (StandardAE, PropertyCoLearningAE,  # noqa: E402,F401
                                train_standard_ae, train_pcl_ae, encode_with_model)
from eval_protocol.models import DualBranchANN  # noqa: E402,F401



# =============================================================
# Property Co-learning AE (Standard)
# =============================================================


# =============================================================
# Improved Property Co-learning AE (VAE-style)
# Uses variational inference for better latent space structure
# and Huber loss for robustness to outliers in chemical data
# =============================================================
class ImprovedPCLAE(nn.Module):
    """
    Variational Property Co-Learning Autoencoder.

    Improvements over PropertyCoLearningAE:
    1. VAE-style latent space (mu, logvar) for smoother interpolation
    2. KL divergence regularization to prevent latent space collapse
    3. Beta-VAE style weighting for controlled latent space structure

    This architecture is more suitable for chemical data where:
    - Molecular representations should vary continuously
    - Outliers in yield data are common
    - Latent space interpolation can reveal structure-activity relationships
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=256, beta=0.01):
        """
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            beta: KL divergence weight (beta-VAE style, default 0.01)
        """
        super().__init__()
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # DRFP values are in [0, 1]
        )

        # Property predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        """Return latent mean (deterministic encoding)."""
        h = self.encoder(x)
        return self.fc_mu(h)

    def forward(self, x):
        """Full forward pass with variational sampling."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)

        return recon, pred, mu, logvar


# =============================================================
# AE trainers
# =============================================================




def train_pcl_ae_improved(X, y, latent_dim, epochs=150, batch_size=128, lr=1e-3,
                          lambda_prop=None, beta=0.01):
    """Train Improved Property Co-learning AE with VAE-style architecture.

    Improvements over train_pcl_ae():
    1. VAE-style latent space with KL divergence regularization
    2. Huber Loss (SmoothL1Loss) for property prediction - more robust to outliers
    3. Gradient clipping for training stability
    4. Cosine annealing learning rate schedule

    Args:
        X: Input features (DRFP + auxiliary features)
        y: Target property (normalized yield, [0, 1])
        latent_dim: Latent space dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_prop: Weight for property prediction loss (default: from config)
        beta: KL divergence weight (beta-VAE style, default 0.01)

    Returns:
        Latent representations (n_samples, latent_dim)
    """
    if lambda_prop is None:
        lambda_prop = float(BEST_LAMBDA_PROP)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize target property for stable training
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_norm = (y - y_mean) / y_std

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_norm).to(device)

    model = ImprovedPCLAE(X.shape[1], latent_dim, beta=beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Use Huber Loss (SmoothL1Loss) - more robust to outliers than MSE
    # DRFP reconstruction still uses BCE (appropriate for binary data)
    recon_criterion = nn.BCEWithLogitsLoss()
    prop_criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss, robust to outliers

    # Cosine annealing schedule for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_prop = 0.0
        epoch_kl = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()

            recon, pred, mu, logvar = model(xb)

            # Reconstruction loss (BCE for DRFP - binary data)
            recon_loss = recon_criterion(recon, xb)

            # Property prediction loss (Huber - robust to outliers)
            prop_loss = prop_criterion(pred, yb)

            # KL divergence regularization (encourages Gaussian latent space)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss with proper scaling
            loss = recon_loss + lambda_prop * prop_loss + beta * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_prop += prop_loss.item()
            epoch_kl += kl_loss.item()

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"  [ImprovedPCL-AE] Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Recon: {epoch_recon/len(loader):.4f} | "
                  f"Prop: {epoch_prop/len(loader):.4f} | "
                  f"KL: {epoch_kl/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        latent = model.encode(X_tensor).cpu().numpy()

    return latent


# Alias for backward compatibility
train_pcl_ae_v2 = train_pcl_ae_improved


# =============================================================
# DualBranchANN
# =============================================================


# =============================================================
# Tree-model factory
# =============================================================
def get_tree_model(name, seed=42):
    """RF / XGB / LGBM.  ``seed`` defaults to 42 for single-pass callers; any loop
    that *repeats* an evaluation must pass a seed that varies with the repeat
    (otherwise every repeat is bit-identical and ``r2_std`` is not a spread over
    independent runs).  ``eval_sklearn_model`` uses ``seed = 1000*rep + fold_id``."""
    if name == 'RF':
        return RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=seed)
    if name == 'XGB':
        return xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=seed, verbosity=0)
    if name == 'LGBM':
        return lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=10,
            device='cpu', verbose=-1, random_state=seed)
    raise ValueError(name)


# =============================================================
# Cross-validated evaluators
# =============================================================
def eval_sklearn_model(model_name, X, y, n_folds=5, n_repeats=3):
    """5-fold CV on the canonical data_split.json manifest (seed=2026, yield-stratified).

    NOTE on ``n_repeats``: the manifest folds are FIXED, so a repeat only re-seeds
    the model (``seed = 1000*rep + fold_id``).  ``r2_std`` is therefore the SD over
    fold x model-seed evaluations (descriptive); it is NOT an SE of the mean and the
    ``n_repeats*n_folds`` evaluations are not independent (training sets overlap).
    Before this fix the tree seed was constant, so repeats were identical copies.
    """
    fold_records = []
    n = len(y)
    folds = kfold_folds(load_manifest())
    for rep in range(n_repeats):
        for fold_id, tr, te in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = get_tree_model(model_name, seed=1000 * rep + int(fold_id))
            model.fit(X_tr, y[tr])
            pred = np.clip(model.predict(X_te), 0, 1)
            r2 = r2_score(y[te], pred)
            mae = mean_absolute_error(y[te], pred)
            rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
            try:
                pr = float(pearsonr(y[te], pred)[0])
            except Exception:
                pr = np.nan
            fold_records.append({
                'rep': rep, 'fold': fold_id, 'r2': r2,
                'mae': mae, 'rmse': rmse, 'pearson': pr,
            })
    fm = pd.DataFrame(fold_records)
    return {
        'r2_mean': float(fm['r2'].mean()),
        'r2_std': float(fm['r2'].std()),
        'mae_mean': float(fm['mae'].mean()),
        'rmse_mean': float(fm['rmse'].mean()),
        'pearson_mean': float(fm['pearson'].mean()),
    }


def eval_dual_branch(X_drfp, X_other, y, n_folds=5, n_repeats=2, hidden=128):
    """DualBranchANN 5-fold CV using data_split.json manifest.

    Early stopping / LR scheduling use a validation split carved out of the
    TRAINING fold only (``eval_protocol.models.DualBranchRegressor``); the test
    fold is never consulted, and the reported R2 is that of the prediction at
    the best *validation* epoch (previously: max-over-epochs test R2, which is
    an optimistically biased selection).  Each repeat uses a different seed.
    """
    from eval_protocol.models import DualBranchRegressor
    fold_records = []
    folds = kfold_folds(load_manifest())
    for rep in range(n_repeats):
        for fold_id, tr, te in folds:
            sd = StandardScaler(); so = StandardScaler()
            Xd_tr = sd.fit_transform(X_drfp[tr])
            Xd_te = sd.transform(X_drfp[te])
            Xo_tr = so.fit_transform(X_other[tr])
            Xo_te = so.transform(X_other[te])
            reg = DualBranchRegressor(
                n_first=X_drfp.shape[1], hidden=hidden,
                random_state=1000 * rep + fold_id, device=str(DEVICE))
            reg.fit(np.hstack([Xd_tr, Xo_tr]).astype(np.float32), y[tr])
            pred = np.clip(reg.predict(np.hstack([Xd_te, Xo_te]).astype(np.float32)), 0, 1)
            r2 = r2_score(y[te], pred)
            mae = mean_absolute_error(y[te], pred)
            rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
            try:
                pr = float(pearsonr(y[te], pred)[0])
            except Exception:
                pr = np.nan
            fold_records.append({
                'rep': rep, 'fold': fold_id, 'r2': float(r2),
                'mae': float(mae), 'rmse': float(rmse), 'pearson': float(pr),
            })
    fm = pd.DataFrame(fold_records)
    return {
        'r2_mean': float(fm['r2'].mean()),
        'r2_std': float(fm['r2'].std()),
        'mae_mean': float(fm['mae'].mean()),
        'rmse_mean': float(fm['rmse'].mean()),
        'pearson_mean': float(fm['pearson'].mean()),
    }


# =============================================================
# DRFP-ablation-meta helper (consumed by 301 / 801)
# =============================================================
def load_best_variant_from_meta():
    """Read drfp_ablation_meta.json; raise informative error if missing.

    Mirrors the same function previously inlined at the top of 301_benchmark.py.
    """
    meta_path = os.path.join(OUTPUT_DIR, 'drfp_ablation_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'Drfp ablation meta file not found: {meta_path}\n'
            f'Run 201_ablation.py first to generate it.'
        )
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta['best_variant'], meta['best_label'], meta['best_r2']


if __name__ == '__main__':
    print('=' * 72)
    print('  utils_benchmark.py -- smoke test')
    print('=' * 72)
    print(f'PROJECT_ROOT  : {PROJECT_ROOT}')
    print(f'DATA_EXTENDED : {DATA_EXTENDED}  (exists={os.path.exists(DATA_EXTENDED)})')
    print(f'OUTPUT_DIR    : {OUTPUT_DIR}')
    print(f'DEVICE        : {DEVICE}')
    print(f'DRFP variants : {list(DRFP_VARIANTS)}')
