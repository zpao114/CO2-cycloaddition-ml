# -*- coding: utf-8 -*-
"""
405_external_validation.py
=============================
外部测试集验证（External Validation）
======================================
按照 JCIM OECD QSPR Principle 4 要求：
  - 按 data_split.json 的标准划分：85% 训练 / 15% 测试（seed=2026）
  - 在完全封闭的测试集上评估所有模型
  - 与训练池内 manifest 5 折 CV 结果对比（7 个模型全部现算），验证模型是否过拟合

输出：
  results_external_validation/
      external_validation_results.csv
      external_vs_internal_comparison.csv
"""

from __future__ import annotations

import os
import sys
import io
import time
import json
import argparse
import logging
import warnings
import random
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold  # noqa: F401  (kept for inner splits; canonical CV uses data_split.json)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from src.data_split import load_manifest, holdout_arrays, kfold_folds, select_train_pool, verify_manifest
from utils_rxn import read_drfp, get_best_drfp_variant
from config import BEST_LAMBDA_PROP, PCL_AE_EPOCHS                     # noqa: E402
from eval_protocol.nets import PropertyCoLearningAE, train_pcl_ae_model  # noqa: E402,F401
from eval_protocol.models import DualBranchANN                         # noqa: E402,F401

DATA_EXTENDED = os.path.join(PROJECT_ROOT, 'results/results_cho_diagnostic/co2_drfp_xtb_extended.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results_external_validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2026          # 外部验证专用种子（不同于 CV 用的 42）
HOLDOUT_RATIO = 0.15  # 15% 测试集

logger = logging.getLogger('external_validation')


# =====================================================================
# 数据加载（复用 08_benchmark.py 的逻辑）
# =====================================================================
def _col(df, key):
    """模糊匹配列名（兼容编码导致的乱码）"""
    return [c for c in df.columns if key.lower() in c.lower()]


def parse_drfp_col(series):
    arrs = []
    for fp_str in series:
        a = read_drfp(fp_str)
        if a is None or a.size == 0:
            arrs.append(np.zeros(2048, dtype=np.float32))
        else:
            arrs.append(a.astype(np.float32))
    return np.array(arrs)


def load_data(use_holdout_train=True):
    """Load data, dynamically selecting the best DRFP variant from ablation meta."""
    meta = get_best_drfp_variant()
    best_col = meta['best_drfp_col']
    best_var = meta['best_variant']
    logger.info(f'\n[Step 0] 加载数据 ...')
    logger.info(f'  DRFP variant: {best_var} ({best_col})')
    df = pd.read_csv(DATA_EXTENDED, encoding='utf-8-sig')
    df = df[df['extraction_status'] == 'valid'].copy()
    df = df.dropna(subset=['yield (%)'])
    df = df[df['yield (%)'] > 0].reset_index(drop=True)

    manifest = load_manifest()
    if use_holdout_train:
        df = select_train_pool(df, manifest)   # verifies row_id order + data hash
        logger.info(f'  [load_data] filtered to holdout train pool: {len(df)} rows (manifest verified)')
    else:
        verify_manifest(df, manifest)           # full table: indices stay absolute
        logger.info(f'  [load_data] full table: {len(df)} rows (manifest verified)')

    n = len(df)
    logger.info(f'  有效样本数: {n}')

    # DRFP（动态选择最优变体）
    X_drfp = parse_drfp_col(df[best_col]).astype(np.float32)
    logger.info(f'  DRFP [{best_col}]: {X_drfp.shape[1]}D')

    # XTB（动态列名匹配）
    xtb_candidates = [
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
    xtb_cols = [c for c in xtb_candidates if c in df.columns]
    X_xtb = np.nan_to_num(df[xtb_cols].values.astype(np.float32), nan=0.0)
    logger.info(f'  XTB: {X_xtb.shape[1]}D')

    # 条件列（模糊匹配温度/压力/时间）
    temp_col = _col(df, 'temperature')[0]
    pres_col = _col(df, 'pressure')[0]
    time_col = _col(df, 'time(')[0] if _col(df, 'time(') else _col(df, 'time')[0] if _col(df, 'time') else None

    cond_base = [temp_col, pres_col]
    if time_col:
        cond_base.append(time_col)
    cat_loading_cols = [c for c in df.columns if 'loading_mol%' in c]
    cond_cols = cond_base + cat_loading_cols
    cond_cols = [c for c in cond_cols if c in df.columns]
    X_cond = np.nan_to_num(df[cond_cols].values.astype(np.float32), nan=0.0)
    logger.info(f'  条件: {X_cond.shape[1]}D ({cond_cols})')

    # 交互特征
    T = X_cond[:, 0:1]
    P = X_cond[:, 1:2]
    inter_parts = []
    ai = xtb_cols.index('activation_proxy') if 'activation_proxy' in xtb_cols else None
    tpi = xtb_cols.index('total_polarity_index') if 'total_polarity_index' in xtb_cols else None
    if ai is not None:
        inter_parts.append(T * X_xtb[:, ai:ai+1])
    if tpi is not None:
        inter_parts.append(P * X_xtb[:, tpi:tpi+1])
    X_inter = np.concatenate(inter_parts, axis=1).astype(np.float32) \
        if inter_parts else np.zeros((n, 0), dtype=np.float32)
    logger.info(f'  交互特征: {X_inter.shape[1]}D')

    X_xtb_cond_inter = np.hstack([X_xtb, X_cond, X_inter]).astype(np.float32)

    y = df['yield (%)'].values.astype(np.float32) / 100.0

    # 保存原始数据（用于测试集分析报告），用模糊匹配
    cat_col = 'catalyst_1_name'
    sys_type_col = 'catalyst_system_type'
    reactant_col = 'reactant_name'
    df_clean = pd.DataFrame({
        'catalyst_1_name': df[cat_col].values,
        'reactant_name': df[reactant_col].values if reactant_col in df.columns else 'unknown',
        'catalyst_system_type': df[sys_type_col].values if sys_type_col in df.columns else 'unknown',
        'temperature': df[temp_col].values,
        'pressure': df[pres_col].values,
        'time': df[time_col].values if time_col else np.zeros(n),
        'yield_pct': df['yield (%)'].values,
    })

    return X_drfp, X_xtb_cond_inter, y, df_clean, xtb_cols


# =====================================================================
# 模型定义（复用 08_benchmark.py）
# =====================================================================
def train_pcl_ae(X_train, y_train, latent_dim=128, lambda_prop=None,
                 epochs=PCL_AE_EPOCHS, batch_size=128, lr=1e-3):
    return train_pcl_ae_model(X_train, y_train, latent_dim, lambda_prop, epochs,
                              batch_size, lr, device=DEVICE)





def get_tree_model(name, seed=None):
    seed = SEED if seed is None else seed
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


# =====================================================================
# 主评估逻辑
# =====================================================================
def evaluate_on_holdout(X_drfp_all, X_xtb_all, y_all,
                        train_idx, test_idx,
                        pcl_lambda=None, latent_dim=128):
    """在 hold-out 集上训练并评估所有模型"""
    if pcl_lambda is None:
        pcl_lambda = float(BEST_LAMBDA_PROP)
    Xd_tr = X_drfp_all[train_idx]
    Xd_te = X_drfp_all[test_idx]
    Xo_tr = X_xtb_all[train_idx]
    Xo_te = X_xtb_all[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    results = {}
    records = []

    # ---- 1. DRFP 原始（无降维）+ RF/XGB/LGBM ----
    logger.info('\n  [Tree models on raw DRFP + XTB]')
    for model_name in ['RF', 'XGB', 'LGBM']:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.hstack([Xd_tr, Xo_tr]))
        X_te = scaler.transform(np.hstack([Xd_te, Xo_te]))
        m = get_tree_model(model_name)
        m.fit(X_tr, y_tr)
        pred = np.clip(m.predict(X_te), 0, 1)
        r2 = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        try:
            pr = pearsonr(y_te, pred)[0]
        except Exception:
            pr = np.nan
        results[model_name] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr}
        records.append({
            'model': model_name, 'r2': r2, 'mae': mae,
            'rmse': rmse, 'pearson': pr,
            'note': 'raw DRFP + XTB + Cond + Inter'
        })
        logger.info(f'    {model_name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')

    # ---- 2. PCA-128 + RF/XGB ----
    logger.info('\n  [PCA-128 + RF/XGB]')
    from sklearn.decomposition import PCA
    sd = StandardScaler()
    Xd_tr_s = sd.fit_transform(Xd_tr)
    pca = PCA(n_components=128, random_state=SEED)
    Xd_tr_pca = pca.fit_transform(Xd_tr_s).astype(np.float32)
    Xd_te_pca = pca.transform(sd.transform(Xd_te)).astype(np.float32)

    for model_name in ['RF', 'XGB']:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.hstack([Xd_tr_pca, Xo_tr]))
        X_te = scaler.transform(np.hstack([Xd_te_pca, Xo_te]))
        m = get_tree_model(model_name)
        m.fit(X_tr, y_tr)
        pred = np.clip(m.predict(X_te), 0, 1)
        r2 = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        try:
            pr = pearsonr(y_te, pred)[0]
        except Exception:
            pr = np.nan
        results[f'PCA-128+{model_name}'] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr}
        records.append({
            'model': f'PCA-128+{model_name}',
            'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr,
            'note': 'PCA-128 DRFP + XTB + Cond + Inter'
        })
        logger.info(f'    PCA-128+{model_name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')

    # ---- 3. PCL-AE-128 + RF ----
    logger.info('\n  [PCL-AE-128 + RF]')
    s_drfp_pcl = StandardScaler()
    Xd_tr_sp = s_drfp_pcl.fit_transform(Xd_tr).astype(np.float32)
    Xd_te_sp = s_drfp_pcl.transform(Xd_te).astype(np.float32)

    pcl_for_rf = train_pcl_ae(Xd_tr_sp, y_tr, latent_dim=latent_dim,
                                 lambda_prop=pcl_lambda)
    with torch.no_grad():
        Zd_tr_rf = pcl_for_rf.encode(torch.FloatTensor(Xd_tr_sp).to(DEVICE)).cpu().numpy()
        Zd_te_rf = pcl_for_rf.encode(torch.FloatTensor(Xd_te_sp).to(DEVICE)).cpu().numpy()

    s_z = StandardScaler(); s_o = StandardScaler()
    Zd_tr_sc = s_z.fit_transform(Zd_tr_rf)
    Zd_te_sc = s_z.transform(Zd_te_rf)
    Xo_tr_sc = s_o.fit_transform(Xo_tr)
    Xo_te_sc = s_o.transform(Xo_te)

    m_rf = get_tree_model('RF')
    m_rf.fit(np.hstack([Zd_tr_sc, Xo_tr_sc]), y_tr)
    pred_rf = np.clip(m_rf.predict(np.hstack([Zd_te_sc, Xo_te_sc])), 0, 1)
    r2 = r2_score(y_te, pred_rf)
    mae = mean_absolute_error(y_te, pred_rf)
    rmse = np.sqrt(mean_squared_error(y_te, pred_rf))
    try:
        pr = pearsonr(y_te, pred_rf)[0]
    except Exception:
        pr = np.nan
    results['PCL-AE-128+RF'] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr}
    records.append({
        'model': 'PCL-AE-128+RF',
        'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr,
        'note': f'PCL-AE-{latent_dim} DRFP + XTB + Cond + Inter (λ={pcl_lambda})'
    })
    logger.info(f'    PCL-AE-128+RF: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')

    # ---- 4. PCL-AE-128 + DualANN ----
    logger.info('\n  [PCL-AE-128 + DualANN]')
    s_drfp_ann = StandardScaler()
    Xd_tr_sa = s_drfp_ann.fit_transform(Xd_tr).astype(np.float32)
    Xd_te_sa = s_drfp_ann.transform(Xd_te).astype(np.float32)
    s_other_ann = StandardScaler()
    Xo_tr_sa = s_other_ann.fit_transform(Xo_tr).astype(np.float32)
    Xo_te_sa = s_other_ann.transform(Xo_te).astype(np.float32)

    pcl_for_ann = train_pcl_ae(Xd_tr_sa, y_tr, latent_dim=latent_dim,
                                  lambda_prop=pcl_lambda)
    with torch.no_grad():
        Zd_tr_ann = pcl_for_ann.encode(torch.FloatTensor(Xd_tr_sa).to(DEVICE)).cpu().numpy()
        Zd_te_ann = pcl_for_ann.encode(torch.FloatTensor(Xd_te_sa).to(DEVICE)).cpu().numpy()
    # Same latent standardisation (train-fitted) as internal_cv_reference, so the
    # internal-vs-external DualANN comparison uses one pipeline.
    s_z_ann = StandardScaler().fit(Zd_tr_ann)
    Zd_tr_ann = s_z_ann.transform(Zd_tr_ann).astype(np.float32)
    Zd_te_ann = s_z_ann.transform(Zd_te_ann).astype(np.float32)

    # inner split for early stopping
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=SEED + 999)
    inner_tr_idx, inner_val_idx = next(inner_kf.split(Zd_tr_ann))
    Zd_tr2 = Zd_tr_ann[inner_tr_idx].astype(np.float32)
    Zd_va  = Zd_tr_ann[inner_val_idx].astype(np.float32)
    Xo_tr2 = Xo_tr_sa[inner_tr_idx].astype(np.float32)
    Xo_va  = Xo_tr_sa[inner_val_idx].astype(np.float32)
    y_tr2  = y_tr[inner_tr_idx].astype(np.float32)
    y_va   = y_tr[inner_val_idx].astype(np.float32)

    Zd_tr2_t = torch.FloatTensor(Zd_tr2).to(DEVICE)
    Zd_va_t  = torch.FloatTensor(Zd_va).to(DEVICE)
    Xo_tr2_t = torch.FloatTensor(Xo_tr2).to(DEVICE)
    Xo_va_t  = torch.FloatTensor(Xo_va).to(DEVICE)
    y_tr2_t  = torch.FloatTensor(y_tr2).to(DEVICE)
    y_va_t   = torch.FloatTensor(y_va).to(DEVICE)
    Zd_te_t  = torch.FloatTensor(Zd_te_ann).to(DEVICE)
    Xo_te_t  = torch.FloatTensor(Xo_te_sa).to(DEVICE)

    # 8 组网格搜索 + 5-fold CV 选 cfg（更稳健的 val 估计）
    search_grid = [
        dict(wd=1e-4, lr=1e-3, ep=200, bs=64),
        dict(wd=1e-3, lr=5e-4, ep=200, bs=64),
        dict(wd=2e-3, lr=5e-4, ep=250, bs=64),
        dict(wd=5e-3, lr=5e-4, ep=250, bs=64),
        dict(wd=1e-3, lr=5e-4, ep=250, bs=32),
        dict(wd=1e-3, lr=1e-3, ep=200, bs=32),
        dict(wd=1e-3, lr=2e-4, ep=300, bs=64),
        dict(wd=3e-3, lr=2e-4, ep=300, bs=64),
    ]
    logger.info(f'    超参数搜索 ({len(search_grid)} 组 × 5-fold CV)...')
    K_FOLDS = 5
    cv_summary = []
    best_inner_r2, best_cfg, best_state, best_test_pred = -np.inf, None, None, None
    for cfg in search_grid:
        fold_r2 = []
        for fold_idx, (fit_idx, val_idx) in enumerate(KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED + 999).split(Zd_tr_ann)):
            Zd_f = Zd_tr_ann[fit_idx].astype(np.float32)
            Zd_va_f = Zd_tr_ann[val_idx].astype(np.float32)
            Xo_f = Xo_tr_sa[fit_idx].astype(np.float32)
            Xo_va_f = Xo_tr_sa[val_idx].astype(np.float32)
            y_f = y_tr[fit_idx].astype(np.float32)
            y_va_f = y_tr[val_idx].astype(np.float32)

            # 80/20 inner split for early stopping
            n_f = len(y_f)
            rng = np.random.RandomState(SEED + 1000 + fold_idx)
            perm = rng.permutation(n_f)
            cut = int(n_f * 0.8)
            it, iv = perm[:cut], perm[cut:]
            Zd_tt, Zd_iv = Zd_f[it], Zd_f[iv]
            Xo_tt, Xo_iv = Xo_f[it], Xo_f[iv]
            y_tt, y_iv = y_f[it], y_f[iv]

            Zd_tt_t = torch.FloatTensor(Zd_tt).to(DEVICE)
            Zd_iv_t = torch.FloatTensor(Zd_iv).to(DEVICE)
            Xo_tt_t = torch.FloatTensor(Xo_tt).to(DEVICE)
            Xo_iv_t = torch.FloatTensor(Xo_iv).to(DEVICE)
            y_tt_t = torch.FloatTensor(y_tt).to(DEVICE)
            Zd_va_f_t = torch.FloatTensor(Zd_va_f).to(DEVICE)
            Xo_va_f_t = torch.FloatTensor(Xo_va_f).to(DEVICE)

            torch.manual_seed(SEED + 7 + fold_idx)
            net = DualBranchANN(Zd_tt.shape[1], Xo_tt.shape[1], hidden=latent_dim).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
            loader = DataLoader(TensorDataset(Zd_tt_t, Xo_tt_t, y_tt_t), batch_size=cfg['bs'], shuffle=True)
            br2, bstate, ni = -np.inf, None, 0
            for ep in range(cfg['ep']):
                net.train()
                for z, xo, yy in loader:
                    opt.zero_grad()
                    loss = nn.MSELoss()(net(z, xo), yy)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    pv = net(Zd_iv_t, Xo_iv_t).cpu().numpy().squeeze()
                r2v = r2_score(y_iv, pv)
                sched.step(1 - r2v)
                if r2v > br2:
                    br2 = r2v
                    bstate = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    ni = 0
                else:
                    ni += 1
                if ni >= 40:
                    break
            net.load_state_dict(bstate)
            net.eval()
            with torch.no_grad():
                pv_fold = net(Zd_va_f_t, Xo_va_f_t).cpu().numpy().squeeze()
            fold_r2.append(r2_score(y_va_f, np.clip(pv_fold, 0, 1)))
        cv_mean = float(np.mean(fold_r2))
        cv_std = float(np.std(fold_r2))
        cv_summary.append((cfg, cv_mean, cv_std, fold_r2))
        # 中间报告（test 也算一次供参考，但只用于查看，不参与选 cfg）
        torch.manual_seed(SEED + 7)
        net = DualBranchANN(Zd_tr_ann.shape[1], Xo_tr_sa.shape[1], hidden=latent_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
        loader = DataLoader(TensorDataset(Zd_tr2_t, Xo_tr2_t, y_tr2_t), batch_size=cfg['bs'], shuffle=True)
        br2, bstate, ni = -np.inf, None, 0
        for ep in range(cfg['ep']):
            net.train()
            for z, xo, yy in loader:
                opt.zero_grad()
                loss = nn.MSELoss()(net(z, xo), yy)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Zd_va_t, Xo_va_t).cpu().numpy().squeeze()
            r2v = r2_score(y_va, pv)
            sched.step(1 - r2v)
            if r2v > br2:
                br2 = r2v
                bstate = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                ni = 0
            else:
                ni += 1
            if ni >= 40:
                break
        net.load_state_dict(bstate)
        net.eval()
        with torch.no_grad():
            pt = net(Zd_te_t, Xo_te_t).cpu().numpy().squeeze()
        r2t = r2_score(y_te, np.clip(pt, 0, 1))
        logger.info(f'      wd={cfg["wd"]:.0e} lr={cfg["lr"]:.0e} ep={cfg["ep"]} bs={cfg["bs"]} '
              f'→ 5fold val={cv_mean:.4f}±{cv_std:.4f} test={r2t:.4f}')
        if cv_mean > best_inner_r2:
            best_inner_r2 = cv_mean
            best_cfg = cfg
            best_state = bstate
            best_test_pred = pt.copy()

    logger.info(f'    最佳 cfg (5-fold CV): wd={best_cfg["wd"]:.0e} lr={best_cfg["lr"]:.0e} '
          f'ep={best_cfg["ep"]} bs={best_cfg["bs"]} (CV={best_inner_r2:.4f})')

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_tr = net(
            torch.FloatTensor(Zd_tr_ann).to(DEVICE),
            torch.FloatTensor(Xo_tr_sa).to(DEVICE)
        ).cpu().numpy()
    pred_te = best_test_pred
    r2_train = r2_score(y_tr, np.clip(pred_tr, 0, 1))
    pred_te = np.clip(pred_te, 0, 1)
    logger.info(f'    DualANN train R²: {r2_train:.4f}')
    r2 = r2_score(y_te, pred_te)
    mae = mean_absolute_error(y_te, pred_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred_te))
    try:
        pr = pearsonr(y_te, pred_te)[0]
    except Exception:
        pr = np.nan
    results['PCL-AE-128+DualANN'] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr}
    records.append({
        'model': 'PCL-AE-128+DualANN',
        'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pr,
        'note': f'PCL-AE-{latent_dim} + DualANN (λ={pcl_lambda})',
    })
    logger.info(f'    PCL-AE-128+DualANN: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')

    # ---- 5. 保存测试集预测 ----
    df_pred = pd.DataFrame({
        'idx': test_idx,
        'y_true': y_te,
        'pred_DualANN': pred_te,
        'pred_PCLAE_RF': pred_rf,
    })

    return results, records, df_pred


# =====================================================================
# 内部 CV 参考：与外部评估同一批 7 个模型，在训练池内用 manifest 5 折现算
# =====================================================================
def internal_cv_reference(X_drfp_all, X_xtb_all, y_all):
    """Manifest 5-fold CV on the train pool for every model reported externally.

    Inputs must already be restricted to the train pool (relative indices from
    ``kfold_folds``).  Scalers, PCA and PCL-AE are fitted inside each fold.
    Returns ``{model_name: (mean_r2, std_r2)}``.
    """
    from sklearn.decomposition import PCA
    logger.info('\n  [内部 5-fold CV 参考基准 (manifest folds, 全部现算)]')

    fold_r2 = {}

    def _rec(name, y_true, pred):
        fold_r2.setdefault(name, []).append(r2_score(y_true, np.clip(pred, 0, 1)))

    manifest_folds = kfold_folds(load_manifest())
    if len(y_all) != len(set(i for _, tr_, va_ in manifest_folds for i in list(tr_) + list(va_))):
        raise ValueError('internal CV inputs are not the manifest train pool')
    for fold_id, tr, te in manifest_folds:
        seed_f = SEED + int(fold_id)
        sd = StandardScaler()
        Xd_tr_s = sd.fit_transform(X_drfp_all[tr]).astype(np.float32)
        Xd_te_s = sd.transform(X_drfp_all[te]).astype(np.float32)

        so = StandardScaler()
        Xo_tr_s = so.fit_transform(X_xtb_all[tr]).astype(np.float32)
        Xo_te_s = so.transform(X_xtb_all[te]).astype(np.float32)

        # --- raw DRFP + XTB trees ---
        sc_raw = StandardScaler()
        Xr_tr = sc_raw.fit_transform(np.hstack([X_drfp_all[tr], X_xtb_all[tr]]))
        Xr_te = sc_raw.transform(np.hstack([X_drfp_all[te], X_xtb_all[te]]))
        for name in ('RF', 'XGB', 'LGBM'):
            m_t = get_tree_model(name, seed=seed_f)
            m_t.fit(Xr_tr, y_all[tr])
            _rec(name, y_all[te], m_t.predict(Xr_te))

        # --- PCA-128 + RF/XGB ---
        pca = PCA(n_components=128, random_state=seed_f)
        Xp_tr = pca.fit_transform(Xd_tr_s).astype(np.float32)
        Xp_te = pca.transform(Xd_te_s).astype(np.float32)
        sc_p = StandardScaler()
        Xpp_tr = sc_p.fit_transform(np.hstack([Xp_tr, X_xtb_all[tr]]))
        Xpp_te = sc_p.transform(np.hstack([Xp_te, X_xtb_all[te]]))
        for name in ('RF', 'XGB'):
            m_t = get_tree_model(name, seed=seed_f)
            m_t.fit(Xpp_tr, y_all[tr])
            _rec(f'PCA-128+{name}', y_all[te], m_t.predict(Xpp_te))

        # --- PCL-AE latent ---
        pcl = train_pcl_ae(Xd_tr_s, y_all[tr], latent_dim=128)
        with torch.no_grad():
            Z_tr = pcl.encode(torch.FloatTensor(Xd_tr_s).to(DEVICE)).cpu().numpy()
            Z_te = pcl.encode(torch.FloatTensor(Xd_te_s).to(DEVICE)).cpu().numpy()

        # --- RF on PCL-AE + XTB ---
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.hstack([Z_tr, X_xtb_all[tr]]))
        X_te = scaler.transform(np.hstack([Z_te, X_xtb_all[te]]))
        m = get_tree_model('RF', seed=seed_f)
        m.fit(X_tr, y_all[tr])
        _rec('PCL-AE-128+RF', y_all[te], m.predict(X_te))

        # --- DualANN on PCL-AE latent + XTB ---
        # 501 模式：Z 再 StandardScaler，从训练集再拆 20% val 做 early stopping
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=SEED + 999)
        inner_tr2, inner_va = next(inner_kf.split(Z_tr))
        Z_tr2 = Z_tr[inner_tr2].astype(np.float32)
        Z_va = Z_tr[inner_va].astype(np.float32)
        Xo_tr2 = Xo_tr_s[inner_tr2].astype(np.float32)
        Xo_va = Xo_tr_s[inner_va].astype(np.float32)
        y_tr2 = y_all[tr][inner_tr2].astype(np.float32)
        y_va = y_all[tr][inner_va].astype(np.float32)

        # 与 501 eval_dual_branch_5fold 一致：Z 也走 StandardScaler
        sd_z = StandardScaler()
        Z_tr2 = sd_z.fit_transform(Z_tr2).astype(np.float32)
        Z_va = sd_z.transform(Z_va).astype(np.float32)
        Z_te = sd_z.transform(Z_te).astype(np.float32)

        Z_te_t = torch.FloatTensor(Z_te).to(DEVICE)
        Xo_te_t = torch.FloatTensor(Xo_te_s).to(DEVICE)

        model = DualBranchANN(Z_tr2.shape[1], Xo_tr2.shape[1], hidden=128).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
        Z_tr2_t = torch.FloatTensor(Z_tr2).to(DEVICE)
        Z_va_t = torch.FloatTensor(Z_va).to(DEVICE)
        Xo_tr2_t = torch.FloatTensor(Xo_tr2).to(DEVICE)
        Xo_va_t = torch.FloatTensor(Xo_va).to(DEVICE)
        y_tr2_t = torch.FloatTensor(y_tr2).to(DEVICE)
        loader = DataLoader(TensorDataset(Z_tr2_t, Xo_tr2_t, y_tr2_t), batch_size=64, shuffle=True)

        # Early stopping / LR schedule use the INNER validation split
        # (carved from the training fold); the test fold is predicted once
        # with the best-validation checkpoint.
        best_r2, best_state, no_imp = -np.inf, None, 0
        for ep in range(200):
            model.train()
            for z, xo, yy in loader:
                opt.zero_grad()
                loss = nn.MSELoss()(model(z, xo), yy)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_v = model(Z_va_t, Xo_va_t).cpu().numpy().squeeze()
            r2 = r2_score(y_va, np.clip(pred_v, 0, 1))
            sched.step(1 - r2)
            if r2 > best_r2:
                best_r2 = r2
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 40:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            best_pred = model(Z_te_t, Xo_te_t).cpu().numpy().squeeze()
        _rec('PCL-AE-128+DualANN', y_all[te], best_pred)
        logger.info(f'    fold {fold_id}: done')

    out = {}
    for name, vals in fold_r2.items():
        out[name] = (float(np.mean(vals)), float(np.std(vals)))
        logger.info(f'    {name:<20} 5-fold CV: R²={out[name][0]:.4f}±{out[name][1]:.4f}')
    return out


# =====================================================================
# 主函数
# =====================================================================
def main():
    t0 = time.time()
    logger.info('=' * 72)
    logger.info('  外部测试集验证 (External Validation)')
    logger.info('  JCIM OECD QSPR Principle 4 — External Test Set')
    logger.info('=' * 72)
    logger.info(f'设备: {DEVICE}')
    logger.info(f'随机种子: {SEED}')
    logger.info(f'测试集比例: {HOLDOUT_RATIO:.0%}')

    # 固定随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # [Step 0] 加载数据
    X_drfp, X_xtb, y, df_clean, xtb_cols = load_data(use_holdout_train=False)
    n = len(y)
    logger.info(f'\n总样本: {n}, 产率范围: [{y.min():.3f}, {y.max():.3f}]')

    # [Step 1] 使用 data_split.json 的权威 holdout（绝对索引，不再二次随机切分）
    logger.info('\n[Step 1] 读取 manifest holdout 划分')
    train_idx, test_idx, _ = holdout_arrays(load_manifest())
    train_idx = np.array(sorted(train_idx), dtype=int)
    test_idx = np.array(sorted(test_idx), dtype=int)
    assert not set(train_idx) & set(test_idx), 'manifest train/test overlap'
    n_test = len(test_idx)
    logger.info(f'  训练集: {len(train_idx)} 样本')
    logger.info(f'  测试集: {n_test} 样本 (manifest holdout)')

    # 测试集分布报告
    df_test = df_clean.iloc[test_idx].copy()
    df_test['y'] = y[test_idx]
    df_train = df_clean.iloc[train_idx].copy()
    df_train['y'] = y[train_idx]

    logger.info(f'\n  测试集产率分布:')
    logger.info(f'    均值: {df_test["y"].mean()*100:.1f}% (训练集: {df_train["y"].mean()*100:.1f}%)')
    logger.info(f'    标准差: {df_test["y"].std()*100:.1f}% (训练集: {df_train["y"].std()*100:.1f}%)')
    logger.info(f'    范围: [{df_test["y"].min()*100:.1f}%, {df_test["y"].max()*100:.1f}%]')

    # 催化剂/底物分布对比
    test_cats = set(df_test['catalyst_system_type'].dropna().unique())
    train_cats = set(df_train['catalyst_system_type'].dropna().unique())
    overlap_cats = test_cats & train_cats
    logger.info(f'\n  催化剂类型覆盖:')
    logger.info(f'    训练集: {len(train_cats)} 类 | 测试集: {len(test_cats)} 类 | 重叠: {len(overlap_cats)}')
    logger.info(f'    测试集独有: {test_cats - train_cats}')

    test_subs = set(df_test['reactant_name'].dropna().unique())
    train_subs = set(df_train['reactant_name'].dropna().unique())
    overlap_subs = test_subs & train_subs
    logger.info(f'\n  底物类型覆盖:')
    logger.info(f'    训练集: {len(train_subs)} 种 | 测试集: {len(test_subs)} 种 | 重叠: {len(overlap_subs)}')
    logger.info(f'    测试集独有: {test_subs - train_subs}')

    # [Step 2] 外部验证评估
    logger.info('\n' + '=' * 72)
    logger.info('  [Step 2] 外部测试集评估')
    logger.info('=' * 72)

    results, records, df_pred = evaluate_on_holdout(
        X_drfp, X_xtb, y, train_idx, test_idx,
        latent_dim=128)

    # [Step 3] 内部 CV 参考
    logger.info('\n' + '=' * 72)
    logger.info('  [Step 3] 内部 5-fold CV 参考基准（全部模型现算）')
    logger.info('=' * 72)
    cv_results = internal_cv_reference(X_drfp[train_idx], X_xtb[train_idx], y[train_idx])

    # [Step 4] 保存结果
    logger.info('\n' + '=' * 72)
    logger.info('  [Step 4] 保存结果')
    logger.info('=' * 72)

    # 结果 CSV
    df_results = pd.DataFrame(records)
    df_results['split'] = 'external_holdout'
    df_results['n_train'] = len(train_idx)
    df_results['n_test'] = n_test
    df_results['seed'] = SEED
    out_csv = os.path.join(OUTPUT_DIR, 'external_validation_results.csv')
    df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f'  已保存: {out_csv}')

    # 测试集预测详细
    df_pred_out = df_clean.iloc[test_idx].reset_index(drop=True).copy()
    df_pred_out['y_true'] = y[test_idx]
    df_pred_out['pred_DualANN'] = df_pred['pred_DualANN'].values
    df_pred_out['error'] = df_pred_out['pred_DualANN'] - df_pred_out['y_true']
    pred_csv = os.path.join(OUTPUT_DIR, 'external_test_predictions.csv')
    df_pred_out.to_csv(pred_csv, index=False, encoding='utf-8-sig')
    logger.info(f'  已保存: {pred_csv}')

    # 内部 vs 外部对比表
    logger.info('\n' + '=' * 72)
    logger.info('  内部 CV vs 外部测试集 对比')
    logger.info('=' * 72)
    logger.info(f'  {"模型":<25} {"内部CV R²":>12} {"外部Test R²":>12} {"差值Δ":>8}')
    logger.info(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*8}')

    internal_r2 = {name: mean for name, (mean, _std) in cv_results.items()}
    missing_cv = [m for m in df_results['model'] if m not in internal_r2]
    if missing_cv:
        raise RuntimeError(f'no internal CV computed for: {missing_cv}')

    comparison = []
    for model_name in df_results['model'].values:
        int_r2 = internal_r2.get(model_name, np.nan)
        ext_r2 = df_results[df_results['model'] == model_name]['r2'].values[0]
        delta = ext_r2 - int_r2 if not np.isnan(int_r2) else np.nan
        flag = '★' if abs(delta) < 0.05 else ('▲' if delta > 0.05 else '▼')
        logger.info(f'  {model_name:<25} {int_r2:>12.4f} {ext_r2:>12.4f} {delta:>+8.4f} {flag}')
        comparison.append({
            'model': model_name,
            'internal_cv_r2': int_r2,
            'external_test_r2': ext_r2,
            'delta_r2': delta,
            'overfit_flag': abs(delta) > 0.12,
        })

    df_comp = pd.DataFrame(comparison)
    comp_csv = os.path.join(OUTPUT_DIR, 'external_vs_internal_comparison.csv')
    df_comp.to_csv(comp_csv, index=False, encoding='utf-8-sig')
    logger.info(f'\n  已保存: {comp_csv}')

    # 摘要报告
    logger.info('\n' + '=' * 72)
    logger.info('  摘要')
    logger.info('=' * 72)
    elapsed = time.time() - t0
    best_model = df_results.loc[df_results['r2'].idxmax()]
    logger.info(f'  最佳外部测试集模型: {best_model["model"]}')
    logger.info(f'  外部测试集 R² = {best_model["r2"]:.4f}')
    logger.info(f'  内部 CV R²      = {internal_r2.get(best_model["model"], "N/A")}')
    delta = best_model['r2'] - internal_r2.get(best_model["model"], np.nan)
    if abs(delta) < 0.05:
        verdict = '✓ 模型无明显过拟合'
    elif delta > 0.05:
        verdict = '✓ 外部表现优于内部（可能数据集划分有利）'
    else:
        verdict = '⚠ 外部表现弱于内部，可能存在过拟合'
    logger.info(f'  判定: {verdict}')
    logger.info(f'\n  总耗时: {elapsed/60:.1f} 分钟')

    return df_results, df_comp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='External validation (JCIM OECD QSPR Principle 4).')
    parser.add_argument('--force', action='store_true', help='强制重新运行 (旧版兼容性,仅记录)')
    parser.add_argument('--dry-run', action='store_true', help='仅语法检查, 不执行模型')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.dry_run:
        logger.info('[DRY-RUN] 405_external_validation: 语法检查通过, 无实际计算')
        sys.exit(0)
    main()
    sys.exit(0)
