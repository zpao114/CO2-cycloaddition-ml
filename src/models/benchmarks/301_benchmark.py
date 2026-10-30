# -*- coding: utf-8 -*-
"""
301 -- Full benchmark pipeline on top of the best DRFP variant.

Reads:
    results_best_pipeline/drfp_ablation_meta.json

Writes:
    results_best_pipeline/full_benchmark_results.csv

Workflow:
    Part A  DRFP dimensional reduction (raw / pca / ae / pcl) x multiple
            downstream feature sets x RF / XGB / LGBM / DualANN
    Part B  XTB-only baseline (sanity check on electronic descriptors)
    Part C  XGB + LGBM + DualANN ensemble on the best feature set

Protocols (--protocol):
    honest           (default) DRFP reduction (scaler/PCA/AE/PCL-AE) fitted on the
                     TRAINING rows of every fold only
    transductive_x   reduction sees every row's X but no test labels
    pretrain_frozen  reduction sees every row's X and y (legacy PCL-AE protocol)
    presplit         legacy code path (whole-table reduction before the split);
                     kept as a control for the Delta_protocol comparison

Usage:
    python 301_benchmark.py [--protocol honest|transductive_x|pretrain_frozen|presplit]
    (201_ablation.py must have run first to produce the meta file)
"""
import os
import sys
import io
import time
import json
import warnings

if sys.stdout.encoding and sys.stdout.encoding.lower().replace('-', '') != 'utf8':
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:
        pass

log_fh = open("results/results_cho_diagnostic/301_rerun_internal.log", "w", encoding="utf-8", buffering=1)
_orig_print = print


def _tee_print(*args, **kwargs):
    end = kwargs.pop("end", "\n")
    sep = kwargs.pop("sep", " ")
    msg = sep.join(str(a) for a in args) + end
    _orig_print(msg, end="")
    log_fh.write(msg)
    log_fh.flush()


print = _tee_print

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from eval_protocol.models import fit_predict_dual  # noqa: E402  (inner-val early stopping)
from eval_protocol.benchmark import (FoldReducer, cv_tree, cv_dual, cv_ensemble,  # noqa: E402
                                     part_a_rows, DUAL_HP_DEFAULT)

from utils_benchmark import (
    parse_drfp_col, load_data, eval_sklearn_model, eval_dual_branch,
    DualBranchANN, X_inter_base, DEVICE, OUTPUT_DIR,
    train_standard_ae, train_pcl_ae, get_tree_model, BEST_LAMBDA_PROP,
)
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold  # noqa: F401  (kept for legacy callers; canonical 5-fold CV uses data_split.json)
from src.data_split import load_manifest, kfold_folds, split_iterator  # noqa: F401
from scipy.stats import pearsonr


def load_best_variant_from_meta():
    """Read drfp_ablation_meta.json; raise informative error if missing."""
    meta_path = os.path.join(OUTPUT_DIR, 'drfp_ablation_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'Drfp ablation meta file not found: {meta_path}\n'
            f'Run 201_ablation.py first to generate it.'
        )
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta['best_variant'], meta['best_label'], meta['best_r2']


def _parse_protocol():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--protocol', default='honest',
                    choices=['honest', 'transductive_x', 'pretrain_frozen', 'presplit'])
    args, _ = ap.parse_known_args()
    return args.protocol


DUAL_HP = DUAL_HP_DEFAULT


TREES_ALL = ['RF', 'XGB', 'LGBM']


def main():
    t0 = time.time()
    PROTOCOL = _parse_protocol()
    print(f'  >> 评估协议 protocol = {PROTOCOL}  |  lambda_prop = {float(BEST_LAMBDA_PROP)}')
    print('=' * 72)
    print('  301 -- CO2 环加成主基准流水线')
    print('  PCA 128/256 + PCL-AE + 全特征组合')
    print('=' * 72)
    print(f'设备: {DEVICE}  |  CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Read best DRFP variant from ablation meta (201 must run first)
    best_variant, best_label, best_ablation_r2 = load_best_variant_from_meta()
    print(f'\n  >> 下游实验使用最优 DRFP 变体: {best_variant} ({best_label}), ablation R^2={best_ablation_r2:.4f}')

    X_drfp, X_xtb, X_cond, X_cat, X_inter, y, df, _groups, _xtb_cols, _drfp_var = load_data(best_variant)
    n = len(y)
    print(f'\n总样本数: {n}, y range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}')
    print(f'  DRFP 变体: {_drfp_var}, {X_drfp.shape[1]}D')


    DRFP_REDUCE_METHODS = [
        ('raw',    None, 'DRFP 原始'),
        ('pca128', 128,  'PCA-128'),
        ('pca256', 256,  'PCA-256'),
        ('ae128',  128,  'AE-128'),
        ('ae256',  256,  'AE-256'),
        ('pcl128', 128,  'PCL-AE-128'),
        ('pcl256', 256,  'PCL-AE-256'),
    ]

    FEATURE_SETS = [
        ('drfp',           lambda mX: mX),
        ('xtb',            None),  # placeholder
        ('cond',           None),
        ('xtb_cond_inter', None),
        ('drfp_xtb',       None),
        ('drfp_xtb_cond',  None),
        ('drfp_xtb_cond_inter',  None),
    ]

    X_xtb_cond = np.hstack([X_xtb, X_cond]).astype(np.float32)
    X_xtb_cond_inter = np.hstack([X_xtb_cond, X_inter]).astype(np.float32)
    featset_mat = {
        'drfp':          X_drfp,
        'xtb':           X_xtb,
        'cond':          X_cond,
        'xtb_cond_inter': X_xtb_cond_inter,
        'drfp_xtb':      np.hstack([X_drfp, X_xtb]).astype(np.float32),
        'drfp_xtb_cond': np.hstack([X_drfp, X_xtb_cond]).astype(np.float32),
        'drfp_xtb_cond_inter': np.hstack([X_drfp, X_xtb_cond_inter]).astype(np.float32),
    }

    all_results = []
    best_r2 = -np.inf
    best_tag = ''
    folds_cv = kfold_folds(load_manifest())

    TREES = ['RF', 'XGB', 'LGBM']

    # ============================================================
    # Part A：DRFP 降维后再拼接 XTB / Cond ...
    # ============================================================
    print('\n' + '=' * 72)
    print('  Part A: DRFP 降维 (raw/pca/ae/pcl) × 多种下游特征集 × 多模型')
    print('=' * 72)

    for method, dim, label in DRFP_REDUCE_METHODS:
        if PROTOCOL != 'presplit':
            for row, tag in part_a_rows(method, label, X_drfp, X_xtb, X_xtb_cond,
                                        X_xtb_cond_inter, y, folds_cv, PROTOCOL,
                                        lam=float(BEST_LAMBDA_PROP), device=str(DEVICE)):
                all_results.append(row)
                if row['r2_mean'] > best_r2:
                    best_r2, best_tag = row['r2_mean'], tag
            continue
        if method == 'raw':
            X_drfp_red = X_drfp
        else:
            scaler_drfp = StandardScaler()
            Xd_s = scaler_drfp.fit_transform(X_drfp).astype(np.float32)
            if method == 'pca128':
                pca = PCA(n_components=128, random_state=42)
                X_drfp_red = pca.fit_transform(Xd_s).astype(np.float32)
            elif method == 'pca256':
                pca = PCA(n_components=256, random_state=42)
                X_drfp_red = pca.fit_transform(Xd_s).astype(np.float32)
            elif method == 'ae128':
                X_drfp_red = train_standard_ae(Xd_s, 128).astype(np.float32)
            elif method == 'ae256':
                X_drfp_red = train_standard_ae(Xd_s, 256).astype(np.float32)
            elif method == 'pcl128':
                X_drfp_red = train_pcl_ae(Xd_s, y, 128).astype(np.float32)
            elif method == 'pcl256':
                X_drfp_red = train_pcl_ae(Xd_s, y, 256).astype(np.float32)
            else:
                X_drfp_red = Xd_s

        suffixed_sets = {
            f'drfp_{label}':       X_drfp_red,
            f'drfp_{label}_xtb':   np.hstack([X_drfp_red, X_xtb]).astype(np.float32),
            f'drfp_{label}_xtbc':  np.hstack([X_drfp_red, X_xtb_cond]).astype(np.float32),
            f'drfp_{label}_full':  np.hstack([X_drfp_red, X_xtb_cond_inter]).astype(np.float32),
        }

        for fsname, Xfull in suffixed_sets.items():
            print(f'\n[{fsname}]  shape={Xfull.shape}')
            for mname in TREES:
                t1 = time.time()
                metrics = eval_sklearn_model(mname, Xfull, y, n_folds=5, n_repeats=3)
                elapsed = time.time() - t1
                tag = f'{label} | {fsname} | {mname}'
                all_results.append({
                    'stage': 'A_DRFP_reduce',
                    'drfp_method': label,
                    'feature_set': fsname,
                    'model': mname,
                    **{k: metrics[k] for k in ['r2_mean', 'r2_std', 'mae_mean', 'rmse_mean', 'pearson_mean']},
                    'feature_dim': Xfull.shape[1],
                    'time_s': elapsed,
                })
                print(f'  {mname:5s}  R²={metrics["r2_mean"]:.4f}±{metrics["r2_std"]:.4f}  '
                      f'MAE={metrics["mae_mean"]:.4f}  ({elapsed:.0f}s)')
                if metrics['r2_mean'] > best_r2:
                    best_r2 = metrics['r2_mean']
                    best_tag = tag

            if Xfull.shape[1] >= 100 and method in ('raw', 'pca128', 'pca256', 'pcl128', 'pcl256'):
                t1 = time.time()
                if 'drfp_' + label + '_xtbc' == fsname or fsname.endswith('_xtbc'):
                    n_drfp = X_drfp_red.shape[1]
                    X_drfp_in = Xfull[:, :n_drfp]
                    X_other = Xfull[:, n_drfp:]
                elif fsname.endswith('_full'):
                    n_drfp = X_drfp_red.shape[1]
                    X_drfp_in = Xfull[:, :n_drfp]
                    X_other = Xfull[:, n_drfp:]
                else:
                    continue  # skip dual branch for drfp-only

                metrics = eval_dual_branch(X_drfp_in, X_other, y, n_folds=5, n_repeats=2)
                elapsed = time.time() - t1
                tag = f'{label} | {fsname} | DualANN'
                all_results.append({
                    'stage': 'A_DRFP_reduce',
                    'drfp_method': label,
                    'feature_set': fsname,
                    'model': 'DualANN',
                    **{k: metrics[k] for k in ['r2_mean', 'r2_std', 'mae_mean', 'rmse_mean', 'pearson_mean']},
                    'feature_dim': Xfull.shape[1],
                    'time_s': elapsed,
                })
                print(f'  DualANN  R²={metrics["r2_mean"]:.4f}±{metrics["r2_std"]:.4f}  '
                      f'MAE={metrics["mae_mean"]:.4f}  ({elapsed:.0f}s)')
                if metrics['r2_mean'] > best_r2:
                    best_r2 = metrics['r2_mean']
                    best_tag = tag

    # ============================================================
    # Part B：不含 DRFP 的纯 XTB / Cond 组合（基线）
    # ============================================================
    print('\n' + '=' * 72)
    print('  Part B: 不含 DRFP 的 XTB/Cond 基线（验证 XTB 自身能力）')
    print('=' * 72)

    for fsname, Xfull in [('xtb', featset_mat['xtb']),
                           ('cond', featset_mat['cond']),
                           ('xtb_cond_inter', featset_mat['xtb_cond_inter'])]:
        print(f'\n[{fsname}]  shape={Xfull.shape}')
        for mname in TREES:
            t1 = time.time()
            metrics = eval_sklearn_model(mname, Xfull, y, n_folds=5, n_repeats=3)
            elapsed = time.time() - t1
            tag = f'PartB | {fsname} | {mname}'
            all_results.append({
                'stage': 'B_no_drfp',
                'drfp_method': 'NONE',
                'feature_set': fsname,
                'model': mname,
                **{k: metrics[k] for k in ['r2_mean', 'r2_std', 'mae_mean', 'rmse_mean', 'pearson_mean']},
                'feature_dim': Xfull.shape[1],
                'time_s': elapsed,
            })
            print(f'  {mname:5s}  R²={metrics["r2_mean"]:.4f}±{metrics["r2_std"]:.4f}  '
                  f'MAE={metrics["mae_mean"]:.4f}  ({elapsed:.0f}s)')
            if metrics['r2_mean'] > best_r2:
                best_r2 = metrics['r2_mean']
                best_tag = tag

    # ============================================================
    # Part C: Ensemble (XGB + LGBM + DualANN 等权平均)
    # 与 Part A/B 一致：固定 manifest 的 5 折 × 3 个模型种子重复（折固定，仅重设模型种子，
    # 并非 15 个独立折），
    # 最终 R² 取每折 pooled R² 的 mean/std，使 Part C 与 A/B 统计可比。
    # ============================================================
    print('\n' + '=' * 72)
    print('  Part C: Ensemble (XGB + LGBM + DualANN 等权平均, 3 repeats × 5 folds)')
    print('=' * 72)

    N_REPEATS_ENS = 3
    N_SPLITS_ENS = 5

    for fsname in ['PCA-256 DRFP + XTB + Cond + Inter',
                   'PCA-128 DRFP + XTB + Cond + Inter',
                   'DRFP + XTB + Cond + Inter']:
        if PROTOCOL != 'presplit':
            _m = 'pca256' if 'PCA-256' in fsname else ('pca128' if 'PCA-128' in fsname else 'raw')
            red = FoldReducer(X_drfp, y, folds_cv, method=_m, scope=PROTOCOL,
                              lam=float(BEST_LAMBDA_PROP), device=str(DEVICE))
            met = cv_ensemble(red, X_xtb_cond_inter, y, n_repeats=N_REPEATS_ENS, seed0=0,
                              device=str(DEVICE), **DUAL_HP)
            print(f'\n[{fsname}]  protocol={PROTOCOL}')
            print(f'  Ensemble (mean of {N_REPEATS_ENS} pooled) R2 = {met["r2_mean"]:.4f} '
                  f'+- {met["r2_std"]:.4f}  MAE={met["mae_mean"]:.4f}  '
                  f'RMSE={met["rmse_mean"]:.4f}  Pearson={met["pearson_mean"]:.4f}')
            all_results.append({
                'stage': 'C_ensemble', 'drfp_method': fsname, 'feature_set': 'ensemble',
                'model': 'Ensemble',
                **{k: met[k] for k in ['r2_mean', 'r2_std', 'mae_mean', 'rmse_mean', 'pearson_mean']},
                'feature_dim': red.dim() + X_xtb_cond_inter.shape[1], 'time_s': 0.0,
            })
            if met['r2_mean'] > best_r2:
                best_r2, best_tag = met['r2_mean'], f'Ensemble on {fsname}'
            continue
        # DRFP 处理 (legacy pre-split path)
        if 'PCA-256' in fsname:
            Xd_red = PCA(n_components=256, random_state=42).fit_transform(
                StandardScaler().fit_transform(X_drfp)
            ).astype(np.float32)
        elif 'PCA-128' in fsname:
            Xd_red = PCA(n_components=128, random_state=42).fit_transform(
                StandardScaler().fit_transform(X_drfp)
            ).astype(np.float32)
        else:
            Xd_red = X_drfp.astype(np.float32)

        Xfull = np.hstack([Xd_red, X_xtb_cond_inter]).astype(np.float32)
        n = len(y)

        rep_records = []
        # Use the canonical 5-fold split from data_split.json for OOF ensemble
        ens_folds = kfold_folds(load_manifest())
        for rep in range(N_REPEATS_ENS):
            oof = np.zeros(n, dtype=np.float32)
            oof_count = np.zeros(n, dtype=np.float32)
            for _fold_i, (tr, te) in enumerate([(tr, te) for _, tr, te in ens_folds]):
                sd = StandardScaler(); so = StandardScaler()
                Xd_tr = sd.fit_transform(Xd_red[tr])
                Xd_te = sd.transform(Xd_red[te])
                Xo_tr = so.fit_transform(X_xtb_cond_inter[tr])
                Xo_te = so.transform(X_xtb_cond_inter[te])
                Xf_tr = np.hstack([Xd_tr, Xo_tr])
                Xf_te = np.hstack([Xd_te, Xo_te])

                xgbm = get_tree_model('XGB', seed=1000 * rep + _fold_i)
                xgbm.fit(Xf_tr, y[tr])
                pred_xgb = np.clip(xgbm.predict(Xf_te), 0, 1)

                lgbm = get_tree_model('LGBM', seed=1000 * rep + _fold_i)
                lgbm.fit(Xf_tr, y[tr])
                pred_lgb = np.clip(lgbm.predict(Xf_te), 0, 1)

                # DualANN: early stopping on an inner split of the TRAIN fold only
                pred_ann = np.clip(fit_predict_dual(
                    Xd_tr, Xo_tr, y[tr], Xd_te, Xo_te, seed=1000 * rep + _fold_i,
                    hidden=128, lr=5e-4, weight_decay=1e-3, batch_size=32,
                    max_epochs=250, patience=40, lr_patience=20, device=str(DEVICE)), 0, 1)

                oof[te] += (pred_xgb + pred_lgb + pred_ann) / 3.0
                oof_count[te] += 1

            mask = oof_count > 0
            oof_avg = np.where(mask, oof / np.maximum(oof_count, 1), 0)
            pooled_r2 = r2_score(y[mask], oof_avg[mask])
            pooled_mae = mean_absolute_error(y[mask], oof_avg[mask])
            pooled_rmse = float(np.sqrt(mean_squared_error(y[mask], oof_avg[mask])))
            try:
                pooled_pr = float(pearsonr(y[mask], oof_avg[mask])[0])
            except Exception:
                pooled_pr = np.nan
            rep_records.append({
                'rep': rep,
                'r2': float(pooled_r2),
                'mae': float(pooled_mae),
                'rmse': float(pooled_rmse),
                'pearson': float(pooled_pr),
            })

        fm = pd.DataFrame(rep_records)
        mean_r2 = float(fm['r2'].mean())
        std_r2 = float(fm['r2'].std())
        mean_mae = float(fm['mae'].mean())
        mean_rmse = float(fm['rmse'].mean())
        mean_pr = float(fm['pearson'].mean())

        print(f'\n[{fsname}]')
        print(f'  Ensemble (mean of 3 pooled) R² = {mean_r2:.4f} ± {std_r2:.4f}  '
              f'MAE={mean_mae:.4f}  RMSE={mean_rmse:.4f}  Pearson={mean_pr:.4f}')

        all_results.append({
            'stage': 'C_ensemble',
            'drfp_method': fsname,
            'feature_set': 'ensemble',
            'model': 'Ensemble',
            'r2_mean': mean_r2,
            'r2_std': std_r2,
            'mae_mean': mean_mae,
            'rmse_mean': mean_rmse,
            'pearson_mean': mean_pr,
            'feature_dim': Xfull.shape[1],
            'time_s': 0.0,
        })
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_tag = f'Ensemble on {fsname}'

    # ============================================================
    # Part D：DRFP 消融实验（HeckLit 风格 Table 1）
    # 每个变体独立训练，输出 R²/RMSE/MAE，量化各组分的贡献
    # ============================================================
    df_all = pd.DataFrame(all_results).sort_values('r2_mean', ascending=False)
    df_all.insert(0, 'protocol', PROTOCOL)
    df_all['lambda_prop'] = float(BEST_LAMBDA_PROP)
    # honest -> canonical file; every other protocol gets its own suffixed file
    _name = 'full_benchmark_results.csv' if PROTOCOL == 'honest' \
        else f'full_benchmark_results_{PROTOCOL}.csv'
    out_csv = os.path.join(OUTPUT_DIR, _name)
    df_all.to_csv(out_csv, index=False, encoding='utf-8-sig')

    elapsed_total = time.time() - t0
    print('\n' + '=' * 72)
    print('  ★ TOP 20 实验 ★')
    print('=' * 72)
    cols_show = ['stage', 'drfp_method', 'feature_set', 'model',
                 'r2_mean', 'r2_std', 'mae_mean', 'rmse_mean', 'pearson_mean']
    print(df_all.head(20)[cols_show].to_string(index=False))

    print(f'\n★ 最佳: {best_tag}, R² = {best_r2:.4f}')
    print(f'汇总已保存到: {out_csv}')
    print(f'总耗时: {elapsed_total/60:.1f} 分钟')

    return df_all

if __name__ == '__main__':
    main()
