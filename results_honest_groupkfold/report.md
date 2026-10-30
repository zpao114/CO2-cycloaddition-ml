# GroupKFold HONEST ablation — report

Generated: 2026-10-29 22:11:51
Per-fold AE training epochs: 150
Seeds: [42, 2024, 7, 9998, 7777]
Catalyst group column: `catalyst_system_type` (5 categories)
Split protocol: `GroupKFold(n_splits=5)` with `groups = f"{c}_{seed}"` (byte-for-byte replica of `generate_si_s3_benchmark_full_v3_1.py` §5)

## 1. Why this experiment

The cached GroupKFold benchmark trains the PCL-AE encoder on ALL rows once, then reuses the resulting 128-D latent across every GroupKFold fold.  This is the same caching pattern whose LOSO effect is measured in `pcl_ae_fair_comparison_multiseed.py` (authoritative Delta_protocol).  This script runs the GroupKFold protocol **honestly**: a fresh PCL-AE encoder is trained on the train fold only, for every fold.  The grouping key is the catalyst system type, so this is a second-split check of the cached-vs-honest gap (catalyst-type holdout, not a cross-substrate claim); the gap need not match the LOSO one.

## 2. Per-(seed, model) sweep

| seed | model | n_folds | honest_r2_mean | cached_r2_mean | delta_r2_mean |
|---|---|---|---|---|---|
| 42 | XGB | 5 | +0.2558 | +0.2449 | -0.0109 |
| 42 | LGBM | 5 | +0.1884 | +0.2881 | +0.0997 |
| 42 | RF | 5 | +0.3312 | +0.3730 | +0.0418 |
| 42 | ConcatMLP | 5 | +0.2484 | +0.3154 | +0.0671 |
| 2024 | XGB | 5 | +0.2770 | +0.2443 | -0.0327 |
| 2024 | LGBM | 5 | +0.2547 | +0.2881 | +0.0334 |
| 2024 | RF | 5 | +0.3534 | +0.3713 | +0.0178 |
| 2024 | ConcatMLP | 5 | +0.2923 | +0.3175 | +0.0252 |
| 7 | XGB | 5 | +0.2776 | +0.2570 | -0.0206 |
| 7 | LGBM | 5 | +0.2495 | +0.2881 | +0.0386 |
| 7 | RF | 5 | +0.3424 | +0.3672 | +0.0249 |
| 7 | ConcatMLP | 5 | +0.2445 | +0.2827 | +0.0382 |
| 9998 | XGB | 5 | +0.2656 | +0.2031 | -0.0626 |
| 9998 | LGBM | 5 | +0.1602 | +0.2881 | +0.1279 |
| 9998 | RF | 5 | +0.3272 | +0.3672 | +0.0399 |
| 9998 | ConcatMLP | 5 | +0.0922 | +0.3283 | +0.2361 |
| 7777 | XGB | 5 | +0.2171 | +0.2332 | +0.0161 |
| 7777 | LGBM | 5 | +0.1089 | +0.2881 | +0.1792 |
| 7777 | RF | 5 | +0.2966 | +0.3722 | +0.0756 |
| 7777 | ConcatMLP | 5 | +0.1034 | +0.3227 | +0.2193 |

## 3. N-seed mean comparison (headline)

| model | n_seeds | honest_r2_mean | honest_r2_std | cached_r2_mean | cached_r2_std | delta_r2_mean |
|---|---|---|---|---|---|---|
| XGB | 5 | +0.2586 ± 0.0249 | +0.2365 ± 0.0205 | -0.0221 |
| LGBM | 5 | +0.1923 ± 0.0616 | +0.2881 ± 0.0000 | +0.0958 |
| RF | 5 | +0.3302 ± 0.0214 | +0.3702 ± 0.0028 | +0.0400 |
| ConcatMLP | 5 | +0.1961 ± 0.0918 | +0.3133 ± 0.0178 | +0.1172 |

## 4. Per-fold (catalyst-held-out) diagnostics

| held-out catalysts | n(seed×fold) | honest R² (mean ± std) | cached R² (mean ± std) | Δ R² (cached − honest) |
|---|---|---|---|---|
| ionic_liquid | 20 | +0.2284 ± 0.0520 | +0.1544 ± 0.1189 | -0.0740 |
| metal_halide | 20 | +0.3020 ± 0.0480 | +0.3733 ± 0.0275 | +0.0713 |
| mixed_system | 20 | +0.1958 ± 0.1568 | +0.4052 ± 0.1761 | +0.2094 |
| organic_base | 20 | +0.3302 ± 0.1607 | +0.3178 ± 0.1240 | -0.0124 |
| unknown | 20 | +0.1651 ± 0.0850 | +0.2594 ± 0.0595 | +0.0943 |

## 5. Interpretation

**PAIRED comparison.** Every honest/cached R² in §3 is on the **same fold with the same seed**: the only difference between the two columns is whether the PCL-AE encoder saw all rows (cached) or only the train fold (honest).  The Δ column is therefore a per-fold leakage measurement, not a mean-of-means comparison.

**Note on seed convention.** This script uses seeds = [42, 2024, 7, 9998, 7777], aligned with `pcl_ae_fair_comparison_multiseed.py`.  The cached and honest arms share the same seeds and folds, so the **paired** gap is the quantity to quote.  GroupKFold by catalyst type is deterministic: the 5 folds are identical for every seed, so seeds vary only encoder / regressor initialisation, not the split.

- **XGB headline**: cached R² = +0.2365, honest R² = +0.2586, paired gap (cached − honest) = -0.0221.
- **Paired Δ (cached − honest; positive = overestimate, same sign as Δprotocol) per model**: XGB -0.022, LGBM +0.096, RF +0.040, ConcatMLP +0.117.
- The gap is model-dependent: cached R² exceeds honest R² for 3/4 models and not for the others. Quote the per-model values; do not describe GroupKFold as a uniform replication of the LOSO inflation.

## 6. Reproducibility

- Data: `results/results_cho_diagnostic/co2_drfp_xtb_extended.csv`
- Cached latent: `results_pcl_ae/pcl_ae_latent.npy`
- Cached baseline CSV: `results/results_si/groupkfold_v3_full_results.csv`
- This script: `src/analysis/honest_groupkfold_ablation.py`
- Per-fold AE: `PropertyCoLearningAE` from `eval_protocol/nets.py`
- λ_prop = 0.2, latent_dim = 128
