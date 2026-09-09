# GroupKFold HONEST ablation — report

Generated: 2026-09-01 15:01:57
Per-fold AE training epochs: 80
Seeds: [42, 2024, 7, 9998, 7777]
Catalyst group column: `catalyst_system_type` (5 categories)
Split protocol: `GroupKFold(n_splits=5)` with `groups = f"{c}_{seed}"` (byte-for-byte replica of `generate_si_s3_benchmark_full_v3_1.py` §5)

## 1. Why this experiment

The paper's §3.1 GroupKFold R² = +0.388 (XGB, F2 PCL-AE) was produced by training the PCL-AE encoder on ALL 2,490 rows once, then reusing the resulting 128-D latent across every GroupKFold fold.  This is the same caching pattern that inflates the LOSO headline by ~+1.07 R² units (see `pcl_ae_fair_comparison.py`).  This script runs the GroupKFold protocol **honestly**: a fresh PCL-AE encoder is trained on the train fold only, for every fold.

## 2. Per-(seed, model) sweep

| seed | model | n_folds | honest_r2_mean | cached_r2_mean | delta_r2_mean |
|---|---|---|---|---|---|
| 42 | XGB | 5 | +0.0081 | +0.4562 | -0.4481 |
| 42 | LGBM | 5 | -0.0156 | +0.4505 | -0.4661 |
| 42 | RF | 5 | +0.0158 | +0.4948 | -0.4790 |
| 42 | DualBranchANN | 5 | -0.0367 | +0.4401 | -0.4768 |
| 2024 | XGB | 5 | -0.0425 | +0.4508 | -0.4933 |
| 2024 | LGBM | 5 | -0.0542 | +0.4505 | -0.5046 |
| 2024 | RF | 5 | -0.0152 | +0.4958 | -0.5110 |
| 2024 | DualBranchANN | 5 | -0.0583 | +0.4353 | -0.4936 |
| 7 | XGB | 5 | +0.0132 | +0.4469 | -0.4338 |
| 7 | LGBM | 5 | -0.0123 | +0.4505 | -0.4628 |
| 7 | RF | 5 | +0.0104 | +0.4959 | -0.4854 |
| 7 | DualBranchANN | 5 | -0.0372 | +0.4332 | -0.4705 |
| 9998 | XGB | 5 | -0.0003 | +0.4545 | -0.4548 |
| 9998 | LGBM | 5 | -0.0023 | +0.4505 | -0.4528 |
| 9998 | RF | 5 | +0.0151 | +0.4960 | -0.4809 |
| 9998 | DualBranchANN | 5 | -0.0471 | +0.4318 | -0.4790 |
| 7777 | XGB | 5 | +0.0166 | +0.4661 | -0.4495 |
| 7777 | LGBM | 5 | -0.0135 | +0.4505 | -0.4639 |
| 7777 | RF | 5 | +0.0221 | +0.4936 | -0.4715 |
| 7777 | DualBranchANN | 5 | -0.0429 | +0.4461 | -0.4890 |

## 3. N-seed mean comparison (headline)

| model | n_seeds | honest_r2_mean | honest_r2_std | cached_r2_mean | cached_r2_std | delta_r2_mean |
|---|---|---|---|---|---|---|
| XGB | 5 | -0.0010 ± 0.0241 | +0.4549 ± 0.0072 | -0.4559 |
| LGBM | 5 | -0.0196 ± 0.0200 | +0.4505 ± 0.0000 | -0.4701 |
| RF | 5 | +0.0096 ± 0.0145 | +0.4952 ± 0.0010 | -0.4856 |
| DualBranchANN | 5 | -0.0445 ± 0.0089 | +0.4373 ± 0.0058 | -0.4818 |

## 4. Per-fold (catalyst-held-out) diagnostics

| held-out catalysts | n(seed×fold) | honest R² (mean ± std) | cached R² (mean ± std) | Δ R² |
|---|---|---|---|---|
| ionic_liquid | 20 | +0.1307 ± 0.0440 | +0.3355 ± 0.0273 | -0.2048 |
| metal_halide | 20 | +0.3706 ± 0.0301 | +0.6105 ± 0.0299 | -0.2399 |
| mixed_system | 20 | +0.0277 ± 0.0494 | +0.4688 ± 0.0168 | -0.4412 |
| organic_base | 20 | -0.4220 ± 0.0554 | +0.2422 ± 0.0639 | -0.6642 |
| unknown | 20 | -0.1762 ± 0.0378 | +0.6404 ± 0.0130 | -0.8165 |

## 5. Interpretation

**PAIRED comparison.** Every honest/cached R² in §3 is on the **same fold with the same seed**: the only difference between the two columns is whether the PCL-AE encoder saw all 2,490 rows (cached) or only the train fold (honest).  The Δ column is therefore a per-fold leakage measurement, not a mean-of-means comparison.

**Note on seed convention.** This script uses seeds = (42, 2024, 7) to align with the LOSO honest script (`pcl_ae_fair_comparison.py`). The cached GroupKFold CSV from §3.1 (`groupkfold_v3_full_results.csv`) uses seeds = (0, 1, 2). The cached 3-seed mean XGB R² therefore differs slightly between this script (+0.4513) and the paper §3.1 (+0.3883); the **paired** leakage Δ is invariant to the seed choice.

- **XGB headline**: cached R² = +0.4549, honest R² = -0.0010, paired leakage Δ = -0.4559.
- GroupKFold leakage is **severe**: honest R² (-0.001) collapses to near zero. The §3.1 cached number **cannot be defended** as an honest out-of-fold R²; the framework's diagnostic value (catalyst-stratified transferability, SHAP direction, §3.2 LOSO root-cause) becomes the primary story. The paper should explicitly downgrade §3.1 from 'GroupKFold R² = 0.388' to 'GroupKFold R² ≤ 0.07 (honest)', and frame §3.1 as the *cached latent reference* used to power the SHAP diagnostics, not as an honest generalisation number.

## 6. Reproducibility

- Data: `results/results_cho_diagnostic/co2_drfp_xtb_extended.csv`
- Cached latent: `results_pcl_ae/pcl_ae_latent.npy`
- Cached baseline CSV: `results/results_si/groupkfold_v3_full_results.csv`
- This script: `src/analysis/honest_groupkfold_ablation.py`
- Per-fold AE: `PropertyCoLearningAE` from `utils_benchmark.py`
- λ_prop = 200.0, latent_dim = 128
