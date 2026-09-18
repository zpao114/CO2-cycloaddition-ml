# CO2 Cycloaddition ML Pipeline

This project is the code and data for a machine-learning study of catalyst
screening for CO2 / epoxide cycloaddition. It includes the cleaned dataset
and the code for dataset processing & analysis, model training & testing,
and mechanistic diagnostics. The work is in preparation for an ACS journal
submission.

The project is implemented in Python and uses both CPU and GPU
computation. The required Python version and Python libraries are listed
in [`requirements.txt`](requirements.txt). GFN2-xTB descriptors additionally
require the `xtb` CLI (installed via conda-forge); ORCA is required only
for the optional DFT validation set.

---

## Project structure

The project consists of 4 first-level directories: `src`, `configs`,
`data`, `figures`, and `dft_validation`.  Additional directories
`results_*` store analysis artefacts; all of them are gitignored except
the 5 that are explicitly referenced in the paper.

| Directory | Description |
|-----------|-------------|
| `src/` | Source code, organized by responsibility (data, models, analysis, dft, visualization, ci_artifacts, helpers) |
| `configs/` | Tuned hyperparameters, data registry, model artifact registry (templates only) |
| `data/` | Cleaned and processed data (CSVs) |
| `figures/` | Paper figures (PDF + PNG), English version only |
| `dft_validation/` | ORCA / xTB validation set: KL divergence summary + raw ORCA / xTB outputs (SI §S7.8 documents the `mult = 2` setup caveat on the five-substrate B3LYP-D3BJ/def2-TZVP single-point energies) |
| `results_honest_groupkfold/` | Honest GroupKFold paired-seed audit (§3.1) |
| `results_external_holdout/` | External-holdout multiseed results (§3.2) |
| `results_pcl_ae_fair_comparison_multiseed/` | PCL-AE fair-comparison multiseed (§3.3) |
| `results_substrate_catalyst_matrix/` | Per-substrate/catalyst-class/yield-bin decomposition (§3.3) |
| `results_permuted_ablation/` | Permutation feature ablation (§3.4) |
| `tests/` | Minimal placeholder |

All remaining `results_*/` directories are gitignored.

---

## src/

The source code is organized by responsibility, not by execution order.
Numbered prefixes (`101_`, `201_`, `301_`, ..., `902_`) indicate the
recommended execution order and are documented in
[`scripts/RUN_ORDER.md`](scripts/RUN_ORDER.md).

### src/data/ — data processing

| File | Purpose |
|------|---------|
| `101_clean.py` | Cleaning of Reaxys-derived reaction table (canonical SMILES, drop NA, dedupe) |
| `102_smiles.py` | Reaction SMILES construction (`reactants>agents>products`) |
| `103_drfp.py` | DRFP (Differential Reaction Fingerprint) encoding, 2048-bit |
| `104b_run_xtb_extended.py` | GFN2-xTB descriptor generation (HOMO, LUMO, charges, dipole) for substrates, catalysts, solvents |
| `105b_xtb_sanity_v2.py` | Sanity checks for xTB outputs (convergence, charge sanity, energy ranges) |
| `107_merge_substrate_xtb.py` | Merge xTB descriptors into the reaction feature table |
| `201_ablation.py` | DRFP / xTB feature-set ablation |
| `201_drfp_ablation.py` | DRFP-only ablation (2048 / 1024 / 512 / 256 bits) |
| `802_pcl_ae_visualization.py` | PCA / t-SNE view of the PCL-AE latent space |

### src/models/ — model training & benchmarks

| Subdirectory | Contents / Purpose |
|--------------|---------------------|
| `benchmarks/` | 5-fold, GroupKFold, sample-size sensitivity, statistical tests, external holdout, year-OOD, control experiments, ablations |
| `persistence/` | Train and persist the PCL-AE encoder and the best pipeline |
| `screening/` | Virtual screening and ranking metrics (top-k recall, NDCG, MRR) |

Key files inside `benchmarks/`: `301_benchmark.py`, `302_groupkfold_validation.py`,
`303_sample_size_sensitivity.py`, `304_statistical_significance.py`,
`306_external_validation.py`, `310_known_top10_baseline.py`,
`801_drfp_ablation_deep_analysis.py`, `803_mordred_ablation.py`,
`804_hierarchical_catalyst_model.py`, `807_cross_model_shap.py`,
`808_catalyst_control.py`, `809_condition_control.py`,
`810_yield_distribution.py`, `_shap_infra.py`.

Key files inside `persistence/`: `401_persist_best_pipeline.py`,
`405_external_validation.py`, `train_pcl_ae.py`.

Key files inside `screening/`: `403b_ranking_metrics.py`.

### src/analysis/ — diagnostic analyses

| Subdirectory | Contents / Purpose |
|--------------|---------------------|
| `loso/` | Leave-one-substrate-out (LOSO) / leave-one-mechanism-out (LOMO) cross-validation and per-substrate SHAP attribution |
| `mechanism/` | Catalyst-mechanism classification, substrate-feature table, transferability matrix |
| `diagnostics/` | Paper-ready figures, substrate × catalyst coverage matrix, CHO mechanistic diagnostic |

Key files inside `loso/`: `700_loso_lomo_cv.py`, `701_per_substrate_shap.py`,
`702_integrated_report.py`, `705_improved_loso.py`,
`705b_substrate_aware_loso.py`, `705c_statistical_loso.py`,
`705d_loso_analysis.py`, `705e_fine_grained_loso.py`,
`706_loso_root_cause_figure.py`.

Key files inside `mechanism/`: `601_catalyst_mechanism_v2.py`,
`602_substrate_features.py`, `603_transferability_matrix.py`.

Key files inside `diagnostics/`: `900_paper_abstract.py`,
`901_substrate_catalyst_matrix.py`, `902_cho_mechanistic_diagnostic.py`,
`generate_shap_for_901.py`, `regenerate_paper_figures.py`.

### src/dft/ — DFT validation

| File | Purpose |
|------|---------|
| `501_generate_dft_inputs.py` | Generate ORCA `.inp` files for the validation set |
| `502_generate_dft_inputs_extended.py` | Extended validation set: transition states, complexes, IL splits |
| `510_parse_dft_outputs.py` | Parse ORCA outputs into a tidy CSV |
| `510b_orca_ts_b_freq_nbo.py` | Specialised parser for TS / frequency / NBO runs |
| `512_xtb_on_dft_geometry.py` | Re-run xTB on DFT-optimised geometries (calibration) |
| `514_dft_vs_xtb_report.py` | xTB vs DFT energy comparison |
| `514b_dft_transition_state.py` | Transition-state analysis for IGE/SO/PO |

### src/visualization/ — paper figures

| File | Purpose |
|------|---------|
| `fig0_graphical_abstract_v3.py` | Graphical abstract (v3) |
| `fig1_protocol_comparison_en.py` | Fig 1: 8 protocols × 2 feature sets (English) |
| `fig2_loso_quality_en.py` | Fig 2: LOSO protocol quality |
| `fig3_coverage_5x5_en.py` | Fig 3: Substrate × catalyst 5×5 coverage |
| `fig4_shap_per_substrate_en.py` | Fig 4: Per-substrate SHAP |
| `fig5_homo_vs_yield_en.py` | Fig 5: HOMO energy vs yield scatter |
| `fig_pcl_ae_architecture.py` | PCL-AE architecture diagram |
| `fig_yrandomization_100perm.py` | Y-randomization (100 permutations) |
| `s6_dft_vs_xtb_grid.py` | DFT-vs-xTB comparison grid |
| `regen_all_v3.py` | One-shot regeneration of all v3 figures |

### src/ci_artifacts/ — SI artifact generators

`generate_bootstrap_substrate_ci.py`, `generate_si_s3_benchmark_full_v3_1.py`,
`generate_y_randomization_v4_100perm.py`, `generate_year_ood_benchmark.py`.
These regenerate the SI tables deterministically from the processed CSVs.

### Top-level src/ helpers

| File | Purpose |
|------|---------|
| `config.py` | Auto-loaded tuned hyperparameters (PCL-AE λ, latent dim, DRFP variant) |
| `paths.py` | Single source of truth for file paths |
| `orchestrator.py` | End-to-end pipeline orchestrator (calls numbered scripts) |
| `CO2_features.py`, `CO2_rxn.py`, `molecule.py`, `utils_features.py`, `utils_rxn.py`, `utils_benchmark.py`, `data_split.py`, `candidate_library.py`, `shap_explanation.py` | Helper modules: SMILES handling, DRFP / xTB feature I/O, reaction / molecule utilities, data-splitting policy, candidate library construction, SHAP explanation glue |

---

## scripts/

> **Note:** The `scripts/` directory is not included in this repository.
> The pipeline execution order is documented in
> [`scripts/RUN_ORDER.md`](scripts/RUN_ORDER.md) (read-only reference).

---

## configs/

| File | Purpose |
|------|---------|
| `data_registry.yaml` | Catalogues processed CSVs (paths, schema, version) |
| `model_artifacts.yaml` | Catalogues trained model artefacts and their hashes |
| `project.yaml` | Local project config (gitignored; contains maintainer contact) |
| `project.yaml.example` | Template; copy to `project.yaml` and edit |

---

## data/

| Subdirectory | Contents |
|--------------|----------|
| `processed/` | Cleaned CSV, reaction SMILES, DRFP, xTB-merged features, catalyst mechanism, bootstrap CIs, statistical test results |

The raw Reaxys export is **not** redistributed.

---

## figures/

Paper figures in English (PDF + PNG + SVG):

| Figure | Description |
|--------|-------------|
| `fig0_graphical_abstract_en` | Graphical abstract |
| `fig1_data_overview_en` | Dataset overview |
| `fig1_protocol_comparison_en` | 8 protocols × 2 feature sets |
| `fig2_loso_root_cause_en` | LOSO / LOMO / KFold quality comparison |
| `fig3_coverage_matrix_en` | Substrate × catalyst coverage |
| `fig4_shap_direction_en` | Per-substrate SHAP attribution |
| `fig5_ablation_en` | Feature ablation results |
| `fig6_energy_levels_en` | Five-substrate DFT energy levels (canonical v2 source) |
| `fig7_latent_space_en` | PCL-AE latent space (128-D, post-fair-comparison) |

---

## dft_validation/

KL divergence summary between DFT and xTB HOMO energies for the
validation set.  Raw ORCA / xTB outputs are not included.

**SP single-point DFT cross-check (current v2 source).** The five-archived-geometry single-point energies and NBO results in `dft_validation/sp_5_subs/sp_5_subs_summary_v2.csv` (the canonical "CHO gap 0.6344 eV" numbers; PO/ECH/IGE gaps 5.20–5.42 eV; SO gap 4.02 eV; LUMO-driven gap shift on CHO) were computed with `* xyzfile -1 1` (charge = −1, mult = 1, closed-shell anion singlet) at the B3LYP-D3BJ/def2-TZVP/SMD(DMSO) level using ORCA 6.1.1. All five SCFs are `fully_converged`. The v2 SP inputs are documented in SI §S7.8 (input templates `sp_*_v2.inp`); the table and per-substrate diagnostics are in SI §S7.6. For the raw `.out` files, NBO populations, NPA charges, and bond orders that underlie the v2 summary table, see SI §S7.6 and §S7.7 of [docs/supplementary/S7_ige_transition_state.md](docs/supplementary/S7_ige_transition_state.md) (the supplementary materials accompanying this manuscript).

**The DFT line therefore does not contribute to the main-text CHO distinctiveness claim**, which rests on the SHAP sign reversal and KL divergence (§3.3 of the paper). The full discussion of the v2 SCF protocol and the geometry-status caveat is in **SI §S7.6, §S7.7 and §S7.8** of [docs/supplementary/S7_ige_transition_state.md](docs/supplementary/S7_ige_transition_state.md). Future work is to repeat the TS optimisation for CHO and SO from longer initial Br–C distances with GFN-FF pre-optimisation, then re-assess CHO's frontier-orbital position at full SCF and geometry convergence.

| File | Contents |
|------|----------|
| `loso_kl_divergence.csv` | KL divergence per substrate for DFT-vs-xTB HOMO comparison |

---

## Reproducing the headline results

```bash
git clone https://github.com/zpao114/CO2-cycloaddition-ml.git
cd CO2-cycloaddition-ml

# Environment
conda create -n co2_ml python=3.10
conda activate co2_ml
pip install -r requirements.txt
conda install -c conda-forge xtb==22.1

# Configure
cp configs/project.yaml.example configs/project.yaml
# edit configs/project.yaml (email, paths)

# Pipeline (PowerShell)
.\scripts\run_pipeline_v2.ps1 -List        # show tiers
.\scripts\run_pipeline_v2.ps1 -DryRun     # preview commands
.\scripts\run_pipeline_v2.ps1 -NoXTB      # skip xTB (~6 hr)
.\scripts\run_pipeline_v2.ps1             # run all
```

For the optional DFT validation set, install ORCA 5.0 or newer separately
and set `ORCA_PATH` in `configs/project.yaml`.

---

## Key results (headline)

2,490 CO2 cycloaddition reactions across 5 substrates and 5 catalyst
mechanism classes (`data/processed/cleaned.csv`, extraction_status='valid').

| Substrate | n | Mean yield | Unique catalysts |
|-----------|---|------------|------------------|
| Styrene oxide | 783 | 84.4% | 83 |
| Epichlorohydrin | 692 | 91.9% | 73 |
| Propylene oxide | 646 | 88.3% | 78 |
| Cyclohexene oxide | 305 | 53.1% | 47 |
| Isopropyl glycidyl ether | 64 | 91.0% | 30 |

Model performance on the headline protocols:

| Protocol | Model | R² | Notes |
|----------|-------|-----|-------|
| 5-fold KFold | DualBranchANN | 0.4106 | MAE = 0.116 |
| External holdout (15%, single seed=2026) | RF | 0.119 | n = 317 (per `results_external_validation/external_validation_results.csv`) |
| External holdout (15%, 4-seed mean) | PCL-AE-128 + DualANN | +0.131 ± 0.128 | n = 317 (per `results_external_holdout/multiseed_v2_summary.csv`); 95% CI [−0.072, +0.334] |
| Year-OOD (train ≤2021, test ≥2022) | RF | +0.425 ± 0.002 | n_train=1,499, n_test=942 (per `results/results_si/year_ood_benchmark.csv`) |
| y-randomization (100 perms) | All models | p < 0.01 | Signal is real |

**LOSO failure is structurally robust.** Cyclohexene oxide (CHO, mean
yield 53.1%) sits in a different yield regime than the terminal epoxides
(mean yield ~88%) and drives LOSO failure. From
`results_step4/summary_protocol.csv` (XGBoost proxy; X1_xTB+mech is the
canonical feature set used in this paper):

| Protocol | R² | MAE | n_train / n_test |
|----------|-----|-----|------------------|
| 5-fold KFold (baseline) | +0.279 | 12.44 | 1992 / 2490 |
| **LOSO** | **−0.063** | 14.34 | 1992 / 2490 |
| LOMO | +0.075 | 14.69 | 1867 / 2490 |
| LOSO × LOMO | +0.203 | 13.41 | 2344 / 2470 |

For the X0_xTB_only feature set the numbers are nearly identical (LOSO
R² ≈ −0.068, LOMO R² ≈ +0.072, LOSO×LOMO R² ≈ +0.195).

Bootstrap 95% CI (B = 1000, per-substrate residuals; from §S3.3
`docs/supplementary/S3_protocol_matrix.md`):

| Protocol | mean R² | 95% CI | Half-width |
|----------|---------|--------|------------|
| LOSO (xTB only) | −0.068 | [−0.100, −0.036] | 0.032 |
| LOSO (xTB + mech) | −0.063 | [−0.096, −0.030] | 0.033 |
| LOSO × LOMO (xTB only) | +0.195 | [+0.155, +0.235] | 0.040 |

Both LOSO CIs lie entirely below zero (X0 upper bound −0.036 < 0),
confirming the failure is statistically significant.

> **Note:** An earlier draft of this README, alongside the corresponding
> section in `docs/RESULTS.md`, included a 4-model × 4-cell per-model
> LOSO/LOMO table (XGB −0.441, LGBM −0.519, RF −2.300, DualBranchANN
> −5.031) and a Bootstrap 95% CI block with `LOSO mean=−0.051,
> LOSO×LOMO mean=+0.217`. These numbers were sourced from an early
> baseline run that is no longer reproducible (`docs/supplementary/
> S3_4_model_benchmark_full.md` documents the discrepancy and supersedes
> them). They have been removed in this revision; the per-protocol
> aggregate numbers above are the canonical XGBoost-proxy results from
> `results_step4/summary_protocol.csv`. If a per-model LOSO comparison
> is needed, re-run `src/analysis/loso/700_loso_lomo_cv.py` across the
> four architectures.

---

## Method summary

| Component | Description |
|-----------|-------------|
| DRFP | Differential Reaction Fingerprint, 2048-bit |
| GFN2-xTB | Semi-empirical electronic descriptors (HOMO, LUMO, charges, dipole) |
| PCL-AE | Property-Co-Learning AutoEncoder, 128-D latent (λ = 200.0) |
| DualBranchANN | Dual-branch ANN: DRFP branch + xTB / condition branch |
| MechanismAwareLOSO | Leave-one-substrate-out with mechanistic interpretation |
| Chemical SHAP | SHAP values mapped to chemical meaning |

Tuned hyperparameters (see `src/config.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| PCL-AE λ | 200.0 | 17-point scan, 3 seeds |
| Latent dimension | 128 | hard-coded |
| DRFP variant | full | reactants + catalyst + solvent |

---

## Citation

A formal citation will be added when the manuscript is accepted. Until
then, please cite this repository as:

```bibtex
@software{co2_cycloaddition_ml,
  title  = {CO2 Cycloaddition ML Pipeline},
  author = {zpao114},
  year   = {2026},
  url    = {https://github.com/zpao114/CO2-cycloaddition-ml}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
