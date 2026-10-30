# CO2 Cycloaddition ML

This repository is the code and dataset for our study on machine-learning
catalyst screening for CO2 / epoxide cycloaddition. It includes the cleaned
reaction dataset and the scripts for data processing, model training and
evaluation. The paper is submitted to *Journal of Chemical Information and
Modeling* (JCIM).

The project is written in Python 3.10 and runs on both CPU and GPU. The
required libraries are listed in `requirements.txt`. GFN2-xTB descriptors
need the `xtb` program (installable from conda-forge); ORCA is only required
for the optional DFT validation part.

The repository contains the following folders.

**data/** — Cleaned dataset used in the paper (`data/processed/`), covering
1033 CO2 cycloaddition reactions.

**src/data/** — Data cleaning and feature generation. `101_clean.py` cleans
the Reaxys-derived reaction table, `102_smiles.py` builds reaction SMILES,
`103_drfp.py` computes the DRFP fingerprints, `104c_run_xtb_registry.py` and
`105b_xtb_sanity_v2.py` generate and sanity-check the GFN2-xTB descriptors,
and `107_merge_substrate_xtb.py` merges them into one feature table.
`data_split.py` produces the train/test split.

**src/models/** — Model training and evaluation. `persistence/train_pcl_ae.py`
trains the property co-learning autoencoder (PCL-AE);
`persistence/401_persist_best_pipeline.py` and
`persistence/405_external_validation.py` persist the best pipeline and run
external validation. `benchmarks/301_benchmark.py` runs the main 5-fold
benchmark.

**src/eval_protocol/** — The RSEP evaluation protocol proposed in the paper.
It compares the pretrain-and-freeze and per-fold-retraining strategies under
identical data splits, seeds and downstream regressors, and reports the
paired difference in performance.

**src/dft/** — Optional DFT validation scripts: ORCA input generation and
parsing, xTB-on-DFT-geometry calibration, and the xTB-vs-DFT comparison.

**src/** — Helper modules (`config.py`, `paths.py`, `CO2_features.py`,
`CO2_rxn.py`, `molecule.py`, `utils_features.py`, `utils_rxn.py`,
`utils_benchmark.py`, `candidate_library.py`).

**figures/** — The figures used in the paper.

**results_honest_groupkfold/** and the other four `results_*/` directories —
The result directories referenced by the paper
(`results_external_holdout/`, `results_pcl_ae_fair_comparison_multiseed/`,
`results_substrate_catalyst_matrix/`, `results_permuted_ablation/`).

## Reproduce

Create a Python 3.10 environment, install `requirements.txt` and the `xtb`
program, then run the numbered scripts under `src/data/`, `src/models/` and
`src/eval_protocol/` in the order of their numeric prefixes.

## Citation

If you use this code, please cite the repository:

```bibtex
@software{co2_cycloaddition_ml,
  title  = {CO2 Cycloaddition ML},
  author = {zpao114},
  year   = {2026},
  url    = {https://github.com/zpao114/CO2-cycloaddition-ml}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
