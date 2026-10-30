# -*- coding: utf-8 -*-
"""Unified honest-evaluation framework (see docs in each submodule)."""
from .splits import (  # noqa: F401
    ManifestMismatchError, SplitIndices, SplitManifest, fingerprint,
    group_kfold_manifest, holdout_manifest, leave_one_group_out_manifest,
    lomo_manifest, loso_manifest, make_row_ids, stratified_kfold_manifest,
)
from .encoders import (  # noqa: F401
    IdentityEncoder, PCAEncoder, PCLAEEncoder, StandardAEEncoder,
)
from .models import (  # noqa: F401
    ConcatMLP, ConcatMLPRegressor, DualBranchANN, DualBranchRegressor, make_tree_model,
)
from .harness import FIT_SCOPES, FoldFit, evaluate, evaluate_all_scopes, fit_fold  # noqa: F401
from .stats import (  # noqa: F401
    cluster_bootstrap_ci, decompose_delta, paired_seed_test, spearman, summarize_oof,
)
from .controls import label_permutation_control, lambda_dose_response, permutation_verdict  # noqa: F401
from .audit import near_duplicate_audit  # noqa: F401
from .features import NUMERIC_FEATURE_COLS, FoldImputer, ImputeScaleModel, fold_standardize  # noqa: F401
