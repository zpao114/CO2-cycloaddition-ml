# -*- coding: utf-8 -*-
"""
eval_protocol/selection.py
==========================
Nested selection: choose a hyper-parameter / design variant (DRFP variant, PCL-AE
lambda, ...) using ONLY the training rows of each outer fold, then score the
outer test fold with the choice that was made without seeing it.

Why: the legacy Stage 1 / Stage 2 of ``201_ablation.py`` took ``idxmax`` over a
sweep scored on the same CV folds that were later reported as performance
(Cawley & Talbot 2010; Varma & Simon 2006).  The max of K noisy scores is an
optimistic estimate of the chosen configuration's performance.

``nested_select`` reports three numbers that make the optimism visible:

* ``outer_r2``      - pooled out-of-fold R2 of "select on inner CV, evaluate on outer fold"
* ``inner_best_r2`` - mean over outer folds of the winning inner-CV score
* ``optimism``      - ``inner_best_r2 - outer_r2``  (selection bias of the sweep score)

Callback contract (single callback for inner and outer evaluation)::

    fit_predict(candidate, train_idx, test_idx) -> predictions for ``test_idx``

All indices refer to rows of the arrays the callback closes over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

Fold = Tuple[int, Sequence[int], Sequence[int]]
FitPredict = Callable[[Any, np.ndarray, np.ndarray], np.ndarray]


def make_inner_folds(train_idx: Sequence[int], k: int = 3, seed: int = 0,
                     groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Inner folds over ``train_idx`` (absolute row indices).

    With ``groups`` (e.g. substrate / catalyst id of ALL rows) the inner split is a
    GroupKFold so that the inner problem has the same extrapolation structure as the
    outer one (LOSO outer -> substrate-grouped inner).
    """
    train_idx = np.asarray(train_idx)
    if groups is not None:
        g = np.asarray(groups)[train_idx]
        k = int(min(k, len(np.unique(g))))
        if k < 2:
            raise ValueError("fewer than 2 groups in the training rows; cannot nest")
        splitter = GroupKFold(n_splits=k).split(train_idx, groups=g)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed).split(train_idx)
    return [(train_idx[a], train_idx[b]) for a, b in splitter]


def _score(candidate, folds, fit_predict: FitPredict, y: np.ndarray) -> float:
    """Pooled out-of-fold R2 of ``candidate`` over ``folds`` (list of (tr, te))."""
    oof, seen = {}, []
    for tr, te in folds:
        oof[tuple(te)] = fit_predict(candidate, np.asarray(tr), np.asarray(te))
        seen.append(np.asarray(te))
    te_all = np.concatenate(seen)
    pred = np.concatenate([oof[tuple(t)] for t in seen])
    return float(r2_score(y[te_all], pred))


def select_once(candidates: Sequence[Any], pool_idx: Sequence[int], fit_predict: FitPredict,
                y: np.ndarray, inner_k: int = 3, seed: int = 0,
                groups: Optional[np.ndarray] = None, log: Callable = lambda *a: None
                ) -> Tuple[Any, Dict[Any, float]]:
    """Pick the best candidate by inner CV on ``pool_idx`` only. Ties -> first."""
    inner = make_inner_folds(pool_idx, inner_k, seed, groups)
    scores: Dict[Any, float] = {}
    for c in candidates:
        scores[c] = _score(c, inner, fit_predict, y)
        log(f"    candidate={c!r}  inner R2={scores[c]:.4f}")
    best = max(candidates, key=lambda c: scores[c])
    return best, scores


@dataclass
class NestedResult:
    per_fold: pd.DataFrame
    outer_r2: float                  # pooled OOF R2 of the whole select-then-fit procedure
    inner_best_r2: float             # mean winning inner score (= the optimistic sweep score)
    optimism: float
    frequency: Dict[Any, int]
    final_choice: Any                # chosen by inner CV on the FULL pool (use for deployment)
    final_scores: Dict[Any, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "outer_r2_nested": self.outer_r2,
            "inner_best_r2_mean": self.inner_best_r2,
            "optimism": self.optimism,
            "selection_frequency": {str(k): int(v) for k, v in self.frequency.items()},
            "final_choice": self.final_choice,
            "final_choice_inner_scores": {str(k): float(v) for k, v in self.final_scores.items()},
            "per_outer_fold_choice": self.per_fold[["fold", "choice", "inner_r2", "outer_r2"]]
            .to_dict(orient="records"),
            **self.meta,
        }


def nested_select(candidates: Sequence[Any], outer_folds: Sequence[Fold],
                  fit_predict: FitPredict, y: np.ndarray, inner_k: int = 3, seed: int = 0,
                  groups: Optional[np.ndarray] = None, pool_idx: Optional[Sequence[int]] = None,
                  log: Callable = lambda *a: None, meta: Optional[Dict[str, Any]] = None
                  ) -> NestedResult:
    """Select-then-evaluate with the selection confined to each outer training fold."""
    y = np.asarray(y, dtype=np.float64)
    rows, oof_idx, oof_pred = [], [], []
    for fold_id, tr, te in outer_folds:
        tr, te = np.asarray(tr), np.asarray(te)
        log(f"  [outer fold {fold_id}] n_train={len(tr)} n_test={len(te)}")
        best, scores = select_once(candidates, tr, fit_predict, y, inner_k,
                                   seed + int(fold_id), groups, log)
        pred = fit_predict(best, tr, te)
        oof_idx.append(te); oof_pred.append(pred)
        rows.append({"fold": int(fold_id), "choice": best, "inner_r2": scores[best],
                     "outer_r2": float(r2_score(y[te], pred)),
                     **{f"inner_r2[{c}]": scores[c] for c in candidates}})
        log(f"    -> chose {best!r}  inner={scores[best]:.4f}  outer={rows[-1]['outer_r2']:.4f}")
    idx, pr = np.concatenate(oof_idx), np.concatenate(oof_pred)
    outer_r2 = float(r2_score(y[idx], pr))
    df = pd.DataFrame(rows)
    freq = df["choice"].value_counts().to_dict()
    pool = np.unique(np.concatenate([np.asarray(tr) for _, tr, _ in outer_folds])) \
        if pool_idx is None else np.asarray(pool_idx)
    final, final_scores = select_once(candidates, pool, fit_predict, y, inner_k, seed, groups)
    return NestedResult(df, outer_r2, float(df["inner_r2"].mean()),
                        float(df["inner_r2"].mean() - outer_r2), freq, final, final_scores,
                        meta or {})


def write_selection_meta(path, result: NestedResult, **extra) -> None:
    payload = {**result.to_meta(), **extra}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
