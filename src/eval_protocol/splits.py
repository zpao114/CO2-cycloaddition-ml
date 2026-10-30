# -*- coding: utf-8 -*-
"""
eval_protocol/splits.py
=======================
Row-ID based split manifests with data-fingerprint verification.

Why
---
The legacy ``data_split.json`` stores *positional* integer indices. If the
upstream CSV is regenerated (101/107/108 re-run) and the row count or order
changes, the manifest silently points at the wrong rows. Here every split is
stored as lists of stable ``row_id`` strings plus a SHA-256 fingerprint of the
row-id set (optionally of the labels too). ``SplitManifest.resolve`` refuses to
return indices if the fingerprint, row-id set, or label hash does not match.

Provided split families
-----------------------
* ``leave_one_group_out_manifest`` (LOSO = group by substrate, LOMO = group by
  mechanism/catalyst class, any other grouping works too)
* ``group_kfold_manifest``  (deterministic, seed-shuffled, size-balanced)
* ``stratified_kfold_manifest`` (yield-quantile strata)
* ``holdout_manifest``      (single stratified train/test split)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

MANIFEST_VERSION = "2.0"


class ManifestMismatchError(RuntimeError):
    """Raised when a manifest does not match the data it is applied to."""


# ---------------------------------------------------------------------------
# Row IDs and fingerprints
# ---------------------------------------------------------------------------
def make_row_ids(df: pd.DataFrame, key_cols: Sequence[str]) -> np.ndarray:
    """Stable content-derived row IDs.

    ``row_id = sha1(key columns joined by '|')[:16]``. If several rows share
    identical key values, an occurrence counter (``#0``, ``#1`` ...) is appended
    in file order so that IDs are unique but still content-anchored.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"key columns not in frame: {missing}")
    base = (
        df[list(key_cols)]
        .astype(str)
        .agg("|".join, axis=1)
        .map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])
    )
    occ = base.groupby(base).cumcount()
    return (base + "#" + occ.astype(str)).to_numpy(dtype=object)


def fingerprint(row_ids: Sequence[str], y: Optional[Sequence[float]] = None) -> str:
    """SHA-256 over the sorted row-id set (and rounded labels, if given)."""
    ids = np.asarray(row_ids, dtype=object)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("row_ids must be unique")
    order = np.argsort(ids.astype(str), kind="mergesort")
    h = hashlib.sha256()
    for i in order:
        h.update(str(ids[i]).encode("utf-8"))
        h.update(b"\n")
        if y is not None:
            h.update(f"{float(np.asarray(y)[i]):.6f}".encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SplitIndices:
    """Resolved split: integer positions into the *current* data table."""
    name: str
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class SplitManifest:
    kind: str
    row_ids: List[str]
    data_hash: str
    splits: List[Dict]
    params: Dict = field(default_factory=dict)
    has_labels_in_hash: bool = False
    version: str = MANIFEST_VERSION

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "kind": self.kind,
            "params": self.params,
            "n_rows": len(self.row_ids),
            "data_hash": self.data_hash,
            "has_labels_in_hash": self.has_labels_in_hash,
            "row_ids": self.row_ids,
            "splits": self.splits,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path) -> "SplitManifest":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("version") != MANIFEST_VERSION:
            raise ManifestMismatchError(
                f"manifest version {d.get('version')!r} != {MANIFEST_VERSION!r}; "
                "regenerate it with eval_protocol.splits"
            )
        return cls(
            kind=d["kind"], row_ids=list(d["row_ids"]), data_hash=d["data_hash"],
            splits=d["splits"], params=d.get("params", {}),
            has_labels_in_hash=bool(d.get("has_labels_in_hash", False)),
            version=d["version"],
        )

    # ------------------------------------------------------------ resolution
    def resolve(
        self,
        row_ids: Sequence[str],
        y: Optional[Sequence[float]] = None,
    ) -> List[SplitIndices]:
        """Map stored row IDs onto positions of the current table.

        Raises ``ManifestMismatchError`` if the current table's row-id set (or,
        when the manifest was built with labels, the labels) differ from the
        ones the manifest was created on.
        """
        ids = np.asarray(row_ids, dtype=object)
        use_y = y if self.has_labels_in_hash else None
        if self.has_labels_in_hash and y is None:
            raise ManifestMismatchError("manifest hash includes labels; pass y")
        got = fingerprint(ids, use_y)
        if got != self.data_hash:
            raise ManifestMismatchError(
                f"data fingerprint mismatch: manifest={self.data_hash[:12]}... "
                f"current={got[:12]}... (rows: manifest={len(self.row_ids)}, "
                f"current={len(ids)}). Regenerate the manifest after changing "
                "upstream data."
            )
        pos = {rid: i for i, rid in enumerate(ids.tolist())}
        out: List[SplitIndices] = []
        for s in self.splits:
            tr = np.array([pos[r] for r in s["train_ids"]], dtype=np.int64)
            te = np.array([pos[r] for r in s["test_ids"]], dtype=np.int64)
            if np.intersect1d(tr, te).size:
                raise ManifestMismatchError(f"split {s['name']} overlaps train/test")
            out.append(SplitIndices(s["name"], int(s["fold"]), tr, te))
        return out


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _finalize(kind, row_ids, y, splits, params, include_labels) -> SplitManifest:
    ids = [str(r) for r in row_ids]
    return SplitManifest(
        kind=kind,
        row_ids=ids,
        data_hash=fingerprint(ids, y if include_labels else None),
        splits=splits,
        params=params,
        has_labels_in_hash=bool(include_labels and y is not None),
    )


def _split_dict(name, fold, ids, tr, te) -> Dict:
    return {
        "name": str(name),
        "fold": int(fold),
        "train_ids": [str(ids[i]) for i in tr],
        "test_ids": [str(ids[i]) for i in te],
    }


def leave_one_group_out_manifest(
    row_ids: Sequence[str],
    groups: Sequence,
    kind: str = "LOGO",
    min_test: int = 1,
    y: Optional[Sequence[float]] = None,
    include_labels_in_hash: bool = False,
) -> SplitManifest:
    """One split per group; the group is the test set (LOSO / LOMO / LOCO)."""
    ids = np.asarray(row_ids, dtype=object)
    g = np.asarray(groups).astype(str)
    if len(ids) != len(g):
        raise ValueError("row_ids and groups differ in length")
    splits = []
    for k, grp in enumerate(sorted(set(g.tolist()))):
        te = np.where(g == grp)[0]
        if len(te) < min_test:
            continue
        tr = np.where(g != grp)[0]
        splits.append(_split_dict(f"{kind}:{grp}", k, ids, tr, te))
    return _finalize(kind, ids, y, splits, {"min_test": min_test}, include_labels_in_hash)


def loso_manifest(row_ids, substrates, **kw) -> SplitManifest:
    """Leave-One-Substrate-Out."""
    return leave_one_group_out_manifest(row_ids, substrates, kind="LOSO", **kw)


def lomo_manifest(row_ids, mechanisms, **kw) -> SplitManifest:
    """Leave-One-Mechanism(-class)-Out."""
    return leave_one_group_out_manifest(row_ids, mechanisms, kind="LOMO", **kw)


def group_kfold_manifest(
    row_ids: Sequence[str],
    groups: Sequence,
    n_splits: int = 5,
    seed: int = 0,
    y: Optional[Sequence[float]] = None,
    include_labels_in_hash: bool = False,
) -> SplitManifest:
    """Deterministic size-balanced GroupKFold (groups shuffled by ``seed``)."""
    ids = np.asarray(row_ids, dtype=object)
    g = np.asarray(groups).astype(str)
    uniq, counts = np.unique(g, return_counts=True)
    if len(uniq) < n_splits:
        raise ValueError(f"{len(uniq)} groups < n_splits={n_splits}")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    # largest groups first (stable on the shuffled order), assign to emptiest fold
    order = perm[np.argsort(-counts[perm], kind="mergesort")]
    load = np.zeros(n_splits, dtype=int)
    fold_of: Dict[str, int] = {}
    for gi in order:
        f = int(np.argmin(load))
        fold_of[uniq[gi]] = f
        load[f] += counts[gi]
    fold_arr = np.array([fold_of[x] for x in g])
    splits = []
    for f in range(n_splits):
        te = np.where(fold_arr == f)[0]
        tr = np.where(fold_arr != f)[0]
        splits.append(_split_dict(f"GroupKFold:{f}", f, ids, tr, te))
    return _finalize("GroupKFold", ids, y, splits,
                     {"n_splits": n_splits, "seed": seed}, include_labels_in_hash)


def _yield_strata(y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, qs, right=True)


def stratified_kfold_manifest(
    row_ids: Sequence[str],
    y: Sequence[float],
    n_splits: int = 5,
    seed: int = 0,
    n_bins: int = 4,
    include_labels_in_hash: bool = False,
) -> SplitManifest:
    from sklearn.model_selection import StratifiedKFold

    ids = np.asarray(row_ids, dtype=object)
    y = np.asarray(y, dtype=float)
    strata = _yield_strata(y, n_bins)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [
        _split_dict(f"StratKFold:{f}", f, ids, tr, te)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), strata))
    ]
    return _finalize("StratKFold", ids, y, splits,
                     {"n_splits": n_splits, "seed": seed, "n_bins": n_bins},
                     include_labels_in_hash)


def repeated_stratified_folds(
    y: Sequence[float],
    n_repeats: int,
    n_splits: int = 5,
    seed: int = 0,
    n_bins: int = 4,
):
    """Independent yield-stratified K-fold partitions, one per repeat.

    Returns ``[[(fold, train_idx, test_idx), ...] for each repeat]`` over
    positions ``0..len(y)-1``.  Repeat ``r`` uses ``random_state = seed + r``, so
    the partitions differ between repeats (re-using one partition ``n_repeats``
    times is pseudo-replication: the paired tests would see correlated copies).
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=float)
    strata = _yield_strata(y, n_bins)
    out = []
    for r in range(int(n_repeats)):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        out.append([(f, tr, te) for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), strata))])
    return out


def holdout_manifest(
    row_ids: Sequence[str],
    y: Sequence[float],
    test_size: float = 0.15,
    seed: int = 0,
    n_bins: int = 4,
    include_labels_in_hash: bool = False,
) -> SplitManifest:
    from sklearn.model_selection import train_test_split

    ids = np.asarray(row_ids, dtype=object)
    y = np.asarray(y, dtype=float)
    tr, te = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed,
        stratify=_yield_strata(y, n_bins),
    )
    splits = [_split_dict("Holdout", 0, ids, np.sort(tr), np.sort(te))]
    return _finalize("Holdout", ids, y, splits,
                     {"test_size": test_size, "seed": seed, "n_bins": n_bins},
                     include_labels_in_hash)
