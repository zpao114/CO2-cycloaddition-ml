"""103_drfp.py — generate 4 DRFP variants from co2_smiles.csv.

Inputs
------
data/processed/co2_smiles.csv   (output of 102_smiles.py)
    - row_id, yield (%), temperature (℃), pressure (MPa), ..., reactant_smiles,
      catalyst_*_smiles, solvent_*_smiles, RXN_SMILES

Outputs
-------
data/processed/co2_drfp.csv
    - Every column of co2_smiles.csv, unchanged and in the same order
      (including the 102 column name ``temperature (℃)``), followed by the
      four DRFP columns.
    - The row set equals co2_smiles.csv exactly. All DRFP vectors are
      recomputed on every run; no previously written co2_drfp.csv is read or
      reused, so a stale file can never leak old rows or old column names.

DRFP variant mapping
--------------------
  drfp          = reactants.catalysts.solvents>>products    (full)
  drfp React    = reactants>>products                        (reactants-only)
  drfp wo cats  = reactants.solvents>>products               (no catalysts)
  drfp wo sols  = reactants.catalysts>>products              (no solvents)

Encoding: DrfpEncoder (radius=3, n_folded_length=2048, default settings).

Idempotent: re-running overwrites co2_drfp.csv cleanly.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from typing import Iterable, List, Optional

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows PowerShell (cp936 default).
# Guarded so importing this module from a notebook/Jupyter doesn't re-wrap.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=not sys.stdout.isatty(),
            )
    except Exception:  # pragma: no cover
        pass

from src.paths import (  # noqa: E402  (after sys.stdout reconfigure)
    DRFP_CSV,
    SMILES_CSV,
    ensure_dir,
)


DRFP_FULL = "drfp"
DRFP_REACT = "drfp React"
DRFP_NO_CATS = "drfp wo cats"
DRFP_NO_SOLS = "drfp wo sols"
DRFP_VARIANTS = [DRFP_FULL, DRFP_REACT, DRFP_NO_CATS, DRFP_NO_SOLS]


def _join(parts: Iterable[Optional[str]]) -> Optional[str]:
    """Join non-empty SMILES with '.'; return None if all parts empty."""
    items = [str(p).strip() for p in parts if isinstance(p, str) and str(p).strip()]
    return ".".join(items) if items else None


def build_variant_rxn(
    row: pd.Series,
    *,
    include_catalysts: bool,
    include_solvents: bool,
) -> Optional[str]:
    """Build an RXN_SMILES string with selected component subsets.

    Format: '<reactants>.<catalysts>.<solvents>>>products'.
    """
    rs = _join([row.get("reactant_smiles")])
    ps = _join([row.get("product_smiles")])
    if not rs:
        return None

    parts_cat: List[str] = []
    if include_catalysts:
        c = _join([row.get(f"catalyst_{i}_smiles") for i in range(1, 5)])
        if c:
            parts_cat.append(c)

    parts_solv: List[str] = []
    if include_solvents:
        s = _join([row.get(f"solvent_{i}_smiles") for i in range(1, 5)])
        if s:
            parts_solv.append(s)

    lhs = ".".join([rs] + parts_cat + parts_solv)
    rhs = ps or ""
    return f"{lhs}>>{rhs}"


def encode_one(encoder, rxn_str: str) -> np.ndarray:
    """Encode a single RXN_SMILES via DrfpEncoder; returns 2048-bit vector."""
    fps = encoder.encode([rxn_str])
    return fps[0]


def drfp_to_str(fp: np.ndarray) -> str:
    """Render DRFP vector as the canonical '[0 0 1 ...]' string used in co2_drfp.csv."""
    return "[" + " ".join(str(int(b)) for b in fp) + "]"


def encode_rows(smiles_subset: pd.DataFrame) -> pd.DataFrame:
    """Encode the 4 DRFP variants for every row in `smiles_subset`."""
    from drfp import DrfpEncoder
    encoder = DrfpEncoder()
    variant_defs = [
        (DRFP_FULL,    dict(include_catalysts=True,  include_solvents=True)),
        (DRFP_REACT,   dict(include_catalysts=False, include_solvents=False)),
        (DRFP_NO_CATS, dict(include_catalysts=False, include_solvents=True)),
        (DRFP_NO_SOLS, dict(include_catalysts=True,  include_solvents=False)),
    ]

    records = []
    n_total = len(smiles_subset)
    encode_log = logging.getLogger("103_drfp.encode")
    for n, (_, row) in enumerate(smiles_subset.iterrows(), 1):
        rid = int(row["row_id"])
        record = {"row_id": rid}
        for col, opts in variant_defs:
            rxn = build_variant_rxn(row, **opts)
            if rxn is None:
                record[col] = np.nan
                continue
            fp = encode_one(encoder, rxn)
            record[col] = drfp_to_str(fp)
        records.append(record)
        if n % 50 == 0 or n == n_total:
            encode_log.info("  encoded %d/%d", n, n_total)

    return pd.DataFrame(records, columns=["row_id"] + DRFP_VARIANTS)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate 4 DRFP variants (full / reactants-only / no_cats / no_sols) "
            "for every row of co2_smiles.csv. Always recomputes from scratch."
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Accepted for pipeline-wide --force compatibility. "
                        "No-op: this script always re-encodes every row.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute DRFP but do not write co2_drfp.csv.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    log = logging.getLogger("103_drfp")

    log.info("Reading %s", SMILES_CSV)
    if not SMILES_CSV.exists():
        log.error("%s not found. Run 102_smiles.py first.", SMILES_CSV)
        return 1

    smiles = pd.read_csv(SMILES_CSV, encoding="utf-8-sig")
    log.info("Loaded %d rows x %d cols from co2_smiles.csv", len(smiles), len(smiles.columns))

    if smiles["row_id"].duplicated().any():
        log.error("co2_smiles.csv has duplicate row_id values; refusing to continue.")
        return 1

    log.info("Encoding %d rows (full recompute; no previous DRFP is reused)...", len(smiles))
    drfp = encode_rows(smiles).set_index("row_id")

    out = smiles.copy()
    for col in DRFP_VARIANTS:
        out[col] = out["row_id"].astype(int).map(drfp[col])
    out = out.sort_values("row_id").reset_index(drop=True)

    n_missing = int(out[DRFP_FULL].isna().sum())
    if n_missing:
        log.warning("%d rows have no full DRFP (empty reactant_smiles)", n_missing)

    if args.dry_run:
        log.info("[dry-run] DRFP computed but not written.")
        return 0

    ensure_dir(DRFP_CSV.parent)
    out.to_csv(DRFP_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%d rows x %d cols)", DRFP_CSV, len(out), len(out.columns))
    log.info("  %d freshly encoded rows", len(out) - n_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
