"""102_smiles.py — add SMILES columns to cleaned.csv, producing co2_smiles.csv.

Inputs
------
cleaned.csv                                 (output of 101_clean.py)
    - reactant_name, product_name, catalyst_*_name, solvent_*, no SMILES

Outputs
-------
co2_smiles.csv
    - cleaned.csv + 11 SMILES columns: reactant_smiles, product_smiles,
      catalyst_{1..4}_smiles, solvent_{1..4}_smiles, RXN_SMILES
smiles_baseline.json                        (diagnostic only)

Strategy
--------
1. Strict exact-name lookup in chem_registry.py (single source of truth,
   shared with 108_rebuild_from_registry.py and 104c_run_xtb_registry.py).
   No substring matching and no RDKit fallback: an unknown name raises
   KeyError so it is reviewed and added to the registry.
2. RXN_SMILES = '<reactants>.<catalysts>.<solvents>>>products'.

Idempotent: re-running overwrites co2_smiles.csv cleanly.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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
    CLEANED_CSV,
    SMILES_BASELINE,
    SMILES_CSV,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import chem_registry as reg  # noqa: E402


def resolve_smiles(name) -> Optional[str]:
    """Strict name -> SMILES lookup via chem_registry.

    Blank / NaN / solvent-free / unresolved entries give None.  A name that is
    not in the registry raises KeyError: new names must be reviewed and added
    to chem_registry.py, never guessed by substring matching.
    """
    return reg.smiles_of(name)


def build_rxn_smiles(row: pd.Series) -> Optional[str]:
    """Assemble RXN_SMILES = 'reactants.catalysts.solvents>>>products'."""
    parts_reactants: List[str] = []
    parts_catalysts: List[str] = []
    parts_solvents:  List[str] = []
    parts_products:  List[str] = []

    if isinstance(row.get("reactant_smiles"), str):
        parts_reactants.append(row["reactant_smiles"])
    if isinstance(row.get("product_smiles"), str):
        parts_products.append(row["product_smiles"])
    for i in range(1, 5):
        cs = row.get(f"catalyst_{i}_smiles")
        if isinstance(cs, str):
            parts_catalysts.append(cs)
    for i in range(1, 5):
        ss = row.get(f"solvent_{i}_smiles")
        if isinstance(ss, str):
            parts_solvents.append(ss)

    if not parts_reactants:
        return None
    lhs = ".".join(parts_reactants + parts_catalysts + parts_solvents)
    rhs = ".".join(parts_products) if parts_products else ""
    return f"{lhs}>>{rhs}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add 11 SMILES columns to cleaned.csv, producing co2_smiles.csv "
            "(strict chem_registry lookup, no fallback) and RXN_SMILES."
        ),
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing co2_smiles.csv even if present.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute SMILES but do not write co2_smiles.csv.")
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
    log = logging.getLogger("102_smiles")

    log.info("Reading %s", CLEANED_CSV)
    if not CLEANED_CSV.exists():
        log.error("%s not found. Run 101_clean.py first.", CLEANED_CSV)
        return 1

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"] == "valid"].copy()
    if "yield (%)" in df.columns:
        df = df.dropna(subset=["yield (%)"])
        df = df[df["yield (%)"] > 0]
    df = df.reset_index(drop=True)
    n_in = len(df)
    log.info("Loaded %d rows x %d cols (after valid+yield>0 filter)", n_in, len(df.columns))

    df["reactant_smiles"] = df["reactant_name"].apply(resolve_smiles)
    df["product_smiles"]  = df["product_name"].apply(resolve_smiles)

    for i in range(1, 5):
        col_name  = f"catalyst_{i}_name"
        col_smiles = f"catalyst_{i}_smiles"
        if col_name in df.columns:
            df[col_smiles] = df[col_name].apply(resolve_smiles)
        else:
            df[col_smiles] = pd.NA

    for i in range(1, 5):
        col_name  = f"solvent_{i}"
        col_smiles = f"solvent_{i}_smiles"
        if col_name in df.columns:
            df[col_smiles] = df[col_name].apply(resolve_smiles)
        else:
            df[col_smiles] = pd.NA

    df["RXN_SMILES"] = df.apply(build_rxn_smiles, axis=1)

    coverage = {
        "reactant_smiles":   df["reactant_smiles"].notna().sum(),
        "product_smiles":    df["product_smiles"].notna().sum(),
        "catalyst_1_smiles": df["catalyst_1_smiles"].notna().sum(),
        "catalyst_2_smiles": df["catalyst_2_smiles"].notna().sum(),
        "RXN_SMILES":        df["RXN_SMILES"].notna().sum(),
    }
    log.info("SMILES coverage:")
    for k, v in coverage.items():
        log.info("  %-18s: %4d / %d", k, v, n_in)

    if args.dry_run:
        log.info("[dry-run] SMILES computed but not written.")
        return 0

    if SMILES_CSV.exists() and not args.force:
        log.warning("%s already exists (use --force to overwrite)", SMILES_CSV)
        return 0

    SMILES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SMILES_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%d rows x %d cols)", SMILES_CSV, len(df), len(df.columns))

    diag = {
        "n_rows": int(n_in),
        "smiles_coverage": {k: int(v) for k, v in coverage.items()},
        "n_lookup_entries": len(reg.REGISTRY),
    }
    try:
        SMILES_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(SMILES_BASELINE, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        log.info("Wrote %s", SMILES_BASELINE)
    except Exception as e:
        log.warning("Skipped smiles_baseline.json: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
