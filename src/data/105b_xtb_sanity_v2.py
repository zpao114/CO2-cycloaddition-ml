#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
105b_xtb_sanity_v2.py — xTB sanity check (geometry + reference + coverage)

Reads the output of 104c_run_xtb_registry.py and audits it row by row.

Inputs (all under --xtb-dir, default results/results_cho_diagnostic)
--------------------------------------------------------------------
  xtb_results_summary.csv          one row per species (104c)
  xtb_runs/<name>.xyz              geometry used for the run
  xtb_runs/<name>.xyz.xtb.stdout   raw xTB output

The summary CSV is the source of truth for which species exist. Nothing is
hard-coded here, so the check follows the registry automatically.

What is checked per row
-----------------------
1. Files: the .xyz and .xtb.stdout exist (rows that failed geometry
   generation have no stdout and are reported as GEOMETRY_FAILED).
2. Geometry: no "all atoms at origin", no pair closer than 0.5 A.
3. Consistency: HOMO/LUMO re-parsed from the stdout match raw_homo_eV /
   raw_lumo_eV stored in the summary (catches a stale summary).
4. Physical plausibility of HOMO/LUMO/gap/dipole (skipped for monatomic
   ions, whose frontier orbitals are not meaningful here).
5. Dipole reference: |mu| of CO2, styrene oxide, DMSO and DMF against GFN2
   values (informational only). No orbital-energy reference is kept: HOMO/LUMO
   from GFN2 differ between gas phase and ALPB by about 0.1 eV, and there is
   no independent literature value to compare with.

Rows that 104c already flagged (qc_flag non-empty: failed geometry,
open-shell, degenerate gap) are reported as QC_FLAGGED. That is an expected
outcome, not a failure of this script.

Outputs (written to --out-dir, default = --xtb-dir)
---------------------------------------------------
  xtb_sanity_summary.csv   machine-readable, one row per species
  xtb_sanity_report.txt    human-readable

Usage
-----
    python 105b_xtb_sanity_v2.py
    python 105b_xtb_sanity_v2.py --xtb-dir results/results_cho_diagnostic
    python 105b_xtb_sanity_v2.py --strict   # exit 1 on any structural warning
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logger = logging.getLogger("105b_xtb_sanity_v2")

# Force UTF-8 for stdout to handle non-ASCII chars in geometry warnings
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

# 104b_run_xtb_extended.py starts with a digit, so it cannot be imported with
# a normal `import`; load only its stdout parser via spec_from_file_location.
from importlib import util as _util
_SPEC_PATH = (Path(__file__).parent / "104b_run_xtb_extended.py").resolve()
_spec = _util.spec_from_file_location("src_data_xtb_104b", str(_SPEC_PATH))
_104b = _util.module_from_spec(_spec)
_spec.loader.exec_module(_104b)
parse_xtb_out = _104b.parse_xtb_out

from src.paths import RESULTS_CHO_DIAGNOSTIC, ensure_dir  # noqa: E402

DEFAULT_XTB_DIR = RESULTS_CHO_DIAGNOSTIC
SUMMARY_NAME = "xtb_results_summary.csv"
RUNS_SUBDIR = "xtb_runs"

# Dipole references from GFN2-xTB gas-phase single points (Debye).
# Keys are matched case-insensitively against the summary `name` column.
REFERENCE_VACUUM = {
    "co2":  {"dipole_D": 0.0},
    "styrene_oxide": {"dipole_D": 2.19},
    "dmso": {"dipole_D": 4.62},
    "dmf":  {"dipole_D": 4.61},
}

# Tolerance for the stdout-vs-summary consistency check (eV).
CONSISTENCY_TOL_EV = 1e-3

CSV_FIELDS = ["name", "role", "smiles", "status", "qc_flag",
              "homo", "lumo", "gap", "dipole", "n_atoms", "notes"]


def _nan_to_none(v) -> Optional[float]:
    """pandas NaN / None / '' -> None, otherwise float."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _read_xyz(xyz_path: str):
    """Return (n_atoms, coords, warnings). coords is None on a parse failure."""
    base = os.path.basename(xyz_path)
    if not os.path.isfile(xyz_path):
        return None, None, [f"{base}: xyz file missing"]
    with open(xyz_path, encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        return None, None, [f"{base}: xyz too short"]
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None, None, [f"{base}: cannot parse atom count"]
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    if len(coords) != n_atoms:
        return n_atoms, None, [f"{base}: n_atoms={n_atoms} but only {len(coords)} coords parsed"]
    return n_atoms, coords, []


def _check_geometry(xyz_path: str, n_atoms: int, coords: List[List[float]]) -> List[str]:
    """Detect broken geometries: all atoms at the origin, or overlapping atoms."""
    base = os.path.basename(xyz_path)
    warns: List[str] = []
    if n_atoms > 1:
        nonzero = sum(1 for c in coords if any(abs(v) > 1e-4 for v in c))
        if nonzero == 0:
            warns.append(f"{base}: ALL ATOMS AT ORIGIN, xTB result is meaningless")
        too_short = 0
        for c1, c2 in combinations(coords, 2):
            d = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
            if d < 0.5:
                too_short += 1
        if too_short:
            warns.append(f"{base}: {too_short} atom pair(s) closer than 0.5 A")
    return warns


def _check_physical_plausibility(name: str, res: Dict, monatomic: bool,
                                 charged: bool = False) -> List[str]:
    """Plausibility of frontier orbitals / dipole. Monatomic ions are skipped.

    The dipole bound is not applied to charged species: their dipole depends on
    the coordinate origin, so a large value is not evidence of a bad result.
    """
    if monatomic:
        return []
    warns: List[str] = []
    H, L = res.get("homo_eV"), res.get("lumo_eV")
    G, M = res.get("gap_eV"), res.get("dipole_D")
    if H is None and L is None:
        return [f"{name}: HOMO and LUMO both missing"]
    if H is None or L is None:
        warns.append(f"{name}: HOMO or LUMO missing")
    if H is not None and L is not None and H >= L:
        warns.append(f"{name}: HOMO ({H:+.3f}) >= LUMO ({L:+.3f})")
    if G is not None and not (0 <= G < 25):
        warns.append(f"{name}: gap {G:.3f} eV out of [0, 25]")
    if M is not None and not charged and not (0 <= M < 25):
        warns.append(f"{name}: |mu| {M:.3f} Debye out of [0, 25]")
    return warns


def _check_consistency(name: str, summary_row: Dict, parsed: Dict) -> List[str]:
    """Compare the stdout re-parse with the raw values stored in the summary."""
    warns: List[str] = []
    for key_sum, key_par in (("raw_homo_eV", "homo_eV"), ("raw_lumo_eV", "lumo_eV")):
        s = _nan_to_none(summary_row.get(key_sum))
        p = _nan_to_none(parsed.get(key_par))
        if s is None and p is None:
            continue
        if (s is None) != (p is None):
            warns.append(f"{name}: {key_sum} present in only one of summary / stdout")
        elif abs(s - p) > CONSISTENCY_TOL_EV:
            warns.append(f"{name}: {key_sum}={s:+.4f} in summary but {p:+.4f} in stdout (stale summary?)")
    return warns


def _check_reference(name: str, res: Dict, tol_eV: float = 1.5, tol_D: float = 2.0) -> List[str]:
    """Compare with literature reference (gas-phase GFN2-xTB). Informational."""
    ref = REFERENCE_VACUUM.get(name.lower())
    if ref is None:
        return []
    warns: List[str] = []
    for key, tol in (("homo_eV", tol_eV), ("lumo_eV", tol_eV),
                     ("gap_eV", tol_eV), ("dipole_D", tol_D)):
        v, rv = res.get(key), ref.get(key)
        if v is None or rv is None:
            continue
        if abs(v - rv) > tol:
            warns.append(f"{name}: {key}={v:+.3f} vs ref {rv:+.3f} (delta={v - rv:+.2f}, tol=+/-{tol})")
    return warns


def _compute_nan_coverage(rows: List[Dict]) -> Dict[str, tuple]:
    """Per-key NaN count over rows that are expected to have values (OK / SUSPECT)."""
    live = [r for r in rows if r["status"] in ("OK", "SUSPECT")]
    out = {}
    for k in ("homo", "lumo", "gap", "dipole"):
        n_nan = sum(1 for r in live if r.get(k) is None)
        out[k] = (len(live), n_nan)
    return out


def _format_text_report(rows: List[Dict], coverage: Dict[str, tuple],
                        warnings: List[str], drift: List[str]) -> str:
    """Human-readable summary for archiving alongside results."""
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = ["=" * 70, "xTB Sanity Report (105b_xtb_sanity_v2.py)", "=" * 70,
             f"Total species    : {len(rows)}"]
    for status in ("OK", "SUSPECT", "QC_FLAGGED", "GEOMETRY_FAILED", "FAIL"):
        lines.append(f"{status:<17s}: {counts.get(status, 0)}")
    lines.append(f"Structural warnings : {len(warnings)}")
    lines.append(f"Informational notes : {len(drift)} (reference drift, geometry notes on masked rows)")
    lines.append("")
    lines.append("NaN coverage (over OK + SUSPECT rows):")
    for k, (n_tot, n_nan) in coverage.items():
        pct = 100.0 * n_nan / max(n_tot, 1)
        lines.append(f"  {k:<10s} {n_nan:>4d}/{n_tot} ({pct:>5.1f}%)")
    flagged = [r for r in rows if r["status"] in ("QC_FLAGGED", "GEOMETRY_FAILED")]
    if flagged:
        lines.append("")
        lines.append(f"Species masked by 104c QC ({len(flagged)}):")
        for r in flagged:
            lines.append(f"  - {r['name']}: {r['qc_flag']}")
    if warnings:
        lines.append("")
        lines.append(f"Structural warnings ({len(warnings)}):")
        lines.extend(f"  - {w}" for w in warnings)
    if drift:
        lines.append("")
        lines.append(f"Informational notes ({len(drift)}):")
        lines.extend(f"  - {w}" for w in drift)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xtb-dir", default=str(DEFAULT_XTB_DIR),
                        help="Directory that holds xtb_results_summary.csv and xtb_runs/ "
                             "(the --out-dir given to 104c).")
    parser.add_argument("--summary", default=SUMMARY_NAME,
                        help="Summary CSV filename, relative to --xtb-dir.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the reports. Defaults to --xtb-dir.")
    parser.add_argument("--report-csv", default="xtb_sanity_summary.csv",
                        help="Per-species CSV (relative to --out-dir). Empty string to skip.")
    parser.add_argument("--report-txt", default="xtb_sanity_report.txt",
                        help="Human-readable report (relative to --out-dir). Empty string to skip.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any structural warning was raised "
                             "(QC-flagged rows and reference drift do not count).")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip reference-value comparison.")
    parser.add_argument("--force", action="store_true",
                        help="Accepted for pipeline-wide --force compatibility. "
                             "No-op: reports are always regenerated.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all checks and print counts; do not write reports.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    xtb_dir = Path(args.xtb_dir)
    out_dir = Path(args.out_dir) if args.out_dir else xtb_dir
    summary_path = xtb_dir / args.summary
    runs_dir = xtb_dir / RUNS_SUBDIR

    logger.info("=" * 70)
    logger.info("105b_xtb_sanity_v2.py | xtb-dir=%s | strict=%s", xtb_dir, args.strict)
    logger.info("=" * 70)

    if not summary_path.is_file():
        logger.error("Summary not found: %s. Run 104c first (--out-dir %s).", summary_path, xtb_dir)
        return 2
    if not runs_dir.is_dir():
        logger.error("Runs directory not found: %s", runs_dir)
        return 2

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    required = {"name", "role", "smiles", "xtb_ok", "homo_eV", "lumo_eV", "gap_eV", "dipole_D"}
    missing_cols = required - set(summary.columns)
    if missing_cols:
        logger.error("Summary is missing columns: %s", sorted(missing_cols))
        return 2
    if summary["name"].duplicated().any():
        dups = summary.loc[summary["name"].duplicated(), "name"].tolist()
        logger.error("Duplicate species names in summary: %s", dups)
        return 2
    logger.info("Loaded %d species from %s", len(summary), summary_path)

    rows: List[Dict] = []
    structural_warnings: List[str] = []
    drift_warnings: List[str] = []

    for rec in summary.to_dict("records"):
        name = str(rec["name"])
        qc_flag = rec.get("qc_flag")
        qc_flag = "" if (qc_flag is None or (isinstance(qc_flag, float) and math.isnan(qc_flag))) else str(qc_flag)
        out_path = runs_dir / f"{name}.xyz.xtb.stdout"
        xyz_path = runs_dir / f"{name}.xyz"
        row = dict(name=name, role=rec["role"], smiles=rec["smiles"], status="OK",
                   qc_flag=qc_flag, homo=None, lumo=None, gap=None, dipole=None,
                   n_atoms=None, notes="")
        notes: List[str] = []

        # Rows that 104c could not even build a geometry for.
        if qc_flag == "geometry_failed":
            row["status"] = "GEOMETRY_FAILED"
            rows.append(row)
            logger.info("  [GEOM_FAIL] %-40s (masked by 104c)", name)
            continue

        # Check 1: files exist
        if not out_path.is_file():
            row["status"] = "FAIL"
            row["notes"] = "no xtb.stdout"
            structural_warnings.append(f"{name}: xtb stdout missing ({out_path.name})")
            rows.append(row)
            logger.warning("  [missing] %s", name)
            continue

        # Check 2: geometry
        n_atoms, coords, geom_warns = _read_xyz(str(xyz_path))
        row["n_atoms"] = n_atoms
        if coords is not None:
            geom_warns += _check_geometry(str(xyz_path), n_atoms, coords)
        for w in geom_warns:
            if qc_flag:
                # Already masked by 104c: record it, but do not count it as a
                # new structural problem (it would trip --strict for no reason).
                logger.info("  [geom]   %s (already masked by 104c: %s)", w, qc_flag)
                drift_warnings.append(f"{w} [masked by 104c: {qc_flag}]")
            else:
                logger.warning("  [geom]   %s", w)
                structural_warnings.append(w)
                notes.append("geometry")
        monatomic = (n_atoms == 1)

        # Parse stdout once; summary values (masked on QC failure) are reported,
        # raw values are used for plausibility / consistency.
        parsed = parse_xtb_out(str(out_path))
        res_raw = dict(homo_eV=_nan_to_none(parsed.get("homo_eV")),
                       lumo_eV=_nan_to_none(parsed.get("lumo_eV")),
                       gap_eV=_nan_to_none(parsed.get("gap_eV")),
                       dipole_D=_nan_to_none(parsed.get("dipole_D")))

        # Check 3: stdout vs summary consistency (needs raw_* columns from 104c)
        if "raw_homo_eV" in summary.columns:
            cons = _check_consistency(name, rec, parsed)
            for w in cons:
                logger.warning("  [stale]  %s", w)
                structural_warnings.append(w)
                notes.append("consistency")

        # Rows masked by 104c QC: expected outcome, not a script failure.
        if qc_flag:
            row["status"] = "QC_FLAGGED"
            row["notes"] = ";".join(notes)
            rows.append(row)
            logger.info("  [QC_FLAG]  %-40s %s", name, qc_flag)
            continue

        # Check 4: plausibility (OK rows only)
        charged = bool(_nan_to_none(rec.get("charge")))
        phys_warns = _check_physical_plausibility(name, res_raw, monatomic, charged)
        for w in phys_warns:
            logger.warning("  [warn]   %s", w)
            structural_warnings.append(w)
            notes.append("plausibility")

        # Check 5: reference drift (informational)
        if not args.no_reference:
            for w in _check_reference(name, res_raw):
                logger.info("  [drift]  %s", w)
                drift_warnings.append(w)

        row.update(homo=_nan_to_none(rec.get("homo_eV")), lumo=_nan_to_none(rec.get("lumo_eV")),
                   gap=_nan_to_none(rec.get("gap_eV")), dipole=_nan_to_none(rec.get("dipole_D")))
        if not bool(rec.get("xtb_ok")):
            notes.append("xtb_ok=False without qc_flag")
            structural_warnings.append(f"{name}: xtb_ok is False but 104c gave no qc_flag")
        row["status"] = "SUSPECT" if notes else "OK"
        row["notes"] = ";".join(notes)
        rows.append(row)

        fmt = lambda v, f: f"{v:{f}}" if v is not None else "  -  "
        logger.info("  [%-8s] %-40s HOMO=%s LUMO=%s gap=%s |mu|=%s", row["status"], name,
                    fmt(row["homo"], "+.4f"), fmt(row["lumo"], "+.4f"),
                    fmt(row["gap"], "+.4f"), fmt(row["dipole"], ".3f"))

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    logger.info("=" * 70)
    logger.info("Total %d | %s | structural warnings %d | drift %d",
                len(rows), " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                len(structural_warnings), len(drift_warnings))
    logger.info("=" * 70)

    coverage = _compute_nan_coverage(rows)
    for key, (n_live, n_nan) in coverage.items():
        logger.info("  %-10s %4d NaN / %d live rows", key, n_nan, n_live)

    if args.dry_run:
        logger.info("[dry-run] Not writing reports.")
        return 1 if (args.strict and structural_warnings) else 0

    if args.report_csv:
        ensure_dir(out_dir)
        csv_path = out_dir / args.report_csv
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in CSV_FIELDS})
        logger.info("Summary CSV: %s", csv_path)
    if args.report_txt:
        ensure_dir(out_dir)
        txt_path = out_dir / args.report_txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(_format_text_report(rows, coverage, structural_warnings, drift_warnings))
        logger.info("Text report: %s", txt_path)

    if structural_warnings and args.strict:
        logger.error("FAILED (strict mode): %d structural warning(s).", len(structural_warnings))
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
