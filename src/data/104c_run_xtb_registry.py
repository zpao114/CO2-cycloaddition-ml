"""104c_run_xtb_registry.py - GFN2-xTB single points for every species the
corrected modeling table actually uses.

Candidates are derived from chem_registry (no hand-maintained list):
  * substrates, CO2, products, solvents  -> one row each (name = snake_case)
  * every catalyst: neutral molecule, or each ionic fragment separately
    (cation / anion), plus the whole neutral salt for metal salts (manual
    geometry) so 107 can fall back to it when a bare metal ion has no orbitals.

Differences from 104b
---------------------
* net charge = RDKit formal charge of the SMILES (104b's regex missed [nH+],
  [OH-], [NH+] ... and mis-assigned e.g. KOH as +1);
* rows are keyed by canonical SMILES, never by a shared display name.
Same engine settings as 104b: GFN2-xTB, --sp, ALPB(DMSO), --acc 1.0, closed shell.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import re
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
HERE = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger  # noqa: E402

import chem_registry as reg  # noqa: E402

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("104c")

METALS = {"Zn", "Mg", "Co", "Fe", "Cu", "Ni", "Al", "Sn", "K", "Na", "Li", "Ca", "Re"}
MIN_PAIR_DIST_A = 0.9


def _load_104b():
    spec = importlib.util.spec_from_file_location("m104b", HERE / "104b_run_xtb_extended.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def canon(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def net_charge(smi: str) -> int:
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smi))


def qc_flag(cand, xyz_path: str, charge: int, res: dict) -> str:
    """Return '' if the closed-shell GFN2 result is physically usable, else a reason.

    Checks: (1) xtb produced HOMO and LUMO; (2) geometry atom count equals the
    SMILES atom count incl. implicit H; (3) no atom pair closer than
    MIN_PAIR_DIST_A (a hand-built or embedded geometry with overlapping atoms
    gives a meaningless energy); (4) even electron count (the run is
    closed-shell, --uhf 0); (5) HOMO-LUMO gap > 0.05 eV (a ~0 gap together with
    HOMO == LUMO means fractional occupation, i.e. an open-shell / degenerate
    system forced into a closed-shell treatment).
    """
    if not res.get("ok") or res.get("homo_eV") is None or res.get("lumo_eV") is None:
        return "no_homo_lumo"
    pt = Chem.GetPeriodicTable()
    with open(xyz_path, encoding="utf-8") as f:
        n = int(f.readline())
        f.readline()
        atoms = [line.split() for line in f if line.strip()]
    z = sum(pt.GetAtomicNumber(a[0]) for a in atoms)
    expected = Chem.AddHs(Chem.MolFromSmiles(cand["smiles"])).GetNumAtoms()
    if n != expected:
        return f"atom_count_{n}_vs_{expected}"
    xyz = [tuple(float(v) for v in a[1:4]) for a in atoms]
    dmin = min((sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
                for a, b in combinations(xyz, 2)), default=None)
    if dmin is not None and dmin < MIN_PAIR_DIST_A:
        return f"atoms_overlap_{dmin:.2f}A"
    if (z - charge) % 2 == 1:
        return "open_shell_odd_electrons"
    gap = res.get("gap_eV")
    if gap is not None and gap < 0.05:
        return "degenerate_gap_lt_0.05eV"
    return ""


def build_candidates():
    cands, seen, used_names = [], set(), set()

    def add(name, smi, role, manual=False):
        key = (canon(smi), role)
        if key in seen:
            return
        seen.add(key)
        if name.lower() in used_names:
            # _slug drops charges and brackets, so distinct species can share a
            # slug; the name keys the .xyz/.stdout files and must be unique.
            q = net_charge(smi)
            name = f"{name}_q{'m' if q < 0 else 'p'}{abs(q)}"
            while name.lower() in used_names:
                name += "_x"
        used_names.add(name.lower())
        c = {"name": name, "smiles": smi, "role": role}
        if manual:
            c["manual_geom"] = True
        cands.append(c)

    for n, e in reg.SUBSTRATES.items():
        add(_slug(n), e.smiles, "substrate")
    add("CO2", "O=C=O", "reactant")
    for n, e in reg.PRODUCTS.items():
        add(_slug(n), e.smiles, "product")
    for n, e in reg.SOLVENTS.items():
        if e.smiles:
            add(_slug(n), e.smiles, "solvent")

    for n, e in reg.CATALYSTS.items():
        if not e.smiles:
            continue
        mol = Chem.MolFromSmiles(e.smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            add(_slug(n), e.smiles, "catalyst")
            continue
        has_metal = any(a.GetSymbol() in METALS for a in mol.GetAtoms())
        for f in frags:
            fs = Chem.MolToSmiles(f)
            q = Chem.GetFormalCharge(f)
            if f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in METALS:
                continue  # bare metal ion: no electrons / orbitals for GFN2 (kept NaN)
            role = "cation" if q > 0 else "anion" if q < 0 else "catalyst"
            add(f"frag_{_slug(fs)}", fs, role)
        if has_metal and e.tier == "exact":
            add(f"salt_{_slug(n)}", e.smiles, "salt", manual=True)
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "smiles_fix_staging"))
    ap.add_argument("--summary", default="xtb_results_summary.csv")
    ap.add_argument("--solvent", default="dmso")
    ap.add_argument("--gfn", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of candidate names (debug)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr, force=True)

    m = _load_104b()
    if m.locate_xtb() is None:
        log.error("xtb not found")
        return 2
    cands = build_candidates()
    if args.only:
        keep = set(args.only.split(","))
        cands = [c for c in cands if c["name"] in keep]
    log.info("%d candidates", len(cands))
    if args.dry_run:
        for c in cands:
            log.info("%-10s %-55s %s q=%+d", c["role"], c["name"][:55], c["smiles"][:70], net_charge(c["smiles"]))
        return 0

    out_dir = Path(args.out_dir)
    run_dir = out_dir / "xtb_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, ok_n = [], 0
    for i, c in enumerate(cands, 1):
        q = net_charge(c["smiles"])
        xyz = str(run_dir / f"{c['name']}.xyz")
        flag = ""
        if not m.smiles_to_xyz(c["smiles"], xyz, manual_geom=c.get("manual_geom", False)):
            log.warning("[%d/%d] geometry failed: %s", i, len(cands), c["name"])
            res = dict(ok=False, homo_eV=None, lumo_eV=None, gap_eV=None, total_e_Eh=None,
                       dipole_D=None, mulliken_q={})
            flag = "geometry_failed"
        else:
            out = m.run_xtb(xyz, q, 1, gfn=args.gfn, solvent=args.solvent or None,
                            level="sp", acc=1.0, timeout=args.timeout)
            res = m.parse_xtb_out(out)
            flag = qc_flag(c, xyz, q, res)
        row = m.summary_row(c, q, args.solvent, args.gfn, res)
        row.update(raw_homo_eV=row["homo_eV"], raw_lumo_eV=row["lumo_eV"],
                   raw_gap_eV=row["gap_eV"], qc_flag=flag)
        if flag:  # unreliable: do not let downstream use the number
            for k in ("homo_eV", "lumo_eV", "gap_eV", "dipole_D"):
                row[k] = None
            row["xtb_ok"] = False
        rows.append(row)
        ok_n += bool(row["xtb_ok"])
        log.info("[%d/%d] %-8s %-45s q=%+d ok=%s qc=%s HOMO=%s LUMO=%s", i, len(cands), c["role"],
                 c["name"][:45], q, row["xtb_ok"], flag or "-",
                 None if res["homo_eV"] is None else round(res["homo_eV"], 3),
                 None if res["lumo_eV"] is None else round(res["lumo_eV"], 3))

    header = list(m.SUMMARY_HEADER) + ["raw_homo_eV", "raw_lumo_eV", "raw_gap_eV", "qc_flag"]
    with open(out_dir / args.summary, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows, %d ok)", out_dir / args.summary, len(rows), ok_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
