# -*- coding: utf-8 -*-
"""
101_clean.py — CO2 cycloaddition raw-data cleaner.

Inputs
------
data/raw/CO2_cycloaddition_merged.csv          (Reaxys-style merged CSV, GBK-encoded)

Outputs
-------
data/processed/cleaned.csv                     (cleaned master table)
data/external/extraction_report.csv            (rows that failed extraction)
data/external/discard_report.csv               (rows that were reasonably discarded)
data/external/cleaned_baseline.json            (run statistics)

Pipeline position
-----------------
TIER 1 first step. Produces ``cleaned.csv`` consumed by 102_smiles, then
103_drfp, 104b_run_xtb, 105b_xtb_sanity and 107_merge.

Encoding
--------
The raw CSV is encoded in GBK / GB2312 / GB18030. The loader tries each of
these in order, falling back to UTF-8, then Latin-1.

Extraction rule: a row is kept if ALL critical info is PRESENT
----------------------------------------------------------------
Both pure Reaxys records ('With A, B in S, Time=...') and prose-led rows with
a reagent-style 'With...' tail are parsed.  A row is kept only when ALL of the
following hold, otherwise it is discarded:

  * the substrate is one of the supported epoxides;
  * the yield is a usable number (exact, approximate 'Ca.', bound '> 99 %', or
    midpoint of a range 'a - b %'; no masses or text-only entries);
  * every reagent equals a CAT_NORMALIZE key AND that key has an exact structure
    in chem_registry.py (matching is whole-token, so polymer/silica/MOF-
    supported catalysts never collapse to a known substring like 'pyridine');
  * the solvent clause is fully resolved;
  * Time / T / p: any of the three missing is fine (Reaxys gaps), but present
    ones must be single numbers with units (ranges and stages are rejected);
  * heating is conventional (no light, microwave, electrolysis, flow).

Rows that start with free procedure prose pass when their tail-only parse
succeeds (the tail lists what went into the flask).
Duplicates are kept once: rows with identical Reaction/Yield/Conditions, and
rows whose parsed output record is identical in every column.
Rows rejected only because one short, plain reagent name is unknown go to
extraction_report.csv; all others go to discard_report.csv.

Usage
-----
    python 101_clean.py
    python 101_clean.py --force                 # ignore existing cleaned.csv
    python 101_clean.py --input <other.csv>     # custom input
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from src.paths import (
        CLEANED_BASELINE,
        CLEANED_CSV,
        DATA_EXTERNAL,
        DISCARD_REPORT,
        EXTRACTION_REPORT,
        RAW_REAXYS_CSV,
        ensure_dir,
    )
except Exception:  # noqa: BLE001
    _ROOT = Path(os.environ.get("CO2_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))))
    RAW_REAXYS_CSV = _ROOT / "data" / "raw" / "CO2_cycloaddition_merged.csv"
    CLEANED_CSV   = _ROOT / "data" / "processed" / "cleaned.csv"
    EXTRACTION_REPORT = _ROOT / "data" / "external" / "extraction_report.csv"
    DISCARD_REPORT    = _ROOT / "data" / "external" / "discard_report.csv"
    CLEANED_BASELINE  = _ROOT / "data" / "external" / "cleaned_baseline.json"
    DATA_EXTERNAL     = _ROOT / "data" / "external"
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

INPUT_ENCODINGS = ("gbk", "gb2312", "gb18030", "utf-8", "latin-1")

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logger = logging.getLogger("101_clean")


# Garbled variants preserved here because the raw CSV contains Unicode
# hyphen/dash characters that get corrupted to '?' during Excel→CSV export.
SUBSTRATE_MAP = {
    "styrene oxide":              ("Styrene oxide",              "4-phenyl-1,3-dioxolan-2-one"),
    "氧化苯乙烯":                ("Styrene oxide",              "4-phenyl-1,3-dioxolan-2-one"),
    "propylene oxide":           ("Propylene oxide",            "4-methyl-1,3-dioxolan-2-one"),
    "甲基环氧乙烷":              ("Propylene oxide",            "4-methyl-1,3-dioxolan-2-one"),
    "epichlorohydrin":          ("Epichlorohydrin",           "4-(chloromethyl)-1,3-dioxolan-2-one"),
    "环氧氯丙烷":                ("Epichlorohydrin",           "4-(chloromethyl)-1,3-dioxolan-2-one"),
    "cyclohexene oxide":        ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烯环氧":                ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷?1,2?环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷?1，2?环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "环己烷-1,2-环氧乙烷":      ("Cyclohexene oxide",         "hexahydrobenzo[d][1,3]dioxol-2-one"),
    "isopropyl glycidyl ether": ("Isopropyl glycidyl ether", "4-(isopropoxymethyl)-1,3-dioxolan-2-one"),
    "缩水甘油异丙醚":            ("Isopropyl glycidyl ether", "4-(isopropoxymethyl)-1,3-dioxolan-2-one"),
    "glycidyl phenyl ether":    ("Glycidyl phenyl ether",     "4-phenoxymethyl-1,3-dioxolan-2-one"),
}


# NOTE: photo/LED patterns must use word boundaries, otherwise matching
# "placed" / "flocculated" / "Led (author name)" causes false positives.
CAT_BLACKLIST_PATTERNS = [
    re.compile(r"zeolitic imidazolate framework", re.I),
    re.compile(r"\bZIF-\d+\b", re.I),
    re.compile(r"\bMOF-\d+\b", re.I),
    re.compile(r"\bMIL-\d+\b", re.I),
    re.compile(r"metal[- ]organic[- ]framework", re.I),
    re.compile(r"polymer[- ]supported", re.I),
    re.compile(r"grafted\s+onto", re.I),
    re.compile(r"grafted\s+with", re.I),
    re.compile(r"on\s+divinylbenzene", re.I),
    re.compile(r"PDVB", re.I),
    re.compile(r"poly[-\s]?divinylbenzene", re.I),
    re.compile(r"polyethyleneimine", re.I),
    re.compile(r"\bLED\b.*?(?:light|irradiation|lamp|Visible|photo|可见|光照|紫外)", re.I),
    re.compile(r"(?:light|irradiation|lamp|photo|紫外)\b.*?\bLED\b", re.I),
    re.compile(r"photo[- ]catal", re.I),
    re.compile(r"UV[- ]irradiation", re.I),
    re.compile(r"visible\s+light", re.I),
    re.compile(r"LED\s+irradiation", re.I),
    re.compile(r"electro[- ]catal", re.I),
]


# Extending this table is the canonical way to capture new literature mentions.
CAT_NORMALIZE: Dict[str, str] = {
    "tetrabutylammonium bromide":        "tetrabutylammonium bromide",
    "tetrabutylammomium bromide":       "tetrabutylammonium bromide",
    "tetrabutylammonium iodide":         "tetrabutylammonium iodide",
    "tetrabutylammonium chloride":       "tetrabutylammonium chloride",
    "tetrabutylammonium fluoride":        "tetrabutylammonium fluoride",
    "tetrabutylammonium hydroxide":       "tetrabutylammonium hydroxide",
    "TBAB":                             "tetrabutylammonium bromide",
    "TBAC":                             "tetrabutylammonium chloride",
    "TBAF":                             "tetrabutylammonium fluoride",
    "TBAI":                             "tetra-(n-butyl)ammonium iodide",
    "n-Bu4NBr":                        "tetrabutylammonium bromide",
    "n-Bu4NI":                         "tetra-(n-butyl)ammonium iodide",
    "tetra-n-propylammonium bromide":      "tetra-n-propylammonium bromide",
    "tetraoctyl ammonium bromide":       "tetraoctyl ammonium bromide",
    "tetrahexylammonium bromide":        "tetrahexylammonium bromide",
    "methyltributylammonium bromide":    "methyltributylammonium bromide",
    "benzyl triethylammonium bromide":   "benzyl triethylammonium bromide",
    "benzyltrimethylammonium bromide":   "benzyltrimethylammonium bromide",
    "tetramethylammonium bromide":        "tetramethylammonium bromide",
    "tetraethylammonium bromide":        "tetraethylammonium bromide",
    "choline iodide":                   "choline iodide",
    "ChI":                             "choline iodide",
    "BMIM":                            "BMIM",
    "[BMIM]Br":                        "[BMIM]Br",
    "[BMIM]Cl":                        "[BMIM]Cl",
    "[BMIM]BF4":                       "[BMIM]BF4",
    "[BMIM]PF6":                       "[BMIM]PF6",
    "EMIM":                            "EMIM",
    "[EMIM]Br":                        "[EMIM]Br",
    "[EMIM]Cl":                        "[EMIM]Cl",
    "HMIM":                            "HMIM",
    "[Py]Br":                          "[Py]Br",
    "1-ethyl-3-methylimidazolium bromide":    "1-ethyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium bromide":    "1-butyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium chloride":  "1-butyl-3-methylimidazolium chloride",
    "1-butyl-3-methylimidazolium iodide":     "1-butyl-3-methylimidazolium iodide",
    "imidazolium bromide":              "imidazolium bromide",
    "imidazolium chloride":             "imidazolium chloride",
    "zinc chloride":                   "zinc chloride",
    "zinc dibromide":                  "zinc dibromide",
    "zinc iodide":                     "zinc iodide",
    "zinc acetate":                   "zinc acetate",
    "ZnCl2":                          "zinc chloride",
    "ZnBr2":                          "zinc dibromide",
    "ZnI2":                           "zinc diiodide",
    "Zn(OAc)2":                      "zinc acetate",
    "Zn(OTf)2":                      "Zn(OTf)2",
    "sodium chloride":                 "sodium chloride",
    "sodium iodide":                   "sodium iodide",
    "NaCl":                           "sodium chloride",
    "NaI":                            "sodium iodide",
    "potassium iodide":                 "potassium iodide",
    "KI":                             "potassium iodide",
    "lithium bromide":                 "lithium bromide",
    "LiBr":                           "lithium bromide",
    "cobalt chloride":                "cobalt chloride",
    "CoCl2":                          "cobalt chloride",
    "CoBr2":                          "cobalt bromide",
    "iron(III) chloride":              "iron(III) chloride",
    "FeCl3":                          "iron(III) chloride",
    "FeCl2":                          "iron(II) chloride",
    "copper(II) chloride":             "copper(II) chloride",
    "CuCl2":                          "copper(II) chloride",
    "CuBr2":                          "copper(II) bromide",
    "nickel(II) chloride":             "nickel(II) chloride",
    "NiCl2":                          "nickel(II) chloride",
    "magnesium chloride":             "magnesium chloride",
    "MgCl2":                         "magnesium chloride",
    "calcium chloride":               "calcium chloride",
    "CaCl2":                         "calcium chloride",
    "aluminum chloride":              "aluminum chloride",
    "AlCl3":                        "aluminum chloride",
    "tin(IV) chloride":              "tin(IV) chloride",
    "SnCl4":                        "tin(IV) chloride",
    "tetraphenylphosphonium bromide": "tetraphenylphosphonium bromide",
    "tetraphenylphosphonium":        "tetraphenylphosphonium",
    "triphenylphosphine":            "triphenylphosphine",
    "DBU":                          "1,8-diazabicyclo[5.4.0]undec-7-ene",
    "DMAP":                         "DMAP",
    "DABCO":                        "DABCO",
    "1,4-diazabicyclo[2.2.2]octane": "1,4-diazabicyclo[2.2.2]octane",
    "triethylamine":                "triethylamine",
    "pyridine":                    "pyridine",
    "1,8-diazabicyclo[5.4.0]undec-7-ene": "1,8-diazabicyclo[5.4.0]undec-7-ene",
    "1,1,3,3-tetramethylguanidine":  "1,1,3,3-tetramethylguanidine",
    "ethylenediaminetetraacetic acid": "ethylenediaminetetraacetic acid",
    "EDTA":                         "ethylenediaminetetraacetic acid",
    "silanol catalyst":              "silanol catalyst",
    "Ph3P":                        "triphenylphosphine",
    "tetra-(n-butyl)ammonium iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)ammonium":              "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)ammonium bromide":      "tetrabutylammonium bromide",
    "tetra-(n-butyl)ammonium chloride":      "tetrabutylammonium chloride",
    "tetra-(n-butyl)ammonium fluoride":      "tetrabutylammonium fluoride",
    "tetra-(n-butyl)ammonium hydroxide":     "tetrabutylammonium hydroxide",
    "tetra-(n-butyl)ammoni um iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra-(n-butyl)amm onium iodide":       "tetra-(n-butyl)ammonium iodide",
    "tetra(n-butyl)ammonium":                "tetra-(n-butyl)ammonium iodide",
    "tetra(n-butyl)ammonium hydroxide":       "tetra-(n-butyl)ammonium hydroxide",
    "tetra-n-butylammonium iodide":          "tetra-(n-butyl)ammonium iodide",
    "tetrabutyl-ammonium iodide":            "tetra-(n-butyl)ammonium iodide",
    "tetrabutyl-ammonium chloride":          "tetrabutylammonium chloride",
    "tetrabutyl-ammonium bromide":           "tetrabutylammonium bromide",
    "tetrabutylammonium iodide":             "tetra-(n-butyl)ammonium iodide",
    "tetrabutylammonium hydrogensulfate":    "tetrabutylammonium hydrogensulfate",
    "n-tetrabutylammonium iodide":           "tetra-(n-butyl)ammonium iodide",
    "tri-n-butyl(2-hydroxyethyl)phosphonium iodide":  "tri-n-butyl(2-hydroxyethyl)phosphonium iodide",
    "tri-n-butyl(2-hydroxyethyl)ammonium iodide":     "tri-n-butyl(2-hydroxyethyl)ammonium iodide",
    "tri-n-butyl(2-hydroxyethyl)ammonium":             "tri-n-butyl(2-hydroxyethyl)ammonium iodide",
    "tetraheptylammonium bromide":           "tetraheptylammonium bromide",
    "tetraoctylammonium bromide":            "tetraoctylammonium bromide",
    "tetrahexylammonium bromide":            "tetrahexylammonium bromide",
    "methyltrioctylammonium hydroquinolate": "methyltrioctylammonium hydroquinolate",
    "methyltributylammonium bromide":        "methyltributylammonium bromide",
    "tetramethylammonium picolinate":       "tetramethylammonium picolinate",
    "phenyltrimethylammonium tribromide":   "phenyltrimethylammonium tribromide",
    "benzyltrimethylammonium bromide":       "benzyltrimethylammonium bromide",
    "benzyltriethylammonium bromide":       "benzyl triethylammonium bromide",
    "octyl-(2-hydroxyethyl)-dimethylammonium bromide": "octyl-(2-hydroxyethyl)-dimethylammonium bromide",
    "octyl(2-hydroxyethyl)dimethylammonium bromide":   "octyl-(2-hydroxyethyl)-dimethylammonium bromide",
    "bis(2-hydroxyethyl) bis-(quaternary ammonium)iodide": "bis(2-hydroxyethyl) bis-(quaternary ammonium)iodide",
    "2-morpholinoethanol hydriodide":       "2-morpholinoethanol hydriodide",
    "2-morpholinoethanol hydroiodide":      "2-morpholinoethanol hydriodide",
    "2-morpholinoethylammonium iodide":      "2-morpholinoethanol hydriodide",
    "tetrahydrofuran-2,5-dimethanol":        "tetrahydrofuran-2,5-dimethanol",
    "PEG6000(NBu3Br)2":                    "PEG6000(NBu3Br)2",
    "BrTBDPEG150TBDBr":                   "BrTBDPEG150TBDBr",
    "choline chloride-PEG600":             "choline chloride-PEG600",
    "triethanolammonium iodide":            "triethanolammonium iodide",
    "triethanolammonium":                  "triethanolammonium iodide",
    "tetra-n-propylammonium bromide":       "tetra-n-propylammonium bromide",
    "tri-n-butylamine":                    "tri-n-butylamine",
    "tributylamine":                       "tri-n-butylamine",
    "1-n-butyl-3-methylimidazolim bromide": "1-butyl-3-methylimidazolium bromide",
    "1-n-butyl-3-methylimidazolium bromide": "1-butyl-3-methylimidazolium bromide",
    "1-butyl-3-methylimidazolium alanine": "1-butyl-3-methylimidazolium alanine",
    "1-hexyl-3-methylimidazolium tetrafluoroborate": "1-hexyl-3-methylimidazolium tetrafluoroborate",
    "1-hexyl-3-methylimidazolium hydrogencarbonat": "1-hexyl-3-methylimidazolium hydrogencarbonate",
    "1-methyl-3-(3-phenylthioureido)pyridinium iodide": "1-methyl-3-(3-phenylthioureido)pyridinium iodide",
    "1-methyl-1-N-propylpyrrolidinium bromide": "1-methyl-1-N-propylpyrrolidinium bromide",
    "1-ethyl-1-methylpyrrolidinium bromide": "1-ethyl-1-methylpyrrolidinium bromide",
    "1-butyl-1-methylpiperidinium bromide": "1-butyl-1-methylpiperidinium bromide",
    "1-methylpiperidinium bromide": "1-methylpiperidinium bromide",
    "3-hydroxy-N-octylpyridinium iodide": "3-hydroxy-N-octylpyridinium iodide",
    "N,N-dimethyl-N(pyrenyl-1-methyl) dodecan-1-ammonium bromide": "N,N-dimethyl-N(pyrenyl-1-methyl) dodecan-1-ammonium bromide",
    "pyridinium": "pyridinium",
    "pyridinium-": "pyridinium",
    "ethylpyridin-1-ium bromide": "ethylpyridinium bromide",
    "1-methylimidazolium bromide": "1-methylimidazolium bromide",
    "1-methylimidazolium": "1-methylimidazolium",
    "imidazolium": "imidazolium",
    "imidazolium alanine": "imidazolium alanine",
    "tetraphenylstibonium bromide": "tetraphenylstibonium bromide",
    "zinc(II)-1,10-phenanthroline-5,6-dione": "zinc phenanthroline dione",
    "zinc(II)-1,10-phenanthroline-5,6-dione-": "zinc phenanthroline dione",
    "zinc phenanthroline dione": "zinc phenanthroline dione",
    "zinc-phenanthroline": "zinc phenanthroline dione",
    "cobalt(II) nitrate hexahydrate": "cobalt(II) nitrate",
    "Co(NO3)2*6H2O": "cobalt(II) nitrate",
    "sodium bromide": "sodium bromide",
    "NaBr": "sodium bromide",
    "tetrahydroxydiboron": "tetrahydroxydiboron",
    "tetrahydroxy diboron": "tetrahydroxydiboron",
    "2,4,6-triamino-s-triazine": "2,4,6-triamino-s-triazine",
    "2,4,6-triamino-s-triazine; zinc(II) iodide": "2,4,6-triamino-s-triazine; zinc iodide",
    "magnesium oxide": "magnesium oxide",
    "MgO": "magnesium oxide",
    "copper cobaltite": "copper cobaltite",
    "cobalt-impregnated 2D-": "cobalt-impregnated 2D material",
    "bromopentacarbonylrhenium": "bromopentacarbonylrhenium",
    "polyethylene glycol-supported hexaalkylguanidinium bromide": "polyethylene glycol-supported hexaalkylguanidinium bromide",
    "R,R-Jacobsen's Co-I salen catalyst": "Jacobsen Co-salen",
    "Jacobsen Co-salen": "Jacobsen Co-salen",
    "salen cobalt": "Jacobsen Co-salen",
    "salen": "salen",
    "Al{C5H2O(O)(O)CH2OH}3": "Al-salan complex",
    "aluminum salen": "aluminum salen",
    "(triphenylphosphoranylidene)acetaldehyde": "(triphenylphosphoranylidene)acetaldehyde",
    "triphenylphosphoranylideneacetaldehyde": "(triphenylphosphoranylidene)acetaldehyde",
    "N-bromosuccinimide": "N-bromosuccinimide",
    "NBS": "N-bromosuccinimide",
    "dibenzoyl peroxide": "dibenzoyl peroxide",
    "dibenzoyl peroxide; N-bromosuccinimide": "dibenzoyl peroxide; N-bromosuccinimide",
    "2,5-dihydroxyterephthalohydrazide": "2,5-dihydroxyterephthalohydrazide",
    "18-crown-6 ether": "18-crown-6 ether",
    "18-crown-6": "18-crown-6 ether",
    "crown ether": "18-crown-6 ether",
    "L-arginine": "L-arginine",
    "L-arginine;": "L-arginine",
    "N-trimethylanilinium bromide": "N-trimethylanilinium bromide",
    "N-trimethylethan-1-aminium iodide": "N-trimethylethan-1-aminium iodide",
    "4-dimethylaminopyridine": "4-dimethylaminopyridine",
    "4-dimethylamino-N-iodopyridinium bromide": "4-dimethylamino-N-iodopyridinium bromide",
    # Spellings found in the raw Conditions text (verified row by row)
    "TBABr":                              "tetrabutylammonium bromide",
    "tetrabutylammoniumbromide":          "tetrabutylammonium bromide",
    "tetrabutylammoniumchloride":         "tetrabutylammonium chloride",
    "tetra-n-tertbutylammonium bromide":  "tetrabutylammonium bromide",
    # Discrete, fully specified compounds that occur as stated reagents in the
    # raw records (each has an exact structure in chem_registry.py).
    "water":                              "water",
    "calcium iodide":                     "calcium iodide",
    "zinc(II) chloride":                  "zinc chloride",
    "zinc(II) iodide":                    "zinc iodide",
    "potassium bromide":                  "potassium bromide",
    "indium(III) bromide":                "indium(III) bromide",
    "chromium dichloride":                "chromium dichloride",
    "choline chloride":                   "choline chloride",
    "tetrabutylphosphonium bromide":      "tetrabutylphosphonium bromide",
    "tetrabutyl phosphonium bromide":     "tetrabutylphosphonium bromide",
    "tetrabutyl ammonium fluoride":       "tetrabutylammonium fluoride",
    "tetrabutyl ammonium bromide":        "tetrabutylammonium bromide",
    "1,1,3,3-tetramethylguanidine hydrobromide": "1,1,3,3-tetramethylguanidine hydrobromide",
    "MTBD":                               "MTBD",
    "TBD":                                "TBD",
    "biphenyl":                           "biphenyl",
    "benzyl bromide":                     "benzyl bromide",
    "phenol":                             "phenol",
    "4-methoxy-phenol":                   "4-methoxyphenol",
    "4-methoxyphenol":                    "4-methoxyphenol",
    "3-(aminomethyl)phenol":              "3-(aminomethyl)phenol",
    "2-Hydroxymethylpyridine":            "2-pyridinemethanol",
    "2-Picolinic acid":                   "picolinic acid",
    "3-Dimethylamino-1-propanol":         "3-dimethylamino-1-propanol",
    "acetic acid":                        "acetic acid",
    "formic acid":                        "formic acid",
    "glycerol":                           "glycerol",
    "Pentaerythritol":                    "pentaerythritol",
    "betaine":                            "betaine",
    "L-histidine":                        "L-histidine",
    "L-phenylalanine":                    "L-phenylalanine",
    "tris(4-bromophenyl)amine":           "tris(4-bromophenyl)amine",
    "1,2,4,5-tetrakis(bromomethyl)benzene": "1,2,4,5-tetrakis(bromomethyl)benzene",
    "1-(bromomethyl)-4-(1,1-dimethylethyl)benzene": "4-tert-butylbenzyl bromide",
    "1-benzyl-3-butylimidazolium bromide": "1-benzyl-3-butylimidazolium bromide",
    "1-octadecyl-3-methylimidazolium bromide": "1-octadecyl-3-methylimidazolium bromide",
    "1-(3-aminopropyl)-3-butylimidazolium iodide": "1-(3-aminopropyl)-3-butylimidazolium iodide",
    "1-ethyl-3-methylimidazolium chloride": "1-ethyl-3-methylimidazolium chloride",
    "1-ethyl-3-methylimidazolium iodide": "1-ethyl-3-methylimidazolium iodide",
}


SOLVENT_NORMALIZE = {
    "neat":               "无溶剂",
    "no solvent":         "无溶剂",
    "acetonitrile":       "acetonitrile",
    "MeCN":              "acetonitrile",
    "methanol":          "methanol",
    "MeOH":              "methanol",
    "ethanol":           "ethanol",
    "isopropanol":       "isopropanol",
    "water":             "water",
    "DMF":              "DMF",
    "toluene":           "toluene",
    "chloroform":         "chloroform",
    "dichloromethane":    "dichloromethane",
    "DCM":               "dichloromethane",
    "DMSO":              "DMSO",
    "acetone":           "acetone",
    "THF":              "THF",
    "1,4-dioxane":      "1,4-dioxane",
    "ethyl acetate":      "ethyl acetate",
    "hexane":             "hexane",
    "cyclohexane":       "cyclohexane",
    "diethyl ether":     "diethyl ether",
    "chlorobenzene":      "chlorobenzene",
    "1-butanol":        "1-butanol",
    "2-butanol":        "2-butanol",
    "supercritical CO2": "supercritical CO2",
    "scCO2":             "supercritical CO2",
    "ionic liquid":      "ionic liquid (IL)",
    # Long-form names exactly as the raw Reaxys records write them
    "N,N-dimethyl-formamide":      "DMF",
    "dimethyl sulfoxide":          "DMSO",
    "dimethylsulfoxide":           "DMSO",
    "tetrahydrofuran":             "THF",
    "isopropyl alcohol":           "isopropanol",
    "1,2-dichloro-ethane":         "1,2-dichloroethane",
    "1,2-dichloro-benzene":        "1,2-dichlorobenzene",
    "1,2-dichloro- ethane":        "1,2-dichloroethane",
    "butanone":                    "butanone",
    "benzonitrile":                "benzonitrile",
    "1-methyl-pyrrolidin-2-one":   "N-methyl-2-pyrrolidone",
    "1,2-dimethoxyethane":         "1,2-dimethoxyethane",
    "dimethyl carbonate":          "dimethyl carbonate",
    "n-heptane":                   "n-heptane",
    "1,3,5-trimethyl-benzene":     "mesitylene",
    "2-methyltetrahydrofuran":     "2-methyltetrahydrofuran",
    "decane":                      "decane",
}


IONIC_KW = [
    "BMIM","EMIM","HMIM","OMIM","DMIM","AMIM",
    "[BMIM]","[EMIM]","[HMIM]","[OMIM]","[DMIM]","[AMIM]",
    "imidazolium","1-butyl-3-methylimidazolium","1-ethyl-3-methylimidazolium",
    "1-hexyl-3-methylimidazolium","1-octyl-3-methylimidazolium",
    "1-methyl-3-octylimidazolium","1-allyl-3-methylimidazolium",
    "1-benzyl-3-methylimidazolium","1-methyl-3-propylimidazolium",
    "[Py]","pyridinium","N-methylpyridinium","N-ethylpyridinium",
    "N-butylpyridinium","N-octylpyridinium",
    "tetrabutylammonium","tetraphenylphosphonium","tetraphenylstibonium",
    "tetraoctylammonium","tetrahexylammonium","tetraheptylammonium",
    "tetra-n-butylammonium","tetra-n-propylammonium","tetraethylammonium",
    "tetramethylammonium","methyltributylammonium","methyltrioctylammonium",
    "benzyl triethylammonium","benzyltrimethylammonium","choline",
    "triethanolammonium","tri-n-butyl(2-hydroxyethyl)ammonium",
    "tri-n-butyl(2-hydroxyethyl)phosphonium",
    "guanidinium","1,1,3,3-tetramethylguanidine","tetramethylguanidine",
    "phosphonium","butyl(triphenyl)phosphonium","methyltriphenylphosphonium",
    "tributylbenzylphosphonium","dodecyltributylphosphonium",
    "TBAB","TBAI","TBAC","TBAF","TBAC","TBD","MTBD","DBU","DMAP","DABCO",
    "tetra-(n-butyl)ammonium","pyrrolidinium","N-methylpyrrolidinium","N-ethylpyrrolidinium",
]
METAL_KW = [
    "zinc","sodium","potassium","lithium","cobalt","iron","copper",
    "nickel","magnesium","calcium","aluminum","tin","manganese","indium","lanthanum",
    "ZnCl","ZnBr","ZnI","NaCl","NaI","KCl","KI","LiCl","LiBr",
    "FeCl","CoCl","CuCl","NiCl","MgCl","MgBr","CaCl","AlCl","SnCl",
    "ZnO","MgO","Al2O3","CaO","TiO2","aluminium",
    "Zn(","Na(","K(","Co(","Fe(","Cu(","Ni(","Mg(","Sn(",
]
# Onium-salt stems matched as substrings (no letter-boundary): systematic IUPAC
# names embed the cation stem after a prefix, e.g. "3-butylimidazolium",
# "tetrabutylphosphonium", "pyrazol-2-ium", "hydrobromide" amine salts.
ONIUM_STEM_RE = re.compile(
    r"imidazol(?:ium|um|-\d-ium)|imidazole\s+(?:bromide|chloride|iodide)"
    r"|pyrazol(?:ium|-\d-ium)|pyrazole\s+(?:bromide|chloride|iodide|hydrobromide)"
    r"|ammonium|ammonio|aminium|anilinium|phosphonium|stibonium|sulfonium"
    r"|pyridinium|pyrrolidinium|guanidinium|cyclopropenium|azonia"
    r"|\d-ium\b|betaine|hydrobromide|hydriodide|hydrochloride|hydroiodide"
)
BASE_KW = [
    "DBU","DMAP","DABCO","TBD","MTBD","triethylamine","pyridine",
    "tri-n-butylamine","tri-n-butylamine","tributylamine","dimethylamine","diethylamine",
    "dimethylpyridine","iminopyridine","aniline",
    "1,8-diazabicyclo[5.4.0]undec-7-ene","diazabicyclo","triphenylphosphine","triphenylphosphine oxide",
]


OUT_COLS = [
    "row_id", "reactant_name", "product_name", "yield (%)",
    "temperature (\u2103)", "pressure (MPa)", "time (h)",
    "catalyst_1_name", "catalyst_1_loading_mol%",
    "catalyst_2_name", "catalyst_2_loading_mol%",
    "catalyst_3_name", "catalyst_3_loading_mol%",
    "catalyst_4_name", "catalyst_4_loading_mol%",
    "reagent_1_name", "reagent_2_name",
    "solvent_1", "solvent_2", "solvent_3", "solvent_4",
    "reference", "publication_year", "extraction_status", "catalyst_system_type",
    "all_solvents_normalized",
]
EXTR_COLS = ["source_row_id", "discard_reason", "discard_type",
             "affected_field", "raw_value", "suggested_fix"]
DISC_COLS = [
    "row_id",
    "catalyst_1_name", "catalyst_2_name", "catalyst_3_name", "catalyst_4_name",
    "catalyst_1_smiles", "catalyst_2_smiles", "catalyst_3_smiles", "catalyst_4_smiles",
    "discard_reason", "discard_type",
]


def _v(v) -> str:
    return "" if v is None else v


_DASHES = "\u2010\u2011\u2012\u2013\u2212\u00ad"
_BREAK_HYPHEN = re.compile("[" + _DASHES + r"]\s+(?=[a-z])")
_DASH_CHARS = re.compile("[" + _DASHES + "]")


def normalize_text(s: str) -> str:
    """Repair PDF/Reaxys text artefacts before any matching.

    The raw export writes every hyphen as U+2010 and leaves line-break hyphens
    inside words ('ammoni\u2010 um iodide', 'tetra\u2010 butyl').  A hyphen followed
    by whitespace and a lower-case letter is a line break: join the word.
    Remaining Unicode dashes become ASCII '-', which is what CAT_NORMALIZE uses.
    """
    if not s:
        return s
    s = _BREAK_HYPHEN.sub("", s)
    return _DASH_CHARS.sub("-", s)


_NUM = r"(\d+(?:\.\d+)?)"
_YIELD_RE = re.compile(
    r"^(?P<pre>ca\.?|>|≥)?" + _NUM + r"(?:-" + _NUM + r")?(?P<pct>%)?(?:spectr|chro-?mat|nmr|gc|hplc|isol)?\.?$", re.I)


def parse_yield(v) -> Tuple[Optional[float], str]:
    """Return (yield_percent, '') or (None, reason).

    Taken at face value: '92.9 %Chromat.', '70 %Spectr.', '0.94 Spectr.', '0.8'
    (a bare number <= 1 is a fraction).  Approximate / bounded values keep the
    number that is written: 'Ca. 80 %' -> 80, '> 99 %' -> 99; a range 'a - b %'
    -> its midpoint.  No yield is invented: '< 5 %' (upper bound only), a mass in
    g, and texts without a number ('Chromat.', '.') have no usable yield.
    """
    if v is None:
        return None, "empty"
    s = normalize_text(str(v).strip())
    compact = re.sub(r"\s+", "", s)
    m = _YIELD_RE.match(compact)
    if not m:
        return None, f"no usable yield: {s[:40]!r}"
    lo, hi, pct = float(m.group(2)), m.group(3), m.group("pct")
    num = (lo + float(hi)) / 2.0 if hi else lo
    if not pct:
        if num > 1.0:
            return None, f"bare number >1 without %: {s[:40]!r}"
        num *= 100.0
    if num > 100.0:
        return None, f">100%: {s[:40]!r}"
    return round(num, 2), ""


# A reagent of the Reaxys record is accepted ONLY when its whole text equals a
# dictionary key AND the canonical name has an exact, complete structure in
# chem_registry.py.  There is no substring matching: a longer name that merely
# contains a known word ('pyridine' in 'poly(4-vinylpyridine)supported iodine')
# can therefore never be reduced to that word.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chem_registry as _reg  # noqa: E402

# Keys whose canonical name would add information the source text does not
# give (a guessed counter-ion) or that name only a cation.
_AMBIGUOUS_KEYS = {
    "BMIM", "EMIM", "HMIM", "imidazolium", "1-methylimidazolium", "pyridinium",
    "pyridinium-", "tetraphenylphosphonium", "tetra-(n-butyl)ammonium",
    "tetra(n-butyl)ammonium", "tri-n-butyl(2-hydroxyethyl)ammonium",
    "triethanolammonium",
}


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _usable_canonical(name: str) -> bool:
    e = _reg.REGISTRY.get(name)
    return bool(e and e.tier == "exact" and e.smiles and "cation only" not in (e.note or "").lower())


def _build_cat_lookup() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in CAT_NORMALIZE.items():
        if k in _AMBIGUOUS_KEYS or not _usable_canonical(v):
            continue
        out[_norm_key(k)] = v
    # every exact registry catalyst is also reachable under its own registry name
    for k in _reg.CATALYSTS:
        if k not in _AMBIGUOUS_KEYS and _usable_canonical(k):
            out.setdefault(_norm_key(k), k)
    return out


_CAT_LOOKUP: Dict[str, str] = {}
_SOLV_LOOKUP: Dict[str, str] = {}


def _init_lookups() -> None:
    _CAT_LOOKUP.clear(); _CAT_LOOKUP.update(_build_cat_lookup())
    _SOLV_LOOKUP.clear()
    for k, v in SOLVENT_NORMALIZE.items():
        if v == "无溶剂" or (v in _reg.REGISTRY and _reg.REGISTRY[v].tier == "exact"):
            _SOLV_LOOKUP[_norm_key(k)] = v


_MATERIAL_RE = re.compile(
    r"poly|support|immobili|graft|framework|\bMOF\b|resin|silica|MCM|SBA|composite|@|nano|"
    r"cellulose|zeolite|\bCOF\b|covalent|mesopor|porous|Amberlyst|loaded|impregnat|modified|"
    r"functionali[sz]ed|hybrid|\bon\b|\bgel\b|membrane|carbon(?!ate|yl|\s+dioxide|\s+tetra)|graphene|clay|magnetic|Fe3O4|"
    r"chitosan|\bPS-|\bPEG|\bIL-|ionic liquid|cross-linked|crosslinked|lattice|network", re.I)

# Narrative text only: heterogeneous supports / polymers / frameworks / immobilised catalysts.
# (Narrower than _MATERIAL_RE, which is meant for a single reagent name, not for prose.)
_PROSE_MATERIAL_RE = re.compile(
    r"polymer|polystyrene|supported\b|immobili[sz]|graft|framework|\bMOFs?\b|\bCOFs?\b|\bZIF\b|"
    r"resin|\bMCM|\bSBA|composite|zeolite|mesoporous|nanoparticle|nanotube|nanosheet|"
    r"cellulose|chitosan|graphene|heterogeneous|cross-?linked", re.I)

# Raw molecular-formula / coordination-compound text ('C24H32Br2CoN6', 'Zn(2+)', '2C2H3N').
# A bare '[' is NOT evidence of a formula (IUPAC names use it: 'bicyclo[5.4.0]'); only
# '[<element symbol>' followed by a digit / bracket / space opens a coordination formula.
_FORMULA_RE = re.compile(
    r"[A-Z][a-z]?\d+[A-Z(]|\d[A-Z][a-z]?\d|\*|\(\d[+-]\)|\d\s?[+-]\)|\{|\[[A-Z][a-z]?\d*[\[\]() ]")
# A component that is plausibly a catalyst (metal, onium cation, halide/hydroxide salt, salen,
# porphyrin, complex): it may NOT be demoted to an auxiliary reagent without a structure.
_CATALYST_LIKE_RE = re.compile(
    r"\b(?:lithium|sodium|potassium|rubidium|cesium|caesium|magnesium|calcium|strontium|barium|"
    r"zinc|cadmium|mercury|copper|silver|gold|nickel|palladium|platinum|cobalt|rhodium|iridium|"
    r"iron|ruthenium|osmium|manganese|chromium|molybdenum|tungsten|vanadium|niobium|tantalum|"
    r"titanium|zirconium|hafnium|scandium|yttrium|lanthan\w*|cerium|ytterbium|gadolinium|"
    r"aluminium|aluminum|gallium|indium|tin|lead|bismuth|antimony|germanium|silicon|boron)\b"
    r"|ammonium|phosphonium|imidazolium|pyrazolium|pyridinium|guanidinium|sulfonium|stibonium|"
    r"arsonium|iminium|salen|porphyr|complex|catalyst|(?:ide|ate)\(|\bcopper\b|"
    r"\b(?:bromide|chloride|iodide|fluoride|hydroxide|acetate|carbonate|oxide|oxo|cobaltite|molybdate)\b|"
    r"\b(?:[A-Z][a-z]?)(?:\([IVX]+\))", re.I)

# Things that may NOT be kept as a plain-text auxiliary reagent: complexes / MOFs / code-named
# materials (metal symbol, brackets, 'XXX-123'), inorganic frameworks, and parse residue.
_AUX_REJECT_WORDS = re.compile(
    r"prussian|lignin|magnesia|schiff|salen|porphyr|\boxide\b|\bblue\b|hydrate\b", re.I)
# case-sensitive: a metal symbol glued to a capital/digit/bracket, e.g. 'ZnF', 'CoTPP', 'Zn(', 'Ni('
_AUX_REJECT_SYMBOL = re.compile(
    r"[()\[\]{}°]|"
    r"(?:Li|Na|Rb|Cs|Mg|Ca|Sr|Ba|Zn|Cd|Hg|Cu|CU|Ag|Au|Ni|Pd|Pt|Co|Rh|Ir|Fe|Ru|Os|Mn|Cr|Mo|Nb|Ta|"
    r"Ti|Zr|Hf|Sc|La|Ce|Yb|Gd|Al|Ga|In|Sn|Pb|Bi|Sb|Ge)(?=[A-Z\d(])|"
    r"\b[A-Za-z]{1,6}-?\d{2,}\b")


def _aux_ok(comp: str) -> bool:
    """True when an unresolved plain-named component may be kept as auxiliary reagent text."""
    if not re.search(r"[A-Za-z]{4,}", comp):          # parse residue such as '4O'
        return False
    return not (_AUX_REJECT_WORDS.search(comp) or _AUX_REJECT_SYMBOL.search(comp))

_HYDRATE = re.compile(r"\s*(?:\*|·|\.)?\s*(?:\d+\s*)?(?:hexa|di|tetra|tri|mono|penta)?hydrate$", re.I)
_LOADING = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*mol\s*%\s*\)?", re.I)


def _split_top(text: str, seps: str = ",;") -> List[str]:
    """Split on ', ' / '; ' at bracket depth 0 (never inside '1,3-' or '(a, b)')."""
    parts, buf, depth = [], [], 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and ch in seps and i + 1 < len(text) and text[i + 1].isspace():
            parts.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _split_solvent_clause(seg: str) -> Tuple[str, Optional[str]]:
    """'A, B in S1, S2' -> ('A, B', 'S1, S2').  The LAST depth-0 ' in ' starts the solvent clause."""
    depth, cut = 0, None
    for m in re.finditer(r"[()\[\]{}]|\s+in\s+", seg):
        t = m.group(0)
        if t in "([{":
            depth += 1
        elif t in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            cut = m
    if cut is None:
        return seg.strip(), None
    return seg[:cut.start()].strip(), seg[cut.end():].strip()


def resolve_catalyst(comp: str) -> Tuple[Optional[str], Optional[float]]:
    """Whole-text lookup of one reagent. Returns (canonical_name or None, loading mol%)."""
    t = comp.strip()
    loading = None
    lm = _LOADING.search(t)
    if lm:
        v = float(lm.group(1))
        loading = round(v, 2) if 0 < v <= 100 else None
        t = (t[:lm.start()] + t[lm.end():]).strip(" ,")
    t = _HYDRATE.sub("", t) if _norm_key(t) not in _CAT_LOOKUP else t
    return _CAT_LOOKUP.get(_norm_key(t)), loading


# A structured Reaxys record reads
#   With <reagent>, <reagent> in <solvent>, Time= 5h, T= 20 °C , p= 750.075Torr , <tags> <authors; journal; (year)>
# Rows that start with free procedure prose are NOT parsed: the reagent list
# appended to such prose is often incomplete (the main catalyst is described
# only in the prose), so a partial list cannot be told apart from a full one.
_FIELD_START = re.compile(r"(?<![A-Za-z0-9])(?:Time|T|p)\s*=")
_FIELD_VAL = {
    "Time": re.compile(r"\s*,?\s*(Time)\s*=\s*(\d+(?:\.\d+)?)\s*(h)(?![A-Za-z])"),
    "T":    re.compile(r"\s*,?\s*(T)\s*=\s*(-?\d+(?:\.\d+)?)\s*(°C)"),
    "p":    re.compile(r"\s*,?\s*(p)\s*=\s*(\d+(?:\.\d+)?)\s*(Torr)(?![A-Za-z])"),
}
_NONTHERMAL = re.compile(
    r"Irradiation|Microwave|electrolysis|Flow reactor|Ultrasound|Photo|\bStage\s*\d|Cooling with ice", re.I)
_TAGS = [
    "Autoclave", "Catalytic behavior", "Reagent/catalyst", "Temperature", "Pressure", "Schlenk technique",
    "Sealed tube", "Green chemistry", "Inert atmosphere", "Solvent", "Supercritical conditions",
    "Ambient temperature", "Molecular sieve", "regioselective reaction", "chemoselective reaction",
    "High pressure", "Glovebox", "Ionic liquid", "Kinetics", "Time", "neat (no solvent)",
    "General procedure:", "Typical procedure:",
]
_TAG_RE = re.compile(r"^\s*,?\s*(?:" + "|".join(re.escape(t) for t in _TAGS) + r")\s*,?\s*", re.I)


def split_record(cond: str) -> Optional[Tuple[str, str]]:
    """('A, B in S', 'Time= ..., rest') for a pure structured row, else None."""
    m = re.match(r"\s*With\s+", cond, re.I)
    if not m:
        return None
    rest = cond[m.end():]
    f = _FIELD_START.search(rest)
    if not f:
        return rest, ""
    return rest[:f.start()].rstrip(" ,"), rest[f.start():]


def parse_fields(tail: str) -> Tuple[Dict[str, float], str, str]:
    """Read the contiguous 'Time= / T= / p=' block.

    Returns (values, remainder_after_block, error).  A field that is present but
    not a single number with its unit (ranges '0 - 160 °C', stages, repeated
    fields) is an error: the record is ambiguous, not merely incomplete.
    """
    vals: Dict[str, float] = {}
    pos = 0
    while True:
        hit = None
        for key, rx in _FIELD_VAL.items():
            m = rx.match(tail, pos)
            if m:
                hit = (key, m); break
        if not hit:
            break
        key, m = hit
        if key in vals:
            return {}, "", f"{key} given twice"
        v = float(m.group(2))
        if key == "p":
            v = round(v * 0.000133322, 5)
        vals[key] = v
        pos = m.end()
    rem = tail[pos:]
    if _FIELD_START.match(rem.lstrip(" ,")):
        return {}, "", "field is a range / not a single value: " + rem.strip()[:40]
    if "T" in vals and not -50 <= vals["T"] <= 400:
        return {}, "", f"T out of range: {vals['T']}"
    if "Time" in vals and not 0 < vals["Time"] <= 720:
        return {}, "", f"time out of range: {vals['Time']}"
    if "p" in vals and not 0 <= vals["p"] <= 50:
        return {}, "", f"p out of range: {vals['p']}"
    return vals, rem, ""


def parse_reference(rem: str) -> Tuple[str, Optional[int]]:
    """(reference text, year) from what follows the Time/T/p block."""
    s = rem.strip()
    while True:
        m = _TAG_RE.match(s)
        if not m:
            break
        s = s[m.end():]
    s = s.strip(" ,")
    ym = re.search(r"\((\d{4})\)", s)
    year = int(ym.group(1)) if ym and 1900 <= int(ym.group(1)) <= 2100 else None
    return (s if len(s) >= 15 else "Unknown reference"), year


def parse_reactant(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    rx = str(row.get("Reaction", "")).strip()
    for k, (r, p) in SUBSTRATE_MAP.items():
        if k.lower() in rx.lower():
            return r, p
    return None, None


def classify(cats: List[str]) -> str:
    """Five-class: ionic_liquid / metal_halide / organic_base / mixed_system / unknown."""
    if not cats:
        return "unknown"
    text = " ".join(str(c).lower() for c in cats if c)

    def hit(keyword: str) -> bool:
        return bool(re.search(r"(?<![a-zA-Z])" + re.escape(keyword.lower()) + r"(?![a-zA-Z])", text))

    ionic = any(hit(k) for k in IONIC_KW) or bool(ONIUM_STEM_RE.search(text))
    metal = any(hit(k) for k in METAL_KW)
    base = any(hit(k) for k in BASE_KW)
    if sum((ionic, metal, base)) >= 2:
        return "mixed_system"
    return "ionic_liquid" if ionic else "metal_halide" if metal else "organic_base" if base else "unknown"


def is_blacklist(text: str) -> bool:
    return bool(text) and any(p.search(text) for p in CAT_BLACKLIST_PATTERNS)


def _disc(rid: int, reason: str) -> Dict:
    return dict(
        row_id=rid,
        catalyst_1_name="", catalyst_2_name="", catalyst_3_name="", catalyst_4_name="",
        catalyst_1_smiles="", catalyst_2_smiles="", catalyst_3_smiles="", catalyst_4_smiles="",
        discard_reason=reason, discard_type="reasonable_discard",
    )


def _resolve_solvents(clause: Optional[str]) -> Tuple[List[str], Optional[str]]:
    """(canonical solvents, offending_text).  None clause -> ([], None): not reported."""
    if clause is None:
        return [], None
    c = clause.strip()
    if re.match(r"neat\b", c, re.I):                       # 'neat (no solvent)', 'neat liquid', ...
        return ["无溶剂"], None
    out: List[str] = []
    for part in _split_top(c):
        hit = _SOLV_LOOKUP.get(_norm_key(part))
        if hit is None:
            return [], part
        if hit not in out:
            out.append(hit)
    return out, None


def process_row(rid: int, row: pd.Series) -> Tuple[Optional[Dict], Optional[Tuple[str, str, str]]]:
    """Return (valid_row, None) or (None, (bucket, reason, raw_text)).

    bucket 'discard' = the record contains something that cannot be represented
    (material, polymer, unknown additive, ambiguous field, other reaction mode);
    bucket 'repair'  = only a short, plain-looking reagent name is unknown, i.e.
    the dictionary could be extended to rescue the row.
    """
    rat, prd = parse_reactant(row)
    if rat is None:
        return None, ("discard", f"底物不在支持范围:{str(row.get('Reaction'))[:60]}", "")
    yv, why = parse_yield(row.get("Yield"))
    if yv is None:
        return None, ("discard", f"产率无法解析:{why}", "")

    con = normalize_text(str(row.get("Conditions", "")).strip())
    parts = split_record(con)
    if parts is not None:
        head, tail = parts
        if not tail:
            return None, ("discard", "缺少 Time/T/p 字段", con[:200])
        if is_blacklist(head):
            return None, ("discard", "催化剂为材料/MOF/聚合物/光电催化", con[:200])
        parse_fn = "structured"
    else:
        # Prose-led row: look for a 'With...' tail at the very end.
        ws = list(re.finditer(r"(?<![A-Za-z])With\s+", con))
        if not ws:
            return None, ("discard", "非结构化记录且无reagent-style尾部", con[:200])
        last_with = ws[-1]
        sub = split_record(con[last_with.start():])
        if sub is None or not sub[1]:
            return None, ("discard", "缺少 Time/T/p 字段", con[:200])
        head, tail = sub
        if is_blacklist(con[last_with.start():][:200]):
            return None, ("discard", "尾部reagents为材料/MOF/聚合物", con[:200])
        # The narrative before the tail may reveal a support / framework / polymer that the
        # trailing reagent list leaves out: then the record is incomplete and is dropped.
        if is_blacklist(con[:last_with.start()]) or _PROSE_MATERIAL_RE.search(con[:last_with.start()]):
            return None, ("discard", "操作文本含材料/MOF/聚合物载体", con[:200])
        parse_fn = "prose_tail"

    reagents, solv_clause = _split_solvent_clause(head)
    vals, rem, err = parse_fields(tail)
    if err:
        return None, ("discard", "条件字段不明确:" + err, con[:200])
    if not vals and not re.search(r"\(\d{4}\)", rem):
        return None, ("discard", "缺少 Time/T/p 字段且无参考年份", con[:200])
    if _NONTHERMAL.search(rem[:200]) or (parse_fn == "structured" and _NONTHERMAL.search(head)):
        return None, ("discard", "非常规加热方式(光照/微波/电解/流动/分段)", con[:200])

    solvents, bad_s = _resolve_solvents(solv_clause)
    if bad_s is not None:
        return None, ("discard", f"溶剂无法确定:{bad_s[:60]}", con[:200])

    names: List[str] = []
    loads: List[Optional[float]] = []
    unresolved: List[str] = []
    for comp in _split_top(reagents):
        name, ld = resolve_catalyst(comp)
        if name is None:
            if _MATERIAL_RE.search(comp):
                return None, ("discard", f"组分为材料/聚合物/负载体系:{comp[:70]}", con[:200])
            if _FORMULA_RE.search(comp):
                # Only an element-count / charge-state string: it names a compound but carries no
                # connectivity, so it cannot be turned into a SMILES (no DRFP, no xTB input).
                return None, ("discard", f"组分仅有分子式/配位式(无连接性):{comp[:70]}", con[:200])
            # a long but plain-looking name is a discrete molecule missing from the
            # registry (-> repair list), not a material; only absurd lengths are noise
            if len(comp) > 120:
                return None, ("discard", f"组分无法确定结构:{comp[:70]}", con[:200])
            if _CATALYST_LIKE_RE.search(comp) or not _aux_ok(comp) or len(comp) > 60:
                # looks like a catalyst (metal / onium / halide salt): must have a structure
                return None, ("repair", f"组分无法确定结构:{comp[:70]}", con[:200])
            sv = _SOLV_LOOKUP.get(_norm_key(comp))
            if sv is not None:                         # a solvent listed among the reagents
                if sv not in solvents and len(solvents) < 4:
                    solvents.append(sv)
                continue
            unresolved.append(comp.strip())
            continue
        if name not in names:
            names.append(name); loads.append(ld)
    if not names:
        bad = unresolved[0] if unresolved else ""
        return None, ("repair" if bad else "discard",
                      f"组分无法确定结构:{bad[:70]}" if bad else "无催化剂", con[:200])
    # Named species without a registry structure are auxiliary reagents only when
    # the row already has a structure-resolved primary catalyst; more than two
    # cannot be kept in the reagent columns, so the row is routed to repair.
    if len(unresolved) > 2:
        return None, ("repair", f"辅助试剂>2:{'; '.join(unresolved)[:70]}", con[:200])
    if len(names) > 4:
        return None, ("discard", f"催化剂组分>4({len(names)})", con[:200])

    ref, year = parse_reference(rem)
    pad = lambda xs, i: xs[i] if i < len(xs) else None  # noqa: E731
    r = {
        "row_id": rid, "reactant_name": rat, "product_name": prd, "yield (%)": yv,
        "temperature (\u2103)": _v(vals.get("T")),
        "pressure (MPa)": _v(vals.get("p")),
        "time (h)": _v(vals.get("Time")),
    }
    for i in range(4):
        r[f"catalyst_{i+1}_name"] = _v(pad(names, i)) or ""
        r[f"catalyst_{i+1}_loading_mol%"] = _v(pad(loads, i)) or ""
    r.update({
        "reagent_1_name": unresolved[0] if len(unresolved) > 0 else "",
        "reagent_2_name": unresolved[1] if len(unresolved) > 1 else "",
        "solvent_1": solvents[0] if len(solvents) > 0 else "",
        "solvent_2": solvents[1] if len(solvents) > 1 else "",
        "solvent_3": solvents[2] if len(solvents) > 2 else "",
        "solvent_4": solvents[3] if len(solvents) > 3 else "",
        "reference": ref, "publication_year": year if year else "",
        "extraction_status": "valid", "catalyst_system_type": classify(names),
        "all_solvents_normalized": "; ".join(solvents),
    })
    return r, None


def clean_dataset(
    df_raw: pd.DataFrame,
    progress_every: int = 500,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Return (valid_rows, extraction_failures, discards, stats)."""
    if not _CAT_LOOKUP:
        _init_lookups()
    valid_r, ext_r, disc_r = [], [], []
    st = dict(total=0, valid=0, disc=0, ext=0, sub=0, yl=0, bl=0, cond=0, dup=0)
    reasons: Counter = Counter()
    seen: Dict[Tuple, int] = {}
    seen_parsed: Dict[Tuple, int] = {}
    for idx, row in df_raw.iterrows():
        st["total"] += 1
        rid = idx + 1
        key = (str(row.get("Reaction")).strip(), str(row.get("Yield")).strip(),
               normalize_text(str(row.get("Conditions")).strip()))
        if key in seen:
            disc_r.append(_disc(rid, f"与第{seen[key]}行完全重复(Reaction/Yield/Conditions 相同)"))
            st["dup"] += 1; st["disc"] += 1
            continue
        seen[key] = rid
        ok, bad = process_row(rid, row)
        if ok is not None:
            # Second-level dedupe on the PARSED record: two raw rows whose text
            # differs but whose every output column (except row_id) is identical
            # are the same experiment (e.g. general-procedure line + summary line).
            pkey = tuple(str(v) for k, v in ok.items() if k != "row_id")
            if pkey in seen_parsed:
                disc_r.append(_disc(rid, f"与第{seen_parsed[pkey]}行解析后全部字段完全一致(重复实验)"))
                st["dup"] += 1; st["disc"] += 1
                continue
            seen_parsed[pkey] = rid
            valid_r.append(ok); st["valid"] += 1
            continue
        bucket, reason, raw_txt = bad
        reasons[reason.split(":")[0]] += 1
        if bucket == "repair":
            ext_r.append(dict(source_row_id=rid, discard_reason=reason,
                              discard_type="extraction_failed_repair_needed", affected_field="catalyst",
                              raw_value=raw_txt, suggested_fix="确认结构后加入 CAT_NORMALIZE 与 chem_registry"))
            st["ext"] += 1
        else:
            disc_r.append(_disc(rid, reason)); st["disc"] += 1
        if reason.startswith("底物"):
            st["sub"] += 1
        elif reason.startswith("产率"):
            st["yl"] += 1
        elif "材料" in reason:
            st["bl"] += 1
        elif "字段" in reason or "Time/T/p" in reason:
            st["cond"] += 1
        if progress_every and rid % progress_every == 0:
            logger.info("  %d/%d valid=%d", rid, len(df_raw), st["valid"])
    st["reasons"] = dict(reasons)
    return valid_r, ext_r, disc_r, st


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    for enc in INPUT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("  encoding %s OK: %d rows", enc, len(df))
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "all-encodings", b"", 0, 1, f"Cannot decode {path} with any of {INPUT_ENCODINGS}"
    )


def save_outputs(
    valid_r: List[Dict], ext_r: List[Dict], disc_r: List[Dict],
    cleaned_csv: Path, ext_csv: Path, disc_csv: Path, baseline_json: Path,
) -> None:
    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid_r, columns=OUT_COLS).to_csv(cleaned_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved %d rows → %s", len(valid_r), cleaned_csv)

    # Always (re)write both reports, header-only when empty, so a report from an
    # earlier run can never survive next to a fresh cleaned.csv.
    ext_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ext_r, columns=EXTR_COLS).to_csv(ext_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved %d extraction-failure rows → %s", len(ext_r), ext_csv)
    pd.DataFrame(disc_r, columns=DISC_COLS).to_csv(disc_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved %d discard rows → %s", len(disc_r), disc_csv)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1.1 — clean the raw CO2-cycloaddition CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite cleaned.csv even if it exists.")
    p.add_argument("--input", type=str, default=str(RAW_REAXYS_CSV),
                   help="Path to the raw Reaxys-style CSV (overrides default).")
    p.add_argument("--output", type=str, default=str(CLEANED_CSV),
                   help="Path to the cleaned CSV (overrides default).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FMT, datefmt=LOG_DATEFMT,
    )


def main(argv: Optional[List[str]] = None) -> int:
    t_start = time.time()
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    output_csv = Path(args.output)
    extraction_csv = DATA_EXTERNAL / "extraction_report.csv"
    discard_csv = DATA_EXTERNAL / "discard_report.csv"
    baseline_json = CLEANED_BASELINE

    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("CO2 环加成原始数据清洗")
    logger.info("=" * 60)

    if not args.force and output_csv.exists():
        try:
            existing = pd.read_csv(output_csv, encoding="utf-8-sig")
            logger.info("[SKIP] %s already exists (%d rows). Use --force to re-run.",
                        output_csv, len(existing))
            return 0
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not read existing %s: %s — re-cleaning.", output_csv, ex)

    logger.info("[1/5] Loading: %s", args.input)
    df_raw = load_raw_csv(Path(args.input))

    logger.info("[2/5] Cleaning %d rows …", len(df_raw))
    valid_r, ext_r, disc_r, st = clean_dataset(
        df_raw, progress_every=300,
    )

    save_outputs(valid_r, ext_r, disc_r, output_csv, extraction_csv, discard_csv, baseline_json)

    logger.info("[3/5] Done in %.1fs.", time.time() - t_start)
    logger.info("=" * 60)
    logger.info("  raw=%d valid=%d discard=%d extraction_failed=%d",
                st["total"], st["valid"], st["disc"], st["ext"])
    logger.info("    substrate=%d yield=%d blacklist=%d conditions=%d",
                st["sub"], st["yl"], st["bl"], st["cond"])
    logger.info("=" * 60)

    summary = dict(
        valid_rows=st["valid"],
        total_rows=st["total"],
        discard_rows=st["disc"],
        ext_rows=st["ext"],
        timestamp=str(datetime.now()),
    )
    ensure_dir(baseline_json.parent)
    with open(baseline_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", baseline_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())