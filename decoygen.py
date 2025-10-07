#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoy generation (RULE‑FIRST permutations; performance‑aware; mutation‑aware) with:

Core semantics (as requested)
-----------------------------
• **Skip logic**: a stage is *skipped* iff its criterion is already met:
  – N‑term good → don’t edit N‑term.
  – C‑term hydrophobic → don’t edit C‑term (Ala C‑term remains frozen).
  – If an internal P/C/M already exists → skip that stage.
  – If an internal R/K already exists **or an R exists anywhere in the sequence (incl. termini)** → skip RK stage.
• **First internal edit at mutant site (non‑frameshift)**:
  – The first internal edit that isn’t skipped must be applied at `mut_idx0`.
  – Never set the mutant site to the original WT or MUT residue.
  – Special case: if mut is P→C, C‑stage always skips; P‑stage skips *that position* but may edit elsewhere later.
• **Proline placement & context**:
  – P is never placed at ends; allowed 1‑based positions ∈ [4 .. n−2].
  – Never edit the residue immediately before a Proline.
• **C/M around P**: prefer (small bonus) having C and M on **opposite sides** of an internal P when possible—
  both orientations considered (M left / C right **and** M right / C left).
• **Tie reducer**: +0.5 raw rule bonus if ≥4 internal hydrophobic residues (I/L/V/F/Y/W/M) beyond termini.

Charge heuristics and rules
---------------------------
• Deterministic, sequence‑only ESI charge predictor (precursor z ∈ {1..4}).
• Predicted b/y fragment charge split per cleavage; optionally:
  – skip b1/b2 and y1/y2,
  – avoid cleavage C‑terminal to Proline (“after Proline”).
• **Rule A (1 point)**: add +1.0 to the rule total **iff**:
  – predicted precursor charge ∈ [2..4], **and**
  – at least one predicted fragment (excluding b1,b2,y1,y2 and after‑Proline cleavages) has charge ≥ 2.
• **Rule B (1 point, new)**: add +1.0 **iff**:
  – predicted precursor charge ∈ {2,3,4}, **and**
  – there are **≥3 predicted fragments** (excluding b1/b2/y1/y2 and honoring “avoid after‑Proline”) with fragment charge **2 or 3**.

Ranking & selection
-------------------
• Final winner per peptide among rule‑best survivors uses continuous features:
  – Confusability (0–1) vs contaminant windows (m/z+RT+frags),
  – Hydrophobic **fraction** (0–1).
  Tie‑break: higher hydrophobic fraction → longer HK substring → fewer D/E → lexicographic.
• Global composite (dataset ranking) = **normalized** combination of:
  – hydrophobicity (KD→[0,1]), confusability (0–1), HK k‑mer fraction (0–1), and optional polymer proximity (0–1).
  No double‑counting (the legacy “fly” is not used in the composite).
• Dataset Top‑N selection: standard thresholding vs originals (unchanged), with constraints.

Performance
-----------
• Beam search:
  – `--term-beam` (default 3) keeps top‑K terminal variants per stage by rule score.
  – `--beam-internal` (default 128) prunes internal stage frontiers per permutation by rule score (and a stable key).
• Charge predictor reduces charge enumeration; fragment pruning avoids tiny ions and post‑Proline cleavages.
• C/M split preference encoded as a rule bonus; does not force ordering.
• Fragment m/z generation uses caching + pruning; windows remain pre‑indexed with sorted fragment m/z and bisection.

Outputs
-------
• CSVs: all originals (scored), all decoys, selected top‑N.
• Plots: colored **edit paths (only steps where edits occurred, not “keeps”)** and scatter/histograms.
• README summarizing rules and units.

Dependencies
------------
numpy, pandas, matplotlib (Agg), tqdm, requests (only if downloads enabled)
External helpers (unchanged API):
  - wiqd_peptide_similarity.enumerate_polymer_series
  - wiqd_constants.AA, H2O, PROTON_MASS as PROTON
  - wiqd_features.kd_gravy, hydrophobic_fraction, mass_neutral, mz_from_mass, mz_from_sequence
  - wiqd_rt_proxy.predict_rt_min
  - wiqd_proteins.load_housekeeping_sequences
  - wiqd_sequence_helpers.collapse_seq_mode
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# ------------------------ Logging ------------------------
LOG = logging.getLogger("decoygen")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------ External helpers ----------------
from wiqd_peptide_similarity import enumerate_polymer_series
from wiqd_constants import AA, H2O, PROTON_MASS as PROTON, AA_HYDRO_CTERM
from wiqd_features import (
    kd_gravy,
    hydrophobic_fraction,
    mz_from_mass,
    mass_neutral,
    mz_from_sequence,
)
from wiqd_rt_proxy import predict_rt_min
from wiqd_proteins import load_housekeeping_sequences
from wiqd_sequence_helpers import collapse_seq_mode

# =============================================================================
# Utilities
# =============================================================================


def ensure_outdir(p: str | os.PathLike):
    os.makedirs(p, exist_ok=True)


def parse_csv_list(s: Optional[str], cast=str) -> Tuple:
    if not s:
        return tuple()
    return tuple(cast(x.strip()) for x in s.split(",") if x.strip())


def clean_seq(s: str) -> str:
    return "".join(ch for ch in str(s).strip().upper() if ch in AA)


def kmer_set(s: str, k: int) -> Set[str]:
    if k <= 0 or len(s) < k:
        return set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


class RobustMinMax:
    def __init__(self, ref: pd.Series, lo_pct: float = 1.0, hi_pct: float = 99.0):
        arr = ref.to_numpy(float)
        self.lo = float(np.nanpercentile(arr, lo_pct)) if len(arr) else 0.0
        self.hi = float(np.nanpercentile(arr, hi_pct)) if len(arr) else 1.0
        if not np.isfinite(self.lo):
            self.lo = np.nanmin(arr) if len(arr) else 0.0
        if not np.isfinite(self.hi):
            self.hi = np.nanmax(arr) if len(arr) else 1.0
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-9

    def transform_scalar(self, v: float) -> float:
        v = min(max(v, self.lo), self.hi)
        return (v - self.lo) / (self.hi - self.lo)

    def transform(self, s: pd.Series) -> pd.Series:
        arr = s.to_numpy(float)
        arr = np.clip(arr, self.lo, self.hi)
        return pd.Series((arr - self.lo) / (self.hi - self.lo), index=s.index)


# =============================================================================
# Contaminant sets (download or FASTA)
# =============================================================================

DOWNLOAD_RECIPES = {
    "albumin": {"type": "uniprot_accessions", "accessions": ["P02768"]},
    "keratins": {
        "type": "uniprot_accessions",
        "accessions": [
            "P04264",
            "P05787",
            "P05783",
            "P08727",
            "P13645",
            "P13647",
            "P02533",
        ],
    },
    "proteases": {
        "type": "uniprot_accessions",
        "accessions": ["P00760", "P07477", "P00761"],
    },
    "mhc_hardware": {
        "type": "uniprot_accessions",
        "accessions": ["P61769", "P01892", "P01899", "P01889", "P13747"],
    },
}


def _parse_fasta_to_acc_seq(text: str) -> Dict[str, str]:
    acc_to_seq: Dict[str, str] = {}
    acc, chunks = None, []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if acc is not None and chunks:
                acc_to_seq[acc] = "".join(chunks)
            header = line[1:].strip()
            m = re.match(r"^\w+\|([^|]+)\|", header)
            acc = m.group(1) if m else header.split()[0]
            chunks = []
        else:
            chunks.append(re.sub(r"[^A-Z]", "", line.strip().upper()))
    if acc is not None and chunks:
        acc_to_seq[acc] = "".join(chunks)
    for k in list(acc_to_seq.keys()):
        acc_to_seq[k] = "".join(ch for ch in acc_to_seq[k] if ch in AA)
    return acc_to_seq


def download_uniprot_accessions(
    accessions: Sequence[str], timeout: int = 30
) -> Dict[str, str]:
    import requests

    if not accessions:
        return {}
    url = "https://rest.uniprot.org/uniprotkb/stream"
    q = " OR ".join([f"accession:{a}" for a in accessions])
    params = {"format": "fasta", "query": q}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return _parse_fasta_to_acc_seq(r.text)


def load_sets(
    set_names: Sequence[str],
    sets_fasta_dir: Optional[str],
    do_download: bool,
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for s in set_names:
        acc2seq: Dict[str, str] = {}
        if sets_fasta_dir:
            f = os.path.join(sets_fasta_dir, f"{s}.fasta")
            if os.path.isfile(f):
                LOG.info(f"[sets] Reading FASTA: {f}")
                with open(f, "r") as fh:
                    acc2seq = _parse_fasta_to_acc_seq(fh.read())
        if not acc2seq and do_download:
            recipe = DOWNLOAD_RECIPES.get(s)
            if not recipe:
                raise ValueError(f"Unknown set '{s}'. Known: {list(DOWNLOAD_RECIPES)}")
            if recipe.get("type") == "uniprot_accessions":
                acc2seq = download_uniprot_accessions(
                    recipe.get("accessions", []), timeout=timeout
                )
        if not acc2seq:
            raise RuntimeError(
                f"No sequences for set '{s}'. Provide --sets-fasta-dir/{s}.fasta or enable --download-sets 1."
            )
        out[s] = acc2seq
        LOG.info(f"[sets] {s}: {len(acc2seq)} sequences")
    return out


# =============================================================================
# Windows & confusability
# =============================================================================


@dataclass
class Window:
    seq: str
    acc: str
    set_name: str
    length: int
    mass: float
    mz_by_z: Dict[int, float]
    rt_min: float
    frag_mz_sorted: List[float] = field(default_factory=list)


def build_windows_index(
    set_name: str,
    acc2seq: Dict[str, str],
    len_min: int,
    len_max: int,
    charges_precursor: Iterable[int],
    gradient_min: float,
    cys_fixed_mod: float,
) -> List[Window]:
    zs = sorted({int(z) for z in charges_precursor})
    out: List[Window] = []
    for acc, s in acc2seq.items():
        seq = "".join(ch for ch in s if ch in AA)
        N = len(seq)
        for L in range(max(1, len_min), min(N, len_max) + 1):
            for i in range(0, N - L + 1):
                sub = seq[i : i + L]
                m = mass_neutral(sub, cys_fixed_mod)
                mz_by = {z: mz_from_mass(m, z) for z in zs}
                rt = predict_rt_min(sub, gradient_min)
                out.append(Window(sub, acc, set_name, L, m, mz_by, rt))
    return out


def index_fragments_for_windows(
    windows: List[Window],
    fragment_kmin: int,
    fragment_kmax: int,
    frag_charges: Iterable[int],
    cys_fixed_mod: float,
):
    frag_zs = sorted({int(z) for z in frag_charges})
    for w in windows:
        arr: List[float] = []
        if w.length <= 1:
            w.frag_mz_sorted = []
            continue
        kmax_eff = min(fragment_kmax, w.length - 1)
        kmin_eff = max(1, min(fragment_kmin, kmax_eff))
        for k in range(kmin_eff, kmax_eff + 1):
            bseq = w.seq[:k]
            yseq = w.seq[-k:]
            for z in frag_zs:
                bmz = (mass_neutral(bseq, cys_fixed_mod) - H2O + z * PROTON) / z
                ymz = (mass_neutral(yseq, cys_fixed_mod) + z * PROTON) / z
                arr.append(bmz)
                arr.append(ymz)
        arr.sort()
        w.frag_mz_sorted = arr


# ---------- Charge heuristic (precursor + fragment charge splits) -------------


@lru_cache(maxsize=100000)
def _basicity_proxy(s: str) -> float:
    w = {"R": 1.0, "K": 0.8, "H": 0.3}
    return 0.35 + sum(w.get(a, 0.0) for a in s)


@lru_cache(maxsize=100000)
def predict_precursor_charge(seq: str, max_z: int = 4) -> int:
    """Sequence‑only positive‑mode precursor charge prediction (1..max_z)."""
    s = _basicity_proxy(seq)
    L = len(seq)
    s += 0.2 * (L >= 12) + 0.3 * (L >= 16)
    s -= 0.2 * ((seq.count("D") + seq.count("E")) >= 3)
    if seq.endswith(("K", "R")):
        s += 0.2
    z = 1
    for t in (1.2, 2.2, 3.2):
        if s >= t:
            z += 1
    return min(z, max_z)


@lru_cache(maxsize=100000)
def _frag_capacity(subseq: str, ion: str) -> float:
    """Relative ability of a fragment to hold charge."""
    w = {"R": 1.0, "K": 0.8, "H": 0.3}
    cap = sum(w.get(a, 0.0) for a in subseq)
    if ion == "b":
        cap += 0.5  # new N-terminus
    else:
        cap += 0.35  # new C-terminus
        if subseq.endswith(("K", "R")):
            cap += 0.4  # tryptic y-ion bias
    return max(cap, 0.0)


@lru_cache(maxsize=200000)
def predict_b_y_charges(
    seq: str, precursor_z: int
) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    For each cleavage (i|i+1), return tuples: (i, zb, zy, L)
    where i is the cleavage index (1..L-1), zb = charge on b, zy on y.
    A simple rule: if z==1, assign all to higher-capacity side; if z>=2, enforce at least 1+ per side,
    remaining goes to higher-capacity side.
    """
    L = len(seq)
    out: List[Tuple[int, int, int, int]] = []
    for i in range(1, L):
        b, y = seq[:i], seq[i:]
        cb, cy = _frag_capacity(b, "b"), _frag_capacity(y, "y")
        z = int(precursor_z)
        if z == 1:
            zb, zy = (1, 0) if cb >= cy else (0, 1)
        else:
            zb = zy = 1
            rem = z - 2
            if rem > 0:
                if cb >= cy:
                    zb += rem
                else:
                    zy += rem
        out.append((i, zb, zy, L))
    return tuple(out)


# -------------------- Confusability (with heuristics & pruning) ---------------


def confusability_in_mz(
    pep: str,
    windows: List[Window],
    full_len_min: int,
    full_len_max: int,
    precursor_charges: Iterable[int],
    ppm_tol: float,
    rt_pred_min: float,
    rt_tol_min: float,
    cys_fixed_mod: float,
    kmin: int,
    kmax: int,
    good_ppm: float,
    frag_types_req: int,
    *,
    # ---- all options are explicit arguments (no function attributes) ----
    use_charge_pred: bool,
    max_pred_z: int,
    skip_short_frags: bool,
    avoid_post_proline: bool,
    require_frag_z2: bool,
    frag_charges: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    """
    Score confusability of 'pep' vs pre-indexed contaminant windows.

    Parameters
    ----------
    use_charge_pred : bool
        If True, use predicted precursor charge (single z) and predicted b/y splits.
        If False, enumerate `precursor_charges` for precursor m/z and `frag_charges` for fragments.
    max_pred_z : int
        Upper bound for predicted precursor charge (typically 4).
    skip_short_frags : bool
        If True, skip b1/b2 and y1/y2.
    avoid_post_proline : bool
        If True, avoid cleavages C-terminal to Proline for fragment matching.
    require_frag_z2 : bool
        If True, require that at least one predicted fragment has charge ≥2; otherwise zero out fragment match terms.
    """
    # Precursor m/z candidates
    z_pred = (
        predict_precursor_charge(pep, max_z=int(max_pred_z))
        if use_charge_pred
        else None
    )
    prec_zs = (
        [z_pred]
        if (use_charge_pred and z_pred is not None)
        else sorted({int(z) for z in precursor_charges})
    )

    assert pep.isupper(), pep
    assert all(ch in AA for ch in pep), pep
    pep_m = mass_neutral(pep, cys_fixed_mod)
    pep_mz_by_z = {z: mz_from_mass(pep_m, z) for z in prec_zs}
    pool = [
        i for i, w in enumerate(windows) if (full_len_min <= w.length <= full_len_max)
    ]

    def _precursor_matched_idxs(win_subset: List[Window]) -> List[int]:
        out = []
        for i, w in enumerate(win_subset):
            hit = False
            for z, pmz in pep_mz_by_z.items():
                wmz = w.mz_by_z.get(z)
                if wmz is None:
                    continue
                if abs(pmz - wmz) / pmz * 1e6 <= ppm_tol:
                    hit = True
                    break
            if hit:
                out.append(i)
        return out

    idx_prec_sub = _precursor_matched_idxs([windows[i] for i in pool])
    prec_idxs = [pool[i] for i in idx_prec_sub]

    # RT co-elution
    def _filter_rt_idxs(idxs: List[int]) -> List[int]:
        lo, hi = rt_pred_min - rt_tol_min, rt_pred_min + rt_tol_min
        return [
            i for i in idxs if (windows[i].rt_min >= lo and windows[i].rt_min <= hi)
        ]

    idx_prec_rt = _filter_rt_idxs(prec_idxs)

    def _best_ppm(idxs):
        best = float("inf")
        for i in idxs:
            w = windows[i]
            for z, pmz in pep_mz_by_z.items():
                wmz = w.mz_by_z.get(z)
                if wmz is None:
                    continue
                ppm = abs(pmz - wmz) / pmz * 1e6
                if ppm < best:
                    best = ppm
        return best

    best_all = _best_ppm(pool) if pool else float("inf")
    best_rt = _best_ppm(idx_prec_rt) if idx_prec_rt else float("inf")

    # Build peptide fragment m/z list
    def _pep_by_frag_mzs() -> List[Tuple[float, str, int, int]]:
        L = len(pep)
        skip_short, avoid_post_pro = bool(skip_short_frags), bool(avoid_post_proline)
        out = []
        if L <= 1:
            return out

        if use_charge_pred and z_pred is not None:
            # use predicted split charges per cleavage
            splits = predict_b_y_charges(pep, z_pred)
            for i, zb, zy, _L in splits:
                if skip_short and (i <= 2 or (L - i) <= 2):  # skip b1/b2 and y1/y2
                    continue
                if avoid_post_pro and pep[i - 1] == "P":  # cleavage after P
                    continue
                bseq, yseq = pep[:i], pep[i:]
                if zb >= 1:
                    bmz = (mass_neutral(bseq, cys_fixed_mod) - H2O + zb * PROTON) / zb
                    out.append((bmz, "b", i, zb))
                if zy >= 1:
                    ymz = (mass_neutral(yseq, cys_fixed_mod) + zy * PROTON) / zy
                    out.append((ymz, "y", L - i, zy))
        else:
            # classic enumeration over (ion,k,z) with CORRECT "after-Proline" checks for b_k and y_k
            frag_zs = list(frag_charges) if frag_charges is not None else prec_zs
            kmin_eff = max(1, min(kmin, min(kmax, L - 1)))
            kmax_eff = min(kmax, L - 1)
            for k in range(kmin_eff, kmax_eff + 1):
                i_b = k  # cleavage index for b_k
                i_y = L - k  # cleavage index for y_k
                if skip_short and (k <= 2 or (L - k) <= 2):  # b1/b2 or y1/y2
                    continue
                skip_b = avoid_post_pro and i_b > 0 and pep[i_b - 1] == "P"
                skip_y = avoid_post_pro and i_y > 0 and pep[i_y - 1] == "P"

                bseq, yseq = pep[:k], pep[-k:]
                for z in frag_zs:
                    if not skip_b:
                        bmz = (mass_neutral(bseq, cys_fixed_mod) - H2O + z * PROTON) / z
                        out.append((bmz, "b", k, z))
                    if not skip_y:
                        ymz = (mass_neutral(yseq, cys_fixed_mod) + z * PROTON) / z
                        out.append((ymz, "y", k, z))
        return out

    fr = _pep_by_frag_mzs()
    n_types = max(1, len(fr))

    def _count_types(idxs: List[int]) -> int:
        seen = set()
        for mz, ion, k, z in fr:
            lo = mz * (1 - ppm_tol * 1e-6)
            hi = mz * (1 + ppm_tol * 1e-6)
            hit = False
            for i in idxs:
                arr = windows[i].frag_mz_sorted
                if not arr:
                    continue
                import bisect

                L = bisect.bisect_left(arr, lo)
                R = bisect.bisect_right(arr, hi)
                if R > L:
                    hit = True
                    break
            if hit:
                seen.add((ion, k, z))
        return len(seen)

    ft_all = _count_types(prec_idxs)
    ft_rt = _count_types(idx_prec_rt)
    frac_all, frac_rt = ft_all / n_types, ft_rt / n_types

    if require_frag_z2:
        has_z2 = any(z >= 2 for (_, _, _, z) in fr)
        if not has_z2:
            ft_all = 0
            ft_rt = 0
            frac_all = 0.0
            frac_rt = 0.0

    # Final confusability score
    if best_all <= good_ppm and ft_rt >= frag_types_req:
        score = 1.0
    else:
        ppm_term = max(0.0, 1.0 - min(best_all, ppm_tol) / ppm_tol)
        frag_term = min(1.0, 0.7 * frac_rt + 0.3 * frac_all)
        score = max(0.0, min(1.0, 0.5 * ppm_term + 0.5 * frag_term))

    return {
        "score": score,
        "best_ppm": best_all,
        "best_ppm_rt": best_rt,
        "n_prec": float(len(prec_idxs)),
        "n_prec_rt": float(len(idx_prec_rt)),
        "frac_frag_rt": frac_rt,
        "frac_frag_all": frac_all,
    }


# =============================================================================
# PEG
# =============================================================================


def polymer_best_for_peptide(
    seq: str,
    candidates: List[Dict[str, object]],
    peptide_z: Iterable[int] = (2, 3),
    ppm_window: float = 10.0,
    cys_fixed_mod: float = 0.0,
) -> Dict[str, object]:
    if not candidates:
        return {
            "polymer_match_flag": 0,
            "polymer_best_abs_ppm": float("inf"),
            "polymer_details": "",
            "polymer_conf": 0.0,
        }
    best = (float("inf"), None, None, None)
    byz: Dict[int, List[Dict[str, object]]] = {}
    for c in candidates:
        z = int(c["z"])
        byz.setdefault(z, []).append(c)
    for z in sorted({int(z) for z in peptide_z}):
        arr = byz.get(z, [])
        if not arr:
            continue
        pep_mz = mz_from_sequence(seq, z, cys_fixed_mod)
        mzs = np.array([float(c["mz"]) for c in arr], dtype=float)
        j = int(np.argmin(np.abs(mzs - pep_mz)))
        d_ppm = (mzs[j] - pep_mz) / pep_mz * 1e6
        if abs(d_ppm) < best[0]:
            best = (abs(d_ppm), arr[j], z, pep_mz)
    absppm, cand, pepz, pep_mz = best
    if cand is None:
        return {
            "polymer_match_flag": 0,
            "polymer_best_abs_ppm": float("inf"),
            "polymer_details": "",
            "polymer_conf": 0.0,
        }
    polymer_conf = max(0.0, 1.0 - min(absppm, ppm_window) / ppm_window)
    detail = (
        f"{cand['family']};end={cand['endgroup']};n={cand['n']};"
        f"adducts={cand['adducts']};polymer_z={cand['z']};pep_z={pepz};"
        f"polymer_mz={float(cand['mz']):.6f};pep_mz={pep_mz:.6f}"
    )
    return {
        "polymer_match_flag": 1 if absppm <= ppm_window else 0,
        "polymer_best_abs_ppm": float(absppm),
        "polymer_conf": polymer_conf,
        "polymer_details": detail,
    }


# =============================================================================
# Housekeeping homology (HK) + longest substring index
# =============================================================================


def build_hk_kmer_bank(
    seqs: Dict[str, str], k: int = 4, collapse: str = "xle"
) -> Set[str]:
    bank = set()
    for _, seq in seqs.items():
        s = collapse_seq_mode(seq, collapse)
        for i in range(0, len(s) - k + 1):
            bank.add(s[i : i + k])
    return bank


def hk_kmer_features(
    peptide: str, bank: Set[str], k: int, collapse: str = "xle"
) -> Dict[str, float]:
    if len(peptide) < k or not bank:
        return {"hk_kmer_hits": 0.0, "hk_kmer_frac": 0.0}
    s = collapse_seq_mode(peptide, collapse)
    kms = [s[i : i + k] for i in range(0, len(s) - k + 1)]
    hits = sum(1 for km in kms if km in bank)
    return {"hk_kmer_hits": float(hits), "hk_kmer_frac": float(hits) / max(1, len(kms))}


def build_hk_substring_banks(
    seqs: Dict[str, str], collapse: str = "xle", Lmax: int = 20
) -> Dict[int, Set[str]]:
    banks: Dict[int, Set[str]] = {L: set() for L in range(1, Lmax + 1)}
    for _, seq in seqs.items():
        s = collapse_seq_mode(seq, collapse)
        n = len(s)
        for L in range(1, min(Lmax, n) + 1):
            for i in range(0, n - L + 1):
                banks[L].add(s[i : i + L])
    return banks


def hk_longest_substring_len(
    peptide: str, banks: Dict[int, Set[str]], collapse: str = "xle"
) -> int:
    if not banks:
        return 0
    s = collapse_seq_mode(peptide, collapse)
    Lmax = min(max(banks.keys()), len(s))
    for L in range(Lmax, 0, -1):
        if L not in banks:
            continue
        for i in range(0, len(s) - L + 1):
            if s[i : i + L] in banks[L]:
                return L
    return 0


# =============================================================================
# Rule features (RAW, no normalization) & editing
# =============================================================================

PROTECTED_SET = {"C", "M", "P"}  # existing residues never overwritten
PROLINE_PRECEDER_WEAKGOOD = {"I", "L", "V", "F", "Y", "W", "M"}


@dataclass
class RuleParams:
    nterm_good: Tuple[str, ...]
    nterm_bad_strong: Tuple[str, ...]
    nterm_bad_weak: Tuple[str, ...]
    nterm_targets: Tuple[str, ...]
    bad_internal: Tuple[str, ...]

    rule_flank_gap_min: int  # kept for compatibility

    # Added knobs (wired from CLI):
    hydro_internal_min: int = 4
    hydro_internal_bonus: float = 0.5
    cm_split_bonus: float = 0.25
    min_precursor_z: int = 2
    max_precursor_z: int = 4
    skip_short_frags_rule: bool = True
    avoid_post_pro_rule: bool = True


def _proline_context_raw(seq: str, edge_near: int = 1) -> Tuple[float, str]:
    """
    Best P context (raw, no scaling), for any internal P:
        +2.0 : GP internal, not near edges
        +1.0 : GP near edge OR any AP
        +0.5 : weak-good (I/L/V/F/Y/W/M before P)
         0.0 : neutral (!K/R before P)
        -1.0 : KR before P
    """
    n = len(seq)
    best_val, best_cls = 0.0, "none"
    for i, a in enumerate(seq):
        if a != "P" or i in (0, n - 1):
            continue
        prev = seq[i - 1]
        near_edge = (i <= edge_near) or (i >= n - 1 - edge_near)
        if prev == "G":
            val = 2.0 if not near_edge else 1.0
            cls = "GP_well" if not near_edge else "GP_edge"
        elif prev == "A":
            val, cls = 1.0, "AP"
        elif prev in PROLINE_PRECEDER_WEAKGOOD:
            val, cls = 0.5, "weak_good"
        elif prev in ("K", "R"):
            val, cls = -1.0, "KR_before_P"
        else:
            val, cls = 0.0, "neutral"
        if (val > best_val) or (val == best_val and best_cls == "none"):
            best_val, best_cls = val, cls
    return best_val, best_cls


def _cm_opposite_sides_bonus(seq: str, bonus: float) -> float:
    """Small preference if there is a P and C/M lie on opposite sides (either orientation)."""
    n = len(seq)
    c_pos = [i for i in range(1, n - 1) if seq[i] == "C"]
    m_pos = [i for i in range(1, n - 1) if seq[i] == "M"]
    p_pos = [i for i in range(1, n - 1) if seq[i] == "P"]
    if not c_pos or not m_pos or not p_pos:
        return 0.0
    for p in p_pos:
        has_left_c = any(i < p for i in c_pos)
        has_right_c = any(i > p for i in c_pos)
        has_left_m = any(i < p for i in m_pos)
        has_right_m = any(i > p for i in m_pos)
        if (has_left_c and has_right_m) or (has_left_m and has_right_c):
            return float(bonus)
    return 0.0


def _charge_rule_ok(seq: str, rp: RuleParams) -> bool:
    """1‑point rule if predicted precursor z in [2..4] AND any predicted fragment has z>=2,
    excluding b1/b2,y1/y2 and after‑Proline cleavages."""
    zpred = predict_precursor_charge(seq, max_z=rp.max_precursor_z)
    if not (rp.min_precursor_z <= zpred <= rp.max_precursor_z):
        return False
    splits = predict_b_y_charges(seq, zpred)
    L = len(seq)
    for i, zb, zy, _L in splits:
        if rp.skip_short_frags_rule and (i <= 2 or (L - i) <= 2):
            continue
        if rp.avoid_post_pro_rule and seq[i - 1] == "P":
            continue
        if zb >= 2 or zy >= 2:
            return True
    return False


def _charge_rule_strict_ok(seq: str, rp: RuleParams) -> bool:
    """
    New coverage rule:
      +1 if predicted precursor z ∈ {2,3,4} AND there are ≥3 predicted fragments
      (excluding b1/b2,y1/y2 and honoring 'avoid after‑Proline') whose fragment charge is 2 or 3.
    """
    zpred = predict_precursor_charge(seq, max_z=rp.max_precursor_z)
    if zpred not in (2, 3, 4):
        return False
    splits = predict_b_y_charges(seq, zpred)
    L = len(seq)
    need = 3
    got = 0
    for i, zb, zy, _L in splits:
        if rp.skip_short_frags_rule and (i <= 2 or (L - i) <= 2):
            continue
        if rp.avoid_post_pro_rule and seq[i - 1] == "P":
            continue
        # Count both b and y if they meet the charge in {2,3}
        if zb in (2, 3):
            got += 1
            if got >= need:
                return True
        if zy in (2, 3):
            got += 1
            if got >= need:
                return True
    return False


def rule_score(seq: str, p: RuleParams) -> Dict[str, float]:
    """
    RAW RULE TOTAL (no normalization; spacing bonus removed):
      +1  if N-term in nterm_good; -2 if strong-bad; -1 if weak-bad
      +2  for best P context GP (well), +1 GP near edge or AP, +0.5 weak-good, 0 neutral, -1 KR-before-P
      +1  if C-term hydrophobic
      -1  if any internal 'bad' residue present (e.g., Q)
      +1  if internal C present
      +1  if internal M present
      +1/0/-1 for basic-site rule (internal R/K vs only H vs none)
      +0.5 if ≥4 internal hydrophobics (tie reducer; configurable)
      +0.25 preference if C and M are on opposite sides of a P (either orientation; configurable)
      +1.0 charge rule if (pred precursor z in [2..4]) and (any fragment has charge ≥2 with pruning)
      +1.0 **new** charge-coverage rule if (pred precursor z ∈ {2,3,4}) and (≥3 fragments have charge 2 or 3 with pruning)
    """
    # (1) N-term
    nterm_aa = seq[0]
    if nterm_aa in p.nterm_good:
        nterm_raw = 1.0
    elif nterm_aa in p.nterm_bad_strong:
        nterm_raw = -2.0
    elif nterm_aa in p.nterm_bad_weak:
        nterm_raw = -1.0
    else:
        nterm_raw = 0.0

    # (2) Proline context
    pro_raw, pro_cls = _proline_context_raw(seq, edge_near=1)

    # (3) C-term hydrophobicity
    cterm = 1.0 if (seq[-1] in AA_HYDRO_CTERM) else 0.0

    # (4) Internal "bad" penalty
    bad_penalty = -1.0 if any((aa in p.bad_internal) for aa in seq[1:-1]) else 0.0

    # (5) Internal C/M bonuses
    c_internal = 1.0 if "C" in seq[1:-1] else 0.0
    m_internal = 1.0 if "M" in seq[1:-1] else 0.0

    # (6) Basic-site rule
    internal_aa = seq[1:-1]
    has_rk = any(a in ("R", "K") for a in internal_aa)
    has_h = "H" in internal_aa
    if has_rk:
        basic_raw, basic_cls = 1.0, "RK"
    elif has_h:
        basic_raw, basic_cls = 0.0, "H_only"
    else:
        basic_raw, basic_cls = -1.0, "none"

    # (7) Internal hydrophobic tie‑bonus
    n_hydro_int = sum(a in ("I", "L", "V", "F", "Y", "W", "M") for a in internal_aa)
    hydro_int_bonus = (
        float(p.hydro_internal_bonus)
        if n_hydro_int >= int(p.hydro_internal_min)
        else 0.0
    )

    # (8) C/M opposite‑sides preference
    cm_bonus = _cm_opposite_sides_bonus(seq, float(p.cm_split_bonus))

    # (9) Charge rule (+1)
    charge_bonus = 1.0 if _charge_rule_ok(seq, p) else 0.0
    # (10) New: ≥3 fragments have charge 2 or 3 (+1)
    charge_frag3_bonus = 1.0 if _charge_rule_strict_ok(seq, p) else 0.0

    total = (
        nterm_raw
        + pro_raw
        + cterm
        + bad_penalty
        + c_internal
        + m_internal
        + basic_raw
        + hydro_int_bonus
        + cm_bonus
        + charge_bonus
        + charge_frag3_bonus
    )
    return {
        "rule_nterm_raw": nterm_raw,
        "rule_proline_raw": pro_raw,
        "rule_proline_class": pro_cls,
        "rule_cterm_raw": cterm,
        "rule_bad_internal_raw": bad_penalty,
        "rule_internal_c_raw": c_internal,
        "rule_internal_m_raw": m_internal,
        "rule_basic_sites_raw": basic_raw,
        "rule_basic_class": basic_cls,
        "rule_hydro_internal_bonus_raw": hydro_int_bonus,
        "rule_cm_split_bonus_raw": cm_bonus,
        "rule_charge_bonus_raw": charge_bonus,
        "rule_charge_frag3_bonus_raw": charge_frag3_bonus,
        "rule_total": total,
    }


@dataclass
class EditParams:
    nterm_targets: Tuple[str, ...]


# =============================================================================
# Stage expansion helpers (enumeration; stages may skip iff criterion met)
# =============================================================================


def _compact_token(stage: str, label: str) -> Optional[str]:
    """Build a compact token from a human-readable label like 'C@5' or 'Nterm:A→R'."""
    if not label or label.endswith(":keep"):
        return None
    if stage in ("Nterm", "Cterm"):
        try:
            src, dst = label.split(":")[1].split("→")
            return ("N" if stage == "Nterm" else "T") + f"{src}>{dst}"
        except Exception:
            return ("N" if stage == "Nterm" else "T") + label.replace(":", "")
    m = re.match(r"([A-Z])@(\d+)", label)
    if m:
        aa, pos = m.group(1), m.group(2)
        return (aa if stage == "RK" and aa in ("R", "K") else stage[0]) + pos
    return stage[0] + "?"


@dataclass
class PathState:
    seq: str
    stage_names: List[str]  # nodes: ["orig","Nterm","Cterm",...]
    stage_seqs: List[str]  # same length as stage_names
    labels: List[str]  # edge labels between nodes (len = len(stage_names)-1)
    tokens: List[str]  # only applied-op tokens (no 'keep')
    # mutation context
    mut_idx0: Optional[int] = None
    is_frameshift: bool = False
    wt_aa: Optional[str] = None
    mut_aa: Optional[str] = None
    did_internal_edit: bool = False  # first-internal-edit tracking

    def extend(
        self, stage: str, new_seq: str, label: str, edited: bool = False
    ) -> "PathState":
        new_names = self.stage_names + [stage]
        new_seqs = self.stage_seqs + [new_seq]
        new_labels = self.labels + [label or f"{stage}:keep"]
        tok = _compact_token(stage, label)
        new_tokens = self.tokens + ([tok] if tok else [])
        return PathState(
            new_seq,
            new_names,
            new_seqs,
            new_labels,
            new_tokens,
            self.mut_idx0,
            self.is_frameshift,
            self.wt_aa,
            self.mut_aa,
            self.did_internal_edit or (edited and stage in ("P", "C", "M", "RK")),
        )


def enumerate_terminals_by_rule(
    seq: str,
    rp: RuleParams,
    ep: EditParams,
    mut_idx0: Optional[int],
    wt_aa: Optional[str],
    mut_aa: Optional[str],
    is_fs: bool,
    term_beam: int = 3,
) -> List[PathState]:
    """Enumerate N-term then C-term; keep up to top-K by rule_total at each stage, with WT/MUT exclusions at the mutant terminus."""
    starts: List[Tuple[str, str]] = []

    # N-term
    a0 = seq[0]
    nterm_keep = (a0 in PROTECTED_SET) or (a0 in rp.nterm_good)
    if nterm_keep:
        nterm_cands = [(seq, "Nterm:keep")]
    else:
        targets = [t for t in ep.nterm_targets if t != a0]
        if not is_fs and mut_idx0 == 0:
            targets = [t for t in targets if t != (wt_aa or "") and t != (mut_aa or "")]
        opts = [t + seq[1:] for t in targets] + [seq]
        scored = [
            (
                s,
                f"Nterm:{a0}→{s[0]}" if s != seq else "Nterm:keep",
                rule_score(s, rp)["rule_total"],
            )
            for s in opts
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        best = scored[0][2]
        nterm_cands = [(s, l) for (s, l, r) in scored if r == best][: max(1, term_beam)]

    # C-term
    out_states: List[PathState] = []
    for sN, labN in nterm_cands:
        last = sN[-1]
        cterm_keep = (
            (last in PROTECTED_SET) or (last == "A") or (last in AA_HYDRO_CTERM)
        )
        if cterm_keep:
            cterm_cands = [(sN, "Cterm:keep")]
        else:
            hydros = [aa for aa in AA_HYDRO_CTERM if aa != last]
            if not is_fs and mut_idx0 == len(sN) - 1:
                hydros = [
                    aa for aa in hydros if aa != (wt_aa or "") and aa != (mut_aa or "")
                ]
            tries = [sN[:-1] + aa for aa in hydros] + [sN]
            scored = [
                (
                    s,
                    f"Cterm:{last}→{s[-1]}" if s != sN else "Cterm:keep",
                    rule_score(s, rp)["rule_total"],
                )
                for s in tries
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            best = scored[0][2]
            cterm_cands = [(s, l) for (s, l, r) in scored if r == best][
                : max(1, term_beam)
            ]

        for sNT, labT in cterm_cands:
            out_states.append(
                PathState(
                    seq=sNT,
                    stage_names=["orig", "Nterm", "Cterm"],
                    stage_seqs=[seq, sN, sNT],
                    labels=[labN, labT],
                    tokens=[_compact_token("Nterm", labN)]
                    * (0 if labN.endswith("keep") else 1)
                    + (
                        [_compact_token("Cterm", labT)]
                        if not labT.endswith("keep")
                        else []
                    ),
                    mut_idx0=mut_idx0,
                    is_frameshift=bool(is_fs),
                    wt_aa=wt_aa,
                    mut_aa=mut_aa,
                    did_internal_edit=False,
                )
            )
    return out_states


def expand_stage(state: PathState, stage: str, rp: RuleParams, args) -> List[PathState]:
    """Expand by one internal stage respecting skip‑if‑met and mutant‑first rules."""
    s = state.seq
    L = len(s)
    out: List[PathState] = []

    def _keep():
        out.append(state.extend(stage, s, f"{stage}:keep", edited=False))

    internal = s[1:-1]

    # --- skip criteria (criterion already met) ---
    if stage == "P" and ("P" in internal):
        _keep()
        return out
    if stage == "C" and ("C" in internal):
        _keep()
        return out
    if stage == "M" and ("M" in internal):
        _keep()
        return out
    if stage == "RK":
        # NEW: skip RK if an internal R/K already exists OR any R exists anywhere (incl. termini)
        if any(a in ("R", "K") for a in internal) or ("R" in s):
            _keep()
            return out

    # special-case: mutation P->C means skip C stage entirely
    if (
        (not state.is_frameshift)
        and stage == "C"
        and (state.wt_aa == "P" and state.mut_aa == "C")
    ):
        _keep()
        return out

    # helper: allowed positions
    p_idxs = [i for i in range(1, L - 1) if s[i] == "P"]
    disallowed = set()
    # never edit residue immediately before a Proline
    for p in p_idxs:
        if p - 1 >= 1:
            disallowed.add(p - 1)

    def _range_all():
        return [
            i
            for i in range(1, L - 1)
            if i not in disallowed and s[i] not in PROTECTED_SET
        ]

    # P placement window (fixed: correct upper/lower bounds to enforce 1-based [4..n-2])
    if stage == "P":
        # 1-based [4..n-2] -> 0-based [3..L-3]; also clamp CLI to spec
        lo = max(3, int(args.p_position_min) - 1)
        hi = L - 3
        allowed = [
            i
            for i in range(lo, hi + 1)
            if i not in disallowed and s[i] not in PROTECTED_SET and s[i] != "P"
        ]
    else:
        allowed = [i for i in _range_all()]

    # optional: split preference around P (we consider BOTH orientations; preference is a rule bonus)
    # We therefore do not prune to only one side; we just avoid forbidden edits.

    # never overwrite existing stage residue; never set to WT/MUT at the mutant site
    def _positions_for_target(target_aa: str) -> List[int]:
        pos = []
        for i in allowed:
            if s[i] == target_aa:
                continue
            if (
                (not state.is_frameshift)
                and (state.mut_idx0 is not None)
                and (i == int(state.mut_idx0))
            ):
                if target_aa in ((state.wt_aa or ""), (state.mut_aa or "")):
                    continue
            pos.append(i)
        return pos

    def _apply_at(i: int, aa: str, label_aa: Optional[str] = None, edited=True):
        new = s[:i] + aa + s[i + 1 :]
        lab = f"{(label_aa or aa)}@{i+1}"
        out.append(state.extend(stage, new, lab, edited=edited))

    # --- first internal edit must be at mutant site (if non‑FS and none done) ---
    first_internal_needed = (
        (not state.is_frameshift)
        and (not state.did_internal_edit)
        and (state.mut_idx0 is not None)
        and (0 < int(state.mut_idx0) < L - 1)
    )

    if stage == "P":
        target = "P"
        if first_internal_needed:
            mi = int(state.mut_idx0)
            if (
                mi in allowed
                and s[mi] not in PROTECTED_SET
                and s[mi] != "P"
                and target not in ((state.wt_aa or ""), (state.mut_aa or ""))
            ):
                _apply_at(mi, "P")
                return out
            _keep()
            return out  # cannot legally place P at mut site as the first internal edit
        for i in _positions_for_target("P"):
            _apply_at(i, "P")
        if len(out) == 0:
            _keep()
        return out

    if stage in ("C", "M"):
        target = stage
        if first_internal_needed:
            mi = int(state.mut_idx0)
            if target in ((state.wt_aa or ""), (state.mut_aa or "")):
                _keep()
                return out
            if mi in _positions_for_target(target):
                _apply_at(mi, target)
                return out
            _keep()
            return out
        for i in _positions_for_target(target):
            _apply_at(i, target)
        if len(out) == 0:
            _keep()
        return out

    if stage == "RK":
        if first_internal_needed:
            mi = int(state.mut_idx0)
            options = []
            for aa in ("R", "K"):
                if aa in ((state.wt_aa or ""), (state.mut_aa or "")):
                    continue
                if mi in _positions_for_target(aa):
                    options.append((mi, aa))
            if options:
                for i, aa in options:
                    _apply_at(i, aa, label_aa=aa)
                return out
            _keep()
            return out
        for i in allowed:
            if s[i] in PROTECTED_SET:
                continue
            for aa in ("R", "K"):
                if s[i] == aa:
                    continue
                _apply_at(i, aa, label_aa=aa)
        if len(out) == 0:
            _keep()
        return out

    _keep()
    return out


# =============================================================================
# Tie-breaker & final scoring (for permutation winners only)
# =============================================================================


def parse_weight_map(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split(":")
        out[k.strip()] = float(v.strip())
    return out


def fly_surface_norm(seq: str) -> float:
    """Map KD mean from [-4.5, 4.5] → [0,1]."""
    if not seq:
        return 0.0
    gravy = kd_gravy(seq)
    return float(min(1.0, max(0.0, (gravy + 4.5) / 9.0)))


def internal_basic_h_sulfur_metrics(seq: str) -> Dict[str, float]:
    core = seq[1:-1] if len(seq) > 2 else ""
    rk = sum(ch in ("R", "K") for ch in core)
    h = sum(ch == "H" for ch in core)
    s_count = sum(ch in ("C", "M") for ch in seq)
    L = max(1, len(seq))
    return {
        "rk_internal": float(rk),
        "h_internal": float(h),
        "protons_score": float(rk) + 0.5 * float(h),
        "sulfur_frac": float(s_count) / float(L),
        "hydro_frac": hydrophobic_fraction(seq),
    }


def _warn_if_fraction(name, arr):
    if len(arr) == 0:
        return
    m, M = float(np.nanmin(arr)), float(np.nanmax(arr))
    if m < -1e-6 or M > 1 + 1e-6:
        LOG.warning(f"[units] {name} expected in [0,1] but spans [{m:.3f}, {M:.3f}]")


def _warn_if_integerish(name, arr):
    if len(arr) == 0:
        return
    dif = np.nanmax(np.abs(arr - np.rint(arr)))
    if dif > 1e-6:
        LOG.warning(
            f"[units] {name} expected integer-valued but non-integer deltas up to {dif:.3g}"
        )


# =============================================================================
# Plotting helpers (full-sequence colored per stage)
# =============================================================================

STAGE_COLOR = {
    "orig": "#000000",
    "Nterm": "#6a3d9a",
    "Cterm": "#d62728",
    "P": "#ff7f0e",
    "C": "#1f77b4",
    "M": "#2ca02c",
    "RK": "#8c564b",
}
BG_COLORS = {
    "orig_not_hit": "#7f7f7f",
    "orig_hit": "#9467bd",
    "decoys_not_selected": "#9ecae1",
    "decoys_selected": "#e41a1c",
}
MONO_FONT_SIZE = 8


def _grid(ax):
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)


def truncate_seq(s: str, maxlen: int = 48) -> str:
    if len(s) <= maxlen:
        return s
    keep = max(8, (maxlen - 3) // 2)
    return f"{s[:keep]}…{s[-keep:]}"


def _build_ledger_history(
    stage_seqs: List[str], stage_names: List[str]
) -> List[List[Optional[str]]]:
    if not stage_seqs:
        return []
    L = len(stage_seqs[0])
    ledgers = []
    cur = [None] * L
    ledgers.append(cur.copy())
    for k in range(1, len(stage_seqs)):
        prev, curseq, stage = stage_seqs[k - 1], stage_seqs[k], stage_names[k]
        for i, (a, b) in enumerate(zip(prev, curseq)):
            if a != b:
                cur[i] = stage
        ledgers.append(cur.copy())
    return ledgers


def _draw_full_sequence(ax, x, y, seq: str, ledger: List[Optional[str]]):
    xoff = 6
    yoff = -14
    for i, ch in enumerate(seq):
        tag = ledger[i] if i < len(ledger) else None
        color = STAGE_COLOR.get(tag or "orig", "#000000")
        ax.annotate(
            ch,
            (x, y),
            textcoords="offset points",
            xytext=(xoff + i * 7, yoff),
            fontsize=MONO_FONT_SIZE,
            family="monospace",
            color=color,
            zorder=4,
        )


def edit_path_plot_colored(
    outdir: str,
    rank_idx: int,
    selected_flag: bool,
    stage_names: List[str],
    xvals: List[float],
    yvals: List[float],
    stage_seqs: List[str],
    title_extra: str = "",
    label_len: int = 48,
    edge_labels: Optional[List[str]] = None,  # NEW: show actual op label per step
):
    ensure_outdir(outdir)
    fig, ax = plt.subplots()

    for i in range(len(stage_names) - 1):
        ax.plot(
            [xvals[i], xvals[i + 1]],
            [yvals[i], yvals[i + 1]],
            lw=1.4,
            color="#777777",
            alpha=0.9,
            zorder=1,
        )

    ledgers = _build_ledger_history(stage_seqs, stage_names)

    for i, st in enumerate(stage_names):
        c = STAGE_COLOR.get(st, "#444444")
        ax.scatter(
            [xvals[i]],
            [yvals[i]],
            s=50 if i == len(stage_names) - 1 else 44,
            color=c,
            edgecolor="k",
            linewidths=0.6,
            zorder=3,
        )
        # Annotate with stage AND edge label for clarity (no label for origin)
        node_label = st
        if edge_labels and i > 0 and (i - 1) < len(edge_labels):
            node_label = f"\n{edge_labels[i - 1]}"
        print(node_label)
        ax.annotate(
            node_label,
            (xvals[i], yvals[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=c,
        )
        _draw_full_sequence(
            ax, xvals[i], yvals[i], truncate_seq(stage_seqs[i], label_len), ledgers[i]
        )

    ax.set_xlabel("fly_surface_norm (0–1, KD-normalized)")
    ax.set_ylabel("confusability (0–1)")
    ttl1 = f"R{rank_idx:03d} — Edit path"
    ttl2 = f"orig: {truncate_seq(stage_seqs[0], label_len)}  →  final: {truncate_seq(stage_seqs[-1], label_len)}"
    if title_extra:
        ttl2 += f"  |  {title_extra}"
    ax.set_title(ttl1 + "\n" + ttl2)

    for name in ["Nterm", "Cterm", "P", "C", "M", "RK"]:
        ax.scatter(
            [], [], color=STAGE_COLOR[name], label=name, edgecolor="k", linewidths=0.4
        )
    ax.legend(frameon=True, fontsize=8)

    _grid(ax)
    fig.tight_layout()
    fname = f"R{rank_idx:03d}__score_{title_extra.replace('Δscore=','').replace(' ','_')}__{'SEL' if selected_flag else 'NOT'}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


# =============================================================================
# Plotting helpers (extracted orchestration)
# =============================================================================


def plot_orig_vs_decoy_scores(
    df_dec: pd.DataFrame,
    outpath: str,
    *,
    selected_col: str = "selected",
    orig_score_col: str = "start_score",
    decoy_score_col: str = "final_score",
    title: str = "Original vs decoy composite score",
    dense_threshold: int = 1500,
) -> None:
    """
    Scatter (or hex) plot of original peptide's score vs cognate decoy's score.
    Highlights selected decoys.

    Expects each decoy row to carry the *original* peptide's score in `orig_score_col`
    (e.g., start_score) and its own decoy score in `decoy_score_col` (e.g., final_score).
    """
    # Sanity checks / safe subset
    if (orig_score_col not in df_dec.columns) or (
        decoy_score_col not in df_dec.columns
    ):
        return
    d = df_dec[[orig_score_col, decoy_score_col, selected_col]].copy()
    d = d.dropna(subset=[orig_score_col, decoy_score_col])
    if d.empty:
        return

    # Basic masks
    sel_mask = (
        d[selected_col] == True
        if selected_col in d.columns
        else pd.Series(False, index=d.index)
    )
    nsel_mask = ~sel_mask

    x = d[orig_score_col].to_numpy(float)
    y = d[decoy_score_col].to_numpy(float)
    x_sel, y_sel = x[sel_mask.to_numpy()], y[sel_mask.to_numpy()]
    x_nsel, y_nsel = x[nsel_mask.to_numpy()], y[nsel_mask.to_numpy()]

    # Figure
    fig, ax = plt.subplots()

    # If very dense, use a hexbin background for the *non-selected* cloud
    total_pts = len(x)
    used_hexbin = False
    if total_pts >= dense_threshold and len(x_nsel) > 0:
        hb = ax.hexbin(
            x_nsel,
            y_nsel,
            gridsize=50,
            mincnt=1,
            linewidths=0.0,
            alpha=0.7,
        )
        used_hexbin = True
        # Add a colorbar only if helpful
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("count (non-selected)")
    else:
        # Regular scatter for non-selected
        if len(x_nsel):
            ax.scatter(
                x_nsel,
                y_nsel,
                s=12,
                alpha=0.45,
                label=f"Decoys (not selected, n={len(x_nsel)})",
                color=BG_COLORS.get("decoys_not_selected", "#9e9e9e"),
            )

    # Selected overlay (always as points)
    if len(x_sel):
        ax.scatter(
            x_sel,
            y_sel,
            s=22,
            alpha=0.9,
            label=f"Decoys (selected, n={len(x_sel)})",
            color=BG_COLORS.get("decoys_selected", "#ff7f0e"),
            edgecolor="k",
            linewidths=0.4,
            zorder=3,
        )

    # y=x reference line
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    lo = np.nanmin([xmin, ymin])
    hi = np.nanmax([xmax, ymax])
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#555555", alpha=0.9, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Correlations (optional; shown in subtitle)
    try:
        if total_pts >= 3:
            r_pear = np.corrcoef(x, y)[0, 1]
            # Spearman is more robust if ranks tie; fall back gracefully
            from scipy.stats import spearmanr  # optional dep, ignore if unavailable

            r_spear = spearmanr(x, y, nan_policy="omit").statistic
            subtitle = f"(Pearson r={r_pear:.3f}, Spearman ρ={r_spear:.3f})"
        else:
            subtitle = ""
    except Exception:
        subtitle = ""

    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.set_xlabel(f"original score [{orig_score_col}]")
    ax.set_ylabel(f"decoy score [{decoy_score_col}]")
    if not used_hexbin:
        ax.legend(frameon=True, fontsize=8)

    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_scatter_groups(
    outpath: str,
    orig_xy: Tuple[np.ndarray, np.ndarray],
    hit_mask: Optional[np.ndarray],
    nsel_xy: Tuple[np.ndarray, np.ndarray],
    sel_xy: Tuple[np.ndarray, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    dense_threshold: int = 1500,
):
    """
    Draw three groups: originals (optionally split by hit mask), non‑selected decoys, selected decoys.
    """
    fig, ax = plt.subplots()
    if len(orig_xy[0]):
        if hit_mask is not None and len(hit_mask) == len(orig_xy[0]):
            x, y = np.asarray(orig_xy[0]), np.asarray(orig_xy[1])
            hm = np.asarray(hit_mask, dtype=bool)
            ax.scatter(
                x[~hm],
                y[~hm],
                s=10,
                alpha=0.35,
                label=f"Originals (not hit, n={int((~hm).sum())})",
                color=BG_COLORS["orig_not_hit"],
            )
            ax.scatter(
                x[hm],
                y[hm],
                s=30,
                alpha=0.85,
                marker="^",
                label=f"Originals (hit, n={int(hm.sum())})",
                color=BG_COLORS["orig_hit"],
                edgecolor="k",
                linewidths=0.4,
            )
        else:
            ax.scatter(
                orig_xy[0],
                orig_xy[1],
                s=10,
                alpha=0.35,
                label=f"Originals (n={len(orig_xy[0])})",
                color=BG_COLORS["orig_not_hit"],
            )
    if len(nsel_xy[0]):
        ax.scatter(
            nsel_xy[0],
            nsel_xy[1],
            s=12,
            alpha=0.45,
            label=f"Decoys (not selected, n={len(nsel_xy[0])})",
            color=BG_COLORS["decoys_not_selected"],
        )
    if len(sel_xy[0]):
        ax.scatter(
            sel_xy[0],
            sel_xy[1],
            s=18,
            alpha=0.80,
            label=f"Decoys (selected, n={len(sel_xy[0])})",
            color=BG_COLORS["decoys_selected"],
            edgecolor="k",
            linewidths=0.4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_hist(df_dec: pd.DataFrame, col: str, outpath: str, title: str) -> None:
    """One‑column histogram."""
    if df_dec.empty or col not in df_dec.columns:
        return
    fig, ax = plt.subplots()
    ax.hist(df_dec[col].dropna().to_numpy(float), bins=30, alpha=0.8, color="#3182bd")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def render_all_plots(
    df_dec: pd.DataFrame,
    outdir: str,
    trace_label_maxlen: int = 48,
    dense_threshold: int = 1500,
    no_progress: bool = False,
) -> None:
    """Render edit‑path plots, scatters, and histograms into <outdir>/plots/."""
    plots_dir = os.path.join(outdir, "plots")
    ep_dir = os.path.join(plots_dir, "edit_paths")
    ensure_outdir(ep_dir)

    df_plot = (
        df_dec.copy().sort_values("final_score", ascending=False).reset_index(drop=True)
    )

    it = df_plot.itertuples(index=False)
    if not no_progress:
        it = tqdm(it, total=len(df_plot), desc="EditPaths(global)")

    for rank, rr in enumerate(it, start=1):
        stages = rr.path_stages.split("|") if rr.path_stages else ["orig"]
        seqs = rr.path_seqs.split("|") if rr.path_seqs else [rr.peptide]
        xvals = [float(x) for x in rr.path_fly.split("|")] if rr.path_fly else [rr.fly0]
        yvals = (
            [float(y) for y in rr.path_conf.split("|")] if rr.path_conf else [rr.conf0]
        )
        edge_labels = rr.edit_path.split(" → ") if rr.edit_path else []
        edit_path_plot_colored(
            ep_dir,
            rank_idx=rank,
            selected_flag=bool(rr.selected),
            stage_names=stages,
            xvals=xvals,
            yvals=yvals,
            stage_seqs=seqs,
            title_extra=f"Δscore={rr.delta_total_score:.3f}",
            label_len=trace_label_maxlen,
            edge_labels=edge_labels,
        )

    orig_hit_mask = (
        df_dec["is_hit"].to_numpy(bool) if "is_hit" in df_dec.columns else None
    )
    sel_mask = df_dec["selected"] == True
    nsel_mask = ~sel_mask

    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_rk_internal_vs_hydro.png"),
        orig_xy=(
            df_dec["rk_internal0"].to_numpy(float),
            df_dec["hydrophobic_fraction0"].to_numpy(float),
        ),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "rk_internal"].to_numpy(float),
            df_dec.loc[nsel_mask, "hydrophobic_fraction"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "rk_internal"].to_numpy(float),
            df_dec.loc[sel_mask, "hydrophobic_fraction"].to_numpy(float),
        ),
        xlabel="internal R/K count (integer)",
        ylabel="hydrophobic fraction (0–1)",
        title="Internal basic sites vs hydrophobicity",
        dense_threshold=dense_threshold,
    )
    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_protons_vs_conf.png"),
        orig_xy=(
            df_dec["protons_score0"].to_numpy(float),
            df_dec["conf0"].to_numpy(float),
        ),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "protons_score"].to_numpy(float),
            df_dec.loc[nsel_mask, "confT"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "protons_score"].to_numpy(float),
            df_dec.loc[sel_mask, "confT"].to_numpy(float),
        ),
        xlabel="side‑chain protons score (R/K + 0.5·H; count‑like)",
        ylabel="confusability (0–1)",
        title="Protons on side chains vs confusability",
        dense_threshold=dense_threshold,
    )
    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_sulfur_vs_hydro.png"),
        orig_xy=(
            df_dec["sulfur_frac0"].to_numpy(float),
            df_dec["hydrophobic_fraction0"].to_numpy(float),
        ),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "sulfur_frac"].to_numpy(float),
            df_dec.loc[nsel_mask, "hydrophobic_fraction"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "sulfur_frac"].to_numpy(float),
            df_dec.loc[sel_mask, "hydrophobic_fraction"].to_numpy(float),
        ),
        xlabel="S-content fraction (C+M)/length (0–1)",
        ylabel="hydrophobic fraction (0–1)",
        title="S-content vs hydrophobicity",
        dense_threshold=dense_threshold,
    )
    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_fly_vs_conf.png"),
        orig_xy=(df_dec["fly0"].to_numpy(float), df_dec["conf0"].to_numpy(float)),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "flyT"].to_numpy(float),
            df_dec.loc[nsel_mask, "confT"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "flyT"].to_numpy(float),
            df_dec.loc[sel_mask, "confT"].to_numpy(float),
        ),
        xlabel="fly_surface_norm (0–1)",
        ylabel="confusability (0–1)",
        title="Confusability vs flyability",
        dense_threshold=dense_threshold,
    )
    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_final_vs_fly.png"),
        orig_xy=(df_dec["fly0"].to_numpy(float), df_dec["start_score"].to_numpy(float)),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "flyT"].to_numpy(float),
            df_dec.loc[nsel_mask, "final_score"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "flyT"].to_numpy(float),
            df_dec.loc[sel_mask, "final_score"].to_numpy(float),
        ),
        xlabel="fly_surface_norm (0–1)",
        ylabel="composite score (arbitrary units)",
        title="Composite score vs flyability",
        dense_threshold=dense_threshold,
    )
    plot_scatter_groups(
        outpath=os.path.join(plots_dir, "scatter_final_vs_conf.png"),
        orig_xy=(
            df_dec["conf0"].to_numpy(float),
            df_dec["start_score"].to_numpy(float),
        ),
        hit_mask=orig_hit_mask,
        nsel_xy=(
            df_dec.loc[nsel_mask, "confT"].to_numpy(float),
            df_dec.loc[nsel_mask, "final_score"].to_numpy(float),
        ),
        sel_xy=(
            df_dec.loc[sel_mask, "confT"].to_numpy(float),
            df_dec.loc[sel_mask, "final_score"].to_numpy(float),
        ),
        xlabel="confusability (0–1)",
        ylabel="composite score (arbitrary units)",
        title="Composite score vs confusability",
        dense_threshold=dense_threshold,
    )

    plot_orig_vs_decoy_scores(
        df_dec=df_dec,
        outpath=os.path.join(plots_dir, "scatter_orig_vs_decoy_scores.png"),
        selected_col="selected",
        orig_score_col="start_score",  # original peptide’s composite score column
        decoy_score_col="final_score",  # cognate decoy’s composite score column
        title="Original vs cognate decoy scores",
        dense_threshold=dense_threshold,
    )
    plot_hist(
        df_dec,
        "delta_total_score",
        os.path.join(plots_dir, "hist_delta_total_score.png"),
        "Distribution of final score improvements",
    )
    plot_hist(
        df_dec,
        "rule_delta",
        os.path.join(plots_dir, "hist_rule_delta.png"),
        "Distribution of rule score changes",
    )


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decoy generation (RULE-FIRST permutations; mutation-aware; charge heuristic; unit-consistent plots)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O & indexing
    p.add_argument("--in", dest="input_csv", required=True, help="Input CSV")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--peptide-col", default="peptide", help="Column with peptide sequence"
    )
    p.add_argument(
        "--mut-idx-col", default=None, help="0-based mutant index (peptide coordinate)"
    )
    p.add_argument(
        "--pos-col",
        default=None,
        help="Peptide position column (interpreted by --indexing)",
    )
    p.add_argument(
        "--mut-notation-col",
        default=None,
        help="Peptide-coordinate mutation notation (e.g., p.G5D, MAP2_fs)",
    )
    p.add_argument(
        "--indexing",
        choices=["0", "1", "C1"],
        default="1",
        help="Position interpretation: 0=N-term 0-based; 1=N-term 1-based; C1=1-based from C-term (1=last)",
    )
    p.add_argument(
        "--require-mut-idx0",
        type=int,
        default=1,
        help="1=fail when mut_idx0 missing/invalid (non-frameshifts); 0=auto-place near middle",
    )
    p.add_argument(
        "--is-hit-col",
        default="is_hit",
        help="Optional boolean/0-1 column marking original hits (for overlay in plots)",
    )

    # Sets
    p.add_argument(
        "--sets",
        default="albumin,keratins,proteases,mhc_hardware",
        help="Comma list of contaminant sets",
    )
    p.add_argument(
        "--sets-fasta-dir", default=None, help="Directory with <set>.fasta files"
    )
    p.add_argument(
        "--download-sets",
        type=int,
        default=1,
        help="1=download UniProt accessions for sets",
    )
    p.add_argument("--download-timeout", type=int, default=30)

    # Matching / confusability
    p.add_argument(
        "--ppm-tol",
        type=float,
        default=30.0,
        help="PPM tolerance for precursor/fragment matching",
    )
    p.add_argument(
        "--good-ppm-strict",
        type=float,
        default=10.0,
        help="PPM threshold for strong matches",
    )
    p.add_argument(
        "--fragment-kmin", type=int, default=2, help="Min fragment length (k)"
    )
    p.add_argument(
        "--fragment-kmax", type=int, default=7, help="Max fragment length (k)"
    )
    p.add_argument(
        "--full-mz-len-min", type=int, default=5, help="Contaminant window min length"
    )
    p.add_argument(
        "--full-mz-len-max", type=int, default=15, help="Contaminant window max length"
    )
    p.add_argument(
        "--rt-tolerance-min",
        type=float,
        default=1.0,
        help="RT co-elution tolerance (min)",
    )
    p.add_argument(
        "--gradient-min", type=float, default=20.0, help="Gradient length (min)"
    )
    p.add_argument(
        "--cys-mod",
        choices=["none", "carbamidomethyl"],
        default="carbamidomethyl",
        help="Fixed mod on Cys",
    )

    # Charges (legacy fallback)
    p.add_argument(
        "--charges",
        default="2,3",
        help="Fallback precursor charges (used if predictor disabled)",
    )
    p.add_argument(
        "--frag-charges",
        default="1,2,3",
        help="Fallback fragment charges (used if predictor disabled)",
    )
    p.add_argument(
        "--frag-types-req",
        type=int,
        default=3,
        help="Min distinct (ion,k,z) for strong match",
    )

    # PEG config
    p.add_argument("--polymer-families", default="PEG,PPG,PTMEG,PDMS")
    p.add_argument("--polymer-endgroups", default="auto")
    p.add_argument("--polymer-adducts", default="H,Na,K,NH4")
    p.add_argument("--polymer-z", default="1,2,3")
    p.add_argument("--polymer-n-min", type=int, default=5)
    p.add_argument("--polymer-n-max", type=int, default=100)
    p.add_argument("--polymer-mz-min", type=float, default=200.0)
    p.add_argument("--polymer-mz-max", type=float, default=1500.0)
    p.add_argument(
        "--polymer-ppm-strict",
        type=float,
        default=10.0,
        help="≤ this ppm → polymer_match_flag=1",
    )

    # Rules
    p.add_argument("--nterm-good", default="RYLFDM")
    p.add_argument("--nterm-bad-strong", default="KP")
    p.add_argument("--nterm-bad-weak", default="STVA")
    p.add_argument("--nterm-targets", default="RYLFDM")
    p.add_argument("--bad-internal", default="Q")
    p.add_argument("--rule-flank-gap-min", type=int, default=2)

    # Beam & preferences
    p.add_argument(
        "--term-beam",
        type=int,
        default=3,
        help="Top-K terminal variants kept by rule score per terminal stage",
    )
    p.add_argument(
        "--beam-internal",
        type=int,
        default=128,
        help="Beam width for internal-stage frontiers per permutation",
    )
    p.add_argument(
        "--hydro-internal-min",
        type=int,
        default=4,
        help="≥ this many internal hydrophobics earns a small bonus",
    )
    p.add_argument(
        "--hydro-internal-bonus",
        type=float,
        default=0.5,
        help="Rule bonus if internal hydrophobics ≥ threshold",
    )
    p.add_argument(
        "--cm-split-bonus",
        type=float,
        default=0.25,
        help="Rule bonus if C and M lie on opposite sides of a P",
    )
    p.add_argument(
        "--p-position-min",
        type=int,
        default=4,
        help="Earliest 1-based P position allowed (max is n-2)",
    )

    # Charge predictor & fragment pruning
    p.add_argument(
        "--use-charge-predictor",
        type=int,
        default=1,
        help="1=use heuristic predicted precursor charge to limit matching",
    )
    p.add_argument("--min-precursor-z", type=int, default=2)
    p.add_argument("--max-precursor-z", type=int, default=4)
    p.add_argument(
        "--require-frag-z2",
        type=int,
        default=1,
        help="1=require ≥1 predicted fragment with z≥2 (used in conf & the charge rule)",
    )
    p.add_argument(
        "--skip-short-frags",
        type=int,
        default=1,
        help="1=skip b1,b2,y1,y2 in fragment matching & in charge rule",
    )
    p.add_argument(
        "--avoid-post-proline-cleavage",
        type=int,
        default=1,
        help="1=avoid cleavages C-terminal to Proline for fragments & the charge rule",
    )

    # Rankings / tie-break (global composite simplified)
    p.add_argument(
        "--rank-weights",
        default="hydro:0.3,conf_mz:0.3,hk:0.3,polymer:0.10",
        help="Weights for global composite (normalized to [0,1]): hydro, conf_mz, hk, polymer",
    )
    p.add_argument("--overall-rule-weight", type=float, default=1.0)
    p.add_argument("--overall-cont-weight", type=float, default=2.0)
    p.add_argument(
        "--perm-combine-weights",
        default="conf:0.5,hydro:0.5",
        help="Weights for combining confusability (0–1) and hydrophobic fraction (0–1) among rule-best survivors",
    )

    # Selection
    p.add_argument("--N", type=int, default=10, help="Number of decoys to select")
    p.add_argument("--dedup-k", type=int, default=5, help="k-mer de-dup (0 disables)")
    p.add_argument(
        "--require-contam-match",
        type=int,
        default=1,
        help="Require conf_mz_multi ≥ 0.01 to select",
    )
    p.add_argument(
        "--enforce-hydrophobic-cterm",
        type=int,
        default=0,
        help="Require hydrophobic C-term at selection time",
    )
    p.add_argument(
        "--min-confusability-ratio",
        type=float,
        default=0.90,
        help="conf_new ≥ ratio × conf_old",
    )
    p.add_argument(
        "--require-hydro-increase",
        type=int,
        default=1,
        help="Require hydrophobic_fraction increase",
    )
    p.add_argument(
        "--hydro-delta-min",
        type=float,
        default=1e-6,
        help="Minimum hydrophobic_fraction increase",
    )

    # Housekeeping
    p.add_argument(
        "--housekeeping-fasta",
        default=None,
        help="Path to housekeeping FASTA (optional)",
    )
    p.add_argument(
        "--download-housekeeping",
        action="store_true",
        help="Download housekeeping proteins from UniProt",
    )
    p.add_argument(
        "--housekeeping-k", type=int, default=4, help="k for housekeeping homology"
    )
    p.add_argument(
        "--housekeeping-collapse",
        choices=["none", "xle", "xle_de_qn", "xle+deqn"],
        default="xle",
    )
    p.add_argument(
        "--xle-collapse",
        action="store_true",
        help="Collapse I/L→J for HK analysis (k-mer + substring)",
    )

    # Viz
    p.add_argument(
        "--dense-threshold",
        type=int,
        default=1500,
        help="Hexbin switch threshold for dense plots",
    )
    p.add_argument("--trace-label-maxlen", type=int, default=48)

    # Verbosity
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


# =============================================================================
# Mutation parsing (PEPTIDE coords)
# =============================================================================

_MUT_NOTATION_RE = re.compile(
    r"(?:(?:p\.)?\s*)?([A-Z*])\s*(\d+)\s*([A-Z*]|fs\*?)", flags=re.IGNORECASE
)


def _convert_pos_to_idx0(pos_val: int, indexing: str, L: int) -> Optional[int]:
    if pos_val is None:
        return None
    try:
        v = int(pos_val)
    except Exception:
        return None
    if indexing == "0":
        idx0 = v
    elif indexing == "1":
        idx0 = v - 1
    elif indexing == "C1":
        idx0 = L - v
    else:
        idx0 = None
    if idx0 is None or idx0 < 0 or idx0 >= L:
        return None
    return idx0


def _parse_mut_notation_peptide(
    seq: str, notation: str, indexing: str
) -> Tuple[Optional[int], Optional[str], Optional[str], bool]:
    assert notation
    assert len(notation) > 0
    assert indexing in ("0", "1", "C1")

    s = str(notation).strip().lower()
    is_protein_target = (s == "taa") or (s == "cta")
    is_fusion = s == "fusion"
    is_fs = bool(re.search(r"(_fs| fs|\d+fs)$", s, flags=re.IGNORECASE))
    modify_any_residue = is_fs or is_protein_target or is_fusion
    m = _MUT_NOTATION_RE.search(s)
    if not m:
        return None, None, None, modify_any_residue
    wt = m.group(1).upper()
    pos = int(m.group(2))
    mut = m.group(3).upper()
    if mut.startswith("FS"):
        return None, wt, None, True
    L = len(seq)
    idx0 = _convert_pos_to_idx0(pos, indexing, L)
    if idx0 is None or idx0 < 0 or idx0 >= L:
        return None, None, None, modify_any_residue
    if wt != seq[idx0]:
        return None, None, None, modify_any_residue
    return idx0, wt, mut, False


def _coerce_is_hit_col(s: pd.Series) -> pd.Series:
    def f(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            return bool(int(v))
        if isinstance(v, str):
            vv = v.strip().lower()
            return vv in ("1", "true", "t", "y", "yes", "hit")
        return False

    return s.map(f)


def resolve_mut_idx_column(
    df: pd.DataFrame,
    peptide_col: str,
    is_hit_col: str,
    mut_idx_col: str,
    pos_col: str,
    mut_notation_col: str,
    indexing: str,
    require_mut_idx0: bool,
    outdir: str,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:

    Ls = df[peptide_col].astype(str).map(len)

    mut0 = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    is_fs_series = pd.Series(False, index=df.index, dtype=bool)
    wt_series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    mut_series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    is_hit_series = pd.Series([False] * len(df), index=df.index, dtype=bool)

    if is_hit_col and is_hit_col in df.columns:
        is_hit_series = _coerce_is_hit_col(df[is_hit_col])

    if mut_idx_col and mut_idx_col in df.columns:
        s = pd.to_numeric(df[mut_idx_col], errors="coerce").astype("Int64")
        mut0 = s.copy()

    if mut0.isna().any() and pos_col and pos_col in df.columns:
        pos_raw = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
        pos_idx0 = []
        for i, pos in enumerate(pos_raw.tolist()):
            L = int(Ls.iloc[i])
            idx = None if pd.isna(pos) else _convert_pos_to_idx0(int(pos), indexing, L)
            pos_idx0.append(idx if idx is not None else pd.NA)
        pos_idx0 = pd.Series(pos_idx0, index=df.index, dtype="Int64")
        mut0 = mut0.fillna(pos_idx0)

    if mut_notation_col and mut_notation_col in df.columns:
        idx_list, wt_list, mut_list, fs_list = [], [], [], []
        for i in range(len(df)):
            seq = str(df.iloc[i][peptide_col])
            note = str(df.iloc[i][mut_notation_col])
            idx0, wt, mut, is_fs = _parse_mut_notation_peptide(seq, note, indexing)
            idx_list.append(idx0 if idx0 is not None else pd.NA)
            wt_list.append(wt if wt is not None else pd.NA)
            mut_list.append(mut if mut is not None else pd.NA)
            fs_list.append(bool(is_fs))
        idx_series = pd.Series(idx_list, index=df.index, dtype="Int64")
        wt_series = pd.Series(wt_list, index=df.index, dtype="object")
        mut_series = pd.Series(mut_list, index=df.index, dtype="object")
        is_fs_series = is_fs_series | pd.Series(fs_list, index=df.index, dtype=bool)

        mut0 = mut0.fillna(idx_series)

    missing_mask = mut0.isna() & (~is_fs_series)
    if missing_mask.any():
        if require_mut_idx0:
            bad = df.loc[
                missing_mask,
                [peptide_col]
                + [c for c in [mut_idx_col, pos_col, mut_notation_col] if c],
            ]
            path = os.path.join(outdir, "missing_mut_idx0_rows.csv")
            ensure_outdir(outdir)
            bad.to_csv(path, index=False)
            raise ValueError(
                f"[mut_idx0] Required but missing/invalid for {int(missing_mask.sum())} non-frameshift rows. See {path}."
            )
        auto = (
            df[peptide_col]
            .astype(str)
            .map(lambda s: max(1, min(len(s) - 2, len(s) // 2)))
        )
        mut0 = mut0.fillna(auto).astype("Int64")

    out = []
    errors = []
    for i, (idx0, L, is_fs) in enumerate(
        zip(mut0.tolist(), Ls.tolist(), is_fs_series.tolist())
    ):
        if is_fs:
            out.append(pd.NA)
            continue
        if idx0 is None or (isinstance(idx0, float) and math.isnan(idx0)):
            out.append(pd.NA)
            continue
        v = int(idx0)
        if v < 0 or v >= L:
            errors.append((i, f"out_of_range({v})_len({L})"))
        out.append(v)
    if errors:
        path = os.path.join(outdir, "invalid_mut_idx0_rows.csv")
        ensure_outdir(outdir)
        pd.DataFrame(
            [
                {"row": i, "reason": r, "peptide": df.iloc[i][peptide_col]}
                for i, r in errors
            ]
        ).to_csv(path, index=False)
        raise ValueError(f"[mut_idx0] Found {len(errors)} invalid indices; see {path}")
    return (
        pd.Series(out, index=df.index, dtype="Int64"),
        is_fs_series.astype(bool),
        wt_series,
        mut_series,
        is_hit_series,
    )


# =============================================================================
# Pipeline
# =============================================================================


def _global_tie(hn, cn, hkn, pn, weights):
    # all inputs are normalized to [0,1]
    return (
        weights.get("hydro", 0.0) * hn
        + weights.get("conf_mz", 0.0) * cn
        + weights.get("hk", 0.0) * hkn
        + weights.get("polymer", 0.0) * pn
    )


def _compress_path_for_display(
    state: PathState, xvals: List[float], yvals: List[float]
) -> Tuple[List[str], List[str], List[str], List[float], List[float]]:
    """
    Keep only nodes where an edit occurred (drop '...:keep' steps), always keep origin.
    Returns: (edge_labels, stage_names, stage_seqs, xvals, yvals)
    """
    kept_idx = [0]
    for i in range(1, len(state.stage_names)):
        if i - 1 < len(state.labels) and not str(state.labels[i - 1]).endswith("keep"):
            kept_idx.append(i)
    stage_names2 = [state.stage_names[i] for i in kept_idx]
    stage_seqs2 = [state.stage_seqs[i] for i in kept_idx]
    xvals2 = [xvals[i] for i in kept_idx]
    yvals2 = [yvals[i] for i in kept_idx]
    edge_labels2 = [state.labels[i - 1] for i in kept_idx[1:]]
    return edge_labels2, stage_names2, stage_seqs2, xvals2, yvals2


def load_csv(
    input_csv: str,
    peptide_col: str,
    is_hit_col: str,
    mut_idx_col: str,
    pos_col: str,
    mut_notation_col: str,
    indexing: str,
    require_mut_idx0: bool,
    outdir: str,
) -> pd.DataFrame:
    df_in = pd.read_csv(input_csv)

    pepcols = peptide_col.split(",")
    for peptide_col in pepcols:
        if peptide_col in df_in.columns:
            break
    else:
        raise ValueError(f"[load_csv] No peptide column found among: {pepcols}")

    pos_cols = pos_col.split(",") if pos_col else []

    for pos_col in pos_cols:
        if pos_col in df_in.columns:
            break
    else:
        raise ValueError(f"[load_csv] No pos_col found among: {pos_cols}")

    df = df_in.copy()
    df[peptide_col] = df[peptide_col].map(clean_seq)
    df = df[df[peptide_col].str.len() > 0].copy()
    LOG.info(f"Loaded {len(df)} rows from {input_csv} with valid peptides.")

    # Mut idx / frameshift / WT/MUT letters (peptide coordinates) + is_hit
    df["mut_idx0"], df["is_frameshift"], df["wt_aa"], df["mut_aa"], df["is_hit"] = (
        resolve_mut_idx_column(
            df,
            peptide_col=peptide_col,
            is_hit_col=is_hit_col,
            mut_idx_col=mut_idx_col,
            pos_col=pos_col,
            mut_notation_col=mut_notation_col,
            indexing=indexing,
            require_mut_idx0=require_mut_idx0,
            outdir=outdir,
        )
    )
    return df, peptide_col, pos_col


def main():
    args = make_parser().parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    ensure_outdir(args.outdir)
    ensure_outdir(os.path.join(args.outdir, "plots", "edit_paths"))

    # Load input
    df, peptide_col, pos_col = load_csv(
        args.input_csv,
        args.peptide_col,
        args.is_hit_col,
        args.mut_idx_col,
        args.pos_col,
        args.mut_notation_col,
        args.indexing,
        args.require_mut_idx0,
        args.outdir,
    )

    # Sets & windows
    set_names = [s.strip() for s in str(args.sets).split(",") if s.strip()]
    acc2seq_by_set = load_sets(
        set_names, args.sets_fasta_dir, bool(args.download_sets), args.download_timeout
    )

    charges_prec_fallback = sorted({int(z) for z in parse_csv_list(args.charges, int)})
    frag_charges_fallback = sorted(
        {int(z) for z in parse_csv_list(args.frag_charges, int)}
    )

    # All confusability options are passed explicitly (never as function attributes)
    conf_kwargs = dict(
        use_charge_pred=bool(args.use_charge_predictor),
        max_pred_z=int(args.max_precursor_z),
        skip_short_frags=bool(args.skip_short_frags),
        avoid_post_proline=bool(args.avoid_post_proline_cleavage),
        require_frag_z2=bool(args.require_frag_z2),
        frag_charges=frag_charges_fallback,
    )

    cys_fixed = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0

    windows_by_set: Dict[str, List[Window]] = {}
    LOG.info("Indexing windows + fragments for contaminant sets …")
    for s in tqdm(set_names, disable=bool(args.no_progress), desc="Sets"):
        wins = build_windows_index(
            s,
            acc2seq_by_set[s],
            args.full_mz_len_min,
            args.full_mz_len_max,
            charges_prec_fallback,
            args.gradient_min,
            cys_fixed,
        )
        index_fragments_for_windows(
            wins,
            args.fragment_kmin,
            args.fragment_kmax,
            frag_charges_fallback,
            cys_fixed,
        )
        windows_by_set[s] = wins
        LOG.info(f"[windows] {s}: {len(wins)} windows indexed")

    # Housekeeping bank + substring banks
    hk_bank: Set[str] = set()
    hk_sub_banks: Dict[int, Set[str]] = {}
    hk_collapse_mode = "xle" if args.xle_collapse else str(args.housekeeping_collapse)
    if args.housekeeping_fasta or args.download_housekeeping:
        hk_seqs = load_housekeeping_sequences(
            args.housekeeping_fasta, args.download_housekeeping
        )
        if hk_seqs:
            hk_bank = build_hk_kmer_bank(
                hk_seqs, k=int(args.housekeeping_k), collapse=hk_collapse_mode
            )
            hk_sub_banks = build_hk_substring_banks(
                hk_seqs, collapse=hk_collapse_mode, Lmax=20
            )
            LOG.info(
                f"[housekeeping] bank size={len(hk_bank)} (k={args.housekeeping_k}, collapse={hk_collapse_mode}); substring banks up to L=20."
            )
        else:
            LOG.info("[housekeeping] no sequences found; HK disabled")
    else:
        LOG.info("[housekeeping] skipped (no FASTA / no download)")

    # PEG-like candidates
    polymer_cands = enumerate_polymer_series(
        families=tuple(
            x.strip() for x in str(args.polymer_families).split(",") if x.strip()
        ),
        endgroups=tuple(
            x.strip() for x in str(args.polymer_endgroups).split(",") if x.strip()
        ),
        adducts=tuple(
            x.strip() for x in str(args.polymer_adducts).split(",") if x.strip()
        ),
        zset=tuple(int(z.strip()) for z in str(args.polymer_z).split(",") if z.strip()),
        n_min=int(args.polymer_n_min),
        n_max=int(args.polymer_n_max),
        mz_min=float(args.polymer_mz_min),
        mz_max=float(args.polymer_mz_max),
    )
    LOG.info(f"[polymer] candidates: {len(polymer_cands)}")

    # Rule/Edit params & weights
    rp = RuleParams(
        nterm_good=tuple(args.nterm_good),
        nterm_bad_strong=tuple(args.nterm_bad_strong),
        nterm_bad_weak=tuple(args.nterm_bad_weak),
        nterm_targets=tuple(args.nterm_targets),
        bad_internal=tuple(args.bad_internal),
        hydro_internal_min=int(args.hydro_internal_min),
        hydro_internal_bonus=float(args.hydro_internal_bonus),
        cm_split_bonus=float(args.cm_split_bonus),
        min_precursor_z=int(args.min_precursor_z),
        max_precursor_z=int(args.max_precursor_z),
        skip_short_frags_rule=bool(args.skip_short_frags),
        avoid_post_pro_rule=bool(args.avoid_post_proline_cleavage),
    )
    ep = EditParams(nterm_targets=tuple(args.nterm_targets))
    weights_global = parse_weight_map(args.rank_weights)
    weights_perm = parse_weight_map(args.perm_combine_weights)
    w_conf = float(weights_perm.get("conf", 0.5))
    w_hyd = float(weights_perm.get("hydro", 0.5))

    # -------------------- Score originals + fit transforms --------------------
    LOG.info(
        "Scoring originals (global composite: normalized hydro + conf + HK + optional polymer) …"
    )
    rows = []

    for seq in tqdm(df[peptide_col], disable=bool(args.no_progress), desc="Originals"):

        r = rule_score(seq, rp)
        hydro_raw = kd_gravy(seq)
        fly = fly_surface_norm(seq)
        rt_pred = predict_rt_min(seq, args.gradient_min)

        conf_per_set = {}
        for s in set_names:
            wins = windows_by_set[s]
            assert len(wins) > 0, f"No windows for set '{s}'"

            d = confusability_in_mz(
                seq,
                wins,
                args.full_mz_len_min,
                args.full_mz_len_max,
                charges_prec_fallback,
                args.ppm_tol,
                rt_pred,
                args.rt_tolerance_min,
                cys_fixed,
                args.fragment_kmin,
                args.fragment_kmax,
                good_ppm=args.good_ppm_strict,
                frag_types_req=args.frag_types_req,
                **conf_kwargs,
            )
            conf_per_set[s] = d["score"]
        conf_mz_multi = max(conf_per_set.values()) if conf_per_set else 0.0

        hk = hk_kmer_features(
            seq, hk_bank, k=int(args.housekeeping_k), collapse=hk_collapse_mode
        )
        polymer = polymer_best_for_peptide(
            seq,
            polymer_cands,
            peptide_z=tuple(charges_prec_fallback),
            ppm_window=args.polymer_ppm_strict,
            cys_fixed_mod=cys_fixed,
        )

        rows.append(
            {
                "seq": seq,
                **r,
                "kd_gravy": hydro_raw,
                "fly_surface_norm": fly,
                "conf_mz_multi": conf_mz_multi,
                "hk_kmer_hits": hk["hk_kmer_hits"],
                "hk_kmer_frac": hk["hk_kmer_frac"],
                "polymer_conf": polymer.get("polymer_conf", 0.0),
                "polymer_match_flag": polymer.get("polymer_match_flag", 0),
                "polymer_best_abs_ppm": polymer.get(
                    "polymer_best_abs_ppm", float("inf")
                ),
                "polymer_details": polymer.get("polymer_details", ""),
            }
        )
    base = pd.DataFrame(rows)

    # Robust transforms (0..1)
    t_hydro = RobustMinMax(base["kd_gravy"])
    t_conf = RobustMinMax(base["conf_mz_multi"])
    t_hk = RobustMinMax(base["hk_kmer_frac"])
    t_poly = RobustMinMax(base["polymer_conf"])

    base["hydro_norm"] = t_hydro.transform(base["kd_gravy"])
    base["confusability_norm"] = t_conf.transform(base["conf_mz_multi"])
    base["hk_kmer_norm"] = t_hk.transform(base["hk_kmer_frac"])
    base["polymer_norm"] = t_poly.transform(base["polymer_conf"])

    base["tie_break_score"] = [
        _global_tie(hn, cn, hkn, pn, weights_global)
        for hn, cn, hkn, pn in zip(
            base["hydro_norm"],
            base["confusability_norm"],
            base["hk_kmer_norm"],
            base["polymer_norm"],
        )
    ]
    base["final_rank_score"] = (
        float(args.overall_rule_weight) * base["rule_total"]
        + float(args.overall_cont_weight) * base["tie_break_score"]
    )

    out_all = pd.concat(
        [df.reset_index(drop=True), base.reset_index(drop=True)], axis=1
    )
    if out_all.columns.duplicated().any():
        out_all = out_all.loc[:, ~out_all.columns.duplicated()]

    all_csv = os.path.join(args.outdir, "all_candidates_scored.csv")
    out_all.to_csv(all_csv, index=False)
    LOG.info(f"Wrote: {all_csv} (n={len(out_all)})")

    # -------------------- Helper caches for conf/hydro ------------------------
    @lru_cache(maxsize=200000)
    def conf_score_cached(s: str) -> float:
        rt_pred = predict_rt_min(s, args.gradient_min)
        confs = []
        for _set in set_names:
            wins = windows_by_set[_set]
            if not wins:
                confs.append(0.0)
                continue
            d = confusability_in_mz(
                s,
                wins,
                args.full_mz_len_min,
                args.full_mz_len_max,
                charges_prec_fallback,
                args.ppm_tol,
                rt_pred,
                args.rt_tolerance_min,
                cys_fixed,
                args.fragment_kmin,
                args.fragment_kmax,
                good_ppm=args.good_ppm_strict,
                frag_types_req=args.frag_types_req,
                **conf_kwargs,
            )
            confs.append(d["score"])
        return max(confs) if confs else 0.0

    @lru_cache(maxsize=200000)
    def hydro_frac_cached(s: str) -> float:
        return hydrophobic_fraction(s)

    # -------------------- RULE-FIRST decoy per peptide ------------------------
    LOG.info("Synthesizing one best decoy per peptide (RULE-FIRST permutations) …")
    dec_rows = []
    pbar2 = tqdm(
        range(len(out_all)), disable=bool(args.no_progress), desc="Decoys(all)"
    )
    for i in pbar2:
        seq = out_all.loc[i, "seq"]
        is_fs = (
            bool(out_all.loc[i, "is_frameshift"])
            if "is_frameshift" in out_all.columns
            else False
        )

        starts = enumerate_terminals_by_rule(
            seq,
            rp,
            ep,
            mut_idx0=(
                None
                if pd.isna(out_all.loc[i, "mut_idx0"])
                else int(out_all.loc[i, "mut_idx0"])
            ),
            wt_aa=(
                None
                if pd.isna(out_all.loc[i, "wt_aa"])
                else str(out_all.loc[i, "wt_aa"])
            ),
            mut_aa=(
                None
                if pd.isna(out_all.loc[i, "mut_aa"])
                else str(out_all.loc[i, "mut_aa"])
            ),
            is_fs=is_fs,
            term_beam=int(args.term_beam),
        )

        # permutations of internal stages (P, C, M, RK), beam-pruned after each stage
        STAGE_ORDERS = list(permutations(("P", "C", "M", "RK"), 4))
        global_best_rule = -1e99
        global_states: Dict[str, List[PathState]] = {}
        for permutation in STAGE_ORDERS:
            frontier: List[PathState] = starts[:]
            for stage in permutation:
                new_frontier: List[PathState] = []
                for st in frontier:
                    new_frontier.extend(expand_stage(st, stage, rp, args))
                # group by sequence and keep canonical path per sequence
                seen: Dict[str, Tuple[Tuple[float, int, str], PathState]] = {}
                for st in new_frontier:
                    r = rule_score(st.seq, rp)["rule_total"]
                    tokstr = ";".join([t for t in st.tokens if t])
                    key = st.seq
                    val = (
                        r,
                        -len(st.tokens),
                        tokstr,
                    )  # pref: higher rule, fewer edits, lexicographically earlier token string
                    if (key not in seen) or (val > seen[key][0]):
                        seen[key] = (val, st)
                cand = [(v[0][0], v[1]) for v in seen.values()]  # (rule_total, state)
                cand.sort(key=lambda t: t[0], reverse=True)
                frontier = [st for (_, st) in cand[: max(1, int(args.beam_internal))]]

            scores = [(st, rule_score(st.seq, rp)["rule_total"]) for st in frontier]
            if not scores:
                continue
            perm_best = max(v for _, v in scores)
            winners = [st for (st, v) in scores if v == perm_best]
            if perm_best > global_best_rule:
                global_best_rule = perm_best
                global_states = {}
            if perm_best == global_best_rule:
                for st in winners:
                    global_states.setdefault(st.seq, []).append(st)

        # If nothing changed, force a single M substitution (best position)
        if (
            (len(global_states) == 1)
            and (seq in global_states)
            and all(len(st.tokens) == 0 for st in global_states[seq])
        ):
            bestM = None
            bestM_rule = -1e99
            for pos in range(0, len(seq)):
                if seq[pos] in PROTECTED_SET or seq[pos] == "M":
                    continue
                s2 = seq[:pos] + "M" + seq[pos + 1 :]
                r = rule_score(s2, rp)["rule_total"]
                if r > bestM_rule:
                    bestM_rule, bestM = r, PathState(
                        seq=s2,
                        stage_names=["orig", "M"],
                        stage_seqs=[seq, s2],
                        labels=["M@%d" % (pos + 1)],
                        tokens=["M%d" % (pos + 1)],
                        mut_idx0=(
                            None
                            if pd.isna(out_all.loc[i, "mut_idx0"])
                            else int(out_all.loc[i, "mut_idx0"])
                        ),
                        is_frameshift=is_fs,
                        wt_aa=(
                            None
                            if pd.isna(out_all.loc[i, "wt_aa"])
                            else str(out_all.loc[i, "wt_aa"])
                        ),
                        mut_aa=(
                            None
                            if pd.isna(out_all.loc[i, "mut_aa"])
                            else str(out_all.loc[i, "mut_aa"])
                        ),
                        did_internal_edit=True,
                    )
            if bestM is not None:
                global_best_rule = bestM_rule
                global_states = {bestM.seq: [bestM]}

        # canonical path per unique sequence
        seq_to_best_state: Dict[str, PathState] = {}
        for candidate_seq, states in global_states.items():
            states2 = []
            for st in states:
                tokstr = ";".join([t for t in st.tokens if t])
                states2.append((len(st.tokens), tokstr, st))
            states2.sort(key=lambda t: (t[0], t[1]))
            seq_to_best_state[candidate_seq] = states2[0][2]

        # Continuous among rule-best: conf (0–1) and hydro fraction (0–1)
        scored: List[Tuple[float, float, float, int, int, str, PathState]] = []
        for candidate_seq, st in seq_to_best_state.items():
            conf = conf_score_cached(candidate_seq)
            hyd = hydro_frac_cached(candidate_seq)
            comb = w_conf * conf + w_hyd * hyd
            hk_long = (
                hk_longest_substring_len(
                    candidate_seq,
                    hk_sub_banks,
                    collapse=(
                        "xle" if args.xle_collapse else args.housekeeping_collapse
                    ),
                )
                if hk_sub_banks
                else 0
            )
            acid_cnt = sum(ch in ("D", "E") for ch in candidate_seq)
            scored.append((comb, conf, hyd, hk_long, acid_cnt, candidate_seq, st))
        if not scored:
            final_seq = seq
            final_state = PathState(
                seq,
                ["orig"],
                [seq],
                [],
                [],
                mut_idx0=(
                    None
                    if pd.isna(out_all.loc[i, "mut_idx0"])
                    else int(out_all.loc[i, "mut_idx0"])
                ),
                is_frameshift=is_fs,
                wt_aa=(
                    None
                    if pd.isna(out_all.loc[i, "wt_aa"])
                    else str(out_all.loc[i, "wt_aa"])
                ),
                mut_aa=(
                    None
                    if pd.isna(out_all.loc[i, "mut_aa"])
                    else str(out_all.loc[i, "mut_aa"])
                ),
            )
        else:
            # comb ↓, hyd ↓, HK ↓, acids ↑, lexicographic ↑
            scored.sort(key=lambda t: (-t[0], -t[2], -t[3], t[4], t[5]))
            final_seq = scored[0][5]
            final_state = scored[0][6]

        # 5) Evaluate endpoints for CSV/plots
        def _eval(s: str) -> Dict[str, float]:
            rt_pred = predict_rt_min(s, args.gradient_min)
            confs = []
            for _set in set_names:
                wins = windows_by_set[_set]
                if not wins:
                    confs.append(0.0)
                    continue
                d = confusability_in_mz(
                    s,
                    wins,
                    args.full_mz_len_min,
                    args.full_mz_len_max,
                    charges_prec_fallback,
                    (
                        args.ppm_tolerance
                        if hasattr(args, "ppm_tolerance")
                        else args.ppm_tol
                    ),
                    rt_pred,
                    args.rt_tolerance_min,
                    cys_fixed,
                    args.fragment_kmin,
                    args.fragment_kmax,
                    good_ppm=args.good_ppm_strict,
                    frag_types_req=args.frag_types_req,
                    **conf_kwargs,
                )
                confs.append(d["score"])
            polymer = polymer_best_for_peptide(
                s,
                polymer_cands,
                peptide_z=tuple(charges_prec_fallback),
                ppm_window=args.polymer_ppm_strict,
                cys_fixed_mod=cys_fixed,
            )
            hk = hk_kmer_features(
                s,
                hk_bank,
                k=int(args.housekeeping_k),
                collapse=("xle" if args.xle_collapse else args.housekeeping_collapse),
            )
            return {
                "fly": fly_surface_norm(s),
                "hydro": hydrophobic_fraction(s),
                "conf": max(confs) if confs else 0.0,
                "hk_frac": hk["hk_kmer_frac"],
                "polymer_conf": polymer.get("polymer_conf", 0.0),
                "polymer_match_flag": polymer.get("polymer_match_flag", 0),
                "polymer_best_abs_ppm": polymer.get(
                    "polymer_best_abs_ppm", float("inf")
                ),
            }

        s0 = seq
        sF = final_seq
        e0 = _eval(s0)
        eF = _eval(sF)
        r_old = rule_score(s0, rp)
        r_new = rule_score(sF, rp)

        # stage path values for plots (x=fly, y=conf)
        xvals_full = [fly_surface_norm(s) for s in final_state.stage_seqs]
        yvals_full = []
        for s in final_state.stage_seqs:
            rt_pred = predict_rt_min(s, args.gradient_min)
            confs = []
            for _set in set_names:
                wins = windows_by_set[_set]
                if not wins:
                    confs.append(0.0)
                    continue
                d = confusability_in_mz(
                    s,
                    wins,
                    args.full_mz_len_min,
                    args.full_mz_len_max,
                    charges_prec_fallback,
                    args.ppm_tol,
                    rt_pred,
                    args.rt_tolerance_min,
                    cys_fixed,
                    args.fragment_kmin,
                    args.fragment_kmax,
                    good_ppm=args.good_ppm_strict,
                    frag_types_req=args.frag_types_req,
                    **conf_kwargs,
                )
                confs.append(d["score"])
            yvals_full.append(max(confs) if confs else 0.0)

        # Compress the path to only actual edits (drop :keep steps); this fixes "final node is RK" confusion
        edge_labels_disp, stage_names_disp, stage_seqs_disp, xvals_disp, yvals_disp = (
            _compress_path_for_display(final_state, xvals_full, yvals_full)
        )

        # Units-correct metrics for original & final
        origm = internal_basic_h_sulfur_metrics(s0)
        finalm = internal_basic_h_sulfur_metrics(sF)

        # Global composite (normalized)
        hydro_old_n = t_hydro.transform_scalar(kd_gravy(s0))
        hydro_new_n = t_hydro.transform_scalar(kd_gravy(sF))
        conf_old_n = t_conf.transform_scalar(e0["conf"])
        conf_new_n = t_conf.transform_scalar(eF["conf"])
        hk_old_n = t_hk.transform_scalar(e0["hk_frac"])
        hk_new_n = t_hk.transform_scalar(eF["hk_frac"])
        poly_old_n = t_poly.transform_scalar(e0["polymer_conf"])
        poly_new_n = t_poly.transform_scalar(eF["polymer_conf"])

        tie_old = _global_tie(
            hydro_old_n, conf_old_n, hk_old_n, poly_old_n, weights_global
        )
        tie_new = _global_tie(
            hydro_new_n, conf_new_n, hk_new_n, poly_new_n, weights_global
        )

        final_old = (
            float(args.overall_rule_weight) * r_old["rule_total"]
            + float(args.overall_cont_weight) * tie_old
        )
        final_new = (
            float(args.overall_rule_weight) * r_new["rule_total"]
            + float(args.overall_cont_weight) * tie_new
        )

        dec_rows.append(
            {
                "peptide": s0,
                "decoy_seq": sF,
                "selected": False,
                "is_frameshift": bool(is_fs),
                "is_hit": (
                    bool(out_all.loc[i, "is_hit"])
                    if "is_hit" in out_all.columns
                    else False
                ),
                "mut_idx0": (
                    -1
                    if pd.isna(out_all.loc[i, "mut_idx0"])
                    else int(out_all.loc[i, "mut_idx0"])
                ),
                "wt_aa": (
                    ""
                    if pd.isna(out_all.loc[i, "wt_aa"])
                    else str(out_all.loc[i, "wt_aa"])
                ),
                "mut_aa": (
                    ""
                    if pd.isna(out_all.loc[i, "mut_aa"])
                    else str(out_all.loc[i, "mut_aa"])
                ),
                # Save compressed/clean path info (edits only)
                "edit_path": " → ".join(edge_labels_disp),
                "path_stages": "|".join(stage_names_disp),
                "path_seqs": "|".join(stage_seqs_disp),
                "path_fly": "|".join(f"{v:.6f}" for v in xvals_disp),
                "path_conf": "|".join(f"{v:.6f}" for v in yvals_disp),
                # endpoints
                "fly0": e0["fly"],
                "conf0": e0["conf"],
                "flyT": eF["fly"],
                "confT": eF["conf"],
                # final continuous components (fractions 0..1)
                "fly_surface_norm": eF["fly"],
                "hydrophobic_fraction": eF["hydro"],  # 0..1
                "conf_mz_multi": eF["conf"],  # 0..1
                "hk_kmer_frac": eF["hk_frac"],  # 0..1
                "polymer_conf": eF["polymer_conf"],  # 0..1
                "polymer_match_flag": int(eF["polymer_match_flag"]),
                "polymer_best_abs_ppm": eF["polymer_best_abs_ppm"],
                # rule totals (raw)
                "rule_total_old": r_old["rule_total"],
                "rule_total_new": r_new["rule_total"],
                "rule_delta": r_new["rule_total"] - r_old["rule_total"],
                # global dataset composite
                "start_score": final_old,
                "final_score": final_new,
                "delta_total_score": final_new - final_old,
                # ORIGINAL metrics (counts/fractions)
                "rk_internal0": origm["rk_internal"],
                "protons_score0": origm["protons_score"],
                "sulfur_frac0": origm["sulfur_frac"],
                "hydrophobic_fraction0": origm["hydro_frac"],
                # FINAL metrics (counts/fractions)
                "rk_internal": finalm["rk_internal"],
                "protons_score": finalm["protons_score"],
                "sulfur_frac": finalm["sulfur_frac"],
            }
        )

    df_dec = pd.DataFrame(dec_rows)
    if not df_dec.empty:
        df_dec["start_rank"] = (
            df_dec["start_score"].rank(ascending=False, method="min").astype(int)
        )
        df_dec["end_rank"] = (
            df_dec["final_score"].rank(ascending=False, method="min").astype(int)
        )
        df_dec["rank_improvement"] = df_dec["start_rank"] - df_dec["end_rank"]
        n_total = max(1, len(df_dec) - 1)
        df_dec["rank_improvement_norm"] = df_dec["rank_improvement"].clip(
            lower=0
        ).astype(float) / float(n_total)

    dec_csv = os.path.join(args.outdir, "decoys_all.csv")
    df_dec.to_csv(dec_csv, index=False)
    LOG.info(f"Wrote: {dec_csv} (n={len(df_dec)})")

    # -------------------- Selection (dataset-level; unchanged) ----------------
    N = int(args.N)
    kdedup = int(args.dedup_k)
    require_contam = bool(args.require_contam_match)
    enforce_cterm = bool(args.enforce_hydrophobic_cterm)

    orig_scores = out_all["final_rank_score"].copy()
    K = min(len(orig_scores), 3 * N)
    worst_orig_idx = orig_scores.nsmallest(K).index.tolist()
    sorted_all = orig_scores.sort_values(ascending=False).reset_index(drop=True)
    threshold_T = (
        float(sorted_all.iloc[N - 1])
        if len(sorted_all) >= N
        else float(sorted_all.iloc[-1])
    )
    LOG.info(
        f"[selection] Top‑N threshold among originals (Nth best) = {threshold_T:.6f}"
    )

    worst_orig_peps = set(out_all.loc[worst_orig_idx, "seq"].tolist())
    candidate_mask = df_dec["peptide"].isin(worst_orig_peps) & (
        df_dec["final_score"] >= threshold_T
    )
    remaining = set(df_dec.index[candidate_mask].tolist())

    selected_idx = []
    seen_kmers = set()

    def hard_constraints_ok(rr) -> bool:
        if require_contam and (float(rr["conf_mz_multi"]) < 0.01):
            return False
        if enforce_cterm and (rr["decoy_seq"][-1] not in "FILVWYM"):
            return False
        conf_ok = float(rr["conf_mz_multi"]) >= float(
            args.min_confusability_ratio
        ) * float(rr["conf0"])
        if not conf_ok:
            return False
        hydro_ok = float(rr["hydrophobic_fraction"]) >= hydrophobic_fraction(
            rr["peptide"]
        ) + float(args.hydro_delta_min)
        if args.require_hydro_increase and not hydro_ok:
            return False
        return True

    while len(selected_idx) < N and remaining:
        best_i = None
        best_score = -1e99
        for irow in list(remaining):
            rr = df_dec.loc[irow]
            if not hard_constraints_ok(rr):
                continue
            if kdedup > 0:
                kms = kmer_set(rr["decoy_seq"], kdedup)
                if kms & seen_kmers:
                    continue
            sc = float(rr["final_score"])
            if sc > best_score:
                best_score = sc
                best_i = irow
        if best_i is None:
            break
        selected_idx.append(best_i)
        df_dec.at[best_i, "selected"] = True
        if kdedup > 0:
            for km in kmer_set(df_dec.loc[best_i, "decoy_seq"], kdedup):
                seen_kmers.add(km)
        remaining.remove(best_i)

    selected = (
        df_dec.loc[selected_idx].copy().sort_values("final_score", ascending=False)
    )
    df_dec.to_csv(dec_csv, index=False)
    sel_csv = os.path.join(args.outdir, "decoys_selected_topN.csv")
    selected.to_csv(sel_csv, index=False)
    LOG.info(f"Wrote: {sel_csv} (n={len(selected)})")

    # -------------------- Unit sanity checks ---------------------------------
    _warn_if_integerish("rk_internal0", df_dec["rk_internal0"].to_numpy(float))
    _warn_if_integerish("rk_internal", df_dec["rk_internal"].to_numpy(float))
    _warn_if_fraction(
        "hydrophobic_fraction0", df_dec["hydrophobic_fraction0"].to_numpy(float)
    )
    _warn_if_fraction(
        "hydrophobic_fraction", df_dec["hydrophobic_fraction"].to_numpy(float)
    )
    _warn_if_fraction("sulfur_frac0", df_dec["sulfur_frac0"].to_numpy(float))
    _warn_if_fraction("sulfur_frac", df_dec["sulfur_frac"].to_numpy(float))
    _warn_if_fraction("conf0", df_dec["conf0"].to_numpy(float))
    _warn_if_fraction("confT", df_dec["confT"].to_numpy(float))

    # -------------------- Plots (extracted) ---------------------------------
    render_all_plots(
        df_dec=df_dec,
        outdir=args.outdir,
        trace_label_maxlen=args.trace_label_maxlen,
        dense_threshold=args.dense_threshold,
        no_progress=bool(args.no_progress),
    )

    # README
    with open(os.path.join(args.outdir, "README.md"), "w") as fh:
        fh.write(
            "\n".join(
                [
                    "# Decoy generation (RULE-FIRST; mutation-aware; charge heuristic)",
                    "",
                    "## Rule components (raw, no normalization)",
                    "- N-term: +1 good; −1 weak-bad; −2 strong-bad",
                    "- Proline context: +2 GP (well-placed), +1 GP edge or AP, +0.5 weak-good, 0 neutral, −1 KR-before-P",
                    "- C-term hydrophobic: +1; Ala C-term is frozen",
                    "- Internal 'bad' (e.g., Q): −1; Internal C: +1; Internal M: +1",
                    "- Basic-site rule: +1 internal R/K; 0 only H; −1 none",
                    "- Tie reducers: +0.5 if ≥4 internal hydrophobics; +0.25 if C/M on opposite sides of P",
                    "- Charge rule A: +1 if predicted precursor z∈[2,4] and ≥1 predicted fragment (w/ pruning) has charge ≥2",
                    "- Charge rule B (new): +1 if predicted precursor z∈{2,3,4} and ≥3 predicted fragments (w/ pruning) have charge 2 or 3",
                    "",
                    "## Skip & mutation logic",
                    "- Skip a stage if its criterion is already met (e.g., P already internal → skip P stage).",
                    "- RK stage additionally skips if **any R exists anywhere** in the sequence (including termini).",
                    "- Non-frameshift: first internal edit must occur at the mutant site, never setting WT or MUT residue.",
                    "- Special: mut P→C → always skip C stage; P stage skips that position but may edit elsewhere later.",
                    "- P placements: 1-based positions in [4 .. n-2]; never edit residue before Proline.",
                    "",
                    "## Termini and permutations",
                    "- Termini: keep top-K by rule (beam) at N-term then C-term.",
                    "- Internal permutations of {P,C,M,RK} with per-stage beam pruning by rule_total.",
                    "- End of permutation: keep all sequences at the permutation's max rule_total; across permutations, keep global rule-best.",
                    "",
                    "## Final winner per peptide",
                    "- Combine **confusability (0–1)** and **hydrophobic fraction (0–1)** with --perm-combine-weights.",
                    "- Ties: higher hydrophobicity → longer HK substring → fewer D/E → lexicographic.",
                    "",
                    "## Global composite (dataset ranking)",
                    "- Normalized combination: hydro + conf_mz + hk (+ optional polymer), weights via --rank-weights.",
                    "",
                    "## Selection",
                    "- Bottom 2*N originals define candidate pool; threshold T = Nth best original; select decoys >= T with constraints.",
                    "",
                    "## Visualization",
                    "- Edit paths now **only include actual edits** (keep steps hidden). Nodes are annotated with both the stage and the exact edit label (e.g., `P@7`).",
                ]
            )
        )
    LOG.info(
        f"[DONE] Selected {len(selected)} / {len(df_dec)} decoys. Outdir: {args.outdir}"
    )


if __name__ == "__main__":
    main()
