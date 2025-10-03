#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoy generation with rule + continuous features, multi-source confusability,
housekeeping k‑mer parity with WIQD, frameshift handling, and rich plotting.

Highlights
----------
• Makes ONE decoy per input peptide via an explicit edit path:
  Nterm → C → P → M → Cterm (skips any step that’s already “good”).
• Frameshifts (“fs” in mutation notation):
  – Do not require a mutation index.
  – If the peptide lacks C, evaluate C placement at ANY position (including terminals)
    and pick the best candidate under the same ranking/tie rules.
• Rule score tiers: N-term tiers (good/bad-weak/bad-strong), proline context,
  hydrophobic C-term, internal penalties/bonuses (bad residues, internal C/M),
  plus a spacing bonus for C/M around P.
• Continuous tie-breakers and ranking signals:
  – Hydrophobicity (KD-derived fly_surface_norm) + KD hydrophobic fraction
  – Multi-source protein confusability (precursor m/z + RT + fragment chaining)
  – Housekeeping homology (HK k-mer fraction), matching WIQD logic
  – Optional PEG-like isobar proximity (families / endgroups / adducts)
  – AA-type diversity metric
• Selection is improvement- & diversity-aware with hard constraints (same defaults):
  – confusability_new ≥ min_ratio × confusability_old
  – hydrophobicity_new ≥ hydrophobicity_old + delta_min (if required)
  – optional hydrophobic C-term enforcement and contaminant-match requirement
  – k-mer de-duplication for the final Top-N
• Rich plots:
  – Classic edit-path plots (confusability vs hydrophobicity) in plots/edit_paths/
  – Per-letter colored stage labels in traces (selected & not selected)
  – Density-aware scatters, score-decomposition, deltas, distributions, falloff diagnostics

Dependencies
------------
numpy, pandas, matplotlib, tqdm, requests (only if --download-sets / --download-housekeeping is used)

Relies on your ecosystem (NO duplication):
  - wiqd_constants: AA, H2O, PROTON_MASS
  - wiqd_features: kd_gravy, hydrophobic_fraction, mass_neutral, mz_from_mass, mz_from_sequence
  - wiqd_rt_proxy: predict_rt_min
  - wiqd_peptide_similarity: enumerate_polymer_series
  - wiqd_proteins: load_housekeeping_sequences
  - wiqd_sequence_helpers: collapse_seq_mode
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass, field
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

# ------------------------ Imports from helpers (no duplication) ---------------
from wiqd_peptide_similarity import enumerate_polymer_series
from wiqd_constants import AA, H2O, PROTON_MASS as PROTON
from wiqd_features import (
    kd_gravy,
    hydrophobic_fraction,
    mz_from_mass,
    mass_neutral,
    mz_from_sequence,
)
from wiqd_rt_proxy import predict_rt_min

# Housekeeping parity with WIQD
from wiqd_proteins import load_housekeeping_sequences
from wiqd_sequence_helpers import collapse_seq_mode


# =============================================================================
# Small utilities
# =============================================================================
def ensure_outdir(p: str | os.PathLike):
    os.makedirs(p, exist_ok=True)


def parse_csv_list(s: Optional[str], cast=str) -> Tuple:
    if not s:
        return tuple()
    return tuple(cast(x.strip()) for x in s.split(",") if x.strip())


def collapse_xle(s: str, on: bool) -> str:
    return s.replace("I", "J").replace("L", "J") if on else s


def clean_seq(s: str) -> str:
    return "".join(ch for ch in str(s).strip().upper() if ch in AA)


def kmer_set(s: str, k: int) -> Set[str]:
    if k <= 0 or len(s) < k:
        return set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


class RobustMinMax:
    """Fit on a reference series; apply to others with clipping (robust bounds)."""

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
# Contaminant sets: accessions to download (or local FASTA files)
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
SET_FRIENDLY = {
    "albumin": "Albumin",
    "keratins": "Keratins",
    "proteases": "Proteases (autolysis)",
    "mhc_hardware": "MHC hardware (B2M/HLA)",
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
    # keep only AAs
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
    """
    Return dict: set_name -> {accession: sequence}. Requires local FASTA or --download-sets 1.
    """
    out: Dict[str, Dict[str, str]] = {}
    for s in set_names:
        if s not in DOWNLOAD_RECIPES:
            raise ValueError(f"Unknown set '{s}'. Available: {list(DOWNLOAD_RECIPES)}")
        acc2seq: Dict[str, str] = {}
        # 1) Local FASTA
        if sets_fasta_dir:
            f = os.path.join(sets_fasta_dir, f"{s}.fasta")
            if os.path.isfile(f):
                LOG.info(f"[sets] Reading FASTA: {f}")
                with open(f, "r") as fh:
                    acc2seq = _parse_fasta_to_acc_seq(fh.read())
        # 2) Download (if allowed)
        if not acc2seq and do_download:
            recipe = DOWNLOAD_RECIPES[s]
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
# Windows & confusability (internal implementation, tuned for speed)
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


def confusability_script1(
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
    frag_charges: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    """Score in [0,1] based on precursor m/z proximity, RT coelution, and chained b/y fragment matches."""
    pep_m = mass_neutral(pep, cys_fixed_mod)
    prec_zs = sorted({int(z) for z in precursor_charges})
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

    frag_zs = list(frag_charges) if frag_charges is not None else prec_zs

    def _pep_by_frag_mzs() -> List[Tuple[float, str, int, int]]:
        L = len(pep)
        out = []
        if L <= 1:
            return out
        kmax_eff = min(kmax, L - 1)
        kmin_eff = max(1, min(kmin, kmax_eff))
        for k in range(kmin_eff, kmax_eff + 1):
            bseq, yseq = pep[:k], pep[-k:]
            for z in frag_zs:
                bmz = (mass_neutral(bseq, cys_fixed_mod) - H2O + z * PROTON) / z
                ymz = (mass_neutral(yseq, cys_fixed_mod) + z * PROTON) / z
                out.append((bmz, "b", k, z))
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
    if best_all <= good_ppm and ft_rt >= frag_types_req:
        score = 1.0
    else:
        ppm_term = max(0.0, 1.0 - min(best_all, ppm_tol) / ppm_tol)
        frag_term = min(1.0, (0.7 * frac_rt + 0.3 * frac_all))
        score = max(0.0, min(1.0, 0.5 * ppm_term + 0.5 * frag_term))
    return {
        "score": score,
        "best_ppm": best_all,
        "best_ppm_rt": best_rt,
        "n_prec": float(len(prec_idxs)),
        "n_prec_rt": float(len(idx_prec_rt)),
        "frac_frag_rt": frac_rt,
        "frac_frag_all": frac_all,
        "prec_idxs": prec_idxs,
        "prec_idxs_rt": idx_prec_rt,
        "frag_types_rt": ft_rt,
    }


# =============================================================================
# PEG-like candidates (dict-based; fixes earlier tuple mismatch)
# =============================================================================
def polymer_best_for_peptide(
    seq: str,
    candidates: List[Dict[str, object]],
    peptide_z: Iterable[int] = (2, 3),
    ppm_window: float = 10.0,
    cys_fixed_mod: float = 0.0,
) -> Dict[str, object]:
    """
    Choose the best PEG-like match among candidates (dicts from enumerate_polymer_series).
    Returns a dict with polymer_match_flag, polymer_best_abs_ppm, polymer_conf, polymer_details.
    """
    if not candidates:
        return {
            "polymer_match_flag": 0,
            "polymer_best_abs_ppm": float("inf"),
            "polymer_details": "",
            "polymer_conf": 0.0,
        }

    best = (float("inf"), None, None, None)  # absppm, cand, z, pep_mz
    # index by z for speed
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
        f"adducts={cand['adducts']};polymer_z={cand['z']};"
        f"pep_z={pepz};polymer_mz={float(cand['mz']):.6f};pep_mz={pep_mz:.6f}"
    )
    return {
        "polymer_match_flag": 1 if absppm <= ppm_window else 0,
        "polymer_best_abs_ppm": float(absppm),
        "polymer_conf": polymer_conf,
        "polymer_details": detail,
    }


# =============================================================================
# Housekeeping homology (parity with WIQD)
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


# =============================================================================
# Rule features & editing (Nterm → C → P → M → Cterm)
# =============================================================================
@dataclass
class RuleParams:
    nterm_good: Tuple[str, ...]
    nterm_bad_strong: Tuple[str, ...]  # {K, P} as -2
    nterm_bad_weak: Tuple[str, ...]  # {S, T, V, A} as -1
    nterm_targets: Tuple[str, ...]
    bad_internal: Tuple[str, ...]  # e.g., {"Q"}
    hydrophobic_cterm_set: Tuple[str, ...]
    rule_flank_gap_min: int
    xle_collapse: bool


PROLINE_PRECEDER_TOP = {"A", "G"}
PROLINE_PRECEDER_GOOD = {"I", "L", "V", "F", "Y", "W", "M"}


def nterm_rule(a0: str, p: RuleParams) -> int:
    if a0 in p.nterm_good:
        return 1
    if a0 in p.nterm_bad_strong:
        return -2
    if a0 in p.nterm_bad_weak:
        return -1
    return 0


def proline_rule(seq: str) -> int:
    n = len(seq)
    idxs = [i for i, a in enumerate(seq) if a == "P" and i not in (0, n - 1)]
    if not idxs:
        return 0
    score = 1  # baseline for internal P
    good_pre = 0
    weak_pre = 0
    only_bad = True
    for i in idxs:
        prev = seq[i - 1]
        if prev in PROLINE_PRECEDER_TOP:
            good_pre += 1
            only_bad = False
        elif prev in PROLINE_PRECEDER_GOOD:
            weak_pre += 1
            only_bad = False
        elif prev not in ("K", "R"):
            only_bad = False
    if good_pre > 0:
        score += 3
    elif weak_pre > 0:
        score += 2
    if only_bad:
        score -= 1
    return score


def cterm_rule(seq: str, p: RuleParams) -> int:
    return 1 if seq and seq[-1] in p.hydrophobic_cterm_set else 0


def internal_residue_penalties(seq: str, p: RuleParams) -> int:
    return -1 if any((aa in p.bad_internal) for aa in seq[1:-1]) else 0


def bonus_internal_C_M(seq: str) -> int:
    s = 0
    if "C" in seq[1:-1]:
        s += 1
    if "M" in seq[1:-1]:
        s += 1
    return s


def bonus_spacing_CM_around_P(seq: str, min_gap: int) -> int:
    n = len(seq)
    idxP = [i for i, a in enumerate(seq) if a == "P" and i not in (0, n - 1)]
    if not idxP or "C" not in seq or "M" not in seq:
        return 0
    for i in idxP:
        idxC = [j for j, a in enumerate(seq) if a == "C"]
        idxM = [j for j, a in enumerate(seq) if a == "M"]
        if not idxC or not idxM:
            continue
        dC = min(abs(j - i) for j in idxC)
        dM = min(abs(j - i) for j in idxM)
        if dC >= min_gap and dM >= min_gap and dC != 1 and dM != 1:
            return 1
    return 0


def rule_score(seq: str, p: RuleParams) -> Dict[str, int]:
    s_nterm = nterm_rule(seq[0], p)
    s_pro = proline_rule(seq)
    s_cterm = cterm_rule(seq, p)
    s_bad = internal_residue_penalties(seq, p)
    s_spacing = bonus_spacing_CM_around_P(seq, p.rule_flank_gap_min)
    total = s_nterm + s_pro + s_cterm + s_bad + bonus_internal_C_M(seq) + s_spacing
    return {
        "rule_nterm": s_nterm,
        "rule_proline": s_pro,
        "rule_cterm": s_cterm,
        "rule_bad_internal": s_bad,
        "rule_internal_c": 1 if "C" in seq[1:-1] else 0,
        "rule_internal_m": 1 if "M" in seq[1:-1] else 0,
        "rule_spacing_cm_p": s_spacing,
        "rule_total": total,
    }


@dataclass
class EditParams:
    nterm_targets: Tuple[str, ...]
    hydrophobic_cterm_set: Tuple[str, ...]
    rule_flank_gap_min: int
    xle_collapse: bool


def choose_best_by_rule_then_tie(
    seq_candidates: List[str],
    rp: RuleParams,
    tie_scorer=None,
) -> str:
    if not seq_candidates:
        raise ValueError("empty candidate list")
    best = []
    best_rule = -1e9
    for s in seq_candidates:
        sc = rule_score(s, rp)["rule_total"]
        if sc > best_rule:
            best_rule = sc
            best = [s]
        elif sc == best_rule:
            best.append(s)
    if len(best) == 1 or tie_scorer is None:
        return best[0]
    tb = [(tie_scorer(s), s) for s in best]
    tb.sort(key=lambda x: x[0], reverse=True)
    return tb[0][1]


def edit_nterm(
    seq: str, rp: RuleParams, ep: EditParams, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    if seq[0] in rp.nterm_good:
        return seq, None
    cands = [seq] + [t + seq[1:] for t in ep.nterm_targets if t != seq[0]]
    best = choose_best_by_rule_then_tie(cands, rp, tie_scorer)
    return (best, f"Nterm:{seq[0]}→{best[0]}") if best != seq else (seq, None)


def edit_mutant_to_c(
    seq: str, mut_idx: Optional[int]
) -> Tuple[str, Optional[str], bool]:
    if mut_idx is None or mut_idx < 0 or mut_idx >= len(seq):
        return seq, None, False
    if seq[mut_idx] == "C":
        return seq, None, True
    new = seq[:mut_idx] + "C" + seq[mut_idx + 1 :]
    return new, f"C@{mut_idx+1}", True


def avoid_positions_near_P(seq: str, min_gap: int) -> set:
    idxP = [i for i, a in enumerate(seq) if a == "P"]
    avoid = set()
    for i in idxP:
        for d in range(-min_gap, min_gap + 1):
            avoid.add(i + d)
    return avoid


def best_single_sub_for_residue(
    seq: str,
    residue: str,
    rp: RuleParams,
    avoid_idx: Optional[set] = None,
    tie_scorer=None,
) -> Tuple[str, Optional[str]]:
    avoid = avoid_idx or set()
    cands = []
    for i in range(0, len(seq)):
        if i in avoid:
            continue
        if seq[i] == residue:
            continue
        cand = seq[:i] + residue + seq[i + 1 :]
        cands.append((cand, i))
    if not cands:
        return seq, None
    best = choose_best_by_rule_then_tie([c for c, _ in cands], rp, tie_scorer)
    if best != seq:
        pos = next(i for (c, i) in cands if c == best)
        return best, f"{residue}@{pos+1}"
    return seq, None


def ensure_internal_c(
    seq: str, rp: RuleParams, min_gap: int, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    if "C" in seq[1:-1]:
        return seq, None
    avoid = avoid_positions_near_P(seq, min_gap)
    idxP = [i for i, a in enumerate(seq) if a == "P"]
    bad_pre = {i - 1 for i in idxP if i - 1 >= 1}
    avoid.update(bad_pre)
    candidates_priority = []
    for i in range(1, len(seq) - 1):
        if i in avoid:
            continue
        a = seq[i]
        if a in ("P", "M"):
            continue
        prio = 2 if a in rp.bad_internal else 1
        candidates_priority.append((prio, i))
    if not candidates_priority:
        return seq, None
    for pr in (2, 1):
        idxs = [i for p, i in candidates_priority if p == pr]
        cands = [seq[:i] + "C" + seq[i + 1 :] for i in idxs]
        if not cands:
            continue
        best = choose_best_by_rule_then_tie(cands + [seq], rp, tie_scorer)
        if best != seq:
            pos = next(i for i in idxs if seq[:i] + "C" + seq[i + 1 :] == best)
            return best, f"C@{pos+1}"
    return seq, None


def ensure_c_anywhere_for_frameshift(
    seq: str, rp: RuleParams, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    """For frameshifts: allow C placement at ANY position (including terminals)."""
    if "C" in seq:
        return seq, None
    cands = []
    for i in range(0, len(seq)):
        if seq[i] == "C":
            continue
        cand = seq[:i] + "C" + seq[i + 1 :]
        cands.append((cand, i))
    if not cands:
        return seq, None
    best = choose_best_by_rule_then_tie([c for c, _ in cands], rp, tie_scorer)
    if best != seq:
        pos = next(i for (c, i) in cands if c == best)
        return best, f"C@{pos+1}"
    return seq, None


def edit_proline(
    seq: str, rp: RuleParams, ep: EditParams, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    if "P" in seq[1:-1]:
        return seq, None
    best, label = best_single_sub_for_residue(
        seq, "P", rp, avoid_idx=None, tie_scorer=tie_scorer
    )
    return best, (label or "P")


def edit_methionine(
    seq: str, rp: RuleParams, ep: EditParams, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    if "M" in seq[1:-1]:
        return seq, None
    avoid = avoid_positions_near_P(seq, ep.rule_flank_gap_min)
    best, label = best_single_sub_for_residue(
        seq, "M", rp, avoid_idx=avoid, tie_scorer=tie_scorer
    )
    return best, (label or "M")


def edit_cterm(
    seq: str, rp: RuleParams, ep: EditParams, tie_scorer=None
) -> Tuple[str, Optional[str]]:
    if seq[-1] in ep.hydrophobic_cterm_set:
        return seq, None
    best = None
    best_rule = rule_score(seq, rp)["rule_total"]
    best_label = None
    for aa in ep.hydrophobic_cterm_set:
        if aa == seq[-1]:
            continue
        cand = seq[:-1] + aa
        sc = rule_score(cand, rp)["rule_total"]
        if sc > best_rule or (
            sc == best_rule
            and tie_scorer
            and tie_scorer(cand) > (tie_scorer(best) if best else -1e9)
        ):
            best, best_rule, best_label = cand, sc, f"Cterm:{seq[-1]}→{aa}"
    if best is not None:
        return best, best_label
    # fallback forced hydrophobic switch
    for aa in ep.hydrophobic_cterm_set:
        if aa != seq[-1]:
            return seq[:-1] + aa, f"Cterm:{seq[-1]}→{aa}"
    return seq, None


def build_edit_path(
    seq: str,
    mut_idx0: Optional[int],
    rp: RuleParams,
    ep: EditParams,
    tie_scorer=None,
    is_frameshift: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Returns (stage_seqs, stage_labels) for stages:
      [orig, after Nterm, after C, after P, after M, after Cterm]
    """
    stage_labels: List[str] = []
    s0 = seq

    # Nterm
    sN, labN = edit_nterm(s0, rp, ep, tie_scorer)
    stage_labels.append(labN or "Nterm:keep")

    # C (frameshift: anywhere; otherwise mutant->C or ensure internal C)
    if is_frameshift:
        sC, labC = ensure_c_anywhere_for_frameshift(sN, rp, tie_scorer)
        stage_labels.append(labC or "C:keep(fs)")
    else:
        sC, labC, _applied_at_mut = edit_mutant_to_c(sN, mut_idx0)
        if not labC:
            sC2, labC2 = ensure_internal_c(sC, rp, ep.rule_flank_gap_min, tie_scorer)
            if labC2:
                sC, labC = sC2, labC2
        stage_labels.append(labC or "C:keep")

    # P
    sP, labP = edit_proline(sC, rp, ep, tie_scorer)
    stage_labels.append(labP or "P:keep")

    # M
    sM, labM = edit_methionine(sP, rp, ep, tie_scorer)
    stage_labels.append(labM or "M:keep")

    # Cterm
    sT, labT = edit_cterm(sM, rp, ep, tie_scorer)
    stage_labels.append(labT or "Cterm:keep")

    return [s0, sN, sC, sP, sM, sT], stage_labels


# =============================================================================
# Tie-breaker & final scoring
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


def aa_type_metric(seq: str) -> float:
    counts = {}
    for a in seq:
        counts[a] = counts.get(a, 0) + 1
    n = len(seq)
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log(p + 1e-12)
    Hmax = math.log(min(len(AA), n))
    return H / Hmax if Hmax > 0 else 0.0


def combine_tie_breaker(
    weights: Dict[str, float],
    hydro_norm: float,
    conf_norm: float,
    hk_norm: float,
    polymer_weight: float,
    aa_types_norm: float,
) -> float:
    """Continuous composite for ranking/tie-breaking. 'term' weight maps to HK k-mer norm."""
    fly = math.sqrt(max(hydro_norm, 0.0) ** 2 + max(conf_norm, 0.0) ** 2)
    return (
        weights.get("fly", 0.0) * fly
        + weights.get("hydro", 0.0) * hydro_norm
        + weights.get("conf_mz", 0.0) * conf_norm
        + weights.get("term", 0.0) * hk_norm
        + weights.get("polymer", 0.0)
        * (polymer_weight if np.isfinite(polymer_weight) else 0.0)
        + weights.get("types", 0.0) * aa_types_norm
    )


# =============================================================================
# Plotting helpers (classic edit-path + dense-aware)
# =============================================================================
STAGE_POINT_COLORS = {
    "orig": "#000000",
    "Nterm": "#6a3d9a",
    "C": "#1f77b4",
    "P": "#ff7f0e",
    "M": "#2ca02c",
    "Cterm": "#d62728",
}
STAGE_LIST = ["orig", "Nterm", "C", "P", "M", "Cterm"]


def _grid(ax):
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)


def truncate_seq(s: str, maxlen: int = 36) -> str:
    if len(s) <= maxlen:
        return s
    keep = max(6, (maxlen - 3) // 2)
    return f"{s[:keep]}…{s[-keep:]}"


def hex_or_scatter_background(ax, x, y, n_points: int, dense_threshold: int = 1500):
    if n_points > dense_threshold:
        hb = ax.hexbin(x, y, gridsize=60, bins="log", alpha=0.9)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
    else:
        ax.scatter(x, y, s=10, alpha=0.25, label=f"not selected (n={n_points})")


def classic_edit_path_plot(
    outdir: str,
    idx_or_rank: int,
    selected_flag: bool,
    seqs: List[str],
    xvals: List[float],  # hydro (fly surface norm)
    yvals: List[float],  # confusability score
    title_extra: str = "",
    max_label_len: int = 36,
):
    """Classic path: confusability vs hydrophobicity (fly surface norm) with stage colors."""
    ensure_outdir(outdir)
    fig, ax = plt.subplots()
    # Lines between stages
    for i in range(len(STAGE_LIST) - 1):
        ax.plot(
            [xvals[i], xvals[i + 1]],
            [yvals[i], yvals[i + 1]],
            lw=1.4,
            color="#777777",
            alpha=0.9,
        )
    # Points + labels
    for i, st in enumerate(STAGE_LIST):
        ax.scatter(
            [xvals[i]],
            [yvals[i]],
            s=42,
            color=STAGE_POINT_COLORS[st],
            edgecolor="k",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            st,
            (xvals[i], yvals[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=STAGE_POINT_COLORS[st],
        )
    # Titles
    ax.set_xlabel("fly_surface_norm (KD-normalized)")
    ax.set_ylabel("confusability score (multi-set)")
    seq0, seqF = seqs[0], seqs[-1]
    ttl1 = f"Edit path (orig → Nterm → C → P → M → Cterm) [{'SELECTED' if selected_flag else 'not selected'}]"
    ttl2 = f"orig: {truncate_seq(seq0)}  →  final: {truncate_seq(seqF)}"
    if title_extra:
        ttl2 += f"  |  {title_extra}"
    ax.set_title(ttl1 + "\n" + ttl2)
    _grid(ax)
    fig.tight_layout()
    fname = f"path_{idx_or_rank:04d}__{'SEL' if selected_flag else 'NOT'}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def density_scatter_sel_vs_not(
    df: pd.DataFrame,
    x: str,
    y: str,
    selcol: str,
    outpath: str,
    dense_threshold: int = 1500,
    title: Optional[str] = None,
):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots()
    msel = df[selcol] == True
    non = df.loc[~msel]
    sel = df.loc[msel]
    if len(non):
        hex_or_scatter_background(
            ax,
            non[x].to_numpy(float),
            non[y].to_numpy(float),
            len(non),
            dense_threshold,
        )
    if len(sel):
        ax.scatter(
            sel[x],
            sel[y],
            s=20,
            alpha=0.95,
            label=f"selected (n={len(sel)})",
            edgecolor="k",
            linewidths=0.4,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decoy generation with rule+continuous features, HK k-mer parity, frameshifts, and classic edit-path plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--in", dest="input_csv", required=True, help="Input CSV")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--peptide-col",
        default="peptide",
        help="Column containing the peptide sequence",
    )
    p.add_argument(
        "--mut-idx-col",
        default=None,
        help="Column with 0-based mutation index (preferred)",
    )
    p.add_argument(
        "--pos-col",
        default=None,
        help="Integer position column (will be converted by --indexing)",
    )
    p.add_argument(
        "--mut-notation-col",
        default=None,
        help="Mutation notation (e.g., L858R, p.G12D, p.G12fs*) — 'fs' triggers frameshift mode",
    )
    p.add_argument(
        "--indexing",
        choices=["0", "1", "C1"],
        default="C1",
        help="How to interpret positions",
    )
    p.add_argument(
        "--require-mut-idx0",
        type=int,
        default=1,
        help="1=fail when mut_idx0 missing/invalid (non-frameshifts only); 0=auto",
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
        help="PPM tolerance for general m/z matching",
    )
    p.add_argument(
        "--good-ppm-strict", type=float, default=10.0, help="PPM for strong matches"
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
    p.add_argument("--charges", default="2,3", help="Precursor charges")
    p.add_argument("--frag-charges", default="1,2,3", help="Fragment charges")
    p.add_argument(
        "--xle-collapse",
        action="store_true",
        help="Collapse I/L to J for legacy kmers (unused now)",
    )
    p.add_argument(
        "--strict-use-rt",
        action="store_true",
        help="When listing proteins, require RT-gated precursors",
    )
    p.add_argument(
        "--frag-types-req",
        type=int,
        default=3,
        help="Min distinct (ion,k,z) fragment types for strong match",
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

    # Rules (per spec)
    p.add_argument("--nterm-good", default="RYLFDM")
    p.add_argument("--nterm-bad-strong", default="KP")  # K/P = -2
    p.add_argument("--nterm-bad-weak", default="STVA")  # S/T/V/A = -1
    p.add_argument("--nterm-targets", default="RYLFDM")  # preferred N-term
    p.add_argument("--bad-internal", default="Q")
    p.add_argument("--rule-flank-gap-min", type=int, default=2)

    # Rankings / tie-break
    p.add_argument(
        "--rank-weights",
        default="fly:0.25,hydro:0.25,conf_mz:0.15,term:0.10,polymer:0.20,types:0.05",
    )
    p.add_argument("--overall-rule-weight", type=float, default=1.0)
    p.add_argument("--overall-cont-weight", type=float, default=1.0)

    # Selection
    p.add_argument("--N", type=int, default=48, help="Number of decoys to select")
    p.add_argument("--dedup-k", type=int, default=4, help="k-mer de-dup (0 disables)")
    p.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.10,
        help="Novelty bonus by category coverage",
    )
    p.add_argument(
        "--improvement-bonus",
        type=float,
        default=0.15,
        help="Bonus for normalized rank improvement",
    )
    p.add_argument(
        "--require-contam-match",
        type=int,
        default=1,
        help="Require ≥1 contaminant to select",
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

    # Percentile stratification (optional summary filters)
    p.add_argument(
        "--percentile-cut",
        type=float,
        default=25.0,
        help="Report thresholds used in summaries",
    )
    p.add_argument(
        "--select-metric",
        choices=["rule", "final"],
        default="final",
        help="Which metric to percentile-filter on when reporting",
    )

    # Housekeeping homology (parity with WIQD)
    p.add_argument(
        "--housekeeping-fasta",
        default=None,
        help="Path to housekeeping FASTA (optional)",
    )
    p.add_argument(
        "--download-housekeeping",
        action="store_true",
        help="Download housekeeping proteins from UniProt (requires internet)",
    )
    p.add_argument("--hk-k", type=int, default=4, help="k for housekeeping homology")
    p.add_argument(
        "--hk-collapse",
        choices=["none", "xle", "xle_de_qn", "xle+deqn"],
        default="xle",
        help="Collapsed alphabet for HK homology",
    )

    # Viz
    p.add_argument(
        "--dense-threshold",
        type=int,
        default=1500,
        help="Switch to hexbin when many non-selected",
    )
    p.add_argument("--trace-label-maxlen", type=int, default=36)

    # Verbosity
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


# =============================================================================
# Pipeline
# =============================================================================
def resolve_mut_idx_column(df: pd.DataFrame, args) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (mut_idx0:int series, is_frameshift:boolean series).

    Frameshift rows (mutation notation contains 'fs' case-insensitive) are
    *exempt* from --require-mut-idx0 and keep mut_idx0 as <NA>.
    """
    pepcol = args.peptide_col
    Ls = df[pepcol].astype(str).map(len)

    def parse_first_int(s: str) -> Optional[int]:
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return None
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else None

    # frameshift mask from mut-notation-col (Gene_AA_Change style)
    fs_mask = pd.Series(False, index=df.index)
    if args.mut_notation_col and args.mut_notation_col in df.columns:
        fs_mask = (
            df[args.mut_notation_col]
            .astype(str)
            .str.contains(r"\bfs\b", case=False, na=False)
        )

    # Primary: mut-idx-col (already 0-based)
    if args.mut_idx_col and args.mut_idx_col in df.columns:
        s = pd.to_numeric(df[args.mut_idx_col], errors="coerce").astype("Int64")
        mut0 = s.copy()
    else:
        mut0 = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # pos-col + indexing
    if mut0.isna().any() and args.pos_col and args.pos_col in df.columns:
        pos = pd.to_numeric(df[args.pos_col], errors="coerce").astype("Int64")
        if args.indexing in ("1", "C1"):
            pos = pos - 1
        mut0 = mut0.fillna(pos)

    # mut-notation-col
    if (
        mut0.isna().any()
        and args.mut_notation_col
        and args.mut_notation_col in df.columns
    ):
        extra = df[args.mut_notation_col].apply(parse_first_int).astype("Int64")
        if args.indexing in ("1", "C1"):
            extra = extra - 1
        mut0 = mut0.fillna(extra)

    # Enforce for non-frameshift only
    missing_mask = mut0.isna() & (~fs_mask)
    if missing_mask.any():
        if args.require_mut_idx0:
            bad = df.loc[
                missing_mask,
                [pepcol]
                + [
                    c
                    for c in [args.mut_idx_col, args.pos_col, args.mut_notation_col]
                    if c
                ],
            ]
            path = os.path.join(args.outdir, "missing_mut_idx0_rows.csv")
            ensure_outdir(args.outdir)
            bad.to_csv(path, index=False)
            raise ValueError(
                f"[mut_idx0] Required but missing/invalid for {int(missing_mask.sum())} non-frameshift rows. See {path}."
            )
        # auto-place near middle (avoid terminals)
        auto = (
            df[pepcol].astype(str).map(lambda s: max(1, min(len(s) - 2, len(s) // 2)))
        )
        mut0 = mut0.fillna(auto).astype("Int64")

    # Range check only for non-frameshifts
    out = []
    errors = []
    for i, (idx0, L, is_fs) in enumerate(
        zip(mut0.tolist(), Ls.tolist(), fs_mask.tolist())
    ):
        if is_fs:
            out.append(pd.NA)  # keep NA for fs rows
            continue
        if idx0 is None or (isinstance(idx0, float) and math.isnan(idx0)):
            errors.append((i, "NaN"))
            out.append(pd.NA)
            continue
        v = int(idx0)
        if v < 0 or v >= L:
            errors.append((i, f"out_of_range({v})_len({L})"))
        out.append(v)
    if errors:
        path = os.path.join(args.outdir, "invalid_mut_idx0_rows.csv")
        ensure_outdir(args.outdir)
        pd.DataFrame(
            [{"row": i, "reason": r, "peptide": df.iloc[i][pepcol]} for i, r in errors]
        ).to_csv(path, index=False)
        raise ValueError(f"[mut_idx0] Found {len(errors)} invalid indices; see {path}")
    return pd.Series(out, index=df.index, dtype="Int64"), fs_mask.astype(bool)


def main():
    args = make_parser().parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    ensure_outdir(args.outdir)
    ensure_outdir(os.path.join(args.outdir, "plots"))
    ensure_outdir(os.path.join(args.outdir, "plots", "edit_paths"))
    ensure_outdir(os.path.join(args.outdir, "plots", "seq_traces_selected"))
    ensure_outdir(os.path.join(args.outdir, "plots", "seq_traces_not_selected"))

    # Load input
    df_in = pd.read_csv(args.input_csv)
    if args.peptide_col not in df_in.columns:
        raise ValueError(
            f"Missing peptide column '{args.peptide_col}'. Available: {list(df_in.columns)[:12]}..."
        )
    df = df_in.copy()
    df[args.peptide_col] = df[args.peptide_col].map(clean_seq)
    df = df[df[args.peptide_col].str.len() > 0].copy()
    LOG.info(f"Loaded {len(df)} rows with valid peptides.")

    # Solve mut_idx0 (+ frameshift flag)
    df["mut_idx0"], df["is_frameshift"] = resolve_mut_idx_column(df, args)

    # Sets & windows
    set_names = [s.strip() for s in str(args.sets).split(",") if s.strip()]
    acc2seq_by_set = load_sets(
        set_names, args.sets_fasta_dir, bool(args.download_sets), args.download_timeout
    )

    charges_prec = sorted({int(z) for z in parse_csv_list(args.charges, int)})
    frag_charges = sorted({int(z) for z in parse_csv_list(args.frag_charges, int)})
    cys_fixed = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0

    windows_by_set: Dict[str, List[Window]] = {}
    LOG.info("Indexing windows + fragments for contaminant sets …")
    for s in tqdm(set_names, disable=bool(args.no_progress), desc="Sets"):
        wins = build_windows_index(
            s,
            acc2seq_by_set[s],
            args.full_mz_len_min,
            args.full_mz_len_max,
            charges_prec,
            args.gradient_min,
            cys_fixed,
        )
        index_fragments_for_windows(
            wins, args.fragment_kmin, args.fragment_kmax, frag_charges, cys_fixed
        )
        windows_by_set[s] = wins
        LOG.info(f"[windows] {s}: {len(wins)} windows indexed")

    # Housekeeping bank (parity with WIQD)
    hk_bank: Set[str] = set()
    if args.housekeeping_fasta or args.download_housekeeping:
        hk_seqs = load_housekeeping_sequences(
            args.housekeeping_fasta, args.download_housekeeping
        )
        if hk_seqs:
            hk_bank = build_hk_kmer_bank(
                hk_seqs, k=int(args.hk_k), collapse=str(args.hk_collapse)
            )
            LOG.info(
                f"[housekeeping] bank size={len(hk_bank)} (k={args.hk_k}, collapse={args.hk_collapse})"
            )
        else:
            LOG.info("[housekeeping] no sequences found; HK metric disabled")
    else:
        LOG.info("[housekeeping] skipped (no FASTA and no download flag)")

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
        zset=tuple(int(z) for z in str(args.polymer_z).split(",") if z.strip()),
        n_min=int(args.polymer_n_min),
        n_max=int(args.polymer_n_max),
        mz_min=float(args.polymer_mz_min),
        mz_max=float(args.polymer_mz_max),
    )
    LOG.info(f"[polymer] candidates: {len(polymer_cands)}")

    # Rule/Edit params
    rp = RuleParams(
        nterm_good=tuple(args.nterm_good),
        nterm_bad_strong=tuple(args.nterm_bad_strong),
        nterm_bad_weak=tuple(args.nterm_bad_weak),
        nterm_targets=tuple(args.nterm_targets),
        bad_internal=tuple(args.bad_internal),
        hydrophobic_cterm_set=tuple("FILVWYM"),
        rule_flank_gap_min=int(args.rule_flank_gap_min),
        xle_collapse=bool(args.xle_collapse),
    )
    ep = EditParams(
        nterm_targets=tuple(args.nterm_targets),
        hydrophobic_cterm_set=tuple("FILVWYM"),
        rule_flank_gap_min=int(args.rule_flank_gap_min),
        xle_collapse=bool(args.xle_collapse),
    )
    weights = parse_weight_map(args.rank_weights)

    # -------------------- Score originals + fit transforms --------------------
    LOG.info("Scoring originals (rule + continuous + HK homology) …")
    rows = []
    pbar = tqdm(range(len(df)), disable=bool(args.no_progress), desc="Originals")
    for i in pbar:
        seq = df.iloc[i][args.peptide_col]
        mut_idx0_val = df.iloc[i]["mut_idx0"]
        mut_idx0 = None if pd.isna(mut_idx0_val) else int(mut_idx0_val)
        r = rule_score(seq, rp)

        hydro_raw = kd_gravy(seq)
        fly = fly_surface_norm(seq)
        # Multi-set confusability
        rt_pred = predict_rt_min(seq, args.gradient_min)
        conf_per_set = {}
        bestppm_per_set = {}
        for s in set_names:
            wins = windows_by_set[s]
            if not wins:
                conf_per_set[s] = 0.0
                bestppm_per_set[s] = float("inf")
                continue
            d = confusability_script1(
                seq,
                wins,
                args.full_mz_len_min,
                args.full_mz_len_max,
                charges_prec,
                args.ppm_tol,
                rt_pred,
                args.rt_tolerance_min,
                cys_fixed,
                args.fragment_kmin,
                args.fragment_kmax,
                good_ppm=args.good_ppm_strict,
                frag_types_req=args.frag_types_req,
                frag_charges=frag_charges,
            )
            conf_per_set[s] = d["score"]
            bestppm_per_set[s] = d["best_ppm"]
        conf_mz_multi = max(conf_per_set.values()) if conf_per_set else 0.0

        # HK homology (WIQD parity)
        hk = hk_kmer_features(
            seq, hk_bank, k=int(args.hk_k), collapse=str(args.hk_collapse)
        )

        # PEG
        polymer = polymer_best_for_peptide(
            seq,
            polymer_cands,
            peptide_z=tuple(charges_prec),
            ppm_window=args.polymer_ppm_strict,
            cys_fixed_mod=cys_fixed,
        )

        # Diversity
        aa_types = aa_type_metric(seq)

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
                "aa_types": aa_types,
            }
        )
    base = pd.DataFrame(rows)

    # Robust transforms (fit on originals)
    t_hydro = RobustMinMax(base["kd_gravy"])
    t_conf = RobustMinMax(base["conf_mz_multi"])
    t_hk = RobustMinMax(base["hk_kmer_frac"])
    t_types = RobustMinMax(base["aa_types"])

    base["hydro_norm"] = t_hydro.transform(base["kd_gravy"])
    base["confusability_norm"] = t_conf.transform(base["conf_mz_multi"])
    base["hk_kmer_norm"] = t_hk.transform(base["hk_kmer_frac"])
    base["aa_types_norm"] = t_types.transform(base["aa_types"])

    # Tie-breaker and final composite
    base["tie_break_score"] = [
        combine_tie_breaker(
            weights=weights,
            hydro_norm=hn,
            conf_norm=cn,
            hk_norm=hkn,
            polymer_weight=0.0,  # polymer is re-evaluated per candidate decoy; keep 0 for originals
            aa_types_norm=an,
        )
        for hn, cn, hkn, an in zip(
            base["hydro_norm"],
            base["confusability_norm"],
            base["hk_kmer_norm"],
            base["aa_types_norm"],
        )
    ]
    base["final_rank_score"] = (
        args.overall_rule_weight * base["rule_total"]
        + args.overall_cont_weight * base["tie_break_score"]
    )

    out_all = pd.concat(
        [df.reset_index(drop=True), base.reset_index(drop=True)], axis=1
    )
    # Safety: drop duplicated columns if any (keep leftmost)
    if out_all.columns.duplicated().any():
        out_all = out_all.loc[:, ~out_all.columns.duplicated()]

    all_csv = os.path.join(args.outdir, "all_candidates_scored.csv")
    out_all.to_csv(all_csv, index=False)
    LOG.info(f"Wrote: {all_csv} (n={len(out_all)})")

    # Tie-scorer used during edits (fast; uses fit transforms)
    def tie_scorer_for_seq(s: str) -> float:
        h = t_hydro.transform_scalar(kd_gravy(s))
        rt_pred = predict_rt_min(s, args.gradient_min)
        confs = []
        for _set in set_names:
            wins = windows_by_set[_set]
            if not wins:
                confs.append(0.0)
                continue
            d = confusability_script1(
                s,
                wins,
                args.full_mz_len_min,
                args.full_mz_len_max,
                charges_prec,
                args.ppm_tol,
                rt_pred,
                args.rt_tolerance_min,
                cys_fixed,
                args.fragment_kmin,
                args.fragment_kmax,
                good_ppm=args.good_ppm_strict,
                frag_types_req=args.frag_types_req,
                frag_charges=frag_charges,
            )
            confs.append(d["score"])
        c = t_conf.transform_scalar(max(confs) if confs else 0.0)
        hkf = hk_kmer_features(
            s, hk_bank, k=int(args.hk_k), collapse=str(args.hk_collapse)
        )
        hk_norm = t_hk.transform_scalar(hkf["hk_kmer_frac"])
        fly = math.sqrt(h * h + c * c)
        return (
            weights.get("fly", 0.0) * fly
            + weights.get("hydro", 0.0) * h
            + weights.get("conf_mz", 0.0) * c
            + weights.get("term", 0.0) * hk_norm
        )

    # -------------------- Make a decoy for EVERY peptide ---------------------
    LOG.info(
        "Synthesizing one decoy per peptide via edit path: Nterm → C → P → M → Cterm …"
    )
    dec_rows = []
    pbar2 = tqdm(
        range(len(out_all)), disable=bool(args.no_progress), desc="Decoys(all)"
    )
    for i in pbar2:
        seq = out_all.loc[i, "seq"]
        # frameshift flag
        is_fs = (
            bool(out_all.loc[i, "is_frameshift"])
            if "is_frameshift" in out_all.columns
            else False
        )
        # mut index (None for frameshift or NA)
        raw_mut = out_all.loc[i, "mut_idx0"] if "mut_idx0" in out_all.columns else pd.NA
        mut_idx0 = None if (pd.isna(raw_mut) or is_fs) else int(raw_mut)

        stage_seqs, labels = build_edit_path(
            seq, mut_idx0, rp, ep, tie_scorer=tie_scorer_for_seq, is_frameshift=is_fs
        )
        s0, sN, sC, sP, sM, sT = stage_seqs
        final = sT

        # Original vs final: recompute components
        def _eval(s: str) -> Dict[str, float]:
            rt_pred = predict_rt_min(s, args.gradient_min)
            confs = []
            for _set in set_names:
                wins = windows_by_set[_set]
                if not wins:
                    confs.append(0.0)
                    continue
                d = confusability_script1(
                    s,
                    wins,
                    args.full_mz_len_min,
                    args.full_mz_len_max,
                    charges_prec,
                    args.ppm_tol,
                    rt_pred,
                    args.rt_tolerance_min,
                    cys_fixed,
                    args.fragment_kmin,
                    args.fragment_kmax,
                    args.good_ppm_strict,
                    args.frag_types_req,
                    frag_charges,
                )
                confs.append(d["score"])
            polymer = polymer_best_for_peptide(
                s,
                polymer_cands,
                peptide_z=tuple(charges_prec),
                ppm_window=args.polymer_ppm_strict,
                cys_fixed_mod=cys_fixed,
            )
            hk = hk_kmer_features(
                s, hk_bank, k=int(args.hk_k), collapse=str(args.hk_collapse)
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

        e0 = _eval(s0)
        e1 = _eval(sN)
        e2 = _eval(sC)
        e3 = _eval(sP)
        e4 = _eval(sM)
        e5 = _eval(sT)

        r_old = rule_score(s0, rp)
        r_new = rule_score(final, rp)

        # Normalize with original-fit scalers where applicable (for tie metrics)
        hydro_old_n = t_hydro.transform_scalar(kd_gravy(s0))
        hydro_new_n = t_hydro.transform_scalar(kd_gravy(final))
        conf_old_n = t_conf.transform_scalar(e0["conf"])
        conf_new_n = t_conf.transform_scalar(e5["conf"])
        hk_old_n = t_hk.transform_scalar(e0["hk_frac"])
        hk_new_n = t_hk.transform_scalar(e5["hk_frac"])
        types_old_n = t_types.transform_scalar(aa_type_metric(s0))
        types_new_n = t_types.transform_scalar(aa_type_metric(final))

        tie_old = combine_tie_breaker(
            weights, hydro_old_n, conf_old_n, hk_old_n, 0.0, types_old_n
        )
        tie_new = combine_tie_breaker(
            weights, hydro_new_n, conf_new_n, hk_new_n, 0.0, types_new_n
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
                "decoy_seq": final,
                "selected": False,
                "is_frameshift": bool(is_fs),
                "mut_idx0": (int(mut_idx0) if (mut_idx0 is not None) else -1),
                "edit_path": " → ".join([lab for lab in labels]),
                # stage values for classic plots
                "fly0": e0["fly"],
                "conf0": e0["conf"],
                "flyN": e1["fly"],
                "confN": e1["conf"],
                "flyC": e2["fly"],
                "confC": e2["conf"],
                "flyP": e3["fly"],
                "confP": e3["conf"],
                "flyM": e4["fly"],
                "confM": e4["conf"],
                "flyT": e5["fly"],
                "confT": e5["conf"],
                # totals
                "fly_surface_norm": e5["fly"],
                "hydrophobic_fraction": e5["hydro"],
                "conf_mz_multi": e5["conf"],
                "hk_kmer_frac": e5["hk_frac"],
                "polymer_conf": e5["polymer_conf"],
                "polymer_match_flag": int(e5["polymer_match_flag"]),
                "polymer_best_abs_ppm": e5["polymer_best_abs_ppm"],
                # rule components
                "rule_total_old": r_old["rule_total"],
                "rule_total_new": r_new["rule_total"],
                "rule_delta": r_new["rule_total"] - r_old["rule_total"],
                # score components
                "start_score": final_old,
                "final_score": final_new,
                "delta_total_score": final_new - final_old,
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

    # -------------------- Selection (diversity/improvement-aware) -------------
    N = int(args.N)
    kdedup = int(args.dedup_k)
    require_contam = bool(args.require_contam_match)
    enforce_cterm = bool(args.enforce_hydrophobic_cterm)
    selected_idx = []
    seen_kmers = set()
    covered_cats: Set[str] = set()
    remaining = set(df_dec.index.tolist())

    def categories_for_row(rr) -> Set[str]:
        cats = set()
        if float(rr["conf_mz_multi"]) >= 0.45:
            cats.add("protein-like")
        if int(rr["polymer_match_flag"]) == 1:
            cats.add("polymer-like")
        return cats

    while len(selected_idx) < N and remaining:
        scored = []
        for i in list(remaining):
            rr = df_dec.loc[i]
            # Hard constraints
            if require_contam and (float(rr["conf_mz_multi"]) < 0.01):
                scored.append((i, -1e9))
                continue
            if enforce_cterm and (rr["decoy_seq"][-1] not in "FILVWYM"):
                scored.append((i, -1e9))
                continue
            # conf ratio
            conf_ok = float(rr["conf_mz_multi"]) >= float(
                args.min_confusability_ratio
            ) * float(rr["conf0"])
            if not conf_ok:
                scored.append((i, -1e9))
                continue
            # hydro increase
            hydro_ok = float(rr["hydrophobic_fraction"]) >= hydrophobic_fraction(
                rr["peptide"]
            ) + float(args.hydro_delta_min)
            if args.require_hydro_increase and not hydro_ok:
                scored.append((i, -1e9))
                continue
            # k-mer dedup
            if kdedup > 0:
                kms = kmer_set(rr["decoy_seq"], kdedup)
                if kms & seen_kmers:
                    scored.append((i, -1e6))
                    continue
            # novelty + improvement bonus
            cats = categories_for_row(rr)
            novelty = len(cats - covered_cats) / 3.0
            aug = (
                float(rr["final_score"])
                + float(args.diversity_bonus) * novelty
                + float(args.improvement_bonus) * float(rr["rank_improvement_norm"])
            )
            scored.append((i, aug))
        if not scored:
            break
        scored.sort(key=lambda t: t[1], reverse=True)
        best_i, best_aug = scored[0]
        if best_aug < -1e8:
            break
        selected_idx.append(best_i)
        df_dec.at[best_i, "selected"] = True
        rr = df_dec.loc[best_i]
        if kdedup > 0:
            for km in kmer_set(rr["decoy_seq"], kdedup):
                seen_kmers.add(km)
        covered_cats |= categories_for_row(rr)
        remaining.remove(best_i)

    selected = (
        df_dec.loc[selected_idx].copy().sort_values("final_score", ascending=False)
    )
    df_dec.to_csv(dec_csv, index=False)  # re-write with selected flags

    sel_csv = os.path.join(args.outdir, "decoys_selected_topN.csv")
    selected.to_csv(sel_csv, index=False)
    LOG.info(f"Wrote: {sel_csv} (n={len(selected)})")

    # -------------------- Plots -----------------------------------------------
    plots_dir = os.path.join(args.outdir, "plots")
    ep_dir = os.path.join(plots_dir, "edit_paths")
    ensure_outdir(ep_dir)

    # selected first (rank order), then a subset of non-selected
    sel_sorted = selected.reset_index(drop=True)
    for rank, rr in enumerate(
        tqdm(
            sel_sorted.itertuples(index=False),
            total=len(sel_sorted),
            disable=bool(args.no_progress),
            desc="EditPaths(SEL)",
        )
    ):
        seqs = [
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.decoy_seq,
        ]  # labels show stages; seq text is illustrative
        xvals = [rr.fly0, rr.flyN, rr.flyC, rr.flyP, rr.flyM, rr.flyT]
        yvals = [rr.conf0, rr.confN, rr.confC, rr.confP, rr.confM, rr.confT]
        classic_edit_path_plot(
            ep_dir,
            rank + 1,
            True,
            seqs,
            xvals,
            yvals,
            title_extra=f"Δscore={rr.delta_total_score:.3f}",
            max_label_len=args.trace_label_maxlen,
        )

    non = df_dec.loc[df_dec["selected"] != True].reset_index(drop=True)
    for i, rr in enumerate(
        tqdm(
            non.itertuples(index=False),
            total=len(non),
            disable=bool(args.no_progress),
            desc="EditPaths(NOT)",
        )
    ):
        seqs = [
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.peptide,
            rr.decoy_seq,
        ]
        xvals = [rr.fly0, rr.flyN, rr.flyC, rr.flyP, rr.flyM, rr.flyT]
        yvals = [rr.conf0, rr.confN, rr.confC, rr.confP, rr.confM, rr.confT]
        classic_edit_path_plot(
            ep_dir,
            i + 1,
            False,
            seqs,
            xvals,
            yvals,
            title_extra=f"Δscore={rr.delta_total_score:.3f}",
            max_label_len=args.trace_label_maxlen,
        )

    # score decomposition for selected
    if len(selected):
        parts = [
            ("fly_surface_norm", weights.get("fly", 0.25)),
            ("hydrophobic_fraction", weights.get("hydro", 0.25)),
            ("conf_mz_multi", weights.get("conf_mz", 0.15)),
            ("hk_kmer_frac", weights.get("term", 0.10)),
            ("polymer_conf", weights.get("polymer", 0.20)),
        ]
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(selected)), 4.2))
        ssel = selected.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        x = np.arange(len(ssel))
        bottom = np.zeros(len(ssel), dtype=float)
        for col, wv in parts:
            contrib = float(wv) * ssel[col].fillna(0).to_numpy(float)
            ax.bar(x, contrib, bottom=bottom, label=f"{col}×{float(wv):.2f}")
            bottom += contrib
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_ylabel("Final score (weighted, illustrative)")
        ax.set_title("Final-score decomposition (selected)")
        ax.legend()
        _grid(ax)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, "selected_final_score_decomposition.png"), dpi=150
        )
        plt.close(fig)

    # density-aware scatter plots
    density_pairs = [
        (
            "fly_surface_norm",
            "conf_mz_multi",
            "scatter_fly_vs_conf.png",
            "confusability vs fly (KD)",
        ),
        ("final_score", "fly_surface_norm", "scatter_final_vs_fly.png", "final vs fly"),
        ("final_score", "conf_mz_multi", "scatter_final_vs_conf.png", "final vs conf"),
        (
            "final_score",
            "hk_kmer_frac",
            "scatter_final_vs_hk.png",
            "final vs housekeeping k-mer frac",
        ),
        (
            "final_score",
            "polymer_conf",
            "scatter_final_vs_polymer.png",
            "final vs polymer-like proximity",
        ),
    ]
    for x, y, f, t in density_pairs:
        density_scatter_sel_vs_not(
            df_dec,
            x,
            y,
            "selected",
            os.path.join(plots_dir, f),
            args.dense_threshold,
            title=t,
        )

    # Δ distributions
    def _hist(col: str, fn: str, title: str):
        if df_dec.empty or col not in df_dec.columns:
            return
        fig, ax = plt.subplots()
        ax.hist(df_dec[col].dropna().to_numpy(float), bins=30, alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)
        _grid(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fn), dpi=150)
        plt.close(fig)

    _hist(
        "delta_total_score",
        "hist_delta_total_score.png",
        "Distribution of final score improvements",
    )
    _hist("rule_delta", "hist_rule_delta.png", "Distribution of rule score changes")

    def _scatter_with_diag(xcol: str, ycol: str, fn: str, title: str):
        if df_dec.empty or (xcol not in df_dec.columns) or (ycol not in df_dec.columns):
            return
        fig, ax = plt.subplots()
        ax.scatter(df_dec[xcol], df_dec[ycol], s=14, alpha=0.6)
        v = np.r_[df_dec[xcol].to_numpy(float), df_dec[ycol].to_numpy(float)]
        mn = float(np.nanmin(v))
        mx = float(np.nanmax(v))
        ax.plot([mn, mx], [mn, mx], ls="--", lw=1, color="gray")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        _grid(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fn), dpi=150)
        plt.close(fig)

    _scatter_with_diag(
        "start_score",
        "final_score",
        "scatter_final_vs_start_score.png",
        "Final vs Start score",
    )
    _scatter_with_diag(
        "start_rank",
        "end_rank",
        "scatter_end_vs_start_rank.png",
        "End rank vs Start rank",
    )

    # README
    with open(os.path.join(args.outdir, "README.md"), "w") as fh:
        fh.write(
            "\n".join(
                [
                    "# Decoy generation with classic edit-path plots (HK/frameshift)",
                    "",
                    "## Sets",
                    f"Sets: {', '.join(set_names)}",
                    "No FASTA fallbacks; use --sets-fasta-dir or --download-sets 1",
                    "",
                    "## Housekeeping homology (WIQD parity)",
                    f"- hk_kmer_frac uses k={args.hk_k} and collapse={args.hk_collapse}.",
                    "- Term weight in rank-weights maps to HK k-mer norm.",
                    "",
                    "## Frameshifts",
                    "- 'fs' in mutation notation: no mut index required; allows placing 'C' at any position if none is present.",
                    "- Steps remain Nterm → C → P → M → Cterm, skipping already-good stages.",
                    "",
                    "## Scores",
                    "- Rule score (integer) and continuous tie-breakers (fly, conf, HK k-mer, polymer, types).",
                    "- Final score = rule_weight × rule_total + cont_weight × tie_break_score.",
                    "",
                    "## Selection",
                    f"- N={args.N}, k-de-dup k={args.dedup_k}, diversity_bonus={args.diversity_bonus}, improvement_bonus={args.improvement_bonus}.",
                    f"- Hard constraints: conf_new ≥ {args.min_confusability_ratio:.2f}×conf_old; "
                    f"{'require hydrophobic C-term; ' if bool(args.enforce_hydrophobic_cterm) else ''}"
                    f"{'hydro_new ≥ hydro_old + δ' if bool(args.require_hydro_increase) else 'no hydro increase required'}.",
                    "",
                    "## Plots",
                    "- Classic edit paths: plots/edit_paths/path_*.png",
                    "- Score decomposition, density scatters, Δ distributions, final vs start, rank plots.",
                ]
            )
        )

    LOG.info(
        f"[DONE] Selected {len(selected)} / {len(df_dec)} decoys. Outdir: {args.outdir}"
    )


if __name__ == "__main__":
    main()
