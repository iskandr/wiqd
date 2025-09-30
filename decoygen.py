#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: decoygen6.py
"""
Decoy generation + multi-source confusability (proteins + PEG) + diversity-aware selection
(With per-letter colored per-peptide traces, density-aware scatters, and improvement- & constraint-aware selection)

What’s new vs. decoygen5:
  • Per-letter colored traces:
      - Labeled trace PNG for **every peptide** (selected and not).
      - Each residue is colored by the **stage at which it changed**:
          orig (unchanged): black
          +C: blue
          +P: orange
          +M/final: green
        (If a predecessor edit is ever enabled again, it will be colored purple.)
      - Filenames include both the original and final sequences.
  • Density-aware scatter plots:
      - For large point clouds, non-selected are shown via hexbin density;
        selected are overlaid as higher-opacity points.
  • Improvement-aware selection:
      - Computes a "start score" from the original sequence using the same weights as the final score.
      - Selection augments the final score with an improvement bonus (configurable).
  • **Hard selection constraints**:
      - Selected decoys must be at least `--min-confusability-ratio` (default 0.90) times as
        contamination‑confusable as their original sequence (i.e., not >10% worse).
      - Selected decoys must be **more hydrophobic** than their original sequence
        by at least `--hydro-delta-min` (default 1e-6), when `--require-hydro-increase=1` (default).
  • **Policy change for P placement (your preference)**:
      - If the only possible P placement has a preceding K/R, the peptide is **dropped** (no hidden predecessor edit).
        Dropped peptides are listed in `dropped_rk_only.csv`.
  • Proton mass hygiene:
      - A single constant is used everywhere: PROTON = 1.007276466879.

New in this revision:
  • **Mandatory input filters (default ON):**
      1) Drop rows where the **reference AA at the mutation index** OR 'C' is anywhere in the mutant peptide.
      2) Drop peptides that contain 'P' anywhere.
      3) Drop peptides that contain 'M' anywhere.
     These occur **before** any C→P→M edits, so "avoid placing M where there's an M" is unnecessary.
  • **Peptide falloff diagnostics:**
      - `peptide_falloff_counts.csv` and `plots/peptide_falloff.png` show counts after each filter and after RK-only drops.
  • **Terminal k‑mer background fraction fix:**
      - `bg_term_kmer_frac` is now the **fraction of sets matched** at N and C (averaged), not {0,0.5,1}.

Example:
  python decoygen6.py \
    --in peptides.csv --peptide-col epitope --pos-col MT_pos --indexing C1 --require-mut-idx0 1 \
    --charges 2,3 --frag-charges 1 --frag-types-req 3 \
    --improvement-bonus 0.15 --min-confusability-ratio 0.90 --hydro-delta-min 1e-6

Outputs:
  - decoy_candidates_all.csv (all scored candidates; 'selected' marks top-N)
  - decoy_selected_topN.csv (final picks)
  - dropped_rk_only.csv (peptides skipped because P would be placed after K/R only)
  - dropped_ref_or_mut_C.csv, dropped_input_has_P.csv, dropped_input_has_M.csv (input filter diagnostics)
  - dedup_exclusions.csv (k-mer de-dup details)
  - stage_counts.csv (per-peptide candidate counts and filters)
  - peptide_falloff_counts.csv; plots/peptide_falloff.png (peptide count falloff across filters)
  - plots/*
      · selected_final_score_decomposition.png
      · density-aware scatter_* plots
      · improvement diagnostics (final vs start score/rank; Δ histograms/boxplots)
      · seq_traces_selected/*.png  (per-selected peptide labeled traces; filenames include sequences)
      · seq_traces_not_selected/*.png (per-not-selected peptide labeled traces)
      · core step plots and counts
"""

import os, sys, math, argparse, re, bisect
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Optional, Set

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------- Logging helpers --------------------------
def logx(msg: str):
    print(msg, flush=True)


# -------------------------- Chemistry constants ----------------------
AA = set("ACDEFGHIKLMNPQRSTVWY")
MONO = {
    "A": 71.037113805,
    "R": 156.10111105,
    "N": 114.04292747,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.05857754,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.09496305,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.07931298,
    "Y": 163.063328575,
    "V": 99.068413945,
}
H2O = 18.010564684
PROTON = 1.007276466879  # single source of truth (NIST/CODATA-level value)

KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
HYDRO_SET = set("AVILMFWYC")


def clean_pep(p):
    return "".join(ch for ch in str(p).strip().upper() if ch in AA)


def mass_neutral(seq: str, cys_fixed_mod: float = 0.0) -> float:
    return sum(MONO[a] for a in seq) + H2O + cys_fixed_mod * seq.count("C")


def mz_from_mass(m: float, z: int) -> float:
    return (m + z * PROTON) / z


def kd_gravy(seq: str) -> float:
    return sum(KD[a] for a in seq) / len(seq) if seq else float("nan")


def hydrophobic_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    return sum(1 for a in seq if a in HYDRO_SET) / len(seq)


def collapse_xle(s: str) -> str:
    # Collapse I/L into J for terminal k-mer comparisons
    return s.replace("I", "J").replace("L", "J")


# RT surrogate
def predict_rt_min(seq: str, gradient_min: float = 20.0) -> float:
    if not seq:
        return float("nan")
    gravy = kd_gravy(seq)
    frac = (gravy + 4.5) / 9.0
    base = 0.5 + (gradient_min - 1.0) * min(1.0, max(0.0, frac))
    length_adj = 0.03 * max(0, len(seq) - 8) * (gradient_min / 20.0)
    basic_adj = (
        -0.15
        * (seq.count("K") + seq.count("R") + 0.3 * seq.count("H"))
        * (gradient_min / 20.0)
    )
    return float(min(max(0.0, base + length_adj + basic_adj), gradient_min))


# -------------------------- Flyability (surface proxy) ----------------
KD_HYDRO = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}


def fly_surface_norm(seq: str) -> float:
    """Map KD hydropathy (mean) from [-4.5,4.5] → [0,1]."""
    if not seq:
        return 0.0
    gravy = sum(KD_HYDRO.get(a, 0.0) for a in seq) / len(seq)
    return float(min(1.0, max(0.0, (gravy + 4.5) / 9.0)))


# -------------------------- PEG families -----------------------------
FAMILIES = {
    "PEG": 44.026214747,  # C2H4O
    "PPG": 58.041865,  # C3H6O
    "PTMEG": 72.057515,  # poly-THF (C4H8O)
    "PDMS": 74.018792,  # (CH3)2SiO
}
ENDGROUPS_BY_FAMILY = {
    "PEG": {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PPG": {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PTMEG": {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PDMS": {"cyclic": 0.0},
}
ADDUCTS = {"H": PROTON, "Na": 22.989218, "K": 38.963158, "NH4": 18.033823}


def peptide_mz_poly(seq: str, z: int) -> float:
    return (mass_neutral(seq) + z * PROTON) / z


def polymer_mz(
    family: str, endgroup: str, n: int, adducts: Tuple[str, ...]
) -> Tuple[int, float]:
    z = len(adducts)
    mass = ENDGROUPS_BY_FAMILY[family][endgroup] + n * FAMILIES[family]
    return z, (mass + sum(ADDUCTS[a] for a in adducts)) / z


def enumerate_polymer_candidates(
    families=("PEG", "PPG", "PTMEG", "PDMS"),
    endgroups=("auto",),
    adducts=("H", "Na", "K", "NH4"),
    zset=(1, 2, 3),
    n_min=5,
    n_max=100,
    mz_min=200.0,
    mz_max=1500.0,
) -> List[Tuple[str, str, int, Tuple[str, ...], int, float]]:
    from itertools import combinations_with_replacement

    def adduct_multisets(z: int):
        labs = [a for a in adducts if a in ADDUCTS]
        yield from combinations_with_replacement(labs, z)

    out = []
    for fam in families:
        egs = (
            tuple(ENDGROUPS_BY_FAMILY[fam].keys())
            if ("auto" in endgroups)
            else tuple([e for e in endgroups if e in ENDGROUPS_BY_FAMILY[fam]])
        )
        for end in egs:
            for n in range(n_min, n_max + 1):
                for z in zset:
                    for ads in adduct_multisets(z):
                        _z, mzv = polymer_mz(fam, end, n, ads)
                        if mz_min <= mzv <= mz_max:
                            out.append((fam, end, n, ads, _z, mzv))
    return out


def peg_best_for_peptide(
    seq: str,
    candidates: List[Tuple[str, str, int, Tuple[str, ...], int, float]],
    peptide_z=(2, 3),
    ppm_window=10.0,
) -> Dict[str, object]:
    if not candidates:
        return {
            "peg_match_flag": 0,
            "peg_best_abs_ppm": float("inf"),
            "peg_details": "",
            "peg_conf": 0.0,
        }
    arr = np.array([c[5] for c in candidates], dtype=float)
    rows = []
    for z in peptide_z:
        pep_mz = peptide_mz_poly(seq, z)
        idx = np.argsort(np.abs(arr - pep_mz))[:200]
        for j in idx:
            fam, end, n, ads, _z, mz_poly = candidates[j]
            d_ppm = (mz_poly - pep_mz) / pep_mz * 1e6
            rows.append((abs(d_ppm), fam, end, n, "+".join(ads), _z, mz_poly, z))
    if not rows:
        return {
            "peg_match_flag": 0,
            "peg_best_abs_ppm": float("inf"),
            "peg_details": "",
            "peg_conf": 0.0,
        }
    rows.sort(key=lambda t: t[0])
    absppm, fam, bend, n, ads, pegz, pegmz, pepz = rows[0]
    peg_conf = max(0.0, 1.0 - min(absppm, ppm_window) / ppm_window)
    detail = f"{fam};end={bend};n={n};adducts={ads};peg_z={pegz};pep_z={pepz};peg_mz={pegmz:.6f}"
    return {
        "peg_match_flag": 1 if absppm <= ppm_window else 0,
        "peg_best_abs_ppm": float(absppm),
        "peg_conf": peg_conf,
        "peg_details": detail,
    }


# -------------------------- Contamination sets -----------------------
SET_FRIENDLY = {
    "albumin": "Albumin",
    "keratins": "Keratins",
    "proteases": "Proteases (autolysis)",
    "mhc_hardware": "MHC hardware (B2M/HLA)",
}
ALBUMIN_P02768_MATURE = (
    "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFA"
    "KTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMC"
    "TAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRL"
    "KCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSIS"
    "SKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLL"
    "LRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVST"
    "PTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVD"
    "ETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCF"
    "AEEGKKLVAASQAALGL"
)
FALLBACK_FASTAS: Dict[str, Dict[str, str]] = {
    "albumin": {"P02768": ALBUMIN_P02768_MATURE},
    "keratins": {"P04264": "MTSYSYRQSSSKSSSSGSSRSGGGGGGYGGGGGAGGYGGQGSSSSS"},
    "proteases": {"P00760": "IVGGYTCGANTVPYQVSLNSGYHFCGAGKTKDSGGP"},
    "mhc_hardware": {"P61769": "MRVTAPRTVLLLAVLAVVHLVHSQSRPHSRPEDFFF"},
}


def parse_fasta(text: str) -> Dict[str, str]:
    seqs = {}
    hdr = None
    buf = []
    for ln in text.splitlines():
        if not ln:
            continue
        if ln[0] == ">":
            if hdr:
                seqs[hdr] = "".join(buf).replace(" ", "").upper()
            hdr = ln[1:].strip()
            buf = []
        else:
            buf.append(ln.strip())
    if hdr:
        seqs[hdr] = "".join(buf).replace(" ", "").upper()
    return seqs


def extract_accession(hdr: str) -> str:
    m = re.match(r"^\w+\|([^|]+)\|", hdr)
    if m:
        return m.group(1)
    tok = hdr.split()[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tok)


def fetch_uniprot(accs: List[str], timeout=45.0) -> Dict[str, str]:
    try:
        import requests

        url = "https://rest.uniprot.org/uniprotkb/stream"
        q = " OR ".join(f"accession:{a}" for a in accs)
        params = {"query": q, "format": "fasta", "includeIsoform": "false"}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        hdr2seq = parse_fasta(r.text)
        out = {}
        for hdr, seq in hdr2seq.items():
            out[extract_accession(hdr)] = seq
        return out
    except Exception as e:
        logx(f"[sets][WARN] UniProt fetch failed ({e}); using fallbacks if present.")
        return {}


def load_set_sequences(set_name: str, args) -> Dict[str, str]:
    # Priority: local FASTA → fetch (if enabled) → fallback embedded
    if args.sets_fasta_dir:
        fpath = os.path.join(args.sets_fasta_dir, f"{set_name}.fasta")
        if os.path.isfile(fpath):
            with open(fpath, "r") as fh:
                hdr2seq = parse_fasta(fh.read())
            out = {}
            for hdr, seq in hdr2seq.items():
                out[extract_accession(hdr)] = seq
            if out:
                logx(f"[sets] {set_name}: loaded {len(out)} sequences from {fpath}")
                return out
    acc_map = {
        "albumin": ["P02768"],
        "keratins": [
            "P04264",
            "P35908",
            "P13645",
            "P05787",
            "P05783",
            "P02533",
            "P08727",
        ],
        "proteases": ["P00760", "Q7M135"],
        "mhc_hardware": ["P61769", "P04439", "P01889", "P10321"],
    }
    if args.download_sets:
        fetched = fetch_uniprot(acc_map.get(set_name, []))
        if fetched:
            logx(f"[sets] {set_name}: fetched {len(fetched)} sequences from UniProt")
            return fetched
    fb = FALLBACK_FASTAS.get(set_name, {})
    if fb:
        logx(f"[sets] {set_name}: using fallback embedded sequences (n={len(fb)})")
    else:
        logx(
            f"[sets][WARN] {set_name}: no data (disable/replace this set or provide FASTA)."
        )
    return fb


# -------------------------- Windows & fragments ----------------------
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
    zs = sorted(set(int(z) for z in charges_precursor))
    out = []
    for acc, s in acc2seq.items():
        clean = "".join(ch for ch in s if ch.isalpha()).upper()
        N = len(clean)
        for L in range(max(1, len_min), min(N, len_max) + 1):
            for i in range(0, N - L + 1):
                sub = clean[i : i + L]
                if not set(sub) <= AA:
                    continue
                m = mass_neutral(sub, cys_fixed_mod)
                mz_by = {z: mz_from_mass(m, z) for z in zs}
                rt = predict_rt_min(sub, gradient_min)
                out.append(
                    Window(
                        seq=sub,
                        acc=acc,
                        set_name=set_name,
                        length=L,
                        mass=m,
                        mz_by_z=mz_by,
                        rt_min=rt,
                    )
                )
    return out


def index_fragments_for_windows(
    windows: List[Window],
    fragment_kmin: int,
    fragment_kmax: int,
    frag_charges: Iterable[int],
    cys_fixed_mod: float,
):
    frag_zs = sorted(set(int(z) for z in frag_charges))
    for w in windows:
        arr = []
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


def precursor_matched_idxs(
    windows: List[Window], pep_mz_by_z: Dict[int, float], ppm_tol: float
) -> List[int]:
    out = []
    for i, w in enumerate(windows):
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


def filter_rt_idxs(
    windows: List[Window], idxs: List[int], rt: float, tol_min: float
) -> List[int]:
    lo, hi = rt - tol_min, rt + tol_min
    return [i for i in idxs if (windows[i].rt_min >= lo and windows[i].rt_min <= hi)]


def pep_by_frag_mzs(
    pep: str, frag_charges: Iterable[int], cys_fixed_mod: float, kmin: int, kmax: int
) -> List[Tuple[float, str, int, int]]:
    """Return list of (mz, ion, k, z) for peptide b/y fragments with given fragment charge states."""
    L = len(pep)
    out = []
    if L <= 1:
        return out
    kmax_eff = min(kmax, L - 1)
    kmin_eff = max(1, min(kmin, kmax_eff))
    frag_zs = sorted(set(int(z) for z in frag_charges))
    for k in range(kmin_eff, kmax_eff + 1):
        bseq = pep[:k]
        yseq = pep[-k:]
        for z in frag_zs:
            bmz = (mass_neutral(bseq, cys_fixed_mod) - H2O + z * PROTON) / z
            ymz = (mass_neutral(yseq, cys_fixed_mod) + z * PROTON) / z
            out.append((bmz, "b", k, z))
            out.append((ymz, "y", k, z))
    return out


def count_fragment_types_over_idxs(
    idxs: List[int],
    windows: List[Window],
    pep_frag_mzs: List[Tuple[float, str, int, int]],
    ppm_tol: float,
) -> int:
    seen = set()
    for mz, ion, k, z in pep_frag_mzs:
        lo = mz * (1 - ppm_tol * 1e-6)
        hi = mz * (1 + ppm_tol * 1e-6)
        hit = False
        for i in idxs:
            arr = windows[i].frag_mz_sorted
            if not arr:
                continue
            L = bisect.bisect_left(arr, lo)
            R = bisect.bisect_right(arr, hi)
            if R > L:
                hit = True
                break
        if hit:
            seen.add((ion, k, z))
    return len(seen)


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
    prec_zs = sorted(set(int(z) for z in precursor_charges))
    pep_mz_by_z = {z: mz_from_mass(pep_m, z) for z in prec_zs}
    idx_pool = [
        i for i, w in enumerate(windows) if (full_len_min <= w.length <= full_len_max)
    ]
    idx_prec = precursor_matched_idxs(
        [windows[i] for i in idx_pool], pep_mz_by_z, ppm_tol
    )
    prec_idxs = [idx_pool[i] for i in idx_prec]
    idx_prec_rt = filter_rt_idxs(windows, prec_idxs, rt_pred_min, rt_tol_min)

    def best_ppm(idxs):
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

    best_all = best_ppm(idx_pool) if idx_pool else float("inf")
    best_rt = best_ppm(idx_prec_rt) if idx_prec_rt else float("inf")

    frag_zs = list(frag_charges) if frag_charges is not None else prec_zs
    fr = pep_by_frag_mzs(pep, frag_zs, cys_fixed_mod, kmin, kmax)
    n_types = max(1, len(fr))
    ft_all = count_fragment_types_over_idxs(prec_idxs, windows, fr, ppm_tol)
    ft_rt = count_fragment_types_over_idxs(idx_prec_rt, windows, fr, ppm_tol)
    frac_all = ft_all / n_types
    frac_rt = ft_rt / n_types

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


# -------------------------- Terminal k-mer bank ----------------------
def build_kmer_banks(
    acc2seq_map: Dict[str, Dict[str, str]],
    k_term: int,
    collapse_xle_flag: bool = True,
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, Set[str]]]]:
    """
    Build an all‑k‑mer bank per set (not only terminals). Query uses terminal N/C k-mers of the peptide.
    """
    kmer_to_accs_by_set = {}
    acc_to_kmers_by_set = {}
    for s, acc2seq in acc2seq_map.items():
        kt = {}
        at = {}
        for acc, seq in acc2seq.items():
            seqc = collapse_xle(seq) if collapse_xle_flag else seq
            kmers = set()
            for i in range(0, len(seqc) - k_term + 1):
                kmers.add(seqc[i : i + k_term])
            at[acc] = kmers
            for km in kmers:
                kt.setdefault(km, set()).add(acc)
        kmer_to_accs_by_set[s] = kt
        acc_to_kmers_by_set[s] = at
    return kmer_to_accs_by_set, acc_to_kmers_by_set


def term_kmer_matches(
    pep: str,
    k_term: int,
    kmer_to_accs_by_set: Dict[str, Dict[str, Set[str]]],
    collapse_xle_flag: bool = True,
) -> Dict[str, object]:
    """Return terminal-kmer presence across sets and fraction-of-sets metric."""
    if len(pep) < k_term:
        return {"bg_term_kmer_frac": 0.0, "matches_by_set": {}, "accessions_by_set": {}}
    N = pep[:k_term]
    C = pep[-k_term:]
    Nc = collapse_xle(N) if collapse_xle_flag else N
    Cc = collapse_xle(C) if collapse_xle_flag else C
    matches_by_set = {}
    accs_by_set = {}
    present_N = present_C = 0
    total_sets = max(1, len(kmer_to_accs_by_set))
    for s, bank in kmer_to_accs_by_set.items():
        hasN = Nc in bank
        hasC = Cc in bank
        matches_by_set[s] = {"N": int(hasN), "C": int(hasC)}
        accs = set()
        if hasN:
            accs |= bank[Nc]
        if hasC:
            accs |= bank[Cc]
        accs_by_set[s] = sorted(list(accs)) if accs else []
        if hasN:
            present_N += 1
        if hasC:
            present_C += 1
    # Average the fraction-of-sets matched at N and C
    fracN = present_N / total_sets
    fracC = present_C / total_sets
    frac = 0.5 * (fracN + fracC)
    return {
        "bg_term_kmer_frac": frac,
        "matches_by_set": matches_by_set,
        "accessions_by_set": accs_by_set,
    }


# -------------------------- Decoy generation (heuristic engine) ------
P_PREF_ORDER = ["G", "A", "L", "V", "I", "S", "T"]  # X in X–P preference (avoid R/K)
BASIC = set("KRH")


def _p_tier(x_before: str) -> Tuple[str, int]:
    if x_before == "G":
        return "GP", 0
    if x_before == "A":
        return "AP", 1
    if x_before in {"L", "V", "I", "S", "T"}:
        return "LVISTP", 2
    return "other", 3


def _eval_seq(
    seq: str,
    sets: List[str],
    windows_by_set: Dict[str, List[Window]],
    charges_prec: List[int],
    frag_charges: List[int],
    args,
    cys_fixed: float,
) -> Dict[str, object]:
    fly = fly_surface_norm(seq)
    hydro = hydrophobic_fraction(seq)
    rt_pred = predict_rt_min(seq, args.gradient_min)
    conf_per_set = {}
    bestppm_per_set = {}
    for s in sets:
        wins = windows_by_set.get(s, [])
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
    return {
        "fly": fly,
        "hydro": hydro,
        "rt": rt_pred,
        "conf_mz_multi": conf_mz_multi,
        "conf_per_set": conf_per_set,
        "bestppm_per_set": bestppm_per_set,
    }


def generate_decoy_best(
    seq: str,
    mut_idx0: int,
    sets: List[str],
    windows_by_set: Dict[str, List[Window]],
    charges_prec: List[int],
    frag_charges: List[int],
    args,
    cys_fixed: float,
) -> Dict[str, object]:
    """
    P→M heuristic constructor:
      1) C at mutation (not last).
      2) P first with tiers, hard-avoid RK-before-P; prefer Q→P; NOT adjacent to C; tie-break ↑hydro→↑fly→↑conf.
         If *all* P candidates would have RK immediately before P, **drop the peptide**.
      3) M next with Q→M preference; ≥1 away from C; enforce P between C and M unless C@0; tie-break ↑hydro→↑fly→↑conf→↑spacing.
    """
    L = len(seq)
    orig = list(seq)
    reasons = []
    eval0 = _eval_seq(
        seq, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
    )

    # Stage 1: Cys at mutation
    c_idx = int(mut_idx0)
    if c_idx >= L:
        c_idx = max(0, L - 2)
        reasons.append(f"mut_idx out of range adjusted→{c_idx}")
    if c_idx == L - 1:
        c_idx = L - 2
        reasons.append("mut_idx at last adjusted to L-2")
    s1 = orig.copy()
    if s1[c_idx] != "C":
        s1[c_idx] = "C"
        reasons.append(f"C@{c_idx}")
    seq1 = "".join(s1)
    eval1 = _eval_seq(
        seq1, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
    )

    # Stage 2: choose P (not 0/last; NOT adjacent to C; and hard-avoid RK-before-P)
    p_cands_raw = []
    for p_idx in range(1, L - 1):
        if abs(p_idx - c_idx) <= 1:  # not adjacent to C
            continue
        xb = s1[p_idx - 1]
        tier_name, tier_rank = _p_tier(xb)
        rk_before = 1 if xb in {"R", "K"} else 0
        s2 = s1.copy()
        s2[p_idx] = "P"
        seq2 = "".join(s2)
        ev2 = _eval_seq(
            seq2, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
        )
        q_to_p = 1 if orig[p_idx] == "Q" else 0
        p_cands_raw.append(
            {
                "p_idx": p_idx,
                "tier_name": tier_name,
                "tier_rank": tier_rank,
                "rk_before": rk_before,
                "q_to_p": q_to_p,
                "seq2": seq2,
                "eval2": ev2,
            }
        )
    any_non_rk = any(c["rk_before"] == 0 for c in p_cands_raw)
    if not any_non_rk:
        # Preferred policy: drop peptides where P can only be placed after K/R
        return {
            "drop_rk_only": 1,
            "drop_reason": "P placement only possible after K/R",
            "seq0": seq,
        }
    p_cands = [c for c in p_cands_raw if c["rk_before"] == 0]
    p_cands.sort(
        key=lambda c: (
            c["tier_rank"],
            -c["q_to_p"],
            -c["eval2"]["hydro"],
            -c["eval2"]["fly"],
            -c["eval2"]["conf_mz_multi"],
        )
    )
    bestP = p_cands[0]
    p_idx = bestP["p_idx"]
    seq2 = bestP["seq2"]
    eval2 = bestP["eval2"]
    reasons.append(
        f"P@{p_idx} (tier {bestP['tier_name']}, {'Q→P' if bestP['q_to_p'] else 'no Q→P'})"
    )

    # Stage 3: choose M (≥1 away from C; not last). Enforce P between C and M unless C at 0.
    banned = {L - 1, c_idx, p_idx, c_idx - 1, c_idx + 1}
    m_cands = []
    for i in range(0, L):
        if i in banned or i < 0 or i >= L:
            continue
        if c_idx != 0:
            if not ((i < p_idx < c_idx) or (c_idx < p_idx < i)):
                continue
        s3 = list(seq2)
        s3[i] = "M"
        seq3 = "".join(s3)
        ev3 = _eval_seq(
            seq3, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
        )
        q_to_m = 1 if orig[i] == "Q" else 0
        spacing = min(abs(i - c_idx), abs(i - p_idx))
        m_cands.append(
            {
                "m_idx": i,
                "seq3": seq3,
                "eval3": ev3,
                "q_to_m": q_to_m,
                "spacing": spacing,
            }
        )
    if not m_cands:
        for i in range(0, L - 1):
            if i in {c_idx, p_idx}:
                continue
            if c_idx != 0 and not ((i < p_idx < c_idx) or (c_idx < p_idx < i)):
                continue
            s3 = list(seq2)
            s3[i] = "M"
            seq3 = "".join(s3)
            ev3 = _eval_seq(
                seq3, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
            )
            q_to_m = 1 if orig[i] == "Q" else 0
            spacing = min(abs(i - c_idx), abs(i - p_idx))
            m_cands.append(
                {
                    "m_idx": i,
                    "seq3": seq3,
                    "eval3": ev3,
                    "q_to_m": q_to_m,
                    "spacing": spacing,
                }
            )
        if not m_cands:
            i = 1 if p_idx != 1 and c_idx != 1 else 2
            i = min(i, L - 2)
            s3 = list(seq2)
            s3[i] = "M"
            seq3 = "".join(s3)
            ev3 = _eval_seq(
                seq3, sets, windows_by_set, charges_prec, frag_charges, args, cys_fixed
            )
            m_cands.append(
                {
                    "m_idx": i,
                    "seq3": seq3,
                    "eval3": ev3,
                    "q_to_m": int(orig[i] == "Q"),
                    "spacing": min(abs(i - c_idx), abs(i - p_idx)),
                }
            )
            reasons.append("NOTE: relaxed M placement due to constraints")
    m_cands.sort(
        key=lambda c: (
            -c["q_to_m"],
            -c["eval3"]["hydro"],
            -c["eval3"]["fly"],
            -c["eval3"]["conf_mz_multi"],
            -c["spacing"],
        )
    )
    bestM = m_cands[0]
    m_idx = bestM["m_idx"]
    seq3 = bestM["seq3"]
    eval3 = bestM["eval3"]
    reasons.append(
        f"M@{m_idx} ({'Q→M' if bestM['q_to_m'] else 'no Q→M'}, spacing={bestM['spacing']})"
    )

    # Deltas for plots
    deltaP_h = eval2["hydro"] - eval1["hydro"]
    deltaP_f = eval2["fly"] - eval1["fly"]
    deltaM_h = eval3["hydro"] - eval2["hydro"]
    deltaM_f = eval3["fly"] - eval2["fly"]

    return {
        "drop_rk_only": 0,
        "seq0": seq,
        "seq1": seq1,
        "seq2": seq2,
        "seq3": seq3,
        "decoy_seq": seq3,
        "c_idx": c_idx,
        "p_idx": p_idx,
        "m_idx": m_idx,
        "reasons": ";".join(reasons),
        "p_tier": bestP["tier_name"],
        "p_rk_before": 0,  # guaranteed non-RK due to policy
        "p_candidates_total": len(p_cands_raw),
        "p_candidates_nonRK": int(sum(1 for c in p_cands_raw if c["rk_before"] == 0)),
        "m_candidates_total": len(m_cands),
        "dist_p_to_c": abs(p_idx - c_idx),
        "dist_m_to_p": abs(m_idx - p_idx),
        "eval0": eval0,
        "eval1": eval1,
        "eval2": eval2,
        "eval3": eval3,
        "deltaP_hydro": deltaP_h,
        "deltaP_fly": deltaP_f,
        "deltaM_hydro": deltaM_h,
        "deltaM_fly": deltaM_f,
        # placeholder if you ever re-enable pre-P predecessor edit
        "preP_idx": None,
    }


# -------------------------- Selection utilities ----------------------
def kmer_set(s: str, k: int) -> Set[str]:
    if k <= 0 or len(s) < k:
        return set()
    return {s[i : i + k] for i in range(0, len(s) - k + 1)}


# -------------------------- Plot helpers -----------------------------
STAGE_COLORS = {"orig": "#000000", "C": "#1f77b4", "P": "#ff7f0e", "M": "#2ca02c"}
PREP_FIX_COLOR = "#9467bd"  # purple if predecessor edit is ever used


def _grid(ax):
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)


def truncate_seq(s: str, maxlen: int = 36) -> str:
    if len(s) <= maxlen:
        return s
    keep = max(6, (maxlen - 3) // 2)
    return f"{s[:keep]}...{s[-keep:]}"


def slugify_seq(s: str, maxlen: int = 40) -> str:
    if len(s) <= maxlen:
        return s
    return f"{s[:maxlen-8]}..{s[-6:]}"


def scatter_sel_vs_not(
    df: pd.DataFrame,
    x: str,
    y: str,
    selcol: str,
    outpath: str,
    title: str = None,
    dense_threshold: int = 1500,
    gridsize: int = 50,
):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots()
    msel = df[selcol] == True
    n_not = int((~msel).sum())
    n_sel = int(msel.sum())

    if n_not > dense_threshold:
        hb = ax.hexbin(
            df.loc[~msel, x], df.loc[~msel, y], gridsize=gridsize, bins="log", alpha=0.9
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
    else:
        if n_not > 0:
            ax.scatter(
                df.loc[~msel, x],
                df.loc[~msel, y],
                s=10,
                alpha=0.25,
                label=f"not selected (n={n_not})",
            )

    if n_sel > 0:
        ax.scatter(
            df.loc[msel, x],
            df.loc[msel, y],
            s=18,
            alpha=0.9,
            label=f"selected (n={n_sel})",
            edgecolor="k",
            linewidths=0.3,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def scatter_with_diag(
    df: pd.DataFrame,
    x: str,
    y: str,
    selcol: str,
    outpath: str,
    title: str = None,
    dense_threshold: int = 1500,
):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots()
    msel = df[selcol] == True
    n_not = int((~msel).sum())
    n_sel = int(msel.sum())
    if n_not > dense_threshold:
        hb = ax.hexbin(
            df.loc[~msel, x], df.loc[~msel, y], gridsize=50, bins="log", alpha=0.9
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")
    else:
        if n_not > 0:
            ax.scatter(
                df.loc[~msel, x],
                df.loc[~msel, y],
                s=10,
                alpha=0.25,
                label=f"not selected (n={n_not})",
            )
    if n_sel > 0:
        ax.scatter(
            df.loc[msel, x],
            df.loc[msel, y],
            s=18,
            alpha=0.9,
            label=f"selected (n={n_sel})",
            edgecolor="k",
            linewidths=0.3,
        )
    if n_not + n_sel > 0:
        vx = df[x].astype(float)
        vy = df[y].astype(float)
        mn = float(np.nanmin([vx.min(), vy.min()]))
        mx = float(np.nanmax([vx.max(), vy.max()]))
        ax.plot([mn, mx], [mn, mx], ls="--", lw=1.0, color="gray")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_hist_selected_vs_not(
    df: pd.DataFrame,
    col: str,
    selcol: str,
    outpath: str,
    bins: int = 30,
    title: Optional[str] = None,
):
    if df.empty or col not in df.columns:
        return
    fig, ax = plt.subplots()
    msel = df[selcol] == True
    if int((~msel).sum()) > 0:
        ax.hist(df.loc[~msel, col].dropna(), bins=bins, alpha=0.4, label="not selected")
    if int(msel.sum()) > 0:
        ax.hist(df.loc[msel, col].dropna(), bins=bins, alpha=0.8, label="selected")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title or f"{col}: selected vs not")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def boxplot_delta_selected_vs_not(
    df: pd.DataFrame, col: str, selcol: str, outpath: str, title: Optional[str] = None
):
    if df.empty or col not in df.columns:
        return
    fig, ax = plt.subplots()
    data = [
        df.loc[df[selcol] != True, col].dropna().values,
        df.loc[df[selcol] == True, col].dropna().values,
    ]
    if len(data[0]) + len(data[1]) == 0:
        plt.close(fig)
        return
    ax.boxplot(data, labels=["not selected", "selected"], showmeans=True)
    ax.set_ylabel(col)
    ax.set_title(title or f"{col} Δ: selected vs not")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ---- Per-letter colored labels for traces ----
def _stage_letter_colors(
    seq_len: int, c_idx: int, p_idx: int, m_idx: int, preP_idx: Optional[int]
) -> List[List[str]]:
    """Return per-position colors for labels at each stage (orig, C, P, M)."""
    stages = 4
    cols = [[STAGE_COLORS["orig"]] * seq_len for _ in range(stages)]
    # C from stage 1 onward
    if 0 <= c_idx < seq_len:
        for s in (1, 2, 3):
            cols[s][c_idx] = STAGE_COLORS["C"]
    # P from stage 2 onward
    if 0 <= p_idx < seq_len:
        for s in (2, 3):
            cols[s][p_idx] = STAGE_COLORS["P"]
    # pre-P predecessor (if ever present) from stage 2 onward
    if preP_idx is not None and 0 <= preP_idx < seq_len:
        for s in (2, 3):
            cols[s][preP_idx] = PREP_FIX_COLOR
    # M at stage 3
    if 0 <= m_idx < seq_len:
        cols[3][m_idx] = STAGE_COLORS["M"]
    return cols


def _annotate_colored_seq(ax, x, y, seq: str, colors: List[str], *, fontsize=8):
    dx = 0
    for ch, col in zip(seq, colors):
        ax.annotate(
            ch,
            (x, y),
            textcoords="offset points",
            xytext=(dx, 5),
            fontsize=fontsize,
            color=col,
            family="DejaVu Sans Mono",
        )
        dx += 6  # ~6 px per mono char


def plot_trace_for_peptide_labeled(
    outdir: str,
    idx_or_rank: int,
    selected_flag: bool,
    seqs: List[str],
    flies: List[float],
    confs: List[float],
    title_extra: str = "",
    max_label_len: int = 36,
    *,
    c_idx: Optional[int] = None,
    p_idx: Optional[int] = None,
    m_idx: Optional[int] = None,
    preP_idx: Optional[int] = None,
):
    """Per-peptide labeled trace; each residue colored by stage it changed."""
    stages = ["orig", "C", "P", "M"]
    seq0, seq3 = seqs[0], seqs[-1]
    fig, ax = plt.subplots()
    xs = list(flies)
    ys = list(confs)
    for i in range(len(stages) - 1):
        ax.plot(
            [xs[i], xs[i + 1]], [ys[i], ys[i + 1]], lw=1.2, color="#777777", alpha=0.8
        )
    for i, st in enumerate(stages):
        ax.scatter(
            [xs[i]],
            [ys[i]],
            s=40,
            color=STAGE_COLORS[st],
            edgecolor="k",
            linewidths=0.5,
            zorder=3,
        )
        seq = seqs[i]
        if all(v is not None for v in (c_idx, p_idx, m_idx)) and len(seq) > 0:
            stage_cols = _stage_letter_colors(len(seq), c_idx, p_idx, m_idx, preP_idx)[
                i
            ]
            if len(seq) > max_label_len:
                keep = max(6, (max_label_len - 3) // 2)
                left, right = seq[:keep], seq[-keep:]
                left_cols, right_cols = stage_cols[:keep], stage_cols[-keep:]
                _annotate_colored_seq(ax, xs[i], ys[i], left, left_cols, fontsize=8)
                ax.annotate(
                    "...",
                    (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=(6 * keep, 5),
                    fontsize=8,
                    color="#444444",
                    family="DejaVu Sans Mono",
                )
                _annotate_colored_seq(ax, xs[i], ys[i], right, right_cols, fontsize=8)
            else:
                _annotate_colored_seq(ax, xs[i], ys[i], seq, stage_cols, fontsize=8)
        else:
            ax.annotate(
                seq,
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=STAGE_COLORS[st],
            )
    ax.set_xlabel("fly_surface_norm")
    ax.set_ylabel("confusability (max across sets)")
    ttl1 = (
        f"Trace (orig → C → P → M) [{'SELECTED' if selected_flag else 'not selected'}]"
    )
    ttl2 = f"orig: {truncate_seq(seq0)}  →  final: {truncate_seq(seq3)}"
    if title_extra:
        ttl2 += f"  |  {title_extra}"
    ax.set_title(ttl1 + "\n" + ttl2)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=STAGE_COLORS[s],
            label=s,
            markeredgecolor="k",
            markersize=6,
        )
        for s in stages
    ]
    ax.legend(handles=handles, title="Stage", loc="best")
    _grid(ax)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    slug0 = slugify_seq(seq0, maxlen=40)
    slug3 = slugify_seq(seq3, maxlen=40)
    fname = f"trace_{idx_or_rank:04d}__{'SEL' if selected_flag else 'NOT'}__orig_{slug0}__final_{slug3}.png"
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


# -------------------------- Mut index resolution ---------------------
def parse_first_int(s: str) -> Optional[int]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


# --- Helpers to parse mutation notation for ref/mut AAs ---
_AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def _aa3_to1(tok: str) -> Optional[str]:
    return _AA3.get(tok.upper())


def parse_ref_mut_letters(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (ref_aa, mut_aa) as 1-letter codes if recognized, else (None, None).
    Handles: 'L858R', 'p.L858R', 'p.G12D', 'p.Gly12Asp'.
    """
    if s is None:
        return (None, None)
    txt = str(s).strip()
    # Single-letter form
    m1 = re.search(r"[pP]\.\s*([A-Z\*])\s*\d+\s*([A-Z\*])", txt) or re.search(
        r"\b([A-Z\*])\s*\d+\s*([A-Z\*])\b", txt
    )
    if m1:
        ref, mut = m1.group(1), m1.group(2)
        return (
            ref if ref and ref.isalpha() else None,
            mut if mut and mut.isalpha() else None,
        )
    # Three-letter form
    m2 = re.search(r"(?:[pP]\.\s*)?([A-Za-z]{3})\s*\d+\s*([A-Za-z]{3})", txt)
    if m2:
        ref3, mut3 = m2.group(1), m2.group(2)
        ref1 = _aa3_to1(ref3)
        mut1 = _aa3_to1(mut3)
        return (ref1, mut1)
    return (None, None)


def resolve_mut_idx_column(df: pd.DataFrame, args) -> pd.Series:
    """
    Return a 0-based mut_idx0 Series (int) or raise ValueError with details.
    Sources of truth (in order):
      1) --mut-idx-col (already 0-based)
      2) --pos-col + --indexing (0|1|C1)
      3) --mut-notation-col + --indexing (extract integer, then convert)
    """
    pepcol = args.peptide_col
    Ls = df[pepcol].astype(str).map(len)

    if args.mut_idx_col and args.mut_idx_col in df.columns:
        s = pd.to_numeric(df[args.mut_idx_col], errors="coerce").astype("Int64")
        mut0 = s.copy()
    else:
        mut0 = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    if mut0.isna().any():
        if args.pos_col and args.pos_col in df.columns:
            pos = pd.to_numeric(df[args.pos_col], errors="coerce").astype("Int64")
            if str(args.indexing).upper() in ("1", "C1"):
                pos = pos - 1
            elif str(args.indexing).upper() in ("0",):
                pass
            else:
                raise ValueError(
                    f"--indexing must be one of 0, 1, C1 (got {args.indexing})"
                )
            mut0 = mut0.fillna(pos)

    if mut0.isna().any():
        if args.mut_notation_col and args.mut_notation_col in df.columns:
            extra = df[args.mut_notation_col].apply(parse_first_int).astype("Int64")
            if str(args.indexing).upper() in ("1", "C1"):
                extra = extra - 1
            mut0 = mut0.fillna(extra)

    missing_mask = mut0.isna()
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
            ].copy()
            path = os.path.join(args.outdir, "missing_mut_idx0_rows.csv")
            os.makedirs(args.outdir, exist_ok=True)
            bad.to_csv(path, index=False)
            raise ValueError(
                f"[mut_idx0] Required but missing/invalid for {int(missing_mask.sum())} rows. "
                f"See {path}. Provide --mut-idx-col (0-based), or --pos-col/--mut-notation-col with --indexing."
            )
        else:
            logx(
                "[mut_idx0] Missing rows will use auto Cys placement near the middle (override enabled)."
            )
            auto = (
                df[pepcol]
                .astype(str)
                .map(lambda s: max(1, min(len(s) - 2, len(s) // 2)))
            )
            mut0 = mut0.fillna(auto).astype("Int64")

    out = []
    errors = []
    for i, (idx0, L) in enumerate(zip(mut0.tolist(), Ls.tolist())):
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
        os.makedirs(args.outdir, exist_ok=True)
        pd.DataFrame(
            [{"row": i, "reason": r, "peptide": df.iloc[i][pepcol]} for i, r in errors]
        ).to_csv(path, index=False)
        raise ValueError(f"[mut_idx0] Found {len(errors)} invalid indices; see {path}")
    return pd.Series(out, index=df.index, dtype=int)


# -------------------------- Main pipeline ----------------------------
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Read input & map columns ----
    df_in = pd.read_csv(args.input_csv)
    if args.peptide_col not in df_in.columns:
        raise ValueError(
            f"Missing peptide column '{args.peptide_col}'. Available: {list(df_in.columns)[:12]}..."
        )
    df = df_in.copy()
    df[args.peptide_col] = df[args.peptide_col].map(clean_pep)
    df = df[df[args.peptide_col].str.len() > 0].copy()

    # ---- Falloff tracker ----
    falloff = []

    def _stage(name: str, n: int):
        falloff.append({"stage": name, "n": int(n)})

    _stage("loaded_nonempty", len(df))

    # ---- Resolve mut_idx0 (before input filters that depend on it) ----
    df["mut_idx0"] = resolve_mut_idx_column(df, args)
    if "is_hit" not in df.columns:
        df["is_hit"] = np.nan

    # ---- Derive ref AA at mut index; parse mut AA from notation (if provided) ----
    pepcol = args.peptide_col

    def _ref_at_mut(row):
        s = row[pepcol]
        i = int(row["mut_idx0"])
        return s[i] if 0 <= i < len(s) else None

    df["_ref_aa_at_mut"] = df.apply(_ref_at_mut, axis=1)
    if args.mut_notation_col and args.mut_notation_col in df.columns:
        tmp = df[args.mut_notation_col].apply(parse_ref_mut_letters)
        df["_mut_aa_from_notation"] = [t[1] for t in tmp]
    else:
        df["_mut_aa_from_notation"] = None

    # ---- Mandatory input filters (defaults ON; can be toggled via CLI) ----
    drop_reports = []

    # 1) Drop if reference OR mutant AA (at the mutation site) is C
    if int(args.drop_ref_or_mut_C) == 1:
        mask_refC = df["_ref_aa_at_mut"] == "C"
        mask_mutC = df["_mut_aa_from_notation"] == "C"
        maskC = mask_refC | mask_mutC
        if maskC.any():
            drop_reports.append(
                (
                    "dropped_ref_or_mut_C.csv",
                    df.loc[
                        maskC,
                        [pepcol, "mut_idx0"]
                        + ([args.mut_notation_col] if args.mut_notation_col else []),
                    ].copy(),
                )
            )
            df = df.loc[~maskC].copy()
        _stage("after_drop_ref_or_mut_C", len(df))

    # 2) Drop peptides that already contain P
    if int(args.drop_inputs_with_P) == 1:
        maskP = df[pepcol].str.contains("P", regex=False)
        if maskP.any():
            drop_reports.append(
                ("dropped_input_has_P.csv", df.loc[maskP, [pepcol]].copy())
            )
            df = df.loc[~maskP].copy()
        _stage("after_drop_any_P", len(df))

    # 3) Drop peptides that already contain M
    if int(args.drop_inputs_with_M) == 1:
        maskM = df[pepcol].str.contains("M", regex=False)
        if maskM.any():
            drop_reports.append(
                ("dropped_input_has_M.csv", df.loc[maskM, [pepcol]].copy())
            )
            df = df.loc[~maskM].copy()
        _stage("after_drop_any_M", len(df))

    # ---- Build/Load contaminant sets ----
    charges_prec = sorted({int(z) for z in args.charges.split(",") if z.strip()})
    frag_charges = (
        charges_prec
        if (args.frag_charges is None or str(args.frag_charges).strip() == "")
        else sorted({int(z) for z in str(args.frag_charges).split(",") if z.strip()})
    )
    cys_fixed = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    acc2seq_by_set = {}
    windows_by_set = {}
    for s in sets:
        acc2seq = load_set_sequences(s, args)
        if not acc2seq:
            windows_by_set[s] = []
            acc2seq_by_set[s] = {}
            continue
        acc2seq_by_set[s] = acc2seq
        wins = build_windows_index(
            s,
            acc2seq,
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
        logx(f"[windows] {s}: {len(wins)} windows indexed")

    # Terminal k-mer banks
    kmer_to_accs_by_set, acc_to_kmers_by_set = build_kmer_banks(
        acc2seq_by_set, args.term_k, collapse_xle_flag=bool(args.xle_collapse)
    )

    # PEG candidates
    peg_cands = enumerate_polymer_candidates(
        families=tuple(x.strip() for x in args.peg_families.split(",") if x.strip()),
        endgroups=tuple(x.strip() for x in args.peg_endgroups.split(",") if x.strip()),
        adducts=tuple(x.strip() for x in args.peg_adducts.split(",") if x.strip()),
        zset=tuple(int(z) for z in args.peg_z.split(",") if z.strip()),
        n_min=args.peg_n_min,
        n_max=args.peg_n_max,
        mz_min=args.peg_mz_min,
        mz_max=args.peg_mz_max,
    )
    logx(f"[peg] candidates: {len(peg_cands)}")

    # ---- Generate decoys & compute features ----
    rows = []
    stage_counts = []
    dropped = []
    for _, r in df.iterrows():
        pep = r[args.peptide_col]
        mut_idx0 = int(r["mut_idx0"])
        di = generate_decoy_best(
            pep,
            mut_idx0,
            sets,
            windows_by_set,
            charges_prec,
            frag_charges,
            args,
            cys_fixed,
        )
        if di.get("drop_rk_only", 0) == 1:
            dropped.append({"peptide": pep, "reason": di.get("drop_reason", "rk_only")})
            continue
        dec = di["decoy_seq"]

        # Core signals from final seq
        fly3 = di["eval3"]["fly"]
        hydro3 = di["eval3"]["hydro"]
        rt_pred3 = di["eval3"]["rt"]
        conf3 = di["eval3"]["conf_mz_multi"]

        # Per-set signals (final)
        conf_per_set3 = di["eval3"]["conf_per_set"]
        bestppm_per_set3 = di["eval3"]["bestppm_per_set"]

        # Strict protein list (final)
        proteins_good_mz = []
        for s in sets:
            wins = windows_by_set.get(s, [])
            if not wins:
                continue
            dtmp = confusability_script1(
                dec,
                wins,
                args.full_mz_len_min,
                args.full_mz_len_max,
                charges_prec,
                args.ppm_tol,
                rt_pred3,
                args.rt_tolerance_min,
                cys_fixed,
                args.fragment_kmin,
                args.fragment_kmax,
                good_ppm=args.good_ppm_strict,
                frag_types_req=args.frag_types_req,
                frag_charges=frag_charges,
            )
            pep_m = mass_neutral(dec, cys_fixed)
            pep_mz_by_z = {z: mz_from_mass(pep_m, z) for z in charges_prec}
            strict_idxs = (
                dtmp["prec_idxs_rt"] if args.strict_use_rt else dtmp["prec_idxs"]
            )
            strict_accs = set()
            for wi in strict_idxs:
                w = wins[wi]
                okppm = False
                for z, pmz in pep_mz_by_z.items():
                    wmz = w.mz_by_z.get(z)
                    if wmz is None:
                        continue
                    ppm = abs(pmz - wmz) / pmz * 1e6
                    if ppm <= args.good_ppm_strict:
                        okppm = True
                        break
                if okppm:
                    strict_accs.add(w.acc)
            if dtmp["frag_types_rt"] >= args.frag_types_req and strict_accs:
                for acc in sorted(strict_accs):
                    proteins_good_mz.append(f"{s}:{acc}")

        # Terminal kmers (final)
        tk3 = term_kmer_matches(
            dec,
            args.term_k,
            kmer_to_accs_by_set,
            collapse_xle_flag=bool(args.xle_collapse),
        )
        term3 = tk3["bg_term_kmer_frac"]
        proteins_term = []
        for s2, accs in tk3["accessions_by_set"].items():
            for a in accs:
                proteins_term.append(f"{s2}:{a}")

        # PEG (final)
        peg3 = peg_best_for_peptide(
            dec,
            peg_cands,
            peptide_z=tuple(charges_prec),
            ppm_window=args.peg_ppm_strict,
        )
        peg_conf3 = peg3.get("peg_conf", 0.0)
        peg_good3 = int(peg3.get("peg_match_flag", 0) == 1)
        peg_ppm3 = peg3.get("peg_best_abs_ppm", float("inf"))
        peg_details3 = peg3.get("peg_details", "")

        # Contaminant categories matched (final)
        cats3 = []
        for s in sets:
            if (
                conf_per_set3.get(s, 0.0) >= args.contam_match_min_score
                or bestppm_per_set3.get(s, float("inf")) <= args.good_ppm_strict
            ):
                cats3.append(SET_FRIENDLY.get(s, s))
        if peg_good3:
            cats3.append("PEG-like")
        cats_sorted = sorted(set(cats3))
        distinct_types = len(cats_sorted)

        # Hydrophobic C-term (final)
        cterm_hydrophobic = 1 if dec[-1] in HYDRO_SET else 0

        # START (original) features for improvement & constraints
        eval0 = di["eval0"]
        conf0 = eval0["conf_mz_multi"]
        fly0 = eval0["fly"]
        hydro0 = eval0["hydro"]
        tk0 = term_kmer_matches(
            pep,
            args.term_k,
            kmer_to_accs_by_set,
            collapse_xle_flag=bool(args.xle_collapse),
        )
        term0 = tk0["bg_term_kmer_frac"]
        peg0 = peg_best_for_peptide(
            pep,
            peg_cands,
            peptide_z=tuple(charges_prec),
            ppm_window=args.peg_ppm_strict,
        )
        peg_conf0 = peg0.get("peg_conf", 0.0)
        cats0 = []
        conf_per_set0 = eval0["conf_per_set"]
        bestppm_per_set0 = eval0["bestppm_per_set"]
        for s in sets:
            if (
                conf_per_set0.get(s, 0.0) >= args.contam_match_min_score
                or bestppm_per_set0.get(s, float("inf")) <= args.good_ppm_strict
            ):
                cats0.append(SET_FRIENDLY.get(s, s))
        types0 = len(set(cats0))

        rows.append(
            {
                # IDs & edits
                "peptide": pep,
                "decoy_seq": dec,
                "selected": False,
                "mut_idx0": di["c_idx"],
                "m_idx": di["m_idx"],
                "p_idx": di["p_idx"],
                "preP_idx": np.nan,  # reserved, currently unused
                "edit_summary": di["reasons"],
                # stage sequences
                "seq0": di["seq0"],
                "seq1": di["seq1"],
                "seq2": di["seq2"],
                "seq3": di["seq3"],
                # stage metrics (store for traces)
                "fly0": fly0,
                "conf0": conf0,
                "fly1": di["eval1"]["fly"],
                "conf1": di["eval1"]["conf_mz_multi"],
                "fly2": di["eval2"]["fly"],
                "conf2": di["eval2"]["conf_mz_multi"],
                # final metrics
                "fly_surface_norm": fly3,
                "hydrophobic_fraction": hydro3,
                "rt_pred_min": rt_pred3,
                "conf_mz_multi": conf3,
                "bg_term_kmer_frac": term3,
                "peg_conf": peg_conf3,
                "peg_match_flag": peg_good3,
                "peg_best_abs_ppm": peg_ppm3,
                "peg_details": peg_details3,
                "contam_categories_matched": ";".join(cats_sorted),
                "contam_categories_count": distinct_types,
                "contam_categories_all": ";".join(cats_sorted),
                "contam_proteins_mz_matched": ";".join(sorted(set(proteins_good_mz))),
                "contam_proteins_termkmer_matched": ";".join(
                    sorted(set(proteins_term))
                ),
                "cterm_hydrophobic": cterm_hydrophobic,
                "p_tier": di["p_tier"],
                "p_rk_before": di["p_rk_before"],
                "dist_p_to_c": di["dist_p_to_c"],
                "dist_m_to_p": di["dist_m_to_p"],
                "deltaP_hydro": di["deltaP_hydro"],
                "deltaP_fly": di["deltaP_fly"],
                "deltaM_hydro": di["deltaM_hydro"],
                "deltaM_fly": di["deltaM_fly"],
                # start components for scoring
                "hydro0": hydro0,
                "term0": term0,
                "peg0": peg_conf0,
                "types0": types0,
                # per-set diagnostics (final)
                **{
                    f"set_{s}__conf_score": di["eval3"]["conf_per_set"].get(s, 0.0)
                    for s in sets
                },
                **{
                    f"set_{s}__best_ppm": di["eval3"]["bestppm_per_set"].get(
                        s, float("inf")
                    )
                    for s in sets
                },
            }
        )

        stage_counts.append(
            {
                "peptide": pep,
                "p_candidates_total": di["p_candidates_total"],
                "p_candidates_nonRK": di["p_candidates_nonRK"],
                "m_candidates_total": di["m_candidates_total"],
            }
        )

    out = pd.DataFrame(rows)
    st_df = pd.DataFrame(stage_counts)

    # ---- FINAL SCORE (weights) ----
    wspec = {}
    for tok in str(args.rank_weights).split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        try:
            wspec[k.strip()] = float(v.strip())
        except:
            pass
    keys = ["fly", "hydro", "conf_mz", "term", "peg", "types"]
    w = {
        "fly": wspec.get("fly", 0.50),
        "hydro": wspec.get("hydro", 0.20),
        "conf_mz": wspec.get("conf_mz", 0.18),
        "term": wspec.get("term", 0.06),
        "peg": wspec.get("peg", 0.04),
        "types": wspec.get("types", 0.02),
    }
    s = sum(w[k] for k in keys)
    s = s if s > 0 else 1.0
    w = {k: (w[k] / s) for k in keys}

    # Final score (stage 3)
    out["final_score"] = (
        w["fly"] * out["fly_surface_norm"].fillna(0).astype(float)
        + w["hydro"] * out["hydrophobic_fraction"].fillna(0).astype(float)
        + w["conf_mz"] * out["conf_mz_multi"].fillna(0).astype(float)
        + w["term"] * out["bg_term_kmer_frac"].fillna(0).astype(float)
        + w["peg"] * out["peg_conf"].fillna(0).astype(float)
        + w["types"]
        * (
            out["contam_categories_count"].fillna(0).astype(float)
            / float(len(sets) + 1)
        )
    )
    # Start score (stage 0)
    out["start_score"] = (
        w["fly"] * out["fly0"].fillna(0).astype(float)
        + w["hydro"] * out["hydro0"].fillna(0).astype(float)
        + w["conf_mz"] * out["conf0"].fillna(0).astype(float)
        + w["term"] * out["term0"].fillna(0).astype(float)
        + w["peg"] * out["peg0"].fillna(0).astype(float)
        + w["types"] * (out["types0"].fillna(0).astype(float) / float(len(sets) + 1))
    )
    # Ranks & improvements
    if not out.empty:
        out["start_rank"] = (
            out["start_score"].rank(ascending=False, method="min").astype(int)
        )
        out["end_rank"] = (
            out["final_score"].rank(ascending=False, method="min").astype(int)
        )
        out["rank_improvement"] = out["start_rank"] - out["end_rank"]
        n_total = max(1, len(out) - 1)
        out["rank_improvement_norm"] = out["rank_improvement"].clip(lower=0).astype(
            float
        ) / float(n_total)
        out["delta_total_fly"] = out["fly_surface_norm"] - out["fly0"]
        out["delta_total_conf"] = out["conf_mz_multi"] - out["conf0"]
        out["delta_total_score"] = out["final_score"] - out["start_score"]
    else:
        for c in [
            "start_rank",
            "end_rank",
            "rank_improvement",
            "rank_improvement_norm",
            "delta_total_fly",
            "delta_total_conf",
            "delta_total_score",
        ]:
            out[c] = []

    # Optional filter for hydrophobic C-term at selection time
    out["_eligible_cterm"] = (
        True if not args.enforce_hydrophobic_cterm else (out["cterm_hydrophobic"] == 1)
    )

    # ---- Selection: greedy with k-mer de-dup + diversity novelty + improvement bonus + HARD CONSTRAINTS ----
    N = int(args.N)
    kdedup = int(args.dedup_k)
    require_contam = bool(args.require_contam_match)
    selected_idx = []
    seen_kmers = set()
    owner = {}
    dedup_exclusions = []
    covered_cats = set()
    remaining = set(out.index.tolist())
    while len(selected_idx) < N and remaining:
        scored = []
        for i in list(remaining):
            rr = out.loc[i]
            # Hard constraints
            if require_contam and (int(rr["contam_categories_count"]) < 1):
                scored.append((i, -1e9, "no_contaminant_match"))
                continue
            if not bool(rr["_eligible_cterm"]):
                scored.append((i, -1e9, "non_hydrophobic_cterm"))
                continue
            # Final confusability >= min_ratio * start confusability
            conf_ok = float(rr["conf_mz_multi"]) >= float(
                args.min_confusability_ratio
            ) * float(rr["conf0"])
            if not conf_ok:
                scored.append((i, -1e9, "confusability_ratio_below_threshold"))
                continue
            # Hydrophobicity increase
            hydro_ok = float(rr["hydrophobic_fraction"]) >= float(rr["hydro0"]) + float(
                args.hydro_delta_min
            )
            if args.require_hydro_increase and not hydro_ok:
                scored.append((i, -1e9, "hydrophobicity_not_increased"))
                continue
            # k-mer de-dup
            if kdedup > 0:
                kms = kmer_set(rr["decoy_seq"], kdedup)
                overlap = kms & seen_kmers
                if overlap:
                    blockers = sorted({owner.get(k, "") for k in overlap if k in owner})
                    dedup_exclusions.append(
                        {
                            "decoy_seq": rr["decoy_seq"],
                            "reason": "kmer_overlap",
                            "blocked_by": ";".join([b for b in blockers if b]),
                            "n_shared_kmers": len(overlap),
                            "shared_kmers": ";".join(sorted(list(overlap))[:50]),
                        }
                    )
                    scored.append((i, -1e9, "kmer_overlap"))
                    continue
            cats = set(
                [c for c in str(rr["contam_categories_matched"]).split(";") if c]
            )
            novelty = len(cats - covered_cats) / float(len(sets) + 1)
            aug = (
                float(rr["final_score"])
                + float(args.diversity_bonus) * novelty
                + float(args.improvement_bonus) * float(rr["rank_improvement_norm"])
            )
            scored.append((i, aug, ""))

        if not scored:
            break
        scored.sort(key=lambda t: t[1], reverse=True)
        best_i, best_aug, _ = scored[0]
        if best_aug < -1e8:
            break
        selected_idx.append(best_i)
        out.at[best_i, "selected"] = True
        rr = out.loc[best_i]
        if kdedup > 0:
            for km in kmer_set(rr["decoy_seq"], kdedup):
                if km not in seen_kmers:
                    seen_kmers.add(km)
                    owner[km] = rr["decoy_seq"]
        covered_cats |= set(
            [c for c in str(rr["contam_categories_matched"]).split(";") if c]
        )
        remaining.remove(best_i)

    selected = out.loc[selected_idx].copy().sort_values("final_score", ascending=False)

    # ---- Clean columns & order ----
    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in df.columns if not c.startswith("_")]
        df = df[keep].copy()
        first = [
            "decoy_seq",
            "peptide",
            "selected",
            "final_score",
            "start_score",
            "start_rank",
            "end_rank",
            "rank_improvement",
            "rank_improvement_norm",
            "delta_total_score",
            "delta_total_fly",
            "delta_total_conf",
            "fly_surface_norm",
            "hydrophobic_fraction",
            "conf_mz_multi",
            "bg_term_kmer_frac",
            "peg_conf",
            "contam_categories_count",
            "contam_categories_matched",
            "contam_categories_all",
            "peg_match_flag",
            "peg_best_abs_ppm",
            "peg_details",
            "contam_proteins_mz_matched",
            "contam_proteins_termkmer_matched",
            "rt_pred_min",
            "mut_idx0",
            "m_idx",
            "p_idx",
            "edit_summary",
            "p_tier",
            "p_rk_before",
            "dist_p_to_c",
            "dist_m_to_p",
            "deltaP_hydro",
            "deltaP_fly",
            "deltaM_hydro",
            "deltaM_fly",
            "cterm_hydrophobic",
            "fly0",
            "hydro0",
            "conf0",
            "term0",
            "peg0",
            "types0",
            "fly1",
            "conf1",
            "fly2",
            "conf2",
            "seq0",
            "seq1",
            "seq2",
            "seq3",
            "preP_idx",
        ]
        ordered = [c for c in first if c in df.columns] + [
            c for c in df.columns if c not in first
        ]
        return df[ordered]

    full = reorder(out)
    sel = reorder(selected)

    # ---- Write outputs ----
    full.to_csv(os.path.join(args.outdir, "decoy_candidates_all.csv"), index=False)
    sel.to_csv(os.path.join(args.outdir, "decoy_selected_topN.csv"), index=False)
    if len(selected_idx) > 0:
        pd.DataFrame({"selected_index": selected_idx}).to_csv(
            os.path.join(args.outdir, "selected_indices.csv"), index=False
        )
    if dropped:
        pd.DataFrame(dropped).to_csv(
            os.path.join(args.outdir, "dropped_rk_only.csv"), index=False
        )
    if "dedup_exclusions" in locals() and dedup_exclusions:
        pd.DataFrame(dedup_exclusions).to_csv(
            os.path.join(args.outdir, "dedup_exclusions.csv"), index=False
        )
    st_df.to_csv(os.path.join(args.outdir, "stage_counts.csv"), index=False)

    # Write input filter drop reports
    for fname, ddf in drop_reports:
        try:
            ddf.to_csv(os.path.join(args.outdir, fname), index=False)
        except Exception as e:
            logx(f"[warn] failed to write {fname}: {e}")

    # ---- Falloff CSV + plot ----
    try:
        # Count after RK-only drop: remaining = attempted(after last input filter) - dropped
        attempted_after_filters = falloff[-1]["n"] if falloff else len(df)
        after_rk_only = int(attempted_after_filters) - int(len(dropped))
        falloff.append({"stage": "after_drop_rk_only", "n": after_rk_only})
        fall_df = pd.DataFrame(falloff)
        fall_df.to_csv(
            os.path.join(args.outdir, "peptide_falloff_counts.csv"), index=False
        )

        if not fall_df.empty:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(fall_df)), fall_df["n"].values, marker="o")
            ax.set_xticks(np.arange(len(fall_df)))
            ax.set_xticklabels(fall_df["stage"].tolist(), rotation=30, ha="right")
            ax.set_ylabel("peptides remaining")
            ax.set_title("Peptide count falloff across filters")
            _grid(ax)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "peptide_falloff.png"))
            plt.close(fig)
    except Exception as e:
        logx(f"[plots][WARN] falloff plot skipped: {e}")

    # ---- Plots ----
    # Decomposition (selected only)
    def final_score_decomp(
        sel_df: pd.DataFrame, parts: List[Tuple[str, float]], outpath: str
    ):
        if sel_df.empty:
            return
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(sel_df)), 4.2))
        ssel = sel_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        x = np.arange(len(ssel))
        bottom = np.zeros(len(ssel), dtype=float)
        for col, wv in parts:
            contrib = wv * ssel[col].values
            ax.bar(x, contrib, bottom=bottom, label=f"{col}×{wv:.2f}")
            bottom += contrib
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_ylabel("Final score (weighted)")
        ax.set_title("Final-score decomposition (selected)")
        ax.legend()
        _grid(ax)
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)

    parts = [
        ("fly_surface_norm", w["fly"]),
        ("hydrophobic_fraction", w["hydro"]),
        ("conf_mz_multi", w["conf_mz"]),
        ("bg_term_kmer_frac", w["term"]),
        ("peg_conf", w["peg"]),
    ]
    if not sel.empty:
        final_score_decomp(
            sel,
            parts,
            os.path.join(plots_dir, "selected_final_score_decomposition.png"),
        )

    scatter_pairs = [
        (
            "fly_surface_norm",
            "bg_term_kmer_frac",
            "scatter_fly_vs_term.png",
            "bg_term_kmer_frac vs fly",
        ),
        (
            "fly_surface_norm",
            "conf_mz_multi",
            "scatter_fly_vs_conf_mz.png",
            "conf_mz_multi vs fly",
        ),
        ("final_score", "fly_surface_norm", "scatter_final_vs_fly.png", "final vs fly"),
        (
            "final_score",
            "conf_mz_multi",
            "scatter_final_vs_conf_mz.png",
            "final vs conf_mz",
        ),
        (
            "final_score",
            "bg_term_kmer_frac",
            "scatter_final_vs_term.png",
            "final vs term",
        ),
        ("final_score", "peg_conf", "scatter_final_vs_peg.png", "final vs peg"),
        (
            "hydrophobic_fraction",
            "conf_mz_multi",
            "scatter_hydro_vs_conf_mz.png",
            "conf_mz vs hydro",
        ),
    ]
    for x, y, f, t in scatter_pairs:
        if not full.empty:
            scatter_sel_vs_not(
                full,
                x,
                y,
                "selected",
                os.path.join(plots_dir, f),
                title=t,
                dense_threshold=args.dense_threshold,
            )

    if not full.empty:
        scatter_with_diag(
            full,
            "start_score",
            "final_score",
            "selected",
            os.path.join(plots_dir, "scatter_final_vs_start_score.png"),
            title="Final vs Start score",
            dense_threshold=args.dense_threshold,
        )
        scatter_with_diag(
            full,
            "start_rank",
            "end_rank",
            "selected",
            os.path.join(plots_dir, "scatter_end_vs_start_rank.png"),
            title="End rank vs Start rank",
            dense_threshold=args.dense_threshold,
        )
        plot_hist_selected_vs_not(
            full,
            "rank_improvement",
            "selected",
            os.path.join(plots_dir, "hist_rank_improvement_sel_vs_not.png"),
            title="Rank improvement (start - end)",
        )
        boxplot_delta_selected_vs_not(
            full,
            "delta_total_fly",
            "selected",
            os.path.join(plots_dir, "box_delta_total_fly.png"),
            title="Δ fly_surface_norm (final - start)",
        )
        boxplot_delta_selected_vs_not(
            full,
            "delta_total_conf",
            "selected",
            os.path.join(plots_dir, "box_delta_total_conf.png"),
            title="Δ confusability (final - start)",
        )
        boxplot_delta_selected_vs_not(
            full,
            "delta_total_score",
            "selected",
            os.path.join(plots_dir, "box_delta_total_score.png"),
            title="Δ final score (final - start)",
        )

    # Core step plots
    try:
        if not full.empty:

            def plot_p_tier_counts(dfX: pd.DataFrame, path: str):
                tiers = ["GP", "AP", "LVISTP", "other"]
                counts = [int((dfX["p_tier"] == t).sum()) for t in tiers]
                fig, ax = plt.subplots()
                ax.bar(tiers, counts)
                ax.set_ylabel("count")
                ax.set_title("P tier counts")
                _grid(ax)
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)

            def plot_rk_filter_counts(df_stage: pd.DataFrame, path: str):
                if df_stage.empty:
                    return
                fig, ax = plt.subplots()
                ax.bar(
                    ["total P", "non-RK"],
                    [
                        df_stage["p_candidates_total"].sum(),
                        df_stage["p_candidates_nonRK"].sum(),
                    ],
                )
                ax.set_ylabel("count (sum over peptides)")
                ax.set_title("RK-before-P filter counts")
                _grid(ax)
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)

            def plot_dist_hist(dfX: pd.DataFrame, path: str):
                fig, ax = plt.subplots()
                ax.hist(dfX["dist_p_to_c"].dropna(), bins=20, alpha=0.7, label="|P−C|")
                ax.hist(dfX["dist_m_to_p"].dropna(), bins=20, alpha=0.7, label="|M−P|")
                ax.set_title("Distance distributions")
                ax.set_xlabel("residues")
                ax.set_ylabel("count")
                ax.legend()
                _grid(ax)
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)

            def plot_delta(
                dfX: pd.DataFrame, col_dh: str, col_df: str, path: str, title: str
            ):
                fig, ax = plt.subplots()
                ax.scatter(
                    dfX[col_dh].fillna(0), dfX[col_df].fillna(0), s=12, alpha=0.6
                )
                ax.axvline(0, lw=0.8)
                ax.axhline(0, lw=0.8)
                ax.set_xlabel("Δhydrophobicity")
                ax.set_ylabel("Δfly_score_norm")
                ax.set_title(title)
                _grid(ax)
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)

            plot_p_tier_counts(full, os.path.join(plots_dir, "p_tier_counts.png"))
            plot_rk_filter_counts(
                st_df, os.path.join(plots_dir, "p_rk_filter_counts.png")
            )
            plot_dist_hist(full, os.path.join(plots_dir, "dist_to_C_and_P_hist.png"))
            plot_delta(
                full,
                "deltaP_hydro",
                "deltaP_fly",
                os.path.join(plots_dir, "delta_hydro_fly_P.png"),
                "Δ metrics after P step",
            )
            plot_delta(
                full,
                "deltaM_hydro",
                "deltaM_fly",
                os.path.join(plots_dir, "delta_hydro_fly_M.png"),
                "Δ metrics after M step",
            )
    except Exception as e:
        logx(f"[plots][WARN] Step-plot generation skipped due to: {e}")

    # ---- Per-peptide labeled traces ----
    traces_sel_dir = os.path.join(plots_dir, "seq_traces_selected")
    traces_not_dir = os.path.join(plots_dir, "seq_traces_not_selected")
    if not sel.empty:
        ssel = sel.reset_index(drop=True)
        for rank, rr in ssel.iterrows():
            flies = [rr["fly0"], rr["fly1"], rr["fly2"], rr["fly_surface_norm"]]
            confs = [rr["conf0"], rr["conf1"], rr["conf2"], rr["conf_mz_multi"]]
            seqs = [rr["seq0"], rr["seq1"], rr["seq2"], rr["seq3"]]
            plot_trace_for_peptide_labeled(
                traces_sel_dir,
                rank + 1,
                True,
                seqs,
                flies,
                confs,
                title_extra=f"Δscore={rr['delta_total_score']:.3f}",
                max_label_len=args.trace_label_maxlen,
                c_idx=int(rr["mut_idx0"]),
                p_idx=int(rr["p_idx"]),
                m_idx=int(rr["m_idx"]),
                preP_idx=(
                    int(rr["preP_idx"])
                    if pd.notna(rr.get("preP_idx", np.nan))
                    else None
                ),
            )
    not_sel = full.loc[full["selected"] != True].reset_index(drop=True)
    for i, rr in not_sel.iterrows():
        flies = [rr["fly0"], rr["fly1"], rr["fly2"], rr["fly_surface_norm"]]
        confs = [rr["conf0"], rr["conf1"], rr["conf2"], rr["conf_mz_multi"]]
        seqs = [rr["seq0"], rr["seq1"], rr["seq2"], rr["seq3"]]
        plot_trace_for_peptide_labeled(
            traces_not_dir,
            i + 1,
            False,
            seqs,
            flies,
            confs,
            title_extra=f"Δscore={rr['delta_total_score']:.3f}",
            max_label_len=args.trace_label_maxlen,
            c_idx=int(rr["mut_idx0"]),
            p_idx=int(rr["p_idx"]),
            m_idx=int(rr["m_idx"]),
            preP_idx=(
                int(rr["preP_idx"]) if pd.notna(rr.get("preP_idx", np.nan)) else None
            ),
        )

    # ---- README / diagnostics ----
    with open(os.path.join(args.outdir, "README.md"), "w") as fh:
        fh.write(
            "\n".join(
                [
                    "# Decoy generation + multi-source confusability (proteins + PEG)",
                    "",
                    "## Input filters applied (before C→P→M edits)",
                    f"- drop_ref_or_mut_C: {args.drop_ref_or_mut_C} (requires --mut-notation-col to evaluate mutant AA; ref AA uses mut_idx0)",
                    f"- drop_inputs_with_P: {args.drop_inputs_with_P}",
                    f"- drop_inputs_with_M: {args.drop_inputs_with_M}",
                    "",
                    "See `peptide_falloff_counts.csv` and `plots/peptide_falloff.png` for counts after each step.",
                    "",
                    "## Deterministic C→P→M construction",
                    "- C@mutation; P first with **GP » AP » [L/V/I/S/T]P » other**, hard-avoid RK-before-P (K/R immediately before P).",
                    "- If P is only possible with a K/R predecessor, the peptide is **dropped**. See `dropped_rk_only.csv`.",
                    "- M next given P; prefer Q→M; ≥1 away from C; enforce **P between C and M** (unless C at 0).",
                    "- Never change the last residue; optional hydrophobic C-term enforced at selection if `--enforce-hydrophobic-cterm 1`.",
                    "",
                    "## Scoring & selection",
                    f"- final_score = fly×{w['fly']:.2f} + hydro×{w['hydro']:.2f} + conf_mz×{w['conf_mz']:.2f} "
                    f"+ term×{w['term']:.2f} + peg×{w['peg']:.2f} + types×{w['types']:.2f}",
                    "- start_score uses the same weights but computed from the original sequence.",
                    f"- Greedy aug_score = final_score + diversity_bonus({args.diversity_bonus}) × novelty + "
                    f"improvement_bonus({args.improvement_bonus}) × normalized_rank_improvement",
                    "",
                    "## Hard constraints on selected decoys",
                    f"- Final confusability ≥ min_confusability_ratio({args.min_confusability_ratio:.2f}) × start confusability.",
                    f"- Final hydrophobic_fraction ≥ start + hydro_delta_min({args.hydro_delta_min:g}) when `--require-hydro-increase=1`.",
                    "",
                    "## Traces",
                    "- Each residue is colored by the stage at which it changed: orig (black), +C (blue), +P (orange), +M (green).",
                    "- Filenames include original and final sequences; both selected and non-selected have traces.",
                    "",
                    "## Plots in ./plots",
                    "- selected_final_score_decomposition.png",
                    "- density-aware scatter_*",
                    "- scatter_final_vs_start_score.png, scatter_end_vs_start_rank.png",
                    "- hist_rank_improvement_sel_vs_not.png",
                    "- box_delta_total_fly/conf/score.png",
                    "- seq_traces_selected/trace_*.png and seq_traces_not_selected/trace_*.png",
                    "- p_tier_counts.png, p_rk_filter_counts.png",
                    "- dist_to_C_and_P_hist.png",
                    "- delta_hydro_fly_P.png, delta_hydro_fly_M.png",
                    "- peptide_falloff.png",
                ]
            )
        )

    logx(
        f"[DONE] Wrote {len(sel)} selected / {len(full)} kept / {len(dropped)} dropped (RK-only P). Outdir: {args.outdir}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Decoy generation + multi-source confusability (proteins + PEG) + diversity-aware selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--in", dest="input_csv", required=True, help="Input CSV")
    ap.add_argument("--outdir", default="decoy_out6", help="Output directory")
    ap.add_argument("--N", type=int, default=48, help="Number of decoys to select")

    # Column mapping & index policy
    ap.add_argument(
        "--peptide-col", default="peptide", help="Column with peptide sequence (AAs)"
    )
    ap.add_argument(
        "--mut-idx-col",
        dest="mut_idx_col",
        default=None,
        help="Column with 0-based mutation index (mut_idx0)",
    )
    ap.add_argument(
        "--pos-col",
        dest="pos_col",
        default=None,
        help="Integer position column to convert to 0-based via --indexing",
    )
    ap.add_argument(
        "--mut-notation-col",
        dest="mut_notation_col",
        default=None,
        help="Mutation notation column (e.g., L858R, p.G12D)",
    )
    ap.add_argument(
        "--indexing",
        default="C1",
        choices=["0", "1", "C1"],
        help="How to interpret --pos-col / --mut-notation-col positions: '0'=0-based; '1' or 'C1'=1-based (subtract 1)",
    )
    ap.add_argument(
        "--require-mut-idx0",
        type=int,
        default=1,
        help="1=FAIL if a row lacks a valid mut_idx0 (default). 0=allow auto Cys placement.",
    )

    # Mandatory input filters (defaults ON)
    ap.add_argument(
        "--drop-ref-or-mut-C",
        type=int,
        default=1,
        help="Drop a row if the reference AA at mut_idx0 or the mutant AA from notation is 'C'.",
    )
    ap.add_argument(
        "--drop-inputs-with-P",
        type=int,
        default=1,
        help="Drop input peptides that contain 'P' anywhere.",
    )
    ap.add_argument(
        "--drop-inputs-with-M",
        type=int,
        default=1,
        help="Drop input peptides that contain 'M' anywhere.",
    )

    # Sets
    ap.add_argument(
        "--sets",
        default="albumin,keratins,proteases,mhc_hardware",
        help="Comma list of contaminant sets",
    )
    ap.add_argument(
        "--download-sets",
        dest="download_sets",
        action="store_true",
        help="Fetch UniProt FASTAs for default accessions",
    )
    ap.add_argument(
        "--sets-fasta-dir",
        default=None,
        help="Folder with <set>.fasta files to override defaults",
    )

    # Matching/RT
    ap.add_argument(
        "--ppm-tol",
        dest="ppm_tol",
        type=float,
        default=30.0,
        help="PPM tolerance for general m/z matching",
    )
    ap.add_argument(
        "--good-ppm-strict",
        type=float,
        default=10.0,
        help="Strict ppm for strong protein matches and contam categories",
    )
    ap.add_argument(
        "--fragment-kmin", type=int, default=2, help="Min b/y fragment length (k)"
    )
    ap.add_argument(
        "--fragment-kmax", type=int, default=7, help="Max b/y fragment length (k)"
    )
    ap.add_argument(
        "--full-mz-len-min", type=int, default=5, help="Contaminant window min length"
    )
    ap.add_argument(
        "--full-mz-len-max", type=int, default=15, help="Contaminant window max length"
    )
    ap.add_argument(
        "--rt-tolerance-min",
        type=float,
        default=1.0,
        help="RT co-elution tolerance (min)",
    )
    ap.add_argument(
        "--gradient-min",
        type=float,
        default=20.0,
        help="Gradient length used by RT surrogate (min)",
    )
    ap.add_argument(
        "--cys-mod",
        choices=["none", "carbamidomethyl"],
        default="carbamidomethyl",
        help="Fixed mod on Cys for contaminant windows",
    )
    ap.add_argument(
        "--charges",
        default="2,3",
        help="Precursor charge states to test (comma). Also for fragments unless --frag-charges is set.",
    )
    ap.add_argument(
        "--frag-charges",
        default=None,
        help="Fragment (b/y) charge states (comma). Default: same as --charges.",
    )
    ap.add_argument(
        "--strict-use-rt",
        action="store_true",
        help="Require RT-gated precursor matches when listing specific proteins",
    )
    ap.add_argument(
        "--frag-types-req",
        dest="frag_types_req",
        type=int,
        default=3,
        help="Minimum distinct (ion,k,z) fragment types to count a match (after RT gating if enabled).",
    )

    # Terminal k-mer
    ap.add_argument(
        "--term-k",
        type=int,
        default=4,
        help="k for terminal k-mer matching (I/L-collapsed if --xle-collapse 1)",
    )
    ap.add_argument(
        "--xle-collapse",
        dest="xle_collapse",
        type=int,
        default=1,
        help="Collapse I/L to J for terminal k-mer bank (1=yes, 0=no)",
    )

    # PEG config
    ap.add_argument("--peg-families", default="PEG,PPG,PTMEG,PDMS")
    ap.add_argument("--peg-endgroups", default="auto")
    ap.add_argument("--peg-adducts", default="H,Na,K,NH4")
    ap.add_argument("--peg-z", default="1,2,3")
    ap.add_argument("--peg-n-min", type=int, default=5)
    ap.add_argument("--peg-n-max", type=int, default=100)
    ap.add_argument("--peg-mz-min", type=float, default=200.0)
    ap.add_argument("--peg-mz-max", type=float, default=1500.0)
    ap.add_argument(
        "--peg-ppm-strict",
        type=float,
        default=10.0,
        help="≤ this ppm → peg_match_flag=1 and peg_conf=1−ppm/threshold",
    )

    # Selection & visualization tuning
    ap.add_argument(
        "--rank-weights",
        default="fly:0.50,hydro:0.20,conf_mz:0.18,term:0.06,peg:0.04,types:0.02",
        help="Comma spec: fly,hydro,conf_mz,term,peg,types (sum normalized internally)",
    )
    ap.add_argument(
        "--dedup-k", type=int, default=4, help="k-mer de-dup k (0 disables)"
    )
    ap.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.10,
        help="Novelty bonus during greedy selection",
    )
    ap.add_argument(
        "--improvement-bonus",
        type=float,
        default=0.15,
        help="Bonus for (normalized) rank improvement from original→final",
    )
    ap.add_argument(
        "--require-contam-match",
        type=int,
        default=1,
        help="Require ≥1 contaminant to select (1=yes, 0=no)",
    )
    ap.add_argument(
        "--enforce-hydrophobic-cterm",
        type=int,
        default=1,
        help="Require final decoy C-terminus to be hydrophobic (1=yes, 0=no)",
    )
    ap.add_argument(
        "--contam-match-min-score",
        dest="contam_match_min_score",
        type=float,
        default=0.45,
        help="Min per-set confusability to count as a contaminant hit (friendly label summary)",
    )
    ap.add_argument(
        "--dense-threshold",
        type=int,
        default=1500,
        help="Switch to hexbin background if non-selected points exceed this",
    )
    ap.add_argument(
        "--trace-label-maxlen",
        type=int,
        default=36,
        help="Max characters for sequence labels in trace plots",
    )

    # Hard selection constraints
    ap.add_argument(
        "--min-confusability-ratio",
        type=float,
        default=0.90,
        help="Require final conf_mz_multi ≥ this × start conf_mz_multi (e.g., 0.90 for ≤10% drop).",
    )
    ap.add_argument(
        "--require-hydro-increase",
        type=int,
        default=1,
        help="1 = require final hydrophobic_fraction ≥ start + hydro_delta_min; 0 = do not enforce.",
    )
    ap.add_argument(
        "--hydro-delta-min",
        type=float,
        default=1e-6,
        help="Minimum required increase in hydrophobic_fraction for selection when --require-hydro-increase=1.",
    )

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
