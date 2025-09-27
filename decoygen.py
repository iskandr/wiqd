#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peptide selector & decoy feature generator

(Albumin acquisition/validation + Cys-substitution-first + Proline-insertion decoys
 + progress bars + confusability + plots + explicit albumin fragment records
 + flexible column mapping + C-terminal indexing + mutation-notation parsing)

m/z assumptions:
  • All theoretical m/z values are for protonated species [M+nH]^n+ in positive-mode ESI,
    as expected for acidic LC mobile phases (e.g., ~0.1% formic acid). No adducts (Na/K/NH4+) are included.
    Fixed mods can be set (e.g., carbamidomethyl Cys).

Pipeline (concise):
  Base filter:
    • Drop rows where WT=='C' or MUT=='C'.
    • Drop rows whose ORIGINAL peptide contains 'P' or 'C' (we will introduce these explicitly).

  Decoy generation (Cys+Pro):
    • Resolve mutation index; build cys-only sequence by placing 'C' at the mutation index.
    • Insert a 'P' at a valid index not in {0,1,last,mut_idx}, preferring positions preceded by 'N' to form 'NP'.
      If multiple NP sites exist, take the first from the N-terminus; otherwise take the first valid non-NP site.
      If no valid site, label as dropped_no_valid_pro_insertion.
    • Form the final decoy sequence = Cys+Pro (length n+1).

  Selection & ranking (on DECoy, i.e., Cys+Pro):
    • Exclude if the decoy is an exact subsequence of albumin.
    • Stage 1: require Proline (always true after insertion) AND hydrophobic C-terminus (default AILMFWYV);
               keep top fraction by decoy_fly_score (default 0.5).
    • Stage 2: rank by
         (1) minimal precursor-level m/z ppm to ANY albumin 5–12mer (ascending),
         (2) maximum k-mer overlap anywhere to albumin (k in 2..11; descending),
         (3) decoy_fly_score (descending).
      Take top N.

Albumin “confusability” (0..1) per peptide:
  • Combines (a) precursor full m/z match to albumin 5–12mers, (b) b/y fragment m/z matches (2–11mers; charges up to +3 if plausible),
    and (c) b/y prefix/suffix k-mer sequence matches (3..n-1).

Author: ChatGPT (GPT-5 Pro)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import pandas as pd
import numpy as np
from pathlib import Path
from textwrap import wrap

# tqdm (progress bar) with graceful fallback
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

# Matplotlib for plots (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Albumin acquisition and verification ----------------

ALBUMIN_P02768_PREPRO_609 = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF"
    "EDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEP"
    "ERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLF"
    "FAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAV"
    "ARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLK"
    "ECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYAR"
    "RHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFE"
    "QLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVV"
    "LNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTL"
    "SEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLV"
    "AASQAALGL"
)

def _read_fasta_first_seq_text(fasta_path: str) -> str:
    with open(fasta_path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith(">")]
    return "".join(lines)

def _parse_fasta_text_to_seq(text: str) -> str:
    text = "".join([ln.strip() for ln in text.splitlines() if not ln.startswith(">")])
    seq = re.sub(r"[^A-Z]", "", text.upper())
    return seq

def fetch_albumin_fasta_from_uniprot(acc: str, timeout: float = 10.0) -> str:
    """Fetch FASTA from UniProt REST; returns sequence string (letters only)."""
    import urllib.request, urllib.error
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        return _parse_fasta_text_to_seq(data)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to fetch UniProt FASTA for {acc}: {e}")
        return ""

def get_albumin_sequence(args) -> str:
    """Acquire albumin based on args.albumin_source and validate length/epitopes."""
    seq = ""
    source_used = None

    if args.albumin_source in ("file", "auto") and args.albumin_fasta:
        try:
            seq = _read_fasta_first_seq_text(args.albumin_fasta).upper()
            source_used = f"file:{args.albumin_fasta}"
        except Exception as e:
            print(f"[WARN] Could not read albumin FASTA '{args.albumin_fasta}': {e}")

    if not seq and args.albumin_source in ("fetch", "auto"):
        for acc in args.albumin_acc.split(","):
            acc = acc.strip()
            if not acc:
                continue
            fetched = fetch_albumin_fasta_from_uniprot(acc)
            if fetched:
                seq = fetched
                source_used = f"uniprot:{acc}"
                break

    if not seq or args.albumin_source == "embedded":
        seq = ALBUMIN_P02768_PREPRO_609
        if not source_used:
            source_used = "embedded:P02768_prepro_609"

    print(f"[INFO] Albumin source: {source_used}; length={len(seq)} aa before any trimming.")
    seq = verify_and_prepare_albumin(seq, args)
    print(f"[INFO] Using albumin sequence length={len(seq)} aa for matching.")
    return seq

def verify_and_prepare_albumin(seq: str, args) -> str:
    expected_mode = args.albumin_expected  # 'prepro'|'mature'|'either'
    use_mode = args.albumin_use            # 'prepro'|'mature'|'auto'
    strict = args.strict_albumin_check

    L = len(seq)
    is_prepro = (L == 609)
    is_mature = (L == 585)
    if expected_mode == "prepro":
        ok_len = is_prepro
    elif expected_mode == "mature":
        ok_len = is_mature
    else:
        ok_len = is_prepro or is_mature

    if not ok_len:
        msg = f"[{'ERROR' if strict else 'WARN'}] Albumin length check failed: got {L}, expected 609 (prepro) or 585 (mature)."
        print(msg)
        if strict:
            raise RuntimeError(msg)

    default_epitopes = [
        "DAHKSEVAHRFKDLGEENFK",
        "KVPQVSTPTLVEVSR",
        "LVNEVTEFAK",
        "QTALVELVK",
        "YLYEIAR",
        "YICENQDSISSK",
    ]
    ep_list = []
    if args.use_default_epitopes:
        ep_list.extend(default_epitopes)
    if args.epitopes:
        ep_list.extend(args.epitopes)
    if args.epitopes_file:
        try:
            with open(args.epitopes_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip().upper()
                    if s and not s.startswith("#"):
                        ep_list.append(s)
        except Exception as e:
            print(f"[WARN] Could not read epitopes file '{args.epitopes_file}': {e}")

    seen = set()
    epitopes = []
    for e in ep_list:
        if e not in seen:
            epitopes.append(e); seen.add(e)

    failures = [ep for ep in epitopes if seq.find(ep) == -1]
    if failures:
        msg = f"[{'ERROR' if strict else 'WARN'}] Albumin epitope(s) not found: {', '.join(failures)}"
        print(msg)
        if strict:
            raise RuntimeError(msg)
    elif epitopes:
        print(f"[INFO] Verified {len(epitopes)} albumin epitope(s).")

    if use_mode == "mature":
        if L == 609:
            seq = seq[24:]
            print("[INFO] Trimmed prepro->mature: now length 585 aa.")
    elif use_mode == "auto":
        if expected_mode == "mature" and L == 609:
            seq = seq[24:]
            print("[INFO] Auto-trimmed to mature sequence (585 aa).")
    return seq

# ---------------- Mass / mz utilities ----------------

AA_MONO = {
    "A": 71.037113805, "R": 156.10111105, "N": 114.04292747, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.05857754, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.09496305,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.07931298, "Y": 163.063328575, "V": 99.068413945,
}
WATER_MONO = 18.010564684
PROTON_MONO = 1.007276466812

KD_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4,
    "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

def calc_peptide_mono_mass(seq: str, aa_masses: Dict[str, float], cys_fixed_mod: float = 0.0) -> float:
    m = 0.0
    for aa in seq:
        if aa not in aa_masses:
            raise ValueError(f"Unknown amino acid '{aa}' in sequence '{seq}'.")
        m += aa_masses[aa]
        if aa == "C" and cys_fixed_mod != 0.0:
            m += cys_fixed_mod
    return m + WATER_MONO

def mz_from_mass(neutral_mass: float, charge: int) -> float:
    if charge <= 0:
        raise ValueError("Charge must be positive integer.")
    return (neutral_mass + charge * PROTON_MONO) / charge

def allowed_fragment_max_charge(seq: str, hard_cap: int = 3) -> int:
    """
    Heuristic: max charge is limited by available basic sites + 1.
    Count R/K (+1 each), H (+0.5), add 1 for the terminus; cap at hard_cap (default 3).
    Ensure at least +1.
    """
    R = seq.count("R"); K = seq.count("K"); H = seq.count("H")
    basic = R + K + 0.5 * H
    max_z = int(max(1, min(hard_cap, 1 + round(basic))))
    return max_z

# ---------------- Indexing and mutation-notation parsing ----------------

def parse_mutation_notation(s: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Parse strings like 'GENE_A123B' or 'GENE-A123B' or 'GENE:A123B' to (wt, pos, mut, gene).
    Returns (wt, pos_int, mut, gene). If parse fails, returns (None, None, None, gene_guess).
    """
    if s is None: return (None, None, None, None)
    s = str(s).strip()
    if not s: return (None, None, None, None)
    import re as _re
    m = _re.search(r'([A-Za-z0-9_.-]+)[_:\-]([A-Za-z])(\d+)([A-Za-z])$', s)
    if m:
        gene = m.group(1)
        wt = m.group(2).upper()
        pos = int(m.group(3))
        mut = m.group(4).upper()
        return (wt, pos, mut, gene)
    m2 = _re.search(r'([A-Za-z])(\d+)([A-Za-z])$', s)
    if m2:
        wt = m2.group(1).upper()
        pos = int(m2.group(2))
        mut = m2.group(3).upper()
        pref = s[:m2.start()] if m2.start() > 0 else None
        if pref:
            pref = _re.sub(r'[_:\-]+$', '', pref)
        gene = pref
        return (wt, pos, mut, gene)
    return (None, None, None, None)

def detect_indexing_mode(peptide: str, pos: int, wt: str, mut: str, default_mode: str = "N1") -> Tuple[int, str]:
    L = len(peptide)
    def _idx_from_mode(position: int, mode: str) -> int:
        if mode in ("N0","0"):  # 0-based from N-terminus
            return position
        if mode in ("N1","1"):  # 1-based from N-terminus
            return position - 1
        if mode == "C1":        # 1-based from C-terminus (1 == last residue)
            return L - position
        if mode == "C0":        # 0-based from C-terminus (0 == last residue)
            return L - 1 - position
        raise ValueError(f"Unknown indexing mode: {mode}")

    modes_try = [default_mode] if default_mode != "auto" else ["N1","N0","C1","C0"]
    for mode in modes_try:
        idx = _idx_from_mode(pos, mode)
        if 0 <= idx < L:
            aa = peptide[idx]
            if aa == mut or aa == wt:
                return (idx, mode)
    idx = _idx_from_mode(pos, modes_try[0])
    if not (0 <= idx < L):
        raise ValueError(f"Mutation position {pos} with mode {modes_try[0]} is out of range for peptide length {L}.")
    return (idx, modes_try[0])

@dataclass
class FlankOverlapResult:
    left_len: int; right_len: int; albumin_center_idx: int
    albumin_left_seq: str; albumin_right_seq: str
    albumin_span_start: int; albumin_span_end: int

def best_flank_overlap_against_albumin(peptide: str, mut_idx0: int, albumin: str) -> FlankOverlapResult:
    left = peptide[:mut_idx0]
    right = peptide[mut_idx0 + 1:]
    best = FlankOverlapResult(0,0,-1,"","",-1,-1)
    N = len(albumin)
    for j in range(N):
        # left suffix
        max_l = 0; max_l_seq=""; max_l_start=j
        upper_l = min(len(left), j)
        for l in range(upper_l, 0, -1):
            if albumin[j-l:j] == left[-l:]:
                max_l=l; max_l_seq=left[-l:]; max_l_start=j-l; break
        # right prefix
        max_r = 0; max_r_seq=""; max_r_end=j+1
        upper_r = min(len(right), N-(j+1))
        for r in range(upper_r, 0, -1):
            if albumin[j+1:j+1+r] == right[:r]:
                max_r=r; max_r_seq=right[:r]; max_r_end=j+1+r; break
        if (max_l+max_r) > (best.left_len+best.right_len):
            best = FlankOverlapResult(max_l, max_r, j, max_l_seq, max_r_seq, max_l_start, max_r_end)
    return best

# ---------------- Matching metrics ----------------

def kmer_set(seq: str, k: int) -> set:
    if k <= 0 or k > len(seq):
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}

def max_kmer_overlap_any(seq: str, albumin: str, kmin: int, kmax: int) -> Dict[str, Optional[object]]:
    """Max k-mer overlap anywhere in seq vs albumin."""
    kmin = max(1, int(kmin)); kmax = max(kmin, int(kmax))
    best = {"max_k": 0, "example_kmer": None, "albumin_start0": None, "albumin_end0": None}
    for k in range(kmax, kmin - 1, -1):
        if k > len(seq): continue
        seen = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in seen: continue
            seen.add(kmer)
            j = albumin.find(kmer)
            if j != -1:
                best.update({"max_k": k, "example_kmer": kmer, "albumin_start0": j, "albumin_end0": j+k})
                return best
    return best

def max_kmer_overlap_b_y(seq: str, albumin: str, kmin: int, kmax: int) -> Dict[str, Optional[object]]:
    """Max k-mer overlap for b- (prefix) and y- (suffix) kmers separately."""
    kmin = max(1, int(kmin)); kmax = max(kmin, int(kmax))
    # b: prefixes
    best_b = {"b_max_k": 0, "b_example_kmer": None, "b_albumin_start0": None, "b_albumin_end0": None}
    for k in range(kmax, kmin - 1, -1):
        if k > len(seq): continue
        kmer = seq[:k]
        j = albumin.find(kmer)
        if j != -1:
            best_b.update({"b_max_k": k, "b_example_kmer": kmer, "b_albumin_start0": j, "b_albumin_end0": j+k})
            break
    # y: suffixes
    best_y = {"y_max_k": 0, "y_example_kmer": None, "y_albumin_start0": None, "y_albumin_end0": None}
    for k in range(kmax, kmin - 1, -1):
        if k > len(seq): continue
        kmer = seq[-k:]
        j = albumin.find(kmer)
        if j != -1:
            best_y.update({"y_max_k": k, "y_example_kmer": kmer, "y_albumin_start0": j, "y_albumin_end0": j+k})
            break
    best = {}
    best.update(best_b); best.update(best_y)
    return best

def min_ppm_between_query_and_albumin_subseqs(
    query_seq: str,
    albumin: str,
    charges: Iterable[int],
    aa_masses: Dict[str, float],
    cys_fixed_mod: float,
) -> Tuple[float, float, Optional[int], Optional[int], Optional[int], Optional[str]]:
    """
    Compute minimal ppm and Da error between a query peptide and any albumin subsequence of the SAME length,
    across the provided charge states.
    Returns (best_ppm, best_da, best_charge, best_start0, best_end0, best_seq).
    """
    L = len(query_seq)
    if L <= 0:
        return (float("inf"), float("inf"), None, None, None, None)
    q_m = calc_peptide_mono_mass(query_seq, aa_masses, cys_fixed_mod)
    q_mz = {z: mz_from_mass(q_m, z) for z in charges}

    best_ppm = float("inf"); best_da = float("inf")
    best_charge = None; best_start = None; best_end = None; best_seq = None

    N = len(albumin)
    if L > N:
        return (best_ppm, best_da, best_charge, best_start, best_end, best_seq)

    for start in range(0, N - L + 1):
        sub = albumin[start:start+L]
        sub_m = calc_peptide_mono_mass(sub, aa_masses, cys_fixed_mod)
        for z in charges:
            sub_mz = mz_from_mass(sub_m, z)
            da_err = abs(sub_mz - q_mz[z])
            ppm_err = (da_err / q_mz[z]) * 1e6
            if ppm_err < best_ppm:
                best_ppm = ppm_err; best_da = da_err
                best_charge = z; best_start = start; best_end = start+L; best_seq = sub

    return (best_ppm, best_da, best_charge, best_start, best_end, best_seq)

def best_ppm_cys_vs_albumin_fragments(
    cys_variant: str,
    albumin: str,
    charges: Iterable[int],
    aa_masses: Dict[str, float],
    cys_fixed_mod: float,
    frag_len_min: int,
    frag_len_max: int,
) -> Dict[str, Optional[float]]:
    """
    Minimal ppm between decoy full m/z and ANY albumin fragment (length in [frag_len_min, frag_len_max]).
    This allows different lengths between target and albumin fragment.
    """
    cys_m = calc_peptide_mono_mass(cys_variant, aa_masses, cys_fixed_mod)
    cys_mz = {z: mz_from_mass(cys_m, z) for z in charges}

    best = {"best_ppm": float("inf"), "best_da": float("inf"), "best_charge": None, "best_len": None,
            "best_seq": None, "best_start0": None, "best_end0": None}
    N = len(albumin)
    frag_len_min = max(1, int(frag_len_min)); frag_len_max = max(frag_len_min, int(frag_len_max))
    for L in range(frag_len_min, frag_len_max + 1):
        if L > N: break
        for start in range(0, N - L + 1):
            sub = albumin[start:start+L]
            sub_m = calc_peptide_mono_mass(sub, aa_masses, cys_fixed_mod)
            for z in charges:
                sub_mz = mz_from_mass(sub_m, z)
                da_err = abs(sub_mz - cys_mz[z])
                ppm_err = (da_err / cys_mz[z]) * 1e6
                if ppm_err < best["best_ppm"]:
                    best.update({"best_ppm": ppm_err, "best_da": da_err, "best_charge": z,
                                 "best_len": L, "best_seq": sub, "best_start0": start, "best_end0": start+L})
    return best

def best_ppm_b_y_fragments(
    cys_variant: str,
    albumin: str,
    aa_masses: Dict[str, float],
    cys_fixed_mod: float,
    by_frag_len_min: int,
    by_frag_len_max: int,
) -> Dict[str, Optional[float]]:
    """
    Minimal ppm for b- and y-fragments of the decoy (prefix/suffix of L in [by_frag_len_min, by_frag_len_max])
    vs albumin subsequences of the SAME length, across allowed charges up to +3 (heuristic).
    """
    by_frag_len_min = max(1, int(by_frag_len_min)); by_frag_len_max = max(by_frag_len_min, int(by_frag_len_max))
    out = {
        "b_best_ppm": float("inf"), "b_best_da": float("inf"), "b_best_len": None, "b_best_charge": None,
        "b_best_seq": None, "b_best_start0": None, "b_best_end0": None,
        "y_best_ppm": float("inf"), "y_best_da": float("inf"), "y_best_len": None, "y_best_charge": None,
        "y_best_seq": None, "y_best_start0": None, "y_best_end0": None,
    }
    # b-frags
    for L in range(by_frag_len_min, by_frag_len_max + 1):
        if L > len(cys_variant): break
        bseq = cys_variant[:L]
        zmax_b = allowed_fragment_max_charge(bseq, hard_cap=3)
        best_ppm_b, best_da_b, best_ch_b, best_s0_b, best_e0_b, best_sub_b = float('inf'), float('inf'), None, None, None, None
        for z in range(1, zmax_b+1):
            ppm, da, ch, s0, e0, sub = min_ppm_between_query_and_albumin_subseqs(
                bseq, albumin, [z], AA_MONO, cys_fixed_mod
            )
            if ppm < best_ppm_b:
                best_ppm_b, best_da_b, best_ch_b, best_s0_b, best_e0_b, best_sub_b = ppm, da, z, s0, e0, sub
        if best_ppm_b < out["b_best_ppm"]:
            out.update({"b_best_ppm": best_ppm_b, "b_best_da": best_da_b, "b_best_len": L, "b_best_charge": best_ch_b,
                        "b_best_seq": best_sub_b, "b_best_start0": best_s0_b, "b_best_end0": best_e0_b})
    # y-frags
    for L in range(by_frag_len_min, by_frag_len_max + 1):
        if L > len(cys_variant): break
        yseq = cys_variant[-L:]
        zmax_y = allowed_fragment_max_charge(yseq, hard_cap=3)
        best_ppm_y, best_da_y, best_ch_y, best_s0_y, best_e0_y, best_sub_y = float('inf'), float('inf'), None, None, None, None
        for z in range(1, zmax_y+1):
            ppm, da, ch, s0, e0, sub = min_ppm_between_query_and_albumin_subseqs(
                yseq, albumin, [z], AA_MONO, cys_fixed_mod
            )
            if ppm < best_ppm_y:
                best_ppm_y, best_da_y, best_ch_y, best_s0_y, best_e0_y, best_sub_y = ppm, da, z, s0, e0, sub
        if best_ppm_y < out["y_best_ppm"]:
            out.update({"y_best_ppm": best_ppm_y, "y_best_da": best_da_y, "y_best_len": L, "y_best_charge": best_ch_y,
                        "y_best_seq": best_sub_y, "y_best_start0": best_s0_y, "y_best_end0": best_e0_y})
    return out

# ---------------- Flyability ----------------

def triangular_len_score(L: int, zero_min: int = 7, peak_min: int = 9, peak_max: int = 11, zero_max: int = 20) -> float:
    if L <= zero_min: return 0.0
    if L >= zero_max: return 0.0
    if L < peak_min: return (L - zero_min) / max(1, (peak_min - zero_min))
    if L <= peak_max: return 1.0
    return max(0.0, (zero_max - L) / max(1, (zero_max - peak_max)))

def compute_flyability_components(seq: str) -> Dict[str, float]:
    L = len(seq)
    if L == 0: return {"charge_norm":0.0,"surface_norm":0.0,"aromatic_norm":0.0,"len_norm":0.0}
    counts = {aa:0 for aa in AA_MONO.keys()}
    for a in seq:
        if a in counts: counts[a]+=1
    R,K,H,D,E = counts.get("R",0), counts.get("K",0), counts.get("H",0), counts.get("D",0), counts.get("E",0)
    charge_sites = 1.0 + R + K + 0.7*H
    acid_penalty = 0.1*(D+E)
    charge_norm = 1.0 - np.exp(-max(0.0, charge_sites - acid_penalty)/2.0)
    gravy = np.mean([KD_HYDRO.get(a,0.0) for a in seq])
    surface_norm = np.clip((gravy + 4.5)/9.0, 0.0, 1.0)
    W,Y,F = counts.get("W",0), counts.get("Y",0), counts.get("F",0)
    aromatic_norm = np.clip((1.0*W + 0.7*Y + 0.5*F)/max(1.0, L/2.0), 0.0, 1.0)
    len_norm = triangular_len_score(L)
    return {"charge_norm":float(charge_norm), "surface_norm":float(surface_norm),
            "aromatic_norm":float(aromatic_norm), "len_norm":float(len_norm)}

def combine_flyability_score(components: Dict[str, float], fly_weights: Dict[str, float]) -> float:
    w = {"charge": fly_weights.get("charge",0.5), "surface": fly_weights.get("surface",0.35),
         "len": fly_weights.get("len",0.1), "aromatic": fly_weights.get("aromatic",0.05)}
    Wsum = max(1e-9, sum(w.values()))
    for k in w: w[k] /= Wsum
    c = components
    fly = w["charge"]*c["charge_norm"] + w["surface"]*c["surface_norm"] + w["len"]*c["len_norm"] + w["aromatic"]*c["aromatic_norm"]
    return float(np.clip(fly, 0.0, 1.0))

# ---------------- Confusability ----------------

def normalize_ppm_to_score(ppm: float, ppm_cap: float) -> float:
    """Map ppm to [0,1] score where 0 ~ poor match (>>cap), 1 ~ perfect (0 ppm)."""
    if ppm is None or not np.isfinite(ppm): return 0.0
    ppm = float(ppm)
    return max(0.0, 1.0 - min(ppm, ppm_cap)/ppm_cap)

def compute_confusability(
    full_ppm: float,
    b_ppm: float,
    y_ppm: float,
    b_kmax: int,
    y_kmax: int,
    kmax: int,
    ppm_cap: float,
    weights: Dict[str, float],
) -> float:
    # Normalize
    full_mz_s = normalize_ppm_to_score(full_ppm, ppm_cap)
    b_mz_s = normalize_ppm_to_score(b_ppm, ppm_cap)
    y_mz_s = normalize_ppm_to_score(y_ppm, ppm_cap)
    denom = max(1, kmax)
    b_k_s = np.clip((float(b_kmax) / denom), 0.0, 1.0)
    y_k_s = np.clip((float(y_kmax) / denom), 0.0, 1.0)

    # Weights (sum to 1)
    w = {
        "full_mz": weights.get("full_mz", 0.4),
        "b_mz": weights.get("b_mz", 0.2),
        "y_mz": weights.get("y_mz", 0.2),
        "b_kmer": weights.get("b_kmer", 0.1),
        "y_kmer": weights.get("y_kmer", 0.1),
    }
    Wsum = max(1e-9, sum(w.values()))
    for k in w: w[k] /= Wsum

    return float(
        w["full_mz"]*full_mz_s +
        w["b_mz"]*b_mz_s +
        w["y_mz"]*y_mz_s +
        w["b_kmer"]*b_k_s +
        w["y_kmer"]*y_k_s
    )

# ---------------- Proline insertion ----------------

def choose_proline_insertion_index(seq: str, mut_idx0: int) -> Tuple[Optional[int], bool]:
    """
    Choose an insertion index for 'P' subject to constraints:
      - not at indices 0 or 1 (avoid first two positions),
      - not at the last position (no appending at end),
      - not at the mutant residue index (mut_idx0) of the *cys-only* sequence.
    Preference: choose an index i where seq[i-1] == 'N' (to create 'NP').
    If multiple NP sites exist, choose the first from N-terminus.
    If none, choose the first valid non-NP site scanning from N-terminus.
    Returns (index, is_np). If no valid site found, returns (None, False).
    """
    L = len(seq)
    if L < 4:
        return (None, False)
    valid = []
    for i in range(L):  # insertion before seq[i]
        if i in (0, 1):  # avoid first two positions
            continue
        if i >= L:       # disallow appending at the very end
            continue
        if i == mut_idx0:
            continue
        valid.append(i)
    for i in valid:
        if i-1 >= 0 and seq[i-1] == "N":
            return (i, True)
    if valid:
        return (valid[0], False)
    return (None, False)

# ---------------- CLI ----------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cys+Pro decoy selector vs albumin with progress, plots, and confusability.")
    # Albumin source/validation
    p.add_argument("--albumin-fasta", required=False, default=None)
    p.add_argument("--albumin-source", choices=["auto","file","fetch","embedded"], default="auto")
    p.add_argument("--albumin-acc", default="P02768")
    p.add_argument("--albumin-expected", choices=["prepro","mature","either"], default="either")
    p.add_argument("--albumin-use", choices=["prepro","mature","auto"], default="prepro")
    p.add_argument("--strict-albumin-check", action="store_true", default=True)
    p.add_argument("--no-strict-albumin-check", action="store_false", dest="strict_albumin_check")
    p.add_argument("--epitope", dest="epitopes", action="append", default=None)
    p.add_argument("--epitopes-file", default=None)
    p.add_argument("--use-default-epitopes", action="store_true", default=True)
    p.add_argument("--no-use-default-epitopes", action="store_false", dest="use_default_epitopes")

    # IO + columns
    p.add_argument("--input", required=True, help="Input CSV/TSV with columns: peptide, mutation_position, wt, mut (configurable).")
    p.add_argument("--peptide-col", default="peptide", help="Column containing peptide sequence (e.g., 'epitope').")
    p.add_argument("--pos-col", default="mutation_position", help="Column with position of mutation within peptide (e.g., 'MT_pos').")
    p.add_argument("--wt-col", default="wt")
    p.add_argument("--mut-col", default="mut")
    p.add_argument("--mut-notation-col", default=None,
                   help="Optional column (e.g., Gene_AA_Change) with pattern like GENE_A123B to derive WT/pos/MUT if explicit cols are missing.")
    p.add_argument("--indexing", choices=["auto","0","1","N0","N1","C0","C1"], default="auto",
                   help="How to interpret mutation position: N0/N1 (from N-term), C0/C1 (from C-term), or auto. '0'='N0', '1'='N1'.")

    # Mass/charges and outputs
    p.add_argument("--charges", default="2,3")
    p.add_argument("--cys-mod", choices=["none","carbamidomethyl"], default="none")
    p.add_argument("--output-prefix", default="peptide_selection")
    p.add_argument("--outdir", default="peptide_selection_out", help="Directory to contain CSVs and plots (will be created).")
    p.add_argument("--N", type=int, default=10, help="Final top-N.")

    # Stage 1 policy (on decoy)
    p.add_argument("--hydrophobic", default="AILMFWYV")
    p.add_argument("--stage1-require-proline", action="store_true", default=True)
    p.add_argument("--no-stage1-require-proline", action="store_false", dest="stage1_require_proline")
    p.add_argument("--stage1-enforce-hydrophobic", action="store_true", default=True)
    p.add_argument("--no-stage1-enforce-hydrophobic", action="store_false", dest="stage1_enforce_hydrophobic")
    p.add_argument("--stage1-keep-frac", type=float, default=0.5)

    # Stage 2 knobs (on decoy)
    p.add_argument("--frag-len-min", type=int, default=5, help="Precursor-vs-albumin fragment min length (default 5).")
    p.add_argument("--frag-len-max", type=int, default=12, help="Precursor-vs-albumin fragment max length (default 12).")
    p.add_argument("--by-frag-len-min", type=int, default=2, help="b/y fragment m/z scan min length (default 2).")
    p.add_argument("--by-frag-len-max", type=int, default=11, help="b/y fragment m/z scan max length (default 11).")
    p.add_argument("--cys-kmer-min", type=int, default=2)
    p.add_argument("--cys-kmer-max", type=int, default=11)

    # Fly weights (for decoy)
    p.add_argument("--fly-weights", default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05")

    # Confusability synthesis
    p.add_argument("--confusability-weights", default="full_mz:0.4,b_mz:0.2,y_mz:0.2,b_kmer:0.1,y_kmer:0.1",
                   help="Weights (sum ~1) for confusability: full_mz,b_mz,y_mz,b_kmer,y_kmer")
    p.add_argument("--ppm-cap", type=float, default=10.0, help="Cap for mapping ppm to 0..1 score (0 ppm->1.0; cap->0.0).")

    # Legacy (not used in selection; kept for reference)
    p.add_argument("--weights", default="overlap:1.0,hydroCTerm:3.0,hasP:1.5,hasNP:1.5,mz:2.0,fly:3.0")
    return p.parse_args(argv)

def parse_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in s.split(","):
        token = token.strip()
        if not token: continue
        if ":" not in token: raise ValueError(f"Bad weight token '{token}'. Use key:value format.")
        k, v = token.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out

# ---------------- Sequence transitions plotting helper ----------------

def plot_sequence_transitions(rows_df: "pd.DataFrame", out_path: Path, max_peptides: int = 25) -> None:
    """
    Draw a single-axes panel showing up to max_peptides peptides.
    ORIGINAL (mutant residue highlighted) → CYS-ONLY (C highlighted) → DECOY (Cys+Pro; P insertion highlighted, C also highlighted)
    with arrows from ORIGINAL→CYS and CYS→DECOY pointing at the modified positions.
    Colors: mutant residue (orange), cysteine (red), proline (blue).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if rows_df is None or rows_df.empty:
        return

    df = rows_df.head(max_peptides).copy()
    plt.figure(figsize=(14, max(6, 0.5 + 0.8*len(df))))
    ax = plt.gca()
    ax.set_axis_off()

    y_gap = 0.9
    y0 = 0.95
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)

    col_mut = "#d95f02"  # orange
    col_cys = "#d62728"  # red
    col_pro = "#1f77b4"  # blue
    default_col = "#000000"

    def draw_seq(seq, y, highlight_positions, x_start=2):
        for i, ch in enumerate(seq):
            x = x_start + i*0.5
            color = highlight_positions.get(i, default_col)
            ax.text(x, y, ch, fontsize=11, family="DejaVu Sans Mono", ha="left", va="center", color=color)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        pep = getattr(row, "peptide")
        mut_idx0 = int(getattr(row, "mut_idx0"))
        cys_seq = getattr(row, "cys_only_seq")
        decoy_seq = getattr(row, "decoy_seq")
        pro_idx0 = getattr(row, "pro_insertion_index0", None)
        if pro_idx0 is None:
            continue
        c_pos_decoy = mut_idx0 + (1 if pro_idx0 <= mut_idx0 else 0)

        base_y = y0 - row_idx * y_gap
        y_orig = base_y
        y_cys  = base_y - 0.25
        y_dec  = base_y - 0.50

        # ORIGINAL
        hi_orig = {mut_idx0: col_mut}
        draw_seq(pep, y_orig, hi_orig)

        # CYS-only
        hi_cys = {mut_idx0: col_cys}
        draw_seq(cys_seq, y_cys, hi_cys)

        # DECOY
        hi_dec = {}
        if 0 <= pro_idx0 < len(decoy_seq):
            hi_dec[pro_idx0] = col_pro
        if 0 <= c_pos_decoy < len(decoy_seq) and c_pos_decoy != pro_idx0:
            hi_dec[c_pos_decoy] = col_cys
        draw_seq(decoy_seq, y_dec, hi_dec)

        ax.text(0.5, y_orig, "ORIG", fontsize=10, ha="left", va="center")
        ax.text(0.5, y_cys,  "CYS",  fontsize=10, ha="left", va="center")
        ax.text(0.5, y_dec,  "CYS+P",fontsize=10, ha="left", va="center")

        x_start = 2
        x_mut  = x_start + mut_idx0 * 0.5
        x_cys  = x_start + mut_idx0 * 0.5
        ax.annotate("", xy=(x_cys, y_cys+0.04), xytext=(x_mut, y_orig-0.04),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
        x_pro  = x_start + pro_idx0 * 0.5
        ax.annotate("", xy=(x_pro, y_dec+0.04), xytext=(x_cys, y_cys-0.04),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
        ax.hlines(y_dec - 0.12, 0, 80, colors="#DDDDDD", linestyles="-", linewidth=0.6)

    ax.set_title("Sequence transitions: mutant (orange) → cysteine (red) → proline (blue)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------- Main ----------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Ensure output directory exists
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Acquire + validate albumin sequence
    albumin = get_albumin_sequence(args)
    if not albumin or len(albumin) < 50:
        raise RuntimeError("Albumin acquisition failed or sequence too short.")

    # Load input
    if args.input.lower().endswith(".tsv"):
        df_in = pd.read_csv(args.input, sep="\t", dtype=str)
    else:
        df_in = pd.read_csv(args.input, dtype=str)

    df_in["_row_id"] = np.arange(len(df_in))

    # Normalize
    required_cols = [args.peptide_col, args.pos_col]  # wt/mut may be parsed from mut-notation
    for c in required_cols:
        if c not in df_in.columns:
            raise KeyError(f"Missing required column '{c}'. Present: {list(df_in.columns)}")
    df_in["_peptide"] = df_in[args.peptide_col].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)

    # WT/MUT columns if present
    have_wt = args.wt_col in df_in.columns
    have_mut = args.mut_col in df_in.columns
    if have_wt:
        df_in["_wt"] = df_in[args.wt_col].astype(str).str.upper().str.strip()
    else:
        df_in["_wt"] = np.nan
    if have_mut:
        df_in["_mut"] = df_in[args.mut_col].astype(str).str.upper().str.strip()
    else:
        df_in["_mut"] = np.nan

    # Position
    def safe_int(x):
        try: return int(float(str(x).strip()))
        except Exception: return np.nan
    df_in["_pos_int"] = df_in[args.pos_col].apply(safe_int)

    # Optionally parse WT/pos/MUT from a combined mutation-notation column
    if args.mut_notation_col and args.mut_notation_col in df_in.columns:
        def _parse_row(row):
            wt, posi, mut, gene = parse_mutation_notation(row[args.mut_notation_col])
            return pd.Series({"_wt_parsed": wt, "_pos_parsed": posi, "_mut_parsed": mut, "_gene_parsed": gene})
        parsed = df_in.apply(_parse_row, axis=1)
        for col in ["_wt_parsed","_pos_parsed","_mut_parsed","_gene_parsed"]:
            df_in[col] = parsed[col]
        if df_in["_wt"].isna().all():
            df_in["_wt"] = df_in["_wt_parsed"]
        if df_in["_mut"].isna().all():
            df_in["_mut"] = df_in["_mut_parsed"]
        # Do not override provided pos col unless it's missing or NaN
        if df_in["_pos_int"].isna().all():
            df_in["_pos_int"] = df_in["_pos_parsed"].astype("float").astype("Int64")
        if "_gene_parsed" in df_in.columns:
            df_in["_gene_from_notation"] = df_in["_gene_parsed"]

    if df_in["_pos_int"].isna().any():
        bad = df_in[df_in["_pos_int"].isna()]
        raise ValueError(f"Non-integer mutation positions encountered:\n{bad[[args.pos_col]].to_string(index=False)}")

    # Charges
    charges = sorted({int(x.strip()) for x in args.charges.split(",") if x.strip()})
    if not charges: raise ValueError("Provide at least one positive charge via --charges.")
    for z in charges:
        if z <= 0: raise ValueError("Charges must be positive integers.")

    # Cys fixed mod
    cys_fixed_mod = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0
    hydro_set = set(args.hydrophobic.upper())
    fly_weights = parse_weights(args.fly_weights)
    legacy_weights = parse_weights(args.weights)  # reference only
    conf_w = parse_weights(args.confusability_weights)

    # Base filter: drop WT=='C' or MUT=='C'; drop peptides containing P or C in ORIGINAL
    total = len(df_in)
    mask_wtmutC = (df_in["_wt"] == "C") | (df_in["_mut"] == "C")
    mask_hasP = df_in["_peptide"].str.contains("P", regex=False)
    mask_hasC = df_in["_peptide"].str.contains("C", regex=False)

    base_keep_mask = (~mask_wtmutC) & (~mask_hasP) & (~mask_hasC)
    base_kept = df_in[base_keep_mask].copy()
    base_dropped = df_in[~base_keep_mask].copy()

    base_dropped["stage_label"] = None
    base_dropped.loc[mask_wtmutC & ~base_keep_mask, "stage_label"] = "dropped_base_wt_or_mut_C"
    base_dropped.loc[(~mask_wtmutC) & mask_hasP & ~mask_hasC, "stage_label"] = "dropped_base_has_proline"
    base_dropped.loc[(~mask_wtmutC) & (~mask_hasP) & mask_hasC, "stage_label"] = "dropped_base_has_cysteine"
    base_dropped.loc[(~mask_wtmutC) & mask_hasP & mask_hasC, "stage_label"] = "dropped_base_has_proline_and_cysteine"

    print(f"[INFO] Base filter:")
    print(f"       • dropped WT/MUT==C : {int(mask_wtmutC.sum())}")
    print(f"       • dropped contains P: {int(mask_hasP.sum())}")
    print(f"       • dropped contains C: {int(mask_hasC.sum())}")
    print(f"       • remaining          : {len(base_kept)} of {total}")

    # Compute records for survivors
    records: List[Dict[str, object]] = []

    for row in tqdm(list(base_kept.itertuples(index=False)), total=len(base_kept), desc="Computing features"):
        pep = row._peptide; wt = str(row._wt) if pd.notna(row._wt) else None; mut = str(row._mut) if pd.notna(row._mut) else None
        pos = int(row._pos_int); row_id = row._row_id

        # Resolve mutation index (relative to original peptide)
        if args.indexing == "auto":
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode="N1")
        elif args.indexing in ("0","N0"):
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode="N0")
        elif args.indexing in ("1","N1"):
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode="N1")
        elif args.indexing == "C1":
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode="C1")
        elif args.indexing == "C0":
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode="C0")
        else:
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt or "", mut or "", default_mode=str(args.indexing))
        if not (0 <= mut_idx0 < len(pep)):
            raise ValueError(f"Resolved mutation index out of range for peptide '{pep}', pos={pos}")

        # Cys-only: replace mutant residue with 'C'
        pep_chars = list(pep); pep_chars[mut_idx0] = "C"; cys_seq = "".join(pep_chars)

        # Pro insertion on cys-only
        pro_idx, used_np = choose_proline_insertion_index(cys_seq, mut_idx0)
        if pro_idx is None:
            can_insert_pro = False
            decoy_seq = cys_seq  # placeholder
        else:
            can_insert_pro = True
            decoy_seq = cys_seq[:pro_idx] + "P" + cys_seq[pro_idx:]

        # Flags on decoy
        decoy_has_P = ("P" in decoy_seq)
        decoy_has_NP = any(decoy_seq[i-1:i+1] == "NP" for i in range(1, len(decoy_seq)))
        decoy_hydro_last = (len(decoy_seq) > 0 and decoy_seq[-1] in hydro_set)

        # Exclude if decoy is exact subsequence of albumin
        decoy_exact_in_albumin = (albumin.find(decoy_seq) != -1)

        # Overlap around mutation (for reference)
        overlap = best_flank_overlap_against_albumin(pep, mut_idx0, albumin)
        overlap_sum = overlap.left_len + overlap.right_len

        # Masses & m/z
        pep_mass = calc_peptide_mono_mass(pep, AA_MONO, cys_fixed_mod=cys_fixed_mod)
        pep_mz = {z: mz_from_mass(pep_mass, z) for z in charges}
        cys_mass = calc_peptide_mono_mass(cys_seq, AA_MONO, cys_fixed_mod=cys_fixed_mod)
        cys_mz = {z: mz_from_mass(cys_mass, z) for z in charges}
        decoy_mass = calc_peptide_mono_mass(decoy_seq, AA_MONO, cys_fixed_mod=cys_fixed_mod)
        decoy_mz = {z: mz_from_mass(decoy_mass, z) for z in charges}

        # Flyability
        decoy_fly_comp = compute_flyability_components(decoy_seq)
        decoy_fly = combine_flyability_score(decoy_fly_comp, fly_weights)
        cys_fly_comp = compute_flyability_components(cys_seq)
        cys_fly = combine_flyability_score(cys_fly_comp, fly_weights)
        prec_fly_comp = compute_flyability_components(pep)
        prec_fly = combine_flyability_score(prec_fly_comp, fly_weights)

        # Stage 2 metrics on decoy
        frag_best = best_ppm_cys_vs_albumin_fragments(
            cys_variant=decoy_seq, albumin=albumin, charges=charges,
            aa_masses=AA_MONO, cys_fixed_mod=cys_fixed_mod,
            frag_len_min=int(args.frag_len_min), frag_len_max=int(args.frag_len_max),
        )
        any_k = max_kmer_overlap_any(decoy_seq, albumin, kmin=int(args.cys_kmer_min), kmax=int(args.cys_kmer_max))
        by_k = max_kmer_overlap_b_y(decoy_seq, albumin, kmin=int(args.cys_kmer_min), kmax=int(args.cys_kmer_max))
        by_ppm = best_ppm_b_y_fragments(
            cys_variant=decoy_seq, albumin=albumin, aa_masses=AA_MONO, cys_fixed_mod=cys_fixed_mod,
            by_frag_len_min=int(args.by_frag_len_min), by_frag_len_max=int(args.by_frag_len_max),
        )

        # Confusability
        conf_score = compute_confusability(
            full_ppm=frag_best["best_ppm"],
            b_ppm=by_ppm["b_best_ppm"],
            y_ppm=by_ppm["y_best_ppm"],
            b_kmax=by_k["b_max_k"],
            y_kmax=by_k["y_max_k"],
            kmax=int(args.cys_kmer_max),
            ppm_cap=float(args.ppm_cap),
            weights=conf_w,
        )

        rec: Dict[str, object] = {}
        # passthrough original columns
        for c in df_in.columns:
            rec[c] = getattr(row, c)

        # core fields
        rec.update({
            "row_id": row_id,
            "peptide": pep, "mut_idx0": mut_idx0, "indexing_used": mode_used, "wt": wt, "mut": mut,
            "cys_only_seq": cys_seq,
            "decoy_seq": decoy_seq,
            "pro_insertion_index0": pro_idx if can_insert_pro else None,
            "pro_insertion_np": bool(used_np) if can_insert_pro else False,
            "decoy_exact_in_albumin": decoy_exact_in_albumin,
            "can_insert_pro": can_insert_pro,
            "length_precursor": len(pep), "length_cys_only": len(cys_seq), "length_decoy": len(decoy_seq),
            "overlap_left_len": overlap.left_len, "overlap_right_len": overlap.right_len, "overlap_sum": overlap_sum,
        })

        # stage1 flags (explicit for audit)
        rec.update({
            "decoy_stage1_has_P": decoy_has_P,
            "decoy_stage1_hydrophobic_last": decoy_hydro_last,
            "decoy_has_NP": decoy_has_NP,
        })

        # fly features
        rec.update({
            "decoy_fly_charge_norm": decoy_fly_comp["charge_norm"],
            "decoy_fly_surface_norm": decoy_fly_comp["surface_norm"],
            "decoy_fly_aromatic_norm": decoy_fly_comp["aromatic_norm"],
            "decoy_fly_len_norm": decoy_fly_comp["len_norm"],
            "decoy_fly_score": decoy_fly,
            "cys_only_fly_score": cys_fly,
            "prec_fly_score": prec_fly,
        })

        # Mass/mz
        rec["pep_mono_mass"] = pep_mass; rec["cys_only_mono_mass"] = cys_mass; rec["decoy_mono_mass"] = decoy_mass
        for z in charges:
            rec[f"pep_mz_z{z}"] = pep_mz[z]
            rec[f"cys_only_mz_z{z}"] = cys_mz[z]
            rec[f"decoy_mz_z{z}"] = decoy_mz[z]

        # Stage2 metrics & albumin fragment records
        rec.update({
            "stage2_full_best_ppm": (None if not np.isfinite(frag_best["best_ppm"]) else float(frag_best["best_ppm"])),
            "stage2_full_best_da": (None if not np.isfinite(frag_best["best_da"]) else float(frag_best["best_da"])),
            "stage2_full_best_charge": frag_best["best_charge"],
            "stage2_full_best_len": frag_best["best_len"],
            "stage2_full_best_seq": frag_best["best_seq"],
            "stage2_full_best_start0": frag_best["best_start0"],
            "stage2_full_best_end0": frag_best["best_end0"],

            "stage2_any_max_k": any_k["max_k"],
            "stage2_any_example_kmer": any_k["example_kmer"],
            "stage2_any_albumin_start0": any_k["albumin_start0"],
            "stage2_any_albumin_end0": any_k["albumin_end0"],

            "stage2_b_best_ppm": (None if not np.isfinite(by_ppm["b_best_ppm"]) else float(by_ppm["b_best_ppm"])),
            "stage2_b_best_len": by_ppm["b_best_len"],
            "stage2_b_best_charge": by_ppm["b_best_charge"],
            "stage2_b_best_seq": by_ppm["b_best_seq"],
            "stage2_b_best_start0": by_ppm["b_best_start0"],
            "stage2_b_best_end0": by_ppm["b_best_end0"],

            "stage2_y_best_ppm": (None if not np.isfinite(by_ppm["y_best_ppm"]) else float(by_ppm["y_best_ppm"])),
            "stage2_y_best_len": by_ppm["y_best_len"],
            "stage2_y_best_charge": by_ppm["y_best_charge"],
            "stage2_y_best_seq": by_ppm["y_best_seq"],
            "stage2_y_best_start0": by_ppm["y_best_start0"],
            "stage2_y_best_end0": by_ppm["y_best_end0"],
        })

        # b/y sequence matches as compact lists (k:pos:kmer for k=3..n-1)
        b_hits = []; y_hits = []
        Ld = len(decoy_seq)
        for k in range(3, max(3, Ld)):
            b = decoy_seq[:k]; jb = albumin.find(b); 
            if jb != -1: b_hits.append(f"{k}:{jb}:{b}")
            y = decoy_seq[-k:]; jy = albumin.find(y);
            if jy != -1: y_hits.append(f"{k}:{jy}:{y}")
        rec["b_seq_hits"] = ";".join(b_hits)
        rec["y_seq_hits"] = ";".join(y_hits)

        rec["albumin_confusability"] = conf_score

        records.append(rec)

    out_df = pd.DataFrame.from_records(records)

    # Stage labels initialization
    out_df["stage_label"] = "after_base"

    # Label rows where proline insertion was not possible
    no_pro_mask = (out_df["can_insert_pro"] == False)
    out_df.loc[no_pro_mask, "stage_label"] = "dropped_no_valid_pro_insertion"
    print(f"[INFO] Proline insertion feasible: {int((~no_pro_mask).sum())} / {len(out_df)}; dropped {int(no_pro_mask.sum())}.")

    # Drop DECoy exact-in-albumin
    exact_mask = out_df["decoy_exact_in_albumin"] == True
    n_exact = int(exact_mask.sum())
    print(f"[INFO] Excluding decoys that are exact albumin subsequences: {n_exact}. Remaining: {len(out_df) - n_exact}.")
    out_df.loc[exact_mask, "stage_label"] = "dropped_decoy_exact_albumin"

    # Stage 1 rule mask (on decoy) among those not failed earlier
    s1_rule_mask = (~exact_mask) & (~no_pro_mask)
    if args.stage1_require_proline:
        s1_rule_mask = s1_rule_mask & (out_df["decoy_stage1_has_P"] if "decoy_stage1_has_P" in out_df.columns else True)
    if args.stage1_enforce_hydrophobic:
        s1_rule_mask = s1_rule_mask & (out_df["decoy_seq"].str[-1].isin(list(hydro_set)))

    fail_s1_rule = (~s1_rule_mask) & (~exact_mask)
    out_df.loc[fail_s1_rule, "stage_label"] = "dropped_stage1_rule"
    n_after_s1_rules = int(s1_rule_mask.sum())
    print(f"[INFO] Stage1 rule pass: {n_after_s1_rules} / {len(out_df)}.")

    # Stage1 keep fraction by decoy_fly_score
    s1_pool = out_df[s1_rule_mask].copy().sort_values(by=["decoy_fly_score"], ascending=False, kind="mergesort")
    frac = float(args.stage1_keep_frac); 
    if not (0.0 < frac <= 1.0): frac = 0.5
    k_keep = max(1, int(np.ceil(frac * len(s1_pool)))) if len(s1_pool) > 0 else 0
    s1_kept = s1_pool.head(k_keep).copy()
    out_df["stage1_selected"] = out_df.index.isin(s1_kept.index)

    dropped_frac_mask = s1_rule_mask & (~out_df["stage1_selected"])
    out_df.loc[dropped_frac_mask, "stage_label"] = "dropped_stage1_frac"
    print(f"[INFO] Stage1 keep-frac: kept {len(s1_kept)} ({(len(s1_kept)/max(1,len(s1_pool))*100):.1f}%) of {len(s1_pool)} passing rules.")

    # Stage 2 final ranking: among s1_kept
    s2 = s1_kept.copy()
    s2["_ppm"] = s2["stage2_full_best_ppm"].fillna(np.inf)
    s2 = s2.sort_values(by=["_ppm", "stage2_any_max_k", "decoy_fly_score"],
                        ascending=[True, False, False], kind="mergesort").drop(columns=["_ppm"])

    # Final top N
    N = int(args.N)
    sel = s2.head(N).copy()
    sel_ids = set(sel["row_id"].tolist())

    # Label selected vs not_selected_stage2
    out_df.loc[out_df["row_id"].isin(sel_ids), "stage_label"] = "selected"
    not_selected_s2_mask = (out_df["stage1_selected"]) & (~out_df["row_id"].isin(sel_ids))
    out_df.loc[not_selected_s2_mask, "stage_label"] = "not_selected_stage2"

    # Build "all with stage labels": add base_dropped with minimal columns
    base_dropped_copy = base_dropped.copy()
    for col in set(out_df.columns) - set(base_dropped_copy.columns):
        base_dropped_copy[col] = np.nan
    all_with_labels = pd.concat([out_df, base_dropped_copy[out_df.columns]], ignore_index=True)

    # ------------------------
    # Write outputs
    # ------------------------
    prefix = args.output_prefix
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    out_df.to_csv(outdir / f"{prefix}_decoy_features.csv", index=False)
    sel.to_csv(outdir / f"{prefix}_selected_top_N.csv", index=False)
    all_with_labels.to_csv(outdir / f"{prefix}_all_with_stage_labels.csv", index=False)

    # Stage counts
    counts = all_with_labels["stage_label"].value_counts().rename_axis("stage").reset_index(name="count")
    counts.to_csv(outdir / f"{prefix}_stage_counts.csv", index=False)

    # Decoy FASTA of Cys+Pro sequences for the final selection
    decoy_fasta = outdir / f"{prefix}_decoys_cysP.fasta"
    with decoy_fasta.open("w", encoding="utf-8") as fh:
        for i, row in sel.reset_index(drop=True).iterrows():
            _extra = []
            for key in ["Gene_AA_Change","HLA","rank","Type"]:
                if key in row and pd.notna(row[key]):
                    _extra.append(f"{key}={row[key]}")
            if "_gene_from_notation" in row and pd.notna(row["_gene_from_notation"]):
                _extra.append(f"gene={row['_gene_from_notation']}")
            extra_str = ("|" + "|".join(_extra)) if _extra else ""
            header = (f">decoy_{i+1}|orig={row['peptide']}|mut_idx0={row['mut_idx0']}|wt={row['wt']}|mut={row['mut']}|"
                      f"pro_idx0={row.get('pro_insertion_index0')}|np={row.get('pro_insertion_np')}{extra_str}")
            seq = str(row["decoy_seq"])
            fh.write(header + "\n")
            for line in wrap(seq, 60):
                fh.write(line + "\n")

    # ------------------------
    # Plots
    # ------------------------
    # 1) Bar chart of counts per stage
    plt.figure()
    stages = counts["stage"].tolist()
    vals = counts["count"].tolist()
    plt.bar(stages, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Counts per stage")
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"{prefix}_counts_per_stage.png")
    plt.close()

    # 2) Histogram of decoy_fly_score by stage (selected vs others)
    plt.figure()
    sel_mask = all_with_labels["stage_label"] == "selected"
    others_mask = all_with_labels["stage_label"] != "selected"
    plt.hist(all_with_labels.loc[sel_mask, "decoy_fly_score"].dropna(), bins=30, alpha=0.7, label="selected")
    plt.hist(all_with_labels.loc[others_mask, "decoy_fly_score"].dropna(), bins=30, alpha=0.7, label="others")
    plt.xlabel("decoy_fly_score")
    plt.ylabel("Frequency")
    plt.title("Flyability (decoy) distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"{prefix}_hist_flyability.png")
    plt.close()

    # 3) Scatter of full-ppm vs decoy_fly_score by stage
    plt.figure()
    for stage in ["selected", "not_selected_stage2", "dropped_stage1_frac", "dropped_stage1_rule", "dropped_decoy_exact_albumin", "dropped_no_valid_pro_insertion", "dropped_base_has_proline", "dropped_base_has_cysteine", "dropped_base_wt_or_mut_C"]:
        m = all_with_labels["stage_label"] == stage
        if m.any():
            plt.scatter(all_with_labels.loc[m, "decoy_fly_score"], all_with_labels.loc[m, "stage2_full_best_ppm"], s=10, label=stage)
    plt.xlabel("decoy_fly_score")
    plt.ylabel("full-ppm (albumin 5–12mer)")
    plt.yscale("log")
    plt.title("full-ppm vs flyability by stage")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(outdir / "plots" / f"{prefix}_scatter_ppm_vs_fly.png")
    plt.close()

    # 4) Histogram of b/y max k
    plt.figure()
    bmax = all_with_labels["stage2_b_max_k"].dropna()
    ymax = all_with_labels["stage2_y_max_k"].dropna()
    if len(bmax) > 0 or len(ymax) > 0:
        bins_b = np.arange(1, int(bmax.max() if len(bmax)>0 else 2)+1)-0.5
        bins_y = np.arange(1, int(ymax.max() if len(ymax)>0 else 2)+1)-0.5
        if len(bmax) > 0:
            plt.hist(bmax, bins=bins_b, alpha=0.7, label="b_max_k")
        if len(ymax) > 0:
            plt.hist(ymax, bins=bins_y, alpha=0.7, label="y_max_k")
        plt.xlabel("max k (prefix/suffix in albumin)")
        plt.ylabel("Frequency")
        plt.title("b/y k-mer overlap with albumin")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "plots" / f"{prefix}_hist_by_k.png")
        plt.close()
    else:
        plt.close()

    # 5) Sequence transitions panel(s) for SELECTED set (up to 25)
    try:
        if not sel.empty:
            plot_sequence_transitions(sel, outdir / "plots" / f"{prefix}_sequence_transitions_selected.png", max_peptides=25)
        elif not s1_kept.empty:
            plot_sequence_transitions(s1_kept, outdir / "plots" / f"{prefix}_sequence_transitions_pool.png", max_peptides=25)
    except Exception as e:
        print(f"[WARN] Sequence transitions plotting failed: {e}")

    # 6) Bar plot of requested exclusion reasons
    try:
        c_wtmutC = int((all_with_labels["stage_label"] == "dropped_base_wt_or_mut_C").sum())
        c_hasP   = int((all_with_labels["stage_label"] == "dropped_base_has_proline").sum())
        c_bottom_half = int((all_with_labels["stage_label"] == "dropped_stage1_frac").sum())
        cats = ["WT/MUT==C", "Contains P (orig)", "Bottom-half fly"]
        vals = [c_wtmutC, c_hasP, c_bottom_half]
        plt.figure()
        plt.bar(cats, vals)
        plt.ylabel("Count")
        plt.title("Exclusions: WT/MUT C, Original P, Bottom-half fly")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / f"{prefix}_bar_exclusions_requested.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Exclusion bar plot failed: {e}")

    # 7) Scatter: decoy fly score vs albumin confusability (selected colored)
    try:
        plt.figure()
        sel_mask2 = all_with_labels["stage_label"] == "selected"
        not_sel_mask2 = ~sel_mask2
        plt.scatter(all_with_labels.loc[not_sel_mask2, "decoy_fly_score"],
                    all_with_labels.loc[not_sel_mask2, "albumin_confusability"],
                    s=12, label="not selected", alpha=0.7)
        plt.scatter(all_with_labels.loc[sel_mask2, "decoy_fly_score"],
                    all_with_labels.loc[sel_mask2, "albumin_confusability"],
                    s=16, label="selected")
        plt.xlabel("decoy_fly_score")
        plt.ylabel("albumin_confusability (0..1)")
        plt.title("Flyability vs Albumin similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "plots" / f"{prefix}_scatter_fly_vs_confusability.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Fly vs confusability scatter failed: {e}")

    # 8) Among top-fly peptides: precursor m/z similarity vs overall similarity
    try:
        ref_df = s1_kept if 's1_kept' in locals() and not s1_kept.empty else out_df
        if not ref_df.empty:
            thr = ref_df["decoy_fly_score"].quantile(0.75)
            top_fly = ref_df[ref_df["decoy_fly_score"] >= thr].copy()
            def _ppm_to_score(x, cap=float(args.ppm_cap)):
                import math
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                    return 0.0
                x = float(x)
                return max(0.0, 1.0 - min(x, cap)/cap)
            top_fly["full_mz_similarity"] = top_fly["stage2_full_best_ppm"].apply(_ppm_to_score)
            plt.figure()
            plt.scatter(top_fly["full_mz_similarity"], top_fly["albumin_confusability"], s=14, label="top-fly")
            mask_sel_top = top_fly["stage_label"] == "selected"
            if mask_sel_top.any():
                plt.scatter(top_fly.loc[mask_sel_top, "full_mz_similarity"],
                            top_fly.loc[mask_sel_top, "albumin_confusability"], s=16, label="selected (top-fly)")
            plt.xlabel("precursor m/z similarity (0..1)")
            plt.ylabel("overall albumin confusability (0..1)")
            plt.title("Top-fly: precursor m/z similarity vs overall similarity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "plots" / f"{prefix}_scatter_topfly_fullmz_vs_confusability.png")
            plt.close()
    except Exception as e:
        print(f"[WARN] Top-fly scatter failed: {e}")

    print(f"[INFO] Output directory: {outdir.resolve()}")
    print(f"[INFO] Wrote CSVs (features, selected, all_with_stage_labels, stage_counts), decoy FASTA, and plots.")

if __name__ == "__main__":
    main()
