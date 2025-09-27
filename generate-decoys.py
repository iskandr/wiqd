#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peptide selector & decoy feature generator

Given a list of candidate (mutant) MS peptide targets with mutation position and
the ref/wt + mutant amino acids, choose a subset of k (default 10) that:
  (A) Maximize subsequence overlap with human albumin on either side of the mutation,
  (B) Prefer a hydrophobic last residue (typical of many MHC-I ligands),
  (C) Prefer peptides containing proline, with a bonus if they contain "NP",
  (D) EXCLUDE peptides that contain cysteine anywhere or had a cysteine as the WT residue.

Then, for ranking/tie-breaking, form a cysteine-substituted peptide by placing 'C'
at the mutation position, and further rank by whether this cysteine-substituted
peptide has a close m/z match to SOME subsequence of albumin (with at least a
minimum k-mer overlap), while ensuring that the cysteine-substituted peptide is
NOT an exact subsequence of albumin. The m/z comparison is done across specified
charge states (default: 2 and 3).

Outputs:
  1) <prefix>_decoy_features.csv  - all candidates with computed features used for selection
  2) <prefix>_selected_top_k.csv  - the top-k subset actually selected

Assumptions & notes:
  * Input must supply mutant peptide sequences, a mutation position, and WT/Mut AAs.
  * Mutation index can be 0-based or 1-based; default is "auto" detection.
  * Albumin sequence is required (FASTA). Use human albumin (UniProt P02768), or a species-appropriate albumin.
  * Monoisotopic residue masses are based on widely used values (e.g., Unimod / Proteome community).
    - Water mass = 18.010564684 Da; Proton mass = 1.007276466812 Da.
    - Optional fixed Cys carbamidomethylation (+57.021464) can be toggled with --cys-mod carbamidomethyl.
  * Hydrophobic C-terminus set defaults to: AILMFWYV (can be overridden).
  * "Some k-mer overlap" for m/z matching uses an exact shared k-mer of length >= --min-kmer-overlap (default 4).
  * No external dependencies beyond Python standard library + pandas + numpy.

Usage example:
  python peptide_selector.py \
    --input candidates.csv \
    --albumin-fasta human_albumin.fasta \
    --peptide-col peptide --pos-col mut_pos --wt-col wt --mut-col mut \
    --k 10 --charges 2,3 --min-kmer-overlap 4 \
    --output-prefix run1

Input format example (CSV):
  peptide,mut_pos,wt,mut
  AAAAANPQL,5,A,N
  ....

Author: ChatGPT (GPT-5 Pro)
"""

from __future__ import annotations

import argparse
import csv
import sys
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import pandas as pd
import numpy as np


# Monoisotopic masses (Da) for standard amino acids (unmodified residues).
# These are standard values used in proteomics pipelines.
AA_MONO = {
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

WATER_MONO = 18.010564684       # H2O
PROTON_MONO = 1.007276466812    # H+


def read_fasta_sequence(fasta_path: str) -> str:
    """Read a FASTA file and return the first sequence (concatenated, uppercase)."""
    seq_lines: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                # header; ignore
                continue
            seq_lines.append(line.strip())
    seq = "".join(seq_lines).upper()
    seq = re.sub(r"[^A-Z]", "", seq)  # retain only letters
    return seq


def calc_peptide_mono_mass(seq: str, aa_masses: Dict[str, float], cys_fixed_mod: float = 0.0) -> float:
    """Compute monoisotopic neutral mass (M) of a peptide sequence: sum(aa) + H2O (+ fixed mods)."""
    m = 0.0
    for aa in seq:
        if aa not in aa_masses:
            raise ValueError(f"Unknown amino acid '{aa}' in sequence '{seq}'.")
        m += aa_masses[aa]
        if aa == "C" and cys_fixed_mod != 0.0:
            m += cys_fixed_mod
    return m + WATER_MONO


def mz_from_mass(neutral_mass: float, charge: int) -> float:
    """Compute m/z from neutral monoisotopic mass at given charge state."""
    if charge <= 0:
        raise ValueError("Charge must be positive integer.")
    return (neutral_mass + charge * PROTON_MONO) / charge


def detect_indexing_mode(peptide: str, pos: int, wt: str, mut: str, default_mode: str = "1") -> Tuple[int, str]:
    """
    Determine 0- or 1-based indexing for the mutation position.
    Returns (mut_index_0based, mode_used_as_str "0" or "1").
    Heuristics:
     - If exactly one of {peptide[pos] == mut, peptide[pos-1] == mut} is true, use that mode.
     - Else if exactly one of {peptide[pos] == wt, peptide[pos-1] == wt} is true, use the opposite (mut not present).
     - Else fallback to default_mode (1-based) unless out of range, then 0-based if valid, else raise.
    """
    L = len(peptide)
    # candidates
    idx0 = pos
    idx1 = pos - 1

    cond0_mut = (0 <= idx0 < L) and (peptide[idx0] == mut)
    cond1_mut = (0 <= idx1 < L) and (peptide[idx1] == mut)

    if cond0_mut ^ cond1_mut:
        return (idx0 if cond0_mut else idx1, "0" if cond0_mut else "1")

    # Try heuristics with WT (less likely to match in mutant peptide)
    cond0_wt = (0 <= idx0 < L) and (peptide[idx0] == wt)
    cond1_wt = (0 <= idx1 < L) and (peptide[idx1] == wt)

    if cond0_wt ^ cond1_wt:
        # If peptide[idx0]==wt implies pos is wrong for mutant; prefer the other
        chosen = idx1 if cond0_wt else idx0
        return (chosen, "1" if cond0_wt else "0")

    # Fallback
    mode = default_mode
    if mode == "1":
        if 0 <= idx1 < L:
            return (idx1, "1")
        elif 0 <= idx0 < L:
            return (idx0, "0")
        else:
            raise ValueError(f"Mutation position {pos} is out of range for peptide length {L}.")
    else:
        if 0 <= idx0 < L:
            return (idx0, "0")
        elif 0 <= idx1 < L:
            return (idx1, "1")
        else:
            raise ValueError(f"Mutation position {pos} is out of range for peptide length {L}.")


@dataclass
class FlankOverlapResult:
    left_len: int
    right_len: int
    albumin_center_idx: int            # 0-based index of the "central" residue in albumin
    albumin_left_seq: str
    albumin_right_seq: str
    albumin_span_start: int            # inclusive, 0-based
    albumin_span_end: int              # exclusive, 0-based (center+1+right_len)


def best_flank_overlap_against_albumin(
    peptide: str, mut_idx0: int, albumin: str
) -> FlankOverlapResult:
    """
    Find the albumin position that maximizes (left overlap + right overlap) with the peptide's flanks.
    Overlap definition:
      - left overlap: longest suffix of peptide[:mut_idx0] that equals albumin[j - L : j]
      - right overlap: longest prefix of peptide[mut_idx0+1:] that equals albumin[j+1 : j+1+R]
    where j is the albumin "center" (analogous to the peptide's mutated residue).
    Returns the best overlap lengths and the aligned albumin subsequences.
    """
    left = peptide[:mut_idx0]
    right = peptide[mut_idx0 + 1:]
    L_left = len(left)
    L_right = len(right)

    best = FlankOverlapResult(
        left_len=0, right_len=0, albumin_center_idx=-1,
        albumin_left_seq="", albumin_right_seq="",
        albumin_span_start=-1, albumin_span_end=-1
    )

    N = len(albumin)
    for j in range(N):
        # Left overlap: suffix of 'left' equals albumin segment ending at j-1 (i.e., [j-L:j])
        max_l = 0
        max_l_seq = ""
        max_l_start = j
        upper_l = min(L_left, j)  # can't extend beyond start
        for l in range(upper_l, 0, -1):
            if albumin[j - l:j] == left[-l:]:
                max_l = l
                max_l_seq = left[-l:]
                max_l_start = j - l
                break

        # Right overlap: prefix of 'right' equals albumin segment starting at j+1
        max_r = 0
        max_r_seq = ""
        max_r_end = j + 1
        upper_r = min(L_right, N - (j + 1))
        for r in range(upper_r, 0, -1):
            if albumin[j + 1:j + 1 + r] == right[:r]:
                max_r = r
                max_r_seq = right[:r]
                max_r_end = j + 1 + r
                break

        if (max_l + max_r) > (best.left_len + best.right_len):
            best = FlankOverlapResult(
                left_len=max_l,
                right_len=max_r,
                albumin_center_idx=j,
                albumin_left_seq=max_l_seq,
                albumin_right_seq=max_r_seq,
                albumin_span_start=max_l_start,
                albumin_span_end=max_r_end,
            )

    return best


def is_subsequence_in_albumin(seq: str, albumin: str) -> Tuple[bool, Optional[int]]:
    """Return (present, start_idx) for exact substring match in albumin (0-based)."""
    idx = albumin.find(seq)
    return (idx != -1, None if idx == -1 else idx)


def kmer_set(seq: str, k: int) -> set:
    """All distinct k-mers of a sequence."""
    if k <= 0 or k > len(seq):
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


@dataclass
class CysMzMatchResult:
    best_da_error: float
    best_ppm_error: float
    best_charge: Optional[int]
    best_albumin_subseq: Optional[str]
    best_albumin_start: Optional[int]  # 0-based inclusive
    best_albumin_end: Optional[int]    # 0-based exclusive


def closest_cys_variant_mz_to_albumin(
    cys_variant: str,
    albumin: str,
    charges: Iterable[int],
    aa_masses: Dict[str, float],
    cys_fixed_mod: float,
    min_kmer_overlap: int,
) -> CysMzMatchResult:
    """
    For the cysteine-substituted peptide, compute the closest m/z match to any albumin
    subsequence of the same length that shares at least one k-mer of length >= min_kmer_overlap.
    EXACT matches (entire sequence identical) are ignored to satisfy the "not fully in albumin" rule.
    """
    L = len(cys_variant)
    present, exact_start = is_subsequence_in_albumin(cys_variant, albumin)
    cys_m = calc_peptide_mono_mass(cys_variant, aa_masses, cys_fixed_mod)
    cys_mz = {z: mz_from_mass(cys_m, z) for z in charges}
    cys_kmers = set()
    for k in range(min_kmer_overlap, L + 1):
        cys_kmers |= kmer_set(cys_variant, k)

    best = CysMzMatchResult(
        best_da_error=float("inf"),
        best_ppm_error=float("inf"),
        best_charge=None,
        best_albumin_subseq=None,
        best_albumin_start=None,
        best_albumin_end=None,
    )

    for start in range(0, len(albumin) - L + 1):
        end = start + L
        sub = albumin[start:end]
        if sub == cys_variant:
            # Ignore exact match by design.
            continue
        # k-mer overlap check
        if min_kmer_overlap > 0:
            ok = False
            # Efficient: test only the minimum k to reduce cost; however we compute across k>=min too.
            # We already built cys_kmers across k>=min, so just test intersection with sub's k-mers for min length.
            # To reduce cost, test min-length first; if none, skip.
            if kmer_set(sub, min_kmer_overlap) & kmer_set(cys_variant, min_kmer_overlap):
                ok = True
            else:
                # try longer kmers opportunistically (rare)
                for k in range(min_kmer_overlap + 1, min(L, 12) + 1):
                    if kmer_set(sub, k) & kmer_set(cys_variant, k):
                        ok = True
                        break
            if not ok:
                continue

        # Compute m/z differences across charges
        sub_m = calc_peptide_mono_mass(sub, aa_masses, cys_fixed_mod)
        for z in charges:
            sub_mz = mz_from_mass(sub_m, z)
            da_err = abs(sub_mz - cys_mz[z])
            ppm_err = (da_err / cys_mz[z]) * 1e6
            if ppm_err < best.best_ppm_error:
                best = CysMzMatchResult(
                    best_da_error=da_err,
                    best_ppm_error=ppm_err,
                    best_charge=z,
                    best_albumin_subseq=sub,
                    best_albumin_start=start,
                    best_albumin_end=end,
                )

    return best


def build_score(
    left_right_overlap_sum: int,
    hydrophobic_last: bool,
    has_P: bool,
    has_NP: bool,
    best_ppm_error: float,
    weights: Dict[str, float],
    ppm_scale: float = 10.0,
) -> float:
    """
    Combine features into a ranking score.
    Overlap dominates; hydrophobic tail and P/NP add bonuses; better m/z closeness (smaller ppm) adds a bonus.
    """
    score = 0.0
    score += weights.get("overlap", 1.0) * float(left_right_overlap_sum)
    if hydrophobic_last:
        score += weights.get("hydroCTerm", 3.0)
    if has_P:
        score += weights.get("hasP", 1.5)
    if has_NP:
        score += weights.get("hasNP", 1.5)
    # m/z closeness bonus: near 0 ppm gives ~weights['mz']; decays linearly to 0 at ppm_scale
    mz_w = weights.get("mz", 2.0)
    if np.isfinite(best_ppm_error):
        mz_bonus = max(0.0, 1.0 - min(best_ppm_error / ppm_scale, 1.0))
        score += mz_w * mz_bonus
    return score


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select top-k mutant peptides vs. albumin and emit decoy features.")
    p.add_argument("--input", required=True, help="Input CSV/TSV with candidate peptides.")
    p.add_argument("--albumin-fasta", required=True, help="FASTA file containing an albumin sequence.")
    p.add_argument("--peptide-col", default="peptide", help="Column name for mutant peptide sequence.")
    p.add_argument("--pos-col", default="mutation_position", help="Column name for mutation position (0- or 1-based).")
    p.add_argument("--wt-col", default="wt", help="Column name for wild-type (reference) AA.")
    p.add_argument("--mut-col", default="mut", help="Column name for mutant AA.")
    p.add_argument("--indexing", choices=["auto", "0", "1"], default="auto",
                   help="Interpretation of mutation position: auto-detect, 0-based, or 1-based.")
    p.add_argument("--k", type=int, default=10, help="Number of peptides to select.")
    p.add_argument("--charges", default="2,3", help="Comma-separated charge states to evaluate, e.g., '2,3'.")
    p.add_argument("--min-kmer-overlap", type=int, default=4, help="Minimum shared k-mer for m/z albumin match.")
    p.add_argument("--hydrophobic", default="AILMFWYV", help="Hydrophobic set for C-terminus preference.")
    p.add_argument("--cys-mod", choices=["none", "carbamidomethyl"], default="none",
                   help="Fixed modification on Cys residues for mass calculations.")
    p.add_argument("--output-prefix", default="peptide_selection", help="Prefix for output CSV files.")
    p.add_argument("--require-hydrophobic-last", action="store_true",
                   help="If set, enforce hydrophobic last residue as a hard filter (default is a soft preference).")
    p.add_argument("--weights", default="overlap:1.0,hydroCTerm:3.0,hasP:1.5,hasNP:1.5,mz:2.0",
                   help="Comma-separated 'key:value' pairs for scoring weights.")
    return p.parse_args(argv)


def parse_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Bad weight token '{token}'. Use key:value format.")
        k, v = token.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Load albumin sequence
    albumin = read_fasta_sequence(args.albumin_fasta)
    if not albumin or len(albumin) < 50:
        raise RuntimeError("Albumin sequence seems too short. Provide a proper FASTA.")

    # Determine delimiter from file extension
    if args.input.lower().endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t", dtype=str)
    else:
        # Let pandas auto-detect for CSV-like files
        df = pd.read_csv(args.input, dtype=str)

    # Clean/standardize columns
    required_cols = [args.peptide_col, args.pos_col, args.wt_col, args.mut_col]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in input. Columns present: {list(df.columns)}")

    # Uppercase sequences and amino acids; coerce pos to int
    df["_peptide"] = df[args.peptide_col].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df["_wt"] = df[args.wt_col].astype(str).str.upper().str.strip()
    df["_mut"] = df[args.mut_col].astype(str).str.upper().str.strip()
    df["_pos_raw"] = df[args.pos_col].astype(str).str.strip()

    # Parse positions robustly
    def safe_int(x):
        try:
            return int(float(x))
        except Exception:
            return np.nan

    df["_pos_int"] = df["_pos_raw"].apply(safe_int)
    if df["_pos_int"].isna().any():
        bad = df[df["_pos_int"].isna()]
        raise ValueError(f"Non-integer mutation positions encountered:\n{bad[[args.pos_col]].to_string(index=False)}")

    # Charges
    charges = sorted({int(x.strip()) for x in args.charges.split(",") if x.strip()})
    if not charges:
        raise ValueError("Provide at least one positive charge in --charges.")
    for z in charges:
        if z <= 0:
            raise ValueError("Charges must be positive integers.")

    # Cys mod
    cys_fixed_mod = 0.0
    if args.cys_mod == "carbamidomethyl":
        cys_fixed_mod = 57.021464

    hydro_set = set(args.hydrophobic.upper())
    weights = parse_weights(args.weights)

    # Feature computation per row
    records: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        pep = row["_peptide"]
        wt = row["_wt"]
        mut = row["_mut"]
        pos = int(row["_pos_int"])

        # Resolve index mode
        if args.indexing == "auto":
            mut_idx0, mode_used = detect_indexing_mode(pep, pos, wt, mut, default_mode="1")
        elif args.indexing == "0":
            mut_idx0, mode_used = pos, "0"
        else:
            mut_idx0, mode_used = pos - 1, "1"

        if not (0 <= mut_idx0 < len(pep)):
            raise ValueError(f"Resolved mutation index out of range for peptide '{pep}', pos={pos}, mode={mode_used}")

        # Basic flags
        contains_C = ("C" in pep)
        wt_is_C = (wt == "C")
        hydrophobic_last = (len(pep) > 0 and pep[-1] in hydro_set)
        has_P = ("P" in pep)
        has_NP = ("NP" in pep)

        # Hard exclusions
        passes_no_cys = (not contains_C) and (not wt_is_C)
        # Optional hard filter for hydrophobic last residue
        passes_hydro_tail = (hydrophobic_last or (not args.require_hydrophobic_last))

        # Flank overlap against albumin
        overlap = best_flank_overlap_against_albumin(pep, mut_idx0, albumin)
        overlap_sum = overlap.left_len + overlap.right_len

        # Masses for mutant peptide
        try:
            pep_mass = calc_peptide_mono_mass(pep, AA_MONO, cys_fixed_mod=cys_fixed_mod)
        except ValueError as e:
            raise ValueError(f"Row {idx}: {e}")
        pep_mz = {z: mz_from_mass(pep_mass, z) for z in charges}

        # Cys-variant (replace mutant residue with 'C')
        pep_chars = list(pep)
        pep_chars[mut_idx0] = "C"
        cys_variant = "".join(pep_chars)

        # Ensure cys-variant is NOT an exact subsequence of albumin
        cys_exact, cys_exact_start = is_subsequence_in_albumin(cys_variant, albumin)

        # m/z matching to albumin subsequences (excluding exact matches), requiring k-mer overlap
        cys_match = closest_cys_variant_mz_to_albumin(
            cys_variant=cys_variant,
            albumin=albumin,
            charges=charges,
            aa_masses=AA_MONO,
            cys_fixed_mod=cys_fixed_mod,
            min_kmer_overlap=int(args.min_kmer_overlap),
        )

        # Build score (m/z bonus uses cys_match.best_ppm_error)
        score = build_score(
            left_right_overlap_sum=overlap_sum,
            hydrophobic_last=hydrophobic_last,
            has_P=has_P,
            has_NP=has_NP,
            best_ppm_error=cys_match.best_ppm_error,
            weights=weights,
            ppm_scale=10.0,
        )

        rec: Dict[str, object] = {}

        # Original data passthrough
        for c in df.columns:
            rec[c] = row[c]

        # Computed features
        rec.update({
            "peptide": pep,
            "length": len(pep),
            "mut_idx0": mut_idx0,
            "indexing_used": mode_used,
            "wt": wt,
            "mut": mut,

            "contains_C": contains_C,
            "wt_is_C": wt_is_C,
            "hydrophobic_C_terminal": hydrophobic_last,
            "has_P": has_P,
            "has_NP": has_NP,
            "passes_no_cys": passes_no_cys,
            "passes_hydrophobic_tail_filter": passes_hydro_tail,

            "overlap_left_len": overlap.left_len,
            "overlap_right_len": overlap.right_len,
            "overlap_sum": overlap_sum,
            "albumin_center_idx0": overlap.albumin_center_idx,
            "albumin_left_seq": overlap.albumin_left_seq,
            "albumin_right_seq": overlap.albumin_right_seq,
            "albumin_span_start0": overlap.albumin_span_start,
            "albumin_span_end0": overlap.albumin_span_end,
            "albumin_span_start1": (overlap.albumin_span_start + 1) if overlap.albumin_span_start >= 0 else None,
            "albumin_span_end1": (overlap.albumin_span_end) if overlap.albumin_span_end >= 0 else None,  # end is exclusive

            "pep_mono_mass": pep_mass,
        })

        for z in charges:
            rec[f"pep_mz_z{z}"] = pep_mz[z]

        # Cys variant details
        rec.update({
            "cys_variant": cys_variant,
            "cys_variant_exact_in_albumin": bool(cys_exact),
            "cys_variant_exact_albumin_start0": cys_exact_start if cys_exact else None,
            "cys_best_da_error": cys_match.best_da_error if np.isfinite(cys_match.best_da_error) else None,
            "cys_best_ppm_error": cys_match.best_ppm_error if np.isfinite(cys_match.best_ppm_error) else None,
            "cys_best_charge": cys_match.best_charge,
            "cys_best_albumin_subseq": cys_match.best_albumin_subseq,
            "cys_best_albumin_start0": cys_match.best_albumin_start,
            "cys_best_albumin_end0": cys_match.best_albumin_end,
            "cys_best_albumin_start1": (cys_match.best_albumin_start + 1) if cys_match.best_albumin_start is not None else None,
            "cys_best_albumin_end1": cys_match.best_albumin_end if cys_match.best_albumin_end is not None else None,
        })

        # m/z for cys variant (neutral mass + charges)
        cys_mass = calc_peptide_mono_mass(cys_variant, AA_MONO, cys_fixed_mod=cys_fixed_mod)
        rec["cys_variant_mono_mass"] = cys_mass
        for z in charges:
            rec[f"cys_variant_mz_z{z}"] = mz_from_mass(cys_mass, z)

        # Final score
        rec["score"] = score

        # Eligibility for final selection
        rec["eligible_for_selection"] = bool(passes_no_cys and passes_hydro_tail and (not cys_exact))

        records.append(rec)

    out_df = pd.DataFrame.from_records(records)

    # Selection: filter eligible, sort by score, break ties by overlap_sum desc, cys_best_ppm_error asc, hydrophobic, NP, P
    sel = out_df[out_df["eligible_for_selection"]].copy()
    sel = sel.sort_values(
        by=["score", "overlap_sum", "cys_best_ppm_error", "hydrophobic_C_terminal", "has_NP", "has_P"],
        ascending=[False, False, True, False, False, False],
        kind="mergesort",  # stable
    ).head(args.k)

    # Write outputs
    prefix = args.output_prefix
    out_df.to_csv(f"{prefix}_decoy_features.csv", index=False)
    sel.to_csv(f"{prefix}_selected_top_k.csv", index=False)

    # Human-friendly summary to stdout
    print(f"[INFO] Albumin length: {len(albumin)} aa")
    print(f"[INFO] Processed {len(out_df)} candidates; {len(sel)} selected (k={args.k}).")
    print(f"[INFO] Wrote: {prefix}_decoy_features.csv and {prefix}_selected_top_k.csv")

if __name__ == "__main__":
    main()
