#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoygen.py — Cys+Pro+Met decoy generator with Proline/Methionine ranked placement,
albumin RT→precursor→fragment chaining, k-mer de-dup, and rich plots.

Key constraints (unchanged):
  • Cys goes at the mutation index.
  • Do not alter the last residue; it must be hydrophobic (A I L M F W Y V).
  • Proline must be between C and M (M–P–C or C–P–M), EXCEPT if C is at index 0:
      allow C–P–M or C–M–P. Always prefer spacing (avoid adjacency).
  • Drop rows with WT/MUT == C; original peptide must have NO C/M/P; C‑term hydrophobic;
    mutation not last; strict mutation letter check by default.

Proline ranked placement (applied BEFORE Methionine; all ties broken lexicographically):
  1) Prefer X–P with X ∈ {G}  (GP)
  2) Then X ∈ {A}            (AP)
  3) Then X ∈ {L,V,I,S,T}    (LVISTP)
  4) Avoid R/K before P when alternatives exist
  5) Prefer replacing Q (Q→P)
  6) ≥1 residue away from C (not adjacent); prefer more distance
  7) Prefer ↑hydrophobicity (KD GRAVY) vs cysteine‑only baseline
  8) Prefer ↑fly_score_norm (normalized mix of charge/surface/len/aromatic)
  9) Prefer ↑albumin confusability (estimated; same chained evidence family)

Methionine ranked placement (given chosen P; ties lexicographically):
  1) Prefer replacing Q (Q→M)
  2) ≥1 residue away from C; prefer more distance (and prefer spacing from P)
  3) Prefer ↑hydrophobicity vs (Cys+P) baseline
  4) Prefer ↑fly_score_norm
  5) Prefer ↑albumin confusability

Albumin metrics (final per‑decoy; same as before):
  • N_precursor_mz, N_precursor_mz_rt (5..15-mers; z∈{1,2,3}; ±ppm; RT ±1 min)
  • N/Frac_fragment_mz_given_precursor[_rt] (b/y, k=2..11; z∈{1,2,3}; ±ppm)
  • Confusability_simple_albumin = mean( Norm(N_precursor_mz),
                                        Norm(N_precursor_mz_rt),
                                        Frac_frag|prec, Frac_frag|prec+rt )
    with cohort 95th‑percentile caps for Norm(*).
"""

from __future__ import annotations
import argparse, re, math, os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# tqdm (progress)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textwrap import wrap
from pathlib import Path

# =================== Chemistry / Mass ===================
AA_MONO = {
    "A": 71.037113805, "R": 156.10111105, "N": 114.04292747, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.05857754, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.09496305,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.07931298, "Y": 163.063328575, "V": 99.068413945,
}
WATER = 18.010564684
PROTON = 1.007276466812

def calc_mass(seq: str, cys_fixed_mod: float = 0.0) -> float:
    m = 0.0
    for a in seq:
        m += AA_MONO[a]
        if a == "C" and cys_fixed_mod:
            m += cys_fixed_mod
    return m + WATER

def mz_from_mass(M: float, z: int) -> float:
    return (M + z*PROTON) / z

# =================== Flyability (normalized components) ===================
KD_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
HYDRO_CTERM_SET = set("AILMFWYV")

def kd_gravy(seq: str) -> float:
    return float(np.mean([KD_HYDRO.get(a, 0.0) for a in seq])) if seq else 0.0

def triangular_len_score(L: int, zero_min: int = 7, peak_min: int = 9, peak_max: int = 11, zero_max: int = 20) -> float:
    if L <= zero_min: return 0.0
    if L >= zero_max: return 0.0
    if L < peak_min: return (L - zero_min) / max(1, (peak_min - zero_min))
    if L <= peak_max: return 1.0
    return max(0.0, (zero_max - L) / max(1, (zero_max - peak_max)))

def fly_components(seq: str) -> Dict[str, float]:
    L = len(seq)
    if L == 0:
        return {"fly_charge_norm":0.0,"fly_surface_norm":0.0,"fly_len_norm":0.0,"fly_aromatic_norm":0.0}
    counts = {a: seq.count(a) for a in AA_MONO}
    R,K,H,D,E = counts["R"],counts["K"],counts["H"],counts["D"],counts["E"]
    charge_sites = 1.0 + R + K + 0.7*H
    acid_penalty = 0.1*(D+E)
    charge = 1.0 - math.exp(-max(0.0, charge_sites - acid_penalty)/2.0)
    gravy = kd_gravy(seq)
    surface = float(np.clip((gravy + 4.5)/9.0, 0.0, 1.0))
    W,Y,F = counts["W"],counts["Y"],counts["F"]
    aromatic = float(np.clip((1.0*W + 0.7*Y + 0.5*F)/max(1.0, L/2.0), 0.0, 1.0))
    length = triangular_len_score(L)
    return {"fly_charge_norm":charge,"fly_surface_norm":surface,"fly_aromatic_norm":aromatic,"fly_len_norm":length}

def fly_score_norm(seq: str, weights: Dict[str,float]) -> float:
    c = fly_components(seq)
    w = {"charge":weights.get("charge",0.5),
         "surface":weights.get("surface",0.35),
         "len":weights.get("len",0.1),
         "aromatic":weights.get("aromatic",0.05)}
    W = max(1e-9, sum(w.values()))
    for k in w: w[k] /= W
    s = w["charge"]*c["fly_charge_norm"] + w["surface"]*c["fly_surface_norm"] + w["len"]*c["fly_len_norm"] + w["aromatic"]*c["fly_aromatic_norm"]
    return float(np.clip(s, 0.0, 1.0))

# =================== RT predictor (absolute, not re‑scaled) ===================
def predict_rt_min(seq: str, gradient_min: float = 20.0) -> float:
    if not seq: return 0.0
    gravy = kd_gravy(seq)
    frac = (gravy + 4.5) / 9.0
    base = 0.5 + max(0.0, gradient_min - 1.0) * min(1.0, max(0.0, frac))
    length_adj = 0.03 * max(0, len(seq) - 8) * (gradient_min / 20.0)
    basic_adj = -0.15 * (seq.count("K") + seq.count("R") + 0.3*seq.count("H")) * (gradient_min / 20.0)
    return float(np.clip(base + length_adj + basic_adj, 0.0, gradient_min))

# =================== Albumin sequence (embedded prepro 609aa) ===================
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

# =================== Fragments (b/y) ===================
def fragment_mz_list(seq: str, kmin: int, kmax: int, charges: List[int]) -> List[Tuple[float,str,int,int]]:
    out = []
    L = len(seq)
    kmax_eff = min(kmax, max(kmin, L-1))
    for k in range(kmin, kmax_eff+1):
        bseq = seq[:k]; yseq = seq[-k:]
        Mb = calc_mass(bseq) - WATER
        My = calc_mass(yseq)
        for z in charges:
            out.append(((Mb + z*PROTON)/z, "b", k, z))
            out.append(((My + z*PROTON)/z, "y", k, z))
    return out

# =================== Parsing & indexing helpers ===================
def parse_mutation_notation(s: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    if s is None: return (None,None,None,None)
    s = str(s).strip()
    if not s: return (None,None,None,None)
    m = re.search(r'([A-Za-z0-9_.-]+)[_:\-]([A-Za-z])(\d+)([A-Za-z])$', s)
    if m:
        return (m.group(2).upper(), int(m.group(3)), m.group(4).upper(), m.group(1))
    m2 = re.search(r'([A-Za-z])(\d+)([A-Za-z])$', s)
    if m2:
        return (m2.group(1).upper(), int(m2.group(2)), m2.group(3).upper(), s[:m2.start()].strip("_:-"))
    return (None,None,None,None)

def resolve_mut_idx(peptide: str, pos: int, mode: str) -> int:
    L = len(peptide)
    if mode in ("N0","0"): return pos
    if mode in ("N1","1"): return pos - 1
    if mode == "C1": return L - pos
    if mode == "C0": return L - 1 - pos
    raise ValueError(f"Unknown indexing mode {mode}")

def detect_indexing(peptide: str, pos: int, wt: str, mut: str) -> Tuple[int,str]:
    for m in ("N1","N0","C1","C0"):
        idx = resolve_mut_idx(peptide, pos, m)
        if 0 <= idx < len(peptide) and peptide[idx] in (wt, mut):
            return idx, m
    idx = resolve_mut_idx(peptide, pos, "N1")
    return idx, "N1"

# =================== Albumin window index (5..15) ===================
def build_albumin_index(albumin: str, kmin: int, kmax: int, charges: List[int], cys_fixed_mod: float, rt_total_min: float):
    rows = []
    for k in range(kmin, kmax+1):
        for s in range(0, len(albumin)-k+1):
            w = albumin[s:s+k]
            m = calc_mass(w, cys_fixed_mod)
            rows.append({"start0": s, "length": k, "window": w, "mass": m, "rt_min": predict_rt_min(w, rt_total_min)})
    df = pd.DataFrame(rows)
    # Precompute albumin window fragments cache
    frag_cache = {}
    for i in range(len(df)):
        frs = fragment_mz_list(df.at[i,"window"], 2, 11, charges)
        frag_cache[i] = [mz for (mz,ion,k,z) in frs]
    return df, frag_cache

def ppm(a: float, b: float) -> float:
    return (abs(a-b)/b*1e6) if b else float("inf")

def any_within_ppm(q: float, arr: List[float], tol_ppm: float) -> bool:
    if not arr: return False
    tol = q * tol_ppm * 1e-6
    # binary search
    import bisect
    i = bisect.bisect_left(arr, q)
    candidates = []
    if i < len(arr): candidates.append(arr[i])
    if i > 0: candidates.append(arr[i-1])
    if i+1 < len(arr): candidates.append(arr[i+1])
    return any(abs(x-q) <= tol for x in candidates)

# =================== Confusability (estimated per candidate; final normalized later) ===================
def estimate_albumin_conf(seq: str, alb_df: pd.DataFrame, alb_frag_cache: Dict[int, List[float]],
                          charges: List[int], ppm_tol: float, rt_total_min: float, cys_fixed_mod: float):
    pep_mass = calc_mass(seq, cys_fixed_mod)
    pep_mz = {z: mz_from_mass(pep_mass, z) for z in charges}
    rt_pep = predict_rt_min(seq, rt_total_min)

    matched = set(); matched_rt = set()
    frag_kmin, frag_kmax = 2, 11
    # precursor matching
    for i, wr in alb_df.iterrows():
        wmz_by_z = {z: mz_from_mass(wr["mass"], z) for z in charges}
        ok = any(ppm(pep_mz[z], wmz_by_z[z]) <= ppm_tol for z in charges)
        if ok:
            matched.add(int(i))
            if abs(wr["rt_min"] - rt_pep) <= 1.0:  # rt_tol default; exact tol will be applied in final
                matched_rt.add(int(i))

    # fragment lists for peptide
    frs_pep = fragment_mz_list(seq, frag_kmin, frag_kmax, charges)
    n_types = float(len(frs_pep)) if frs_pep else 1.0

    # helper to gather albumin frags for a subset
    def gather(idxset: set) -> List[float]:
        out = []
        for j in idxset:
            arr = alb_frag_cache.get(int(j))
            if arr: out.extend(arr)
        out.sort()
        return out

    prec_fr = gather(matched)
    prec_rt_fr = gather(matched_rt)

    # count types that match within ppm
    seen_prec = set(); seen_prec_rt = set()
    for (mz,ion,k,z) in frs_pep:
        if prec_fr and any_within_ppm(mz, prec_fr, ppm_tol):    seen_prec.add((ion,k,z))
        if prec_rt_fr and any_within_ppm(mz, prec_rt_fr, ppm_tol): seen_prec_rt.add((ion,k,z))

    N_prec = float(len(matched))
    N_prec_rt = float(len(matched_rt))
    Frac_prec = float(len(seen_prec))/n_types
    Frac_prec_rt = float(len(seen_prec_rt))/n_types

    # simple, un‑capped estimate used only for *within-peptide* P/M ranking
    conf_est = (N_prec + N_prec_rt + Frac_prec + Frac_prec_rt)/4.0
    return dict(
        N_prec=N_prec, N_prec_rt=N_prec_rt,
        Frac_prec=Frac_prec, Frac_prec_rt=Frac_prec_rt,
        conf_est=conf_est,
        N_frag_prec_rt=float(len(seen_prec_rt))
    )

# =================== Proline context tiers & filters ===================
P_TIER = {
    "G": 0,  # GP best
    "A": 1,  # AP next
    # L/V/I/S/T next
    "L": 2, "V": 2, "I": 2, "S": 2, "T": 2,
}

def tier_for_x_before_p(x_prev: str) -> int:
    return P_TIER.get(x_prev, 3)

# =================== Main ===================
def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Cys+Pro+Met decoy generator with ranked P/M placement and albumin chaining")
    # IO
    ap.add_argument("--input", required=True)
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--pos-col", default="mutation_position")
    ap.add_argument("--wt-col", default="wt")
    ap.add_argument("--mut-col", default="mut")
    ap.add_argument("--mut-notation-col", default=None)
    ap.add_argument("--indexing", choices=["auto","N0","N1","C0","C1","0","1"], default="auto")
    ap.add_argument("--outdir", default="decoygen_out")
    ap.add_argument("--output-prefix", default="decoygen")

    # Chemistry / MS
    ap.add_argument("--charges", default="1,2,3")
    ap.add_argument("--ppm", type=float, default=30.0)
    ap.add_argument("--alb-kmin", type=int, default=5)
    ap.add_argument("--alb-kmax", type=int, default=15)
    ap.add_argument("--frag-kmin", type=int, default=2)
    ap.add_argument("--frag-kmax", type=int, default=11)
    ap.add_argument("--rt-total-min", type=float, default=20.0)
    ap.add_argument("--rt-tol", type=float, default=1.0)
    ap.add_argument("--cys-mod", choices=["none","carbamidomethyl"], default="none")

    # Selection preferences / constraints
    ap.add_argument("--dedup-k", type=int, default=4)
    ap.add_argument("--avoid-RK-before-P", dest="avoid_rk_p", action="store_true", default=True)
    ap.add_argument("--allow-RK-before-P", dest="avoid_rk_p", action="store_false")

    # Fly weights
    ap.add_argument("--fly-weights", default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05")

    # Selection count
    ap.add_argument("--N", type=int, default=10)

    # Strict mutation check
    ap.add_argument("--strict-mutation-check", action="store_true", default=True)
    ap.add_argument("--no-strict-mutation-check", dest="strict-mutation-check", action="store_false")

    args = ap.parse_args(argv)

    outdir = Path(args.outdir); plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    # Albumin sequence
    albumin = ALBUMIN_P02768_PREPRO_609
    if len(albumin) not in (609,585):
        print(f"[WARN] Embedded albumin length={len(albumin)} (expected 609/585). Proceeding.")

    # Read input
    if args.input.lower().endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t", dtype=str)
    else:
        df = pd.read_csv(args.input, dtype=str)
    df["_row_id"] = np.arange(len(df))

    # Map columns
    if args.peptide_col not in df.columns or args.pos_col not in df.columns:
        raise KeyError(f"Missing required columns: peptide-col={args.peptide_col}, pos-col={args.pos_col}")
    df["_peptide"] = df[args.peptide_col].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)

    def to_int(x):
        try: return int(float(str(x).strip()))
        except Exception: return np.nan
    df["_pos_int"] = df[args.pos_col].apply(to_int)

    df["_wt"] = (df[args.wt_col].astype(str).str.upper().str.strip() if args.wt_col in df.columns else np.nan)
    df["_mut"] = (df[args.mut_col].astype(str).str.upper().str.strip() if args.mut_col in df.columns else np.nan)

    if args.mut_notation_col and args.mut_notation_col in df.columns:
        parsed = df[args.mut_notation_col].apply(parse_mutation_notation).apply(pd.Series)
        parsed.columns = ["_wt_parsed","_pos_parsed","_mut_parsed","_gene_parsed"]
        df = pd.concat([df, parsed], axis=1)
        if df["_wt"].isna().all(): df["_wt"] = df["_wt_parsed"]
        if df["_mut"].isna().all(): df["_mut"] = df["_mut_parsed"]
        df["_pos_int"] = np.where(pd.isna(df["_pos_int"]), df["_pos_parsed"], df["_pos_int"])

    if df["_pos_int"].isna().any():
        bad = df[df["_pos_int"].isna()]
        raise ValueError(f"Non-integer mutation positions encountered:\n{bad[[args.pos_col]].to_string(index=False)}")

    # Charges & mods
    charges = sorted({int(x.strip()) for x in args.charges.split(",") if x.strip()})
    if not charges: raise ValueError("Provide at least one positive charge via --charges.")
    cys_fixed_mod = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0

    # ---- Stage A: base filters + mutation index/letter sanity ----
    n_total = len(df)
    drops = dict(wtmutC=0, hasP=0, hasC=0, hasM=0, nonhydro_cterm=0, mut_last=0, mut_mismatch=0)
    stage_labels = []
    survivors = []

    for row in tqdm(df.to_dict("records"), total=len(df), desc="Parsing & base filtering"):
        pep = row["_peptide"]
        wt = str(row.get("_wt")) if pd.notna(row.get("_wt")) else None
        mut = str(row.get("_mut")) if pd.notna(row.get("_mut")) else None
        pos = int(row["_pos_int"])

        # WT/MUT cannot be C
        if (wt == "C") or (mut == "C"):
            drops["wtmutC"] += 1; row["stage_label"] = "dropped_base_wt_or_mut_C"; stage_labels.append(row); continue
        # original peptide must not contain P/C/M
        if "P" in pep:
            drops["hasP"] += 1; row["stage_label"] = "dropped_base_has_proline"; stage_labels.append(row); continue
        if "C" in pep:
            drops["hasC"] += 1; row["stage_label"] = "dropped_base_has_cysteine"; stage_labels.append(row); continue
        if "M" in pep:
            drops["hasM"] += 1; row["stage_label"] = "dropped_base_has_methionine"; stage_labels.append(row); continue
        # C-term must be hydrophobic
        if not pep or pep[-1] not in HYDRO_CTERM_SET:
            drops["nonhydro_cterm"] += 1; row["stage_label"] = "dropped_base_nonhydrophobic_Cterm"; stage_labels.append(row); continue
        # locate mutation
        if args.indexing == "auto":
            mut_idx0, idx_mode = detect_indexing(pep, pos, wt or "", mut or "")
        else:
            mut_idx0 = resolve_mut_idx(pep, pos, args.indexing); idx_mode = args.indexing
        # mutation not last
        if mut_idx0 == len(pep)-1:
            drops["mut_last"] += 1; row["stage_label"] = "dropped_mutation_at_last"; stage_labels.append(row); continue
        # validate letter
        letter = pep[mut_idx0]
        match = "none"
        if wt and letter == wt: match = "equals_wt"
        if mut and letter == mut: match = ("equals_mut" if match=="none" else "equals_both")
        if match == "none" and args.__dict__.get("strict-mutation-check", True):
            drops["mut_mismatch"] += 1; row["stage_label"] = "dropped_mutation_letter_mismatch"; stage_labels.append(row); continue

        row["_mut_idx0"] = mut_idx0; row["_indexing_used"] = idx_mode; row["_mut_letter"] = letter; row["_mut_match"] = match
        survivors.append(row)

    print("[INFO] Base filters summary:")
    kept = len(survivors)
    print(f"  total={n_total}  kept={kept}  dropped={n_total-kept}")
    for k,v in drops.items():
        print(f"    • {k}: {v}")

    if not survivors:
        raise RuntimeError("No peptides left after base filters.")

    # ---- Precompute albumin windows + frag cache for candidate ranking ----
    alb_df, alb_frag_cache = build_albumin_index(ALBUMIN_P02768_PREPRO_609, int(args.alb_kmin), int(args.alb_kmax),
                                                 charges, cys_fixed_mod, float(args.rt_total_min))

    # Fly weights
    fly_w = {k.strip(): float(v.strip()) for k,v in (tok.split(":") for tok in args.fly_weights.split(","))}

    # ---- Stage B: choose Proline then Methionine per peptide (with your ranked preferences) ----
    p_enum_records = []  # for plots/stats
    m_enum_records = []
    decoy_rows = []

    for row in tqdm(survivors, total=len(survivors), desc="Selecting Proline and Methionine"):
        pep = row["_peptide"]; L = len(pep); mut_idx0 = row["_mut_idx0"]
        # put cysteine at mutation
        cys_seq = pep[:mut_idx0] + "C" + pep[mut_idx0+1:]
        base_gravy = kd_gravy(cys_seq)
        base_fly = fly_score_norm(cys_seq, fly_w)

        # -- enumerate P positions --
        p_positions = [i for i in range(1, L-1) if i != mut_idx0 and abs(i - mut_idx0) >= 2]  # not first/last; not adjacent to C
        p_cands = []
        for p_idx in p_positions:
            x_prev = cys_seq[p_idx-1]
            rk_before = x_prev in ("R","K")
            tier = tier_for_x_before_p(x_prev)
            replaceQ = (cys_seq[p_idx] == "Q")
            # create candidate seq with P
            sP = cys_seq[:p_idx] + "P" + cys_seq[p_idx+1:]
            hyd_delta = kd_gravy(sP) - base_gravy
            flyP = fly_score_norm(sP, fly_w)
            fly_delta = flyP - base_fly
            conf = estimate_albumin_conf(sP, alb_df, alb_frag_cache, charges, float(args.ppm),
                                         float(args.rt_total_min), cys_fixed_mod)
            p_cands.append(dict(p_idx=p_idx, x_prev=x_prev, tier=tier, rk_before=rk_before, replaceQ=replaceQ,
                                distC=abs(p_idx - mut_idx0), hyd_delta=hyd_delta, fly=flyP, fly_delta=fly_delta,
                                conf_est=conf["conf_est"], chain_rt=conf["N_frag_prec_rt"], seqP=sP))
            p_enum_records.append(dict(peptide=pep, mut_idx0=mut_idx0, p_idx=p_idx, x_prev=x_prev, tier=tier,
                                       rk_before=rk_before, replaceQ=replaceQ, hyd_delta=hyd_delta,
                                       fly=flyP, fly_delta=fly_delta, conf_est=conf["conf_est"]))

        if not p_cands:
            row["stage_label"] = "dropped_no_P_positions"; stage_labels.append(row); continue

        # avoid R/K before P if alternatives exist
        if args.avoid_rk_p and any(not c["rk_before"] for c in p_cands):
            p_cands = [c for c in p_cands if not c["rk_before"]]

        # rank P lexicographically
        p_cands.sort(key=lambda c: (c["tier"],
                                    0 if c["replaceQ"] else 1,    # prefer Q→P
                                    -c["distC"],                   # farther from C
                                    -c["hyd_delta"],               # ↑hydrophobicity
                                    -c["fly"],                     # ↑fly_score_norm
                                    -c["conf_est"]))               # ↑albumin conf
        bestP = p_cands[0]
        p_idx = bestP["p_idx"]; seqP = bestP["seqP"]

        # -- enumerate M positions (respect P-between-C rule, with special case for C at 0) --
        m_positions = [i for i in range(1, L-1) if i not in (mut_idx0, p_idx) and abs(i - mut_idx0) >= 2]  # not first/last; not C/P; not adjacent to C
        m_cands = []
        for m_idx in m_positions:
            # P between C and M (unless C at 0, then allow C–P–M or C–M–P)
            p_between = (min(mut_idx0, m_idx) < p_idx < max(mut_idx0, m_idx))
            if not p_between and mut_idx0 != 0:
                continue  # enforce in general case
            # candidate M
            sPM = seqP[:m_idx] + "M" + seqP[m_idx+1:]
            hyd_delta_M = kd_gravy(sPM) - kd_gravy(seqP)
            flyPM = fly_score_norm(sPM, fly_w)
            fly_delta_M = flyPM - fly_score_norm(seqP, fly_w)
            confM = estimate_albumin_conf(sPM, alb_df, alb_frag_cache, charges, float(args.ppm),
                                          float(args.rt_total_min), cys_fixed_mod)
            replaceQ_M = (seqP[m_idx] == "Q")
            m_cands.append(dict(m_idx=m_idx, replaceQ=replaceQ_M, distC=abs(m_idx - mut_idx0),
                                distP=abs(m_idx - p_idx), hyd_delta=hyd_delta_M, fly=flyPM, fly_delta=fly_delta_M,
                                conf_est=confM["conf_est"], chain_rt=confM["N_frag_prec_rt"], seqPM=sPM,
                                p_between=p_between))
            m_enum_records.append(dict(peptide=pep, p_idx=p_idx, m_idx=m_idx, replaceQ=replaceQ_M,
                                       distC=abs(m_idx - mut_idx0), distP=abs(m_idx - p_idx),
                                       hyd_delta=hyd_delta_M, fly=flyPM, fly_delta=fly_delta_M,
                                       conf_est=confM["conf_est"], p_between=p_between))
        if not m_cands:
            row["stage_label"] = "dropped_no_M_positions"; stage_labels.append(row); continue

        # rank M lexicographically (your order)
        m_cands.sort(key=lambda c: (0 if c["replaceQ"] else 1,
                                    -c["distC"],        # farther from C
                                    -c["distP"],        # prefer spacing from P
                                    -c["hyd_delta"],    # ↑hydrophobicity
                                    -c["fly"],          # ↑fly_score_norm
                                    -c["conf_est"]))    # ↑albumin conf
        bestM = m_cands[0]
        m_idx = bestM["m_idx"]; decoy = bestM["seqPM"]

        # record decoy row
        decoy_rows.append(dict(row,
            peptide=pep, mut_idx0=mut_idx0, indexing_used=row["_indexing_used"], match_check=row["_mut_match"],
            cys_seq=cys_seq, decoy_seq=decoy,
            p_index0=p_idx, m_index0=m_idx,
            p_tier=int(bestP["tier"]), p_prev_res=str(cys_seq[p_idx-1]),
            p_replaceQ=bool(bestP["replaceQ"]), p_dist_to_C=int(bestP["distC"]),
            p_hyd_delta=float(bestP["hyd_delta"]), p_fly=float(bestP["fly"]),
            p_conf_est=float(bestP["conf_est"]),
            m_replaceQ=bool(bestM["replaceQ"]), m_dist_to_C=int(bestM["distC"]), m_dist_to_P=int(bestM["distP"]),
            m_hyd_delta=float(bestM["hyd_delta"]), m_fly=float(bestM["fly"]),
            m_conf_est=float(bestM["conf_est"]),
            decoy_fly_score=float(bestM["fly"]), decoy_fly_score_norm=float(bestM["fly"]),  # explicit alias
        ))

    if not decoy_rows:
        raise RuntimeError("No decoys after P/M selection.")

    out_df = pd.DataFrame(decoy_rows)

    # ---- Stage C: Final albumin chaining with cohort‑normalized Confusability_simple_albumin ----
    ppm_tol = float(args.ppm); rt_tol = float(args.rt_tol); rt_total_min = float(args.rt_total_min)
    # precursor matches (sets) and fragment chained counts using b/y fragment types
    N_prec_list = []; N_prec_rt_list = []
    N_frag_prec_list = []; Frac_frag_prec_list = []
    N_frag_prec_rt_list = []; Frac_frag_prec_rt_list = []

    # Pre‑gather albumin frags for arbitrary subset
    def gather(idxset: set) -> List[float]:
        out = []; 
        for j in idxset:
            arr = alb_frag_cache.get(int(j))
            if arr: out.extend(arr)
        out.sort(); return out

    for i, r in tqdm(out_df.iterrows(), total=len(out_df), desc="Final albumin chaining"):
        seq = r["decoy_seq"]
        pep_mass = calc_mass(seq, cys_fixed_mod)
        pep_mz = {z: mz_from_mass(pep_mass, z) for z in charges}
        rt_pep = predict_rt_min(seq, rt_total_min)
        matched = set(); matched_rt = set()
        for j, wr in alb_df.iterrows():
            wmz = {z: mz_from_mass(wr["mass"], z) for z in charges}
            if any(ppm(pep_mz[z], wmz[z]) <= ppm_tol for z in charges):
                matched.add(int(j))
                if abs(wr["rt_min"] - rt_pep) <= rt_tol:
                    matched_rt.add(int(j))

        # peptide fragments
        frs_pep = fragment_mz_list(seq, int(args.frag_kmin), int(args.frag_kmax), charges)
        n_types = float(len(frs_pep)) if frs_pep else 1.0

        prec_fr = gather(matched); prec_rt_fr = gather(matched_rt)

        seen_prec = set(); seen_prec_rt = set()
        for (mz,ion,k,z) in frs_pep:
            if prec_fr and any_within_ppm(mz, prec_fr, ppm_tol): seen_prec.add((ion,k,z))
            if prec_rt_fr and any_within_ppm(mz, prec_rt_fr, ppm_tol): seen_prec_rt.add((ion,k,z))

        N_prec = float(len(matched)); N_prec_rt = float(len(matched_rt))
        Frac_prec = float(len(seen_prec))/n_types; Frac_prec_rt = float(len(seen_prec_rt))/n_types

        N_prec_list.append(N_prec); N_prec_rt_list.append(N_prec_rt)
        N_frag_prec_list.append(float(len(seen_prec))); N_frag_prec_rt_list.append(float(len(seen_prec_rt)))
        Frac_frag_prec_list.append(Frac_prec); Frac_frag_prec_rt_list.append(Frac_prec_rt)

    out_df["N_precursor_mz"] = N_prec_list
    out_df["N_precursor_mz_rt"] = N_prec_rt_list
    out_df["N_fragment_mz_given_precursor"] = N_frag_prec_list
    out_df["N_fragment_mz_given_precursor_rt"] = N_frag_prec_rt_list
    out_df["Frac_fragment_mz_given_precursor"] = Frac_frag_prec_list
    out_df["Frac_fragment_mz_given_precursor_rt"] = Frac_frag_prec_rt_list

    # Confusability_simple with 95th percentile caps
    def cap95(s: pd.Series) -> float:
        p = float(pd.to_numeric(s, errors="coerce").quantile(0.95))
        return p if p > 0 else 1.0
    c1 = cap95(out_df["N_precursor_mz"]); c2 = cap95(out_df["N_precursor_mz_rt"])
    out_df["Norm_N_precursor_mz"] = (out_df["N_precursor_mz"]/c1).clip(0,1)
    out_df["Norm_N_precursor_mz_rt"] = (out_df["N_precursor_mz_rt"]/c2).clip(0,1)
    out_df["Confusability_simple_albumin"] = (
        out_df["Norm_N_precursor_mz"] + out_df["Norm_N_precursor_mz_rt"] +
        out_df["Frac_fragment_mz_given_precursor"].clip(0,1) +
        out_df["Frac_fragment_mz_given_precursor_rt"].clip(0,1)
    ) / 4.0

    # Sort by confusability then fly; greedy k‑mer de‑dup
    s2 = out_df.sort_values(by=["Confusability_simple_albumin","decoy_fly_score"], ascending=[False, False], kind="mergesort")
    N_out = int(args.N); k = int(args.dedup_k)
    if k > 0:
        def kmerset(s, k): return {s[i:i+k] for i in range(0, max(0, len(s)-k+1))}
        selected_rows = []; seen = set()
        for _, rr in s2.iterrows():
            if len(selected_rows) >= N_out: break
            kms = kmerset(rr["decoy_seq"], k)
            if seen.isdisjoint(kms):
                selected_rows.append(rr); seen.update(kms)
        sel = pd.DataFrame(selected_rows) if selected_rows else s2.head(N_out).copy()
    else:
        sel = s2.head(N_out).copy()

    # ---- Outputs ----
    prefix = args.output_prefix
    out_path = Path(args.outdir)
    out_df.to_csv(out_path / f"{prefix}_decoy_features.csv", index=False)
    sel.to_csv(out_path / f"{prefix}_selected_top_N.csv", index=False)

    # Stage labels book-keeping (keepers vs drops)
    dropped_df = pd.DataFrame(stage_labels)
    dropped_df = dropped_df.assign(peptide=dropped_df.get("_peptide"))
    all_with = pd.concat([
        out_df.assign(stage_label="selected").loc[sel.index],
        out_df.drop(sel.index, errors="ignore").assign(stage_label="not_selected"),
        dropped_df
    ], ignore_index=True, sort=False)
    all_with.to_csv(out_path / f"{prefix}_all_with_stage_labels.csv", index=False)

    counts = all_with["stage_label"].value_counts().rename_axis("stage").reset_index(name="count")
    counts.to_csv(out_path / f"{prefix}_stage_counts.csv", index=False)

    # FASTA of selected
    fasta = out_path / f"{prefix}_decoys_cysPM.fasta"
    with fasta.open("w") as fh:
        for i, r in sel.reset_index(drop=True).iterrows():
            extras = []
            for k_ in ["Gene_AA_Change","HLA","rank","Type"]:
                if k_ in r and pd.notna(r[k_]): extras.append(f"{k_}={r[k_]}")
            if "_gene_parsed" in r and pd.notna(r["_gene_parsed"]): extras.append(f"gene={r['_gene_parsed']}")
            hdr = (f">decoy_{i+1}|orig={r['peptide']}|mut_idx0={int(r['mut_idx0'])}|"
                   f"wt={str(r.get('_wt'))}|mut={str(r.get('_mut'))}|"
                   f"p_idx0={int(r['p_index0'])}|m_idx0={int(r['m_index0'])}|"
                   f"indexing={r['indexing_used']}|" + "|".join(extras))
            fh.write(hdr+"\n")
            for line in wrap(str(r["decoy_seq"]), 60): fh.write(line+"\n")

    # ---- Plots ----
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Sequence transitions (orig → Cys → Cys+P → Cys+P+M)
    try:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.55*len(sel))))
        ax.axis("off")
        col_mut="#d95f02"; col_cys="#d62728"; col_pro="#1f77b4"; col_met="#2ca02c"; default="#000"
        y = 0.95; dy = 0.25
        for _, r in sel.head(25).iterrows():
            pep = r["peptide"]; cys = r["cys_seq"]; decoy = r["decoy_seq"]
            mut_i = int(r["mut_idx0"]); p_i = int(r["p_index0"]); m_i = int(r["m_index0"])
            def draw(seq, yy, hi):
                x0 = 2.0
                for i, ch in enumerate(seq):
                    ax.text(x0 + i*0.5, yy, ch, ha="left", va="center",
                            fontfamily="DejaVu Sans Mono", fontsize=10, color=hi.get(i, default))
            draw(pep, y, {mut_i:col_mut}); y -= dy
            draw(cys, y, {mut_i:col_cys}); y -= dy
            sP = cys[:p_i]+"P"+cys[p_i+1:]
            draw(sP, y, {mut_i:col_cys, p_i:col_pro}); y -= dy
            draw(decoy, y, {mut_i:col_cys, p_i:col_pro, m_i:col_met}); y -= dy*0.5
        ax.set_title("Sequence transitions: mutant (orange) → Cys (red) → add Pro (blue) → add Met (green)")
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_sequence_transitions_selected.png", dpi=170); plt.close(fig)
    except Exception as e:
        print(f"[WARN] transitions plot failed: {e}")

    # Exclusion bars
    try:
        labels = ["WT/MUT==C","has P","has C","has M","nonhydro C-term","mut at last","mut mismatch","no P","no M"]
        vals = [
            int((all_with["stage_label"]=="dropped_base_wt_or_mut_C").sum()),
            int((all_with["stage_label"]=="dropped_base_has_proline").sum()),
            int((all_with["stage_label"]=="dropped_base_has_cysteine").sum()),
            int((all_with["stage_label"]=="dropped_base_has_methionine").sum()),
            int((all_with["stage_label"]=="dropped_base_nonhydrophobic_Cterm").sum()),
            int((all_with["stage_label"]=="dropped_mutation_at_last").sum()),
            int((all_with["stage_label"]=="dropped_mutation_letter_mismatch").sum()),
            int((all_with["stage_label"]=="dropped_no_P_positions").sum()),
            int((all_with["stage_label"]=="dropped_no_M_positions").sum()),
        ]
        fig, ax = plt.subplots()
        ax.bar(labels, vals); ax.set_ylabel("Count"); ax.set_title("Exclusions per stage")
        ax.tick_params(axis='x', rotation=30); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_bar_exclusions.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] exclusion bar plot failed: {e}")

    # Proline tier counts & RK filtering effects
    try:
        p_enum = pd.DataFrame(p_enum_records)
        if not p_enum.empty:
            tier_names = {0:"GP",1:"AP",2:"LVISTP",3:"other"}
            p_enum["tier_name"] = p_enum["tier"].map(tier_names)
            fig, ax = plt.subplots()
            p_enum["tier_name"].value_counts().reindex(["GP","AP","LVISTP","other"]).plot(kind="bar", ax=ax)
            ax.set_title("Enumerated Proline tiers"); ax.set_ylabel("count"); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_tier_enumeration.png"); plt.close(fig)

            fig, ax = plt.subplots()
            p_enum["rk_before"].value_counts().reindex([True,False]).plot(kind="bar", ax=ax)
            ax.set_xticklabels(["R/K before P","not R/K before P"], rotation=0)
            ax.set_title("R/K immediately before P (enumerated)"); ax.set_ylabel("count"); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_rk_before_counts.png"); plt.close(fig)

            fig, ax = plt.subplots()
            ax.hist(p_enum["hyd_delta"], bins=25, alpha=0.8)
            ax.set_title("ΔKD(GRAVY) after Proline placement (vs Cys‑only)"); ax.set_xlabel("ΔGRAVY"); ax.set_ylabel("frequency")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_delta_hydrophobicity_hist.png"); plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(p_enum["fly_delta"], p_enum["conf_est"], s=10, alpha=0.6)
            ax.set_xlabel("Δfly_score_norm (Pro step)"); ax.set_ylabel("albumin confusability (estimate)")
            ax.set_title("Proline step: Δfly vs confusability (all candidates)")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_scatter_deltafly_vs_conf.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] P-step plots failed: {e}")

    # Methionine step plots
    try:
        m_enum = pd.DataFrame(m_enum_records)
        if not m_enum.empty:
            fig, ax = plt.subplots()
            ax.hist(m_enum["distC"], bins=range(0, m_enum["distC"].max()+2), align="left")
            ax.set_title("Distance from C (Methionine candidates)"); ax.set_xlabel("|M - C|"); ax.set_ylabel("count")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_M_dist_to_C_hist.png"); plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(m_enum["fly_delta"], m_enum["conf_est"], s=10, alpha=0.6)
            ax.set_xlabel("Δfly_score_norm (M step)"); ax.set_ylabel("albumin confusability (estimate)")
            ax.set_title("Methionine step: Δfly vs confusability (all candidates)")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_M_scatter_deltafly_vs_conf.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] M-step plots failed: {e}")

    # Final scatter: fly vs Confusability_simple_albumin (selected highlighted)
    try:
        fig, ax = plt.subplots()
        mask_sel = out_df.index.isin(sel.index)
        ax.scatter(out_df.loc[~mask_sel,"decoy_fly_score"], out_df.loc[~mask_sel,"Confusability_simple_albumin"], s=12, alpha=0.6, label="not selected")
        ax.scatter(out_df.loc[mask_sel,"decoy_fly_score"], out_df.loc[mask_sel,"Confusability_simple_albumin"], s=16, label="selected")
        ax.set_xlabel("fly_score_norm (decoy)"); ax.set_ylabel("Confusability_simple_albumin (0..1)")
        ax.set_title("Flyability vs Albumin confusability")
        ax.legend(); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_scatter_fly_vs_confusability.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] final scatter failed: {e}")

    print(f"[INFO] Output directory: {outdir.resolve()}")
    print("[INFO] Wrote: *_decoy_features.csv, *_selected_top_N.csv, *_all_with_stage_labels.csv, *_stage_counts.csv, FASTA, plots/")

if __name__ == "__main__":
    main()

