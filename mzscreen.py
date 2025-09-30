#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WIQD — What Is Qing Doing?  (Script‑1‑aligned + contamination score)

Hit vs non‑hit peptide analysis against contaminant protein sets (ALONE), with:

  • Original Script‑2 metrics (fractions/scores)
  • Script‑1‑style metrics (counts; chained fragments; 95th‑percentile‑capped confusability)
  • NEW: Contamination evidence score in [0,1] (±RT), mostly 0, near 1 when strong explanation exists:
        - 1.0 when: precursor ≤ good_ppm (default 20) AND ≥ N fragment types at ≤ good_ppm
        - 0 < score < 1 when: precursor ≤ max_ppm (default 30) AND ≥ N fragment types at ≤ max_ppm
        - 0 otherwise
        Works with and without RT gating.

Extras:
  • --split_keratin       → splits 'keratins' into one source per accession
  • Plots for the new score (±RT) per set + cross‑set bar summaries
  • Enrichment summary CSV for every source (±RT): Δmean(hit−non), MWU p/q, prevalence at score≥0.9

Recommended to align with Script 1:
  --ppm_tol 30 --charges 1,2,3
  --full_mz_len_min 5 --full_mz_len_max 15
  --fragment_kmin 2 --fragment_kmax 7
  --cys_mod none
  --xle_collapse            # optional I/L collapse for sequence containment

Author: you
"""

import os, sys, math, argparse, datetime, random, re, bisect, json
from typing import List, Dict, Optional, Iterable, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- tqdm ----------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# ---------- Banner ----------
def banner(name: str = "WIQD SETS (alone, S1‑aligned + score)", title: str = "What Is Qing Doing?", args=None):
    w=82; bar="+"+"="*(w-2)+"+"
    def c(s): return f"|{s:^{w-2}}|"
    print("\n".join([bar,c(title),c(f"({name})"),bar]))
    if args:
        keys = ["input_csv","outdir","sets","download_sets","sets_fasta_dir","sets_accessions_json",
                "ppm_tol","charges","full_mz_len_min","full_mz_len_max","fragment_kmin","fragment_kmax",
                "rt_tolerance_min","gradient_min","require_precursor_match_for_fragment","cys_mod",
                "alpha","summary_top_n","min_nnz_topn","xle_collapse","split_keratin",
                "score_prec_ppm_good","score_prec_ppm_max","score_frag_types_req"]
        print("Args:", " | ".join(f"{k}={getattr(args,k)}" for k in keys if hasattr(args,k)))
    print()

# ---------- Matplotlib defaults ----------
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (6.8, 4.4),
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9
})
def _grid(ax): ax.grid(True, ls="--", lw=0.5, alpha=0.5)

# ---------- Chemistry ----------
AA = set("ACDEFGHIKLMNPQRSTVWY")
MONO = {
    "A": 71.037113805,"R":156.10111105,"N":114.04292747,"D":115.026943065,
    "C":103.009184505,"E":129.042593135,"Q":128.05857754,"G":57.021463735,
    "H":137.058911875,"I":113.084064015,"L":113.084064015,"K":128.09496305,
    "M":131.040484645,"F":147.068413945,"P":97.052763875,"S":87.032028435,
    "T":101.047678505,"W":186.07931298,"Y":163.063328575,"V":99.068413945,
}
H2O = 18.010564684
PROTON = 1.007276466812

def clean_pep(p): return "".join(ch for ch in str(p).strip().upper() if ch in AA)
def neutral_mass(seq, cys_fixed_mod: float=0.0): return sum(MONO[a] for a in seq) + H2O + cys_fixed_mod*seq.count("C")
def mz_from_mass(m, z): return (m + z*PROTON)/z
def ppm_diff(a: float, b: float) -> float: return (abs(a-b)/a*1e6) if a>0 else float("inf")

def collapse_xle(s: str) -> str:
    # Script‑1 behavior for sequence containment: I/L -> J
    return "".join(("J" if c in ("I","L") else c) for c in s)

def frag_b_mz(seq: str, z: int, cys_fixed_mod: float) -> float:
    return (neutral_mass(seq, cys_fixed_mod) - H2O + z*PROTON) / z

def frag_y_mz(seq: str, z: int, cys_fixed_mod: float) -> float:
    return (neutral_mass(seq, cys_fixed_mod) + z*PROTON) / z

def within_ppm_lists(x: float, sorted_list: List[float], ppm_tol: float) -> bool:
    if not sorted_list: return False
    tol = x * ppm_tol * 1e-6
    i = bisect.bisect_left(sorted_list, x)
    if i < len(sorted_list) and abs(sorted_list[i]-x) <= tol: return True
    if i>0 and abs(sorted_list[i-1]-x) <= tol: return True
    if i+1<len(sorted_list) and abs(sorted_list[i+1]-x) <= tol: return True
    return False

def within_ppm_array(x: float, arr_sorted: List[float], ppm_tol: float) -> bool:
    if not arr_sorted: return False
    tol = x * ppm_tol * 1e-6
    lo, hi = x - tol, x + tol
    L = bisect.bisect_left(arr_sorted, lo)
    R = bisect.bisect_right(arr_sorted, hi)
    return R > L

# ---------- RT surrogate ----------
def predict_rt_min(seq: str, gradient_min: float=20.0) -> float:
    if not seq: return float("nan")
    kd = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    gravy = sum(kd[a] for a in seq)/len(seq)
    frac = (gravy + 4.5)/9.0
    base = 0.5 + (gradient_min-1.0)*min(1.0, max(0.0, frac))
    length_adj = 0.03*max(0,len(seq)-8)*(gradient_min/20.0)
    basic_adj  = -0.15*(seq.count("K")+seq.count("R")+0.3*seq.count("H"))*(gradient_min/20.0)
    return float(min(max(0.0, base+length_adj+basic_adj), gradient_min))

# ---------- FASTA helpers ----------
def parse_fasta(text: str) -> Dict[str,str]:
    seqs={}; hdr=None; buf=[]
    for ln in text.splitlines():
        if not ln: continue
        if ln[0]==">":
            if hdr: seqs[hdr]="".join(buf).replace(" ","").upper()
            hdr = ln[1:].strip(); buf=[]
        else:
            buf.append(ln.strip())
    if hdr: seqs[hdr]="".join(buf).replace(" ","").upper()
    return seqs

def extract_accession(hdr: str) -> str:
    m = re.match(r"^\w+\|([^|]+)\|", hdr)
    if m: return m.group(1)
    tok = hdr.split()[0]
    if "|" in tok:
        parts = tok.split("|")
        for p in parts:
            if re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", p) or re.match(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$", p):
                return p
    return re.sub(r"[^A-Za-z0-9._-]+","_", tok)

def fetch_uniprot_fasta_for_accessions(accs: List[str], timeout: float=45.0) -> Dict[str,str]:
    if not accs: return {}
    try:
        import requests
        url = "https://rest.uniprot.org/uniprotkb/stream"
        q = " OR ".join(f"accession:{a}" for a in accs)
        params = {"query": q, "format": "fasta", "includeIsoform": "false"}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        hdr_to_seq = parse_fasta(r.text)
        acc_to_seq = {}
        for hdr, seq in hdr_to_seq.items():
            acc_to_seq[extract_accession(hdr)] = seq
        return acc_to_seq
    except Exception as e:
        print(f"[wiqd][WARN] UniProt fetch failed for {len(accs)} accs: {e}")
        return {}

# ---------- Window model ----------
@dataclass
class Window:
    seq: str
    length: int
    mass: float
    mz_by_z: Dict[int,float]
    rt_min: float
    frag_mz_sorted: List[float] = field(default_factory=list)  # b/y-only per window

def build_windows_index_for_sequences(
    sequences: List[str], len_min: int, len_max: int, charges: Iterable[int],
    gradient_min: float, cys_fixed_mod: float
) -> List[Window]:
    zs = sorted(set(int(z) for z in charges))
    out: List[Window] = []
    for s in sequences:
        clean = "".join(ch for ch in s if ch.isalpha()).upper()
        if not clean: continue
        N = len(clean)
        for L in range(max(1,len_min), min(N, len_max)+1):
            for i in range(0, N-L+1):
                sub = clean[i:i+L]
                if not set(sub) <= AA: continue
                m = neutral_mass(sub, cys_fixed_mod)
                mz_by = {z: mz_from_mass(m, z) for z in zs}
                rt = predict_rt_min(sub, gradient_min)
                out.append(Window(seq=sub, length=L, mass=m, mz_by_z=mz_by, rt_min=rt))
    return out

def index_fragments_for_windows(
    windows: List[Window],
    fragment_kmin: int, fragment_kmax: int,
    charges: Iterable[int], cys_fixed_mod: float
) -> List[float]:
    """Populate each Window.frag_mz_sorted with b/y-only fragment m/z and return a global ANY-window sorted list."""
    frag_all: List[float] = []
    for w in windows:
        arr: List[float] = []
        kmax_eff = max(fragment_kmin, min(fragment_kmax, w.length - 1))
        if kmax_eff >= fragment_kmin:
            for k in range(fragment_kmin, kmax_eff + 1):
                bseq = w.seq[:k]
                yseq = w.seq[-k:]
                for z in charges:
                    arr.append(frag_b_mz(bseq, z, cys_fixed_mod))
                    arr.append(frag_y_mz(yseq, z, cys_fixed_mod))
        arr.sort()
        w.frag_mz_sorted = arr
        frag_all.extend(arr)
    frag_all.sort()
    return frag_all

# ---------- Matching primitives (Script‑2 originals) ----------
def peptide_fragment_mz_cache(pep: str, charges: Iterable[int], cys_fixed_mod: float, kmin: int, kmax: int):
    Lp=len(pep); kmin=max(1,int(kmin)); kmax=min(int(kmax), Lp)
    mz_by_z: Dict[int,List[float]] = {int(z): [] for z in charges}
    for k in range(kmin, kmax+1):
        for i in range(0, Lp-k+1):
            sub = pep[i:i+k]
            m = neutral_mass(sub, cys_fixed_mod)
            for z in list(mz_by_z.keys()):
                mz_by_z[z].append(mz_from_mass(m, z))
    for z in list(mz_by_z.keys()):
        mz_by_z[z].sort()
    return {"mz_by_z": mz_by_z, "kmin": kmin, "kmax": kmax}

def filter_rt_idxs(idx_list: List[int], windows: List[Window], rt: float, tol_min: float) -> List[int]:
    lo, hi = rt - tol_min, rt + tol_min
    return [i for i in idx_list if (windows[i].rt_min >= lo and windows[i].rt_min <= hi)]

def filter_rt(windows: List[Window], rt: float, tol_min: float) -> List[Window]:
    if not (windows and (rt==rt)): return []
    lo, hi = rt - tol_min, rt + tol_min
    return [w for w in windows if (w.rt_min >= lo and w.rt_min <= hi)]

def count_precursor_matches(windows: List[Window], pep_mz_by_z: Dict[int,float], ppm_tol: float) -> Tuple[int,int]:
    matches=0; candidates=len(windows)
    for w in windows:
        hit=False
        for z, pmz in pep_mz_by_z.items():
            wmz = w.mz_by_z.get(z)
            if wmz is None: continue
            if ppm_diff(pmz, wmz) <= ppm_tol:
                hit=True; break
        if hit: matches+=1
    return matches, candidates

def best_precursor_error(windows: List[Window], pep_mz_by_z: Dict[int,float]) -> Dict[str,Optional[float]]:
    best = {"ppm": float("inf"), "da": float("inf"), "best_charge": None, "best_length": None}
    for w in windows:
        for z, pmz in pep_mz_by_z.items():
            wmz = w.mz_by_z.get(z)
            if wmz is None: continue
            da = abs(wmz - pmz)
            ppmv = (da/pmz)*1e6
            if ppmv < best["ppm"]:
                best.update({"ppm": ppmv, "da": da, "best_charge": z, "best_length": w.length})
    return best

def fragment_any_mz_match_counts(
    windows: List[Window], pep_frag_cache: dict, ppm_tol: float, cys_fixed_mod: float,
    kmin: int, kmax: int
) -> Tuple[int,int]:
    cand=0; mz_match=0
    mz_lists_by_z = pep_frag_cache["mz_by_z"]
    for w in windows:
        kmax_w = min(kmax, w.length)
        for k in range(kmin, kmax_w+1):
            for i in range(0, w.length-k+1):
                sub = w.seq[i:i+k]
                m = neutral_mass(sub, cys_fixed_mod)
                cand += 1
                hit=False
                for z, mz_list in mz_lists_by_z.items():
                    mzv = mz_from_mass(m, z)
                    if within_ppm_lists(mzv, mz_list, ppm_tol):
                        hit=True; break
                if hit: mz_match += 1
    return mz_match, cand

def ppm_to_score(ppm_val: float, ppm_tol: float) -> float:
    if not (isinstance(ppm_val,float) and ppm_val==ppm_val): return 0.0
    return max(0.0, 1.0 - min(ppm_val, ppm_tol)/ppm_tol)

# ---------- Script‑1‑style helpers ----------
def pep_b_y_fragments(pep: str, charges: Iterable[int], cys_fixed_mod: float,
                      kmin: int, kmax: int, xle_collapse_flag: bool):
    """Return (fr_mzs, fr_keys):
       fr_mzs: list of (mz, ion, k, z) for peptide's b/y fragments
       fr_keys: list of (ion, k, seq_c) for sequence containment checks
    """
    L = len(pep)
    kmax_eff = max(kmin, min(kmax, L - 1))
    fr_mzs = []
    fr_keys = []
    for k in range(kmin, kmax_eff + 1):
        bseq = pep[:k]
        yseq = pep[-k:]
        bseq_c = collapse_xle(bseq) if xle_collapse_flag else bseq
        yseq_c = collapse_xle(yseq) if xle_collapse_flag else yseq
        fr_keys.append(("b", k, bseq_c))
        fr_keys.append(("y", k, yseq_c))
        for z in charges:
            fr_mzs.append((frag_b_mz(bseq, z, cys_fixed_mod), "b", k, z))
            fr_mzs.append((frag_y_mz(yseq, z, cys_fixed_mod), "y", k, z))
    return fr_mzs, fr_keys

def precursor_matched_window_indices(
    windows: List[Window], pep_mz_by_z: Dict[int,float], ppm_tol: float
) -> set:
    hits = set()
    for i, w in enumerate(windows):
        ok = False
        for z, pmz in pep_mz_by_z.items():
            wmz = w.mz_by_z.get(z)
            if wmz is None: continue
            if ppm_diff(pmz, wmz) <= ppm_tol:
                ok = True; break
        if ok:
            hits.add(i)
    return hits

def count_fragment_types_over_window_subset(
    window_idxs: set, windows: List[Window],
    fr_mzs: list, ppm_tol: float
) -> int:
    """Return #unique (ion,k,z) types matched in the given window subset (ppm)."""
    if not window_idxs or not fr_mzs: return 0
    import bisect
    seen = set()
    for (mz, ion, k, z) in fr_mzs:
        lo = mz * (1 - ppm_tol*1e-6); hi = mz * (1 + ppm_tol*1e-6)
        hit = False
        for widx in window_idxs:
            arr = windows[widx].frag_mz_sorted
            if not arr: continue
            L = bisect.bisect_left(arr, lo)
            R = bisect.bisect_right(arr, hi)
            if R > L: hit = True; break
        if hit:
            seen.add((ion, k, z))
    return len(seen)

# ---------- NEW: Contamination score ----------
def contamination_score(
    windows: List[Window],
    pep_mz_by_z: Dict[int,float],
    fr_mzs: list,
    # pools are lists of indices into windows
    prec_pool_idxs: List[int],
    rt_pool_idxs: List[int],
    good_ppm: float,
    max_ppm: float,
    frag_types_req: int,
) -> Dict[str, float]:
    """Return {contam_score, contam_score_rt} in [0,1]. Mostly 0; near 1 with strong chained evidence."""
    def best_ppm_over_idx(idx_list):
        best = float("inf")
        for i in idx_list:
            w = windows[i]
            for z, pmz in pep_mz_by_z.items():
                wmz = w.mz_by_z.get(z)
                if wmz is None: continue
                best = min(best, (abs(wmz - pmz)/pmz)*1e6)
        return best

    def score_for_pool(pool_idxs):
        if not pool_idxs: return 0.0
        # precursor windows within thresholds
        wpool = [windows[i] for i in pool_idxs]
        prec20 = precursor_matched_window_indices(wpool, pep_mz_by_z, good_ppm)
        prec30 = precursor_matched_window_indices(wpool, pep_mz_by_z, max_ppm)
        has_prec20 = len(prec20) > 0
        has_prec30 = len(prec30) > 0
        if not has_prec30:
            return 0.0
        # chained fragments within thresholds
        m20 = count_fragment_types_over_window_subset({pool_idxs[i] for i in prec20}, windows, fr_mzs, good_ppm) if has_prec20 else 0
        m30 = count_fragment_types_over_window_subset({pool_idxs[i] for i in prec30}, windows, fr_mzs, max_ppm)
        # exact 1.0 condition
        if has_prec20 and m20 >= frag_types_req:
            return 1.0
        # otherwise interpolate
        best_ppm = best_ppm_over_idx(pool_idxs)
        prec_lin = 1.0 if best_ppm <= good_ppm else (0.0 if best_ppm >= max_ppm else (max_ppm - best_ppm)/(max_ppm - good_ppm))
        # fragment quality: prefer 20ppm matches but allow 30ppm evidence to contribute
        frag_lin_20 = min(1.0, m20/frag_types_req) if frag_types_req>0 else 0.0
        frag_lin_30 = min(1.0, m30/frag_types_req) if frag_types_req>0 else 0.0
        frag_lin = 0.7*frag_lin_20 + 0.3*frag_lin_30  # weight precise fragments more
        return float(max(0.0, min(1.0, prec_lin * frag_lin)))

    # Build deterministic pools
    prec_pool_idxs = list(sorted(set(prec_pool_idxs)))
    rt_pool_idxs   = list(sorted(set(rt_pool_idxs)))
    return {
        "contam_score":    score_for_pool(prec_pool_idxs),
        "contam_score_rt": score_for_pool(rt_pool_idxs)
    }

# ---------- Script‑1‑style per‑set metrics (+ score) ----------
def compute_set_metrics_script1_like(
    pep: str,
    windows_all: List[Window],
    frag_any_sorted: List[float],
    full_len_min: int, full_len_max: int,
    rt_pred_min: float, rt_tol_min: float,
    charges: Iterable[int], pep_mz_by_z: Dict[int,float],
    ppm_tol: float,
    cys_fixed_mod: float,
    fragment_kmin: int, fragment_kmax: int,
    xle_collapse_flag: bool,
    # score tunables
    good_ppm: float, max_ppm: float, frag_types_req: int,
):
    # Precursor candidate windows by length
    prec_pool_idx = [i for i,w in enumerate(windows_all) if full_len_min <= w.length <= full_len_max]
    if not prec_pool_idx:
        out = {k: 0.0 for k in [
            "N_precursor_mz","N_precursor_mz_rt","N_fragment_mz_any","Frac_fragment_mz_any",
            "N_fragment_mz_rt","Frac_fragment_mz_rt","N_fragment_mz_given_precursor","Frac_fragment_mz_given_precursor",
            "N_fragment_mz_given_precursor_rt","Frac_fragment_mz_given_precursor_rt","N_fragment_sequence_given_precursor",
            "Frac_fragment_sequence_given_precursor","N_fragment_sequence_given_precursor_rt","Frac_fragment_sequence_given_precursor_rt",
        ]}
        out.update({"contam_score":0.0,"contam_score_rt":0.0,"best_ppm":float("inf"),"best_ppm_rt":float("inf")})
        return out

    # RT-gated pool
    rt_pool_idx = filter_rt_idxs(prec_pool_idx, windows_all, rt_pred_min, rt_tol_min)

    # Precursor-matched windows (30ppm baseline for counts)
    matched_rel = precursor_matched_window_indices([windows_all[i] for i in prec_pool_idx], pep_mz_by_z, ppm_tol)
    matched_win_idxs = {prec_pool_idx[i] for i in matched_rel}
    matched_rt_rel = precursor_matched_window_indices([windows_all[i] for i in rt_pool_idx], pep_mz_by_z, ppm_tol)
    matched_win_idxs_rt = {rt_pool_idx[i] for i in matched_rt_rel}

    N_precursor_mz = float(len(matched_win_idxs))
    N_precursor_mz_rt = float(len(matched_win_idxs_rt))

    # Peptide b/y fragments
    fr_mzs, fr_keys = pep_b_y_fragments(pep, charges, cys_fixed_mod, fragment_kmin, fragment_kmax, xle_collapse_flag)
    n_types = float(len(fr_mzs)) if fr_mzs else 1.0

    # ANY-window (global) fragment matches (ungated)
    seen_any = set()
    for (mz, ion, k, z) in fr_mzs:
        if within_ppm_array(mz, frag_any_sorted, ppm_tol):
            seen_any.add((ion,k,z))
    N_fragment_mz_any = float(len(seen_any))
    Frac_fragment_mz_any = N_fragment_mz_any / n_types

    # ANY-window RT-gated
    frag_rt_all: List[float] = []
    for i in range(len(windows_all)):
        if abs(windows_all[i].rt_min - rt_pred_min) <= rt_tol_min:
            frag_rt_all.extend(windows_all[i].frag_mz_sorted)
    frag_rt_all.sort()
    seen_rt = set()
    for (mz, ion, k, z) in fr_mzs:
        if within_ppm_array(mz, frag_rt_all, ppm_tol):
            seen_rt.add((ion,k,z))
    N_fragment_mz_rt = float(len(seen_rt))
    Frac_fragment_mz_rt = N_fragment_mz_rt / n_types

    # Chained fragment m/z
    N_fragment_mz_given_precursor = float(count_fragment_types_over_window_subset(matched_win_idxs, windows_all, fr_mzs, ppm_tol))
    Frac_fragment_mz_given_precursor = N_fragment_mz_given_precursor / n_types
    N_fragment_mz_given_precursor_rt = float(count_fragment_types_over_window_subset(matched_win_idxs_rt, windows_all, fr_mzs, ppm_tol))
    Frac_fragment_mz_given_precursor_rt = N_fragment_mz_given_precursor_rt / n_types

    # Sequence containment among precursor‑matched windows
    frag_seq_set = set(fr_keys)
    def seq_c(s): return collapse_xle(s) if xle_collapse_flag else s
    matched_seq = set()
    for (ion,k,seqc) in frag_seq_set:
        for widx in matched_win_idxs:
            if seqc in seq_c(windows_all[widx].seq):
                matched_seq.add((ion,k,seqc)); break
    N_fragment_sequence_given_precursor = float(len(matched_seq))
    Frac_fragment_sequence_given_precursor = N_fragment_sequence_given_precursor / max(1.0, float(len(frag_seq_set)))

    matched_seq_rt = set()
    for (ion,k,seqc) in frag_seq_set:
        for widx in matched_win_idxs_rt:
            if seqc in seq_c(windows_all[widx].seq):
                matched_seq_rt.add((ion,k,seqc)); break
    N_fragment_sequence_given_precursor_rt = float(len(matched_seq_rt))
    Frac_fragment_sequence_given_precursor_rt = N_fragment_sequence_given_precursor_rt / max(1.0, float(len(frag_seq_set)))

    # Best ppm (all windows; and RT‑gated)
    best_all = best_precursor_error([windows_all[i] for i in prec_pool_idx], pep_mz_by_z)
    best_rt  = best_precursor_error([windows_all[i] for i in rt_pool_idx],  pep_mz_by_z) if rt_pool_idx else {"ppm": float("inf")}

    # NEW contamination score (±RT)
    score_dict = contamination_score(
        windows=windows_all,
        pep_mz_by_z=pep_mz_by_z,
        fr_mzs=fr_mzs,
        prec_pool_idxs=prec_pool_idx,
        rt_pool_idxs=rt_pool_idx,
        good_ppm=good_ppm, max_ppm=max_ppm, frag_types_req=frag_types_req
    )

    out = {
        "N_precursor_mz": N_precursor_mz,
        "N_precursor_mz_rt": N_precursor_mz_rt,
        "N_fragment_mz_any": N_fragment_mz_any,
        "Frac_fragment_mz_any": Frac_fragment_mz_any,
        "N_fragment_mz_rt": N_fragment_mz_rt,
        "Frac_fragment_mz_rt": Frac_fragment_mz_rt,
        "N_fragment_mz_given_precursor": N_fragment_mz_given_precursor,
        "Frac_fragment_mz_given_precursor": Frac_fragment_mz_given_precursor,
        "N_fragment_mz_given_precursor_rt": N_fragment_mz_given_precursor_rt,
        "Frac_fragment_mz_given_precursor_rt": Frac_fragment_mz_given_precursor_rt,
        "N_fragment_sequence_given_precursor": N_fragment_sequence_given_precursor,
        "Frac_fragment_sequence_given_precursor": Frac_fragment_sequence_given_precursor,
        "N_fragment_sequence_given_precursor_rt": N_fragment_sequence_given_precursor_rt,
        "Frac_fragment_sequence_given_precursor_rt": Frac_fragment_sequence_given_precursor_rt,
        "contam_score": score_dict["contam_score"],
        "contam_score_rt": score_dict["contam_score_rt"],
        "best_ppm": float(best_all["ppm"]),
        "best_ppm_rt": float(best_rt["ppm"]),
    }
    return out

# ---------- Default sets ----------
DEFAULT_SETS: Dict[str, List[str]] = {
    "actin_tubulin": ["P60709","P63261","Q71U36","P68363","P07437","P68371","P08670"],
    "keratins": ["P04264","P35908","P35527","P13645","P05787","P05783","P02533","P08727","Q04695","P04259"],
    "protease_autolysis": ["P00760","Q7M135"],
    "immunoglobulins": ["P01857","P01859","P01860","P01861","P01876","P01877","P01871","P01834","P0DOY2"],
    "transferrin": ["P02787"],
    "hemoglobin": ["P69905","P68871"],
    "apolipoproteins": ["P02647","P02652","P02649","P02654","P02655","P02656","P04114"],
    "streptavidin_avidin_birA": ["P22629","P02701","P06709"],
    "protein_A_G": ["P02976","P06654"],
    "caseins_gelatin": ["P02662","P02666","P02668","P02452","P08123","P02461"],
    "mhc_hardware": ["P61769","P04439","P01889","P10321"],
    "albumin": ["P02768"],
}
SET_FRIENDLY = {
    "actin_tubulin": "Actin/Tubulin (&c.)",
    "keratins": "Keratins",
    "protease_autolysis": "Protease autolysis (trypsin/Lys‑C)",
    "immunoglobulins": "Immunoglobulins",
    "transferrin": "Transferrin",
    "hemoglobin": "Hemoglobin",
    "apolipoproteins": "Apolipoproteins",
    "streptavidin_avidin_birA": "Streptavidin/Avidin/BirA",
    "protein_A_G": "Protein A/G",
    "caseins_gelatin": "Caseins / gelatin",
    "mhc_hardware": "MHC hardware (B2M/HLA)",
    "albumin": "Albumin",
}
SET_HINTS = {
    "keratins": "Skin/hair shed contamination; abundant tryptic peptides can steal MS/MS duty cycle.",
    "protease_autolysis": "Protease autolysis series (trypsin/Lys‑C).",
    "immunoglobulins": "Ig constant‑region peptides; trace blood or antibody reagents.",
    "transferrin": "Serum/plasma carryover.",
    "hemoglobin": "Red blood cell contamination; handling artifacts.",
    "apolipoproteins": "Lipoprotein/serum background; plasma workflows.",
    "streptavidin_avidin_birA": "Biotin workflows; streptavidin/avidin/BirA shedding.",
    "protein_A_G": "Antibody capture resin leachates.",
    "caseins_gelatin": "Milk/gelatin blocking reagents.",
    "mhc_hardware": "β2‑microglobulin/HLA hardware shedding in immunopeptidomics.",
    "actin_tubulin": "High‑abundance cytoskeleton; lysis/shear background.",
    "albumin": "Dominant serum protein; common carryover/bleed‑through.",
}

# ---------- Load SET sequences ----------
def load_set_sequences(set_name: str, args) -> Dict[str,str]:
    """
    Returns a dict accession->sequence (upper AA letters only).
    Priority:
      1) --sets_fasta_dir/<set_name>.fasta
      2) --sets_accessions_json (override DEFAULT_SETS)
      3) DEFAULT_SETS via UniProt (if --download_sets)
    """
    acc_map = DEFAULT_SETS.copy()
    if args.sets_accessions_json:
        try:
            with open(args.sets_accessions_json, "r") as fh:
                user_map = json.load(fh)
            for k,v in user_map.items():
                if isinstance(v, list) and all(isinstance(x,str) for x in v):
                    acc_map[k] = v
        except Exception as e:
            print(f"[sets][WARN] failed to read --sets_accessions_json: {e}")

    # Option 1: local FASTA
    if args.sets_fasta_dir:
        fpath = os.path.join(args.sets_fasta_dir, f"{set_name}.fasta")
        if os.path.isfile(fpath):
            with open(fpath, "r") as fh:
                hdr_to_seq = parse_fasta(fh.read())
            acc_to_seq = {}
            for hdr, seq in hdr_to_seq.items():
                acc_to_seq[extract_accession(hdr)] = seq
            if acc_to_seq:
                print(f"[sets] {set_name}: loaded {len(acc_to_seq)} sequences from FASTA")
                return acc_to_seq

    # Option 2/3: accessions (download if requested)
    accs = acc_map.get(set_name, [])
    if args.download_sets and accs:
        acc_to_seq = fetch_uniprot_fasta_for_accessions(accs)
        if acc_to_seq:
            print(f"[sets] {set_name}: fetched {len(acc_to_seq)} sequences from UniProt")
            return acc_to_seq
        else:
            print(f"[sets][WARN] UniProt returned 0 sequences for {set_name}")

    # No data
    if not args.download_sets:
        print(f"[sets][WARN] {set_name}: no FASTA and download disabled; skipping.")
    return {}

# ---------- Resolve overlaps & make sets disjoint ----------
def resolve_disjoint_sets(seqs_by_set: Dict[str, Dict[str,str]], sets_order: List[str]):
    """
    Make sets disjoint by accession, then by exact sequence. Earlier sets win.
    Returns:
      resolved: {set: {acc: seq}}
      overlap_matrix: pd.DataFrame of pre-dedup accession overlaps
      report_lines: human-readable report
    """
    # Accession overlap matrix (pre-dedup)
    acc_sets = {s: set(d.keys()) for s, d in seqs_by_set.items()}
    sets = [s for s in sets_order if s in acc_sets]
    mat = pd.DataFrame(0, index=sets, columns=sets, dtype=int)
    for i, si in enumerate(sets):
        for sj in sets[i:]:
            inter = len(acc_sets[si].intersection(acc_sets[sj]))
            mat.loc[si, sj] = inter
            mat.loc[sj, si] = inter

    # Resolve by accession then by exact sequence content
    assigned_acc = {}
    assigned_seq = {}
    resolved = {s: {} for s in sets_order}
    removed = {s: [] for s in sets_order}
    for s in sets_order:
        for acc, seq in seqs_by_set.get(s, {}).items():
            if acc in assigned_acc:
                removed[s].append(acc); continue
            if seq in assigned_seq:
                removed[s].append(acc); continue
            assigned_acc[acc] = s
            assigned_seq[seq] = s
            resolved[s][acc] = seq

    # Report
    lines = ["# Set overlap & resolution (disjoint sets)",
             f"Sets (ordered): {', '.join(sets_order)}",
             "Resolution rule: earlier set wins; duplicates by accession or exact sequence removed from later sets.",
             ""]
    for s in sets_order:
        n0 = len(seqs_by_set.get(s, {}))
        n_rm = len(removed.get(s, []))
        n1 = len(resolved.get(s, {}))
        lines.append(f"- {s}: original={n0}, removed={n_rm}, final={n1}")
    return resolved, mat, lines

# ---------- Stats ----------
def mannwhitney_u_p(x, y):
    n1, n2 = len(x), len(y)
    if n1==0 or n2==0: return (float("nan"), float("nan"))
    combined = x+y
    if min(combined)==max(combined): return (float("nan"), float("nan"))
    data = [(v,0) for v in x] + [(v,1) for v in y]; data.sort(key=lambda t:t[0])
    R=[0.0]*(n1+n2); i=0
    while i<len(data):
        j=i
        while j<len(data) and data[j][0]==data[i][0]: j+=1
        rank=(i+1+j)/2.0
        for k in range(i,j): R[k]=rank
        i=j
    R1=sum(R[:n1]); U1=R1 - n1*(n1+1)/2.0; U2=n1*n2 - U1; U=min(U1,U2)
    i=0; T=0
    while i<len(data):
        j=i
        while j<len(data) and data[j][0]==data[i][0]: j+=1
        t=j-i
        if t>1: T += t*(t*t-1)
        i=j
    mu=n1*n2/2.0
    sigma2 = n1*n2*(n1+n2+1)/12.0 - (n1*n2*T)/(12.0*(n1+n2)*(n1+n2-1)) if (n1+n2)>1 else 0.0
    sigma = (sigma2**0.5) if sigma2>0 else float("nan")
    if not (sigma==sigma) or sigma<=0: return (U, float("nan"))
    z=(U-mu+0.5)/sigma
    p=2.0*(1.0 - 0.5*(1.0+math.erf(abs(z)/math.sqrt(2.0))))
    return (U, max(0.0, min(1.0, p)))

def cliffs_delta(x, y):
    # small helper for effect size (optional)
    x=np.asarray(x); y=np.asarray(y)
    n1=len(x); n2=len(y)
    if n1==0 or n2==0: return float("nan")
    gt = sum((xi>yj) for xi in x for yj in y)
    lt = sum((xi<yj) for xi in x for yj in y)
    return (gt - lt) / (n1*n2)

def bh_fdr(pvals: Dict[str,float]) -> Dict[str,float]:
    valid=[(k,v) for k,v in pvals.items() if isinstance(v,float) and v==v]
    m=len(valid)
    if m==0: return {k: float("nan") for k in pvals}
    valid.sort(key=lambda kv: kv[1])
    qs=[0.0]*m
    for i,(_,p) in enumerate(valid): qs[i] = p*m/(i+1)
    for i in range(m-2,-1,-1): qs[i]=min(qs[i], qs[i+1])
    out={}
    for i,(k,_) in enumerate(valid): out[k]=min(qs[i],1.0)
    for k in pvals.keys():
        if k not in out: out[k]=float("nan")
    return out

# ---------- Plot helpers ----------
def pretty_feature_name(set_name: str, metric_key: str) -> str:
    label_set = SET_FRIENDLY.get(set_name, set_name.replace("_"," "))
    m = {
        "precursor_mz_match_fraction": f"{label_set}: precursor m/z (fraction)",
        "precursor_mz_match_fraction_rt": f"{label_set}: precursor m/z (fraction, RT‑gated)",
        "fragment_mz_match_fraction": f"{label_set}: fragment m/z (fraction)",
        "fragment_mz_match_fraction_rt": f"{label_set}: fragment m/z (fraction, RT‑gated)",
        "confusability_simple_mean": f"{label_set}: simple confusability",
        "contam_score": f"{label_set}: contamination score",
        "contam_score_rt": f"{label_set}: contamination score (RT‑gated)",
    }
    return m.get(metric_key, f"{label_set}: {metric_key.replace('_',' ')}")

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)

def save_boxplot_simple(x, y, title, ylabel, outpath):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.boxplot([x,y], showmeans=True, meanline=True)
    jitter=0.08
    ax.scatter([1+(random.random()-0.5)*2*jitter for _ in x], x, alpha=0.35, s=10)
    ax.scatter([2+(random.random()-0.5)*2*jitter for _ in y], y, alpha=0.35, s=10)
    ax.set_xticks([1,2]); ax.set_xticklabels(["non-hit","hit"])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0 if "score" in ylabel.lower() else None)
    _grid(ax); fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08); plt.close(fig)

def save_summary_plot(stats_df: pd.DataFrame, alpha: float, outpath: str,
                      top_n: int=20, min_nnz_topn: int=4):
    # Uncapped −log10(p/q); highlight q line if provided
    def neglog10(p):
        try:
            if 0.0 < float(p) <= 1.0: return -math.log10(float(p))
        except Exception: pass
        return 0.0
    df=stats_df.copy()
    if "nnz_nonhit" in df and "nnz_hit" in df:
        df = df[(df["nnz_nonhit"]>=min_nnz_topn) & (df["nnz_hit"]>=min_nnz_topn)]
    df["score"] = df.apply(lambda r: neglog10(r["q_value"]) if (isinstance(r["q_value"],float) and r["q_value"]>0) else neglog10(r["p_value"]), axis=1)
    df = df[df["score"]>0].copy()
    if df.empty:
        fig,ax=plt.subplots(figsize=(8,3.5)); ax.set_title("No signals"); ax.set_xticks([]); ax.set_yticks([]); _grid(ax)
        fig.savefig(outpath); plt.close(fig); return
    top = df.sort_values("score", ascending=False).head(int(top_n))
    fig, ax = plt.subplots(figsize=(max(8,len(top)*0.34),5))
    ax.bar(range(len(top)), top["score"].tolist())
    labels = [r["feature"] for _, r in top.iterrows()]
    ax.set_xticks(range(len(top)), labels, rotation=90)
    ax.set_ylabel("−log10(p or q)"); ax.set_title("Feature significance summary")
    if alpha and alpha>0:
        cutoff = -math.log10(alpha); ax.axhline(cutoff, ls="--", lw=1.0); ax.text(0.0, cutoff*1.02, f"q={alpha:g}", va="bottom")
    _grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def save_heatmap(df: pd.DataFrame, title: str, outpath: str, vmin=-1.0, vmax=1.0, cmap="coolwarm"):
    if df.empty:
        fig,ax=plt.subplots(figsize=(6,3)); ax.set_title(title); ax.text(0.5,0.5,"(no data)",ha="center",va="center"); ax.axis("off")
        fig.savefig(outpath); plt.close(fig); return
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(df.columns)), max(4, 0.6*len(df.index))))
    im = ax.imshow(df.values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Pearson r")
    plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def save_bar_enrichment(enrich_df: pd.DataFrame, outpath: str, col: str, title: str):
    if enrich_df.empty or col not in enrich_df.columns: 
        fig,ax=plt.subplots(); ax.set_title("No data"); fig.savefig(outpath); plt.close(fig); return
    df = enrich_df.copy()
    df = df.sort_values(col, ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.35*len(df)), 5))
    ax.bar(range(len(df)), df[col].values.tolist())
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["set"].tolist(), rotation=90)
    ax.set_ylabel(col.replace("_"," ")); ax.set_title(title)
    _grid(ax); fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

# ---------- Analysis ----------
def run(args):
    # Setup
    assert os.path.isfile(args.input_csv), "--in not found"
    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    subdirs = {
        "precursor": os.path.join(plots_dir, "precursor"),
        "fragment": os.path.join(plots_dir, "fragment"),
        "confusability": os.path.join(plots_dir, "confusability"),
        "sets": os.path.join(plots_dir, "sets"),
        "correlations": os.path.join(plots_dir, "correlations"),
        "overlaps": os.path.join(plots_dir, "overlaps"),
        "score": os.path.join(plots_dir, "score")
    }
    for d in subdirs.values(): os.makedirs(d, exist_ok=True)

    # Read input
    df = pd.read_csv(args.input_csv)
    if args.min_score is not None and "is_hit" not in df.columns:
        score_col = next((c for c in df.columns if c.lower()=="score"), None)
        if score_col is None: raise ValueError("--min_score provided but no 'score' column")
        df["is_hit"] = (pd.to_numeric(df[score_col], errors="coerce") >= float(args.min_score)).astype(int)
    assert "peptide" in df.columns and "is_hit" in df.columns, "input must have 'peptide,is_hit' (or --min_score)"
    df["peptide"] = df["peptide"].map(clean_pep)
    df = df[df["peptide"].str.len()>0].copy()
    df["is_hit"] = df["is_hit"].astype(int)
    assert set(df["is_hit"].unique()) <= {0,1}, "is_hit must be 0/1"
    g0n = (df["is_hit"]==0).sum(); g1n = (df["is_hit"]==1).sum()

    # Chemistry config
    charges = sorted({int(z.strip()) for z in args.charges.split(",") if z.strip()})
    assert charges and all(z>0 for z in charges)
    cys_fixed_mod = 57.021464 if args.cys_mod=="carbamidomethyl" else 0.0

    # Core per‑peptide mass/RT
    feat_rows=[]
    for pep in tqdm(df["peptide"], desc="[1/9] Core peptide chemistry & RT"):
        m = neutral_mass(pep, cys_fixed_mod)
        feat_rows.append({
            "peptide": pep,
            "length": len(pep),
            "mz_z1": mz_from_mass(m,1),
            "mz_z2": mz_from_mass(m,2),
            "mz_z3": mz_from_mass(m,3),
            "rt_pred_min": predict_rt_min(pep, args.gradient_min),
        })
    feat = pd.DataFrame(feat_rows)

    # Resolve set contents (load)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    seqs_by_set_raw: Dict[str, Dict[str,str]] = {}
    for s in tqdm(sets, desc="[2/9] Loading SET sequences"):
        seqs_by_set_raw[s] = load_set_sequences(s, args)

    # Optional: split keratins into one source per accession
    if args.split_keratin and "keratins" in seqs_by_set_raw:
        ker = seqs_by_set_raw.pop("keratins")
        new_names=[]
        for acc, seq in ker.items():
            nm = f"keratin_{acc}"
            seqs_by_set_raw[nm] = {acc: seq}
            new_names.append(nm)
            SET_FRIENDLY[nm] = f"Keratins:{acc}"
            SET_HINTS[nm] = SET_HINTS.get("keratins","Keratins")
        sets = [s for s in sets if s!="keratins"] + new_names
        print(f"[sets] split_keratin: created {len(new_names)} sources: {', '.join(new_names[:6])}{'...' if len(new_names)>6 else ''}")

    # Overlap analysis + disjoint sets
    resolved, overlap_mat, overlap_report = resolve_disjoint_sets(seqs_by_set_raw, sets)
    if not overlap_mat.empty:
        overlap_mat.to_csv(os.path.join(args.outdir, "set_overlap_matrix.csv"))
        save_heatmap(overlap_mat.astype(float), "Pre‑dedup accession overlap", os.path.join(subdirs["overlaps"], "overlap_accessions.png"), vmin=0, vmax=max(1.0, overlap_mat.values.max()))
    with open(os.path.join(args.outdir, "set_overlap_report.txt"), "w") as fh:
        fh.write("\n".join(overlap_report))
    with open(os.path.join(args.outdir, "sets_resolved.json"), "w") as fh:
        json.dump({k: sorted(list(v.keys())) for k,v in resolved.items()}, fh, indent=2)

    # Build windows per (disjoint) set + Script‑1‑style fragment indices
    windows_by_set: Dict[str, List[Window]] = {}
    frag_any_by_set: Dict[str, List[float]] = {}
    for s in tqdm(sets, desc="[3/9] Building windows per set"):
        seqs = list(resolved.get(s, {}).values())
        if not seqs:
            windows_by_set[s] = []
            frag_any_by_set[s] = []
            print(f"[sets][WARN] {s}: empty after resolving; skipping downstream metrics.")
            continue
        windows_by_set[s] = build_windows_index_for_sequences(
            sequences=seqs,
            len_min=min(args.full_mz_len_min, args.fragment_kmin),
            len_max=max(args.full_mz_len_max, args.fragment_kmax),
            charges=charges,
            gradient_min=args.gradient_min,
            cys_fixed_mod=cys_fixed_mod,
        )
        frag_any_by_set[s] = index_fragments_for_windows(
            windows_by_set[s],
            fragment_kmin=args.fragment_kmin, fragment_kmax=args.fragment_kmax,
            charges=charges, cys_fixed_mod=cys_fixed_mod,
        )
        print(f"[sets] {s}: indexed {len(windows_by_set[s])} windows, {len(frag_any_by_set[s])} b/y fragments")

    # Per‑peptide metrics vs each set (Script‑2 + Script‑1‑style + score)
    per_set_rows: Dict[str, List[Dict[str,object]]] = {s: [] for s in sets}
    for idx, pep in tqdm(list(enumerate(feat["peptide"])), total=len(feat), desc="[4/9] Per‑peptide metrics vs sets"):
        pep_rt = feat.iloc[idx]["rt_pred_min"]
        pmass = neutral_mass(pep, cys_fixed_mod)
        pep_mz_by_z = {z: mz_from_mass(pmass, z) for z in charges}
        frag_cache = peptide_fragment_mz_cache(pep, charges, cys_fixed_mod, args.fragment_kmin, args.fragment_kmax)

        for s in sets:
            wins = windows_by_set.get(s, [])
            if not wins:
                per_set_rows[s].append({
                    "peptide": pep,
                    # Script‑1-style (zeros)
                    "N_precursor_mz": 0.0, "N_precursor_mz_rt": 0.0,
                    "N_fragment_mz_any": 0.0, "Frac_fragment_mz_any": 0.0,
                    "N_fragment_mz_rt": 0.0,  "Frac_fragment_mz_rt": 0.0,
                    "N_fragment_mz_given_precursor": 0.0, "Frac_fragment_mz_given_precursor": 0.0,
                    "N_fragment_mz_given_precursor_rt": 0.0, "Frac_fragment_mz_given_precursor_rt": 0.0,
                    "N_fragment_sequence_given_precursor": 0.0, "Frac_fragment_sequence_given_precursor": 0.0,
                    "N_fragment_sequence_given_precursor_rt": 0.0, "Frac_fragment_sequence_given_precursor_rt": 0.0,
                    "contam_score": 0.0, "contam_score_rt": 0.0,
                    # Script‑2 originals (NaN so they don't skew means)
                    "precursor_mz_match_fraction": float("nan"),
                    "precursor_mz_match_fraction_rt": float("nan"),
                    "fragment_mz_match_fraction": float("nan"),
                    "fragment_mz_match_fraction_rt": float("nan"),
                    "precursor_confusability": float("nan"),
                    "precursor_confusability_rt": float("nan"),
                    "confusability_simple_mean": float("nan"),
                    "precursor_best_ppm": float("nan"),
                    "precursor_best_ppm_rt": float("nan"),
                    "best_ppm": float("inf"), "best_ppm_rt": float("inf"),
                })
            else:
                # Script‑1‑style metrics (+ score)
                d1 = compute_set_metrics_script1_like(
                    pep=pep,
                    windows_all=wins,
                    frag_any_sorted=frag_any_by_set[s],
                    full_len_min=args.full_mz_len_min, full_len_max=args.full_mz_len_max,
                    rt_pred_min=pep_rt, rt_tol_min=args.rt_tolerance_min,
                    charges=charges, pep_mz_by_z=pep_mz_by_z,
                    ppm_tol=args.ppm_tol,
                    cys_fixed_mod=cys_fixed_mod,
                    fragment_kmin=args.fragment_kmin, fragment_kmax=args.fragment_kmax,
                    xle_collapse_flag=args.xle_collapse,
                    good_ppm=args.score_prec_ppm_good, max_ppm=args.score_prec_ppm_max, frag_types_req=args.score_frag_types_req
                )
                # Script‑2 original metrics
                prec_pool_all = [w for w in wins if args.full_mz_len_min <= w.length <= args.full_mz_len_max]
                prec_pool_rt  = filter_rt(prec_pool_all, pep_rt, args.rt_tolerance_min)
                prec_hits_all, prec_cand_all = count_precursor_matches(prec_pool_all, pep_mz_by_z, args.ppm_tol)
                prec_hits_rt,  prec_cand_rt  = count_precursor_matches(prec_pool_rt,  pep_mz_by_z, args.ppm_tol)
                prec_frac_all = (prec_hits_all / max(1, prec_cand_all)) if prec_cand_all>0 else 0.0
                prec_frac_rt  = (prec_hits_rt  / max(1, prec_cand_rt))  if prec_cand_rt>0  else 0.0
                best_all = best_precursor_error(prec_pool_all, pep_mz_by_z)
                best_rt  = best_precursor_error(prec_pool_rt,  pep_mz_by_z) if prec_pool_rt else {"ppm": float("inf")}
                conf_prec_all = ppm_to_score(best_all["ppm"], args.ppm_tol) if best_all["ppm"]==best_all["ppm"] else 0.0
                conf_prec_rt  = ppm_to_score(best_rt["ppm"],  args.ppm_tol) if best_rt["ppm"]==best_rt["ppm"]  else 0.0

                frag_pool_all = prec_pool_all
                frag_pool_rt  = prec_pool_rt
                if args.require_precursor_match_for_fragment:
                    def gate(wpool):
                        gated=[]
                        for w in wpool:
                            ok=False
                            for z, pmz in pep_mz_by_z.items():
                                wmz = w.mz_by_z.get(z)
                                if wmz is None: continue
                                if ppm_diff(pmz, wmz) <= args.ppm_tol:
                                    ok=True; break
                            if ok: gated.append(w)
                        return gated
                    frag_pool_all = gate(prec_pool_all)
                    frag_pool_rt  = gate(prec_pool_rt)

                frag_cache = peptide_fragment_mz_cache(pep, charges, cys_fixed_mod, args.fragment_kmin, args.fragment_kmax)
                frag_mz_hits_all, frag_cand_all = fragment_any_mz_match_counts(
                    frag_pool_all, frag_cache, args.ppm_tol, cys_fixed_mod, args.fragment_kmin, args.fragment_kmax
                )
                frag_mz_hits_rt,  frag_cand_rt  = fragment_any_mz_match_counts(
                    frag_pool_rt,  frag_cache, args.ppm_tol, cys_fixed_mod, args.fragment_kmin, args.fragment_kmax
                )
                frag_frac_all = (frag_mz_hits_all / max(1, frag_cand_all)) if frag_cand_all>0 else 0.0
                frag_frac_rt  = (frag_mz_hits_rt  / max(1, frag_cand_rt )) if frag_cand_rt >0 else 0.0
                conf_simple_mean = (conf_prec_all + conf_prec_rt + frag_frac_all + frag_frac_rt) / 4.0

                d2 = {
                    "precursor_mz_match_fraction": prec_frac_all,
                    "precursor_mz_match_fraction_rt": prec_frac_rt,
                    "fragment_mz_match_fraction": frag_frac_all,
                    "fragment_mz_match_fraction_rt": frag_frac_rt,
                    "precursor_confusability": conf_prec_all,
                    "precursor_confusability_rt": conf_prec_rt,
                    "confusability_simple_mean": conf_simple_mean,
                    "precursor_best_ppm": best_all["ppm"],
                    "precursor_best_ppm_rt": best_rt["ppm"],
                }
                d_all = dict(d1); d_all.update(d2); d_all["peptide"]=pep
                per_set_rows[s].append(d_all)

    # Merge all set features into feat
    for s in sets:
        alias = f"set_{s}__"
        df_s = pd.DataFrame(per_set_rows[s])
        df_s = df_s.add_prefix(alias).rename(columns={f"{alias}peptide":"peptide"})
        feat = feat.merge(df_s, on="peptide", how="left")

    # Albumin convenience aliases (optional)
    if "albumin" in sets:
        cols_map = {
            "precursor_mz_match_fraction": "alb_precursor_mz_match_fraction",
            "precursor_mz_match_fraction_rt": "alb_precursor_mz_match_fraction_rt",
            "fragment_mz_match_fraction": "alb_fragment_mz_match_fraction",
            "fragment_mz_match_fraction_rt": "alb_fragment_mz_match_fraction_rt",
            "precursor_confusability": "alb_precursor_confusability",
            "precursor_confusability_rt": "alb_precursor_confusability_rt",
            "confusability_simple_mean": "alb_confusability_simple_mean",
            "N_precursor_mz": "alb_N_precursor_mz",
            "N_precursor_mz_rt": "alb_N_precursor_mz_rt",
            "contam_score": "alb_contam_score",
            "contam_score_rt": "alb_contam_score_rt",
        }
        for k,v in cols_map.items():
            src = f"set_albumin__{k}"
            if src in feat.columns:
                feat[v] = feat[src]

    # Join labels & write features
    feat = feat.merge(df[["peptide","is_hit"]], on="peptide", how="left")

    # Add Script‑1‑style confusability from counts + chained fragment fractions
    def add_confusability_like_script1(feat_df: pd.DataFrame, sets_list: list) -> pd.DataFrame:
        out = feat_df.copy()
        for s in sets_list:
            base = f"set_{s}__"
            c1 = f"{base}N_precursor_mz"
            c2 = f"{base}N_precursor_mz_rt"
            f1 = f"{base}Frac_fragment_mz_given_precursor"
            f2 = f"{base}Frac_fragment_mz_given_precursor_rt"
            if not all(c in out.columns for c in [c1,c2,f1,f2]): continue
            p95_c1 = out[c1].quantile(0.95); p95_c2 = out[c2].quantile(0.95)
            p95_c1 = float(p95_c1) if p95_c1>0 else 1.0
            p95_c2 = float(p95_c2) if p95_c2>0 else 1.0
            out[f"{base}Norm_N_precursor_mz"]    = (out[c1] / p95_c1).clip(0,1)
            out[f"{base}Norm_N_precursor_mz_rt"] = (out[c2] / p95_c2).clip(0,1)
            out[f"{base}Confusability_simple__script1"] = (
                out[f"{base}Norm_N_precursor_mz"].clip(0,1)
                + out[f"{base}Norm_N_precursor_mz_rt"].clip(0,1)
                + out[f1].clip(0,1)
                + out[f2].clip(0,1)
            ) / 4.0
        return out

    feat = add_confusability_like_script1(feat, sets)
    feat.to_csv(os.path.join(args.outdir, "features.csv"), index=False)

    # ---------- Stats & plotting ----------
    g0=feat[feat["is_hit"]==0]; g1=feat[feat["is_hit"]==1]
    rows=[]; pmap={}
    now=datetime.datetime.now().isoformat(timespec="seconds")

    # Script‑2 metrics (original) — per‑set plots and tests
    metric_keys = ["precursor_mz_match_fraction","precursor_mz_match_fraction_rt",
                   "fragment_mz_match_fraction","fragment_mz_match_fraction_rt",
                   "confusability_simple_mean"]

    for s in tqdm(sets, desc="[5/9] Stats & plots per set (Script‑2 metrics)"):
        base = f"set_{s}__"
        for mk in metric_keys:
            col = f"{base}{mk}"
            if col not in feat.columns: continue
            x = g0[col].dropna().tolist(); y = g1[col].dropna().tolist()
            if len(x)==0 and len(y)==0: continue
            U,p = mannwhitney_u_p(x,y)
            mean0=float(pd.Series(x).mean()) if x else float("nan")
            mean1=float(pd.Series(y).mean()) if y else float("nan")
            nnz0=sum(1 for v in x if isinstance(v,(int,float)) and v!=0)
            nnz1=sum(1 for v in y if isinstance(v,(int,float)) and v!=0)
            featurename = f"{s}:{mk}"
            rows.append({"feature": featurename, "set": s, "metric": mk, "type":"numeric", "test_used":"mannwhitney_u",
                        "n_nonhit": len(x), "n_hit": len(y),
                        "mean_nonhit": mean0, "mean_hit": mean1,
                        "p_value": p, "q_value": float("nan"),
                        "nnz_nonhit": nnz0, "nnz_hit": nnz1, "computed_at": now})
            pmap[featurename]=p

        # Per‑set summary figure (2×2 for Script‑2 fraction metrics)
        cols = [f"{base}precursor_mz_match_fraction", f"{base}precursor_mz_match_fraction_rt",
                f"{base}fragment_mz_match_fraction", f"{base}fragment_mz_match_fraction_rt"]
        labels = ["precursor","precursor_rt","fragment","fragment_rt"]
        vals = []
        for c in cols:
            if c in feat.columns:
                vals.append((g0[c].dropna().tolist(), g1[c].dropna().tolist()))
            else:
                vals.append(([],[]))
        fig, axes = plt.subplots(2,2, figsize=(7.6,5.6), constrained_layout=True)
        for i,(ax,(x,y),lab) in enumerate(zip(axes.ravel(), vals, labels)):
            if len(x)+len(y)>0:
                ax.boxplot([x,y], showmeans=True, meanline=True)
                jitter=0.08
                ax.scatter([1+(random.random()-0.5)*2*jitter for _ in x], x, alpha=0.25, s=9)
                ax.scatter([2+(random.random()-0.5)*2*jitter for _ in y], y, alpha=0.25, s=9)
                ax.set_xticks([1,2]); ax.set_xticklabels(["non-hit","hit"])
            ax.set_title(lab); ax.set_ylabel("fraction"); _grid(ax)
        fig.suptitle(SET_FRIENDLY.get(s, s.replace("_"," ")))
        fig.savefig(os.path.join(subdirs["sets"], f"{safe_name(s)}.png")); plt.close(fig)

    # Script‑1‑style metrics + NEW contamination score
    script1_metrics = [
        "N_precursor_mz","N_precursor_mz_rt",
        "N_fragment_mz_any","Frac_fragment_mz_any",
        "N_fragment_mz_rt","Frac_fragment_mz_rt",
        "N_fragment_mz_given_precursor","Frac_fragment_mz_given_precursor",
        "N_fragment_mz_given_precursor_rt","Frac_fragment_mz_given_precursor_rt",
        "contam_score","contam_score_rt",
        "Norm_N_precursor_mz","Norm_N_precursor_mz_rt",
        "Confusability_simple__script1",
    ]
    for s in tqdm(sets, desc="[6/9] Stats (S1 metrics + score) & plots"):
        base = f"set_{s}__"
        # Tests
        for mk in script1_metrics:
            col = f"{base}{mk}"
            if col not in feat.columns: continue
            x = g0[col].dropna().tolist(); y = g1[col].dropna().tolist()
            if len(x)==0 and len(y)==0: continue
            U,p = mannwhitney_u_p(x,y)
            mean0=float(pd.Series(x).mean()) if x else float("nan")
            mean1=float(pd.Series(y).mean()) if y else float("nan")
            nnz0=sum(1 for v in x if isinstance(v,(int,float)) and v!=0)
            nnz1=sum(1 for v in y if isinstance(v,(int,float)) and v!=0)
            featurename = f"{s}:{mk}"
            rows.append({"feature": featurename, "set": s, "metric": mk, "type":"numeric", "test_used":"mannwhitney_u",
                        "n_nonhit": len(x), "n_hit": len(y),
                        "mean_nonhit": mean0, "mean_hit": mean1,
                        "p_value": p, "q_value": float("nan"),
                        "nnz_nonhit": nnz0, "nnz_hit": nnz1, "computed_at": now})
            pmap[featurename]=p

        # Plots for new contamination score (±RT)
        for mk in ["contam_score","contam_score_rt"]:
            col = f"{base}{mk}"
            if col in feat.columns:
                x = g0[col].dropna().tolist(); y = g1[col].dropna().tolist()
                title = pretty_feature_name(s, mk)
                outp = os.path.join(subdirs["score"], f"{safe_name(s)}{'_rt' if mk.endswith('_rt') else ''}.png")
                save_boxplot_simple(x,y,title,"score [0..1]",outp)

    # Assemble stats, FDR, and global summaries
    stats_df = pd.DataFrame(rows)
    qmap = bh_fdr(pmap)
    stats_df["q_value"] = stats_df["feature"].map(qmap)
    stats_df.to_csv(os.path.join(args.outdir, "stats_summary.csv"), index=False)

    save_summary_plot(
        stats_df, args.alpha,
        os.path.join(args.outdir, "summary_feature_significance.png"),
        top_n=args.summary_top_n, min_nnz_topn=args.min_nnz_topn
    )

    # ---------- Cross‑set enrichment summary using the new score ----------
    enrich_rows=[]
    for s in sets:
        for mk in ["contam_score","contam_score_rt"]:
            col = f"set_{s}__{mk}"
            if col not in feat.columns: continue
            x = pd.to_numeric(g0[col], errors="coerce")
            y = pd.to_numeric(g1[col], errors="coerce")
            x = x[~x.isna()]; y = y[~y.isna()]
            if len(x)==0 and len(y)==0: continue
            U,p = mannwhitney_u_p(x.tolist(), y.tolist())
            q = qmap.get(f"{s}:{mk}", float("nan"))
            delta_mean = float(y.mean() - x.mean()) if (len(x)>0 and len(y)>0) else float("nan")
            cd = cliffs_delta(y.tolist(), x.tolist())
            prev_non = float((x >= 0.9).mean()) if len(x)>0 else float("nan")
            prev_hit = float((y >= 0.9).mean()) if len(y)>0 else float("nan")
            enrich_rows.append({
                "set": s, "metric": mk,
                "mean_nonhit": float(x.mean()) if len(x)>0 else float("nan"),
                "mean_hit": float(y.mean()) if len(y)>0 else float("nan"),
                "delta_mean_hit_minus_non": delta_mean,
                "cliffs_delta": cd,
                "p_value": p, "q_value": q,
                "prevalence_score_ge_0_9_nonhit": prev_non,
                "prevalence_score_ge_0_9_hit": prev_hit,
                "n_nonhit": int(len(x)), "n_hit": int(len(y)),
            })
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(os.path.join(args.outdir, "contam_enrichment_summary.csv"), index=False)

    # Bar summaries: Δmean for score (±RT)
    if not enrich_df.empty:
        sub = enrich_df[enrich_df["metric"]=="contam_score"][["set","delta_mean_hit_minus_non"]].copy()
        if not sub.empty:
            save_bar_enrichment(sub.rename(columns={"delta_mean_hit_minus_non":"delta_mean"}),
                                os.path.join(plots_dir,"summary_contam_score_delta.png"),
                                col="delta_mean", title="Enrichment by source (score, hits−non‑hits)")
        subrt = enrich_df[enrich_df["metric"]=="contam_score_rt"][["set","delta_mean_hit_minus_non"]].copy()
        if not subrt.empty:
            save_bar_enrichment(subrt.rename(columns={"delta_mean_hit_minus_non":"delta_mean"}),
                                os.path.join(plots_dir,"summary_contam_score_rt_delta.png"),
                                col="delta_mean", title="Enrichment by source (score RT, hits−non‑hits)")

    # ---------- Correlations ----------
    conf_cols = {}
    for s in sets:
        col = f"set_{s}__confusability_simple_mean"
        if col in feat.columns:
            conf_cols[f"{s} (S2)"] = feat[col]
        col2 = f"set_{s}__Confusability_simple__script1"
        if col2 in feat.columns:
            conf_cols[f"{s} (S1)"] = feat[col2]
        col3 = f"set_{s}__contam_score"
        if col3 in feat.columns:
            conf_cols[f"{s} (score)"] = feat[col3]
    conf_df = pd.DataFrame(conf_cols)
    corr_sets = conf_df.corr().fillna(0.0)
    save_heatmap(corr_sets, "Correlation across sets (confusability & score)", os.path.join(subdirs["correlations"], "corr_sets.png"))

    # ---------- README with diagnostics ----------
    alpha = args.alpha
    diag_lines = ["# WIQD SETS (alone) — Summary & Diagnostics (S1‑aligned + score)",
                  f"Input peptides: {len(df)} | n(non‑hit)={g0n} | n(hit)={g1n}",
                  f"Sets evaluated: {', '.join(sets)}", ""]
    # Top signals table (any feature)
    diag_lines.append("## Strongest set×metric signals (q<α first, then p)")
    top = stats_df.sort_values(["q_value","p_value"], na_position="last").head(30)
    if top.empty:
        diag_lines.append("(no significant differences)")
    else:
        for _, r in top.iterrows():
            diag_lines.append(f"- {r['feature']}: mean(non-hit)={r['mean_nonhit']:.3g}, mean(hit)={r['mean_hit']:.3g}, p={r['p_value']:.3g}, q={r['q_value']:.3g}")

    # Contamination‑score enrichment summary (all sources)
    diag_lines.append("\n## Contamination score enrichment by source (hits − non‑hits)")
    if enrich_df.empty:
        diag_lines.append("(no score signals)")
    else:
        sub = enrich_df[enrich_df["metric"]=="contam_score"].sort_values("delta_mean_hit_minus_non", ascending=False)
        for _, r in sub.iterrows():
            diag_lines.append(f"- {SET_FRIENDLY.get(r['set'], r['set'])}: Δmean={r['delta_mean_hit_minus_non']:.3f}, p={r['p_value']:.3g}, q={r['q_value']:.3g}, prev≥0.9 (hit/non)={r['prevalence_score_ge_0_9_hit']:.2f}/{r['prevalence_score_ge_0_9_nonhit']:.2f}")

    # Files guide
    diag_lines += ["",
        "## Where to look",
        "1) `summary_feature_significance.png` — strongest set×metric signals (uncapped −log10 p/q).",
        "2) `contam_enrichment_summary.csv` — per‑source enrichment with the new [0,1] score (±RT).",
        "3) Plots:",
        "   - contamination score: `plots/score/<set>.png` and `<set>_rt.png`",
        "   - precursor/fragment fractions: `plots/precursor/*.png`, `plots/fragment/*.png`",
        "   - confusability: `plots/confusability/*.png`",
        "   - per‑set summary (fractions): `plots/sets/<set>.png`",
        "   - cross‑set bars: `summary_contam_score_delta.png` and `_rt_*.png`",
        "4) Overlaps: `set_overlap_matrix.csv` + `plots/overlaps/overlap_accessions.png`.",
        "",
        "## New contamination score (definition)",
        f"- Parameters: good_ppm={args.score_prec_ppm_good:g}, max_ppm={args.score_prec_ppm_max:g}, frag_types_req={args.score_frag_types_req}",
        "- 1.0 when: ≥1 precursor‑matched window at ≤good_ppm AND ≥frag_types_req b/y fragment types (k=2..7, z=1..3) at ≤good_ppm, chained to those windows.",
        "- Partial credit when: best precursor ≤max_ppm AND ≥frag_types_req fragment types at ≤max_ppm, scaled by proximity and fragment count.",
        "- 0 otherwise. RT‑gated variant uses only windows within ±RT tolerance.",
    ]
    with open(os.path.join(args.outdir, "README_SETS.md"), "w") as fh:
        fh.write("\n".join(diag_lines))

def main():
    ap = argparse.ArgumentParser(
        description="WIQD — hit vs non‑hit peptide analysis against contaminant SETS (alone; Script‑1‑aligned core + [0,1] score)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with peptide,is_hit (or provide --min_score and a score column)")
    ap.add_argument("--outdir", default="wiqd_sets_out", help="Output directory")

    # Sets
    ap.add_argument("--sets", type=str,
                    default="keratins,protease_autolysis,immunoglobulins,transferrin,hemoglobin,apolipoproteins,streptavidin_avidin_birA,protein_A_G,caseins_gelatin,mhc_hardware,albumin,actin_tubulin",
                    help="Comma‑sep list of SET names to evaluate (alone)")
    ap.add_argument("--download_sets", action="store_true", help="Fetch SET sequences from UniProt (recommended)")
    ap.add_argument("--sets_fasta_dir", type=str, default=None, help="Directory of per‑set FASTAs named <set>.fasta")
    ap.add_argument("--sets_accessions_json", type=str, default=None, help="JSON mapping {set_name: [accessions,...]} to override defaults")
    ap.add_argument("--split_keratin", action="store_true", help="Split 'keratins' set into one source per accession")

    # Matching controls
    ap.add_argument("--ppm_tol", type=float, default=30.0, help="PPM tolerance for m/z matching")
    ap.add_argument("--charges", type=str, default="1,2,3", help="Comma‑sep charge states to test")
    ap.add_argument("--full_mz_len_min", type=int, default=5, help="Window min length for **precursor** matching")
    ap.add_argument("--full_mz_len_max", type=int, default=15, help="Window max length for **precursor** matching")
    ap.add_argument("--fragment_kmin", type=int, default=2, help="Min length of b/y **fragment** (Script‑1) and peptide‑substring (Script‑2) checks")
    ap.add_argument("--fragment_kmax", type=int, default=7, help="Max length of b/y **fragment** (Script‑1) and peptide‑substring (Script‑2) checks")
    ap.add_argument("--rt_tolerance_min", type=float, default=1.0, help="RT co‑elution tolerance (minutes)")
    ap.add_argument("--gradient_min", type=float, default=20.0, help="Assumed gradient length (minutes) in RT surrogate")
    ap.add_argument("--require_precursor_match_for_fragment", action="store_true",
                    help="(Script‑2 only) If set, fragment matching only considers windows that ALSO match precursor m/z within tolerance")

    # Chemistry
    ap.add_argument("--cys_mod", choices=["none","carbamidomethyl"], default="carbamidomethyl",
                    help="Fixed mod on Cys for mass calc")
    ap.add_argument("--xle_collapse", action="store_true",
                    help="Treat I/L as indistinguishable for sequence containment (I/L→J), like Script 1")

    # Score tunables
    ap.add_argument("--score_prec_ppm_good", type=float, default=20.0, help="Good precursor ppm threshold for score=1")
    ap.add_argument("--score_prec_ppm_max",  type=float, default=30.0, help="Max precursor ppm to get non‑zero score")
    ap.add_argument("--score_frag_types_req", type=int, default=3, help="Min # of b/y fragment types at threshold to consider 'several'")

    # Stats & summaries
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold (for guidance lines/diagnostics)")
    ap.add_argument("--summary_top_n", type=int, default=24, help="Top‑N features to show in summary plot")
    ap.add_argument("--min_nnz_topn", type=int, default=4, help="Min nnz per group to include in Top‑N summary")
    ap.add_argument("--min_score", type=float, default=None, help="If set, derive is_hit = 1[score >= min_score] from a 'score' column")
    args = ap.parse_args()
    banner(args=args)
    run(args)

if __name__ == "__main__":
    main()
