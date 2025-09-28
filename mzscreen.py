#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiqd_mzscreen_sets_chained.py — Protein-SET m/z screen with chained fragment evidence

What it does (per peptide × protein set × mode):
  1) Precursor m/z matches to protein subsequences (length 5..15 by default) at z∈{1,2,3}, ±ppm:
       - N_precursor_mz
       - N_precursor_mz_rt             (also requires RT co-elution within ±rt_tol on a rt_total_min column)

  2) Fragment m/z matches (b/y, k=2..7, z∈{1,2,3}, ±ppm):
       - N_fragment_mz_any,  Frac_fragment_mz_any       (any window)
       - N_fragment_mz_rt,   Frac_fragment_mz_rt        (any window, RT-gated)

     NEW (chained to the precursor-matched window set):
       - N_fragment_mz_given_precursor,      Frac_fragment_mz_given_precursor
       - N_fragment_mz_given_precursor_rt,   Frac_fragment_mz_given_precursor_rt   (precursor + RT)

  3) Fragment **sequence** containment (b_k prefix / y_k suffix strings) among precursor-matched windows:
       - N_fragment_sequence_given_precursor,      Frac_fragment_sequence_given_precursor
       - N_fragment_sequence_given_precursor_rt,   Frac_fragment_sequence_given_precursor_rt  (precursor + RT)

  4) Simple confusability (bounded [0,1]):
       Confusability_simple = mean( Norm_N_precursor_mz,
                                    Norm_N_precursor_mz_rt,
                                    Frac_fragment_mz_given_precursor,
                                    Frac_fragment_mz_given_precursor_rt )
     where Norm_N_precursor_* are capped at each set×mode 95th percentile then scaled to [0,1].

Outputs (all per-peptide, normalized where applicable):
  - per_peptide_scores__ALL.csv
  - effects_summary__ALL.csv            (Cliff’s δ, group means, prevalence; if is_hit present)
  - per_peptide_per_protein_counts__ALL.csv  (for protein–protein correlation)
  - plots/
      * bar_delta__<metric>__mode_<MODE>.png
      * box_<metric>__<SET>__mode_<MODE>__FLAG.png
      * heatmap_score_corr__<SET>__mode_<MODE>.png
      * heatmap_protein_corr__<SET>__mode_<MODE>.png
  - SUMMARY_STATS.md

Defaults: subseq 5..15, fragment k=2..7, charges 1/2/3, ±30 ppm, RT ±1 min on a 20 min column.
"""

import os, argparse, re, datetime, json, math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Plot style ----
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (8.5, 5.4),
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

def add_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# ---- tqdm optional ----
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# ---- Mass constants (monoisotopic) ----
PROTON_MASS = 1.007276466812
WATER_MASS = 18.010564684
MASS = {
    "A": 71.037113805, "R": 156.101111050, "N": 114.042927470, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.058577540, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.094963050,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.079312980, "Y": 163.063328575, "V": 99.068413945,
    "U": 150.953633405
}

# ---- Base/common accessions ----
COMMON_ACCESSIONS = [
    "P60709","P04406","P07437","P11142","P08238","P68104","P05388","P62805",
    "P11021","P14625","P27824","P27797","P07237","P30101","Q15084","Q15061","Q9BS26",
    "P02768","P01857","P01859","P01860","P01861","P01876","P01877","P01871","P04229",
    "P01854","P01834","P0DOY2"
]

# ---- Default sets (by accession) ----
DEFAULT_SETS = {
    "albumin": ["P02768"],
    "antibody": ["P01857","P01859","P01860","P01861","P01876","P01877","P01871","P01834","P0DOY2"],
    "keratin": ["P04264","P02533","P04259","P02538","P05787","P05783","P08727","Q04695","P35908","P35527"],
    # serum without albumin
    "serum": ["P02647","P02652","P02649","P02787","P02671","P02675","P02679","P01009","P68871","P69905"],
    "cytoskeleton": ["P60709","P68104","P07437","P68371","Q71U36","Q9BQE3"],
    "chaperone": ["P11142","P07900","P08238","P10809","P38646"]
}

# ---- FASTA helpers ----
def parse_fasta(text: str) -> Dict[str, str]:
    seqs = {}
    hdr=None; buf=[]
    for line in text.splitlines():
        if not line: continue
        if line[0] == ">":
            if hdr:
                seqs[hdr] = "".join(buf).replace(" ", "").upper()
            hdr = line[1:].strip()
            buf = []
        else:
            buf.append(line.strip())
    if hdr: seqs[hdr] = "".join(buf).replace(" ", "").upper()
    return seqs

def open_maybe_gzip(path: str):
    import gzip
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def extract_accession(header: str) -> str:
    m = re.match(r"^\w+\|([^|]+)\|", header)
    return m.group(1) if m else header.split()[0]

def short_name(hdr: str) -> str:
    m = re.match(r"^\w+\|([^|]+)\|([^ ]+)", hdr)
    return m.group(2) if m else hdr.split()[0]

# ---- Protein loading ----
def load_common_proteins(prot_fasta: Optional[str], download_common: bool, accessions_path: Optional[str]) -> Dict[str, str]:
    seqs = {}
    if prot_fasta and os.path.isfile(prot_fasta):
        with open_maybe_gzip(prot_fasta) as fh:
            return parse_fasta(fh.read())
    accs = []
    if accessions_path and os.path.isfile(accessions_path):
        with open(accessions_path, "r") as f:
            for line in f:
                t=line.strip()
                if t and not t.startswith("#"): accs.append(t.split()[0])
    else:
        accs = list(dict.fromkeys(COMMON_ACCESSIONS))
    if download_common:
        try:
            import requests
            url = "https://rest.uniprot.org/uniprotkb/stream"
            query = " OR ".join(f"accession:{a}" for a in accs)
            params = {"query": query, "format": "fasta", "includeIsoform": "false"}
            print("[sets-chained] downloading base/common proteins via UniProt REST ...")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            seqs = parse_fasta(r.text)
        except Exception as e:
            print(f"[sets-chained] WARNING: download_common failed ({e}); using embedded fallback.")
    if not seqs:
        # Minimal embedded fallback (HSPA8)
        seqs = {"sp|P11142|HSP7C_HUMAN": (
            "MSKGPAVGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVA"
            "MNPTNTVFDAKRLIGRRFDDAVVQSDMKHWPFMVVNDAGRPKVQVEYKGETKSFYPEEVS"
            "SMVLTKMKEIAEAYLGKTVTNAVVTVPAYFNDSQRQATKDAGTIAGLNVLRIINEPTAAA"
            "IAYGLDKKVGAERNVLIFDLGGGTFDVSILTIEDGIFEVKSTAGDTHLGGEDFDNRMVNH"
            "FIAEFKRKHKKDISENKRAVRRLRTACERAKRTLSSSTQASIEIDSLYEGIDFYTSITRA"
            "RFEELNADLFRGTLDPVEKALRDAKLDKSQIHDIVLVGGSTRIPKIQKLLQDFFNGKELN"
            "KSINPDEAVAYGAAVQAAILSGDKSENVQDLLLLDVTPLSLGIETAGGVMTVLIKRNTTI"
            "PTKQTQTFTTYSDNQPGVLIQVYEGERAMTKDNNLLGKFELTGIPPAPRGVPQIEVTFDI"
            "DANGILNVSAVDKSTGKENKITITNDKGRLSKEDIERMVQEAEKYKAEDEKQRDKVSSKN"
            "SLESYAFNMKATVEDEKLQGKINDEDKQKILDKCNEIINWLDKNQTAEKEEFEHQQKELE"
            "KVCNPIITKLYQSAGGMPGGMPGGFPGGGAPPSGGASSGPTIEEVD"
        )}
    return seqs

def download_uniprot_by_accessions(accessions: List[str]) -> Dict[str, str]:
    try:
        import requests
        url = "https://rest.uniprot.org/uniprotkb/stream"
        query = " OR ".join(f"accession:{a}" for a in accessions)
        params = {"query": query, "format": "fasta", "includeIsoform": "false"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return parse_fasta(r.text)
    except Exception as e:
        print(f"[sets-chained] WARNING: UniProt download failed for set accessions ({e}).")
        return {}

def select_seqs_by_accessions(seqs: Dict[str,str], accessions: List[str]) -> Dict[str,str]:
    accset = set(accessions)
    return {hdr: s for hdr, s in seqs.items() if extract_accession(hdr) in accset}

def merge_seqs(a: Dict[str,str], b: Dict[str,str]) -> Dict[str,str]:
    out = dict(a); out.update(b); return out

def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")

# ---- Chemistry helpers ----
def calc_mass(seq: str) -> float:
    m = 0.0
    for a in seq:
        if a not in MASS: return float("nan")
        m += MASS[a]
    return m + WATER_MASS

def mz_from_mass(neutral_mass: float, z: int) -> float:
    return (neutral_mass + z*PROTON_MASS)/z

def ppm_window_mz(center_mz: float, ppm: float) -> Tuple[float,float]:
    delta = center_mz * ppm * 1e-6
    return center_mz - delta, center_mz + delta

def collapse_xle(s: str) -> str:
    return "".join("J" if c in ("L","I") else c for c in s)

# ---- Fragment m/z helpers ----
def frag_b_mz(seq: str, z: int) -> float:
    # b-ion: peptide mass without water + z*H divided by z
    return (calc_mass(seq) - WATER_MASS + z*PROTON_MASS) / z

def frag_y_mz(seq: str, z: int) -> float:
    # y-ion: peptide mass (includes water) + z*H divided by z
    return (calc_mass(seq) + z*PROTON_MASS) / z

# ---- RT model (simple KD-based, scaled to column length) ----
KD = {
    "I": 4.5,"V": 4.2,"L": 3.8,"F": 2.8,"C": 2.5,"M": 1.9,"A": 1.8,"G": -0.4,"T": -0.7,"S": -0.8,
    "W": -0.9,"Y": -1.3,"P": -1.6,"H": -3.2,"E": -3.5,"Q": -3.5,"D": -3.5,"N": -3.5,"K": -3.9,"R": -4.5,"U": 0.0
}
HYDRO = set(list("AVILMWFY"))

def rt_raw_score(seq: str) -> float:
    if not seq: return 0.0
    vals = [KD.get(a, 0.0) for a in seq]
    mean_kd = float(np.mean(vals)) if vals else 0.0
    frac_hyd = sum(a in HYDRO for a in seq)/max(1,len(seq))
    return mean_kd + 0.5*frac_hyd

def map_to_minutes(raw_vals: np.ndarray, rt_total_min: float) -> np.ndarray:
    if raw_vals.size == 0: return raw_vals
    vmin = float(np.min(raw_vals)); vmax = float(np.max(raw_vals))
    if math.isclose(vmin, vmax): return np.full_like(raw_vals, rt_total_min/2.0)
    return (raw_vals - vmin) / (vmax - vmin) * rt_total_min

# ---- Window and fragment indices ----
def build_window_index(seqs: Dict[str,str], kmin: int, kmax: int) -> pd.DataFrame:
    rows = []
    for hdr, seq in tqdm(seqs.items(), desc="[sets-chained] indexing protein windows"):
        clean = "".join(c for c in seq if c.isalpha()).upper()
        L = len(clean)
        if L == 0: continue
        for k in range(kmin, kmax+1):
            if L < k: continue
            for start in range(0, L-k+1):
                window = clean[start:start+k]
                if any(ch not in MASS for ch in window): continue
                rows.append({
                    "protein": hdr,
                    "accession": extract_accession(hdr),
                    "protein_short": short_name(hdr),
                    "start": start,
                    "length": k,
                    "window": window,
                    "mass": calc_mass(window),
                    "rt_raw": rt_raw_score(window),
                })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.sort_values("mass", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def build_fragment_index(win_df: pd.DataFrame, frag_kmin: int, frag_kmax: int, frag_charges: List[int]) -> pd.DataFrame:
    idx_rows = []
    for widx, wr in tqdm(win_df.iterrows(), total=len(win_df), desc="[sets-chained] indexing window fragments"):
        wseq = wr["window"]; Lw = len(wseq)
        for k in range(frag_kmin, min(frag_kmax, Lw-1)+1):
            # b_k
            subseq_b = wseq[:k]
            for z in frag_charges:
                idx_rows.append({"frag_mz": frag_b_mz(subseq_b, z), "ion": "b", "k": k, "z": z,
                                 "window_idx": int(widx), "rt_raw": wr["rt_raw"]})
            # y_k
            subseq_y = wseq[-k:]
            for z in frag_charges:
                idx_rows.append({"frag_mz": frag_y_mz(subseq_y, z), "ion": "y", "k": k, "z": z,
                                 "window_idx": int(widx), "rt_raw": wr["rt_raw"]})
    frag_df = pd.DataFrame(idx_rows)
    if frag_df.empty: return frag_df
    frag_df.sort_values("frag_mz", inplace=True, kind="mergesort")
    frag_df.reset_index(drop=True, inplace=True)
    return frag_df

# ---- Effect size (Cliff's δ) ----
def mannwhitney_auc_cliffs(x_hit: pd.Series, x_non: pd.Series):
    x = pd.to_numeric(x_hit, errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(x_non, errors="coerce").fillna(0.0).to_numpy()
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return np.nan, np.nan
    comb = np.concatenate([x, y])
    ranks = pd.Series(comb).rank(method="average").to_numpy()
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    auc = U1 / (n1*n2)
    delta = 2.0*auc - 1.0
    return float(auc), float(delta)

def classify_delta(delta: float) -> str:
    if pd.isna(delta): return "NA"
    a=abs(delta)
    if a<0.147: return "negligible"
    if a<0.33:  return "small"
    if a<0.474: return "medium"
    return "large"

# ---- Matching primitives ----
def binary_search_mass(masses_sorted: np.ndarray, mmin: float, mmax: float) -> Tuple[int,int]:
    import bisect
    left = bisect.bisect_left(masses_sorted, mmin)
    right = bisect.bisect_right(masses_sorted, mmax)
    return left, right

def screen_precursor_pairs(peptides_df: pd.DataFrame, win_df: pd.DataFrame,
                           charges: List[int], prec_ppm: float) -> Dict[str, set]:
    """
    Return mapping: pairs[pep] = set of window indices with ≥1 precursor m/z match (any allowed z).
    """
    pairs = {pep: set() for pep in peptides_df["peptide"]}
    masses_sorted = win_df["mass"].values
    for _, r in tqdm(peptides_df.iterrows(), total=len(peptides_df), desc="[sets-chained] precursor m/z matching"):
        pep = r["peptide"]; pmass = r["mass"]
        for z in charges:
            q_mz = mz_from_mass(pmass, z)
            mzmin, mzmax = ppm_window_mz(q_mz, prec_ppm)
            mmin = z*mzmin - z*PROTON_MASS
            mmax = z*mzmax - z*PROTON_MASS
            left, right = binary_search_mass(masses_sorted, min(mmin,mmax), max(mmin,mmax))
            if right <= left: 
                continue
            idxs = win_df.index.values[left:right]
            pairs[pep].update(idxs.tolist())
    return pairs

# ---- Core per-set×mode computation ----
def compute_per_peptide_metrics_for_set(
    set_name: str, mode: str,
    df_peptides: pd.DataFrame,
    base_seqs: Dict[str,str], set_seqs: Dict[str,str],
    kmin_subseq: int, kmax_subseq: int,
    charges: List[int], prec_ppm: float,
    frag_kmin: int, frag_kmax: int, frag_charges: List[int], frag_ppm: float,
    rt_total_min: float, rt_tol: float,
    xle_collapse: bool = False,
    prot_corr_top: int = 30
):
    """
    Returns:
      scores_rows: list of dicts per peptide with all metrics (see header)
      per_protein_df: per-peptide per-protein precursor-match counts
    """
    # Sequences by mode
    seqs = dict(set_seqs) if mode == "alone" else merge_seqs(base_seqs, set_seqs)
    if not seqs: return [], pd.DataFrame()

    # Windows and fragments
    win_df = build_window_index(seqs, kmin_subseq, kmax_subseq)
    if win_df.empty: return [], pd.DataFrame()
    frag_df = build_fragment_index(win_df, frag_kmin, frag_kmax, frag_charges)

    # Precursor matched window sets
    pairs = screen_precursor_pairs(df_peptides, win_df, charges, prec_ppm)

    # Predict RT jointly for peptides and windows (shared scaling)
    pep_rt_raw = df_peptides["peptide"].apply(rt_raw_score).values
    all_rt_raw = np.concatenate([pep_rt_raw, win_df["rt_raw"].values])
    all_rt_min = map_to_minutes(all_rt_raw, rt_total_min)
    pep_rt_min = all_rt_min[:len(pep_rt_raw)]
    win_rt_min = all_rt_min[len(pep_rt_raw):]
    win_df = win_df.copy()
    win_df["rt_min"] = win_rt_min

    frag_df = frag_df.copy()
    frag_df["rt_min"] = win_df.loc[frag_df["window_idx"].values, "rt_min"].values

    # Fast access
    pep_list = df_peptides["peptide"].tolist()
    pep_mass = df_peptides["mass"].values
    pep_rt_map = {pep_list[i]: pep_rt_min[i] for i in range(len(pep_list))}
    win_seq_list = win_df["window"].tolist()
    win_seq_list_xle = [collapse_xle(s) if xle_collapse else s for s in win_seq_list]

    # Fragment search helpers
    frag_mz_vals_all = frag_df["frag_mz"].values  # sorted
    def any_frag_mz_match_global(q_mz: float, ppm: float) -> bool:
        # binary search in global sorted list
        import bisect
        delta = q_mz * ppm * 1e-6
        lo, hi = q_mz - delta, q_mz + delta
        L = bisect.bisect_left(frag_mz_vals_all, lo)
        R = bisect.bisect_right(frag_mz_vals_all, hi)
        return R > L

    def peptide_fragments(pep: str):
        Lp = len(pep)
        kmax_eff = max(frag_kmin, min(frag_kmax, Lp-1))
        ks = list(range(frag_kmin, kmax_eff+1)) if kmax_eff >= frag_kmin else []
        fr_mzs = []   # (mz, ion, k, z)
        fr_keys = []  # (ion, k, seq_c)
        for k in ks:
            bseq = pep[:k]; yseq = pep[-k:]
            bseq_c = collapse_xle(bseq) if xle_collapse else bseq
            yseq_c = collapse_xle(yseq) if xle_collapse else yseq
            fr_keys.append(("b", k, bseq_c)); fr_keys.append(("y", k, yseq_c))
            for z in frag_charges:
                fr_mzs.append((frag_b_mz(bseq, z), "b", k, z))
                fr_mzs.append((frag_y_mz(yseq, z), "y", k, z))
        return fr_mzs, fr_keys

    # Count fragment m/z matches restricted to a subset of windows (for chaining)
    # Build a quick mapping window_idx -> sorted fragment m/z array (numpy)
    fr_by_win = None
    if not frag_df.empty:
        fr_by_win = (frag_df.groupby("window_idx")["frag_mz"]
                     .apply(lambda s: np.sort(s.values.astype(float))).to_dict())

    def count_fragment_mz_over_window_subset(window_idxs: set, fr_mzs: List[Tuple[float,str,int,int]], ppm: float) -> int:
        if not window_idxs or not fr_mzs or fr_by_win is None:
            return 0
        import bisect
        matched = set()
        # Concatenate per-window arrays only logically; search each separately
        for (mz, ion, k, z) in fr_mzs:
            delta = mz * ppm * 1e-6
            lo, hi = mz - delta, mz + delta
            hit = False
            for w in window_idxs:
                arr = fr_by_win.get(int(w))
                if arr is None or arr.size == 0: 
                    continue
                L = bisect.bisect_left(arr, lo)
                R = bisect.bisect_right(arr, hi)
                if R > L:
                    hit = True; break
            if hit:
                matched.add((ion,k,z))
        return len(matched)

    # Per-protein precursor burden (for correlation)
    per_protein_rows = []

    # Compute metrics per peptide
    scores_rows = []
    for i, pep in enumerate(tqdm(pep_list, desc=f"[sets-chained] scoring peptides (set={set_name}, mode={mode})")):
        rt_pep = pep_rt_map[pep]
        pmass = pep_mass[i]

        # Precursor-matched window idxs
        matched_win_idxs = set(pairs.get(pep, set()))
        N_precursor_mz = float(len(matched_win_idxs))

        # For protein correlation
        for w in matched_win_idxs:
            per_protein_rows.append({
                "set_name": set_name, "mode": mode,
                "query_peptide": pep,
                "protein_short": win_df.at[w, "protein_short"],
                "accession": win_df.at[w, "accession"],
                "n_windows": 1
            })

        # RT-gated subset
        rt_ok_win_idxs = {w for w in matched_win_idxs if abs(win_df.at[w,"rt_min"] - rt_pep) <= rt_tol}
        N_precursor_mz_rt = float(len(rt_ok_win_idxs))

        # Peptide fragments (m/z list + seq keys)
        fr_mzs, fr_keys = peptide_fragments(pep)
        n_fragment_types = float(len(fr_mzs))  # denominator for Frac_fragment_* (types are (ion,k,z))

        # (Ungated) fragment m/z vs ANY window
        matched_any = 0
        if fr_mzs:
            seen = set()
            for (mz, ion, k, z) in fr_mzs:
                if any_frag_mz_match_global(mz, frag_ppm):
                    seen.add((ion,k,z))
            matched_any = len(seen)
        N_fragment_mz_any = float(matched_any)
        Frac_fragment_mz_any = N_fragment_mz_any / max(1.0, n_fragment_types)

        # RT-gated fragment m/z vs ANY window (filter frag_df by RT proximity)
        matched_rt = 0
        if fr_mzs and not frag_df.empty:
            # Build a boolean mask of fragment rows that are within RT tolerance to peptide RT
            close = frag_df.index[np.abs(frag_df["rt_min"].values - rt_pep) <= rt_tol]
            if close.size > 0:
                arr = np.sort(frag_df.loc[close, "frag_mz"].values.astype(float))
                import bisect
                seen = set()
                for (mz, ion, k, z) in fr_mzs:
                    delta = mz * frag_ppm * 1e-6
                    lo, hi = mz - delta, mz + delta
                    L = bisect.bisect_left(arr, lo); R = bisect.bisect_right(arr, hi)
                    if R > L: seen.add((ion,k,z))
                matched_rt = len(seen)
        N_fragment_mz_rt = float(matched_rt)
        Frac_fragment_mz_rt = N_fragment_mz_rt / max(1.0, n_fragment_types)

        # NEW: Chained fragment m/z (restricted to precursor-matched windows)
        N_fragment_mz_given_precursor = float(count_fragment_mz_over_window_subset(matched_win_idxs, fr_mzs, frag_ppm))
        Frac_fragment_mz_given_precursor = N_fragment_mz_given_precursor / max(1.0, n_fragment_types)

        # NEW: Chained fragment m/z with RT (restricted to precursor+RT windows)
        N_fragment_mz_given_precursor_rt = float(count_fragment_mz_over_window_subset(rt_ok_win_idxs, fr_mzs, frag_ppm))
        Frac_fragment_mz_given_precursor_rt = N_fragment_mz_given_precursor_rt / max(1.0, n_fragment_types)

        # Fragment sequence containment given precursor / precursor+RT (I/L collapse optional)
        frag_seq_set = set(fr_keys)  # tuples ("b"/"y", k, seq_c)
        # given precursor
        matched_seq_given_prec = set()
        if matched_win_idxs and frag_seq_set:
            for ion, k, seq_c in frag_seq_set:
                for w in matched_win_idxs:
                    if seq_c in win_seq_list_xle[w]:
                        matched_seq_given_prec.add((ion,k,seq_c)); break
        N_fragment_sequence_given_precursor = float(len(matched_seq_given_prec))
        Frac_fragment_sequence_given_precursor = N_fragment_sequence_given_precursor / max(1.0, float(len(frag_seq_set)))

        # given precursor + RT
        matched_seq_given_prec_rt = set()
        if rt_ok_win_idxs and frag_seq_set:
            for ion, k, seq_c in frag_seq_set:
                for w in rt_ok_win_idxs:
                    if seq_c in win_seq_list_xle[w]:
                        matched_seq_given_prec_rt.add((ion,k,seq_c)); break
        N_fragment_sequence_given_precursor_rt = float(len(matched_seq_given_prec_rt))
        Frac_fragment_sequence_given_precursor_rt = N_fragment_sequence_given_precursor_rt / max(1.0, float(len(frag_seq_set)))

        scores_rows.append({
            "set_name": set_name, "mode": mode, "query_peptide": pep,
            # precursor counts
            "N_precursor_mz": N_precursor_mz,
            "N_precursor_mz_rt": N_precursor_mz_rt,
            # fragment m/z (global)
            "N_fragment_mz_any": N_fragment_mz_any,
            "Frac_fragment_mz_any": Frac_fragment_mz_any,
            "N_fragment_mz_rt": N_fragment_mz_rt,
            "Frac_fragment_mz_rt": Frac_fragment_mz_rt,
            # fragment m/z (chained to precursor windows)
            "N_fragment_mz_given_precursor": N_fragment_mz_given_precursor,
            "Frac_fragment_mz_given_precursor": Frac_fragment_mz_given_precursor,
            "N_fragment_mz_given_precursor_rt": N_fragment_mz_given_precursor_rt,
            "Frac_fragment_mz_given_precursor_rt": Frac_fragment_mz_given_precursor_rt,
            # fragment sequence containment (chained)
            "N_fragment_sequence_given_precursor": N_fragment_sequence_given_precursor,
            "Frac_fragment_sequence_given_precursor": Frac_fragment_sequence_given_precursor,
            "N_fragment_sequence_given_precursor_rt": N_fragment_sequence_given_precursor_rt,
            "Frac_fragment_sequence_given_precursor_rt": Frac_fragment_sequence_given_precursor_rt,
            # bookkeeping
            "n_fragment_types_considered": n_fragment_types,
            "rt_min_peptide": float(rt_pep),
        })

    per_protein_df = pd.DataFrame(per_protein_rows)
    if not per_protein_df.empty:
        per_protein_df = (per_protein_df.groupby(["set_name","mode","query_peptide","protein_short","accession"])
                          ["n_windows"].sum().reset_index())

    return scores_rows, per_protein_df

# ---- Confusability and effects ----
def add_confusability(scores_df: pd.DataFrame) -> pd.DataFrame:
    out = scores_df.copy()

    # helper to get 95th-percentile cap per group
    def cap(series):
        if series.empty: return 1.0
        p95 = float(series.quantile(0.95))
        return p95 if p95 > 0 else 1.0

    rows = []
    for (sname, mode), sub in out.groupby(["set_name","mode"], dropna=False):
        c1 = cap(sub["N_precursor_mz"])
        c2 = cap(sub["N_precursor_mz_rt"])
        norm1 = (sub["N_precursor_mz"] / c1).clip(0,1)
        norm2 = (sub["N_precursor_mz_rt"] / c2).clip(0,1)
        # Simple mean of 4 evidences (two normalized counts + two fractions)
        conf_simple = (norm1 + norm2 + sub["Frac_fragment_mz_given_precursor"].clip(0,1)
                       + sub["Frac_fragment_mz_given_precursor_rt"].clip(0,1)) / 4.0
        tmp = sub.copy()
        tmp["Norm_N_precursor_mz"] = norm1.astype(float)
        tmp["Norm_N_precursor_mz_rt"] = norm2.astype(float)
        tmp["Confusability_simple"] = conf_simple.astype(float)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else out

def compute_effects(scores_df: pd.DataFrame, peptides_df: pd.DataFrame,
                    effect_threshold: float, prevalence_threshold: float, min_group_n: int) -> pd.DataFrame:
    if "is_hit" not in peptides_df.columns:
        return pd.DataFrame()
    tmp = scores_df.merge(
        peptides_df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    metrics = [
        # precursor
        "N_precursor_mz","N_precursor_mz_rt",
        # fragment m/z (global)
        "N_fragment_mz_any","Frac_fragment_mz_any","N_fragment_mz_rt","Frac_fragment_mz_rt",
        # fragment m/z (chained)
        "N_fragment_mz_given_precursor","Frac_fragment_mz_given_precursor",
        "N_fragment_mz_given_precursor_rt","Frac_fragment_mz_given_precursor_rt",
        # fragment sequence (chained)
        "N_fragment_sequence_given_precursor","Frac_fragment_sequence_given_precursor",
        "N_fragment_sequence_given_precursor_rt","Frac_fragment_sequence_given_precursor_rt",
        # confusability
        "Confusability_simple"
    ]
    out_rows = []
    for (sname, mode), sub in tmp.groupby(["set_name","mode"], dropna=False):
        for m in metrics:
            x_hit = sub.loc[sub["is_hit"]==1, m]
            x_non = sub.loc[sub["is_hit"]==0, m]
            n_hit, n_non = len(x_hit), len(x_non)
            mean_hit = float(pd.to_numeric(x_hit, errors="coerce").mean()) if n_hit else np.nan
            mean_non = float(pd.to_numeric(x_non, errors="coerce").mean()) if n_non else np.nan
            prop_nz_hit = float((pd.to_numeric(x_hit, errors="coerce") > 0).mean()) if n_hit else np.nan
            prop_nz_non = float((pd.to_numeric(x_non, errors="coerce") > 0).mean()) if n_non else np.nan
            auc, delta = mannwhitney_auc_cliffs(x_hit, x_non) if (n_hit and n_non) else (np.nan, np.nan)
            prevalence = max(prop_nz_hit if not np.isnan(prop_nz_hit) else 0.0,
                             prop_nz_non if not np.isnan(prop_nz_non) else 0.0)
            enriched = (n_hit>=min_group_n and n_non>=min_group_n and
                        (not np.isnan(delta)) and delta>=effect_threshold and prevalence>=prevalence_threshold)
            out_rows.append(dict(set_name=sname, mode=mode, metric=m,
                                 n_hit=n_hit, n_non=n_non,
                                 mean_hit=mean_hit, mean_non=mean_non,
                                 delta_mean=(mean_hit-mean_non) if (not np.isnan(mean_hit) and not np.isnan(mean_non)) else np.nan,
                                 prop_nonzero_hit=prop_nz_hit, prop_nonzero_non=prop_nz_non,
                                 auc=auc, cliffs_delta=delta, size_class=classify_delta(delta),
                                 prevalence=prevalence, enriched=bool(enriched)))
    return pd.DataFrame(out_rows)

# ---- Plotting ----
def plot_delta_bars(effects_df: pd.DataFrame, metric: str, mode: str, thr: float, outdir: str):
    sub = effects_df[(effects_df["metric"]==metric) & (effects_df["mode"]==mode)].copy()
    if sub.empty: return
    sub.sort_values("cliffs_delta", ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(max(8,0.6*len(sub)+3), 5.2))
    x = np.arange(len(sub))
    ax.bar(x, sub["cliffs_delta"].values)
    ax.axhline(thr, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(sub["set_name"].values, rotation=45, ha="right")
    ax.set_ylabel("Cliff's δ (hits > non-hits)")
    ax.set_title(f"Effect sizes — {metric} — mode={mode}")
    for i, r in enumerate(sub.itertuples(index=False)):
        if bool(r.enriched):
            ax.text(i, r.cliffs_delta + 0.02, "•", ha="center", va="bottom")
    add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"bar_delta__{safe_slug(metric)}__mode_{safe_slug(mode)}.png"))
    plt.close(fig)

def plot_flagged_boxplots(scores_df: pd.DataFrame, peptides_df: pd.DataFrame,
                          effects_df: pd.DataFrame, max_flag_plots: int, outdir: str):
    if "is_hit" not in peptides_df.columns: return
    tmp = scores_df.merge(
        peptides_df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    flagged = effects_df.loc[effects_df["enriched"]==True].copy()
    if flagged.empty: return
    flagged["abs_delta"] = flagged["cliffs_delta"].abs()
    flagged.sort_values(["metric","mode","abs_delta"], ascending=[True, True, False], inplace=True)
    ctr = 0
    for _, r in flagged.iterrows():
        if ctr >= max_flag_plots: break
        sname, mode, metric = r["set_name"], r["mode"], r["metric"]
        sub = tmp[(tmp["set_name"]==sname) & (tmp["mode"]==mode)]
        if sub.empty or metric not in sub.columns: continue
        non = pd.to_numeric(sub.loc[sub["is_hit"]==0, metric], errors="coerce").fillna(0.0).values
        hit = pd.to_numeric(sub.loc[sub["is_hit"]==1, metric], errors="coerce").fillna(0.0).values
        fig, ax = plt.subplots()
        ax.boxplot([non, hit], labels=[f"non-hit (n={len(non)})", f"hit (n={len(hit)})"], showmeans=True, meanline=True)
        ax.set_ylabel(f"{metric} per peptide")
        ax.set_title(f"{metric} — {sname} — mode={mode} (δ={r['cliffs_delta']:+.3f}, Δmean={r['delta_mean']:+.2f})")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"box_{safe_slug(metric)}__{safe_slug(sname)}__mode_{safe_slug(mode)}__FLAG.png"))
        plt.close(fig)
        ctr += 1

def plot_score_corr(scores_df: pd.DataFrame, set_name: str, mode: str, outdir: str):
    metrics = [
        "Norm_N_precursor_mz","Norm_N_precursor_mz_rt",
        "Frac_fragment_mz_any","Frac_fragment_mz_rt",
        "Frac_fragment_mz_given_precursor","Frac_fragment_mz_given_precursor_rt",
        "Confusability_simple"
    ]
    sub = scores_df[(scores_df["set_name"]==set_name) & (scores_df["mode"]==mode)][metrics]
    if sub.empty: return
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.4*len(metrics), 1.0*len(metrics)+1.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
    ax.set_title(f"Score correlations — {set_name} — mode={mode}")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heatmap_score_corr__{safe_slug(set_name)}__mode_{safe_slug(mode)}.png"))
    plt.close(fig)

def plot_protein_corr(per_protein_df: pd.DataFrame, set_name: str, mode: str, topN: int, outdir: str):
    sub = per_protein_df[(per_protein_df["set_name"]==set_name) & (per_protein_df["mode"]==mode)]
    if sub.empty: return
    totals = (sub.groupby(["protein_short","accession"])["n_windows"].sum()
                .sort_values(ascending=False).head(topN))
    keep = totals.index.tolist()
    sub = sub.set_index(["protein_short","accession"]).loc[keep].reset_index()
    mat = sub.pivot_table(index="query_peptide", columns=["protein_short","accession"], values="n_windows", fill_value=0)
    if mat.shape[1] < 2: return
    corr = mat.corr()
    fig, ax = plt.subplots(figsize=(max(8, 0.45*corr.shape[1]+3), max(6, 0.45*corr.shape[0]+2)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(corr.shape[1])); ax.set_yticks(range(corr.shape[0]))
    col_labels = [f"{a[0]}({a[1]})" for a in corr.columns]
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticklabels(col_labels)
    ax.set_title(f"Protein vs protein correlation (precursor matches) — {set_name} — mode={mode}")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heatmap_protein_corr__{safe_slug(set_name)}__mode_{safe_slug(mode)}.png"))
    plt.close(fig)

# ---- Orchestrator ----
def main():
    ap = argparse.ArgumentParser(description="Protein-SET m/z screen with chained fragment evidence and simple confusability.")
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with peptide[, is_hit]")
    ap.add_argument("--outdir", default="mzscreen_sets_chained_out", help="Output directory")

    ap.add_argument("--kmin_subseq", type=int, default=5, help="Minimum protein subsequence length (default 5)")
    ap.add_argument("--kmax_subseq", type=int, default=15, help="Maximum protein subsequence length (default 15)")
    ap.add_argument("--charges", type=str, default="1,2,3", help="Precursor charges to test (default '1,2,3')")
    ap.add_argument("--prec_ppm", type=float, default=30.0, help="Precursor m/z tolerance in ppm (default 30)")

    ap.add_argument("--frag_kmin", type=int, default=2, help="Minimum b/y fragment length (default 2)")
    ap.add_argument("--frag_kmax", type=int, default=7, help="Maximum b/y fragment length (default 7)")
    ap.add_argument("--frag_charges", type=str, default="1,2,3", help="Fragment charges to test (default '1,2,3')")
    ap.add_argument("--frag_ppm", type=float, default=30.0, help="Fragment m/z tolerance in ppm (default 30)")

    ap.add_argument("--rt_total_min", type=float, default=20.0, help="Simulated LC gradient length in minutes (default 20)")
    ap.add_argument("--rt_tol", type=float, default=1.0, help="RT co-elution tolerance in minutes (default 1.0)")
    ap.add_argument("--xle_collapse", action="store_true", help="Treat I/L as indistinguishable for sequence containment (default off)")

    ap.add_argument("--prot_fasta", type=str, default=None, help="User-provided FASTA to source set/base proteins (optional)")
    ap.add_argument("--download_common", action="store_true", help="Allow UniProt downloads for missing accessions")
    ap.add_argument("--accessions", type=str, default=None, help="Text file of accessions to define/extend the base panel (optional)")

    ap.add_argument("--sets", type=str, default="albumin,antibody,keratin,serum,cytoskeleton,chaperone", help="Comma-separated set names")
    ap.add_argument("--set_mode", type=str, choices=["alone","union","both"], default="both", help="Test sets alone, union-with-base, or both")
    ap.add_argument("--sets_config", type=str, default=None, help="JSON: {\"set_name\": {\"accessions\": [\"P12345\", ...]}, ...}")

    ap.add_argument("--effect_threshold", type=float, default=0.147, help="Cliff's δ threshold to flag enrichment (default 0.147)")
    ap.add_argument("--prevalence_threshold", type=float, default=0.10, help="Min fraction with ≥1 match in either group (default 0.10)")
    ap.add_argument("--min_group_n", type=int, default=10, help="Min peptides per group for effects (default 10)")

    ap.add_argument("--prot_corr_top", type=int, default=30, help="Top-N proteins to show in protein-correlation heatmap (default 30)")
    ap.add_argument("--max_flag_plots", type=int, default=36, help="Max # of flagged boxplots to emit (default 36)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)

    # Read peptides
    df = pd.read_csv(args.input_csv)
    if "peptide" not in df.columns:
        raise ValueError("Input CSV must include a 'peptide' column.")
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df = df[df["peptide"].str.len()>0].copy()
    df["mass"] = df["peptide"].apply(calc_mass)
    has_hit = "is_hit" in df.columns
    if has_hit:
        df["is_hit"] = df["is_hit"].astype(int)

    # Load base panel
    base_seqs = load_common_proteins(args.prot_fasta, args.download_common, args.accessions)
    all_fasta_seqs = {}
    if args.prot_fasta and os.path.isfile(args.prot_fasta):
        with open_maybe_gzip(args.prot_fasta) as fh:
            all_fasta_seqs = parse_fasta(fh.read())

    # Build set definitions (allow overrides)
    set_defs = dict(DEFAULT_SETS)
    if args.sets_config:
        user_cfg = json.load(open(args.sets_config, "r"))
        for sname, spec in user_cfg.items():
            if isinstance(spec, dict):
                accs = spec.get("accessions", [])
            elif isinstance(spec, list):
                accs = spec
            else:
                raise ValueError(f"Invalid sets_config entry for '{sname}'.")
            set_defs[sname] = list(dict.fromkeys(accs))

    sets_to_run = [s.strip() for s in args.sets.split(",") if s.strip()]
    modes = ["alone","union"] if args.set_mode == "both" else [args.set_mode]

    # Resolve sequences per set (download if needed)
    set_sequences = {}
    for s in sets_to_run:
        accs = set_defs.get(s, [])
        seqs = {}
        if all_fasta_seqs and accs:
            seqs = merge_seqs(seqs, select_seqs_by_accessions(all_fasta_seqs, accs))
        if accs:
            seqs = merge_seqs(seqs, select_seqs_by_accessions(base_seqs, accs))
        if args.download_common and accs:
            have = set(extract_accession(h) for h in seqs.keys())
            need = [a for a in accs if a not in have]
            if need:
                print(f"[sets-chained] downloading {len(need)} proteins for set '{s}' ...")
                seqs = merge_seqs(seqs, download_uniprot_by_accessions(need))
        set_sequences[s] = seqs

    charges = [int(x) for x in args.charges.split(",") if x.strip()]
    frag_charges = [int(x) for x in args.frag_charges.split(",") if x.strip()]

    # Run each set × mode
    scores_all = []
    perprot_all = []
    for s in sets_to_run:
        for mode in modes:
            rows, perprot = compute_per_peptide_metrics_for_set(
                set_name=s, mode=mode,
                df_peptides=df[["peptide","mass"]],
                base_seqs=base_seqs, set_seqs=set_sequences.get(s, {}),
                kmin_subseq=args.kmin_subseq, kmax_subseq=args.kmax_subseq,
                charges=charges, prec_ppm=args.prec_ppm,
                frag_kmin=args.frag_kmin, frag_kmax=args.frag_kmax, frag_charges=frag_charges, frag_ppm=args.frag_ppm,
                rt_total_min=args.rt_total_min, rt_tol=args.rt_tol,
                xle_collapse=args.xle_collapse,
                prot_corr_top=args.prot_corr_top
            )
            if rows:
                scores_all.extend(rows)
            if not perprot.empty:
                perprot_all.append(perprot)

    if not scores_all:
        raise RuntimeError("No results computed. Check sets, FASTA/download options, or peptide list.")

    scores_df = pd.DataFrame(scores_all)
    scores_df = add_confusability(scores_df)
    scores_path = os.path.join(args.outdir, "per_peptide_scores__ALL.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"[sets-chained] Wrote: {scores_path}")

    if perprot_all:
        perprot_df = pd.concat(perprot_all, ignore_index=True)
        perprot_path = os.path.join(args.outdir, "per_peptide_per_protein_counts__ALL.csv")
        perprot_df.to_csv(perprot_path, index=False)
        print(f"[sets-chained] Wrote: {perprot_path}")
    else:
        perprot_df = pd.DataFrame()

    # Effects and plots
    effects_df = compute_effects(scores_df, df, args.effect_threshold, args.prevalence_threshold, args.min_group_n) if has_hit else pd.DataFrame()
    eff_path = os.path.join(args.outdir, "effects_summary__ALL.csv")
    if not effects_df.empty:
        effects_df.sort_values(["metric","mode","cliffs_delta"], ascending=[True, True, False], inplace=True)
        effects_df.to_csv(eff_path, index=False)
        print(f"[sets-chained] Wrote: {eff_path}")

        # δ bars per metric × mode
        metrics_unique = effects_df["metric"].drop_duplicates().tolist()
        for m in metrics_unique:
            for mode in modes:
                plot_delta_bars(effects_df, m, mode, args.effect_threshold, plots_dir)

        # Flagged boxplots
        plot_flagged_boxplots(scores_df, df, effects_df, args.max_flag_plots, plots_dir)

    # Correlations
    for s in sets_to_run:
        for mode in modes:
            plot_score_corr(scores_df, s, mode, plots_dir)
            if not perprot_df.empty:
                plot_protein_corr(perprot_df, s, mode, args.prot_corr_top, plots_dir)

    # SUMMARY
    summary_path = os.path.join(args.outdir, "SUMMARY_STATS.md")
    lines = []
    lines += [
        "# WIQD protein‑set screen — chained fragment evidence",
        "",
        f"*Generated:* {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"*Input:* `{os.path.basename(args.input_csv)}`",
        f"*Sets:* {', '.join(sets_to_run)}",
        f"*Modes:* {', '.join(modes)}",
        "",
        "## Metrics (per peptide)",
        f"- **N_precursor_mz** — # protein subseq windows (k={args.kmin_subseq}..{args.kmax_subseq}) sharing precursor m/z (z={args.charges}; ±{args.prec_ppm} ppm).",
        f"- **N_precursor_mz_rt** — same, but require |ΔRT| ≤ {args.rt_tol} min on a {args.rt_total_min}‑min gradient.",
        f"- **N/Frac_fragment_mz_any** — # and fraction of peptide b/y fragment **m/z** (k={args.frag_kmin}..{args.frag_kmax}, z={args.frag_charges}) matching ANY window fragments (±{args.frag_ppm} ppm).",
        f"- **N/Frac_fragment_mz_rt** — as above, but only windows with |ΔRT| ≤ {args.rt_tol} min.",
        "- **N/Frac_fragment_mz_given_precursor** — fragment m/z matching **restricted to the windows that passed the precursor screen**.",
        "- **N/Frac_fragment_mz_given_precursor_rt** — restricted to windows that passed **precursor + RT**.",
        "- **N/Frac_fragment_sequence_given_precursor[_rt]** — b/y **string containment** among precursor‑(±RT) windows (I/L collapse optional).",
        "",
        "## Confusability (simple)",
        "- **Confusability_simple** = mean( Norm_N_precursor_mz, Norm_N_precursor_mz_rt, Frac_fragment_mz_given_precursor, Frac_fragment_mz_given_precursor_rt ).",
        "- `Norm_*` are per set×mode counts capped at their 95th percentile and scaled to [0,1].",
        "",
        "## Notes",
        "- All outputs are **per‑peptide** (counts) or **fractions**; group comparisons use Cliff’s δ and prevalence.",
        "- Fragment m/z does not require same length or same end; sequence containment checks use b_k (prefix) and y_k (suffix).",
        "- RT model is a simple KD‑based surrogate; intended only for co‑elution filtering.",
        "",
        "See `plots/` for δ bars (per metric × mode), flagged boxplots, and correlation heatmaps.",
    ]
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[sets-chained] Wrote: {summary_path}")

if __name__ == "__main__":
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()

