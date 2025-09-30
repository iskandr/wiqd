#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoygen.py — Cys+Pro+Met decoy generator with:
  • ranked Pro/M placement (Pro tiers, avoid R/K|P, spacing),
  • albumin RT→precursor→fragment m/z “chaining” confusability,
  • NEW: XLE-collapsed terminal k-mer burden vs enlarged background protein set,
  • transparent, weighted FINAL SCORE = w_conf * conf + w_term * term + w_fly * fly,
  • verbose selection explanations + comprehensive plots.

USAGE (typical):
  python decoygen.py \
    --input neo_table.tsv \
    --peptide-col epitope --pos-col MT_pos --mut-notation-col Gene_AA_Change \
    --indexing C1 \
    --ppm 30 --charges 1,2,3 \
    --alb-kmin 5 --alb-kmax 15 \
    --frag-kmin 2 --frag-kmax 11 \
    --rt-total-min 20 --rt-tol 1 \
    --bg-source embedded --term-kmin 3 --term-kmax 7 --collapse xle \
    --rank-weights conf:0.6,term:0.25,fly:0.15 \
    --dedup-k 4 --avoid-RK-before-P \
    --N 10 \
    --outdir run_out --output-prefix wiqd_decoys

PPM rule (scale-invariant):
  match if |m/z_query – m/z_ref| ≤ ppm_tol * m/z_ref / 1e6

Dependencies: numpy, pandas, matplotlib (headless), tqdm (optional)
"""

from __future__ import annotations
import argparse, re, math, os, bisect, textwrap
from typing import Dict, List, Tuple, Optional, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tqdm optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): return it

# =================== Chemistry & Mass ===================
AA_MONO = {
    "A": 71.037113805, "R": 156.10111105, "N": 114.04292747, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.05857754, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.09496305,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.07931298, "Y": 163.063328575, "V": 99.068413945,
}
WATER = 18.010564684
PROTON = 1.007276466812
HYDRO_CTERM_SET = set("AILMFWYV")   # last residue must be hydrophobic

def calc_mass(seq: str, cys_fixed_mod: float = 0.0) -> float:
    m = 0.0
    for a in seq:
        m += AA_MONO[a]
        if a == "C" and cys_fixed_mod:
            m += cys_fixed_mod
    return m + WATER

def mz_from_mass(M: float, z: int) -> float:
    return (M + z*PROTON) / z

# =================== Flyability & Hydrophobicity ===================
KD_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

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
    R,K,H = seq.count("R"), seq.count("K"), seq.count("H")
    D,E   = seq.count("D"), seq.count("E")
    charge_sites = 1.0 + R + K + 0.7*H
    acid_penalty = 0.1*(D+E)
    charge = 1.0 - math.exp(-max(0.0, charge_sites - acid_penalty)/2.0)
    gravy = kd_gravy(seq)
    surface = float(np.clip((gravy + 4.5)/9.0, 0.0, 1.0))
    W,Y,F = seq.count("W"), seq.count("Y"), seq.count("F")
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

def fly_surface_norm_only(seq: str) -> float:
    return fly_components(seq)["fly_surface_norm"]

def hydrophobic_fraction(seq: str, hyd_set: set) -> float:
    if not seq: return 0.0
    return sum(1 for a in seq if a in hyd_set) / len(seq)

def pick_hydro_metric_fn(mode: str, hyd_set: set):
    if mode == "fraction":  # default
        return lambda s: hydrophobic_fraction(s, hyd_set)
    return kd_gravy

def count_penalized(seq: str, pen_set: set) -> int:
    return sum(1 for a in seq if a in pen_set)

# =================== RT predictor ===================
def predict_rt_min(seq: str, gradient_min: float = 20.0) -> float:
    if not seq: return 0.0
    gravy = kd_gravy(seq)
    frac = (gravy + 4.5) / 9.0
    base = 0.5 + max(0.0, gradient_min - 1.0) * min(1.0, max(0.0, frac))
    length_adj = 0.03 * max(0, len(seq) - 8) * (gradient_min / 20.0)
    basic_adj = -0.15 * (seq.count("K") + seq.count("R") + 0.3*seq.count("H")) * (gradient_min / 20.0)
    return float(np.clip(base + length_adj + basic_adj, 0.0, gradient_min))

# =================== Albumin sequence (embedded prepro 609 aa) ===================
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
    # Precompute albumin window fragments — ANY contiguous fragment m/z per window (2..11) * charges
    frag_cache = {}
    for i in range(len(df)):
        frs = fragment_mz_list(df.at[i,"window"], 2, 11, charges)
        lst = [mz for (mz,ion,k,z) in frs]; lst.sort()
        frag_cache[i] = lst
    return df, frag_cache

def ppm(a: float, b: float) -> float:
    return (abs(a-b)/b*1e6) if b else float("inf")

def any_within_ppm(q: float, arr: List[float], tol_ppm: float) -> bool:
    if not arr: return False
    tol = q * tol_ppm * 1e-6
    i = bisect.bisect_left(arr, q)
    if i < len(arr) and abs(arr[i] - q) <= tol: return True
    if i > 0 and abs(arr[i-1] - q) <= tol: return True
    if i+1 < len(arr) and abs(arr[i+1] - q) <= tol: return True
    return False

# =================== Albumin confusability (estimate for ranking) ===================
def estimate_albumin_conf(seq: str, alb_df: pd.DataFrame, alb_frag_cache: Dict[int, List[float]],
                          charges: List[int], ppm_tol: float, rt_total_min: float, cys_fixed_mod: float):
    pep_mass = calc_mass(seq, cys_fixed_mod)
    pep_mz = {z: mz_from_mass(pep_mass, z) for z in charges}
    rt_pep = predict_rt_min(seq, rt_total_min)

    matched = set(); matched_rt = set()
    for i, wr in alb_df.iterrows():
        wmz_by_z = {z: mz_from_mass(wr["mass"], z) for z in charges}
        if any(ppm(pep_mz[z], wmz_by_z[z]) <= ppm_tol for z in charges):
            matched.add(int(i))
            if abs(wr["rt_min"] - rt_pep) <= 1.0:  # estimate uses ±1.0 min
                matched_rt.add(int(i))

    frs_pep = fragment_mz_list(seq, 2, 11, charges)
    n_types = float(len(frs_pep)) if frs_pep else 1.0

    def gather(idxset: set) -> List[float]:
        out = []
        for j in idxset:
            arr = alb_frag_cache.get(int(j))
            if arr: out.extend(arr)
        out.sort(); return out

    prec_fr = gather(matched); prec_rt_fr = gather(matched_rt)
    seen_prec = set(); seen_prec_rt = set()
    for (mz,ion,k,z) in frs_pep:
        if prec_fr and any_within_ppm(mz, prec_fr, ppm_tol):    seen_prec.add((ion,k,z))
        if prec_rt_fr and any_within_ppm(mz, prec_rt_fr, ppm_tol): seen_prec_rt.add((ion,k,z))

    N_prec = float(len(matched))
    N_prec_rt = float(len(matched_rt))
    Frac_prec = float(len(seen_prec))/n_types
    Frac_prec_rt = float(len(seen_prec_rt))/n_types
    conf_est = (N_prec + N_prec_rt + Frac_prec + Frac_prec_rt)/4.0
    return dict(N_prec=N_prec, N_prec_rt=N_prec_rt, Frac_prec=Frac_prec, Frac_prec_rt=Frac_prec_rt,
                conf_est=conf_est, N_frag_prec_rt=float(len(seen_prec_rt)))

# =================== Background terminals (XLE-collapsed k-mers) ===================
DEFAULT_BG_ACCS = [
    # albumin + frequent backgrounds in serum/plasma/lab environments
    "P02768",             # Albumin
    "P01857","P01859",    # IgG heavy/kappa
    "P04264","P02533",    # Keratin 1/14
    "P60709",             # Beta-actin
    "P07437",             # Trypsin (common carryover)
    "P69905","P68871",    # Hemoglobin alpha/beta
    "P02787",             # Transferrin
    "P11021","P07237",    # HSPA5/PDIA1
]

EMBEDDED_BG = {
    # Minimal embedded fallbacks (fragments are fine for k-mer banking)
    "sp|P02768|ALBU_HUMAN": ALBUMIN_P02768_PREPRO_609,
    "sp|P60709|ACTB_HUMAN": "MEEEIAALVIDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLK",
    "sp|P04406|G3P_HUMAN":  "MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGK",
    "sp|P07237|PDIA1_HUMAN":"MDSKGSSQKGKGGQKKDGKDEDKDLPGKKPKVLVLVGIWGAALLLQGAKELKDEL",
    "sp|P01857|IGHG1_HUMAN":"QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAMS",
    "sp|P04264|K1C1_HUMAN":  "MSGRGGMGGGKMSSG",
    "sp|P68871|HBB_HUMAN":   "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQ",
}

def parse_fasta_text(text: str) -> Dict[str,str]:
    seqs = {}
    hdr=None; buf=[]
    for ln in text.splitlines():
        if not ln: continue
        if ln.startswith(">"):
            if hdr:
                seqs[hdr] = "".join(buf).replace(" ", "").upper()
            hdr = ln[1:].strip()
            buf=[]
        else:
            buf.append(ln.strip())
    if hdr:
        seqs[hdr] = "".join(buf).replace(" ", "").upper()
    return seqs

def fetch_uniprot_fasta(acc: str, timeout: float = 15.0) -> str:
    import urllib.request
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        seqs = parse_fasta_text(data)
        if not seqs: return ""
        return list(seqs.values())[0]
    except Exception:
        return ""

def load_background_sequences(source: str, fasta: Optional[str], accs_csv: str) -> Dict[str,str]:
    if source == "none": return {}
    if source in ("file","auto") and fasta and os.path.isfile(fasta):
        with open(fasta, "r") as fh:
            return parse_fasta_text(fh.read())
    seqs = {}
    if source in ("fetch","auto"):
        accs = [a.strip() for a in accs_csv.split(",") if a.strip()]
        for a in accs:
            s = fetch_uniprot_fasta(a)
            if s:
                seqs[f"uniprot|{a}"] = s
    if not seqs:
        seqs = dict(EMBEDDED_BG)
    return seqs

def collapse_xle(s: str) -> str:
    return s.replace("L","J").replace("I","J")

def build_term_kmer_banks(seqs: Dict[str,str], kmin: int, kmax: int, collapse: str = "xle") -> Dict[int,set]:
    banks: Dict[int,set] = {}
    for k in range(kmin, kmax+1):
        banks[k] = set()
    for name, seq in seqs.items():
        s = seq.upper()
        if collapse == "xle":
            s = collapse_xle(s)
        for k in range(kmin, kmax+1):
            if len(s) < k: continue
            for i in range(0, len(s)-k+1):
                banks[k].add(s[i:i+k])
    return banks

def term_kmer_features(peptide: str, banks: Dict[int,set], kmin: int, kmax: int, collapse: str = "xle") -> Dict[str,float]:
    L = len(peptide)
    if L < max(2, kmin):
        return dict(bg_bkmer_hits=0.0, bg_ykmer_hits=0.0, bg_term_kmer_hits=0.0, bg_term_kmer_frac=0.0)
    kmax_eff = min(kmax, L-1)
    seen_b = set(); seen_y = set()
    for k in range(kmin, kmax_eff+1):
        b = peptide[:k]; y = peptide[-k:]
        if collapse == "xle":
            b = collapse_xle(b); y = collapse_xle(y)
        if b in banks.get(k, set()): seen_b.add((k,b))
        if y in banks.get(k, set()): seen_y.add((k,y))
    total = 2*max(0, kmax_eff-kmin+1)
    hits = len(seen_b) + len(seen_y)
    frac = hits / max(1, total)
    return dict(bg_bkmer_hits=float(len(seen_b)), bg_ykmer_hits=float(len(seen_y)),
                bg_term_kmer_hits=float(hits), bg_term_kmer_frac=float(frac))

# =================== Proline tiers (X before P) ===================
P_TIER = {"G":0, "A":1, "L":2, "V":2, "I":2, "S":2, "T":2}  # lower is better
def tier_for_x_before_p(x_prev: str) -> int:
    return P_TIER.get(x_prev, 3)

# =================== Small helpers ===================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s))

def wraptext(s: str, width=90): return "\n".join(textwrap.wrap(s, width=width))

# =================== Pairwise scatter helpers (extended with 'term') ===================
def compute_basic_props(seq: str, charges: List[int], alb_df: pd.DataFrame,
                        alb_frag_cache: Dict[int, List[float]],
                        ppm_tol: float, rt_total_min: float,
                        hyd_fn, use_surface: bool, fly_w: Dict[str,float],
                        cys_fixed_mod: float,
                        term_banks: Dict[int,set], term_kmin: int, term_kmax: int, collapse: str):
    if not seq:
        return {"hyd_frac": np.nan, "gravy": np.nan, "fly_surface": np.nan, "fly_comp": np.nan,
                "conf_est": np.nan, "mass": np.nan, "rt": np.nan, "length": 0, "term": np.nan}
    hyd = hyd_fn(seq)
    grv = kd_gravy(seq)
    fly_surf = fly_surface_norm_only(seq)
    fly_comp = fly_score_norm(seq, fly_w)
    conf = estimate_albumin_conf(seq, alb_df, alb_frag_cache, charges, ppm_tol, rt_total_min, cys_fixed_mod)
    term = term_kmer_features(seq, term_banks, term_kmin, term_kmax, collapse)["bg_term_kmer_frac"]
    M = calc_mass(seq, cys_fixed_mod); rt = predict_rt_min(seq, rt_total_min)
    return {
        "hyd_frac": float(hyd), "gravy": float(grv),
        "fly_surface": float(fly_surf), "fly_comp": float(fly_comp),
        "conf_est": float(conf["conf_est"]), "term": float(term),
        "mass": float(M), "rt": float(rt), "length": int(len(seq)),
    }

def feature_label(tok: str) -> str:
    return {
        "hyd_frac": "Hydrophobic fraction",
        "gravy": "GRAVY (KD)",
        "fly_surface": "fly_surface_norm",
        "fly_comp": "fly_score (composite)",
        "conf_est": "Albumin confusability (estimate)",
        "term": "Background terminal k-mer fraction",
        "mass": "Monoisotopic mass (Da)",
        "rt": "Predicted RT (min)",
        "length": "Length (aa)",
    }.get(tok, tok)

def make_pair_scatter(scatter_df: pd.DataFrame, feats: List[str], outdir: Path, prefix: str, max_pairs: int = 30, sample_n: int = 0):
    pairs = []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            pairs.append((feats[i], feats[j]))
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    df = scatter_df.copy()
    if sample_n and sample_n > 0:
        dfs = []
        for cat, sub in df.groupby("category", dropna=False):
            dfs.append(sub.sample(sample_n, random_state=123) if len(sub) > sample_n else sub)
        df = pd.concat(dfs, ignore_index=True)
    markers = {"candidate": "o", "decoy_not_selected": "x", "decoy_selected": "^"}
    outdir.mkdir(parents=True, exist_ok=True)
    for (x, y) in pairs:
        if x not in df.columns or y not in df.columns: continue
        fig, ax = plt.subplots()
        for cat in ["candidate", "decoy_not_selected", "decoy_selected"]:
            sub = df[df["category"] == cat]
            if sub.empty: continue
            ax.scatter(sub[x], sub[y], s=16, marker=markers.get(cat, "o"), alpha=0.6, label=cat.replace("_"," "))
        ax.set_xlabel(feature_label(x)); ax.set_ylabel(feature_label(y))
        ax.set_title(f"{feature_label(x)} vs {feature_label(y)}")
        ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_pairs_scatter__{x}__vs__{y}.png", dpi=170)
        plt.close(fig)

# =================== Main ===================
def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Cys+Pro+Met decoy generator (+albumin chaining + background terminal k-mers)")
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

    # Hydrophobicity / flyability choices
    ap.add_argument("--hydro-metric", choices=["fraction","gravy"], default="fraction",
                    help="Hydrophobicity metric: fraction (%% hydrophobic residues) or KD‑GRAVY.")
    ap.add_argument("--hydro-set", default="AVILMFWYV",
                    help="Residues counted as hydrophobic for --hydro-metric=fraction (default AVILMFWYV).")
    ap.add_argument("--fly-score-mode", choices=["surface","composite"], default="surface",
                    help="Ranking fly signal: 'surface' uses fly_surface_norm; 'composite' uses the weighted mix.")
    ap.add_argument("--fly-weights", default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05")

    # Oxygen-rich side chain penalty (grouped)
    ap.add_argument("--penalize-set", default="QDE",
                    help="Residues to penalize jointly in ranking (default QDE).")
    ap.add_argument("--penalty-weight", type=float, default=1.0,
                    help="Relative weight for penalized residues in tie-breaking.")
    ap.add_argument("--penalty-prioritize-delta", action="store_true",
                    help="Rank Δpenalized before absolute post-count (default off).")

    # Background terminal k-mers (enlarged set)
    ap.add_argument("--bg-source", choices=["embedded","fetch","file","auto","none"], default="embedded",
                    help="Source for background proteins used to build terminal k-mer bank.")
    ap.add_argument("--bg-fasta", default=None, help="FASTA file if --bg-source file/auto")
    ap.add_argument("--bg-acc", default=",".join(DEFAULT_BG_ACCS), help="Comma-separated UniProt accessions if --bg-source fetch/auto")
    ap.add_argument("--term-kmin", type=int, default=3, help="Min terminal k-mer length for background")
    ap.add_argument("--term-kmax", type=int, default=7, help="Max terminal k-mer length for background")
    ap.add_argument("--collapse", choices=["none","xle"], default="xle", help="Alphabet collapse for term k-mers (default xle)")

    # Selection preferences / constraints
    ap.add_argument("--dedup-k", type=int, default=4)
    ap.add_argument("--avoid-RK-before-P", dest="avoid_rk_p", action="store_true", default=True)
    ap.add_argument("--allow-RK-before-P", dest="avoid_rk_p", action="store_false")

    # Selection count & scoring
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--rank-weights", default="conf:0.6,term:0.25,fly:0.15",
                    help="Weights for FINAL score; format: conf:x,term:y,fly:z (auto-normalized).")

    # Strict mutation check
    ap.add_argument("--strict-mutation-check", dest="strict_mutation_check", action="store_true", default=True)
    ap.add_argument("--no-strict-mutation-check", dest="strict_mutation_check", action="store_false")

    # Pairwise scatter options
    ap.add_argument("--no-scatter-pairs", dest="scatter_pairs", action="store_false", default=True)
    ap.add_argument("--scatter-features",
                    default="hyd_frac,fly_surface,conf_est,term,mass,rt,length",
                    help="Comma-separated features to pairwise plot among {hyd_frac,gravy,fly_surface,fly_comp,conf_est,term,mass,rt,length}.")
    ap.add_argument("--scatter-max-pairs", type=int, default=30)
    ap.add_argument("--scatter-sample", type=int, default=0)

    args = ap.parse_args(argv)

    outdir = Path(args.outdir); plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    # Selection explanation log
    explain_lines = []
    def logx(s): 
        print(s); explain_lines.append(s)

    # Albumin (embedded)
    albumin = ALBUMIN_P02768_PREPRO_609
    L_alb = len(albumin)
    logx(f"[INFO] Albumin embedded source: P02768 (prepro); length={L_alb} aa.")
    logx(f"[INFO] Using albumin sequence length={L_alb} aa for matching.")

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

    # Background proteins for terminal k-mers
    bg_seqs = load_background_sequences(args.bg_source, args.bg_fasta, args.bg_acc)
    if args.bg_source != "none":
        logx(f"[INFO] Background set: {len(bg_seqs)} protein(s) from {args.bg_source}; building terminal k-mer bank (k={args.term_kmin}..{args.term_kmax}, collapse={args.collapse}) …")
    term_banks = build_term_kmer_banks(bg_seqs, int(args.term_kmin), int(args.term_kmax), args.collapse) if bg_seqs else {}

    # Fly weights & hydrophobicity metric
    fly_w = {k.strip(): float(v.strip()) for k,v in (tok.split(":") for tok in args.fly_weights.split(","))}
    hyd_set = set(list(args.hydro_set.strip().upper()))
    pen_set = set(list(args.penalize_set.strip().upper()))
    hyd_fn = pick_hydro_metric_fn(args.hydro_metric, hyd_set)
    use_surface = (args.fly_score_mode == "surface")
    pen_w = float(args.penalty_weight)
    pen_delta_first = bool(args.penalty_prioritize_delta)

    # Build albumin index and frag cache
    alb_df, alb_frag_cache = build_albumin_index(ALBUMIN_P02768_PREPRO_609, int(args.alb_kmin), int(args.alb_kmax),
                                                 charges, cys_fixed_mod, float(args.rt_total_min))

    # ---- Stage A: base filters + mutation index sanity ----
    n_total = len(df)
    drops = dict(wtmutC=0, hasP=0, hasC=0, hasM=0, nonhydro_cterm=0, mut_last=0, mut_mismatch=0)
    stage_labels = []; survivors = []

    logx("[INFO] Stage A: base filtering …")
    for row in tqdm(df.to_dict("records"), total=len(df), desc="Base filters"):
        pep = row["_peptide"]
        wt = str(row.get("_wt")) if pd.notna(row.get("_wt")) else None
        mut = str(row.get("_mut")) if pd.notna(row.get("_mut")) else None
        pos = int(row["_pos_int"])

        if (wt == "C") or (mut == "C"):
            drops["wtmutC"] += 1; row["stage_label"] = "dropped_base_wt_or_mut_C"; stage_labels.append(row); continue
        if "P" in pep:
            drops["hasP"] += 1; row["stage_label"] = "dropped_base_has_proline"; stage_labels.append(row); continue
        if "C" in pep:
            drops["hasC"] += 1; row["stage_label"] = "dropped_base_has_cysteine"; stage_labels.append(row); continue
        if "M" in pep:
            drops["hasM"] += 1; row["stage_label"] = "dropped_base_has_methionine"; stage_labels.append(row); continue
        if not pep or pep[-1] not in HYDRO_CTERM_SET:
            drops["nonhydro_cterm"] += 1; row["stage_label"] = "dropped_base_nonhydrophobic_Cterm"; stage_labels.append(row); continue

        if args.indexing == "auto":
            mut_idx0, idx_mode = detect_indexing(pep, pos, wt or "", mut or "")
        else:
            mut_idx0 = resolve_mut_idx(pep, pos, args.indexing); idx_mode = args.indexing
        if mut_idx0 == len(pep)-1:
            drops["mut_last"] += 1; row["stage_label"] = "dropped_mutation_at_last"; stage_labels.append(row); continue

        letter = pep[mut_idx0]
        match = "none"
        if wt and letter == wt: match = "equals_wt"
        if mut and letter == mut: match = ("equals_mut" if match=="none" else "equals_both")
        if match == "none" and args.strict_mutation_check:
            drops["mut_mismatch"] += 1; row["stage_label"] = "dropped_mutation_letter_mismatch"; stage_labels.append(row); continue

        row["_mut_idx0"] = mut_idx0; row["_indexing_used"] = idx_mode; row["_mut_letter"] = letter; row["_mut_match"] = match
        survivors.append(row)

    kept = len(survivors)
    logx("[INFO] Base filters summary:")
    logx(f"  total={n_total}  kept={kept}  dropped={n_total-kept}")
    for k,v in drops.items(): logx(f"    • {k}: {v}")
    if not survivors: raise RuntimeError("No peptides left after base filters.")

    # ---- Stage B: select Proline (tiered) then Methionine ----
    logx("[INFO] Stage B: selecting Proline tiered (GP>AP>LVISTP, avoid R/K|P, ≥1 from C) then Methionine (≥1 from C; P between C and M) …")

    p_enum_records = []; m_enum_records = []; decoy_rows = []
    examples_to_explain = 10  # write verbose top choices for first 10 peptides

    for r_i, row in enumerate(tqdm(survivors, total=len(survivors), desc="Pro/M selection")):
        pep = row["_peptide"]; L = len(pep); mut_idx0 = row["_mut_idx0"]

        # Put C at mutation
        cys_seq = pep[:mut_idx0] + "C" + pep[mut_idx0+1:]
        base_hyd = hyd_fn(cys_seq)
        base_fly = (fly_surface_norm_only(cys_seq) if use_surface else fly_score_norm(cys_seq, fly_w))
        base_pen = count_penalized(cys_seq, pen_set)

        # ---------- PROLINE ----------
        p_positions = [i for i in range(2, L-1) if i != mut_idx0 and abs(i - mut_idx0) >= 2]
        p_cands = []
        for p_idx in p_positions:
            x_prev = cys_seq[p_idx-1]
            rk_before = (x_prev in ("R","K"))
            tier = tier_for_x_before_p(x_prev)
            replace_pen = (cys_seq[p_idx] in pen_set)
            sP = cys_seq[:p_idx] + "P" + cys_seq[p_idx+1:]
            hydP = hyd_fn(sP); hyd_delta = hydP - base_hyd
            flyP = (fly_surface_norm_only(sP) if use_surface else fly_score_norm(sP, fly_w))
            fly_delta = flyP - base_fly
            pen_post = count_penalized(sP, pen_set); pen_delta = pen_post - base_pen
            conf = estimate_albumin_conf(sP, alb_df, alb_frag_cache, charges, float(args.ppm),
                                         float(args.rt_total_min), cys_fixed_mod)
            p_cands.append(dict(
                p_idx=p_idx, x_prev=x_prev, tier=tier, rk_before=rk_before,
                replace_pen=replace_pen, distC=abs(p_idx - mut_idx0),
                hyd=hydP, hyd_delta=hyd_delta, fly=flyP, fly_delta=fly_delta,
                pen_post=pen_post, pen_delta=pen_delta, conf_est=conf["conf_est"],
                seqP=sP
            ))
            p_enum_records.append(dict(peptide=pep, mut_idx0=mut_idx0, p_idx=p_idx, x_prev=x_prev,
                                       tier=tier, rk_before=rk_before, replace_pen=replace_pen,
                                       hyd_delta=hyd_delta, fly=flyP, fly_delta=fly_delta, conf_est=conf["conf_est"]))

        if not p_cands:
            row["stage_label"] = "dropped_no_P_positions"; stage_labels.append(row); continue

        if args.avoid_rk_p and any(not c["rk_before"] for c in p_cands):
            p_cands = [c for c in p_cands if not c["rk_before"]]

        def p_key(c):
            key = [c["tier"], 0 if c["replace_pen"] else 1]
            if pen_delta_first:
                key += [pen_w * c["pen_delta"], c["pen_post"]]
            else:
                key += [c["pen_post"], pen_w * c["pen_delta"]]
            key += [-c["distC"], -c["hyd_delta"], -c["fly"], -c["conf_est"]]
            return tuple(key)

        p_cands.sort(key=p_key)
        bestP = p_cands[0]
        p_idx = bestP["p_idx"]; seqP = bestP["seqP"]

        if r_i < examples_to_explain:
            logx(f"[EXPLAIN][{r_i+1}] {pep}  → Cys@{mut_idx0}  → Pro@{p_idx} (prev={bestP['x_prev']}, tier={bestP['tier']}, "
                 f"Δhydro={bestP['hyd_delta']:+.3f}, flyP={bestP['fly']:.3f}, conf_est={bestP['conf_est']:.3f}, "
                 f"{'replaced_penalized' if bestP['replace_pen'] else 'kept_nonpenalized'})")

        # ---------- METHIONINE ----------
        m_positions = [i for i in range(1, L-1) if i not in (mut_idx0, p_idx) and abs(i - mut_idx0) >= 2]
        m_cands = []
        for m_idx in m_positions:
            p_between = (min(mut_idx0, m_idx) < p_idx < max(mut_idx0, m_idx))
            if not p_between and mut_idx0 != 0:
                continue
            sPM = seqP[:m_idx] + "M" + seqP[m_idx+1:]
            hydPM = hyd_fn(sPM); hyd_delta_M = hydPM - hyd_fn(seqP)
            flyPM = (fly_surface_norm_only(sPM) if use_surface else fly_score_norm(sPM, fly_w))
            fly_delta_M = flyPM - (fly_surface_norm_only(seqP) if use_surface else fly_score_norm(seqP, fly_w))
            pen_post_M = count_penalized(sPM, pen_set); pen_delta_M = pen_post_M - count_penalized(seqP, pen_set)
            confM = estimate_albumin_conf(sPM, alb_df, alb_frag_cache, charges, float(args.ppm),
                                          float(args.rt_total_min), cys_fixed_mod)
            replace_pen_M = (seqP[m_idx] in pen_set)
            m_cands.append(dict(
                m_idx=m_idx, replace_pen=replace_pen_M,
                distC=abs(m_idx - mut_idx0), distP=abs(m_idx - p_idx),
                hyd=hydPM, hyd_delta=hyd_delta_M, fly=flyPM, fly_delta=fly_delta_M,
                pen_post=pen_post_M, pen_delta=pen_delta_M, conf_est=confM["conf_est"],
                seqPM=sPM, p_between=p_between
            ))
            m_enum_records.append(dict(peptide=pep, p_idx=p_idx, m_idx=m_idx, replace_pen=replace_pen_M,
                                       distC=abs(m_idx - mut_idx0), distP=abs(m_idx - p_idx),
                                       hyd_delta=hyd_delta_M, fly=flyPM, fly_delta=fly_delta_M,
                                       conf_est=confM["conf_est"], p_between=p_between))

        if not m_cands:
            row["stage_label"] = "dropped_no_M_positions"; stage_labels.append(row); continue

        def m_key(c):
            key = [0 if c["replace_pen"] else 1, -c["distC"], -c["distP"]]
            if pen_delta_first:
                key += [pen_w * c["pen_delta"], c["pen_post"]]
            else:
                key += [c["pen_post"], pen_w * c["pen_delta"]]
            key += [-c["hyd_delta"], -c["fly"], -c["conf_est"]]
            return tuple(key)

        m_cands.sort(key=m_key)
        bestM = m_cands[0]
        m_idx = bestM["m_idx"]; decoy = bestM["seqPM"]

        if r_i < examples_to_explain:
            logx(f"[EXPLAIN][{r_i+1}]    … then Met@{m_idx} (Δhydro={bestM['hyd_delta']:+.3f}, flyM={bestM['fly']:.3f}, "
                 f"conf_est={bestM['conf_est']:.3f}, {'replaced_penalized' if bestM['replace_pen'] else 'kept_nonpenalized'})")
            logx(f"[EXPLAIN][{r_i+1}] Final decoy: {decoy}")

        decoy_rows.append(dict(row,
            peptide=pep, mut_idx0=mut_idx0, indexing_used=row["_indexing_used"], match_check=row["_mut_match"],
            cys_seq=cys_seq, decoy_seq=decoy,
            p_index0=p_idx, m_index0=m_idx,
            p_tier=int(bestP["tier"]), p_prev_res=str(cys_seq[p_idx-1]),
            p_replace_pen=bool(bestP["replace_pen"]), p_dist_to_C=int(bestP["distC"]),
            p_hyd_frac=float(bestP["hyd"]), p_hyd_frac_delta=float(bestP["hyd_delta"]),
            p_fly_signal=float(bestP["fly"]), p_conf_est=float(bestP["conf_est"]),
            p_pen_post=int(bestP["pen_post"]), p_pen_delta=int(bestP["pen_delta"]),
            m_replace_pen=bool(bestM["replace_pen"]), m_dist_to_C=int(bestM["distC"]), m_dist_to_P=int(bestM["distP"]),
            m_hyd_frac=float(bestM["hyd"]), m_hyd_frac_delta=float(bestM["hyd_delta"]),
            m_fly_signal=float(bestM["fly"]), m_conf_est=float(bestM["conf_est"]),
            m_pen_post=int(bestM["pen_post"]), m_pen_delta=int(bestM["pen_delta"]),
            decoy_fly_surface_norm=float(bestM["fly"]) if use_surface else float(fly_surface_norm_only(decoy)),
            decoy_fly_score=float(bestM["fly"]) if not use_surface else float(fly_score_norm(decoy, fly_w)),
        ))

    if not decoy_rows:
        raise RuntimeError("No decoys after Pro/M selection.")

    out_df = pd.DataFrame(decoy_rows)

    # ---- Stage C: albumin chaining (final, with RT gate args.rt_tol) ----
    ppm_tol = float(args.ppm); rt_tol = float(args.rt_tol); rt_total_min = float(args.rt_total_min)

    for i, r in tqdm(out_df.iterrows(), total=len(out_df), desc="Albumin chaining"):
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
        frs_pep = fragment_mz_list(seq, int(args.frag_kmin), int(args.frag_kmax), charges)
        n_types = float(len(frs_pep)) if frs_pep else 1.0
        def gather(idxset: set) -> List[float]:
            out = []
            for jj in idxset:
                arr = alb_frag_cache.get(int(jj))
                if arr: out.extend(arr)
            out.sort(); return out
        prec_fr = gather(matched); prec_rt_fr = gather(matched_rt)
        seen_prec = set(); seen_prec_rt = set()
        for (mz,ion,k,z) in frs_pep:
            if prec_fr and any_within_ppm(mz, prec_fr, ppm_tol): seen_prec.add((ion,k,z))
            if prec_rt_fr and any_within_ppm(mz, prec_rt_fr, ppm_tol): seen_prec_rt.add((ion,k,z))
        out_df.loc[i, "N_precursor_mz"] = float(len(matched))
        out_df.loc[i, "N_precursor_mz_rt"] = float(len(matched_rt))
        out_df.loc[i, "N_fragment_mz_given_precursor"] = float(len(seen_prec))
        out_df.loc[i, "N_fragment_mz_given_precursor_rt"] = float(len(seen_prec_rt))
        out_df.loc[i, "Frac_fragment_mz_given_precursor"] = float(len(seen_prec))/n_types
        out_df.loc[i, "Frac_fragment_mz_given_precursor_rt"] = float(len(seen_prec_rt))/n_types

    # Confusability_simple_albumin (capped counts)
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

    # ---- Stage D: NEW background terminal k-mer burden (XLE) ----
    if term_banks:
        feats = []
        for _, r in tqdm(out_df.iterrows(), total=len(out_df), desc="Background terminal k-mers"):
            t = term_kmer_features(r["decoy_seq"], term_banks, int(args.term_kmin), int(args.term_kmax), args.collapse)
            feats.append(t)
        tdf = pd.DataFrame(feats)
        out_df = pd.concat([out_df.reset_index(drop=True), tdf.reset_index(drop=True)], axis=1)
    else:
        out_df["bg_term_kmer_frac"] = 0.0
        out_df["bg_term_kmer_hits"] = 0.0
        out_df["bg_bkmer_hits"] = 0.0
        out_df["bg_ykmer_hits"] = 0.0

    # ---- Stage E: FINAL SCORE (transparent weighted mix) ----
    # conf_norm in [0,1] already; term in [0,1]; fly_norm in [0,1]
    w_spec = {k.strip(): float(v.strip()) for k,v in (tok.split(":") for tok in args.rank_weights.split(","))}
    w_conf = w_spec.get("conf", 0.6); w_term = w_spec.get("term", 0.25); w_fly = w_spec.get("fly", 0.15)
    Wsum = max(1e-9, w_conf + w_term + w_fly); w_conf/=Wsum; w_term/=Wsum; w_fly/=Wsum

    fly_col = "decoy_fly_surface_norm" if (args.fly_score_mode == "surface") else "decoy_fly_score"
    out_df["final_score"] = (w_conf*out_df["Confusability_simple_albumin"] +
                             w_term*out_df["bg_term_kmer_frac"] +
                             w_fly*out_df[fly_col])

    logx("[INFO] Signal balancing for FINAL score:")
    logx(f"  final_score = {w_conf:.2f}*Confusability_simple_albumin"
         f" + {w_term:.2f}*bg_term_kmer_frac"
         f" + {w_fly:.2f}*{fly_col}")

    # Sort by final score; greedy k-mer de-dup
    s2 = out_df.sort_values(by=["final_score", fly_col, "Confusability_simple_albumin"],
                            ascending=[False, False, False], kind="mergesort")
    N_out = int(args.N); kdedup = int(args.dedup_k)
    if kdedup > 0:
        def kmerset(s: str, k: int): return {s[i:i+k] for i in range(0, max(0, len(s)-k+1))}
        selected_rows = []; seen = set()
        for _, rr in s2.iterrows():
            if len(selected_rows) >= N_out: break
            kms = kmerset(rr["decoy_seq"], kdedup)
            if seen.isdisjoint(kms):
                selected_rows.append(rr); seen.update(kms)
        sel = pd.DataFrame(selected_rows) if selected_rows else s2.head(N_out).copy()
    else:
        sel = s2.head(N_out).copy()

    # Rank labels
    out_df = s2.copy()
    out_df["rank_final"] = np.arange(1, len(out_df)+1)
    sel = sel.copy()
    sel["rank_final"] = out_df.set_index("decoy_seq").loc[sel["decoy_seq"], "rank_final"].values

    # ---- Outputs ----
    prefix = args.output_prefix
    out_path = Path(args.outdir)
    out_df.to_csv(out_path / f"{prefix}_decoy_features.csv", index=False)
    sel.to_csv(out_path / f"{prefix}_selected_top_N.csv", index=False)

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
            hdr = (f">decoy_{i+1}|rank={int(r['rank_final'])}|orig={r['peptide']}|mut_idx0={int(r['mut_idx0'])}|"
                   f"wt={str(r.get('_wt'))}|mut={str(r.get('_mut'))}|"
                   f"p_idx0={int(r['p_index0'])}|m_idx0={int(r['m_index0'])}|"
                   f"indexing={r['indexing_used']}|" + "|".join(extras))
            fh.write(hdr+"\n")
            for line in textwrap.wrap(str(r["decoy_seq"]), 60): fh.write(line+"\n")

    # ---- Write selection explanations ----
    with open(out_path / "SELECTION_EXPLAIN.md", "w") as fh:
        fh.write("\n".join(explain_lines + [
            "",
            "Signals:",
            f"- Confusability_simple_albumin ∈ [0,1] (95th-percentile-capped counts + fractions).",
            f"- bg_term_kmer_frac ∈ [0,1] (XLE-collapsed b/y k-mers vs enlarged background set).",
            f"- {fly_col} ∈ [0,1] (surface-normalized if --fly-score-mode surface; composite otherwise).",
            "",
            f"FINAL SCORE = {w_conf:.2f}*Confusability_simple_albumin + {w_term:.2f}*bg_term_kmer_frac + {w_fly:.2f}*{fly_col}",
            f"Greedy k-mer de-dup (k={kdedup}) applied after sorting.",
        ]))

    # ---- Plots ----
    plots_dir.mkdir(exist_ok=True, parents=True)

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

    # Sequence transitions (orig → Cys → +P → +M) for up to 25 selected
    try:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.55*len(sel.head(25)))))
        ax.axis("off")
        y = 0.95; dy = 0.25
        for _, r in sel.head(25).iterrows():
            pep = r["peptide"]; cys = r["cys_seq"]; decoy = r["decoy_seq"]
            mut_i = int(r["mut_idx0"]); p_i = int(r["p_index0"]); m_i = int(r["m_index0"])
            def draw(seq, yy, hi):
                x0 = 2.0
                for i, ch in enumerate(seq):
                    ax.text(x0 + i*0.5, yy, ch, ha="left", va="center",
                            fontfamily="DejaVu Sans Mono", fontsize=10)
            draw(pep, y, {mut_i:1}); y -= dy
            draw(cys, y, {mut_i:1}); y -= dy
            sP = cys[:p_i]+"P"+cys[p_i+1:]
            draw(sP, y, {mut_i:1, p_i:1}); y -= dy
            draw(decoy, y, {mut_i:1, p_i:1, m_i:1}); y -= dy*0.5
        ax.set_title("Sequence transitions: mutant → Cys → add Pro → add Met")
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_sequence_transitions_selected.png", dpi=170); plt.close(fig)
    except Exception as e:
        print(f"[WARN] transitions plot failed: {e}")

    # Proline tier enumeration, RK|P counts, Δhydro histogram, Δfly vs conf scatter
    try:
        p_enum = pd.DataFrame(p_enum_records)
        if not p_enum.empty:
            tier_names = {0:"GP",1:"AP",2:"LVISTP",3:"other"}
            p_enum["tier_name"] = p_enum["tier"].map(tier_names)
            fig, ax = plt.subplots(); p_enum["tier_name"].value_counts().reindex(["GP","AP","LVISTP","other"]).plot(kind="bar", ax=ax)
            ax.set_title("Enumerated Proline tiers"); ax.set_ylabel("count"); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_tier_enumeration.png"); plt.close(fig)

            fig, ax = plt.subplots(); p_enum["rk_before"].value_counts().reindex([True,False]).plot(kind="bar", ax=ax)
            ax.set_xticklabels(["R/K before P","not R/K before P"], rotation=0)
            ax.set_title("R/K immediately before P (enumerated)"); ax.set_ylabel("count")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5); fig.tight_layout()
            fig.savefig(plots_dir / f"{prefix}_P_rk_before_counts.png"); plt.close(fig)

            fig, ax = plt.subplots(); ax.hist(p_enum["hyd_delta"], bins=25, alpha=0.85)
            xlab = "Δ hydrophobic fraction (vs Cys-only)" if args.hydro_metric=="fraction" else "Δ GRAVY (vs Cys-only)"
            ax.set_title(f"Hydrophobicity change after Pro placement"); ax.set_xlabel(xlab); ax.set_ylabel("frequency")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5); fig.tight_layout()
            fig.savefig(plots_dir / f"{prefix}_P_delta_hydro_hist.png"); plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(p_enum["fly_delta"], p_enum["conf_est"], s=10, alpha=0.6)
            ax.set_xlabel("Δ fly signal (Pro step)"); ax.set_ylabel("albumin confusability (estimate)")
            ax.set_title("Pro step: Δfly vs confusability (all candidates)"); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
            fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_P_scatter_deltafly_vs_conf.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] P-step plots failed: {e}")

    # Methionine step plots
    try:
        m_enum = pd.DataFrame(m_enum_records)
        if not m_enum.empty:
            fig, ax = plt.subplots()
            bins = range(0, (int(m_enum["distC"].max()) if not m_enum["distC"].empty else 1)+2)
            ax.hist(m_enum["distC"], bins=bins, align="left", alpha=0.85)
            ax.set_title("Distance from C (Methionine candidates)"); ax.set_xlabel("|M - C|"); ax.set_ylabel("count")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5); fig.tight_layout()
            fig.savefig(plots_dir / f"{prefix}_M_dist_to_C_hist.png"); plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(m_enum["fly_delta"], m_enum["conf_est"], s=10, alpha=0.6)
            ax.set_xlabel("Δ fly signal (M step)"); ax.set_ylabel("albumin confusability (estimate)")
            ax.set_title("M step: Δfly vs confusability (all candidates)")
            ax.grid(True, ls="--", lw=0.5, alpha=0.5); fig.tight_layout()
            fig.savefig(plots_dir / f"{prefix}_M_scatter_deltafly_vs_conf.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] M-step plots failed: {e}")

    # Final fly vs Confusability scatter (selected highlighted)
    try:
        fig, ax = plt.subplots()
        mask_sel = out_df.index.isin(sel.index)
        ax.scatter(out_df.loc[~mask_sel,fly_col], out_df.loc[~mask_sel,"Confusability_simple_albumin"],
                   s=12, alpha=0.6, label="not selected")
        ax.scatter(out_df.loc[mask_sel,fly_col], out_df.loc[mask_sel,"Confusability_simple_albumin"],
                   s=16, label="selected")
        ax.set_xlabel(f"{fly_col} (decoy)"); ax.set_ylabel("Confusability_simple_albumin (0..1)")
        ax.set_title("Fly signal vs Albumin confusability"); ax.legend()
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_scatter_fly_vs_confusability.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] final scatter failed: {e}")

    # NEW: Background terminal k-mers — hist + scatter vs confusability (selected highlighted)
    try:
        fig, ax = plt.subplots()
        mask_sel = out_df.index.isin(sel.index)
        ax.hist(out_df.loc[~mask_sel,"bg_term_kmer_frac"], bins=25, alpha=0.6, label="not selected")
        ax.hist(out_df.loc[mask_sel,"bg_term_kmer_frac"], bins=25, alpha=0.6, label="selected")
        ax.set_xlabel("Background terminal k-mer fraction"); ax.set_ylabel("count")
        ax.set_title("Terminal k-mer fraction vs background"); ax.legend()
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_hist_bg_term_kmer_frac.png"); plt.close(fig)

        fig, ax = plt.subplots()
        ax.scatter(out_df.loc[~mask_sel,"bg_term_kmer_frac"], out_df.loc[~mask_sel,"Confusability_simple_albumin"],
                   s=12, alpha=0.6, label="not selected")
        ax.scatter(out_df.loc[mask_sel,"bg_term_kmer_frac"], out_df.loc[mask_sel,"Confusability_simple_albumin"],
                   s=16, label="selected")
        ax.set_xlabel("Background terminal k-mer fraction"); ax.set_ylabel("Confusability_simple_albumin")
        ax.set_title("Terminal k-mer (background) vs Albumin confusability"); ax.legend()
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_scatter_term_vs_conf.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] background k-mer plots failed: {e}")

    # NEW: Final score decomposition (stacked bars) for selected
    try:
        fig, ax = plt.subplots(figsize=(max(8, 0.6*len(sel)), 4.2))
        ssel = sel.sort_values("final_score", ascending=False).reset_index(drop=True)
        conf_part = w_conf * ssel["Confusability_simple_albumin"]
        term_part = w_term * ssel["bg_term_kmer_frac"]
        fly_part  = w_fly  * ssel[fly_col]
        x = np.arange(len(ssel))
        ax.bar(x, conf_part, label="conf*w_conf")
        ax.bar(x, term_part, bottom=conf_part, label="term*w_term")
        ax.bar(x, fly_part,  bottom=conf_part+term_part, label="fly*w_fly")
        ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in x], rotation=0)
        ax.set_ylabel("Final score (weighted)"); ax.set_title("Selected decoys: Final score decomposition")
        ax.legend(); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        fig.tight_layout(); fig.savefig(plots_dir / f"{prefix}_bar_final_score_decomposition.png"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] final score decomposition plot failed: {e}")

    # Pairwise scatter plots (candidates vs decoys, selected vs not) incl. 'term'
    try:
        if args.scatter_pairs:
            surv_df = pd.DataFrame(survivors) if isinstance(survivors, list) else pd.DataFrame()
            cand_rows = []
            for _, r in tqdm(surv_df.iterrows(), total=len(surv_df), desc="Props for candidates (scatter)"):
                seq = r["_peptide"]
                props = compute_basic_props(seq, charges, alb_df, alb_frag_cache,
                                            ppm_tol, rt_total_min, hyd_fn, use_surface, fly_w, cys_fixed_mod,
                                            term_banks, int(args.term_kmin), int(args.term_kmax), args.collapse)
                props.update({"category": "candidate", "seq": seq})
                cand_rows.append(props)
            cand_df = pd.DataFrame(cand_rows)

            out_df_with_cat = out_df.copy()
            mask_sel = out_df_with_cat.index.isin(sel.index)
            out_df_with_cat["category"] = np.where(mask_sel, "decoy_selected", "decoy_not_selected")

            dec_rows = []
            for _, r in tqdm(out_df_with_cat.iterrows(), total=len(out_df_with_cat), desc="Props for decoys (scatter)"):
                seq = r["decoy_seq"]
                props = compute_basic_props(seq, charges, alb_df, alb_frag_cache,
                                            ppm_tol, rt_total_min, hyd_fn, use_surface, fly_w, cys_fixed_mod,
                                            term_banks, int(args.term_kmin), int(args.term_kmax), args.collapse)
                props.update({"category": r["category"], "seq": seq})
                dec_rows.append(props)
            dec_df = pd.DataFrame(dec_rows)

            scatter_df = pd.concat([cand_df, dec_df], ignore_index=True, sort=False)
            feat_tokens = [t.strip() for t in str(args.scatter_features).split(",") if t.strip()]
            valid = {"hyd_frac","gravy","fly_surface","fly_comp","conf_est","term","mass","rt","length"}
            feats = [t for t in feat_tokens if t in valid] or ["hyd_frac","fly_surface","conf_est","term","mass","rt","length"]
            make_pair_scatter(scatter_df, feats, plots_dir, prefix, int(args.scatter_max_pairs), int(args.scatter_sample))
    except Exception as e:
        print(f"[WARN] pairwise scatter failed: {e}")

    print(f"[INFO] Output directory: {outdir.resolve()}")
    print("[INFO] Wrote: *_decoy_features.csv, *_selected_top_N.csv, *_all_with_stage_labels.csv, *_stage_counts.csv, FASTA, plots/, SELECTION_EXPLAIN.md")

if __name__ == "__main__":
    main()

