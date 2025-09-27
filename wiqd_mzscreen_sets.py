#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiqd_mzscreen_sets.py — m/z-subsequence screen by protein SETS
Focus: for each protein set, compare per-peptide m/z-sharing subsequence burden
between hits and non-hits, with 'alone' vs 'union-with-base' modes.

Primary metric:
  - n_set_matches: number of matching subsequence windows (per peptide) to the set.

Outputs:
  - per_peptide_set_counts__ALL.csv    # per-peptide n_set_matches per set×mode (per-peptide = normalized)
  - set_enrichment_summary__ALL.csv    # effect sizes & flags (per set×mode)
  - plots/bar_sets__delta__mode_<MODE>.png
  - plots/box_set__<SET>__mode_<MODE>__FLAG.png              # only for flagged enriched sets
  - plots/bar_frac_any__<SET>__mode_<MODE>__FLAG.png         # only for flagged enriched sets
  - SUMMARY_REPORT.md

Usage (example):
  python wiqd_mzscreen_sets.py \
    --in peptides.csv \
    --outdir out_sets \
    --kmin 8 --kmax 12 \
    --charges 2,3 \
    --tol 10 --unit ppm \
    --download_common \
    --sets antibody,keratin,serum,cytoskeleton,chaperone,albumin \
    --set_mode both \
    --effect_threshold 0.147 --prevalence_threshold 0.10 --min_group_n 10
"""

import os, argparse, re, datetime, json
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- UI ----------
def banner(script_name: str, args: argparse.Namespace = None):
    title = "What Is Qing Doing?"
    bar = "+" + "="*66 + "+"
    lines = [bar, "|{:^66}|".format(title), "|{:^66}|".format(f"({script_name})"), bar]
    print("\n".join(lines))
    print(f"Start: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if args:
        arg_summary = [
            f"input={getattr(args, 'input_csv', None)}",
            f"outdir={getattr(args, 'outdir', None)}",
            f"k={getattr(args, 'kmin', None)}..{getattr(args, 'kmax', None)}",
            f"charges={getattr(args, 'charges', None)} tol={getattr(args, 'tol', None)} {getattr(args, 'unit', None)}",
            f"prot_fasta={getattr(args, 'prot_fasta', None)} download_common={getattr(args, 'download_common', None)}",
            f"sets={getattr(args, 'sets', None)} mode={getattr(args, 'set_mode', None)}",
            f"sets_config={getattr(args, 'sets_config', None)}",
            f"effect_thr={getattr(args, 'effect_threshold', None)} prev_thr={getattr(args, 'prevalence_threshold', None)}",
            f"min_group_n={getattr(args, 'min_group_n', None)}",
        ]
        print("Args: " + " | ".join(arg_summary))
    print()

# ---- Plot defaults (clean, readable) ----
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (8.5, 5.2),
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

# ---- Base/common panel ----
COMMON_ACCESSIONS = [
    "P60709","P04406","P07437","P11142","P08238","P68104","P05388","P62805",
    "P11021","P14625","P27824","P27797","P07237","P30101","Q15084","Q15061","Q9BS26",
    "P02768","P01857","P01859","P01860","P01861","P01876","P01877","P01871","P04229",
    "P01854","P01834","P0DOY2"
]

# ---- Default SET definitions (by UniProt accession) ----
DEFAULT_SETS = {
    # 'base' is implicit; used for 'union' and can be selected explicitly if desired
    "albumin": ["P02768"],
    "antibody": ["P01857","P01859","P01860","P01861","P01876","P01877","P01871","P01834","P0DOY2"],
    "keratin": ["P04264","P02533","P04259","P02538","P05787","P05783","P08727","Q04695","P35908","P35527"],
    # serum without albumin (albumin is its own set)
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
            print("[sets] downloading base/common proteins via UniProt REST ...")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            seqs = parse_fasta(r.text)
        except Exception as e:
            print(f"[sets] WARNING: download_common failed ({e}); using embedded fallback.")
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
        print(f"[sets] WARNING: UniProt download failed for set accessions ({e}).")
        return {}

def select_seqs_by_accessions(seqs: Dict[str,str], accessions: List[str]) -> Dict[str,str]:
    accset = set(accessions)
    out = {}
    for hdr, s in seqs.items():
        if extract_accession(hdr) in accset:
            out[hdr] = s
    return out

def merge_seqs(a: Dict[str,str], b: Dict[str,str]) -> Dict[str,str]:
    out = dict(a); out.update(b); return out

def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")

# ---- Mass & m/z helpers ----
def calc_mass(seq: str) -> float:
    m = 0.0
    for a in seq:
        if a not in MASS: return float("nan")
        m += MASS[a]
    return m + WATER_MASS

def mz_from_mass(neutral_mass: float, z: int) -> float:
    return (neutral_mass + z*PROTON_MASS)/z

def mz_window_to_mass_range(center_mz: float, z: int, tol: float, unit: str) -> Tuple[float,float]:
    delta_mz = center_mz * tol * 1e-6 if unit.lower()=="ppm" else tol
    mzmin, mzmax = center_mz - delta_mz, center_mz + delta_mz
    mmin = z*mzmin - z*PROTON_MASS
    mmax = z*mzmax - z*PROTON_MASS
    return (min(mmin,mmax), max(mmin,mmax))

def collapse_xle(s: str) -> str:
    return "".join("J" if c in ("L","I") else c for c in s)

# ---- Window index ----
def build_window_index(seqs: Dict[str,str], kmin: int, kmax: int) -> pd.DataFrame:
    rows = []
    for hdr, seq in tqdm(seqs.items(), desc="[sets] indexing windows"):
        clean = "".join(c for c in seq if c.isalpha()).upper()
        L = len(clean)
        for k in range(kmin, kmax+1):
            if L < k: continue
            for start in range(0, L-k+1):
                window = clean[start:start+k]
                if any(ch not in MASS for ch in window): continue
                mass = calc_mass(window)
                rows.append({
                    "protein": hdr,
                    "accession": extract_accession(hdr),
                    "protein_short": short_name(hdr),
                    "start": start,
                    "length": k,
                    "window": window,
                    "mass": mass,
                    "window_xle": collapse_xle(window),
                })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.sort_values("mass", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def binary_search_mass(df: pd.DataFrame, mmin: float, mmax: float) -> Tuple[int,int]:
    import bisect
    masses = df["mass"].values
    left = bisect.bisect_left(masses, mmin)
    right = bisect.bisect_right(masses, mmax)
    return left, right

# ---- Effect size (Cliff's δ via Mann–Whitney U) ----
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

# ---- Core analysis ----
def screen_set(seqs: Dict[str,str], peptides_df: pd.DataFrame,
               kmin: int, kmax: int, charges: List[int], tol: float, unit: str) -> pd.DataFrame:
    """
    Return per-peptide count of m/z-sharing windows against 'seqs'.
    Columns: query_peptide, n_set_matches
    """
    win_df = build_window_index(seqs, kmin, kmax)
    if win_df.empty:
        return pd.DataFrame(columns=["query_peptide","n_set_matches"])
    rows = []
    for _, r in tqdm(peptides_df.iterrows(), total=len(peptides_df), desc="[sets] screening peptides"):
        pep = r["peptide"]; q_mass = r["mass"]; pep_xle = collapse_xle(pep)
        n = 0
        for z in charges:
            q_mz = mz_from_mass(q_mass, z)
            mmin, mmax = mz_window_to_mass_range(q_mz, z, tol, unit)
            left, right = binary_search_mass(win_df, mmin, mmax)
            if right <= left: 
                continue
            sub = win_df.iloc[left:right]
            # Count all windows in this range for this charge as matches
            n += len(sub)
        rows.append({"query_peptide": pep, "n_set_matches": int(n)})
    return pd.DataFrame(rows)

def compute_set_effects(per_pep_counts: pd.DataFrame, peptides_df: pd.DataFrame,
                        effect_threshold: float, prevalence_threshold: float, min_group_n: int) -> pd.DataFrame:
    """
    Per set×mode, compute per-peptide mean, prevalence (>=1), Cliff's δ between hits and non-hits,
    and flag enrichment when hits > non-hits with non-trivial effect size & adequate prevalence/N.
    """
    if "is_hit" not in peptides_df.columns:
        return pd.DataFrame()

    # Pivot into columns so that each set×mode is a separate metric series
    # but we will compute effects by grouping on (set_name, mode) directly from long form
    out_rows = []
    for (set_name, mode), sub in per_pep_counts.groupby(["set_name","mode"], dropna=False):
        # Join is_hit
        tmp = sub.merge(
            peptides_df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
            on="query_peptide", how="left"
        )
        x_hit = tmp.loc[tmp["is_hit"]==1, "n_set_matches"].astype(float)
        x_non = tmp.loc[tmp["is_hit"]==0, "n_set_matches"].astype(float)
        n_hit, n_non = len(x_hit), len(x_non)
        mean_hit, mean_non = float(x_hit.mean()) if n_hit else np.nan, float(x_non.mean()) if n_non else np.nan
        prop_nz_hit = float((x_hit>0).mean()) if n_hit else np.nan
        prop_nz_non = float((x_non>0).mean()) if n_non else np.nan
        auc, delta = mannwhitney_auc_cliffs(x_hit, x_non) if (n_hit and n_non) else (np.nan, np.nan)
        prevalence = max(prop_nz_hit if not np.isnan(prop_nz_hit) else 0.0,
                         prop_nz_non if not np.isnan(prop_nz_non) else 0.0)
        enriched = (n_hit>=min_group_n and n_non>=min_group_n and
                    (not np.isnan(delta)) and delta>=effect_threshold and prevalence>=prevalence_threshold)
        out_rows.append(dict(
            set_name=set_name, mode=mode,
            n_hit=n_hit, n_non=n_non,
            mean_hit=mean_hit, mean_non=mean_non,
            delta_mean=mean_hit-mean_non if (not np.isnan(mean_hit) and not np.isnan(mean_non)) else np.nan,
            prop_nonzero_hit=prop_nz_hit, prop_nonzero_non=prop_nz_non,
            auc=auc, cliffs_delta=delta, size_class=classify_delta(delta),
            prevalence=prevalence, enriched=bool(enriched)
        ))
    return pd.DataFrame(out_rows)

# ---- Plots ----
def plot_delta_bars(effects_df: pd.DataFrame, mode: str, effect_threshold: float, plots_dir: str):
    sub = effects_df.loc[effects_df["mode"]==mode].copy()
    if sub.empty: return
    sub.sort_values("cliffs_delta", ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(sub)+3), 5.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["cliffs_delta"].values)
    ax.axhline(effect_threshold, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(sub["set_name"].values, rotation=45, ha="right")
    ax.set_ylabel("Cliff's δ (hits > non-hits)")
    ax.set_title(f"Enrichment effect size by protein set — mode={mode}")
    # mark enriched cells with • above bar
    for i, r in enumerate(sub.itertuples(index=False)):
        if bool(r.enriched):
            ax.text(i, r.cliffs_delta + 0.02, "•", ha="center", va="bottom")
    add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bar_sets__delta__mode_{safe_slug(mode)}.png"))
    plt.close(fig)

def plot_flagged_set_boxplots(per_pep_counts: pd.DataFrame, peptides_df: pd.DataFrame,
                              effects_df: pd.DataFrame, plots_dir: str):
    if "is_hit" not in peptides_df.columns: return
    flagged = effects_df.loc[effects_df["enriched"]==True].copy()
    if flagged.empty: return
    tmp = per_pep_counts.merge(
        peptides_df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    for _, r in flagged.iterrows():
        set_name, mode = r["set_name"], r["mode"]
        sub = tmp.loc[(tmp["set_name"]==set_name) & (tmp["mode"]==mode)].copy()
        if sub.empty: continue
        non = sub.loc[sub["is_hit"]==0, "n_set_matches"].astype(float).values
        hit = sub.loc[sub["is_hit"]==1, "n_set_matches"].astype(float).values
        fig, ax = plt.subplots()
        ax.boxplot([non, hit], labels=[f"non-hit (n={len(non)})", f"hit (n={len(hit)})"], showmeans=True, meanline=True)
        ax.set_ylabel("n_set_matches per peptide")
        ax.set_title(f"{set_name} — mode={mode}  (δ={r['cliffs_delta']:+.3f}, mean Δ={r['delta_mean']:+.2f})")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"box_set__{safe_slug(set_name)}__mode_{safe_slug(mode)}__FLAG.png"))
        plt.close(fig)

def plot_flagged_set_prevalence(per_pep_counts: pd.DataFrame, peptides_df: pd.DataFrame,
                                effects_df: pd.DataFrame, plots_dir: str):
    if "is_hit" not in peptides_df.columns: return
    flagged = effects_df.loc[effects_df["enriched"]==True].copy()
    if flagged.empty: return
    tmp = per_pep_counts.merge(
        peptides_df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    for _, r in flagged.iterrows():
        set_name, mode = r["set_name"], r["mode"]
        sub = tmp.loc[(tmp["set_name"]==set_name) & (tmp["mode"]==mode)].copy()
        if sub.empty: continue
        non = (sub.loc[sub["is_hit"]==0, "n_set_matches"] > 0).mean() if (sub["is_hit"]==0).any() else np.nan
        hit = (sub.loc[sub["is_hit"]==1, "n_set_matches"] > 0).mean() if (sub["is_hit"]==1).any() else np.nan
        fig, ax = plt.subplots()
        ax.bar([0,1], [non, hit])
        ax.set_xticks([0,1]); ax.set_xticklabels(["non-hit", "hit"])
        ax.set_ylim(0,1)
        ax.set_ylabel("Fraction of peptides with ≥1 match")
        ax.set_title(f"{set_name} — mode={mode}  (non-hit={non:.2f}, hit={hit:.2f})")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"bar_frac_any__{safe_slug(set_name)}__mode_{safe_slug(mode)}__FLAG.png"))
        plt.close(fig)

# ---- Analysis orchestrator ----
def analyze(input_csv: str, outdir: str, kmin: int, kmax: int, charges: List[int], tol: float, unit: str,
            prot_fasta: Optional[str], download_common: bool, accessions_path: Optional[str],
            sets: List[str], set_mode: str, sets_config: Optional[str],
            effect_threshold: float, prevalence_threshold: float, min_group_n: int):
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)

    # Read peptides
    print("[sets] reading input CSV ...")
    df = pd.read_csv(input_csv)
    if "peptide" not in df.columns:
        raise ValueError("Input CSV must include at least 'peptide' column.")
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df = df[df["peptide"].str.len()>0].copy()
    df["mass"] = df["peptide"].apply(calc_mass)
    has_hit = "is_hit" in df.columns
    if has_hit: df["is_hit"] = df["is_hit"].astype(int)

    # Base panel sequences
    base_seqs = load_common_proteins(prot_fasta, download_common, accessions_path)
    all_fasta_seqs = {}
    if prot_fasta and os.path.isfile(prot_fasta):
        with open_maybe_gzip(prot_fasta) as fh:
            all_fasta_seqs = parse_fasta(fh.read())

    # Build set definitions (JSON can override/add)
    set_defs = dict(DEFAULT_SETS)
    if sets_config:
        with open(sets_config, "r") as fh:
            user_cfg = json.load(fh)
        for sname, spec in user_cfg.items():
            if isinstance(spec, dict):
                accs = spec.get("accessions", [])
            elif isinstance(spec, list):
                accs = spec
            else:
                raise ValueError(f"Invalid sets_config format for set '{sname}'.")
            set_defs[sname] = list(dict.fromkeys(accs))

    # Resolve which sets to run
    sets_to_run = [s.strip() for s in sets if s.strip()]
    for s in sets_to_run:
        if s not in set_defs and s != "base":
            print(f"[sets] WARNING: set '{s}' not found in defaults/config; it will be skipped unless 'download_common' can fetch it.")

    # Which modes to run
    if set_mode == "both":
        modes = ["alone","union"]
    else:
        modes = [set_mode]

    # Screen per set×mode
    pp_rows = []  # per-peptide rows
    for sname in sets_to_run:
        accs = set_defs.get(sname, [])
        # Select sequences for this set (from provided FASTA, then from base, then optional download)
        set_seqs = {}
        if all_fasta_seqs and accs:
            set_seqs = merge_seqs(set_seqs, select_seqs_by_accessions(all_fasta_seqs, accs))
        if accs:
            set_seqs = merge_seqs(set_seqs, select_seqs_by_accessions(base_seqs, accs))
        if download_common and accs:
            have = set(extract_accession(h) for h in set_seqs.keys())
            need = [a for a in accs if a not in have]
            if need:
                print(f"[sets] downloading {len(need)} proteins for set '{sname}' ...")
                set_seqs = merge_seqs(set_seqs, download_uniprot_by_accessions(need))

        for mode in modes:
            if mode == "alone":
                seqs = dict(set_seqs)
            elif mode == "union":
                seqs = merge_seqs(base_seqs, set_seqs)
            else:
                raise ValueError("Unknown set mode.")
            if not seqs:
                print(f"[sets] WARNING: set '{sname}' (mode={mode}) has no sequences; skipping.")
                continue

            print(f"[sets] === Set: {sname} (mode={mode}) ===")
            per_pep = screen_set(seqs, df[["peptide","mass"]], kmin, kmax, charges, tol, unit)
            if per_pep.empty:
                print(f"[sets] NOTE: no matches for set '{sname}' (mode={mode}).")
            # Ensure every peptide has a row (fill zeros)
            base_index = pd.DataFrame({"query_peptide": df["peptide"].values})
            per_pep = base_index.merge(per_pep, on="query_peptide", how="left")
            per_pep["n_set_matches"] = per_pep["n_set_matches"].fillna(0).astype(int)
            per_pep["set_name"] = sname
            per_pep["mode"] = mode
            pp_rows.append(per_pep)

    if not pp_rows:
        raise RuntimeError("No sets produced results. Check set names/config and FASTA/download options.")
    per_pep_all = pd.concat(pp_rows, ignore_index=True)
    per_pep_path = os.path.join(outdir, "per_peptide_set_counts__ALL.csv")
    per_pep_all.to_csv(per_pep_path, index=False)
    print(f"[sets] Wrote per-peptide counts: {per_pep_path}")

    # Effects and flags
    effects = compute_set_effects(per_pep_all, df, effect_threshold, prevalence_threshold, min_group_n) if has_hit else pd.DataFrame()
    eff_path = os.path.join(outdir, "set_enrichment_summary__ALL.csv")
    if not effects.empty:
        effects.sort_values(["mode","cliffs_delta"], ascending=[True, False], inplace=True)
        effects.to_csv(eff_path, index=False)
        print(f"[sets] Wrote enrichment summary: {eff_path}")
    else:
        print("[sets] 'is_hit' column not found; skipping enrichment summary and plots.")

    # Plots
    if not effects.empty:
        for mode in modes:
            plot_delta_bars(effects, mode, effect_threshold, plots_dir)
        plot_flagged_set_boxplots(per_pep_all, df, effects, plots_dir)
        plot_flagged_set_prevalence(per_pep_all, df, effects, plots_dir)

    # Summary report
    report_path = os.path.join(outdir, "SUMMARY_REPORT.md")
    lines = []
    lines += [
        "# Protein set enrichment summary",
        "",
        f"*Generated:* {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"*Input:* `{os.path.basename(input_csv)}`",
        f"*Sets:* {', '.join(sets_to_run)}",
        f"*Modes:* {', '.join(modes)}",
        f"*k:* {kmin}..{kmax}  |  *charges:* {','.join(map(str,charges))}  |  *tol:* {tol} {unit}",
        "",
        "Interpretation:",
        "- All metrics are **per peptide** or **normalized by group size**.",
        "- **Cliff’s δ** > 0 means *higher in hits*; threshold for non‑trivial enrichment is shown as a dashed line in the δ plots.",
        "",
    ]
    if not effects.empty:
        enriched = effects.loc[effects["enriched"]==True].copy()
        if enriched.empty:
            lines += ["**No sets passed the enrichment threshold** "
                      f"(δ ≥ {effect_threshold}, prevalence ≥ {int(prevalence_threshold*100)}%, n≥{min_group_n} per group)."]
        else:
            lines += ["## Sets with non‑trivial enrichment (hits > non‑hits)\n"]
            enriched["abs_delta"] = enriched["cliffs_delta"].abs()
            enriched.sort_values(["mode","abs_delta"], ascending=[True, False], inplace=True)
            tbl = ["| Mode | Set | δ | Mean(hit) | Mean(non) | NZ%(hit) | NZ%(non) | Size |",
                   "|:--|:--|--:|--:|--:|--:|--:|:--|"]
            for r in enriched.itertuples(index=False):
                tbl.append(f"| {r.mode} | {r.set_name} | {r.cliffs_delta:+.3f} | "
                           f"{r.mean_hit:.2f} | {r.mean_non:.2f} | "
                           f"{100*(r.prop_nonzero_hit or 0):.0f}% | {100*(r.prop_nonzero_non or 0):.0f}% | {r.size_class} |")
            lines += tbl + [
                "",
                "See per‑set plots in `plots/`:",
                "- `bar_sets__delta__mode_alone.png`, `bar_sets__delta__mode_union.png`",
                "- `box_set__<SET>__mode_<MODE>__FLAG.png`",
                "- `bar_frac_any__<SET>__mode_<MODE>__FLAG.png`",
                ""
            ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[sets] Wrote: {report_path}")

    print(f"[sets] Done. Results in: {outdir}")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Compare per-peptide m/z-sharing subsequence burden for protein sets (hits vs non-hits), with 'alone' vs 'union-with-base' modes.")
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with peptide[, is_hit]")
    ap.add_argument("--outdir", default="mzscreen_sets_out", help="Output directory")
    ap.add_argument("--kmin", type=int, default=7, help="Minimum window k (default 7)")
    ap.add_argument("--kmax", type=int, default=12, help="Maximum window k (default 12)")
    ap.add_argument("--charges", type=str, default="2,3", help="Comma-separated charge states to test (default '2,3')")
    ap.add_argument("--tol", type=float, default=10.0, help="Tolerance value (default 10)")
    ap.add_argument("--unit", type=str, choices=["ppm","da"], default="ppm", help="Tolerance unit (ppm or da)")
    ap.add_argument("--prot_fasta", type=str, default=None, help="FASTA of proteins to screen (optional). If provided, sets will try to select from here first.")
    ap.add_argument("--download_common", action="store_true", help="Allow UniProt downloads for missing accessions (requires internet).")
    ap.add_argument("--accessions", type=str, default=None, help="Text file with UniProt accessions to download for the base panel (optional).")
    ap.add_argument("--sets", type=str, default="albumin,antibody,keratin,serum,cytoskeleton,chaperone",
                    help="Comma-separated set names (default built-ins). You can also include 'base'.")
    ap.add_argument("--set_mode", type=str, choices=["alone","union","both"], default="both",
                    help="Test each set alone, in union with base, or both (default both).")
    ap.add_argument("--sets_config", type=str, default=None,
                    help="JSON mapping: {\"set_name\": {\"accessions\": [\"P12345\", ...]}, ...}")
    # Enrichment thresholds
    ap.add_argument("--effect_threshold", type=float, default=0.147,
                    help="Threshold on Cliff's δ to call enrichment (default 0.147 ≈ 'small').")
    ap.add_argument("--prevalence_threshold", type=float, default=0.10,
                    help="Minimum fraction (either group) with ≥1 match to consider (default 0.10).")
    ap.add_argument("--min_group_n", type=int, default=10,
                    help="Minimum peptides per group (hits and non-hits) required (default 10).")
    args = ap.parse_args()

    banner("WIQD m/z screen — protein SETS", args)
    charges = [int(x) for x in args.charges.split(",") if x.strip()]
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    analyze(
        input_csv=args.input_csv, outdir=args.outdir,
        kmin=args.kmin, kmax=args.kmax, charges=charges, tol=args.tol, unit=args.unit,
        prot_fasta=args.prot_fasta, download_common=args.download_common, accessions_path=args.accessions,
        sets=sets, set_mode=args.set_mode, sets_config=args.sets_config,
        effect_threshold=args.effect_threshold, prevalence_threshold=args.prevalence_threshold, min_group_n=args.min_group_n
    )

if __name__ == "__main__":
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()

