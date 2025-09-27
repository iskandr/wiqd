#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiqd_mzscreen.py — m/z-subsequence screen against common proteins
Streamlined for per-peptide, per-category/protein burden comparison (hits vs non-hits),
with a dedicated 'albumin' category and normalized plots.

USAGE (example):
  python wiqd_mzscreen.py \
    --in peptides.csv \
    --outdir out_mzscreen \
    --kmin 8 --kmax 12 \
    --charges 2,3 \
    --tol 10 --unit ppm \
    --download_common \
    --prep_scenarios clean,antibody,keratin,serum,cytoskeleton,chaperone \
    --scenario_mode subset \
    --effect_threshold 0.147 \
    --prevalence_threshold 0.10 \
    --min_group_n 10 \
    --top_proteins 25

Input CSV must include at least:
  - peptide (string)
Optional:
  - is_hit (0/1)

Outputs (in --outdir):
  - mzscreen_matches__<SCENARIO>.csv
  - mzscreen_matches__ALL.csv
  - per_peptide_category_counts__ALL.csv
  - per_peptide_protein_counts__ALL.csv
  - group_summary__ALL.csv                      [if is_hit present]
  - mzscreen_category_effects__ALL.csv          [if is_hit present]
  - plots/*.png (normalized, per-peptide or per-group)
  - SUMMARY_REPORT.md                           [if is_hit present]
  - README.txt
"""

import os, argparse, re, datetime, json
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- ASCII UI ----------
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
            f"scenarios={getattr(args, 'prep_scenarios', None)} mode={getattr(args, 'scenario_mode', None)}",
            f"prep_config={getattr(args, 'prep_config', None)}",
            f"effect_thr={getattr(args, 'effect_threshold', None)} prev_thr={getattr(args, 'prevalence_threshold', None)}",
            f"min_group_n={getattr(args, 'min_group_n', None)} topN={getattr(args, 'top_proteins', None)}",
        ]
        print("Args: " + " | ".join(arg_summary))
    print()

# ---- Plot defaults ----
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.figsize": (7.6, 4.8),
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

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

# ---- Base accessions ----
COMMON_ACCESSIONS = [
    "P60709","P04406","P07437","P11142","P08238","P68104","P05388","P62805",
    "P11021","P14625","P27824","P27797","P07237","P30101","Q15084","Q15061","Q9BS26",
    "P02768","P01857","P01859","P01860","P01861","P01876","P01877","P01871","P04229",
    "P01854","P01834","P0DOY2"
]

# ---- Scenario definitions (by UniProt accession) ----
DEFAULT_PREP_SCENARIOS = {
    "clean": list(dict.fromkeys(COMMON_ACCESSIONS)),
    "antibody": [
        "P01857","P01859","P01860","P01861","P01876","P01877","P01871",
        "P01834","P0DOY2"
    ],
    "keratin": [
        "P04264","P02533","P04259","P02538","P05787","P05783","P08727",
        "Q04695","P35908","P35527"
    ],
    # Serum EXCLUDING albumin (albumin is its own category below)
    "serum": [
        "P02647","P02652","P02649","P02787","P02671","P02675","P02679",
        "P01009","P68871","P69905"
    ],
    "cytoskeleton": [
        "P60709","P68104","P07437","P68371","Q71U36","Q9BQE3"
    ],
    "chaperone": [
        "P11142","P07900","P08238","P10809","P38646"
    ],
    # NOTE: albumin is handled as a separate CATEGORY; it can also be part of 'clean'
}

# ---- Category map (adds a dedicated 'albumin') ----
CATEGORY_ACCESSIONS = {
    "albumin": {"P02768"},  # ALBU_HUMAN
    "antibody": set(DEFAULT_PREP_SCENARIOS["antibody"]),
    "keratin": set(DEFAULT_PREP_SCENARIOS["keratin"]),
    "serum": set(DEFAULT_PREP_SCENARIOS["serum"]),  # excludes albumin by design
    "cytoskeleton": set(DEFAULT_PREP_SCENARIOS["cytoskeleton"]),
    "chaperone": set(DEFAULT_PREP_SCENARIOS["chaperone"]),
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
            print("[mzscreen] downloading common proteins via UniProt REST...")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            seqs = parse_fasta(r.text)
        except Exception as e:
            print(f"[mzscreen] WARNING: download_common failed ({e}); using embedded fallback.")
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
        print(f"[mzscreen] WARNING: UniProt download failed for scenario accessions ({e}).")
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

def protein_category(accession: str) -> str:
    # Albumin first so it doesn't fall into 'serum'
    if accession in CATEGORY_ACCESSIONS["albumin"]:
        return "albumin"
    for cat, aset in CATEGORY_ACCESSIONS.items():
        if cat == "albumin": continue
        if accession in aset: return cat
    return "other"

# ---- Mass / m/z helpers ----
def calc_mass(seq: str) -> float:
    m = sum(MASS.get(a, float('nan')) for a in seq)
    if np.isnan(m): return float('nan')
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

# ---- Index windows ----
def build_window_index(seqs: Dict[str,str], kmin: int, kmax: int) -> pd.DataFrame:
    rows = []
    for hdr, seq in tqdm(seqs.items(), desc="[mzscreen] indexing protein windows"):
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

# ---- Plot helper ----
def add_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# ---- Per-peptide metrics (simplified) ----
def per_peptide_category_counts(match_all: pd.DataFrame, peptides: List[str], scenarios: List[str]) -> pd.DataFrame:
    """
    Returns long-form table per (scenario, peptide, category) with counts, plus a wide form with n_cat_* per row.
    Also returns total n_matches per peptide/scenario.
    """
    # total matches per peptide/scenario
    tot = (match_all.groupby(["scenario","query_peptide"])
                   .size().reset_index(name="n_matches"))

    # category counts
    cat = (match_all.groupby(["scenario","query_peptide","protein_category"])
                   .size().reset_index(name="n"))
    wide = (cat.pivot_table(index=["scenario","query_peptide"],
                            columns="protein_category",
                            values="n", fill_value=0)
              .reset_index())
    # normalize column names
    wide.rename(columns={c: f"n_cat_{safe_slug(str(c))}" for c in wide.columns if c not in ("scenario","query_peptide")}, inplace=True)

    # ensure (scenario, peptide) completeness
    base = pd.MultiIndex.from_product([scenarios, peptides],
                                      names=["scenario","query_peptide"]).to_frame(index=False)
    out = base.merge(tot, on=["scenario","query_peptide"], how="left") \
              .merge(wide, on=["scenario","query_peptide"], how="left")
    # fill NaNs with 0 for counts; peptides with no matches get n_matches=0
    count_cols = [c for c in out.columns if c.startswith("n_")]
    for c in count_cols:
        out[c] = out[c].fillna(0).astype(float)
    return out

def per_peptide_protein_counts(match_all: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form (scenario, peptide, protein_short, accession, count_windows).
    Only present when count > 0; implicit zeros for others.
    """
    df = (match_all.groupby(["scenario","query_peptide","protein_short","accession"])
                 .size().reset_index(name="n_windows"))
    return df

# ---- Group summary (simple, normalized) ----
def group_summary_simple(per_pep_counts: pd.DataFrame, df_peptides: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per (scenario, is_hit) for:
      - n_matches
      - n_cat_* columns
    Outputs mean (per-peptide), median, and proportion nonzero (per-peptide).
    """
    out_rows = []
    has_hit = "is_hit" in df_peptides.columns
    metrics = ["n_matches"] + [c for c in per_pep_counts.columns if c.startswith("n_cat_")]

    if not has_hit:
        # Only overall summaries
        for scn, sub in per_pep_counts.groupby("scenario", dropna=False):
            for m in metrics:
                vals = pd.to_numeric(sub[m], errors="coerce").fillna(0.0)
                out_rows.append(dict(scenario=scn, is_hit=np.nan, metric=m,
                                     n=len(vals),
                                     mean=float(vals.mean()),
                                     median=float(vals.median()),
                                     prop_nonzero=float((vals>0).mean())))
        return pd.DataFrame(out_rows)

    # Merge is_hit onto per_pep_counts
    tmp = per_pep_counts.merge(
        df_peptides[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )

    for (scn, ih), sub in tmp.groupby(["scenario","is_hit"], dropna=False):
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").fillna(0.0)
            out_rows.append(dict(scenario=scn, is_hit=int(ih),
                                 metric=m,
                                 n=len(vals),
                                 mean=float(vals.mean()),
                                 median=float(vals.median()),
                                 prop_nonzero=float((vals>0).mean())))
    return pd.DataFrame(out_rows)

# ---- Effect sizes on categories (Cliff's δ) ----
def mannwhitney_auc_cliffs(x_hit: pd.Series, x_non: pd.Series):
    x = pd.to_numeric(x_hit, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(x_non, errors="coerce").dropna().to_numpy()
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return np.nan, np.nan
    comb = np.concatenate([x, y])
    ranks = pd.Series(comb).rank(method="average").to_numpy()
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    auc = U1 / (n1*n2)
    delta = 2.0*auc - 1.0
    return float(auc), float(delta)

def compute_category_effects(per_pep_counts: pd.DataFrame, df_peptides: pd.DataFrame,
                             effect_threshold: float, prevalence_threshold: float, min_group_n: int) -> pd.DataFrame:
    if "is_hit" not in df_peptides.columns: return pd.DataFrame()
    tmp = per_pep_counts.merge(
        df_peptides[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    cat_cols = [c for c in per_pep_counts.columns if c.startswith("n_cat_")]
    rows=[]
    for scn, sub in tmp.groupby("scenario", dropna=False):
        for col in cat_cols:
            cat = col.replace("n_cat_","")
            x_hit = sub.loc[sub["is_hit"]==1, col].fillna(0.0)
            x_non = sub.loc[sub["is_hit"]==0, col].fillna(0.0)
            n_hit, n_non = len(x_hit), len(x_non)
            mean_hit, mean_non = float(x_hit.mean()), float(x_non.mean())
            p_nz_hit, p_nz_non = float((x_hit>0).mean()), float((x_non>0).mean())
            if n_hit>=min_group_n and n_non>=min_group_n:
                auc, delta = mannwhitney_auc_cliffs(x_hit, x_non)
                prevalence = max(p_nz_hit, p_nz_non)
                nontrivial = (not np.isnan(delta)) and (abs(delta)>=effect_threshold) and (prevalence>=prevalence_threshold)
            else:
                auc, delta, nontrivial = np.nan, np.nan, False
            rows.append(dict(scenario=scn, category=cat,
                             n_hit=n_hit, n_non=n_non,
                             mean_hit=mean_hit, mean_non=mean_non,
                             prop_nonzero_hit=p_nz_hit, prop_nonzero_non=p_nz_non,
                             auc=auc, cliffs_delta=delta, nontrivial=bool(nontrivial)))
    return pd.DataFrame(rows)

def classify_delta(delta: float) -> str:
    if pd.isna(delta): return "NA"
    a=abs(delta)
    if a<0.147: return "negligible"
    if a<0.33:  return "small"
    if a<0.474: return "medium"
    return "large"

# ---- Plots ----
def plot_box_per_peptide_metric(per_pep_counts: pd.DataFrame, df_peptides: pd.DataFrame, metric: str, plots_dir: str):
    if "is_hit" not in df_peptides.columns: return
    tmp = per_pep_counts.merge(
        df_peptides[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    for scn, sub in tmp.groupby("scenario", dropna=False):
        non = pd.to_numeric(sub.loc[sub["is_hit"]==0, metric], errors="coerce").fillna(0.0).values
        hit = pd.to_numeric(sub.loc[sub["is_hit"]==1, metric], errors="coerce").fillna(0.0).values
        if len(non)==0 and len(hit)==0: continue
        fig, ax = plt.subplots()
        ax.boxplot([non, hit], labels=[f"non-hit (n={len(non)})", f"hit (n={len(hit)})"], showmeans=True, meanline=True)
        ax.set_ylabel(f"{metric} per peptide")
        ax.set_title(f"{metric} per peptide — scenario={scn}")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"box_{safe_slug(metric)}__by_hit__{safe_slug(scn)}.png"))
        plt.close(fig)

def plot_category_effects_heatmap(effect_df: pd.DataFrame, plots_dir: str):
    if effect_df is None or effect_df.empty: return
    mat = effect_df.pivot(index="scenario", columns="category", values="cliffs_delta")
    flags = effect_df.pivot(index="scenario", columns="category", values="nontrivial").fillna(False)
    mat = mat.sort_index(); mat = mat.reindex(columns=sorted(mat.columns))
    flags = flags.reindex(index=mat.index, columns=mat.columns).fillna(False)
    fig_h = 1.2 + 0.6 * max(1, mat.shape[0]); fig_w = 1.6 + 1.1 * max(1, mat.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_title("Category effect size (Cliff's δ): hits vs non-hits")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Cliff's δ (hit > non-hit)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if bool(flags.iloc[i, j]):
                ax.text(j, i, "•", ha="center", va="center", fontsize=12, color="black")
    ax.grid(False); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "heatmap_category_effects__cliffs_delta.png"))
    plt.close(fig)

def plot_flagged_category_boxplots(per_pep_counts: pd.DataFrame, df_peptides: pd.DataFrame,
                                   effect_df: pd.DataFrame, plots_dir: str, max_plots: int = 24):
    if effect_df is None or effect_df.empty: return
    tmp = per_pep_counts.merge(
        df_peptides[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
        on="query_peptide", how="left"
    )
    flagged = effect_df.loc[effect_df["nontrivial"]==True].copy()
    if flagged.empty: return
    flagged["priority"] = flagged["cliffs_delta"].abs()
    flagged.sort_values(["priority","scenario","category"], ascending=[False, True, True], inplace=True)
    emitted = 0
    for _, r in flagged.iterrows():
        if emitted >= max_plots: break
        col = f"n_cat_{safe_slug(str(r['category']))}"
        scn = r["scenario"]
        sub = tmp.loc[tmp["scenario"]==scn]
        if col not in sub.columns: continue
        non = pd.to_numeric(sub.loc[sub["is_hit"]==0, col], errors="coerce").fillna(0.0).values
        hit = pd.to_numeric(sub.loc[sub["is_hit"]==1, col], errors="coerce").fillna(0.0).values
        if len(non)==0 and len(hit)==0: continue
        fig, ax = plt.subplots()
        ax.boxplot([non, hit], labels=[f"non-hit (n={len(non)})", f"hit (n={len(hit)})"], showmeans=True, meanline=True)
        sign = "↑hits" if (not pd.isna(r["cliffs_delta"]) and r["cliffs_delta"]>0) else "↑non-hits"
        ax.set_title(f"{r['category']} — scenario={scn} (δ={r['cliffs_delta']:+.3f}, {sign})")
        ax.set_ylabel(f"{col} per peptide")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"box_{col}__by_hit__{safe_slug(scn)}__FLAG.png"))
        plt.close(fig)
        emitted += 1

# ---- Protein-level comparison (normalized) ----
def protein_group_summary(per_pep_protein: pd.DataFrame, df_peptides: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    For a scenario, compute per-protein:
      - mean per-peptide windows for hits, non-hits (zeros included)
      - delta = mean_hit - mean_non
      - fraction of peptides with any match (hits, non-hits)
    """
    if per_pep_protein.empty: return pd.DataFrame()
    # group sizes
    has_hit = "is_hit" in df_peptides.columns
    if not has_hit: return pd.DataFrame()

    pep_hit = set(df_peptides.loc[df_peptides["is_hit"]==1, "peptide"])
    pep_non = set(df_peptides.loc[df_peptides["is_hit"]==0, "peptide"])
    n_hit = len(pep_hit); n_non = len(pep_non)
    if n_hit==0 or n_non==0: return pd.DataFrame()

    sub = per_pep_protein.loc[per_pep_protein["scenario"]==scenario].copy()
    # Sum counts per protein within each group
    sub_hit = sub.loc[sub["query_peptide"].isin(pep_hit)].groupby(["protein_short","accession"])["n_windows"].agg(["sum","count"])
    sub_non = sub.loc[sub["query_peptide"].isin(pep_non)].groupby(["protein_short","accession"])["n_windows"].agg(["sum","count"])
    # align indexes
    all_idx = sub.groupby(["protein_short","accession"]).size().index
    sub_hit = sub_hit.reindex(all_idx, fill_value=0)
    sub_non = sub_non.reindex(all_idx, fill_value=0)
    # compute normalized metrics
    mean_hit = sub_hit["sum"] / max(1, n_hit)
    mean_non = sub_non["sum"] / max(1, n_non)
    frac_hit = sub_hit["count"].groupby(level=[0,1]).size() if isinstance(sub_hit["count"], pd.Series) else pd.Series(0, index=all_idx)
    # 'count' above is #rows in per_pep_protein (each row = peptide with >0), but if a peptide has multiple windows for same protein, it is a single row already.
    # Convert to unique peptide counts:
    any_hit = sub.loc[sub["query_peptide"].isin(pep_hit)].groupby(["protein_short","accession"])["query_peptide"].nunique()
    any_non = sub.loc[sub["query_peptide"].isin(pep_non)].groupby(["protein_short","accession"])["query_peptide"].nunique()
    any_hit = any_hit.reindex(all_idx, fill_value=0)
    any_non = any_non.reindex(all_idx, fill_value=0)
    frac_any_hit = any_hit / max(1, n_hit)
    frac_any_non = any_non / max(1, n_non)

    out = pd.DataFrame({
        "protein_short": [i[0] for i in all_idx],
        "accession": [i[1] for i in all_idx],
        "mean_per_pep_hit": mean_hit.values.astype(float),
        "mean_per_pep_non": mean_non.values.astype(float),
        "delta": (mean_hit - mean_non).values.astype(float),
        "frac_any_hit": frac_any_hit.values.astype(float),
        "frac_any_non": frac_any_non.values.astype(float),
        "n_hit": n_hit, "n_non": n_non
    })
    # sort by combined burden
    out["combined_mean"] = (out["mean_per_pep_hit"] + out["mean_per_pep_non"]) / 2.0
    out.sort_values(["combined_mean","delta"], ascending=[False, False], inplace=True)
    return out

def plot_protein_ranked_bars(prot_summary: pd.DataFrame, scenario: str, topN: int, plots_dir: str):
    if prot_summary is None or prot_summary.empty: return
    sub = prot_summary.head(topN).copy()
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(sub)+3), 5.2))
    ax.bar(x-0.2, sub["mean_per_pep_non"].values, width=0.4, label=f"non-hit (n={int(sub['n_non'].iloc[0])})")
    ax.bar(x+0.2, sub["mean_per_pep_hit"].values, width=0.4, label=f"hit (n={int(sub['n_hit'].iloc[0])})")
    labels = [f"{p} ({a})" for p,a in zip(sub["protein_short"].values, sub["accession"].values)]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Avg # m/z-sharing windows per peptide")
    ax.set_title(f"Top proteins by per-peptide burden (hits vs non-hits) — scenario={scenario}")
    ax.legend(); add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bar_proteins_ranked__by_mean__{safe_slug(scenario)}.png"))
    plt.close(fig)

    # By delta (hit - non-hit)
    sub2 = prot_summary.sort_values("delta", ascending=False).head(topN)
    x = np.arange(len(sub2))
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(sub2)+3), 5.2))
    ax.bar(x, sub2["delta"].values)
    ax.set_xticks(x); ax.set_xticklabels([f"{p} ({a})" for p,a in zip(sub2["protein_short"].values, sub2["accession"].values)], rotation=90)
    ax.set_ylabel("Δ mean per peptide (hit − non-hit)")
    ax.set_title(f"Top proteins by Δ per-peptide burden — scenario={scenario}")
    add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bar_proteins_ranked__by_delta__{safe_slug(scenario)}.png"))
    plt.close(fig)

    # Fractions with ≥1 match
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(sub)+3), 5.2))
    ax.bar(x-0.2, sub["frac_any_non"].values, width=0.4, label=f"non-hit (n={int(sub['n_non'].iloc[0])})")
    ax.bar(x+0.2, sub["frac_any_hit"].values, width=0.4, label=f"hit (n={int(sub['n_hit'].iloc[0])})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Fraction of peptides with ≥1 match")
    ax.set_ylim(0,1)
    ax.set_title(f"Top proteins: fraction of peptides with ≥1 match — scenario={scenario}")
    ax.legend(); add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bar_proteins_ranked__by_frac_any__{safe_slug(scenario)}.png"))
    plt.close(fig)

# ---- Main analysis ----
def analyze(input_csv: str, outdir: str, kmin: int, kmax: int, charges: List[int], tol: float, unit: str,
            prot_fasta: Optional[str], download_common: bool, accessions_path: Optional[str],
            only_hits: bool,
            prep_scenarios: List[str],
            scenario_mode: str,
            prep_config: Optional[str],
            effect_threshold: float,
            prevalence_threshold: float,
            min_group_n: int,
            top_proteins: int):
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Read peptides
    print("[mzscreen] reading input CSV...")
    df = pd.read_csv(input_csv)
    if "peptide" not in df.columns:
        raise ValueError("Input CSV must include at least 'peptide' column.")
    if only_hits:
        if "is_hit" not in df.columns:
            raise ValueError("--only_hits specified but 'is_hit' column is missing.")
        df = df[df["is_hit"].astype(int)==1].copy()
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df = df[df["peptide"].str.len()>0].copy()
    df["mass"] = df["peptide"].apply(calc_mass)
    has_hit = "is_hit" in df.columns
    if has_hit: df["is_hit"] = df["is_hit"].astype(int)

    # Scenarios
    scenario_defs = dict(DEFAULT_PREP_SCENARIOS)
    if prep_config:
        if not os.path.isfile(prep_config):
            raise FileNotFoundError(f"--prep_config file not found: {prep_config}")
        user_cfg = json.load(open(prep_config,"r"))
        for scn, spec in user_cfg.items():
            if not isinstance(spec, dict): raise ValueError(f"prep_config scenario '{scn}' must be an object.")
            accs = spec.get("accessions", [])
            if accs and not isinstance(accs, list):
                raise ValueError(f"prep_config scenario '{scn}': 'accessions' must be a list.")
            scenario_defs[scn] = list(dict.fromkeys(accs))
    scenarios = [s.strip() for s in prep_scenarios if s.strip()]
    for s in scenarios:
        if s not in scenario_defs and s != "clean":
            print(f"[mzscreen] WARNING: scenario '{s}' not in defaults/config; proceeding with empty list.")

    # Base protein set
    base_seqs = load_common_proteins(prot_fasta, download_common, accessions_path)
    all_fasta_seqs = {}
    if prot_fasta and os.path.isfile(prot_fasta):
        with open_maybe_gzip(prot_fasta) as fh:
            all_fasta_seqs = parse_fasta(fh.read())

    # Run per-scenario screens
    all_match_rows = []
    for scn in scenarios:
        print(f"[mzscreen] === Scenario: {scn} (mode={scenario_mode}) ===")
        scn_accs = scenario_defs.get(scn, [])
        scn_seqs = {}
        if scn == "clean":
            scn_seqs = dict(base_seqs)
        else:
            if all_fasta_seqs and scn_accs:
                scn_seqs = merge_seqs(scn_seqs, select_seqs_by_accessions(all_fasta_seqs, scn_accs))
            if scn_accs:
                scn_seqs = merge_seqs(scn_seqs, select_seqs_by_accessions(base_seqs, scn_accs))
            if download_common and scn_accs:
                have = set(extract_accession(h) for h in scn_seqs.keys())
                need = [a for a in scn_accs if a not in have]
                if need:
                    print(f"[mzscreen] downloading {len(need)} proteins for scenario '{scn}'...")
                    scn_seqs = merge_seqs(scn_seqs, download_uniprot_by_accessions(need))
            if scenario_mode == "extend":
                scn_seqs = merge_seqs(base_seqs, scn_seqs)

        if not scn_seqs:
            print(f"[mzscreen] WARNING: Scenario '{scn}' has no proteins; skipping.")
            continue

        win_df = build_window_index(scn_seqs, kmin, kmax)
        if win_df.empty:
            print(f"[mzscreen] WARNING: Scenario '{scn}' produced zero windows; skipping.")
            continue

        # Annotate categories
        win_df["protein_category"] = win_df["accession"].apply(protein_category)

        # Screen peptides
        rows = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"[mzscreen] screening peptides ({scn})"):
            pep = r["peptide"]; q_mass = r["mass"]; pep_xle = collapse_xle(pep)
            for z in charges:
                q_mz = mz_from_mass(q_mass, z)
                mmin, mmax = mz_window_to_mass_range(q_mz, z, tol, unit)
                left, right = binary_search_mass(win_df, mmin, mmax)
                if right <= left: continue
                sub = win_df.iloc[left:right]
                for _, wr in sub.iterrows():
                    hit_mz = mz_from_mass(wr["mass"], z)
                    rows.append({
                        "scenario": scn,
                        "query_peptide": pep,
                        "charge": z,
                        "window_peptide": wr["window"],
                        "window_len": int(wr["length"]),
                        "window_start": int(wr["start"]),
                        "protein": wr["protein"],
                        "accession": wr["accession"],
                        "protein_short": wr["protein_short"],
                        "protein_category": wr["protein_category"],
                        "window_mz": hit_mz,
                        "identical_seq": int(pep == wr["window"]),
                        "xle_equal": int(pep_xle == wr["window_xle"]),
                    })
        match_df = pd.DataFrame(rows).sort_values(["query_peptide","charge","window_len"])
        scn_match_path = os.path.join(outdir, f"mzscreen_matches__{safe_slug(scn)}.csv")
        match_df.to_csv(scn_match_path, index=False)
        print(f"[mzscreen] Wrote: {scn_match_path}")
        all_match_rows.append(match_df)

    if not all_match_rows:
        raise RuntimeError("No scenario produced matches. Check FASTA/download options and scenarios.")
    all_matches = pd.concat(all_match_rows, ignore_index=True)
    all_matches_path = os.path.join(outdir, "mzscreen_matches__ALL.csv")
    all_matches.to_csv(all_matches_path, index=False)

    # ---- Simplified per-peptide counts ----
    peptides = df["peptide"].drop_duplicates().tolist()
    scenarios_present = sorted(all_matches["scenario"].drop_duplicates().tolist())
    per_pep_cats = per_peptide_category_counts(all_matches, peptides, scenarios_present)
    per_pep_cats_path = os.path.join(outdir, "per_peptide_category_counts__ALL.csv")
    per_pep_cats.to_csv(per_pep_cats_path, index=False)

    per_pep_prot = per_peptide_protein_counts(all_matches)
    per_pep_prot_path = os.path.join(outdir, "per_peptide_protein_counts__ALL.csv")
    per_pep_prot.to_csv(per_pep_prot_path, index=False)

    # ---- Group summaries (normalized) ----
    group_sum = group_summary_simple(per_pep_cats, df)
    group_sum_path = os.path.join(outdir, "group_summary__ALL.csv")
    group_sum.to_csv(group_sum_path, index=False)
    print(f"[mzscreen] Wrote: {per_pep_cats_path} | {per_pep_prot_path} | {group_sum_path}")

    # ---- Core normalized plots ----
    os.makedirs(plots_dir, exist_ok=True)
    # 1) Total burden per peptide: n_matches
    plot_box_per_peptide_metric(per_pep_cats, df, "n_matches", plots_dir)
    # 2) Category burdens per peptide: n_cat_albumin, n_cat_antibody, ...
    for c in [col for col in per_pep_cats.columns if col.startswith("n_cat_")]:
        plot_box_per_peptide_metric(per_pep_cats, df, c, plots_dir)

    # 3) Category effect summary & flagged boxplots
    summary_lines = []
    if has_hit:
        cat_effects = compute_category_effects(per_pep_cats, df, effect_threshold, prevalence_threshold, min_group_n)
        cat_effects["size_class"] = cat_effects["cliffs_delta"].apply(classify_delta)
        cat_effects_path = os.path.join(outdir, "mzscreen_category_effects__ALL.csv")
        cat_effects.to_csv(cat_effects_path, index=False)
        plot_category_effects_heatmap(cat_effects, plots_dir)
        plot_flagged_category_boxplots(per_pep_cats, df, cat_effects, plots_dir, max_plots=24)

        # Build textual guidance for flagged categories
        flagged = cat_effects.loc[cat_effects["nontrivial"]==True].copy()
        if flagged.empty:
            summary_lines.append("**No scenario × category pairs met the non‑trivial threshold** "
                                 f"(|δ| ≥ {effect_threshold}, prevalence ≥ {int(prevalence_threshold*100)}%, "
                                 f"n≥{min_group_n} per group). Consider relaxing thresholds.")
        else:
            flagged["abs_delta"] = flagged["cliffs_delta"].abs()
            flagged.sort_values(["abs_delta","scenario","category"], ascending=[False, True, True], inplace=True)
            summary_lines.append("### Category highlights (non‑trivial differences)\n")
            summary_lines.append(f"Criteria: |δ| ≥ **{effect_threshold}**, prevalence ≥ **{int(prevalence_threshold*100)}%**, n≥**{min_group_n}** per group.\n")
            lines_tbl = ["| Rank | Scenario | Category | δ | Mean(hit) | Mean(non) | NZ%(hit) | NZ%(non) | Size |",
                         "|---:|:--|:--|--:|--:|--:|--:|--:|:--|"]
            for i, r in enumerate(flagged.head(12).itertuples(index=False), 1):
                lines_tbl.append(
                    f"| {i} | {r.scenario} | {r.category} | {r.cliffs_delta:+.3f} | "
                    f"{r.mean_hit:.2f} | {r.mean_non:.2f} | {100*r.prop_nonzero_hit:.0f}% | {100*r.prop_nonzero_non:.0f}% | {r.size_class} |"
                )
            summary_lines.extend(lines_tbl)
            summary_lines.append("")

    # 4) Protein-level ranked comparisons (per scenario), normalized to group size
    if has_hit:
        for scn in scenarios_present:
            ps = protein_group_summary(per_pep_prot, df, scn)
            if ps is None or ps.empty: continue
            # Write per-scenario protein summary for auditability
            ps_path = os.path.join(outdir, f"protein_group_summary__{safe_slug(scn)}.csv")
            ps.to_csv(ps_path, index=False)
            plot_protein_ranked_bars(ps, scn, topN=top_proteins, plots_dir=plots_dir)
            if summary_lines is not None:
                # Add quick suggestion bullets
                head_delta = ps.sort_values("delta", ascending=False).head(5)
                head_mean  = ps.head(5)
                summary_lines.append(f"### Protein comparisons — scenario={scn}")
                summary_lines.append("- **Top Δ (hit − non‑hit)** (per‑peptide normalized): " +
                                     ", ".join([f"{r['protein_short']}({r['accession']}): {r['delta']:+.2f}" for _, r in head_delta.iterrows()]))
                summary_lines.append("- **Top average burden** (either group): " +
                                     ", ".join([f"{r['protein_short']}({r['accession']}): {r['combined_mean']:.2f}" for _, r in head_mean.iterrows()]))
                summary_lines.append(f"  See plots: "
                                     f"`bar_proteins_ranked__by_mean__{safe_slug(scn)}.png`, "
                                     f"`bar_proteins_ranked__by_delta__{safe_slug(scn)}.png`, "
                                     f"`bar_proteins_ranked__by_frac_any__{safe_slug(scn)}.png`.\n")

    # ---- README ----
    lines = []
    lines.append("# m/z-subsequence screen against common proteins (simplified, normalized)\n")
    lines.append(f"- Input peptides: {len(df)}")
    lines.append(f"- Scenarios analyzed: {', '.join(scenarios_present)}")
    lines.append(f"- k-range: {kmin}..{kmax}; charge states: {','.join(map(str,charges))}; tolerance: {tol} {unit}")
    lines.append("- Categories include a dedicated **albumin** bucket (P02768); **serum** excludes albumin.\n")
    lines.append("## Outputs")
    lines.append("- mzscreen_matches__<SCENARIO>.csv  (raw matches per scenario)")
    lines.append("- mzscreen_matches__ALL.csv          (all scenarios concatenated)")
    lines.append("- per_peptide_category_counts__ALL.csv  (per-peptide counts: n_matches and n_cat_*)")
    lines.append("- per_peptide_protein_counts__ALL.csv   (per-peptide counts per protein)")
    if has_hit:
        lines.append("- group_summary__ALL.csv               (normalized summaries for hits vs non-hits)")
        lines.append("- mzscreen_category_effects__ALL.csv   (Cliff’s δ on per-peptide category counts)")
        lines.append("- protein_group_summary__<SCENARIO>.csv (per-protein normalized means, deltas, and fractions)")
        lines.append("- plots/box_n_matches__by_hit__<SCENARIO>.png")
        lines.append("- plots/box_n_cat_<category>__by_hit__<SCENARIO>.png")
        lines.append("- plots/heatmap_category_effects__cliffs_delta.png")
        lines.append("- plots/box_n_cat_<category>__by_hit__<SCENARIO>__FLAG.png (only for flagged cells)")
        lines.append("- plots/bar_proteins_ranked__by_mean__<SCENARIO>.png  (hits vs non-hits, side-by-side)")
        lines.append("- plots/bar_proteins_ranked__by_delta__<SCENARIO>.png (Δ per-peptide burden)")
        lines.append("- plots/bar_proteins_ranked__by_frac_any__<SCENARIO>.png (fraction with ≥1 match)")
        lines.append("- SUMMARY_REPORT.md  (what to inspect with explicit filenames)")
    with open(os.path.join(outdir, "README.txt"), "w") as fh:
        fh.write("\n".join(lines))

    # ---- SUMMARY_REPORT.md ----
    if has_hit:
        report_path = os.path.join(outdir, "SUMMARY_REPORT.md")
        header = [
            "# Summary report",
            "",
            f"*Generated:* {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"*Input:* `{os.path.basename(input_csv)}`",
            f"*Scenarios:* {', '.join(scenarios_present)}",
            f"*k:* {kmin}..{kmax}  |  *charges:* {','.join(map(str,charges))}  |  *tol:* {tol} {unit}",
            "",
            "All plots are **per-peptide** (boxplots) or **normalized by group size** (bars), so group-size differences cannot confound interpretation.",
            "Categories include **albumin** as a separate bucket; **serum** excludes albumin.",
            "",
            "Global summaries:",
            "- **plots/heatmap_category_effects__cliffs_delta.png** — Cliff’s δ heatmap with • markers for non-trivial cells.",
            "- Per-scenario protein comparisons: see `bar_proteins_ranked__*.png` files.",
            "",
        ]
        with open(report_path, "w") as fh:
            fh.write("\n".join(header + (summary_lines or ["(No specific hit vs non‑hit differences were flagged under current thresholds.)"])))
        print(f"[mzscreen] Wrote summary report: {report_path}")

    print(f"[mzscreen] Done.\n  {all_matches_path}\n  {per_pep_cats_path}\n  {per_pep_prot_path}\n  {group_sum_path}\n  plots/[*.png]")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Screen peptides for m/z-sharing subsequences against common proteins, with per-peptide normalized comparisons and a dedicated 'albumin' category.")
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with peptide[, is_hit]")
    ap.add_argument("--outdir", default="mzscreen_out", help="Output directory")
    ap.add_argument("--kmin", type=int, default=7, help="Minimum window k (default 7)")
    ap.add_argument("--kmax", type=int, default=12, help="Maximum window k (default 12)")
    ap.add_argument("--charges", type=str, default="2,3", help="Comma-separated charge states to test (default '2,3')")
    ap.add_argument("--tol", type=float, default=10.0, help="Tolerance value (default 10)")
    ap.add_argument("--unit", type=str, choices=["ppm","da"], default="ppm", help="Tolerance unit (ppm or da)")
    ap.add_argument("--prot_fasta", type=str, default=None, help="FASTA of proteins to screen (optional). If provided, scenarios will try to select from here first.")
    ap.add_argument("--download_common", action="store_true", help="Allow downloads from UniProt for scenario accessions (requires internet)")
    ap.add_argument("--accessions", type=str, default=None, help="Text file with UniProt accessions to download for the 'clean' base panel (optional)")
    ap.add_argument("--only_hits", action="store_true", help="Restrict analysis to rows with is_hit==1 (comparisons disabled)")
    ap.add_argument("--prep_scenarios", type=str, default="clean,antibody,keratin,serum", help="Comma-separated scenario names. Add 'cytoskeleton' and/or 'chaperone' as desired.")
    ap.add_argument("--scenario_mode", type=str, choices=["subset","extend"], default="subset",
                    help="Interpret scenarios as 'subset' (scenario proteins only) or 'extend' (base panel + scenario proteins).")
    ap.add_argument("--prep_config", type=str, default=None, help="JSON file defining/overriding scenarios: {'scen': {'accessions': [...]}, ...}")
    # Effect-size & filtering parameters
    ap.add_argument("--effect_threshold", type=float, default=0.147,
                    help="Threshold on |Cliff's delta| to call a category 'non-trivial' (default 0.147 ≈ 'small').")
    ap.add_argument("--prevalence_threshold", type=float, default=0.10,
                    help="Minimum fraction of peptides with nonzero category count in either group (default 0.10).")
    ap.add_argument("--min_group_n", type=int, default=10,
                    help="Minimum peptides per group (hits and non-hits) within a scenario (default 10).")
    ap.add_argument("--top_proteins", type=int, default=25,
                    help="Top N proteins to plot in ranked comparisons (default 25).")
    args = ap.parse_args()

    banner("WIQD m/z screen (simplified & normalized)", args)
    charges = [int(x) for x in args.charges.split(",") if x.strip()]
    scenarios = [s.strip() for s in args.prep_scenarios.split(",") if s.strip()]
    analyze(
        args.input_csv, args.outdir, args.kmin, args.kmax, charges, args.tol, args.unit,
        args.prot_fasta, args.download_common, args.accessions, args.only_hits,
        scenarios, args.scenario_mode, args.prep_config,
        args.effect_threshold, args.prevalence_threshold, args.min_group_n,
        args.top_proteins
    )

if __name__ == "__main__":
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()

