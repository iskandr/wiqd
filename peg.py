#!/usr/bin/env python3
"""
PEG-like contaminant isobar scoring + family sweeps + normalized plots.

- Reads a CSV with peptide sequences and a hit column.
- Scores best isobars per peptide vs polymer families (PEG/PPG/PTMEG/PDMS).
- Makes normalized |Δppm| plots and ECDF.
- Sweeps families / endgroups / adduct sets / charge sets; summarizes precision/recall, F1,
  OR/RR, KS statistic; renders PR and Pareto curves; bar/heatmap summaries.
- Ranks HIT peptides by "PEG-likeness" score and plots top-N.

USAGE (minimal):
  python peg_isobar_cli.py --in my_peptides.csv

Key columns (configurable):
  --pepcol peptide   --hitcol is_hit

What to inspect first:
  plots/ecdf_abs_delta_ppm_by_is_hit.png   <-- If HIT curve lies above/left <10–20 ppm,
                                               hits are enriched in near-PEG isobars.

"""

from __future__ import annotations
import argparse
import logging
import math
import os
from collections import Counter
from datetime import datetime
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

# Non-interactive backend (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# -------------------- Logging --------------------
logger = logging.getLogger("peg_isobar_cli")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# -------------------- Constants (monoisotopic) --------------------
H_PROTON = 1.007276466879
H2O = 18.010564684

# Families: repeat masses (monoisotopic)
FAMILIES = {
    "PEG":   44.026214747,  # C2H4O
    "PPG":   58.041865,     # C3H6O
    "PTMEG": 72.057515,     # poly-THF (C4H8O)
    "PDMS":  74.018792,     # (CH3)2SiO
}

# End-group masses per family (neutral). Polyethers share same caps; PDMS: include cyclic (0).
ENDGROUPS_BY_FAMILY: Dict[str, Dict[str, float]] = {
    "PEG":   {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PPG":   {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PTMEG": {"diol": 18.010564684, "monoMe": 32.026214748, "diMe": 46.041864812},
    # PDMS cyclic oligomers D_n have zero end mass; add a "cyclic" channel.
    "PDMS":  {"cyclic": 0.0},
}

# Positive-mode cation adducts (mass contributions to the ion)
ADDUCTS: Dict[str, float] = {
    "H":    1.007276466879,
    "Na":  22.989218,
    "K":   38.963158,
    "NH4": 18.033823,
}

# Monoisotopic residue masses (peptide-bonded residues; water added once at the end)
AA: Dict[str, float] = {
    "G": 57.021463735, "A": 71.037113805, "S": 87.032028435, "P": 97.052763875,
    "V": 99.068413945, "T": 101.047678505, "C": 103.009184505, "L": 113.084064015,
    "I": 113.084064015, "N": 114.042927470, "D": 115.026943065, "Q": 128.058577540,
    "K": 128.094963050, "E": 129.042593135, "M": 131.040484645, "H": 137.058911875,
    "F": 147.068413915, "R": 156.101111050, "Y": 163.063328575, "W": 186.079312980,
}

# -------------------- Mass utilities --------------------
def peptide_mass(seq: str) -> float:
    return sum(AA[aa] for aa in seq) + H2O

def peptide_mz(seq: str, z: int) -> float:
    return (peptide_mass(seq) + z * H_PROTON) / z

def _adduct_multisets(z: int, allowed: Iterable[str]) -> Iterable[Tuple[str, ...]]:
    labels = [a for a in allowed if a in ADDUCTS]
    yield from combinations_with_replacement(labels, z)

def polymer_mass(family: str, endgroup: str, n: int) -> float:
    return ENDGROUPS_BY_FAMILY[family][endgroup] + n * FAMILIES[family]

def polymer_mz(family: str, endgroup: str, n: int, adducts: Tuple[str, ...]) -> Tuple[int, float]:
    z = len(adducts)
    return z, (polymer_mass(family, endgroup, n) + sum(ADDUCTS[a] for a in adducts)) / z

def gaussian_score_delta(delta: float, fwhm: Optional[float]) -> Optional[float]:
    if fwhm is None or fwhm <= 0:
        return None
    sigma = fwhm / 2.35482004503
    try:
        return float(math.exp(-0.5 * (delta / sigma) ** 2))
    except OverflowError:
        return 0.0

# -------------------- Helpers & sanity --------------------
def parse_csv_list_of_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def parse_csv_list_of_str(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())

def parse_semicolon_sets(s: str) -> List[Tuple[str, ...]]:
    """Parse 'A;A,B;B,C,D' -> list of tuples."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part: continue
        out.append(tuple(x.strip() for x in part.split(",") if x.strip()))
    return out

def aa_only(seq: str) -> bool:
    return isinstance(seq, str) and len(seq) > 0 and all(c in AA for c in seq)

def coerce_is_hit(x) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer, float, np.floating)):
        if pd.isna(x): return None
        return 1 if float(x) != 0.0 else 0
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "y", "yes", "hit"):
        return 1
    if s in ("0", "false", "f", "n", "no", "miss", "nonhit", "non-hit", "not"):
        return 0
    return None

def validate_config(args):
    if args.q1_fwhm <= 0:
        raise ValueError("--q1-fwhm must be > 0")
    if args.ppm_window <= 0:
        raise ValueError("--ppm-window must be > 0")
    if len(args.peptide_charges) == 0 or any(z <= 0 for z in args.peptide_charges):
        raise ValueError("--peptide-charges must be positive integers")
    if args.n_range[0] > args.n_range[1]:
        raise ValueError("--n-range start must be <= end")
    if args.mz_range[0] >= args.mz_range[1]:
        raise ValueError("--mz-range min must be < max")
    for fam in args.families:
        if fam not in FAMILIES:
            raise ValueError(f"--families includes unknown '{fam}'. Valid: {list(FAMILIES)}")
    for fam in args.families:
        for e in args.endgroups:
            if e == "auto": continue
            if e not in ENDGROUPS_BY_FAMILY.get(fam, {}):
                raise ValueError(f"--endgroups includes '{e}' not valid for {fam}. "
                                 f"Valid for {fam}: {list(ENDGROUPS_BY_FAMILY.get(fam,{}))}")
    for a in args.adducts:
        if a not in ADDUCTS:
            raise ValueError(f"--adducts includes unknown '{a}'. Valid: {list(ADDUCTS)}")

def ensure_single_hit_col(df: pd.DataFrame, hit_col: str, outdir: Path, behavior: str = "error") -> pd.DataFrame:
    """If multiple columns named hit_col exist, verify agreement and collapse to one."""
    dup_names = [c for c in df.columns if c == hit_col]
    if len(dup_names) <= 1:
        df[hit_col] = df[hit_col].apply(coerce_is_hit).astype("Int64")
        return df
    dup_df = df.loc[:, dup_names].copy()
    for c in dup_df.columns:
        dup_df[c] = dup_df[c].apply(coerce_is_hit).astype("Int64")
    mask_all_valid = dup_df.notna().all(axis=1)
    agrees = (dup_df.loc[mask_all_valid].nunique(axis=1) == 1)
    n_valid = int(mask_all_valid.sum())
    n_disagree = int((~agrees).sum())
    if n_valid > 0:
        logger.info(f"[hit-col] duplicates={dup_names}, rows both non-null={n_valid}, disagreements={n_disagree}")
    if n_disagree > 0:
        out = pd.concat([df, dup_df.add_prefix("DUP_")], axis=1).loc[mask_all_valid & (~agrees)]
        path = outdir / "hit_disagreements.csv"
        out.to_csv(path, index=False)
        msg = f"Duplicate hit columns disagree on {n_disagree}/{n_valid} rows (see {path})."
        if behavior == "error":
            raise ValueError(msg)
        elif behavior not in ("warn_prefer_input", "warn_prefer_nonnull"):
            raise ValueError(f"Unknown --hit-conflict behavior '{behavior}'")
        else:
            logger.warning(msg + f" Proceeding with '{behavior}'.")
    cons = dup_df.bfill(axis=1).iloc[:, 0] if behavior == "warn_prefer_nonnull" else dup_df.iloc[:, 0]
    df_out = df.drop(columns=dup_names)
    df_out[hit_col] = cons.astype("Int64")
    return df_out

# -------------------- Candidate enumeration & scoring --------------------
def build_family_candidates(
    families: Iterable[str],
    peg_charges: Iterable[int],
    allowed_adducts: Iterable[str],
    endgroups: Iterable[str],  # may include "auto"
    n_min: int, n_max: int,
    mz_min: float, mz_max: float,
    show_progress: bool = True,
) -> List[Tuple[str, str, int, Tuple[str, ...], int, float]]:
    """
    Build candidates across families; returns list of tuples:
      (family, endgroup, n, adducts_tuple, peg_z, mz)
    """
    # Resolve endgroups per family (respect 'auto')
    endgroups_map: Dict[str, List[str]] = {}
    for fam in families:
        if "auto" in endgroups:
            endgroups_map[fam] = list(ENDGROUPS_BY_FAMILY[fam].keys())
        else:
            endgroups_map[fam] = [e for e in endgroups if e in ENDGROUPS_BY_FAMILY[fam]]

    adducts_list = list(allowed_adducts)
    k = len(adducts_list)
    total_combos = 0
    for z in peg_charges:
        total_combos += math.comb(k + z - 1, z)
    total_iters = sum(len(endgroups_map[fam]) for fam in families) * (n_max - n_min + 1) * total_combos
    candidates: List[Tuple[str, str, int, Tuple[str, ...], int, float]] = []

    pbar = tqdm(total=total_iters, disable=not show_progress, desc="Enumerating candidates")
    for fam in families:
        for end in endgroups_map[fam]:
            for n in range(n_min, n_max + 1):
                for z in peg_charges:
                    for ads in _adduct_multisets(z, allowed=adducts_list):
                        peg_z, mz = polymer_mz(fam, end, n, ads)
                        if mz_min <= mz <= mz_max:
                            candidates.append((fam, end, n, ads, peg_z, mz))
                        pbar.update(1)
    pbar.close()
    logger.info(f"[candidates] kept={len(candidates)} in m/z [{mz_min}, {mz_max}]")
    if not candidates:
        raise RuntimeError("No candidates generated. Check ranges/filters.")
    return candidates

def best_isobar_for_peptide(
    seq: str,
    candidates: List[Tuple[str, str, int, Tuple[str, ...], int, float]],
    cand_mz: np.ndarray,
    peptide_charges: Iterable[int],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str = "ppm",
    preselect: int = 200,
) -> Dict[str, object]:
    """Return ONE best row across candidate set for this peptide."""
    rows: List[Dict[str, object]] = []
    for pep_z in peptide_charges:
        pep_mz = peptide_mz(seq, pep_z)
        idx = np.argsort(np.abs(cand_mz - pep_mz))[:preselect]
        for j in idx:
            fam, end, n, ads, peg_z, mz_poly = candidates[j]
            d_mz = mz_poly - pep_mz
            d_ppm = (d_mz / pep_mz) * 1e6
            score_q1 = gaussian_score_delta(d_mz, q1_fwhm_da)
            score_ppm = gaussian_score_delta(d_ppm, ppm_window)
            score_used = score_ppm if score_priority == "ppm" else score_q1
            rows.append({
                "peptide": seq,
                "pep_z": pep_z,
                "pep_mz": pep_mz,
                "family": fam,
                "peg_z": peg_z,
                "peg_endgroup": end,
                "peg_n": n,
                "peg_adducts": "+".join(ads),
                "peg_mz": mz_poly,
                "delta_mDa": 1000.0 * d_mz,
                "delta_ppm": d_ppm,
                "score_q1": score_q1,
                "score_ppm": score_ppm,
                "score_used": score_used,
            })
    rows.sort(key=lambda r: (-(r["score_used"] if r["score_used"] is not None else 0), abs(r["delta_mDa"])))
    best = rows[0]
    for k, nd in [("pep_mz",6),("peg_mz",6),("delta_mDa",3),("delta_ppm",2),("score_q1",4),("score_ppm",4),("score_used",4)]:
        best[k] = round(best[k], nd)
    if abs(best["delta_ppm"]) > 50:
        best["warning"] = f"Large ppm deviation ({best['delta_ppm']:+.1f} ppm)"
    return best

def best_isobars_for_peptides(
    peptides: List[str],
    peptide_charges: Iterable[int],
    candidates: List[Tuple[str, str, int, Tuple[str, ...], int, float]],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str,
    preselect: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    cand_mz = np.array([c[5] for c in candidates], dtype=float)
    rows = []
    pbar = tqdm(peptides, disable=not show_progress, desc="Scoring peptides")
    for seq in pbar:
        rows.append(
            best_isobar_for_peptide(
                seq, candidates=candidates, cand_mz=cand_mz,
                peptide_charges=peptide_charges, q1_fwhm_da=q1_fwhm_da,
                ppm_window=ppm_window, score_priority=score_priority,
                preselect=preselect
            )
        )
    df = pd.DataFrame(rows)
    want = ["peptide","pep_z","pep_mz","family","peg_z","peg_endgroup","peg_n","peg_adducts",
            "peg_mz","delta_mDa","delta_ppm","score_q1","score_ppm","score_used","warning"]
    for c in want:
        if c not in df.columns: df[c] = np.nan
    return df[want]

# -------------------- Derived features & stats --------------------
def adduct_feature_counts(adduct_str: str) -> Dict[str, int]:
    parts = [p for p in str(adduct_str).split("+") if p]
    c = Counter(parts)
    return {"cnt_H": c.get("H",0), "cnt_Na": c.get("Na",0), "cnt_K": c.get("K",0), "cnt_NH4": c.get("NH4",0)}

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_delta_ppm"] = df["delta_ppm"].abs()
    df["abs_delta_mDa"] = df["delta_mDa"].abs()
    counts = df["peg_adducts"].apply(adduct_feature_counts).apply(pd.Series)
    df = pd.concat([df, counts], axis=1)
    for ion in ("H","Na","K","NH4"):
        df[f"has_{ion}"] = (df[f"cnt_{ion}"] > 0).astype(int)
    df["adduct_count"] = df[["cnt_H","cnt_Na","cnt_K","cnt_NH4"]].sum(axis=1)
    return df

def point_biserial_corr(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1); x = np.asarray(x).reshape(-1)
    n = min(len(y), len(x))
    if n == 0: return 0.0
    y = y[:n]; x = x[:n]
    if np.nanstd(y) == 0 or np.nanstd(x) == 0: return 0.0
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2: return 0.0
    return float(np.corrcoef(y[mask], x[mask])[0,1])

def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS D statistic (no SciPy)."""
    x = np.sort(x); y = np.sort(y)
    xs = np.r_[x, y]
    xs = np.unique(xs)
    Fx = np.searchsorted(x, xs, side='right')/len(x) if len(x) else np.zeros_like(xs, dtype=float)
    Fy = np.searchsorted(y, xs, side='right')/len(y) if len(y) else np.zeros_like(xs, dtype=float)
    return float(np.max(np.abs(Fx - Fy))) if len(xs) else 0.0

# -------------------- Binning / thresholds --------------------
def ppm_bin_labels(edges: List[float]) -> List[str]:
    labels = [f"≤{int(e)}" for e in edges]
    labels.append(f">{int(edges[-1])}")
    return labels
def compute_ppm_bin_fractions(df: pd.DataFrame, hit_col: str, edges: List[float]) -> pd.DataFrame:
    """
    Bin |Δppm| into [0,e1], (e1,e2], ... , (elast, inf). Return counts and *fractions per hit group*.
    Robust to empty/missing groups and preserves `group_total`.
    """
    # Sanity checks
    if hit_col not in df.columns:
        raise KeyError(f"compute_ppm_bin_fractions: '{hit_col}' not in df.columns")
    if "abs_delta_ppm" not in df.columns:
        raise KeyError("compute_ppm_bin_fractions: 'abs_delta_ppm' not in df.columns")

    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()

    # Define bins and labels
    bins = [-np.inf] + edges + [np.inf]
    labels = [f"≤{int(e)}" for e in edges] + [f">{int(edges[-1])}"]

    # Assign bins (ordered categorical)
    sub["ppm_bin"] = pd.cut(
        sub["abs_delta_ppm"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )

    # Build a complete (group × bin) grid so missing combos are present with count=0
    groups = np.sort(sub[hit_col].dropna().unique())
    idx = pd.MultiIndex.from_product([groups, labels], names=[hit_col, "ppm_bin"])

    # Count and reindex to full grid
    counts = (
        sub.groupby([hit_col, "ppm_bin"], observed=False)
           .size()
           .reindex(idx, fill_value=0)
           .rename("count")
           .reset_index()
    )

    # Totals from the reindexed counts (so they *always* exist)
    totals = (
        counts.groupby(hit_col, observed=False)["count"]
              .sum()
              .rename("group_total")
              .reset_index()
    )

    # Merge totals and compute normalized fractions
    out = counts.merge(totals, on=hit_col, how="left")
    out["fraction"] = out["count"] / out["group_total"].replace(0, np.nan)

    # Ensure ppm_bin stays ordered categorical for stable plotting
    out["ppm_bin"] = pd.Categorical(out["ppm_bin"], categories=labels, ordered=True)

    # Optional diagnostics (uncomment if helpful)
    # logger.info("[ppm bins] group totals:\n" + out[[hit_col, "group_total"]].drop_duplicates().to_string(index=False))

    return out



def threshold_summary(df: pd.DataFrame, hit_col: str, thresholds: List[float]) -> pd.DataFrame:
    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()
    rows = []
    n_hit = int(sub[hit_col].sum())
    n_nonhit = int((1 - sub[hit_col]).sum())
    for T in thresholds:
        inB = (sub["abs_delta_ppm"] <= T)
        a = int(((sub[hit_col] == 1) & inB).sum())
        b = n_hit - a
        c = int(((sub[hit_col] == 0) & inB).sum())
        d = n_nonhit - c
        p1 = a / n_hit if n_hit else np.nan
        p2 = c / n_nonhit if n_nonhit else np.nan
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan
        a1,b1,c1,d1 = a+0.5, b+0.5, c+0.5, d+0.5
        or_ = (a1*d1)/(b1*c1)
        se_log_or = math.sqrt(1/a1 + 1/b1 + 1/c1 + 1/d1)
        ci_lo = math.exp(math.log(or_) - 1.96*se_log_or)
        ci_hi = math.exp(math.log(or_) + 1.96*se_log_or)
        rows.append({
            "threshold_ppm": T,
            "n_hit_total": n_hit, "n_nonhit_total": n_nonhit,
            "n_hit_leT": a, "n_nonhit_leT": c,
            "frac_hit_leT": p1, "frac_nonhit_leT": p2,
            "precision": (a / (a + c)) if (a + c) else np.nan,
            "recall": p1,
            "F1": (2*(a/(a+c))*p1/((a/(a+c))+p1)) if (a+c) and (p1==p1) and ((a/(a+c))+p1)>0 else np.nan,
            "risk_ratio": rr,
            "odds_ratio": or_, "or_95ci_lo": ci_lo, "or_95ci_hi": ci_hi,
            "nonhit_capture_frac": (c / n_nonhit) if n_nonhit else np.nan,
        })
    return pd.DataFrame(rows)

# -------------------- Plotting --------------------
def save_plot(fig, outpath: Path):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_ecdf_by_hit(df: pd.DataFrame, hit_col: str, outdir: Path, thresholds: List[float], x_max: float = 200.0):
    sub = df[[hit_col, "abs_delta_ppm"]].dropna()
    if sub.empty: return
    fig = plt.figure()
    for label, grp in sub.groupby(hit_col):
        x = np.sort(grp["abs_delta_ppm"].to_numpy(float))
        y = np.arange(1, len(x)+1)/len(x)
        plt.step(x, y, where="post", label=f"{hit_col}={label}")
    for T in thresholds: plt.axvline(T, linestyle="--", linewidth=1)
    plt.xlim(0, x_max); plt.ylim(0, 1)
    plt.xlabel("|Δppm|"); plt.ylabel("cumulative fraction ≤ x (within group)")
    plt.title("ECDF of |Δppm| by hit status")
    plt.legend()
    save_plot(fig, outdir / "ecdf_abs_delta_ppm_by_is_hit.png")

def plot_ppm_bin_fractions(df_bins: pd.DataFrame, hit_col: str, outdir: Path, edges: List[float]):
    if df_bins.empty: return
    bin_order = ppm_bin_labels(edges)
    fig = plt.figure(figsize=(max(6, 1.1*len(bin_order)), 4.5))
    groups = sorted(df_bins[hit_col].dropna().unique())
    width = 0.35 if len(groups) == 2 else 0.25
    x = np.arange(len(bin_order))
    for i, g in enumerate(groups):
        sub = df_bins[df_bins[hit_col] == g].copy()
        sub["ppm_bin"] = pd.Categorical(sub["ppm_bin"], categories=bin_order, ordered=True)
        sub = sub.set_index("ppm_bin").reindex(bin_order)
        y = sub["fraction"].to_numpy(float)
        plt.bar(x + (i - (len(groups)-1)/2)*width, y, width=width, label=f"{hit_col}={int(g)}")
    plt.xticks(x, bin_order); plt.ylim(0, 1)
    plt.xlabel("abs(Δppm) bins"); plt.ylabel("fraction within group")
    plt.title("Fraction in |Δppm| bins by hit status (normalized)")
    plt.legend()
    save_plot(fig, outdir / "frac_bins_abs_delta_ppm_by_is_hit.png")

def plot_pr_curve(thr_df: pd.DataFrame, family_label: str, outpath: Path):
    """Precision-Recall curve using threshold rows in order."""
    if thr_df.empty: return
    fig = plt.figure()
    plt.plot(thr_df["recall"].to_numpy(float), thr_df["precision"].to_numpy(float), marker="o")
    plt.xlabel("Recall (hit coverage)"); plt.ylabel("Precision (hit fraction among explained)")
    plt.title(f"PR curve: {family_label}")
    save_plot(fig, outpath)

def plot_pareto(thr_df: pd.DataFrame, family_label: str, outpath: Path):
    """Non-hit capture vs hit coverage (Pareto). Lower-left is better."""
    if thr_df.empty: return
    fig = plt.figure()
    x = thr_df["nonhit_capture_frac"].to_numpy(float)
    y = thr_df["recall"].to_numpy(float)
    plt.plot(x, y, marker="o")
    plt.xlabel("Non-hit capture fraction (≤T)"); plt.ylabel("Hit coverage (≤T)")
    plt.title(f"Pareto: {family_label}")
    save_plot(fig, outpath)

def plot_bar_metric(df: pd.DataFrame, group_col: str, metric_col: str, title: str, outpath: Path, top_k: int = 12):
    """Bar plot of metric by group (top_k)."""
    if df.empty: return
    sub = df.sort_values(metric_col, ascending=False).head(top_k)
    fig = plt.figure(figsize=(10, min(0.5*len(sub)+2, 12)))
    y = np.arange(len(sub))
    plt.barh(y, sub[metric_col].to_numpy(float))
    plt.yticks(y, sub[group_col].astype(str).tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(metric_col); plt.title(title)
    save_plot(fig, outpath)

def plot_heatmap_metric(pivot: pd.DataFrame, title: str, outpath: Path):
    if pivot.empty: return
    fig = plt.figure(figsize=(max(6, 0.9*pivot.shape[1]), max(4, 0.45*pivot.shape[0])))
    plt.imshow(pivot.to_numpy(float), aspect="auto")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150); plt.close(fig)

# -------------------- Sweeps --------------------
def format_tuple_str(t: Tuple[str, ...]) -> str:
    return ",".join(t)

def sweep_families_configs(
    df_in: pd.DataFrame, peptides: List[str],
    peptide_charges: Tuple[int, ...],
    families: Tuple[str, ...],
    endgroup_sets: List[Tuple[str, ...]],
    adduct_sets: List[Tuple[str, ...]],
    peg_charge_sets: List[Tuple[int, ...]],
    n_range: Tuple[int, int],
    mz_range: Tuple[float, float],
    q1_fwhm: float, ppm_window: float,
    thresholds: List[float],
    preselect: int,
    outdir: Path,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Sweep across families x endgroup_sets x adduct_sets x peg_charge_sets.
    For each config:
      - build candidates
      - score best (ppm priority)
      - join hits
      - compute threshold summary, KS statistic
    Returns a long DataFrame with metrics per config.
    """
    rows = []
    total = len(families) * len(endgroup_sets) * len(adduct_sets) * len(peg_charge_sets)
    pbar = tqdm(total=total, disable=not show_progress, desc="Sweeping configs")
    for fam, egs, ads, zset in product(families, endgroup_sets, adduct_sets, peg_charge_sets):
        # Resolve endgroups for family
        fam_egs = tuple(ENDGROUPS_BY_FAMILY[fam].keys()) if ("auto" in egs) else tuple([e for e in egs if e in ENDGROUPS_BY_FAMILY[fam]])
        if not fam_egs:
            pbar.update(1); continue
        try:
            candidates = build_family_candidates(
                families=(fam,),
                peg_charges=zset,
                allowed_adducts=ads,
                endgroups=fam_egs,
                n_min=n_range[0], n_max=n_range[1],
                mz_min=mz_range[0], mz_max=mz_range[1],
                show_progress=False
            )
            if not candidates:
                pbar.update(1); continue
            df_best = best_isobars_for_peptides(
                peptides, peptide_charges=peptide_charges,
                candidates=candidates, q1_fwhm_da=q1_fwhm,
                ppm_window=ppm_window, score_priority="ppm",
                preselect=preselect, show_progress=False
            )
            # Join hits
            join = df_in[[args.pepcol, args.hitcol]].merge(df_best, left_on=args.pepcol, right_on="peptide", how="left")
            join[args.hitcol] = join[args.hitcol].apply(coerce_is_hit)
            dj = add_derived_features(join)

            # Metrics at thresholds
            thr = threshold_summary(dj, hit_col=args.hitcol, thresholds=thresholds)
            # KS on abs_delta_ppm by group
            v_hit = dj.loc[dj[args.hitcol]==1, "abs_delta_ppm"].dropna().to_numpy(float)
            v_non = dj.loc[dj[args.hitcol]==0, "abs_delta_ppm"].dropna().to_numpy(float)
            ksD = ks_statistic(v_hit, v_non)

            # Flatten metrics for each threshold
            for _, r in thr.iterrows():
                rows.append({
                    "family": fam,
                    "endgroups": format_tuple_str(fam_egs),
                    "adducts": format_tuple_str(ads),
                    "peg_charges": ",".join(map(str, zset)),
                    "threshold_ppm": r["threshold_ppm"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "F1": r["F1"],
                    "risk_ratio": r["risk_ratio"],
                    "odds_ratio": r["odds_ratio"],
                    "nonhit_capture_frac": r["nonhit_capture_frac"],
                    "ks_D": ksD,
                    "n_hit_total": r["n_hit_total"],
                    "n_nonhit_total": r["n_nonhit_total"],
                    "n_hit_leT": r["n_hit_leT"],
                    "n_nonhit_leT": r["n_nonhit_leT"],
                })
        except Exception as e:
            logger.warning(f"Sweep config failed: family={fam}, egs={egs}, ads={ads}, z={zset}: {e}")
        pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    return df

# -------------------- Hits sorted by score --------------------
def make_hits_sorted_by_score(
    df_in: pd.DataFrame,
    df_best: pd.DataFrame,        # best-per-peptide table (Q1 or PPM)
    pepcol: str,
    hit_col: str,
    mode: str,                    # "ppm", "q1", "combo"
    top_n: int,
    outdir: Path
):
    keep_cols = ["peptide","family","score_ppm","score_q1","delta_ppm","delta_mDa",
                 "peg_endgroup","peg_z","peg_n","peg_adducts","peg_mz","pep_z","pep_mz"]
    for c in keep_cols:
        if c not in df_best.columns:
            df_best[c] = np.nan
    tmp = df_best.merge(df_in[[pepcol, hit_col]], left_on="peptide", right_on=pepcol, how="left")
    hits = tmp[tmp[hit_col] == 1].copy()
    if hits.empty:
        logger.warning("No hits found to rank for the PEG score plot.")
        return
    if mode == "ppm":
        hits["peg_artifact_score"] = hits["score_ppm"]
    elif mode == "q1":
        hits["peg_artifact_score"] = hits["score_q1"]
    elif mode == "combo":
        hits["peg_artifact_score"] = 0.5*(hits["score_ppm"].fillna(0) + hits["score_q1"].fillna(0))
    else:
        raise ValueError(f"Unknown peg-score mode '{mode}'")
    hits = hits.dropna(subset=["peg_artifact_score"])
    if hits.empty:
        logger.warning(f"No finite PEG scores for hits using mode '{mode}'."); return
    hits_sorted = hits.sort_values("peg_artifact_score", ascending=False).head(top_n)
    csv_path = outdir / f"hits_sorted_by_peg_score_{mode}.csv"
    hits_sorted.to_csv(csv_path, index=False)
    logger.info(f"Wrote: {csv_path}")
    fig = plt.figure(figsize=(10, min(0.5*len(hits_sorted)+2, 12)))
    y = np.arange(len(hits_sorted))
    plt.barh(y, hits_sorted["peg_artifact_score"].to_numpy(float))
    plt.yticks(y, hits_sorted["peptide"].astype(str).tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(f"PEG-like score ({mode})")
    plt.title(f"Top {min(top_n, len(hits_sorted))} HIT peptides by PEG-like score ({mode})")
    save_plot(fig, outdir / f"hits_sorted_by_peg_score_{mode}.png")

# -------------------- Category hit-rate (optional) --------------------
def category_hit_rates(df_join: pd.DataFrame, hit_col: str) -> pd.DataFrame:
    rows = []
    if "family" in df_join.columns:
        for grp, sub in df_join.groupby("family"):
            n = len(sub); hits = sub[hit_col].sum(skipna=True)
            rows.append({"category":"family","level":grp,"n":n,"hit_rate":(hits/n if n else np.nan)})
    return pd.DataFrame(rows).sort_values(["category","level"]).reset_index(drop=True)

# -------------------- CLI --------------------
def make_parser():
    p = argparse.ArgumentParser(description="Score polymer isobars from CSV; sweeps + normalized plots + ECDF.")
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV with peptide and hit columns.")
    p.add_argument("--pepcol", default="peptide", help="Peptide column name (default: peptide).")
    p.add_argument("--hitcol", default="is_hit", help="Hit column name (default: is_hit).")
    p.add_argument("--out", dest="outdir", default=None, help="Output directory (default auto-descriptive).")

    # Families + chemistry
    p.add_argument("--families", default="PEG,PPG,PTMEG,PDMS", help="Comma list of families to consider.")
    p.add_argument("--endgroups", default="auto", help="Comma list (e.g., diol,monoMe,diMe) or 'auto' per family.")
    p.add_argument("--adducts", default="H,Na,K,NH4", help="Comma list of allowed adduct monomers for base scoring.")
    p.add_argument("--peptide-charges", default="2,3", help="Comma list of peptide z (default: 2,3).")
    p.add_argument("--peg-charges", default="1,2,3", help="Comma list of PEG/polymer z (default: 1,2,3).")
    p.add_argument("--n-range", default="5,100", help="Oligomer repeat inclusive range (default 5,100).")
    p.add_argument("--mz-range", default="200,1500", help="m/z window for candidates (default 200,1500).")

    # Scoring and performance
    p.add_argument("--q1-fwhm", type=float, default=0.7, help="Q1 FWHM in Th (SRM score).")
    p.add_argument("--ppm-window", type=float, default=10.0, help="PPM window (HRAM-like score).")
    p.add_argument("--preselect", type=int, default=200, help="Preselect N nearest candidates per peptide charge.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--no-plots", action="store_true", help="Skip plotting.")

    # PPM bins and ECDF
    p.add_argument("--ppm-bins", default="5,10,15,20,30", help="ppm edges for summaries (e.g., 10,20,30).")
    p.add_argument("--max-ppm-plot", type=float, default=200.0, help="x-axis max for ECDF plot.")

    # Duplicate hit handling
    p.add_argument("--hit-conflict", default="error",
                   choices=["error","warn_prefer_input","warn_prefer_nonnull"],
                   help="Behavior when duplicate hit columns disagree.")

    # Ranking plot
    p.add_argument("--peg-score", default="ppm", choices=["ppm","q1","combo"],
                   help="Score used to rank hits in top-N plot (default: ppm).")
    p.add_argument("--top-hits", type=int, default=30, help="Top-N HITs to show (default: 30).")

    # SWEEPS
    p.add_argument("--do-sweep", action="store_true", help="Run parameter sweeps across families/configs.")
    p.add_argument("--sweep-adduct-sets", default="H;H,NH4;Na;Na,K;H,Na;H,Na,K",
                   help="Semicolon-separated adduct sets (e.g., 'H;H,NH4;Na;Na,K;H,Na;H,Na,K').")
    p.add_argument("--sweep-endgroup-sets", default="auto;diol;monoMe;diMe;diol,monoMe",
                   help="Semicolon-separated end-group sets (use 'auto' for family defaults).")
    p.add_argument("--sweep-peg-charge-sets", default="1;2;3;1,2;2,3;1,2,3",
                   help="Semicolon-separated polymer charge sets.")
    p.add_argument("--sweep-topK", type=int, default=12, help="Plots: show top-K configs per summary metric (default: 12).")

    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p

def main():
    global args  # used inside sweep for column names
    parser = make_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse lists/ranges
    args.families        = parse_csv_list_of_str(args.families)
    args.endgroups       = parse_csv_list_of_str(args.endgroups)
    args.adducts         = parse_csv_list_of_str(args.adducts)
    args.peptide_charges = parse_csv_list_of_ints(args.peptide_charges)
    args.peg_charges     = parse_csv_list_of_ints(args.peg_charges)
    args.n_range         = tuple(int(x) for x in args.n_range.split(","))
    args.mz_range        = tuple(float(x) for x in args.mz_range.split(","))
    args.ppm_bins        = [float(x) for x in args.ppm_bins.split(",") if x.strip()]
    validate_config(args)

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    # Outdir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        stem = in_path.stem
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(f"{stem}__polymer_sweep__q1-{args.q1_fwhm}_ppm-{args.ppm_window}_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    (outdir / "sweep_plots").mkdir(exist_ok=True)
    logger.info(f"Output directory: {outdir.resolve()}")
    logger.info(f"Python: {os.sys.version.split()[0]}, numpy: {np.__version__}, pandas: {pd.__version__}, matplotlib: {matplotlib.__version__}")

    # Load input
    df_in = pd.read_csv(in_path)
    logger.info(f"Loaded input rows: {len(df_in)}; columns: {list(df_in.columns)}")
    if args.pepcol not in df_in.columns:
        raise ValueError(f"Missing peptide column '{args.pepcol}'.")
    if args.hitcol not in df_in.columns:
        raise ValueError(f"Missing hit column '{args.hitcol}'.")

    # Coerce hits -> single column
    df_in["_is_hit_coerced"] = df_in[args.hitcol].apply(coerce_is_hit).astype("Int64")
    df_in = df_in.drop(columns=[args.hitcol], errors="ignore").rename(columns={"_is_hit_coerced": args.hitcol})

    # Validate peptide sequences
    df_in["_valid_pep"] = df_in[args.pepcol].apply(aa_only)
    invalid = df_in[~df_in["_valid_pep"]]
    if not invalid.empty:
        invalid.to_csv(outdir / "invalid_peptides.csv", index=False)
        logger.warning(f"Invalid peptides: {len(invalid)} (wrote invalid_peptides.csv)")

    peptides = sorted(set(df_in.loc[df_in["_valid_pep"], args.pepcol].astype(str)))
    logger.info(f"Unique valid peptides to score: {len(peptides)}")

    # Build base candidate set (for combined family scoring if no sweep)
    base_candidates = build_family_candidates(
        families=args.families, peg_charges=args.peg_charges,
        allowed_adducts=args.adducts, endgroups=args.endgroups,
        n_min=args.n_range[0], n_max=args.n_range[1],
        mz_min=args.mz_range[0], mz_max=args.mz_range[1],
        show_progress=not args.no_progress
    )

    # Best-per-peptide (ppm priority for closeness; also compute Q1 priority)
    df_best_ppm = best_isobars_for_peptides(
        peptides, peptide_charges=args.peptide_charges,
        candidates=base_candidates, q1_fwhm_da=args.q1_fwhm, ppm_window=args.ppm_window,
        score_priority="ppm", preselect=args.preselect, show_progress=not args.no_progress
    )
    df_best_ppm.to_csv(outdir / "best_per_peptide_ppm.csv", index=False)

    df_best_q1 = best_isobars_for_peptides(
        peptides, peptide_charges=args.peptide_charges,
        candidates=base_candidates, q1_fwhm_da=args.q1_fwhm, ppm_window=args.ppm_window,
        score_priority="q1", preselect=args.preselect, show_progress=not args.no_progress
    )
    df_best_q1.to_csv(outdir / "best_per_peptide_q1.csv", index=False)

    # Join back to input with hit column
    df_join = df_in.merge(df_best_ppm, how="left", left_on=args.pepcol, right_on="peptide", suffixes=("", "__best"))
    df_join = ensure_single_hit_col(df_join, hit_col=args.hitcol, outdir=outdir, behavior=args.hit_conflict)
    dj = add_derived_features(df_join)
    dj.to_csv(outdir / "input_joined_ppm.csv", index=False)

    # Basic normalized plots & summaries
    bins_df = compute_ppm_bin_fractions(dj, hit_col=args.hitcol, edges=args.ppm_bins)
    bins_df.to_csv(outdir / "ppm_bin_fractions_by_hit.csv", index=False)

    thr_df = threshold_summary(dj, hit_col=args.hitcol, thresholds=args.ppm_bins)
    thr_df.to_csv(outdir / "ppm_threshold_summary.csv", index=False)

    if not args.no_plots:
        plot_ppm_bin_fractions(bins_df, hit_col=args.hitcol, outdir=outdir / "plots", edges=args.ppm_bins)
        plot_ecdf_by_hit(dj, hit_col=args.hitcol, outdir=outdir / "plots", thresholds=args.ppm_bins, x_max=args.max_ppm_plot)

    # Rank hits by PEG-like score (choose source by args.peg_score)
    rank_source = df_best_ppm if args.peg_score in ("ppm","combo") else df_best_q1
    try:
        make_hits_sorted_by_score(df_in=df_in, df_best=rank_source, pepcol=args.pepcol,
                                  hit_col=args.hitcol, mode=args.peg_score, top_n=args.top_hits, outdir=outdir)
    except Exception as e:
        logger.warning(f"Hits-by-score plot failed: {e}")

    # -------------------- Parameter sweep (optional) --------------------
    if args.do_sweep:
        endgroup_sets = parse_semicolon_sets(args.sweep_endgroup_sets)
        adduct_sets   = parse_semicolon_sets(args.sweep_adduct_sets)
        peg_z_sets    = [tuple(int(x) for x in s.split(",")) for s in args.sweep_peg_charge_sets.split(";")]

        sweep = sweep_families_configs(
            df_in=df_in, peptides=peptides,
            peptide_charges=tuple(args.peptide_charges),
            families=tuple(args.families),
            endgroup_sets=endgroup_sets,
            adduct_sets=adduct_sets,
            peg_charge_sets=peg_z_sets,
            n_range=args.n_range, mz_range=args.mz_range,
            q1_fwhm=args.q1_fwhm, ppm_window=args.ppm_window,
            thresholds=args.ppm_bins, preselect=args.preselect,
            outdir=outdir, show_progress=not args.no_progress
        )
        sweep_path = outdir / "sweep_metrics.csv"
        sweep.to_csv(sweep_path, index=False)
        logger.info(f"Wrote sweep metrics: {sweep_path}")

        if not args.no_plots and not sweep.empty:
            # Summary bars: precision@10, recall@10, F1@10 by (family|config). Show top-K.
            t10 = sweep[sweep["threshold_ppm"] == min(args.ppm_bins)].copy()
            t10["config"] = (t10["family"] + " | eg=" + t10["endgroups"]
                             + " | ad=" + t10["adducts"] + " | z=" + t10["peg_charges"])

            plot_bar_metric(t10, "config", "precision", "Precision @ ≤{} ppm (top configs)".format(int(min(args.ppm_bins))),
                            outdir / "sweep_plots" / "precision_at_minppm_top.png", top_k=args.sweep_topK)
            plot_bar_metric(t10, "config", "F1", "F1 @ ≤{} ppm (top configs)".format(int(min(args.ppm_bins))),
                            outdir / "sweep_plots" / "F1_at_minppm_top.png", top_k=args.sweep_topK)
            plot_bar_metric(t10, "config", "nonhit_capture_frac", "Non-hit capture @ ≤{} ppm (lower is better)".format(int(min(args.ppm_bins))),
                            outdir / "sweep_plots" / "nonhit_capture_at_minppm_top.png", top_k=args.sweep_topK)

            # Best config per family by F1@minppm
            best_per_family = (t10.sort_values("F1", ascending=False)
                                  .groupby("family", as_index=False).first())
            best_per_family.to_csv(outdir / "sweep_best_per_family_at_minppm.csv", index=False)

            # Heatmap: for each family, precision@minppm over adduct_sets x peg_z_sets (endgroups collapsed to "auto")
            for fam in args.families:
                sub = t10[(t10["family"] == fam) & (t10["endgroups"].isin(["auto", "diol,monoMe,diMe"]))].copy()
                if sub.empty: continue
                piv = sub.pivot_table(index="adducts", columns="peg_charges", values="precision", aggfunc="max")
                plot_heatmap_metric(piv, f"{fam}: Precision @ ≤{int(min(args.ppm_bins))} ppm (max over endgroups)",
                                    outdir / "sweep_plots" / f"{fam}_precision_heatmap_minppm.png")

            # PR & Pareto curves for top-K configs by F1@minppm
            top_cfgs = t10.sort_values("F1", ascending=False).head(args.sweep_topK)
            for _, cfg in top_cfgs.iterrows():
                fam, egs, ads, zset = cfg["family"], cfg["endgroups"], cfg["adducts"], cfg["peg_charges"]
                # Filter sweep entries for this config across all thresholds
                thr_curve = sweep[(sweep["family"]==fam) &
                                  (sweep["endgroups"]==egs) &
                                  (sweep["adducts"]==ads) &
                                  (sweep["peg_charges"]==zset)].sort_values("threshold_ppm")
                plot_pr_curve(thr_curve, f"{fam} [{egs}] [{ads}] z={zset}",
                              outdir / "sweep_plots" / f"PR_{fam}_{egs}_{ads}_z{zset}.png")
                plot_pareto(thr_curve, f"{fam} [{egs}] [{ads}] z={zset}",
                            outdir / "sweep_plots" / f"Pareto_{fam}_{egs}_{ads}_z{zset}.png")

    # Final diagnostics
    total_rows = len(df_join)
    n_hits = int(pd.to_numeric(df_join[args.hitcol], errors="coerce").fillna(0).astype(int).sum())
    logger.info(f"Joined rows: {total_rows}, hits={n_hits} ({(n_hits/total_rows*100 if total_rows else 0):.1f}%)")
    logger.info("Done.")

if __name__ == "__main__":
    main()

