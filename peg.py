#!/usr/bin/env python3
"""
PEG–isobar scoring CLI with diagnostics, tqdm progress, normalized ppm-bin plots,
ECDF, and robust duplicate hit-column handling with agreement checks.

Example:
    python peg_isobar_cli.py \
        --in my_peptides.csv \
        --pepcol peptide \
        --hitcol is_hit

Key outputs:
- best_per_peptide_q1.csv           (SRM/Q1-priority selection)
- best_per_peptide_ppm.csv          (HRAM/ppm-priority selection)
- input_joined_q1.csv               (input joined with Q1 selection)
- ppm_bin_fractions_by_hit.csv      (normalized fractions in ppm bins by hit group)
- ppm_threshold_summary.csv         (≤10/20/30 ppm summaries, OR/RR/CIs/p-values)
- plots/
    - ecdf_abs_delta_ppm_by_is_hit.png     <-- THE ONE PLOT to inspect
    - frac_bins_abs_delta_ppm_by_is_hit.png
    - (other diagnostic histograms & bars)

"""

from __future__ import annotations
import argparse
import logging
import math
import os
from collections import Counter
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

# Use non-interactive backend for headless/CI
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

# PEG repeat and end-groups (neutral masses)
EO = 44.026214747  # exact monoisotopic mass of C2H4O
PEG_ENDGROUPS: Dict[str, float] = {
    "diol":   18.010564684,   # HO-(CH2CH2O)n-H
    "monoMe": 32.026214748,   # CH3O-(CH2CH2O)n-H
    "diMe":   46.041864812,   # CH3O-(CH2CH2O)n-OCH3
}

# Positive-mode cation adducts (mass contributions)
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

def polymer_mass(endgroup: str, n: int) -> float:
    return PEG_ENDGROUPS[endgroup] + n * EO

def polymer_mz(endgroup: str, n: int, adducts: Tuple[str, ...]) -> Tuple[int, float]:
    z = len(adducts)
    return z, (polymer_mass(endgroup, n) + sum(ADDUCTS[a] for a in adducts)) / z

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
    if len(args.peg_charges) == 0 or any(z <= 0 for z in args.peg_charges):
        raise ValueError("--peg-charges must be positive integers")
    if args.n_range[0] > args.n_range[1]:
        raise ValueError("--n-range start must be <= end")
    if args.mz_range[0] >= args.mz_range[1]:
        raise ValueError("--mz-range min must be < max")
    for e in args.endgroups:
        if e not in PEG_ENDGROUPS:
            raise ValueError(f"--endgroups includes unknown '{e}'. Valid: {list(PEG_ENDGROUPS)}")
    for a in args.adducts:
        if a not in ADDUCTS:
            raise ValueError(f"--adducts includes unknown '{a}'. Valid: {list(ADDUCTS)}")

def find_duplicate_named_columns(df: pd.DataFrame, name: str) -> List[str]:
    return [c for c in df.columns if c == name]

def ensure_single_hit_col(df: pd.DataFrame, hit_col: str, outdir: Path, behavior: str = "error") -> pd.DataFrame:
    """
    If multiple columns named hit_col exist, verify agreement (rows where *both* non-NA).
    On disagreement:
      - 'error' (default): raise and write rows to hit_disagreements.csv
      - 'warn_prefer_input': warn and prefer the first column
      - 'warn_prefer_nonnull': warn and take rowwise first non-null
    Returns df with a single consolidated hit_col (Int64 {0,1,NA}).
    """
    dup_names = find_duplicate_named_columns(df, hit_col)
    if len(dup_names) <= 1:
        # Normalize dtype
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
        logger.info(f"[hit-col] duplicates={dup_names}, rows with both non-null={n_valid}, disagreements={n_disagree}")
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

    if behavior == "warn_prefer_nonnull":
        cons = dup_df.bfill(axis=1).iloc[:, 0]
    else:
        cons = dup_df.iloc[:, 0]

    df_out = df.drop(columns=dup_names)
    df_out[hit_col] = cons.astype("Int64")
    return df_out

# -------------------- Candidate enumeration & scoring --------------------
def build_polymer_candidates(
    peg_charges: Iterable[int],
    allowed_adducts: Iterable[str],
    allowed_endgroups: Iterable[str],
    n_min: int, n_max: int,
    mz_min: float, mz_max: float,
    show_progress: bool = True,
) -> List[Tuple[str, int, Tuple[str, ...], int, float]]:
    endgroups = list(allowed_endgroups)
    adducts_list = list(allowed_adducts)
    candidates = []

    # Rough count for tqdm
    k = len(adducts_list)
    total_combos = 0
    for z in peg_charges:
        total_combos += math.comb(k + z - 1, z)
    total_iters = len(endgroups) * (n_max - n_min + 1) * total_combos

    pbar = tqdm(total=total_iters, disable=not show_progress, desc="Enumerating PEG candidates")
    for end in endgroups:
        for n in range(n_min, n_max + 1):
            for z in peg_charges:
                for ads in _adduct_multisets(z, allowed=adducts_list):
                    peg_z, mz = polymer_mz(end, n, ads)
                    if mz_min <= mz <= mz_max:
                        candidates.append((end, n, ads, peg_z, mz))
                    pbar.update(1)
    pbar.close()

    logger.info(f"[candidates] kept={len(candidates)} within m/z [{mz_min}, {mz_max}]")
    if len(candidates) == 0:
        raise RuntimeError("No PEG candidates were generated within the specified ranges.")

    # Quick summaries
    dfc = pd.DataFrame({
        "end": [c[0] for c in candidates],
        "n":   [c[1] for c in candidates],
        "z":   [c[3] for c in candidates],
        "mz":  [c[4] for c in candidates],
        "adducts": ["+".join(c[2]) for c in candidates],
    })
    logger.info("[candidates] by end-group:\n" + dfc["end"].value_counts().to_string())
    logger.info("[candidates] by PEG charge:\n" + dfc["z"].value_counts().sort_index().to_string())
    logger.info(f"[candidates] m/z range: {dfc['mz'].min():.4f} .. {dfc['mz'].max():.4f}")
    return candidates

def best_peg_isobar_for_peptide(
    seq: str,
    candidates: List[Tuple[str, int, Tuple[str, ...], int, float]],
    cand_mz: np.ndarray,
    peptide_charges: Iterable[int],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str,
    preselect: int = 200,
) -> Dict[str, object]:
    assert score_priority in ("q1", "ppm")
    rows: List[Dict[str, object]] = []

    for pep_z in peptide_charges:
        pep_mz = peptide_mz(seq, pep_z)
        idx = np.argsort(np.abs(cand_mz - pep_mz))[:preselect]
        for j in idx:
            end, n, ads, peg_z, peg_mz = candidates[j]
            d_mz = peg_mz - pep_mz
            d_ppm = (d_mz / pep_mz) * 1e6
            score_q1 = gaussian_score_delta(d_mz, q1_fwhm_da)
            score_ppm = gaussian_score_delta(d_ppm, ppm_window)
            score_used = score_q1 if score_priority == "q1" else score_ppm
            rows.append({
                "peptide": seq,
                "pep_z": pep_z,
                "pep_mz": pep_mz,
                "peg_z": peg_z,
                "peg_endgroup": end,
                "peg_n": n,
                "peg_adducts": "+".join(ads),
                "peg_mz": peg_mz,
                "delta_mDa": 1000.0 * d_mz,
                "delta_ppm": d_ppm,
                "score_q1": score_q1,
                "score_ppm": score_ppm,
                "score_used": score_used,
                "score_priority": score_priority,
            })

    rows.sort(key=lambda r: (-(r["score_used"] if r["score_used"] is not None else 0), abs(r["delta_mDa"])))
    best = rows[0]
    for k, nd in [("pep_mz",6),("peg_mz",6),("delta_mDa",3),("delta_ppm",2),("score_q1",4),("score_ppm",4),("score_used",4)]:
        best[k] = round(best[k], nd)
    if abs(best["delta_ppm"]) > 50:
        best["warning"] = f"Large ppm deviation ({best['delta_ppm']:+.1f} ppm)"
    return best

def best_peg_isobars_for_peptides(
    peptides: List[str],
    peptide_charges: Iterable[int],
    candidates: List[Tuple[str, int, Tuple[str, ...], int, float]],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str,
    preselect: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    cand_mz = np.array([c[4] for c in candidates], dtype=float)
    rows = []
    pbar = tqdm(peptides, disable=not show_progress, desc="Scoring peptides")
    for seq in pbar:
        rows.append(
            best_peg_isobar_for_peptide(
                seq,
                candidates=candidates,
                cand_mz=cand_mz,
                peptide_charges=peptide_charges,
                q1_fwhm_da=q1_fwhm_da,
                ppm_window=ppm_window,
                score_priority=score_priority,
                preselect=preselect,
            )
        )
    df = pd.DataFrame(rows)
    cols = ["peptide","pep_z","pep_mz","peg_z","peg_endgroup","peg_n","peg_adducts","peg_mz",
            "delta_mDa","delta_ppm","score_q1","score_ppm","score_used","score_priority","warning"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols]

# -------------------- Derived features & stats --------------------
def adduct_feature_counts(adduct_str: str) -> Dict[str, int]:
    parts = [p for p in str(adduct_str).split("+") if p]
    c = Counter(parts)
    return {"cnt_H": c.get("H", 0), "cnt_Na": c.get("Na", 0), "cnt_K": c.get("K", 0), "cnt_NH4": c.get("NH4", 0)}

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
    y = np.asarray(y).reshape(-1)
    x = np.asarray(x).reshape(-1)
    n = min(len(y), len(x))
    if n == 0: return 0.0
    y = y[:n]; x = x[:n]
    if np.nanstd(y) == 0 or np.nanstd(x) == 0: return 0.0
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2: return 0.0
    return float(np.corrcoef(y[mask], x[mask])[0,1])

def compute_effects_table(df_join: pd.DataFrame, hit_col: str) -> pd.DataFrame:
    work = add_derived_features(df_join)
    work[hit_col] = work[hit_col].apply(coerce_is_hit)
    work = work.dropna(subset=[hit_col])
    work[hit_col] = work[hit_col].astype(int)

    effects = []
    numeric_feats = ["pep_mz","peg_mz","peg_n","abs_delta_ppm","abs_delta_mDa","pep_z","peg_z",
                     "adduct_count","cnt_H","cnt_Na","cnt_K","cnt_NH4","has_H","has_Na","has_K","has_NH4"]
    for f in numeric_feats:
        if f in work.columns:
            xy = work[[hit_col,f]].dropna()
            if not xy.empty:
                r = point_biserial_corr(xy[hit_col].to_numpy(), xy[f].to_numpy(float))
                effects.append({"feature": f, "kind": "numeric", "metric": "corr", "value": r})

    if "peg_endgroup" in work.columns:
        d = pd.get_dummies(work["peg_endgroup"], prefix="end", drop_first=False)
        d = d.loc[work.index]
        mask = work[hit_col].notna()
        ycat = work.loc[mask, hit_col].to_numpy()
        dcat = d.loc[mask]
        for col in dcat.columns:
            r = point_biserial_corr(ycat, dcat[col].to_numpy(float))
            effects.append({"feature": col, "kind": "categorical(end)", "metric": "corr_dummy", "value": r})

    eff = pd.DataFrame(effects)
    if eff.empty: return eff
    eff["abs_value"] = eff["value"].abs()
    return eff.sort_values("abs_value", ascending=False).reset_index(drop=True)

# -------------------- NEW: ppm bins, ECDF, and stats --------------------
def ppm_bin_labels(edges: List[float]) -> List[str]:
    labels = []
    prev = 0.0
    for e in edges:
        labels.append(f"≤{int(e)}")
        prev = e
    labels.append(f">{int(edges[-1])}")
    return labels

def compute_ppm_bin_fractions(df: pd.DataFrame, hit_col: str, edges: List[float]) -> pd.DataFrame:
    """
    Bin |Δppm| into [0,e1], (e1,e2], ... , (elast, inf). Return counts and *fractions per hit group*.
    """
    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()
    bins = [-np.inf] + edges + [np.inf]
    sub["ppm_bin"] = pd.cut(sub["abs_delta_ppm"], bins=bins, labels=ppm_bin_labels(edges), include_lowest=True)

    # counts per (hit, bin)
    tab = sub.groupby([hit_col, "ppm_bin"]).size().rename("count").reset_index()
    totals = sub.groupby(hit_col).size().rename("group_total").reset_index()
    out = tab.merge(totals, on=hit_col, how="left")
    out["fraction"] = out["count"] / out["group_total"]
    # ensure all bins present per group
    all_bins = out["ppm_bin"].cat.categories
    groups = out[hit_col].unique()
    out = out.set_index([hit_col, "ppm_bin"]).reindex(
        pd.MultiIndex.from_product([groups, all_bins]), fill_value=0
    ).reset_index()
    # recompute fraction & totals after reindex fill
    out = out.merge(totals, on=hit_col, how="left")
    out["fraction"] = out["count"] / out["group_total"].replace(0, np.nan)
    return out

def threshold_summary(df: pd.DataFrame, hit_col: str, thresholds: List[float]) -> pd.DataFrame:
    """
    For each threshold T, consider B = {abs_delta_ppm ≤ T}. Compute:
      n_hit_B, n_hit_notB, n_nonhit_B, n_nonhit_notB,
      p_hit_B, p_nonhit_B, risk_ratio, odds_ratio (with 95% CI, Haldane correction),
      z-test for p_hit_B vs p_nonhit_B (two-proportion).
    """
    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()
    rows = []
    # totals
    n_hit = int(sub[hit_col].sum())
    n_nonhit = int((1 - sub[hit_col]).sum())
    for T in thresholds:
        inB = (sub["abs_delta_ppm"] <= T)
        a = int(((sub[hit_col] == 1) & inB).sum())
        b = n_hit - a
        c = int(((sub[hit_col] == 0) & inB).sum())
        d = n_nonhit - c

        # Fractions
        p1 = a / n_hit if n_hit else np.nan
        p2 = c / n_nonhit if n_nonhit else np.nan

        # Risk ratio
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan

        # Odds ratio with Haldane–Anscombe correction
        a1,b1,c1,d1 = a+0.5, b+0.5, c+0.5, d+0.5
        or_ = (a1*d1)/(b1*c1)
        se_log_or = math.sqrt(1/a1 + 1/b1 + 1/c1 + 1/d1)
        ci_lo = math.exp(math.log(or_) - 1.96*se_log_or)
        ci_hi = math.exp(math.log(or_) + 1.96*se_log_or)

        # Two-proportion z-test (pooled)
        n1, n2 = n_hit, n_nonhit
        p_pool = (a + c) / (n1 + n2) if (n1+n2) else np.nan
        denom = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)) if (p_pool==p_pool and n1 and n2 and p_pool not in (0,1)) else np.nan
        z = (p1 - p2) / denom if (denom and denom==denom and denom>0) else np.nan
        # two-sided p-value via normal approx
        if z == z:
            from math import erf, sqrt
            pval = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
        else:
            pval = np.nan

        rows.append({
            "threshold_ppm": T,
            "n_hit_total": n_hit, "n_nonhit_total": n_nonhit,
            "n_hit_leT": a, "n_hit_gtT": b, "n_nonhit_leT": c, "n_nonhit_gtT": d,
            "frac_hit_leT": p1, "frac_nonhit_leT": p2,
            "risk_ratio": rr,
            "odds_ratio": or_, "or_95ci_lo": ci_lo, "or_95ci_hi": ci_hi,
            "z_value": z, "p_value_two_sided": pval,
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
    for T in thresholds:
        plt.axvline(T, linestyle="--", linewidth=1)
    plt.xlim(0, x_max)
    plt.ylim(0, 1)
    plt.xlabel("|Δppm|")
    plt.ylabel("cumulative fraction ≤ x (within group)")
    plt.title("ECDF of |Δppm| by hit status")
    plt.legend()
    save_plot(fig, outdir / "ecdf_abs_delta_ppm_by_is_hit.png")

def plot_ppm_bin_fractions(df_bins: pd.DataFrame, hit_col: str, outdir: Path):
    """
    Grouped bars: for each ppm bin (x), show fraction within group (y) for hit=0 and hit=1.
    Fractions sum to 1 per hit group (normalized).
    """
    if df_bins.empty: return
    # Ensure consistent ordering of bins
    bin_order = list(df_bins["ppm_bin"].cat.categories)
    fig = plt.figure(figsize=(max(6, 1.1*len(bin_order)), 4.5))
    groups = sorted(df_bins[hit_col].dropna().unique())
    width = 0.35 if len(groups) == 2 else 0.25
    x = np.arange(len(bin_order))
    for i, g in enumerate(groups):
        sub = df_bins[df_bins[hit_col] == g].set_index("ppm_bin").reindex(bin_order)
        y = sub["fraction"].to_numpy(float)
        plt.bar(x + (i - (len(groups)-1)/2)*width, y, width=width, label=f"{hit_col}={int(g)}")
    plt.xticks(x, bin_order)
    plt.ylim(0, 1)
    plt.xlabel("abs(Δppm) bins")
    plt.ylabel("fraction within group")
    plt.title("Fraction in |Δppm| bins by hit status (normalized)")
    plt.legend()
    save_plot(fig, outdir / "frac_bins_abs_delta_ppm_by_is_hit.png")

# -------------------- CLI --------------------
def make_parser():
    p = argparse.ArgumentParser(description="Score PEG isobars from CSV; normalized ppm-bin plots and ECDF included.")
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV with peptide and hit columns.")
    p.add_argument("--pepcol", default="peptide", help="Peptide column name (default: peptide).")
    p.add_argument("--hitcol", default="is_hit", help="Hit column name (default: is_hit).")
    p.add_argument("--out", dest="outdir", default=None, help="Output directory (default: auto-descriptive).")

    # Scoring knobs
    p.add_argument("--peptide-charges", default="2,3", help="Comma list of peptide z (default: 2,3).")
    p.add_argument("--peg-charges", default="1,2,3", help="Comma list of PEG z (default: 1,2,3).")
    p.add_argument("--endgroups", default="diol,monoMe,diMe", help="Comma list of PEG end-groups.")
    p.add_argument("--adducts", default="H,Na,K,NH4", help="Comma list of allowed adducts.")
    p.add_argument("--n-range", default="5,100", help="EO repeat inclusive range, e.g., 5,100.")
    p.add_argument("--mz-range", default="200,1500", help="m/z bounds for candidate enumeration.")
    p.add_argument("--q1-fwhm", type=float, default=0.7, help="Q1 FWHM in Th (SRM score).")
    p.add_argument("--ppm-window", type=float, default=10.0, help="PPM window (HRAM-like score).")
    p.add_argument("--preselect", type=int, default=200, help="Preselect N closest candidates / peptide charge.")

    # Plots and normalization
    p.add_argument("--ppm-bins", default="10,20,30", help="ppm bin edges (comma), e.g., 10,20,30.")
    p.add_argument("--max-ppm-plot", type=float, default=200.0, help="x-axis max for ECDF plot.")
    p.add_argument("--no-plots", action="store_true", help="Skip plotting.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")

    # Duplicate hit handling
    p.add_argument("--hit-conflict", default="error",
                   choices=["error", "warn_prefer_input", "warn_prefer_nonnull"],
                   help="Action when duplicate hit columns disagree (see README).")

    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p

def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse lists/ranges
    args.peptide_charges = parse_csv_list_of_ints(args.peptide_charges)
    args.peg_charges     = parse_csv_list_of_ints(args.peg_charges)
    args.endgroups       = parse_csv_list_of_str(args.endgroups)
    args.adducts         = parse_csv_list_of_str(args.adducts)
    args.n_range         = tuple(int(x) for x in args.n_range.split(","))
    args.mz_range        = tuple(float(x) for x in args.mz_range.split(","))
    args.ppm_bins        = [float(x) for x in args.ppm_bins.split(",") if x.strip()]

    validate_config(args)

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    # Default descriptive outdir if not provided
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        stem = in_path.stem
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(f"{stem}__peg_scoring__q1-{args.q1_fwhm}_ppm-{args.ppm_window}_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    logger.info(f"Output directory: {outdir.resolve()}")

    logger.info(f"Python: {os.sys.version.split()[0]}, numpy: {np.__version__}, pandas: {pd.__version__}, matplotlib: {matplotlib.__version__}")

    # ---- Load input
    df_in = pd.read_csv(in_path)
    logger.info(f"Loaded input rows: {len(df_in)}; columns: {list(df_in.columns)}")

    if args.pepcol not in df_in.columns:
        raise ValueError(f"Missing peptide column '{args.pepcol}' in input.")
    if args.hitcol not in df_in.columns:
        raise ValueError(f"Missing hit column '{args.hitcol}' in input.")

    # Duplicate column names?
    dup_counts = pd.Series(df_in.columns, dtype=str).value_counts()
    dups = dup_counts[dup_counts > 1]
    if not dups.empty:
        logger.warning(f"Duplicate column names detected: {dups.to_dict()}")

    # Coerce hit to {0,1} and replace original
    df_in["_is_hit_coerced"] = df_in[args.hitcol].apply(coerce_is_hit).astype("Int64")
    df_in = df_in.drop(columns=[args.hitcol], errors="ignore").rename(columns={"_is_hit_coerced": args.hitcol})

    # Validate sequences; write invalids
    df_in["_valid_pep"] = df_in[args.pepcol].apply(aa_only)
    invalid = df_in[~df_in["_valid_pep"]]
    if len(invalid) > 0:
        invalid_path = outdir / "invalid_peptides.csv"
        invalid.to_csv(invalid_path, index=False)
        logger.warning(f"Invalid peptide sequences: {len(invalid)} rows. Wrote to {invalid_path}")

    # Unique peptides to score
    peptides = sorted(set(df_in.loc[df_in["_valid_pep"], args.pepcol].astype(str)))
    logger.info(f"Unique valid peptides to score: {len(peptides)}")

    # ---- Build PEG candidates
    candidates = build_polymer_candidates(
        peg_charges=args.peg_charges,
        allowed_adducts=args.adducts,
        allowed_endgroups=args.endgroups,
        n_min=args.n_range[0], n_max=args.n_range[1],
        mz_min=args.mz_range[0], mz_max=args.mz_range[1],
        show_progress=not args.no_progress,
    )

    # ---- Best-per-peptide (Q1 priority)
    df_q1 = best_peg_isobars_for_peptides(
        peptides,
        peptide_charges=args.peptide_charges,
        candidates=candidates,
        q1_fwhm_da=args.q1_fwhm,
        ppm_window=args.ppm_window,
        score_priority="q1",
        preselect=args.preselect,
        show_progress=not args.no_progress,
    )
    df_q1.to_csv(outdir / "best_per_peptide_q1.csv", index=False)
    logger.info(f"Wrote: {outdir / 'best_per_peptide_q1.csv'}")

    # ---- Best-per-peptide (ppm priority)
    df_ppm = best_peg_isobars_for_peptides(
        peptides,
        peptide_charges=args.peptide_charges,
        candidates=candidates,
        q1_fwhm_da=args.q1_fwhm,
        ppm_window=args.ppm_window,
        score_priority="ppm",
        preselect=args.preselect,
        show_progress=not args.no_progress,
    )
    df_ppm.to_csv(outdir / "best_per_peptide_ppm.csv", index=False)
    logger.info(f"Wrote: {outdir / 'best_per_peptide_ppm.csv'}")

    # ---- Join Q1 selection back to input and enforce single hit col with agreement check
    df_join = df_in.merge(df_q1, how="left", left_on=args.pepcol, right_on="peptide", suffixes=("", "__dfq1"))
    df_join = ensure_single_hit_col(df_join, hit_col=args.hitcol, outdir=outdir, behavior=args.hit_conflict)
    join_path = outdir / "input_joined_q1.csv"
    df_join.to_csv(join_path, index=False)
    logger.info(f"Wrote: {join_path}")

    # ---- Derived features (abs_delta_ppm, adduct features) for plots/stats
    dj = add_derived_features(df_join)

    # ---- Normalized ppm-bin fractions and threshold summaries
    bins_df = compute_ppm_bin_fractions(dj, hit_col=args.hitcol, edges=args.ppm_bins)
    bins_df.to_csv(outdir / "ppm_bin_fractions_by_hit.csv", index=False)
    logger.info(f"Wrote: {outdir / 'ppm_bin_fractions_by_hit.csv'}")

    thr_df = threshold_summary(dj, hit_col=args.hitcol, thresholds=args.ppm_bins)
    thr_df.to_csv(outdir / "ppm_threshold_summary.csv", index=False)
    logger.info(f"Wrote: {outdir / 'ppm_threshold_summary.csv'}")

    # ---- Plots
    if not args.no_plots:
        plots_dir = outdir / "plots"
        try:
            plot_ppm_bin_fractions(bins_df, hit_col=args.hitcol, outdir=plots_dir)
        except Exception as e:
            logger.warning(f"Normalized ppm-bin fraction plot failed: {e}")
        try:
            plot_ecdf_by_hit(dj, hit_col=args.hitcol, outdir=plots_dir, thresholds=args.ppm_bins, x_max=args.max_ppm_plot)
        except Exception as e:
            logger.warning(f"ECDF plot failed: {e}")

    # ---- Final diagnostics
    total_rows = len(df_join)
    n_hits = int(pd.to_numeric(df_join[args.hitcol], errors="coerce").fillna(0).astype(int).sum())
    logger.info(f"Joined rows: {total_rows}, hits={n_hits} ({(n_hits/total_rows*100 if total_rows else 0):.1f}%)")
    logger.info("Done.")

if __name__ == "__main__":
    main()

