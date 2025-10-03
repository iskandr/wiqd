# polymer_helpers.py
from __future__ import annotations

import logging
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

# Third-party
import numpy as np
import pandas as pd

# Non-interactive backend (headless safe)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Helper APIs & constants (external dependency)
from wiqd_peptide_similarity import (
    pep_mass,
    mz_from_mass,
    enumerate_polymer_series,
    best_polymer_match_for_reference,
    MONO,  # residue masses
    ADDUCT_MASS,  # {'H':..., 'Na':..., 'K':..., 'NH4':...}
    POLYMER_REPEAT_MASS,
    POLYMER_ENDGROUP,
)

logger = logging.getLogger("polymer.helpers")


# -------------------- Gaussian utility --------------------
def gaussian_score_delta(delta: float, fwhm: Optional[float]) -> Optional[float]:
    """Return exp(-0.5*(Δ/σ)^2) with σ=FWHM/2.355; None if fwhm invalid."""
    if fwhm is None or fwhm <= 0:
        return None
    sigma = fwhm / 2.35482004503
    try:
        return float(math.exp(-0.5 * (delta / sigma) ** 2))
    except OverflowError:
        return 0.0


# -------------------- Parsers & sanity --------------------
def parse_csv_list_of_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_csv_list_of_str(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def parse_semicolon_sets(s: str) -> List[Tuple[str, ...]]:
    """Parse 'A;A,B;B,C,D' -> list of tuples."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(tuple(x.strip() for x in part.split(",") if x.strip()))
    return out


def aa_only(seq: str) -> bool:
    return isinstance(seq, str) and len(seq) > 0 and all(c in MONO for c in seq)


def coerce_is_hit(x) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer, float, np.floating)):
        if pd.isna(x):
            return None
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

    valid_fams = set(POLYMER_REPEAT_MASS.keys())
    for fam in args.families:
        if fam not in valid_fams:
            raise ValueError(
                f"--families includes unknown '{fam}'. Valid: {sorted(valid_fams)}"
            )
    for fam in args.families:
        if "auto" in args.endgroups:
            continue
        valid_eg = set(POLYMER_ENDGROUP.get(fam, {}).keys())
        for e in args.endgroups:
            if e not in valid_eg:
                raise ValueError(
                    f"--endgroups includes '{e}' not valid for {fam}. "
                    f"Valid for {fam}: {sorted(valid_eg)}"
                )
    valid_adducts = set(ADDUCT_MASS.keys())
    for a in args.adducts:
        if a not in valid_adducts:
            raise ValueError(
                f"--adducts includes unknown '{a}'. Valid: {sorted(valid_adducts)}"
            )


def ensure_single_hit_col(
    df: pd.DataFrame, hit_col: str, outdir: Path, behavior: str = "error"
) -> pd.DataFrame:
    """If multiple columns named hit_col exist, verify agreement and collapse to one."""
    dup_names = [c for c in df.columns if c == hit_col]
    if len(dup_names) <= 1:
        df[hit_col] = df[hit_col].apply(coerce_is_hit).astype("Int64")
        return df
    dup_df = df.loc[:, dup_names].copy()
    for c in dup_df.columns:
        dup_df[c] = dup_df[c].apply(coerce_is_hit).astype("Int64")
    mask_all_valid = dup_df.notna().all(axis=1)
    agrees = dup_df.loc[mask_all_valid].nunique(axis=1) == 1
    n_valid = int(mask_all_valid.sum())
    n_disagree = int((~agrees).sum())
    if n_valid > 0:
        logger.info(
            f"[hit-col] duplicates={dup_names}, rows both non-null={n_valid}, disagreements={n_disagree}"
        )
    if n_disagree > 0:
        out = pd.concat([df, dup_df.add_prefix("DUP_")], axis=1).loc[
            mask_all_valid & (~agrees)
        ]
        path = outdir / "hit_disagreements.csv"
        out.to_csv(path, index=False)
        msg = f"Duplicate hit columns disagree on {n_disagree}/{n_valid} rows (see {path})."
        if behavior == "error":
            raise ValueError(msg)
        elif behavior not in ("warn_prefer_input", "warn_prefer_nonnull"):
            raise ValueError(f"Unknown --hit-conflict behavior '{behavior}'")
        else:
            logger.warning(msg + f" Proceeding with '{behavior}'.")
    cons = (
        dup_df.bfill(axis=1).iloc[:, 0]
        if behavior == "warn_prefer_nonnull"
        else dup_df.iloc[:, 0]
    )
    df_out = df.drop(columns=dup_names)
    df_out[hit_col] = cons.astype("Int64")
    return df_out


# -------------------- Candidate enumeration (via helper) --------------------
def build_family_candidates(
    families: Iterable[str],
    peg_charges: Iterable[int],
    allowed_adducts: Iterable[str],
    endgroups: Iterable[str],  # may include "auto"
    n_min: int,
    n_max: int,
    mz_min: float,
    mz_max: float,
    show_progress: bool = True,
) -> List[Dict]:
    """
    Wraps `enumerate_polymer_series` from the helper.

    Returns list[dict]:
      {'family','endgroup','n','adducts','z','mz'}
    """
    cands = enumerate_polymer_series(
        families=tuple(families),
        endgroups=tuple(endgroups),
        adducts=tuple(allowed_adducts),
        zset=tuple(peg_charges),
        n_min=int(n_min),
        n_max=int(n_max),
        mz_min=float(mz_min),
        mz_max=float(mz_max),
    )
    logger.info(f"[candidates] kept={len(cands)} in m/z [{mz_min}, {mz_max}]")
    if not cands:
        raise RuntimeError("No candidates generated. Check ranges/filters.")
    return cands


# -------------------- Top-K & best-per-peptide scoring -----------------------
def _ppm_from_delta_da(delta_da: float, pep_mz: float) -> float:
    return (delta_da / pep_mz) * 1e6 if pep_mz > 0 else np.nan


def top_polymer_matches_for_peptide(
    seq: str,
    candidates: List[Dict],
    peptide_charges: Iterable[int],
    q1_fwhm_da: float,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
    top_n: int,
) -> List[Dict]:
    """
    Use helper to get Top-K polymer matches for a peptide. Returns list of dicts:
      {'family','endgroup','n','adducts','z','mz','weight','delta_da'}
    where 'weight' is the Q1 Gaussian overlap averaged across modification scenarios.
    """
    res = best_polymer_match_for_reference(
        seq,
        candidates=candidates,
        peptide_z=tuple(peptide_charges),
        q1_fwhm=("da", float(q1_fwhm_da)),
        q1_tol=None,
        top_n=int(top_n),
        allow_heavy_first_residue=bool(allow_heavy_first_residue),
        allow_mtraq=bool(allow_mtraq),
    )
    return list(res.get("top") or [])


def best_isobar_for_peptide(
    seq: str,
    candidates: List[Dict],
    peptide_charges: Iterable[int],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str = "ppm",  # or 'q1'
    preselect: int = 200,  # Top-K from helper
    q3_fwhm: Optional[Tuple[str, float]] = None,  # e.g., ("ppm", 10.0) or ("da", 0.1)
    allow_heavy_first_residue: bool = False,
    allow_mtraq: bool = False,
) -> Dict[str, object]:
    """
    Return ONE best row across candidate set for this peptide.
    We fetch Top-K by Q1 weight from the helper and then (optionally) pick the
    row with highest PPM-Gaussian within those Top-K if score_priority='ppm'.
    """
    tops = top_polymer_matches_for_peptide(
        seq,
        candidates=candidates,
        peptide_charges=peptide_charges,
        q1_fwhm_da=q1_fwhm_da,
        allow_heavy_first_residue=allow_heavy_first_residue,
        allow_mtraq=allow_mtraq,
        top_n=preselect,
    )
    base = {
        "peptide": seq,
        "pep_z": np.nan,
        "pep_mz": np.nan,
        "family": None,
        "peg_z": np.nan,
        "peg_endgroup": None,
        "peg_n": np.nan,
        "peg_adducts": None,
        "peg_mz": np.nan,
        "delta_mDa": np.nan,
        "delta_ppm": np.nan,
        "score_q1": np.nan,
        "score_ppm": np.nan,
        "score_used": np.nan,
        "warning": "No candidate within windows" if not tops else None,
    }
    if not tops:
        return base

    # Compute ppm scores inside Top-K and choose according to priority
    chosen = None
    best_metric = -np.inf
    for c in tops:
        z = int(c["z"])
        pep_mz_val = mz_from_mass(
            pep_mass(seq), z
        )  # normalized by unmodified peptide m/z
        d_da = float(c["delta_da"])
        d_ppm = _ppm_from_delta_da(d_da, pep_mz_val)
        score_q1 = float(c["weight"])
        # in polymer_helpers.best_isobar_for_peptide signature:
        # add: q3_fwhm: Optional[Tuple[str, float]] = None  # e.g., ("ppm", 10.0) or ("da", 0.1)

        # inside the loop where score_ppm is computed:
        if q3_fwhm is None:
            score_ppm = gaussian_score_delta(d_ppm, ppm_window)  # current behavior
        else:
            units, w = q3_fwhm
            if units == "ppm":
                score_ppm = gaussian_score_delta(d_ppm, w)
            elif units == "da":
                score_ppm = gaussian_score_delta(abs(d_da), w)
            else:
                raise ValueError("q3_fwhm units must be 'ppm' or 'da'")

        metric = score_ppm if score_priority == "ppm" else score_q1
        if metric is not None and float(metric) > best_metric:
            best_metric = float(metric)
            chosen = (c, z, pep_mz_val, d_da, d_ppm, score_q1, score_ppm)

    if chosen is None:
        return base

    c, z, pep_mz_val, d_da, d_ppm, score_q1, score_ppm = chosen
    row = {
        **base,
        "pep_z": z,
        "pep_mz": round(pep_mz_val, 6),
        "family": c["family"],
        "peg_z": z,
        "peg_endgroup": c["endgroup"],
        "peg_n": int(c["n"]),
        "peg_adducts": c["adducts"],
        "peg_mz": round(float(c["mz"]), 6),
        "delta_mDa": round(1000.0 * d_da, 3),
        "delta_ppm": round(d_ppm, 2),
        "score_q1": round(score_q1, 4),
        "score_ppm": round(score_ppm if score_ppm is not None else np.nan, 4),
    }
    row["score_used"] = row["score_ppm"] if score_priority == "ppm" else row["score_q1"]
    if abs(row["delta_ppm"]) > 50:
        row["warning"] = f"Large ppm deviation ({row['delta_ppm']:+.1f} ppm)"
    return row


def best_isobars_for_peptides(
    peptides: List[str],
    peptide_charges: Iterable[int],
    candidates: List[Dict],
    q1_fwhm_da: float,
    ppm_window: float,
    score_priority: str,
    preselect: int,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
    show_progress: bool = True,
) -> pd.DataFrame:
    from tqdm.auto import tqdm  # import locally to keep helpers lightweight

    rows = []
    pbar = tqdm(peptides, disable=not show_progress, desc="Scoring peptides")
    for seq in pbar:
        rows.append(
            best_isobar_for_peptide(
                seq,
                candidates=candidates,
                peptide_charges=peptide_charges,
                q1_fwhm_da=q1_fwhm_da,
                ppm_window=ppm_window,
                score_priority=score_priority,
                preselect=preselect,
                allow_heavy_first_residue=allow_heavy_first_residue,
                allow_mtraq=allow_mtraq,
            )
        )
    df = pd.DataFrame(rows)
    want = [
        "peptide",
        "pep_z",
        "pep_mz",
        "family",
        "peg_z",
        "peg_endgroup",
        "peg_n",
        "peg_adducts",
        "peg_mz",
        "delta_mDa",
        "delta_ppm",
        "score_q1",
        "score_ppm",
        "score_used",
        "warning",
    ]
    for c in want:
        if c not in df.columns:
            df[c] = np.nan
    return df[want]


def topk_table_for_all_peptides(
    peptides: List[str],
    peptide_charges: Iterable[int],
    candidates: List[Dict],
    q1_fwhm_da: float,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
    top_k: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Export Top-K candidates per peptide directly from helper output."""
    from tqdm.auto import tqdm

    rows = []
    pbar = tqdm(peptides, disable=not show_progress, desc="Top-K per peptide")
    for seq in pbar:
        tops = top_polymer_matches_for_peptide(
            seq,
            candidates=candidates,
            peptide_charges=peptide_charges,
            q1_fwhm_da=q1_fwhm_da,
            allow_heavy_first_residue=allow_heavy_first_residue,
            allow_mtraq=allow_mtraq,
            top_n=top_k,
        )
        for r in tops:
            rows.append(
                {
                    "peptide": seq,
                    "family": r["family"],
                    "peg_endgroup": r["endgroup"],
                    "peg_n": int(r["n"]),
                    "peg_adducts": r["adducts"],
                    "peg_z": int(r["z"]),
                    "peg_mz": float(r["mz"]),
                    "q1_weight": float(r["weight"]),  # averaged over scenarios
                    "delta_da_mean": float(r["delta_da"]),
                }
            )
    return pd.DataFrame(rows)


# -------------------- Derived features & stats --------------------
def adduct_feature_counts(adduct_str: str) -> Dict[str, int]:
    parts = [p for p in str(adduct_str).split("+") if p]
    c = Counter(parts)
    return {
        "cnt_H": c.get("H", 0),
        "cnt_Na": c.get("Na", 0),
        "cnt_K": c.get("K", 0),
        "cnt_NH4": c.get("NH4", 0),
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_delta_ppm"] = df["delta_ppm"].abs()
    df["abs_delta_mDa"] = df["delta_mDa"].abs()
    counts = df["peg_adducts"].apply(adduct_feature_counts).apply(pd.Series)
    df = pd.concat([df, counts], axis=1)
    for ion in ("H", "Na", "K", "NH4"):
        df[f"has_{ion}"] = (df[f"cnt_{ion}"] > 0).astype(int)
    df["adduct_count"] = df[["cnt_H", "cnt_Na", "cnt_K", "cnt_NH4"]].sum(axis=1)
    return df


def point_biserial_corr(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1)
    x = np.asarray(x).reshape(-1)
    n = min(len(y), len(x))
    if n == 0:
        return 0.0
    y = y[:n]
    x = x[:n]
    if np.nanstd(y) == 0 or np.nanstd(x) == 0:
        return 0.0
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(y[mask], x[mask])[0, 1])


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS D statistic (no SciPy)."""
    x = np.sort(x)
    y = np.sort(y)
    xs = np.r_[x, y]
    xs = np.unique(xs)
    Fx = (
        (np.searchsorted(x, xs, side="right") / len(x))
        if len(x)
        else np.zeros_like(xs, dtype=float)
    )
    Fy = (
        (np.searchsorted(y, xs, side="right") / len(y))
        if len(y)
        else np.zeros_like(xs, dtype=float)
    )
    return float(np.max(np.abs(Fx - Fy))) if len(xs) else 0.0


# -------------------- Binning / thresholds (ppm and weight) -------------------
def ppm_bin_labels(edges: List[float]) -> List[str]:
    labels = [f"≤{int(e)}" for e in edges]
    labels.append(f">{int(edges[-1])}")
    return labels


def compute_ppm_bin_fractions(
    df: pd.DataFrame, hit_col: str, edges: List[float]
) -> pd.DataFrame:
    if hit_col not in df.columns:
        raise KeyError(f"compute_ppm_bin_fractions: '{hit_col}' not in df.columns")
    if "abs_delta_ppm" not in df.columns:
        raise KeyError("compute_ppm_bin_fractions: 'abs_delta_ppm' not in df.columns")
    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()
    bins = [-np.inf] + edges + [np.inf]
    labels = [f"≤{int(e)}" for e in edges] + [f">{int(edges[-1])}"]
    sub["ppm_bin"] = pd.cut(
        sub["abs_delta_ppm"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    groups = np.sort(sub[hit_col].dropna().unique())
    idx = pd.MultiIndex.from_product([groups, labels], names=[hit_col, "ppm_bin"])
    counts = (
        sub.groupby([hit_col, "ppm_bin"], observed=False)
        .size()
        .reindex(idx, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = (
        counts.groupby(hit_col, observed=False)["count"]
        .sum()
        .rename("group_total")
        .reset_index()
    )
    out = counts.merge(totals, on=hit_col, how="left")
    out["fraction"] = out["count"] / out["group_total"].replace(0, np.nan)
    out["ppm_bin"] = pd.Categorical(out["ppm_bin"], categories=labels, ordered=True)
    return out


def weight_bin_labels(edges: List[float]) -> List[str]:
    labels = [f"≤{e:g}" for e in edges]
    labels.append(f">{edges[-1]:g}")
    return labels


def compute_weight_bin_fractions(
    df: pd.DataFrame, hit_col: str, edges: List[float], weight_col: str = "score_q1"
) -> pd.DataFrame:
    if hit_col not in df.columns:
        raise KeyError(f"compute_weight_bin_fractions: '{hit_col}' not in df.columns")
    if weight_col not in df.columns:
        raise KeyError(
            f"compute_weight_bin_fractions: '{weight_col}' not in df.columns"
        )
    sub = df[[hit_col, weight_col]].dropna().copy()
    bins = [-np.inf] + edges + [np.inf]
    labels = [f"≤{e:g}" for e in edges] + [f">{edges[-1]:g}"]
    sub["wbin"] = pd.cut(
        sub[weight_col], bins=bins, labels=labels, include_lowest=True, ordered=True
    )
    groups = np.sort(sub[hit_col].dropna().unique())
    idx = pd.MultiIndex.from_product([groups, labels], names=[hit_col, "wbin"])
    counts = (
        sub.groupby([hit_col, "wbin"], observed=False)
        .size()
        .reindex(idx, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = (
        counts.groupby(hit_col, observed=False)["count"]
        .sum()
        .rename("group_total")
        .reset_index()
    )
    out = counts.merge(totals, on=hit_col, how="left")
    out["fraction"] = out["count"] / out["group_total"].replace(0, np.nan)
    out["wbin"] = pd.Categorical(out["wbin"], categories=labels, ordered=True)
    return out


def threshold_summary_ppm(
    df: pd.DataFrame, hit_col: str, thresholds: List[float]
) -> pd.DataFrame:
    """Lower is better: abs_delta_ppm ≤ T."""
    sub = df[[hit_col, "abs_delta_ppm"]].dropna().copy()
    rows = []
    n_hit = int(sub[hit_col].sum())
    n_nonhit = int((1 - sub[hit_col]).sum())
    for T in thresholds:
        inB = sub["abs_delta_ppm"] <= T
        a = int(((sub[hit_col] == 1) & inB).sum())
        b = n_hit - a
        c = int(((sub[hit_col] == 0) & inB).sum())
        d = n_nonhit - c
        p1 = a / n_hit if n_hit else np.nan
        p2 = c / n_nonhit if n_nonhit else np.nan
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan
        a1, b1, c1, d1 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_ = (a1 * d1) / (b1 * c1)
        se_log_or = math.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
        ci_lo = math.exp(math.log(or_) - 1.96 * se_log_or)
        ci_hi = math.exp(math.log(or_) + 1.96 * se_log_or)
        rows.append(
            {
                "threshold": T,
                "precision": (a / (a + c)) if (a + c) else np.nan,
                "recall": p1,
                "F1": (
                    (2 * (a / (a + c)) * p1 / ((a / (a + c)) + p1))
                    if (a + c) and (p1 == p1) and ((a / (a + c)) + p1) > 0
                    else np.nan
                ),
                "risk_ratio": rr,
                "odds_ratio": or_,
                "or_95ci_lo": ci_lo,
                "or_95ci_hi": ci_hi,
                "nonhit_capture_frac": (c / n_nonhit) if n_nonhit else np.nan,
                "n_hit_total": n_hit,
                "n_nonhit_total": n_nonhit,
                "n_hit_in": a,
                "n_nonhit_in": c,
            }
        )
    return pd.DataFrame(rows)


def threshold_summary_weight(
    df: pd.DataFrame,
    hit_col: str,
    thresholds: List[float],
    weight_col: str = "score_q1",
) -> pd.DataFrame:
    """Higher is better: weight ≥ T."""
    sub = df[[hit_col, weight_col]].dropna().copy()
    rows = []
    n_hit = int(sub[hit_col].sum())
    n_nonhit = int((1 - sub[hit_col]).sum())
    for T in thresholds:
        inB = sub[weight_col] >= T
        a = int(((sub[hit_col] == 1) & inB).sum())
        b = n_hit - a
        c = int(((sub[hit_col] == 0) & inB).sum())
        d = n_nonhit - c
        p1 = a / n_hit if n_hit else np.nan
        p2 = c / n_nonhit if n_nonhit else np.nan
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan
        a1, b1, c1, d1 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_ = (a1 * d1) / (b1 * c1)
        se_log_or = math.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
        ci_lo = math.exp(math.log(or_) - 1.96 * se_log_or)
        ci_hi = math.exp(math.log(or_) + 1.96 * se_log_or)
        rows.append(
            {
                "threshold": T,
                "precision": (a / (a + c)) if (a + c) else np.nan,
                "recall": p1,
                "F1": (
                    (2 * (a / (a + c)) * p1 / ((a / (a + c)) + p1))
                    if (a + c) and (p1 == p1) and ((a / (a + c)) + p1) > 0
                    else np.nan
                ),
                "risk_ratio": rr,
                "odds_ratio": or_,
                "or_95ci_lo": ci_lo,
                "or_95ci_hi": ci_hi,
                "nonhit_capture_frac": (c / n_nonhit) if n_nonhit else np.nan,
                "n_hit_total": n_hit,
                "n_nonhit_total": n_nonhit,
                "n_hit_in": a,
                "n_nonhit_in": c,
            }
        )
    return pd.DataFrame(rows)


# -------------------- Plotting --------------------
def save_plot(fig, outpath: Path):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_ecdf_metric_by_hit(
    df: pd.DataFrame,
    hit_col: str,
    value_col: str,
    outpath: Path,
    thresholds: List[float],
    x_label: str,
    x_max: Optional[float] = None,
):
    sub = df[[hit_col, value_col]].dropna()
    if sub.empty:
        return
    fig = plt.figure()
    for label, grp in sub.groupby(hit_col):
        x = np.sort(grp[value_col].to_numpy(float))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=f"{hit_col}={label}")
    for T in thresholds:
        plt.axvline(T, linestyle="--", linewidth=1)
    if x_max is not None:
        plt.xlim(0, x_max)
    plt.ylim(0, 1)
    plt.xlabel(x_label)
    plt.ylabel("cumulative fraction ≤ x (within group)")
    plt.title(f"ECDF of {x_label} by hit status")
    plt.legend()
    save_plot(fig, outpath)


def plot_ppm_bin_fractions(
    df_bins: pd.DataFrame, hit_col: str, outpath: Path, edges: List[float]
):
    if df_bins.empty:
        return
    bin_order = ppm_bin_labels(edges)
    fig = plt.figure(figsize=(max(6, 1.1 * len(bin_order)), 4.5))
    groups = sorted(df_bins[hit_col].dropna().unique())
    width = 0.35 if len(groups) == 2 else 0.25
    x = np.arange(len(bin_order))
    for i, g in enumerate(groups):
        sub = df_bins[df_bins[hit_col] == g].copy()
        sub["ppm_bin"] = pd.Categorical(
            sub["ppm_bin"], categories=bin_order, ordered=True
        )
        sub = sub.set_index("ppm_bin").reindex(bin_order)
        y = sub["fraction"].to_numpy(float)
        plt.bar(
            x + (i - (len(groups) - 1) / 2) * width,
            y,
            width=width,
            label=f"{hit_col}={int(g)}",
        )
    plt.xticks(x, bin_order)
    plt.ylim(0, 1)
    plt.xlabel("abs(Δppm) bins")
    plt.ylabel("fraction within group")
    plt.title("Fraction in |Δppm| bins by hit status (normalized)")
    plt.legend()
    save_plot(fig, outpath)


def plot_weight_bin_fractions(
    df_bins: pd.DataFrame, hit_col: str, outpath: Path, edges: List[float]
):
    if df_bins.empty:
        return
    bin_order = weight_bin_labels(edges)
    fig = plt.figure(figsize=(max(6, 1.1 * len(bin_order)), 4.5))
    groups = sorted(df_bins[hit_col].dropna().unique())
    width = 0.35 if len(groups) == 2 else 0.25
    x = np.arange(len(bin_order))
    for i, g in enumerate(groups):
        sub = df_bins[df_bins[hit_col] == g].copy()
        sub["wbin"] = pd.Categorical(sub["wbin"], categories=bin_order, ordered=True)
        sub = sub.set_index("wbin").reindex(bin_order)
        y = sub["fraction"].to_numpy(float)
        plt.bar(
            x + (i - (len(groups) - 1) / 2) * width,
            y,
            width=width,
            label=f"{hit_col}={int(g)}",
        )
    plt.xticks(x, bin_order)
    plt.ylim(0, 1)
    plt.xlabel("Q1 weight bins")
    plt.ylabel("fraction within group")
    plt.title("Fraction in Q1 weight bins by hit status (normalized)")
    plt.legend()
    save_plot(fig, outpath)


def plot_pr_curve(thr_df: pd.DataFrame, title: str, outpath: Path):
    if thr_df.empty:
        return
    fig = plt.figure()
    plt.plot(
        thr_df["recall"].to_numpy(float),
        thr_df["precision"].to_numpy(float),
        marker="o",
    )
    plt.xlabel("Recall (hit coverage)")
    plt.ylabel("Precision (hit fraction among explained)")
    plt.title(title)
    save_plot(fig, outpath)


def plot_pareto(thr_df: pd.DataFrame, title: str, outpath: Path):
    if thr_df.empty:
        return
    fig = plt.figure()
    x = thr_df["nonhit_capture_frac"].to_numpy(float)
    y = thr_df["recall"].to_numpy(float)
    plt.plot(x, y, marker="o")
    plt.xlabel("Non-hit capture fraction (in set)")
    plt.ylabel("Hit coverage (in set)")
    plt.title(title)
    save_plot(fig, outpath)


def plot_bar_metric(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    title: str,
    outpath: Path,
    top_k: int = 12,
):
    if df.empty:
        return
    sub = df.sort_values(metric_col, ascending=False).head(top_k)
    fig = plt.figure(figsize=(10, min(0.5 * len(sub) + 2, 12)))
    y = np.arange(len(sub))
    plt.barh(y, sub[metric_col].to_numpy(float))
    plt.yticks(y, sub[group_col].astype(str).tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(metric_col)
    plt.title(title)
    save_plot(fig, outpath)


def plot_scatter_absppm_vs_weight(df: pd.DataFrame, hit_col: str, outpath: Path):
    sub = df[[hit_col, "abs_delta_ppm", "score_q1"]].dropna()
    if sub.empty:
        return
    fig = plt.figure()
    for label, grp in sub.groupby(hit_col):
        plt.scatter(
            grp["abs_delta_ppm"].to_numpy(float),
            grp["score_q1"].to_numpy(float),
            s=12,
            alpha=0.6,
            label=f"{hit_col}={label}",
        )
    plt.xlabel("|Δppm|")
    plt.ylabel("Q1 weight (avg over scenarios)")
    plt.title("Weight vs |Δppm| by hit status")
    plt.legend()
    save_plot(fig, outpath)


def plot_hist_n_by_hit(df: pd.DataFrame, hit_col: str, outpath: Path):
    sub = df[[hit_col, "peg_n"]].dropna()
    if sub.empty:
        return
    fig = plt.figure()
    for label, grp in sub.groupby(hit_col):
        plt.hist(
            grp["peg_n"].to_numpy(int), bins=30, alpha=0.6, label=f"{hit_col}={label}"
        )
    plt.xlabel("polymer repeat n (best match)")
    plt.ylabel("count")
    plt.title("Distribution of PEG-like repeat n by hit status")
    plt.legend()
    save_plot(fig, outpath)


# -------------------- Sweeps --------------------
def format_tuple_str(t: Tuple[str, ...]) -> str:
    return ",".join(t)


def sweep_families_configs(
    df_in: pd.DataFrame,
    peptides: List[str],
    peptide_charges: Tuple[int, ...],
    families: Tuple[str, ...],
    endgroup_sets: List[Tuple[str, ...]],
    adduct_sets: List[Tuple[str, ...]],
    peg_charge_sets: List[Tuple[int, ...]],
    n_range: Tuple[int, int],
    mz_range: Tuple[float, float],
    q1_fwhm: float,
    ppm_window: float,
    thresholds_ppm: List[float],
    preselect: int,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
    pepcol: str,
    hit_col: str,
    outdir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Sweep across families x endgroup_sets x adduct_sets x peg_charge_sets.
    Computes metrics using **ppm thresholds** (as in original).
    """
    from tqdm.auto import tqdm

    rows = []
    total = len(families) * len(endgroup_sets) * len(adduct_sets) * len(peg_charge_sets)
    pbar = tqdm(total=total, disable=not show_progress, desc="Sweeping configs")
    for fam, egs, ads, zset in product(
        families, endgroup_sets, adduct_sets, peg_charge_sets
    ):
        fam_egs = (
            tuple(POLYMER_ENDGROUP[fam].keys())
            if ("auto" in egs)
            else tuple([e for e in egs if e in POLYMER_ENDGROUP[fam]])
        )
        if not fam_egs:
            pbar.update(1)
            continue
        try:
            candidates = build_family_candidates(
                families=(fam,),
                peg_charges=zset,
                allowed_adducts=ads,
                endgroups=fam_egs,
                n_min=n_range[0],
                n_max=n_range[1],
                mz_min=mz_range[0],
                mz_max=mz_range[1],
                show_progress=False,
            )
            if not candidates:
                pbar.update(1)
                continue
            df_best = best_isobars_for_peptides(
                peptides,
                peptide_charges=peptide_charges,
                candidates=candidates,
                q1_fwhm_da=q1_fwhm,
                ppm_window=ppm_window,
                score_priority="ppm",
                preselect=preselect,
                allow_heavy_first_residue=allow_heavy_first_residue,
                allow_mtraq=allow_mtraq,
                show_progress=False,
            )
            join = df_in[[pepcol, hit_col]].merge(
                df_best, left_on=pepcol, right_on="peptide", how="left"
            )
            join[hit_col] = join[hit_col].apply(coerce_is_hit)
            dj = add_derived_features(join)

            thr = threshold_summary_ppm(dj, hit_col=hit_col, thresholds=thresholds_ppm)

            v_hit = dj.loc[dj[hit_col] == 1, "abs_delta_ppm"].dropna().to_numpy(float)
            v_non = dj.loc[dj[hit_col] == 0, "abs_delta_ppm"].dropna().to_numpy(float)
            ksD = ks_statistic(v_hit, v_non)

            for _, r in thr.iterrows():
                rows.append(
                    {
                        "family": fam,
                        "endgroups": format_tuple_str(fam_egs),
                        "adducts": format_tuple_str(ads),
                        "peg_charges": ",".join(map(str, zset)),
                        "threshold_ppm": r["threshold"],
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "F1": r["F1"],
                        "risk_ratio": r["risk_ratio"],
                        "odds_ratio": r["odds_ratio"],
                        "nonhit_capture_frac": r["nonhit_capture_frac"],
                        "ks_D": ksD,
                        "n_hit_total": r["n_hit_total"],
                        "n_nonhit_total": r["n_nonhit_total"],
                        "n_hit_leT": r["n_hit_in"],
                        "n_nonhit_leT": r["n_nonhit_in"],
                    }
                )
        except Exception as e:
            logger.warning(
                f"Sweep config failed: family={fam}, egs={egs}, ads={ads}, z={zset}: {e}"
            )
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def sweep_families_configs_weight(
    df_in: pd.DataFrame,
    peptides: List[str],
    peptide_charges: Tuple[int, ...],
    families: Tuple[str, ...],
    endgroup_sets: List[Tuple[str, ...]],
    adduct_sets: List[Tuple[str, ...]],
    peg_charge_sets: List[Tuple[int, ...]],
    n_range: Tuple[int, int],
    mz_range: Tuple[float, float],
    q1_fwhm: float,
    weight_thresholds: List[float],
    preselect: int,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
    pepcol: str,
    hit_col: str,
    outdir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Optional sweep using **weight thresholds** (≥T).
    """
    from tqdm.auto import tqdm

    rows = []
    total = len(families) * len(endgroup_sets) * len(adduct_sets) * len(peg_charge_sets)
    pbar = tqdm(
        total=total, disable=not show_progress, desc="Sweeping configs (weight)"
    )
    for fam, egs, ads, zset in product(
        families, endgroup_sets, adduct_sets, peg_charge_sets
    ):
        fam_egs = (
            tuple(POLYMER_ENDGROUP[fam].keys())
            if ("auto" in egs)
            else tuple([e for e in egs if e in POLYMER_ENDGROUP[fam]])
        )
        if not fam_egs:
            pbar.update(1)
            continue
        try:
            candidates = build_family_candidates(
                families=(fam,),
                peg_charges=zset,
                allowed_adducts=ads,
                endgroups=fam_egs,
                n_min=n_range[0],
                n_max=n_range[1],
                mz_min=mz_range[0],
                mz_max=mz_range[1],
                show_progress=False,
            )
            if not candidates:
                pbar.update(1)
                continue

            # score_priority doesn't affect score_q1 column; we ask for ppm to match earlier flow
            df_best = best_isobars_for_peptides(
                peptides,
                peptide_charges=peptide_charges,
                candidates=candidates,
                q1_fwhm_da=q1_fwhm,
                ppm_window=10.0,
                score_priority="ppm",
                preselect=preselect,
                allow_heavy_first_residue=allow_heavy_first_residue,
                allow_mtraq=allow_mtraq,
                show_progress=False,
            )
            join = df_in[[pepcol, hit_col]].merge(
                df_best, left_on=pepcol, right_on="peptide", how="left"
            )
            join[hit_col] = join[hit_col].apply(coerce_is_hit)
            dj = add_derived_features(join)

            thr = threshold_summary_weight(
                dj, hit_col=hit_col, thresholds=weight_thresholds, weight_col="score_q1"
            )

            for _, r in thr.iterrows():
                rows.append(
                    {
                        "family": fam,
                        "endgroups": format_tuple_str(fam_egs),
                        "adducts": format_tuple_str(ads),
                        "peg_charges": ",".join(map(str, zset)),
                        "threshold_weight": r["threshold"],
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "F1": r["F1"],
                        "risk_ratio": r["risk_ratio"],
                        "odds_ratio": r["odds_ratio"],
                        "nonhit_capture_frac": r["nonhit_capture_frac"],
                        "n_hit_total": r["n_hit_total"],
                        "n_nonhit_total": r["n_nonhit_total"],
                        "n_hit_in": r["n_hit_in"],
                        "n_nonhit_in": r["n_nonhit_in"],
                    }
                )
        except Exception as e:
            logger.warning(
                f"Sweep(weight) failed: family={fam}, egs={egs}, ads={ads}, z={zset}: {e}"
            )
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# -------------------- Hits sorted by score --------------------
def make_hits_sorted_by_score(
    df_in: pd.DataFrame,
    df_best: pd.DataFrame,
    pepcol: str,
    hit_col: str,
    mode: str,  # "ppm", "q1", "combo"
    top_n: int,
    outdir: Path,
):
    keep_cols = [
        "peptide",
        "family",
        "score_ppm",
        "score_q1",
        "delta_ppm",
        "delta_mDa",
        "peg_endgroup",
        "peg_z",
        "peg_n",
        "peg_adducts",
        "peg_mz",
        "pep_z",
        "pep_mz",
    ]
    for c in keep_cols:
        if c not in df_best.columns:
            df_best[c] = np.nan
    tmp = df_best.merge(
        df_in[[pepcol, hit_col]], left_on="peptide", right_on=pepcol, how="left"
    )
    hits = tmp[tmp[hit_col] == 1].copy()
    if hits.empty:
        logger.warning("No hits found to rank for the PEG score plot.")
        return
    if mode == "ppm":
        hits["peg_artifact_score"] = hits["score_ppm"]
    elif mode == "q1":
        hits["peg_artifact_score"] = hits["score_q1"]
    elif mode == "combo":
        hits["peg_artifact_score"] = 0.5 * (
            hits["score_ppm"].fillna(0) + hits["score_q1"].fillna(0)
        )
    else:
        raise ValueError(f"Unknown peg-score mode '{mode}'")
    hits = hits.dropna(subset=["peg_artifact_score"])
    if hits.empty:
        logger.warning(f"No finite PEG scores for hits using mode '{mode}'.")
        return
    hits_sorted = hits.sort_values("peg_artifact_score", ascending=False).head(top_n)
    csv_path = outdir / f"hits_sorted_by_peg_score_{mode}.csv"
    hits_sorted.to_csv(csv_path, index=False)
    logger.info(f"Wrote: {csv_path}")
    fig = plt.figure(figsize=(10, min(0.5 * len(hits_sorted) + 2, 12)))
    y = np.arange(len(hits_sorted))
    plt.barh(y, hits_sorted["peg_artifact_score"].to_numpy(float))
    plt.yticks(y, hits_sorted["peptide"].astype(str).tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(f"PEG-like score ({mode})")
    plt.title(
        f"Top {min(top_n, len(hits_sorted))} HIT peptides by PEG-like score ({mode})"
    )
    save_plot(fig, outdir / f"hits_sorted_by_peg_score_{mode}.png")


# -------------------- Category hit-rate (optional) --------------------
def category_hit_rates(df_join: pd.DataFrame, hit_col: str) -> pd.DataFrame:
    rows = []
    if "family" in df_join.columns:
        for grp, sub in df_join.groupby("family"):
            n = len(sub)
            hits = sub[hit_col].sum(skipna=True)
            rows.append(
                {
                    "category": "family",
                    "level": grp,
                    "n": n,
                    "hit_rate": (hits / n if n else np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["category", "level"]).reset_index(drop=True)
