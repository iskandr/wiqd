#!/usr/bin/env python3
"""
Triple-profile (Loose/Medium/Strict) Q1/Q3 comparison with per-family overlays
and a grid of peptide-mod scenario × polymer family × polymer mods.

Profiles (fixed):
  • Loose : Q1 FWHM = 0.4 Da,  Q3 FWHM = 0.1 Da   (per-peptide |ΔDa| gating)
  • Medium: Q1 FWHM = 100 ppm, Q3 FWHM = 30 ppm  (per-peptide |Δppm| gating)
  • Strict: Q1 FWHM = 30 ppm,  Q3 FWHM = 10 ppm  (per-peptide |Δppm| gating)

Peptide-mod scenarios scored independently:
  • none         (no uncertain ref-only mods)
  • heavy        (allow heavy first residue; mTRAQ off)
  • mtraq        (allow mTRAQ Δ0/Δ8; heavy off)
  • heavy_mtraq  (allow both heavy first residue and mTRAQ)

Outputs
-------
CSV (per profile & scenario):
  best_per_peptide_{profile}_{scenario}.csv
  input_joined_{profile}_{scenario}.csv

Threshold summaries (dense; avoid dot-like curves):
  thresholds_Q3_{profile}_{scenario}_{units}.csv
  thresholds_Q3_per_family_{profile}_{scenario}_{units}.csv

Grids (enrichment tables):
  grid_enrichment_long.csv
  (per-family heatmaps in plots/family_grids/)

Plots (overall & per-family overlays; per-profile grids):
  plots/overlay_PR_Q3_overall.png
  plots/overlay_Pareto_Q3_overall.png
  plots/family/overlay_PR_Q3_{family}.png
  plots/family/overlay_Pareto_Q3_{family}.png
  plots/family_grids/heatmap_{profile}_{family}.png
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

# Headless-safe matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Local helpers (no modification required)
import polymer_helpers as H

# Scorer entrypoint for Q1 units
from wiqd_peptide_similarity import (
    pep_mass,
    mz_from_mass,
    best_polymer_match_for_reference,
)


# --------------------------- Logging ---------------------------
logger = logging.getLogger("polymer.triple_compare")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# --------------------------- CLI ---------------------------
def make_parser():
    p = argparse.ArgumentParser(
        description="Loose/Medium/Strict Q1/Q3 overlays with per-family splits and mod grid."
    )
    p.add_argument(
        "--in",
        dest="in_csv",
        required=True,
        help="Input CSV with peptide and hit columns.",
    )
    p.add_argument(
        "--pepcol", default="peptide", help="Peptide column (default: peptide)."
    )
    p.add_argument("--hitcol", default="is_hit", help="Hit column (default: is_hit).")
    p.add_argument(
        "--out",
        dest="outdir",
        default=None,
        help="Output directory (default uses input stem).",
    )

    # Chemistry/candidate space (same semantics as polymer.py)
    p.add_argument(
        "--families",
        default="PEG,PPG,PTMEG,PDMS",
        help="Comma list of families to consider.",
    )
    p.add_argument(
        "--endgroups",
        default="auto",
        help="Comma list (e.g., diol,monoMe,diMe) or 'auto' per family.",
    )
    p.add_argument(
        "--adducts", default="H,Na,K,NH4", help="Comma list of allowed adduct monomers."
    )
    p.add_argument(
        "--peptide-charges",
        default="2,3",
        help="Comma list of peptide z (default: 2,3).",
    )
    p.add_argument(
        "--peg-charges",
        default="1,2,3",
        help="Comma list of polymer z (default: 1,2,3).",
    )
    p.add_argument(
        "--n-range",
        default="5,100",
        help="Oligomer repeat inclusive range (default 5,100).",
    )
    p.add_argument(
        "--mz-range",
        default="200,1500",
        help="m/z window for candidates (default 200,1500).",
    )

    # Retrieval controls
    p.add_argument(
        "--preselect",
        type=int,
        default=200,
        help="Top-K candidates per peptide (from scorer).",
    )
    p.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    p.add_argument("--no-plots", action="store_true", help="Skip plotting.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # Dense threshold grids (piecewise linspace specs)
    p.add_argument(
        "--ppm-thresholds",
        default="0.5-10x25,10-60x25,60-200x30",
        help="Piecewise linspace for ppm thresholds (a-bxN,comma-separated).",
    )
    p.add_argument(
        "--da-thresholds",
        default="0.001-0.02x25,0.02-0.2x30,0.2-0.5x20",
        help="Piecewise linspace for Da thresholds (a-bxN,comma-separated).",
    )

    return p


# --------------------------- Profiles & Scenarios ---------------------------
@dataclass
class Profile:
    name: str
    q1: Tuple[str, float]  # ("da", 0.4) or ("ppm", 100.0)
    q3: Tuple[str, float]  # ("da", 0.1) or ("ppm", 30.0)

    def q1_label(self) -> str:
        u, v = self.q1
        return f"{v:g} {u.upper()}"

    def q3_label(self) -> str:
        u, v = self.q3
        return f"{v:g} {u.upper()}"


@dataclass(frozen=True)
class Scenario:
    name: str
    allow_heavy: bool
    allow_mtraq: bool


SCENARIOS: List[Scenario] = [
    Scenario("none", False, False),
    Scenario("heavy", True, False),
    Scenario("mtraq", False, True),
    Scenario("heavy_mtraq", True, True),
]

PROFILES: List[Profile] = [
    Profile("loose", q1=("da", 0.4), q3=("da", 0.1)),
    Profile("medium", q1=("ppm", 100), q3=("ppm", 30)),
    Profile("strict", q1=("ppm", 30), q3=("ppm", 10)),
]


# --------------------------- Utils ---------------------------
def parse_piecewise(spec: str) -> List[float]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rng, n = part.split("x")
            a, b = rng.split("-")
            a = float(a)
            b = float(b)
            n = int(n)
        except Exception as e:
            raise ValueError(f"Bad threshold segment '{part}': {e}")
        seg = np.linspace(a, b, n)
        out.extend(seg.tolist())
    out = np.unique(np.asarray(out, dtype=float))
    return [float(x) for x in out]


def gaussian_q3(delta_da: float, delta_ppm: float, q3: Tuple[str, float]) -> float:
    unit, fwhm = q3
    if unit == "da":
        return H.gaussian_score_delta(abs(delta_da), fwhm)
    elif unit == "ppm":
        return H.gaussian_score_delta(abs(delta_ppm), fwhm)
    else:
        raise ValueError("q3 units must be 'da' or 'ppm'")


# --------------------------- Scoring ---------------------------
def score_one_peptide(
    seq: str,
    candidates: List[Dict],
    peptide_charges: Iterable[int],
    q1: Tuple[str, float],
    q3: Tuple[str, float],
    preselect: int,
    scen: Scenario,
) -> Dict[str, object]:
    """
    Call scorer with Q1 units (da/ppm). Compute per-peptide Q3 score (da/ppm) and
    pick best candidate by Q3 score. Also carry family/endgroup/adducts/z/n/mz.
    """
    res = best_polymer_match_for_reference(
        seq,
        candidates=candidates,
        peptide_z=tuple(peptide_charges),
        q1_fwhm=q1,  # ("da", ... ) or ("ppm", ... )
        q1_tol=None,
        top_n=int(preselect),
        allow_heavy_first_residue=scen.allow_heavy,
        allow_mtraq=scen.allow_mtraq,
    )
    tops = list(res.get("top") or [])
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
        "score_q3": np.nan,
        "score_used": np.nan,
        "scenario": scen.name,
        "warning": "No candidate within windows" if not tops else None,
    }
    if not tops:
        return base

    chosen = None
    best_metric = -np.inf
    for c in tops:
        z = int(c["z"])
        pep_mz = mz_from_mass(pep_mass(seq), z)
        d_da = float(c["delta_da"])
        d_ppm = (d_da / pep_mz) * 1e6 if pep_mz > 0 else np.nan
        s_q1 = float(
            c["weight"]
        )  # from scorer (already averaged over ref-mod scenarios)
        s_q3 = gaussian_q3(d_da, d_ppm, q3)  # our per-peptide Q3 gate

        if s_q3 is not None and float(s_q3) > best_metric:
            best_metric = float(s_q3)
            chosen = (c, z, pep_mz, d_da, d_ppm, s_q1, s_q3)

    if chosen is None:
        return base

    c, z, pep_mz, d_da, d_ppm, s_q1, s_q3 = chosen
    out = {
        **base,
        "pep_z": z,
        "pep_mz": round(pep_mz, 6),
        "family": c["family"],
        "peg_z": z,
        "peg_endgroup": c["endgroup"],
        "peg_n": int(c["n"]),
        "peg_adducts": c["adducts"],
        "peg_mz": round(float(c["mz"]), 6),
        "delta_mDa": round(1000.0 * d_da, 3),
        "delta_ppm": round(d_ppm, 2),
        "score_q1": round(s_q1, 4),
        "score_q3": round(s_q3 if s_q3 is not None else np.nan, 4),
        "score_used": round(s_q3 if s_q3 is not None else np.nan, 4),
    }
    return out


def score_profile_scenario(
    prof: Profile,
    scen: Scenario,
    peptides: List[str],
    peptide_charges: Tuple[int, ...],
    candidates: List[Dict],
    df_in: pd.DataFrame,
    pepcol: str,
    hitcol: str,
    preselect: int,
    progress: bool,
    outdir: Path,
) -> Dict[str, pd.DataFrame]:
    from tqdm.auto import tqdm

    logger.info(
        f"[{prof.name} | {scen.name}] Q1={prof.q1_label()}, Q3={prof.q3_label()}"
    )

    rows = []
    it = tqdm(peptides, disable=not progress, desc=f"Scoring ({prof.name}/{scen.name})")
    for seq in it:
        rows.append(
            score_one_peptide(
                seq, candidates, peptide_charges, prof.q1, prof.q3, preselect, scen
            )
        )
    df_best = pd.DataFrame(rows)
    df_best.to_csv(
        outdir / f"best_per_peptide_{prof.name}_{scen.name}.csv", index=False
    )

    join = df_in.merge(
        df_best, how="left", left_on=pepcol, right_on="peptide", suffixes=("", "__best")
    )
    join = H.ensure_single_hit_col(
        join, hit_col=hitcol, outdir=outdir, behavior="error"
    )

    # Derived features
    dj = H.add_derived_features(join)
    dj["abs_delta_da"] = dj["delta_mDa"].abs() / 1000.0
    dj["polymer_mods"] = (
        dj["peg_endgroup"].astype(str) + "|" + dj["peg_adducts"].astype(str)
    )
    dj["profile"] = prof.name
    dj["q1_label"] = prof.q1_label()
    dj["q3_label"] = prof.q3_label()
    dj["scenario"] = df_best.get("scenario", scen.name)
    dj.to_csv(outdir / f"input_joined_{prof.name}_{scen.name}.csv", index=False)

    return {"df_best": df_best, "dj": dj}


# --------------------------- Threshold summaries ---------------------------
def threshold_summary_on_metric(
    df: pd.DataFrame, hit_col: str, value_col: str, thresholds: List[float]
) -> pd.DataFrame:
    """Lower is better on arbitrary value_col (e.g., abs_delta_da)."""
    sub = df[[hit_col, value_col]].dropna().copy()
    rows = []
    n_hit = int(sub[hit_col].sum())
    n_nonhit = int((1 - sub[hit_col]).sum())
    for T in thresholds:
        inB = sub[value_col] <= T
        a = int(((sub[hit_col] == 1) & inB).sum())
        b = n_hit - a
        c = int(((sub[hit_col] == 0) & inB).sum())
        d = n_nonhit - c
        p1 = a / n_hit if n_hit else np.nan
        p2 = c / n_nonhit if n_nonhit else np.nan
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan
        a1, b1, c1, d1 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_ = (a1 * d1) / (b1 * c1)
        se = np.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
        ci_lo = float(np.exp(np.log(or_) - 1.96 * se))
        ci_hi = float(np.exp(np.log(or_) + 1.96 * se))
        rows.append(
            dict(
                threshold=T,
                precision=(a / (a + c)) if (a + c) else np.nan,
                recall=p1,
                F1=(
                    (2 * (a / (a + c)) * p1 / ((a / (a + c)) + p1))
                    if (a + c) and p1 == p1 and ((a / (a + c)) + p1) > 0
                    else np.nan
                ),
                risk_ratio=rr,
                odds_ratio=or_,
                or_95ci_lo=ci_lo,
                or_95ci_hi=ci_hi,
                nonhit_capture_frac=(c / n_nonhit) if n_nonhit else np.nan,
                n_hit_total=n_hit,
                n_nonhit_total=n_nonhit,
                n_hit_in=a,
                n_nonhit_in=c,
            )
        )
    return pd.DataFrame(rows)


# --------------------------- Plot helpers ---------------------------
def overlay_pr(
    ax,
    A: pd.DataFrame,
    B: pd.DataFrame,
    C: pd.DataFrame,
    labA: str,
    labB: str,
    labC: str,
    title: str,
):
    if not A.empty:
        ax.plot(A["recall"], A["precision"], marker="o", linestyle="-", label=labA)
    if not B.empty:
        ax.plot(B["recall"], B["precision"], marker="s", linestyle="--", label=labB)
    if not C.empty:
        ax.plot(C["recall"], C["precision"], marker="^", linestyle=":", label=labC)
    ax.set_xlabel("Recall (hit coverage)")
    ax.set_ylabel("Precision (hit fraction in set)")
    ax.set_title(title)
    ax.legend()


def overlay_pareto(
    ax,
    A: pd.DataFrame,
    B: pd.DataFrame,
    C: pd.DataFrame,
    labA: str,
    labB: str,
    labC: str,
    title: str,
):
    if not A.empty:
        ax.plot(
            A["nonhit_capture_frac"], A["recall"], marker="o", linestyle="-", label=labA
        )
    if not B.empty:
        ax.plot(
            B["nonhit_capture_frac"],
            B["recall"],
            marker="s",
            linestyle="--",
            label=labB,
        )
    if not C.empty:
        ax.plot(
            C["nonhit_capture_frac"], C["recall"], marker="^", linestyle=":", label=labC
        )
    ax.set_xlabel("Non-hit capture fraction (in set)")
    ax.set_ylabel("Hit coverage (in set)")
    ax.set_title(title)
    ax.legend()


def heatmap_family_grid(
    df_long: pd.DataFrame,
    profile: str,
    family: str,
    scenarios: List[str],
    outpath: Path,
    value_col: str = "hit_rate",
):
    """
    Plot scenario (rows) × polymer_mods (cols) heatmap per family.
    value_col: 'hit_rate' (bounded [0,1]) or e.g. 'log2_or' (unbounded).
    """
    sub = df_long[
        (df_long["profile"] == profile) & (df_long["family"] == family)
    ].copy()
    if sub.empty:
        return
    # Order columns by total n descending to emphasize populated mods
    mod_order = (
        sub.groupby("polymer_mods")["n"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Limit to top 30 mods to keep width manageable
    mod_order = mod_order[:30]
    sub = sub[sub["polymer_mods"].isin(mod_order)]
    # Ensure scenario order
    sub["scenario"] = pd.Categorical(
        sub["scenario"], categories=scenarios, ordered=True
    )
    piv = sub.pivot(index="scenario", columns="polymer_mods", values=value_col)
    # Fill NaN with 0 for visualization
    M = piv.fillna(0).to_numpy(float)
    fig = plt.figure(figsize=(max(8, 0.35 * M.shape[1]), 2 + 0.7 * M.shape[0]))
    ax = plt.gca()
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(x) for x in piv.index])
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(x) for x in piv.columns], rotation=60, ha="right")
    ax.set_title(f"{family} – {profile} – {value_col}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    H.save_plot(fig, outpath)


# --------------------------- Enrichment grid ---------------------------
def grid_enrichment(
    dj: pd.DataFrame,
    hit_col: str,
    profile: str,
    scenario: str,
) -> pd.DataFrame:
    """
    Return long-format per-cell enrichment for:
      cell = (profile, scenario, family, polymer_mods)
    Metrics: n, n_hit, hit_rate, risk_ratio, odds_ratio, CI, log2_or.
    """
    cols_needed = ["family", "polymer_mods", hit_col]
    if any(c not in dj.columns for c in cols_needed):
        return pd.DataFrame()

    rows = []
    # work per family × polymer_mods cell
    all_n = len(dj)
    all_hit = int(
        pd.to_numeric(dj[hit_col], errors="coerce").fillna(0).astype(int).sum()
    )
    all_non = all_n - all_hit

    for (fam, pm), sub in dj.groupby(["family", "polymer_mods"], dropna=True):
        n = len(sub)
        if n == 0:
            continue
        a = int(
            pd.to_numeric(sub[hit_col], errors="coerce").fillna(0).astype(int).sum()
        )
        b = n - a
        # outside cell
        c = all_hit - a
        d = all_non - b

        # hit rate
        hr = a / n if n else np.nan

        # risk ratio (cell vs outside)
        p1 = a / n if n else np.nan
        p2 = (c / (c + d)) if (c + d) else np.nan
        rr = (p1 / p2) if (p1 == p1 and p2 and p2 > 0) else np.nan

        # odds ratio with Haldane-Anscombe correction
        a1, b1, c1, d1 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_ = (a1 * d1) / (b1 * c1)
        se = np.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
        ci_lo = float(np.exp(np.log(or_) - 1.96 * se))
        ci_hi = float(np.exp(np.log(or_) + 1.96 * se))

        rows.append(
            dict(
                profile=profile,
                scenario=scenario,
                family=fam,
                polymer_mods=pm,
                n=n,
                n_hit=a,
                hit_rate=hr,
                risk_ratio=rr,
                odds_ratio=or_,
                or_95ci_lo=ci_lo,
                or_95ci_hi=ci_hi,
                log2_or=np.log2(or_) if or_ > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows)


# --------------------------- Main ---------------------------
def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse lists/ranges; basic validation via helper
    args.families = H.parse_csv_list_of_str(args.families)
    args.endgroups = H.parse_csv_list_of_str(args.endgroups)
    args.adducts = H.parse_csv_list_of_str(args.adducts)
    args.peptide_charges = H.parse_csv_list_of_ints(args.peptide_charges)
    args.peg_charges = H.parse_csv_list_of_ints(args.peg_charges)
    args.n_range = tuple(int(x) for x in args.n_range.split(","))
    args.mz_range = tuple(float(x) for x in args.mz_range.split(","))
    thr_ppm = parse_piecewise(args.ppm_thresholds)
    thr_da = parse_piecewise(args.da_thresholds)

    class _A:
        pass

    a = _A()
    a.q1_fwhm = 0.7
    a.ppm_window = 10.0
    a.peptide_charges = args.peptide_charges
    a.peg_charges = args.peg_charges
    a.n_range = args.n_range
    a.mz_range = args.mz_range
    a.families = args.families
    a.endgroups = args.endgroups
    a.adducts = args.adducts
    H.validate_config(a)

    # IO
    in_path = Path(args.in_csv)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(in_path.stem + "__triple_q1q3_compare")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    (outdir / "plots" / "family").mkdir(exist_ok=True)
    (outdir / "plots" / "family_grids").mkdir(exist_ok=True)

    # Load input
    df_in = pd.read_csv(in_path)
    if args.pepcol not in df_in.columns:
        raise ValueError(f"Missing peptide column '{args.pepcol}'.")
    if args.hitcol not in df_in.columns:
        raise ValueError(f"Missing hit column '{args.hitcol}'.")
    df_in["_is_hit_coerced"] = df_in[args.hitcol].apply(H.coerce_is_hit).astype("Int64")
    df_in = df_in.drop(columns=[args.hitcol], errors="ignore").rename(
        columns={"_is_hit_coerced": args.hitcol}
    )
    df_in["_valid_pep"] = df_in[args.pepcol].apply(H.aa_only)
    peptides = sorted(set(df_in.loc[df_in["_valid_pep"], args.pepcol].astype(str)))

    inv = df_in[~df_in["_valid_pep"]]
    if not inv.empty:
        inv.to_csv(outdir / "invalid_peptides.csv", index=False)
        logger.warning(f"Invalid peptides saved (n={len(inv)})")

    # Candidates once
    base_candidates = H.build_family_candidates(
        families=args.families,
        peg_charges=args.peg_charges,
        allowed_adducts=args.adducts,
        endgroups=args.endgroups,
        n_min=args.n_range[0],
        n_max=args.n_range[1],
        mz_min=args.mz_range[0],
        mz_max=args.mz_range[1],
        show_progress=not args.no_progress,
    )

    # Score all (profile × scenario)
    results = {}  # (profile, scenario) -> {"df_best","dj"}
    for prof in PROFILES:
        for scen in SCENARIOS:
            res = score_profile_scenario(
                prof,
                scen,
                peptides,
                tuple(args.peptide_charges),
                base_candidates,
                df_in,
                args.pepcol,
                args.hitcol,
                args.preselect,
                progress=not args.no_progress,
                outdir=outdir,
            )
            results[(prof.name, scen.name)] = res

    # -------------------- Dense threshold overlays --------------------
    # Overall overlays using SCENARIO="none" (baseline, no uncertain mods)
    def thr_for(prof_name: str, scen_name: str) -> pd.DataFrame:
        dj = results[(prof_name, scen_name)]["dj"]
        units = next(p.q3[0] for p in PROFILES if p.name == prof_name)
        if units == "da":
            t = threshold_summary_on_metric(
                dj, hit_col=args.hitcol, value_col="abs_delta_da", thresholds=thr_da
            )
            t.to_csv(
                outdir / f"thresholds_Q3_{prof_name}_{scen_name}_da.csv", index=False
            )
            return t
        else:
            t = H.threshold_summary_ppm(dj, hit_col=args.hitcol, thresholds=thr_ppm)
            t.to_csv(
                outdir / f"thresholds_Q3_{prof_name}_{scen_name}_ppm.csv", index=False
            )
            return t

    T_loose = thr_for("loose", "none")
    T_medium = thr_for("medium", "none")
    T_strict = thr_for("strict", "none")

    if not args.no_plots:
        fig = plt.figure()
        ax = plt.gca()
        overlay_pr(
            ax,
            T_loose,
            T_medium,
            T_strict,
            "Loose (Q1=0.4 Da; Q3=0.1 Da)",
            "Medium (Q1=100 ppm; Q3=30 ppm)",
            "Strict (Q1=30 ppm; Q3=10 ppm)",
            "PR by Q3 threshold (scenario: none)",
        )
        H.save_plot(fig, outdir / "plots" / "overlay_PR_Q3_overall.png")

        fig = plt.figure()
        ax = plt.gca()
        overlay_pareto(
            ax,
            T_loose,
            T_medium,
            T_strict,
            "Loose (Q1=0.4 Da; Q3=0.1 Da)",
            "Medium (Q1=100 ppm; Q3=30 ppm)",
            "Strict (Q1=30 ppm; Q3=10 ppm)",
            "Pareto by Q3 threshold (scenario: none)",
        )
        H.save_plot(fig, outdir / "plots" / "overlay_Pareto_Q3_overall.png")

    # -------------------- Per-family overlays (scenario: none) --------------------
    fams_present = []
    for prof in PROFILES:
        fams_present.extend(
            results[(prof.name, "none")]["dj"]["family"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    fams_present = sorted(sorted(set(fams_present)))

    for fam in fams_present:

        def fam_thr(prof_name: str) -> pd.DataFrame:
            dj = results[(prof_name, "none")]["dj"]
            sub = dj[dj["family"] == fam]
            if sub.empty:
                return pd.DataFrame()
            units = next(p.q3[0] for p in PROFILES if p.name == prof_name)
            if units == "da":
                return threshold_summary_on_metric(
                    sub,
                    hit_col=args.hitcol,
                    value_col="abs_delta_da",
                    thresholds=thr_da,
                )
            else:
                return H.threshold_summary_ppm(
                    sub, hit_col=args.hitcol, thresholds=thr_ppm
                )

        L = fam_thr("loose")
        M = fam_thr("medium")
        S = fam_thr("strict")

        if not args.no_plots:
            fig = plt.figure()
            ax = plt.gca()
            overlay_pr(
                ax,
                L,
                M,
                S,
                "Loose",
                "Medium",
                "Strict",
                f"PR by Q3 threshold — {fam} (scenario: none)",
            )
            H.save_plot(fig, outdir / "plots" / "family" / f"overlay_PR_Q3_{fam}.png")

            fig = plt.figure()
            ax = plt.gca()
            overlay_pareto(
                ax,
                L,
                M,
                S,
                "Loose",
                "Medium",
                "Strict",
                f"Pareto by Q3 threshold — {fam} (scenario: none)",
            )
            H.save_plot(
                fig, outdir / "plots" / "family" / f"overlay_Pareto_Q3_{fam}.png"
            )

    # -------------------- Grid of peptide-mod × family × polymer_mods --------------------
    grids = []
    for prof in PROFILES:
        for scen in SCENARIOS:
            dj = results[(prof.name, scen.name)]["dj"]
            g = grid_enrichment(
                dj, hit_col=args.hitcol, profile=prof.name, scenario=scen.name
            )
            if not g.empty:
                grids.append(g)
    grid_long = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    if not grid_long.empty:
        grid_long.to_csv(outdir / "grid_enrichment_long.csv", index=False)

        # Per-family heatmaps (top 30 polymer_mods columns by total n)
        if not args.no_plots:
            scen_order = [s.name for s in SCENARIOS]
            for fam in sorted(grid_long["family"].dropna().unique()):
                for prof in ["loose", "medium", "strict"]:
                    heatmap_family_grid(
                        grid_long,
                        profile=prof,
                        family=fam,
                        scenarios=scen_order,
                        outpath=outdir
                        / "plots"
                        / "family_grids"
                        / f"heatmap_{prof}_{fam}.png",
                        value_col="hit_rate",  # change to "log2_or" if you prefer enrichment contrast
                    )

    # Done
    # Summary log
    for key, res in results.items():
        prof, scen = key
        dj = res["dj"]
        total_rows = len(dj)
        n_hits = int(
            pd.to_numeric(dj[args.hitcol], errors="coerce").fillna(0).astype(int).sum()
        )
        logger.info(
            f"[{prof}/{scen}] joined rows={total_rows}, hits={n_hits} ({(n_hits/total_rows*100 if total_rows else 0):.1f}%)"
        )

    logger.info("All outputs written.")


if __name__ == "__main__":
    main()
