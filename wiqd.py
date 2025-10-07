#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WIQD — What Is Qing Doing?
Hit vs non-hit peptide analysis with rigorous stats and plots.

This version streamlines albumin metrics and proline/synthesis flags:

Albumin features
  • RETAINED (counts only; no fractions):
      - alb_candidates_total, alb_candidates_rt
      - alb_precursor_mz_match_count, alb_precursor_mz_match_count_rt
      - alb_frag_candidates_mz, alb_frag_candidates_mz_rt
      - alb_frag_mz_match_count, alb_frag_mz_match_count_rt
      - Best precursor metrics (for context):
          * alb_precursor_mz_best_ppm, alb_precursor_mz_best_da,
            alb_precursor_mz_best_len, alb_precursor_mz_best_charge,
            alb_confusability_precursor_mz (ppm→score)
  • REMOVED:
      - All albumin sequence matching metrics (alb_frag_seq_*)
      - Any albumin *_frac fields
      - Composite albumin confusability by sequence/mz/overall

Fragmentation flags
  • Replace the 11 proline-related presence flags with:
      - frag_dbl_proline_after (any X–P among AP,SP,TP,GP,DP,EP,KP,RP,PP)
      - frag_dbl_proline_before (any P–K or P–R; PP sets both)
  • Add a single numeric:
      - hard_to_synthesize_residues = total count of specified synthesis-risk doublets

Other changes
  • Boxplot deprecation fix: use tick_labels (Matplotlib ≥3.9)
  • Show nnz in plot annotations; means at 3 sig figs (SD removed)
  • Summary Top‑N gate: --min_nnz_topn (default 4)
  • Default permutations: 5,000 (faster)
  • Default cysteine fixed mod: carbamidomethyl (+57.021464), realistic for HPLC‑MS
  • Bug fixes: flyability clamping; pI formula for glutamate
  • “Any fragment” logic for albumin m/z stays (no b/y orientation); RT gating supported;
    optional precursor m/z gating for fragments via --require_precursor_match_for_frag.

Refactor notes
  • Centralize utilities in:
      - wiqd_features (mass/mz/GRAVY/hydrophobicity, composition, pI/charge, etc.)
      - wiqd_fly_score (flyability components & mix)
      - wiqd_sequence_helpers (peptide cleaning; I/L collapse; collapsed alphabets)
      - wiqd_stats (MW U, perm-U, Fisher, FDR, effect sizes, fmt_p/fmt_q)
      - wiqd_proteins (albumin & housekeeping sequence I/O)
      - wiqd_motifs (doublet motif sets)
"""

import os, math, argparse, datetime, random, re, bisect
from typing import List, Dict, Optional, Iterable
from dataclasses import dataclass

import pandas as pd

# =================== tqdm ===================
try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(it, **kwargs):
        return it


# =================== Matplotlib ===================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== Matplotlib defaults ===================
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.figsize": (6.8, 4.4),
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

# =================== Shared helpers & constants ===================
from wiqd_constants import (
    MONO,
    AA,
    EISEN,
    HOPP,
)

from wiqd_util import banner

from wiqd_sequence_helpers import (
    clean_pep as clean_peptide,
    collapse_xle,
    collapse_seq_mode,
)

from wiqd_features import (
    # masses & m/z
    mass_neutral,
    mz_from_mass,
    mass_monoisotopic,
    # hydrophobicity & composition
    kd_gravy,
    hydrophobic_fraction,
    mean_scale,
    kd_stdev,
    max_hydro_streak,
    # charge/pI
    isoelectric_point,
    net_charge_at_pH,
    # composition/physchem
    aliphatic_index,
    aromaticity,
    cterm_hydrophobic,
    tryptic_end,
    has_RP_or_KP,
    xle_fraction,
    aa_fraction,
    hbd_hba_counts,
    elemental_counts,
    approx_M1_rel_abundance,
    count_basic,
    count_acidic,
    basicity_proxy,
)

from wiqd_fly_score import (
    compute_flyability_components,
    combine_flyability_score,
)

from wiqd_esi import (
    es_chargeability_score,
)

from wiqd_rt_proxy import predict_rt_min as RT

from wiqd_motifs import FRAG_DOUBLETS, SYN_RISK_DOUBLETS

from wiqd_stats import (
    mannwhitney_u_p,
    mannwhitney_U_only,
    perm_pvalue_U,
    cliffs_delta,
    cohens_d_and_g,
    fisher_exact,
    bh_fdr,
    fmt_p,
    fmt_q,
)

from wiqd_proteins import (
    load_albumin_sequence,
    load_housekeeping_sequences,
)


# =================== Small plot helper ===================
def _apply_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


# =================== Fragment confusability (independent of albumin) =========
DELTA_KQ = abs(MONO["K"] - MONO["Q"])
DELTA_F_Mox = abs(MONO["F"] - (MONO["M"] + 15.994915))
DELTA_ND = abs(MONO["N"] - MONO["D"])
DELTA_QE = abs(MONO["Q"] - MONO["E"])
DELTA_LI = 0.0
AMBIG_CLASSES = {
    "LI": ({"L", "I"}, DELTA_LI),
    "KQ": ({"K", "Q"}, DELTA_KQ),
    "FMox": ({"F", "M"}, DELTA_F_Mox),
    "ND": ({"N", "D"}, DELTA_ND),
    "QE": ({"Q", "E"}, DELTA_QE),
}


def fragment_confusability_features(
    pep: str, frag_tol_da: float = 0.02, frag_tol_ppm: float = None
):
    L = len(pep)
    if L < 2:
        return {
            "confusable_bion_frac": float("nan"),
            "confusable_yion_frac": float("nan"),
            "confusable_ion_frac": float("nan"),
        }
    prefix = [0.0] * (L + 1)
    for i, ch in enumerate(pep, start=1):
        prefix[i] = prefix[i - 1] + MONO[ch]

    def tol_at(mass):
        return (
            (frag_tol_ppm * mass * 1e-6) if (frag_tol_ppm is not None) else frag_tol_da
        )

    present = {
        key
        for key, (members, _) in AMBIG_CLASSES.items()
        if any((aa in members) for aa in pep)
    }
    if not present:
        return {
            "confusable_bion_frac": 0.0,
            "confusable_yion_frac": 0.0,
            "confusable_ion_frac": 0.0,
        }
    conf_b = conf_y = 0
    total = L - 1
    for i in range(1, L):
        b_mass = prefix[i]
        y_mass = prefix[L] - prefix[i]
        informative_b = informative_y = False
        for key in present:
            members, delta = AMBIG_CLASSES[key]
            if any((aa in members) for aa in pep[:i]):
                if delta > tol_at(b_mass):
                    informative_b = True
            if any((aa in members) for aa in pep[i:]):
                if delta > tol_at(y_mass):
                    informative_y = True
            if informative_b and informative_y:
                break
        if not informative_b:
            conf_b += 1
        if not informative_y:
            conf_y += 1
    frac_b = conf_b / total
    frac_y = conf_y / total
    return {
        "confusable_bion_frac": frac_b,
        "confusable_yion_frac": frac_y,
        "confusable_ion_frac": 0.5 * (frac_b + frac_y),
    }


# =================== Albumin window index & helpers ===================
@dataclass
class AlbWindow:
    seq: str
    start0: int
    end0: int
    length: int
    mass: float
    mz_by_z: Dict[int, float]
    rt_min: float
    collapsed: str


def _calc_neutral_mass_with_cys(seq: str, cys_fixed_mod: float) -> float:
    return mass_neutral(seq, cys_fixed_mod)


def _ppm(a: float, b: float) -> float:
    return abs(a - b) / a * 1e6 if a > 0 else float("inf")


def build_albumin_windows_index(
    albumin: str,
    len_min: int,
    len_max: int,
    cys_fixed_mod: float,
    charges: Iterable[int],
    gradient_min: float,
    collapse: str,
) -> List[AlbWindow]:
    windows: List[AlbWindow] = []
    N = len(albumin)
    len_min = max(1, int(len_min))
    len_max = max(len_min, int(len_max))
    charges = list(sorted(set(int(z) for z in charges)))
    for L in range(len_min, min(N, len_max) + 1):
        for s in range(0, N - L + 1):
            sub = albumin[s : s + L]
            m = _calc_neutral_mass_with_cys(sub, cys_fixed_mod)
            mz_by_z = {z: mz_from_mass(m, z) for z in charges}
            rt = RT(sub, gradient_min=gradient_min)
            windows.append(
                AlbWindow(
                    seq=sub,
                    start0=s,
                    end0=s + L,
                    length=L,
                    mass=m,
                    mz_by_z=mz_by_z,
                    rt_min=rt,
                    collapsed=collapse_seq_mode(sub, collapse),
                )
            )
    return windows


def _filter_rt(windows: List[AlbWindow], rt: float, tol: float) -> List[AlbWindow]:
    return [w for w in windows if abs(w.rt_min - rt) <= tol]


def _filter_precursor_mz(
    windows: List[AlbWindow],
    pep_mz_by_z: Dict[int, float],
    ppm_tol: float,
) -> List[AlbWindow]:
    kept = []
    for w in windows:
        ok = False
        for z, pmz in pep_mz_by_z.items():
            wmz = w.mz_by_z.get(z)
            if wmz is None:
                continue
            if _ppm(pmz, wmz) <= ppm_tol:
                ok = True
                break
        if ok:
            kept.append(w)
    return kept


def _count_precursor_mz_matches(
    windows: List[AlbWindow],
    pep_mz_by_z: Dict[int, float],
    ppm_tol: float,
) -> int:
    cnt = 0
    for w in windows:
        ok = False
        for z, pmz in pep_mz_by_z.items():
            wmz = w.mz_by_z.get(z)
            if wmz is None:
                continue
            if _ppm(pmz, wmz) <= ppm_tol:
                ok = True
                break
        if ok:
            cnt += 1
    return cnt


def ppm_to_score(ppm_val: float, ppm_tol: float) -> float:
    if not isinstance(ppm_val, float) or ppm_val != ppm_val:
        return 0.0
    return max(0.0, 1.0 - min(ppm_val, ppm_tol) / ppm_tol)


# =================== ANY-fragment (m/z only) matching ===================
def _build_peptide_anyfrag_mz_cache(
    pep: str,
    charges: Iterable[int],
    cys_fixed_mod: float,
    kmin: int,
    kmax: int,
):
    """Precompute peptide ANY contiguous fragments (all positions) m/z lists by charge."""
    Lp = len(pep)
    kmin = max(1, int(kmin))
    kmax = min(int(kmax), max(0, Lp))
    mz_lists_by_z: Dict[int, List[float]] = {int(z): [] for z in charges}
    for k in range(kmin, kmax + 1):
        for i in range(0, Lp - k + 1):
            sub = pep[i : i + k]
            m = mass_neutral(sub, cys_fixed_mod)
            for z in mz_lists_by_z.keys():
                mz = mz_from_mass(m, z)
                mz_lists_by_z[z].append(mz)
    for z in mz_lists_by_z.keys():
        mz_lists_by_z[z].sort()
    return {"mz_lists_by_z": mz_lists_by_z}


def _any_within_ppm(x: float, sorted_list: List[float], ppm_tol: float) -> bool:
    if not sorted_list:
        return False
    tol = x * ppm_tol * 1e-6
    i = bisect.bisect_left(sorted_list, x)
    if i < len(sorted_list) and abs(sorted_list[i] - x) <= tol:
        return True
    if i > 0 and abs(sorted_list[i - 1] - x) <= tol:
        return True
    if i + 1 < len(sorted_list) and abs(sorted_list[i + 1] - x) <= tol:
        return True
    return False


def _fragment_any_mz_match_counts_over(
    matched_windows: List[AlbWindow],
    pep_frag_mz: dict,
    charges: Iterable[int],
    cys_fixed_mod: float,
    ppm_tol: float,
):
    """
    Count ANY-fragment m/z matches between albumin windows and the peptide:
      - Consider all contiguous subsequences (k in configured range) of albumin windows
      - A match occurs if the sub-fragment m/z is within ppm_tol to ANY peptide fragment m/z
        at ANY charge in `charges`.
    Returns: {"cand": total_fragments_considered, "mz_count": matched_fragments}
    """
    mz_lists_by_z = pep_frag_mz["mz_lists_by_z"]
    cand = 0
    mz_count = 0
    for w in matched_windows:
        Lw = w.length
        for k in range(1, Lw + 1):
            for i in range(0, Lw - k + 1):
                sub = w.seq[i : i + k]
                cand += 1
                m = mass_neutral(sub, cys_fixed_mod)
                hit = False
                for z in mz_lists_by_z.keys():
                    mz = mz_from_mass(m, z)
                    if _any_within_ppm(mz, mz_lists_by_z[z], ppm_tol):
                        hit = True
                        break
                if hit:
                    mz_count += 1
    return {"cand": cand, "mz_count": mz_count}


# =================== Homology (housekeeping k-mer bank) ===================
def build_kmer_bank(seqs: Dict[str, str], k: int = 6, collapse: str = "xle"):
    bank = set()
    for name, seq in tqdm(
        seqs.items(), desc=f"[4/8] Building hk k-mer bank (k={k}, collapse={collapse})"
    ):
        s = collapse_seq_mode(seq, collapse)
        for i in range(0, len(s) - k + 1):
            bank.add(s[i : i + k])
    return bank


def homology_features_with_params(
    peptide: str, kmer_bank: set, k: int, collapse: str = "xle"
):
    if len(peptide) < k or not kmer_bank:
        return {"hk_kmer_hits": 0, "hk_kmer_frac": 0.0}
    collapsed = collapse_seq_mode(peptide, collapse)
    kms = [collapsed[i : i + k] for i in range(0, len(collapsed) - k + 1)]
    hits = sum(1 for km in kms if km in kmer_bank)
    return {"hk_kmer_hits": hits, "hk_kmer_frac": hits / max(1, len(kms))}


# =================== Doublet accounting (frag & synthesis) ===================
def _count_doublets(seq: str):
    counts = {}
    for i in range(len(seq) - 1):
        di = seq[i : i + 2]
        counts[di] = counts.get(di, 0) + 1
    return counts


def doublet_features(peptide: str):
    L = len(peptide)
    bonds = max(1, L - 1)
    counts = _count_doublets(peptide)
    out = {}
    # Fragmentation doublets (keep counts/fracs; collapse presence → 2 flags)
    frag_total = 0
    for di in FRAG_DOUBLETS:
        c = counts.get(di, 0)
        frag_total += c
        out[f"frag_dbl_{di}_count"] = float(c)
        out[f"frag_dbl_{di}_frac"] = float(c) / bonds
    out["frag_doublet_total_count"] = float(frag_total)
    out["frag_doublet_total_frac"] = float(frag_total) / bonds
    # Proline aggregation (binary presence)
    proline_after_set = {
        "AP",
        "SP",
        "TP",
        "GP",
        "DP",
        "EP",
        "KP",
        "RP",
        "PP",
        "VP",
        "HP",
        "IP",
        "LP",
    }
    proline_before_set = {"PK", "PR", "PP"}  # PP counts for both
    out["frag_dbl_proline_after"] = (
        1 if any(counts.get(di, 0) > 0 for di in proline_after_set) else 0
    )
    out["frag_dbl_proline_before"] = (
        1 if any(counts.get(di, 0) > 0 for di in proline_before_set) else 0
    )
    # Synthesis-risk doublets (counts/fracs + consolidated metric)
    syn_total = 0
    for di in SYN_RISK_DOUBLETS:
        c = counts.get(di, 0)
        syn_total += c
        out[f"synth_dbl_{di}_count"] = float(c)
        out[f"synth_dbl_{di}_frac"] = float(c) / bonds
    out["synth_doublet_total_count"] = float(syn_total)
    out["synth_doublet_total_frac"] = float(syn_total) / bonds
    out["hard_to_synthesize_residues"] = float(syn_total)
    return out


# =================== Pretty names & formatting ===================
def pretty_feature_name(feat: str) -> str:
    if feat.startswith("frag_dbl_") and feat.endswith("_count"):
        core = feat[len("frag_dbl_") : -len("_count")]
        return f"Fragmentation doublet {core} (count)"
    if feat.startswith("frag_dbl_") and feat.endswith("_frac"):
        core = feat[len("frag_dbl_") : -len("_frac")]
        return f"Fragmentation doublet {core} (fraction of bonds)"
    if feat == "frag_doublet_total_count":
        return "Fragmentation doublets (total count)"
    if feat == "frag_doublet_total_frac":
        return "Fragmentation doublets (fraction of bonds)"
    if feat == "frag_dbl_proline_after":
        return "Proline after (X–P) present"
    if feat == "frag_dbl_proline_before":
        return "Proline before (P–K/R) present"

    if feat.startswith("synth_dbl_") and feat.endswith("_count"):
        core = feat[len("synth_dbl_") : -len("_count")]
        return f"Synthesis-risk doublet {core} (count)"
    if feat.startswith("synth_dbl_") and feat.endswith("_frac"):
        core = feat[len("synth_dbl_") : -len("_frac")]
        return f"Synthesis-risk doublet {core} (fraction of bonds)"
    if feat == "synth_doublet_total_count":
        return "Synthesis-risk doublets (total count)"
    if feat == "synth_doublet_total_frac":
        return "Synthesis-risk doublets (fraction of bonds)"
    if feat == "hard_to_synthesize_residues":
        return "Hard-to-synthesize doublets (count)"

    special = {
        "pI": "pI",
        "gravy": "GRAVY (Kyte–Doolittle)",
        "eisenberg_hydro": "Eisenberg hydrophobicity",
        "hopp_woods_hydrophilicity": "Hopp–Woods hydrophilicity",
        "kd_stdev": "Hydrophobicity stdev (KD)",
        "hydrophobic_fraction": "Hydrophobic fraction",
        "max_hydrophobic_run": "Max hydrophobic run length",
        "charge_pH2_0": "Net charge @ pH 2.0",
        "charge_pH2_7": "Net charge @ pH 2.7",
        "charge_pH7_4": "Net charge @ pH 7.4",
        "charge_pH10_0": "Net charge @ pH 10.0",
        "aliphatic_index": "Aliphatic index",
        "aromaticity": "Aromaticity fraction",
        "xle_fraction": "I/L fraction",
        "basic_count": "# basic residues (K/R/H)",
        "acidic_count": "# acidic residues (D/E)",
        "basicity_proxy": "Gas-phase basicity (heuristic)",
        "HBD_sidechain": "# H-bond donors (side-chain)",
        "HBA_sidechain": "# H-bond acceptors (side-chain)",
        "elem_C": "C atoms",
        "elem_H": "H atoms",
        "elem_N": "N atoms",
        "elem_O": "O atoms",
        "elem_S": "S atoms",
        "approx_Mplus1_rel": "Approx. M+1 relative abundance",
        "mz_z1": "m/z (z=1)",
        "mz_z2": "m/z (z=2)",
        "mz_z3": "m/z (z=3)",
        "mass_defect": "Mass defect",
        "hk_kmer_hits": "Housekeeping k-mer hits",
        "hk_kmer_frac": "Housekeeping k-mer fraction",
        "mass_mono": "Monoisotopic mass (Da)",
        "length": "Peptide length",
        "frac_L_or_I": "Fraction of L or I (isobaric 113.084)",
        "frac_K_or_Q": "Fraction of K or Q (~36 mDa apart)",
        "frac_F_or_Mox": "Fraction of F or oxidizable M (F vs M+O)",
        "frac_N_or_D": "Fraction of N or D (deamidation)",
        "frac_Q_or_E": "Fraction of Q or E (deamidation)",
        "frac_ambiguous_union": "Fraction of ambiguous AAs (union of pairs)",
        "has_L_or_I": "Has L or I (isobaric)",
        "has_K_or_Q": "Has K or Q (~36 mDa)",
        "has_F_or_Mox": "Has F or M (oxidizable)",
        "has_N_or_D": "Has N or D (deamidation)",
        "has_Q_or_E": "Has Q or E (deamidation)",
        "has_ambiguous_union": "Has any ambiguous AA",
        "confusable_bion_frac": "Confusable b-ions (fraction)",
        "confusable_yion_frac": "Confusable y-ions (fraction)",
        "confusable_ion_frac": "Confusable ions overall (fraction)",
        "fly_score": "Flyability score",
        "fly_charge_norm": "Fly: charge norm",
        "fly_surface_norm": "Fly: surface norm",
        "fly_aromatic_norm": "Fly: aromatic norm",
        "fly_len_norm": "Fly: length norm",
        "rt_pred_min": "Predicted RT (min)",
        "es_chargeability": "ESI chargeability score",
        # Albumin (counts only)
        "alb_candidates_total": "Albumin candidate windows (total)",
        "alb_candidates_rt": "Albumin candidate windows (RT‑gated)",
        "alb_precursor_mz_match_count": "# albumin windows matching precursor m/z",
        "alb_precursor_mz_match_count_rt": "# windows matching precursor m/z (RT‑gated)",
        "alb_frag_candidates_mz": "Albumin fragments considered (m/z)",
        "alb_frag_mz_match_count": "Albumin fragments m/z‑match (count)",
        "alb_frag_candidates_mz_rt": "Albumin fragments considered (m/z, RT‑gated)",
        "alb_frag_mz_match_count_rt": "Albumin fragments m/z‑match (count, RT‑gated)",
        "alb_precursor_mz_best_ppm": "Precursor vs albumin 5–15mer best ppm",
        "alb_precursor_mz_best_da": "Precursor vs albumin 5–15mer best Da",
        "alb_precursor_mz_best_len": "Precursor vs albumin best length (5–15)",
        "alb_precursor_mz_best_charge": "Precursor vs albumin best charge",
        "alb_confusability_precursor_mz": "Albumin confusability (precursor m/z score)",
    }
    if feat in special:
        return special[feat]
    if feat.startswith("frac_"):
        return f"Fraction of {feat.split('_',1)[1]}"
    return feat.replace("_", " ")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)


# =================== Plot helpers ===================
def save_boxplot(x, y, feature, pval, qval, outpath):
    fig, ax = plt.subplots(constrained_layout=True)
    # Matplotlib 3.9 deprecation: use tick_labels
    bp = ax.boxplot(
        [x, y], tick_labels=["non-hit", "hit"], showmeans=True, meanline=True
    )
    jitter = 0.08
    xs0 = [1 + (random.random() - 0.5) * 2 * jitter for _ in x]
    xs1 = [2 + (random.random() - 0.5) * 2 * jitter for _ in y]
    ax.scatter(xs0, x, alpha=0.4, s=12)
    ax.scatter(xs1, y, alpha=0.4, s=12)
    ax.set_title(pretty_feature_name(feature))
    ax.set_ylabel(pretty_feature_name(feature))
    n0, n1 = len(x), len(y)
    nnz0 = sum(1 for v in x if (isinstance(v, (int, float)) and v != 0))
    nnz1 = sum(1 for v in y if (isinstance(v, (int, float)) and v != 0))
    m0 = (sum(x) / n0) if n0 else float("nan")
    m1 = (sum(y) / n1) if n1 else float("nan")
    text = f"MW/perm(U) p={fmt_p(pval)}, q={fmt_q(qval)} | n={n0}/{n1} | nnz={nnz0}/{nnz1} | μ: {m0:.3g} vs {m1:.3g}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    try:
        for med in bp.get("medians", []):
            med.set_color("C1")
        for mean in bp.get("means", []):
            mean.set_color("C2")
        handles = (
            [bp["medians"][0], bp["means"][0]]
            if bp.get("medians") and bp.get("means")
            else []
        )
        labels = ["Median", "Mean"] if handles else []
        if handles:
            ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=9)
    except Exception:
        pass
    _apply_grid(ax)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_barplot_props(feature, a, b, c, d, pval, qval, outpath):
    n_hit = a + b
    n_non = c + d
    p_hit = a / max(1, n_hit)
    p_non = c / max(1, n_non)
    fig, ax = plt.subplots(constrained_layout=True)
    xs = [0, 1]
    ax.bar(xs, [p_non, p_hit])
    pts_non = [1] * c + [0] * d
    pts_hit = [1] * a + [0] * b
    ax.scatter(
        [0 + (random.random() - 0.5) * 0.16 for _ in pts_non], pts_non, alpha=0.4, s=12
    )
    ax.scatter(
        [1 + (random.random() - 0.5) * 0.16 for _ in pts_hit], pts_hit, alpha=0.4, s=12
    )
    ax.set_xticks(xs, ["non-hit", "hit"])
    ax.set_ylabel(f"Proportion with {pretty_feature_name(feature)}")
    ax.set_title(pretty_feature_name(feature))
    text = f"Fisher p={fmt_p(pval)}, q={fmt_q(qval)} | counts: {c}/{n_non} vs {a}/{n_hit} | nnz={c}/{a}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    ax.set_ylim(0, 1.05)
    _apply_grid(ax)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_grouped_bar(
    labels, vals_nonhit, vals_hit, title, ylabel, outpath, sig_mask=None
):
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.45), 4.4))
    idx = list(range(len(labels)))
    width = 0.4
    ax.bar([i - width / 2 for i in idx], vals_nonhit, width=width, label="non-hit")
    ax.bar([i + width / 2 for i in idx], vals_hit, width=width, label="hit")
    ax.set_xticks(idx, labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if sig_mask:
        for i, sig in enumerate(sig_mask):
            if sig:
                y = max(
                    vals_nonhit[i] if not math.isnan(vals_nonhit[i]) else 0.0,
                    vals_hit[i] if not math.isnan(vals_hit[i]) else 0.0,
                )
                ax.text(i, y * 1.02 + 1e-6, "*", ha="center", va="bottom")
    _apply_grid(ax)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def save_summary_plot(
    stats_df: pd.DataFrame,
    alpha: float,
    outpath: str,
    n_nonhit=None,
    n_hit=None,
    top_n: int = 20,
    min_nnz_topn: int = 4,
):
    def neglog10(x):
        try:
            x = float(x)
            if 0.0 < x <= 1.0:
                return -math.log10(x)
        except Exception:
            pass
        return 0.0

    df = stats_df.copy()
    # Filter by nnz threshold (both groups)
    if "nnz_nonhit" in df.columns and "nnz_hit" in df.columns:
        df = df[(df["nnz_nonhit"] >= min_nnz_topn) & (df["nnz_hit"] >= min_nnz_topn)]
    df["score"] = df.apply(
        lambda r: (
            neglog10(r["q_value"])
            if (isinstance(r["q_value"], float) and r["q_value"] > 0)
            else neglog10(r["p_value"])
        ),
        axis=1,
    )
    df = df[df["score"] > 0].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.set_title("Feature summary — no non-zero signal scores")
        ax.set_ylabel("Signal score")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No detectable differences\n(all q≥α or insufficient nnz)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        _apply_grid(ax)
        plt.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
        return
    top = df.sort_values("score", ascending=False).head(
        int(top_n) if isinstance(top_n, (int, float)) else 20
    )
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.32), 5))
    ax.bar(range(len(top)), top["score"].tolist())
    labels = []
    for _, r in top.iterrows():
        if r["type"] == "numeric":
            mh = r.get("mean_hit", float("nan"))
            mn = r.get("mean_nonhit", float("nan"))
            arrow = (
                "↑"
                if (isinstance(mh, float) and isinstance(mn, float) and mh > mn)
                else (
                    "↓"
                    if (isinstance(mh, float) and isinstance(mn, float) and mh < mn)
                    else ""
                )
            )
        else:
            ph = r.get("prop_hit", float("nan"))
            pn = r.get("prop_nonhit", float("nan"))
            arrow = (
                "↑"
                if (isinstance(ph, float) and isinstance(pn, float) and ph > pn)
                else (
                    "↓"
                    if (isinstance(ph, float) and isinstance(pn, float) and ph < pn)
                    else ""
                )
            )
        labels.append(f"{r['feature']}{arrow}")
    ax.set_xticks(range(len(top)), labels, rotation=90)
    ax.set_ylabel("Signal score")
    title = "Feature significance summary"
    if n_nonhit is not None and n_hit is not None:
        title += f" | n(non-hit)={n_nonhit}, n(hit)={n_hit} | nnz≥{min_nnz_topn}"
    ax.set_title(title)
    if alpha and alpha > 0:
        cutoff = -math.log10(alpha)
        ax.axhline(cutoff, linestyle="--", linewidth=1.0)
        ax.text(0.0, cutoff * 1.02, f"q={alpha:g}", va="bottom")
    _apply_grid(ax)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# =================== Core feature computation ===================
def compute_features(
    peptides: List[str], fly_weights: Dict[str, float], args
) -> pd.DataFrame:
    rows = []
    for pep in tqdm(peptides, desc="[1/8] Computing core & flyability ..."):
        if not pep:
            continue
        Lp = len(pep)
        mass = mass_monoisotopic(
            pep
        )  # unmodified by design (albumin applies Cys mod internally)
        C, H, N, O, S = elemental_counts(pep)
        m1 = approx_M1_rel_abundance(C, H, N, O, S)
        hyd_frac = hydrophobic_fraction(pep)
        kd_std = kd_stdev(pep)
        hydro_run = max_hydro_streak(pep)
        # Ambiguous/isobaric fractions
        frac_L_or_I = (pep.count("L") + pep.count("I")) / Lp
        frac_K_or_Q = (pep.count("K") + pep.count("Q")) / Lp
        frac_F_or_Mox = (pep.count("F") + pep.count("M")) / Lp
        frac_N_or_D = (pep.count("N") + pep.count("D")) / Lp
        frac_Q_or_E = (pep.count("Q") + pep.count("E")) / Lp
        ambig_union_set = set("LIKQFMNDE")
        frac_ambiguous_union = sum(1 for a in pep if a in ambig_union_set) / Lp
        # Flyability
        fcomp = compute_flyability_components(pep)
        fly_score = combine_flyability_score(fcomp, fly_weights)
        donors, acceptors = hbd_hba_counts(pep)
        row = {
            "peptide": pep,
            "length": Lp,
            "mass_mono": mass,
            "gravy": kd_gravy(pep),
            "eisenberg_hydro": mean_scale(pep, EISEN),
            "hopp_woods_hydrophilicity": mean_scale(pep, HOPP),
            "kd_stdev": kd_std,
            "hydrophobic_fraction": hyd_frac,
            "max_hydrophobic_run": hydro_run,
            "pI": isoelectric_point(pep),
            "charge_pH2_0": net_charge_at_pH(pep, 2.0),
            "charge_pH2_7": net_charge_at_pH(pep, 2.7),
            "charge_pH7_4": net_charge_at_pH(pep, 7.4),
            "charge_pH10_0": net_charge_at_pH(pep, 10.0),
            "aliphatic_index": aliphatic_index(pep),
            "aromaticity": aromaticity(pep),
            "cterm_hydrophobic": cterm_hydrophobic(pep),
            "tryptic_end": tryptic_end(pep),
            "has_RP_or_KP": has_RP_or_KP(pep),
            "xle_fraction": xle_fraction(pep),
            "basic_count": count_basic(pep),
            "acidic_count": count_acidic(pep),
            "basicity_proxy": basicity_proxy(pep),
            "proline_count": pep.count("P"),
            "proline_internal": 1 if (("P" in pep[1:-1]) if Lp > 2 else False) else 0,
            "HBD_sidechain": donors,
            "HBA_sidechain": acceptors,
            "elem_C": C,
            "elem_H": H,
            "elem_N": N,
            "elem_O": O,
            "elem_S": S,
            "approx_Mplus1_rel": m1,
            "mz_z1": mz_from_mass(mass, 1),
            "mz_z2": mz_from_mass(mass, 2),
            "mz_z3": mz_from_mass(mass, 3),
            "mass_defect": mass - int(mass),
            "frac_L_or_I": frac_L_or_I,
            "frac_K_or_Q": frac_K_or_Q,
            "frac_F_or_Mox": frac_F_or_Mox,
            "frac_N_or_D": frac_N_or_D,
            "frac_Q_or_E": frac_Q_or_E,
            "frac_ambiguous_union": frac_ambiguous_union,
            "fly_charge_norm": fcomp["fly_charge_norm"],
            "fly_surface_norm": fcomp["fly_surface_norm"],
            "fly_aromatic_norm": fcomp["fly_aromatic_norm"],
            "fly_len_norm": fcomp["fly_len_norm"],
            "fly_score": fly_score,
            "rt_pred_min": RT(pep),
            "es_chargeability": es_chargeability_score(pep),
        }
        for a in sorted(AA):
            row[f"frac_{a}"] = aa_fraction(pep, a)
            row[f"has_{a}"] = int(a in pep)
            row[f"Nterm_{a}"] = int(pep[0] == a)
            row[f"Cterm_{a}"] = int(pep[-1] == a)

        # Doublet features (with proline collapse + hardness)
        row.update(doublet_features(pep))

        rows.append(row)
    return pd.DataFrame(rows)


# =================== Assumption checks ===================
def write_assumptions_report(
    outdir: str,
    df_in: pd.DataFrame,
    feat: pd.DataFrame,
    args,
    albumin_seq: Optional[str],
):
    lines = []
    lines.append("# WIQD Assumptions & Sanity Checks")
    lines.append(
        f"Input peptides: {len(df_in)} | unique: {df_in['peptide'].nunique() if 'peptide' in df_in.columns else 'NA'}"
    )
    if "is_hit" in df_in.columns:
        uniq = sorted(pd.Series(df_in["is_hit"]).dropna().unique().tolist())
        lines.append(f"is_hit unique values: {uniq}")
        if set(uniq) - {0, 1}:
            lines.append(
                "WARN: is_hit has values outside {0,1}. They were coerced if possible."
            )
    if albumin_seq:
        L = len(albumin_seq)
        lines.append(f"Albumin length used: {L}")
        if L not in (585, 609):
            lines.append("WARN: albumin length is not 585 or 609; verify source/trim.")
    missing = feat.isna().mean().sort_values(ascending=False).head(20)
    lines.append("Top features by missingness (fraction missing):")
    for k, v in missing.items():
        lines.append(f"  {k}: {v:.3f}")
    lines.append("Albumin metrics are counts (no sequence matching; no fractions).")
    lines.append(
        f"RT gating applied where *_rt present: ±{args.rt_tolerance_min} min on a {args.gradient_min}‑min gradient."
    )
    if args.require_precursor_match_for_frag:
        lines.append(
            "Fragment metrics are restricted to albumin windows that match precursor m/z within ppm tolerance (ungated and RT‑gated)."
        )
    with open(os.path.join(outdir, "ASSUMPTIONS.txt"), "w") as fh:
        fh.write("\n".join(lines))


# =================== Feature type inference (auto) ===================
def _norm_to_unit_set(vals, eps: float = 1e-12):
    """
    Normalize a small set of unique values to canonical {-1, 0, 1} (and bools),
    keeping anything else as-is. NaNs are ignored by the caller.
    """
    out = set()
    for v in vals:
        # skip NaN safely (works for floats and pandas NA scalars)
        try:
            if v != v:  # NaN != NaN
                continue
        except Exception:
            pass
        if isinstance(v, bool):
            out.add(1 if v else 0)
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                if abs(fv - 1.0) <= eps:
                    out.add(1.0)
                    continue
                if abs(fv) <= eps:
                    out.add(0.0)
                    continue
                if abs(fv + 1.0) <= eps:
                    out.add(-1.0)
                    continue
        out.add(v)
    return out


def infer_feature_types_auto(
    df: pd.DataFrame,
    exclude: Iterable[str] = ("peptide", "is_hit"),
    eps: float = 1e-12,
):
    """
    Return (numeric_feats, binary_feats, binary_scheme) where:
      - numeric_feats: numeric, non-binary columns
      - binary_feats: binary columns
      - binary_scheme: dict feature -> "01" or "pm1"
    All-NaN columns are ignored.
    """
    binary_feats = []
    numeric_feats = []
    binary_scheme = {}  # feature -> "01" (0/1) or "pm1" (-1/1)

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        # Only consider numeric/bool; skip objects/strings
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            continue

        uniq = pd.unique(s.dropna())
        if len(uniq) == 0:
            # all missing; ignore this feature
            continue

        norm = _norm_to_unit_set(uniq, eps=eps)
        if norm <= {0.0, 1.0}:
            binary_feats.append(col)
            binary_scheme[col] = "01"
        elif norm <= {-1.0, 1.0}:
            binary_feats.append(col)
            binary_scheme[col] = "pm1"
        else:
            numeric_feats.append(col)

    # Ensure separation
    keep_numeric = set(numeric_feats) - set(binary_feats)
    numeric_feats = [c for c in df.columns if c in keep_numeric]
    return numeric_feats, binary_feats, binary_scheme


def normalize_binary_series(s: pd.Series, scheme: str, eps: float = 1e-12) -> pd.Series:
    """
    Convert a binary-typed series to {0.0, 1.0} with NaNs preserved.
    - scheme "01": values near {0,1} (and bools) -> 0/1
    - scheme "pm1": values near {-1,1} -> 0/1 via -1→0,  1→1
    """
    if pd.api.types.is_bool_dtype(s):
        return s.map(lambda x: 1.0 if bool(x) else 0.0)

    s_num = pd.to_numeric(s, errors="coerce")

    if scheme == "01":

        def map01(x):
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                if abs(x - 1.0) <= eps:
                    return 1.0
                if abs(x) <= eps:
                    return 0.0
            return float("nan")

        return s_num.map(map01)

    if scheme == "pm1":

        def map_pm1(x):
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                if abs(x - 1.0) <= eps:
                    return 1.0
                if abs(x + 1.0) <= eps:
                    return 0.0
            return float("nan")

        return s_num.map(map_pm1)

    # Fallback (shouldn't happen with our inference)
    return s_num


# =================== Analysis ===================
def run_analysis(args):
    # Preconditions
    assert os.path.isfile(args.input_csv), f"Input file not found: {args.input_csv}"
    assert args.alpha is None or (0 < args.alpha <= 1.0)
    assert args.permutes >= 1000
    assert args.tie_thresh is None or (0 <= args.tie_thresh <= 1.0)
    assert args.ppm_tol > 0
    assert 1 <= args.full_mz_len_min <= args.full_mz_len_max
    assert 1 <= args.by_mz_len_min <= args.by_mz_len_max
    assert 1 <= args.by_seq_len_min <= args.by_seq_len_max
    assert args.rt_tolerance_min >= 0
    assert args.gradient_min > 0

    # Output dirs
    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Read data
    df = pd.read_csv(args.input_csv)
    assert "peptide" in df.columns, "Input CSV must have column 'peptide'."
    # Optionally derive is_hit
    if args.min_score is not None:
        score_col = None
        for c in df.columns:
            if c.lower() == "score":
                score_col = c
                break
        if score_col is None:
            raise ValueError("--min_score provided but no 'score' column found.")
        df["is_hit"] = (
            pd.to_numeric(df[score_col], errors="coerce") >= float(args.min_score)
        ).astype(int)
    assert (
        "is_hit" in df.columns
    ), "Input CSV must include peptide,is_hit (or provide --min_score)."

    # Clean
    df["peptide"] = df["peptide"].map(clean_peptide)
    df = df[df["peptide"].str.len() > 0].copy()
    df["is_hit"] = df["is_hit"].astype(int)
    bad_is_hit = set(df["is_hit"].unique()) - {0, 1}
    assert not bad_is_hit, f"is_hit must be 0/1 only; seen: {sorted(bad_is_hit)}"

    # Chemistry
    charges = sorted({int(x.strip()) for x in args.charges.split(",") if x.strip()})
    assert charges and all(z > 0 for z in charges)
    cys_fixed_mod = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0

    # Fly weights
    fly_weights = {}
    for tok in args.fly_weights.split(","):
        if ":" in tok:
            k, v = tok.split(":", 1)
            fly_weights[k.strip()] = float(v.strip())

    # Core + flyability
    feat = compute_features(df["peptide"].tolist(), fly_weights, args)

    # Ambiguous pair presence flags (kept)
    amb_rows = []
    for p in tqdm(feat["peptide"], desc="[2/8] Ambiguous pair flags ..."):
        amb_union_set = set("LIKQFMNDE")
        amb_rows.append(
            {
                "has_L_or_I": 1 if any(c in "LI" for c in p) else 0,
                "has_K_or_Q": 1 if any(c in "KQ" for c in p) else 0,
                "has_F_or_Mox": 1 if any(c in "FM" for c in p) else 0,
                "has_N_or_D": 1 if any(c in "ND" for c in p) else 0,
                "has_Q_or_E": 1 if any(c in "QE" for c in p) else 0,
                "has_ambiguous_union": 1 if any(c in amb_union_set for c in p) else 0,
            }
        )
    feat = pd.concat([feat, pd.DataFrame(amb_rows)], axis=1)

    # Fragment confusability (independent of albumin)
    conf_rows = [
        fragment_confusability_features(
            p, frag_tol_da=args.frag_tol_da, frag_tol_ppm=args.frag_tol_ppm
        )
        for p in tqdm(feat["peptide"], desc="[3/8] Fragment confusability ...")
    ]
    feat = pd.concat([feat, pd.DataFrame(conf_rows)], axis=1)

    # Housekeeping homology
    seqs = load_housekeeping_sequences(
        args.housekeeping_fasta, args.download_housekeeping
    )
    bank = build_kmer_bank(seqs, k=args.k, collapse=args.collapse)
    hk_rows = [
        homology_features_with_params(p, bank, k=args.k, collapse=args.collapse)
        for p in tqdm(feat["peptide"], desc="[4/8] Homology per peptide ...")
    ]
    feat = pd.concat([feat, pd.DataFrame(hk_rows)], axis=1)

    # Albumin metrics (counts only)
    albumin = load_albumin_sequence(args) if args.albumin_source != "none" else ""
    alb_rows = []
    if albumin:
        # Unified albumin window index covering all lengths needed (min/max spans precursor/frag ranges)
        idx_len_min = min(args.full_mz_len_min, args.by_mz_len_min, args.by_seq_len_min)
        idx_len_max = max(args.full_mz_len_max, args.by_mz_len_max, args.by_seq_len_max)
        assert idx_len_min >= 1 and idx_len_max >= idx_len_min
        alb_windows_all = build_albumin_windows_index(
            albumin=albumin,
            len_min=idx_len_min,
            len_max=idx_len_max,
            cys_fixed_mod=cys_fixed_mod,
            charges=charges,
            gradient_min=args.gradient_min,
            collapse=args.collapse,
        )
        assert (
            len(alb_windows_all) > 0
        ), "Albumin windows index is empty; check albumin sequence"

        # Pre-filter pools for precursor length range (for counts and optional fragment gating)
        precursor_len_pool_all = [
            w
            for w in alb_windows_all
            if args.full_mz_len_min <= w.length <= args.full_mz_len_max
        ]

        for _, r in tqdm(
            feat.iterrows(), total=len(feat), desc="[5/8] Albumin metrics ..."
        ):
            pep = r["peptide"]
            pep_m = mass_neutral(pep, cys_fixed_mod)
            pep_mz_by_z = {z: mz_from_mass(pep_m, z) for z in charges}
            pep_rt = RT(pep, gradient_min=args.gradient_min)

            # Precompute peptide ANY-fragment m/z cache (across configured by_mz range on peptide)
            pep_frag_mz = _build_peptide_anyfrag_mz_cache(
                pep, charges, cys_fixed_mod, args.by_mz_len_min, args.by_mz_len_max
            )

            # RT pools
            rt_pool_precursor_len = _filter_rt(
                precursor_len_pool_all, pep_rt, args.rt_tolerance_min
            )

            # Count precursor matches (ungated and RT-gated) over the length-limited pools
            prec_match_count_all = _count_precursor_mz_matches(
                precursor_len_pool_all, pep_mz_by_z, args.ppm_tol
            )
            prec_match_count_rt = _count_precursor_mz_matches(
                rt_pool_precursor_len, pep_mz_by_z, args.ppm_tol
            )

            # Fragment pools
            frag_pool_all = precursor_len_pool_all
            frag_pool_rt = rt_pool_precursor_len
            if args.require_precursor_match_for_frag:
                frag_pool_all = _filter_precursor_mz(
                    precursor_len_pool_all, pep_mz_by_z, args.ppm_tol
                )
                frag_pool_rt = _filter_precursor_mz(
                    rt_pool_precursor_len, pep_mz_by_z, args.ppm_tol
                )

            # Best precursor metrics (context)
            def precursor_mm_over(windows: List[AlbWindow]):
                best = {"ppm": float("inf"), "da": float("inf"), "z": None, "L": None}
                for w in windows:
                    if (w.length < args.full_mz_len_min) or (
                        w.length > args.full_mz_len_max
                    ):
                        continue
                    for z, pmz in pep_mz_by_z.items():
                        wmz = w.mz_by_z.get(z)
                        if wmz is None:
                            continue
                        da = abs(wmz - pmz)
                        ppmv = (da / pmz) * 1e6
                        if ppmv < best["ppm"]:
                            best.update({"ppm": ppmv, "da": da, "z": z, "L": w.length})
                return best

            pre_all = precursor_mm_over(alb_windows_all)
            cand_rt_all = _filter_rt(alb_windows_all, pep_rt, args.rt_tolerance_min)
            pre_rt = precursor_mm_over(cand_rt_all)

            # ANY-fragment m/z counts (ungated + RT-gated)
            counts_mz_all = _fragment_any_mz_match_counts_over(
                frag_pool_all, pep_frag_mz, charges, cys_fixed_mod, args.ppm_tol
            )
            counts_mz_rt = _fragment_any_mz_match_counts_over(
                frag_pool_rt, pep_frag_mz, charges, cys_fixed_mod, args.ppm_tol
            )

            # Score (ppm→[0,1]) for best precursor m/z
            conf_prec_all = (
                ppm_to_score(pre_all["ppm"], args.ppm_tol)
                if pre_all["ppm"] == pre_all["ppm"]
                else 0.0
            )
            conf_prec_rt = (
                ppm_to_score(pre_rt["ppm"], args.ppm_tol)
                if pre_rt["ppm"] == pre_rt["ppm"]
                else 0.0
            )

            alb_rows.append(
                {
                    "alb_candidates_total": len(alb_windows_all),
                    "alb_candidates_rt": len(cand_rt_all),
                    # Precursor counts
                    "alb_precursor_mz_match_count": prec_match_count_all,
                    "alb_precursor_mz_match_count_rt": prec_match_count_rt,
                    # Best precursor metrics (context)
                    "alb_precursor_mz_best_ppm": pre_all["ppm"],
                    "alb_precursor_mz_best_da": pre_all["da"],
                    "alb_precursor_mz_best_charge": pre_all["z"],
                    "alb_precursor_mz_best_len": pre_all["L"],
                    "alb_confusability_precursor_mz": conf_prec_all,
                    # ANY-fragment (m/z) counts
                    "alb_frag_candidates_mz": counts_mz_all["cand"],
                    "alb_frag_mz_match_count": counts_mz_all["mz_count"],
                    # RT‑gated counterparts
                    "alb_precursor_mz_best_ppm_rt": pre_rt["ppm"],
                    "alb_precursor_mz_best_da_rt": pre_rt["da"],
                    "alb_precursor_mz_best_charge_rt": pre_rt["z"],
                    "alb_precursor_mz_best_len_rt": pre_rt["L"],
                    "alb_confusability_precursor_mz_rt": conf_prec_rt,
                    "alb_frag_candidates_mz_rt": counts_mz_rt["cand"],
                    "alb_frag_mz_match_count_rt": counts_mz_rt["mz_count"],
                    # Back-compat alias retained
                    "alb_same_len_best_ppm": pre_all["ppm"],
                    "alb_same_len_best_da": pre_all["da"],
                    "alb_same_len_best_charge": pre_all["z"],
                }
            )
        feat = pd.concat([feat, pd.DataFrame(alb_rows)], axis=1)
    else:
        feat = feat.assign(alb_precursor_mz_best_ppm=float("nan"))

    # Replace infinities
    feat.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

    # Merge labels
    feat = feat.merge(df[["peptide", "is_hit"]], on="peptide", how="left")
    feat.to_csv(os.path.join(args.outdir, "features.csv"), index=False)

    # ---- Stats & plots ----
    g0 = feat[feat["is_hit"] == 0]
    g1 = feat[feat["is_hit"] == 1]
    n_nonhit = len(g0)
    n_hit = len(g1)

    numeric_feats = [
        "length",
        "mass_mono",
        "gravy",
        "eisenberg_hydro",
        "hopp_woods_hydrophilicity",
        "kd_stdev",
        "hydrophobic_fraction",
        "max_hydrophobic_run",
        "frag_doublet_total_count",
        "frag_doublet_total_frac",
        "synth_doublet_total_count",
        "synth_doublet_total_frac",
        "hard_to_synthesize_residues",
        "pI",
        "charge_pH2_0",
        "charge_pH2_7",
        "charge_pH7_4",
        "charge_pH10_0",
        "aliphatic_index",
        "aromaticity",
        "xle_fraction",
        "basic_count",
        "acidic_count",
        "basicity_proxy",
        "n_to_proline_bonds",
        "c_to_acidic_bonds",
        "HBD_sidechain",
        "HBA_sidechain",
        "elem_C",
        "elem_H",
        "elem_N",
        "elem_O",
        "elem_S",
        "approx_Mplus1_rel",
        "mz_z1",
        "mz_z2",
        "mz_z3",
        "mass_defect",
        "hk_kmer_hits",
        "hk_kmer_frac",
        "frac_L_or_I",
        "frac_K_or_Q",
        "frac_F_or_Mox",
        "frac_N_or_D",
        "frac_Q_or_E",
        "frac_ambiguous_union",
        "confusable_bion_frac",
        "confusable_yion_frac",
        "confusable_ion_frac",
        "fly_charge_norm",
        "fly_surface_norm",
        "fly_aromatic_norm",
        "fly_len_norm",
        "fly_score",
        # RT
        "rt_pred_min",
        # ESI
        "es_chargeability",
        # albumin
        "alb_candidates_total",
        "alb_candidates_rt",
        "alb_precursor_mz_match_count",
        "alb_precursor_mz_match_count_rt",
        "alb_frag_candidates_mz",
        "alb_frag_candidates_mz_rt",
        "alb_frag_mz_match_count",
        "alb_frag_mz_match_count_rt",
        "alb_precursor_mz_best_ppm",
        "alb_precursor_mz_best_da",
        "alb_precursor_mz_best_len",
        "alb_confusability_precursor_mz",
        "alb_precursor_mz_best_ppm_rt",
        "alb_precursor_mz_best_da_rt",
        "alb_precursor_mz_best_len_rt",
        "alb_confusability_precursor_mz_rt",
    ]
    if albumin == "":
        numeric_feats = [f for f in numeric_feats if not f.startswith("alb_")]

    # Append per-motif doublet numeric features
    for di in FRAG_DOUBLETS:
        for suffix in ("count", "frac"):
            col = f"frag_dbl_{di}_{suffix}"
            if col in feat.columns:
                numeric_feats.append(col)
    for di in SYN_RISK_DOUBLETS:
        for suffix in ("count", "frac"):
            col = f"synth_dbl_{di}_{suffix}"
            if col in feat.columns:
                numeric_feats.append(col)

    # Ensure dedup & existence
    seen = set()
    numeric_feats = [
        f
        for f in numeric_feats
        if (f not in seen and not seen.add(f) and f in feat.columns)
    ]

    # Binary feats
    binary_feats = (
        [
            "cterm_hydrophobic",
            "tryptic_end",
            "has_RP_or_KP",
            "has_L_or_I",
            "has_K_or_Q",
            "has_F_or_Mox",
            "has_N_or_D",
            "has_Q_or_E",
            "has_ambiguous_union",
            "frag_dbl_proline_after",
            "frag_dbl_proline_before",
        ]
        + [f"has_{a}" for a in sorted(AA)]
        + [f"Nterm_{a}" for a in sorted(AA)]
    )

    # Stats
    pmap = {}
    rows = []
    now = datetime.datetime.now().isoformat(timespec="seconds")

    # Numeric features
    for f in tqdm(numeric_feats, desc="[6/8] Testing numeric ..."):
        x = g0[f].dropna().tolist()
        y = g1[f].dropna().tolist()
        med0 = float(pd.Series(x).median()) if x else float("nan")
        med1 = float(pd.Series(y).median()) if y else float("nan")
        mean0 = float(pd.Series(x).mean()) if x else float("nan")
        mean1 = float(pd.Series(y).mean()) if y else float("nan")
        uniq0 = len(set(x))
        uniq1 = len(set(y))
        nnz0 = sum(1 for v in x if (isinstance(v, (int, float)) and v != 0))
        nnz1 = sum(1 for v in y if (isinstance(v, (int, float)) and v != 0))
        degenerate = False
        reason = ""
        U = float("nan")
        p_param = float("nan")
        p_perm = float("nan")
        cliffs = float("nan")
        d = float("nan")
        g_ = float("nan")
        test_used = "none"
        if len(x) == 0 or len(y) == 0:
            degenerate = True
            reason = "empty_group"
            tie_fraction = float("nan")
            is_discrete = 0
        else:
            combined = x + y
            if min(combined) == max(combined):
                degenerate = True
                reason = "all_values_identical"
                tie_fraction = float("nan")
                is_discrete = 0
            else:
                from collections import Counter

                cnt = Counter(combined)
                ties = sum(c for c in cnt.values() if c > 1)
                tie_fraction = (ties / len(combined)) if combined else 0.0
                is_discrete = (
                    1 if all(abs(v - round(v)) < 1e-12 for v in combined) else 0
                )
                use_permU = (args.numeric_test == "perm_U") or (
                    args.numeric_test == "auto"
                    and (is_discrete or tie_fraction > args.tie_thresh)
                )
                if use_permU and args.numeric_test != "mw":
                    U = mannwhitney_U_only(x, y)
                    p_perm = perm_pvalue_U(
                        x,
                        y,
                        iters=args.permutes,
                        rng_seed=123,
                        progress_desc=f"[wiqd] permU {f}",
                    )
                    p_val = p_perm
                    test_used = "perm_U"
                else:
                    U, p_param = mannwhitney_u_p(x, y)
                    p_val = p_param
                    test_used = "mannwhitney_u"
                cliffs = cliffs_delta(x, y)
                d, g_ = cohens_d_and_g(x, y)
        pmap[f] = p_val if not degenerate else float("nan")
        rows.append(
            {
                "feature": f,
                "type": "numeric",
                "test_used": test_used,
                "n_nonhit": len(x),
                "n_hit": len(y),
                "unique_nonhit": uniq0,
                "unique_hit": uniq1,
                "median_nonhit": med0,
                "median_hit": med1,
                "mean_nonhit": mean0,
                "mean_hit": mean1,
                "sd_nonhit": float("nan"),
                "sd_hit": float("nan"),
                "var_nonhit": float("nan"),
                "var_hit": float("nan"),
                "U": U,
                "p_parametric": p_param,
                "p_perm_U": p_perm,
                "p_value": p_val if not degenerate else float("nan"),
                "q_value": float("nan"),
                "effect_cliffs_delta": cliffs,
                "effect_cohens_d": d,
                "effect_hedges_g": g_,
                "tie_fraction": (tie_fraction if not degenerate else float("nan")),
                "is_discrete": is_discrete if not degenerate else 0,
                "counts": "",
                "prop_nonhit": float("nan"),
                "prop_hit": float("nan"),
                "degenerate": 1 if degenerate else 0,
                "degenerate_reason": reason,
                "computed_at": now,
                "nnz_nonhit": nnz0,
                "nnz_hit": nnz1,
            }
        )

    # Binary features
    for f in tqdm(binary_feats, desc="[7/8] Testing binary ..."):
        a = int((g1[f] == 1).sum())
        b = int((g1[f] == 0).sum())
        c = int((g0[f] == 1).sum())
        d_ = int((g0[f] == 0).sum())
        n_hit = a + b
        n_non = c + d_
        prop_hit = a / max(1, n_hit)
        prop_non = c / max(1, n_non)
        degenerate = False
        reason = ""
        p_val = float("nan")
        test_used = "fisher_exact"
        if n_hit == 0 or n_non == 0:
            degenerate = True
            reason = "empty_group"
        elif (a + b + c + d_) == 0:
            degenerate = True
            reason = "no_data"
        elif (a == 0 and c == 0) or (b == 0 and d_ == 0):
            degenerate = True
            reason = "constant_feature"
        else:
            p_val = fisher_exact(a, b, c, d_)
        pmap[f] = p_val if not degenerate else float("nan")
        rows.append(
            {
                "feature": f,
                "type": "binary",
                "test_used": test_used,
                "n_nonhit": n_non,
                "n_hit": n_hit,
                "unique_nonhit": float("nan"),
                "unique_hit": float("nan"),
                "median_nonhit": float("nan"),
                "median_hit": float("nan"),
                "mean_nonhit": prop_non,
                "mean_hit": prop_hit,
                "sd_nonhit": float("nan"),
                "sd_hit": float("nan"),
                "var_nonhit": float("nan"),
                "var_hit": float("nan"),
                "U": float("nan"),
                "p_parametric": p_val,
                "p_perm_U": float("nan"),
                "p_value": p_val if not degenerate else float("nan"),
                "q_value": float("nan"),
                "effect_cliffs_delta": float("nan"),
                "effect_cohens_d": float("nan"),
                "effect_hedges_g": float("nan"),
                "tie_fraction": float("nan"),
                "is_discrete": 0,
                "counts": f"hit: {a}/{n_hit}; non-hit: {c}/{n_non}",
                "prop_nonhit": prop_non,
                "prop_hit": prop_hit,
                "degenerate": 1 if degenerate else 0,
                "degenerate_reason": reason,
                "computed_at": now,
                "nnz_nonhit": c,
                "nnz_hit": a,
            }
        )

    # FDR
    qmap = bh_fdr(pmap)
    for r in rows:
        r["q_value"] = qmap[r["feature"]]
    stats_df = pd.DataFrame(rows)

    # Order & save stats
    order = [
        "feature",
        "type",
        "test_used",
        "n_nonhit",
        "n_hit",
        "unique_nonhit",
        "unique_hit",
        "median_nonhit",
        "median_hit",
        "mean_nonhit",
        "mean_hit",
        "sd_nonhit",
        "sd_hit",
        "var_nonhit",
        "var_hit",
        "U",
        "p_parametric",
        "p_perm_U",
        "p_value",
        "q_value",
        "effect_cliffs_delta",
        "effect_cohens_d",
        "effect_hedges_g",
        "tie_fraction",
        "is_discrete",
        "counts",
        "prop_nonhit",
        "prop_hit",
        "degenerate",
        "degenerate_reason",
        "computed_at",
        "nnz_nonhit",
        "nnz_hit",
    ]
    for col in order:
        if col not in stats_df.columns:
            stats_df[col] = (
                ""
                if col
                in (
                    "feature",
                    "type",
                    "test_used",
                    "degenerate_reason",
                    "counts",
                    "computed_at",
                )
                else float("nan")
            )
    stats_df = stats_df[order]
    stats_df.to_csv(os.path.join(args.outdir, "stats_summary.csv"), index=False)

    # Per-feature plots
    for f in tqdm(numeric_feats, desc="[8/8] Plot numeric ..."):
        x = g0[f].dropna().tolist()
        y = g1[f].dropna().tolist()
        if len(x) == 0 and len(y) == 0:
            continue
        pv = stats_df.loc[stats_df["feature"] == f, "p_value"].values[0]
        qv = stats_df.loc[stats_df["feature"] == f, "q_value"].values[0]
        sig = isinstance(qv, float) and qv == qv and qv < args.alpha
        save_boxplot(
            x,
            y,
            f,
            pv,
            qv,
            os.path.join(plots_dir, f"{safe_name(f)}__box{'.SIG' if sig else ''}.png"),
        )
    for f in tqdm(binary_feats, desc="[8/8] Plot binary ..."):
        a = int((g1[f] == 1).sum())
        b = int((g1[f] == 0).sum())
        c = int((g0[f] == 1).sum())
        d_ = int((g0[f] == 0).sum())
        pv = stats_df.loc[stats_df["feature"] == f, "p_value"].values[0]
        qv = stats_df.loc[stats_df["feature"] == f, "q_value"].values[0]
        sig = isinstance(qv, float) and qv == qv and qv < args.alpha
        save_barplot_props(
            f,
            a,
            b,
            c,
            d_,
            pv,
            qv,
            os.path.join(plots_dir, f"{safe_name(f)}__bar{'.SIG' if sig else ''}.png"),
        )

    # AA composition (original)
    aa_labels = sorted(AA)
    mean_non = [
        float(g0[f"frac_{a}"].mean()) if not g0.empty else float("nan")
        for a in aa_labels
    ]
    mean_hit = [
        float(g1[f"frac_{a}"].mean()) if not g1.empty else float("nan")
        for a in aa_labels
    ]
    sig_mask = [
        (
            (
                stats_df.loc[stats_df["feature"] == f"frac_{a}", "q_value"].values[0]
                < args.alpha
            )
            if not stats_df.loc[stats_df["feature"] == f"frac_{a}", "q_value"].empty
            else False
        )
        for a in aa_labels
    ]
    save_grouped_bar(
        aa_labels,
        mean_non,
        mean_hit,
        "AA composition (mean per peptide) | '*' = q<α",
        "mean fraction",
        os.path.join(
            plots_dir,
            "aa_composition.SIG.png" if any(sig_mask) else "aa_composition.png",
        ),
        sig_mask=sig_mask,
    )

    # Summary plot (with nnz filter)
    save_summary_plot(
        stats_df,
        args.alpha,
        os.path.join(args.outdir, "summary_feature_significance.png"),
        n_nonhit=n_nonhit,
        n_hit=n_hit,
        top_n=args.summary_top_n,
        min_nnz_topn=args.min_nnz_topn,
    )

    # Extra comparative scatter (safe)
    try:
        if albumin and "alb_precursor_mz_best_ppm" in feat.columns:
            fig, ax = plt.subplots()
            ax.set_title("Flyability vs Albumin precursor best ppm")
            ax.set_xlabel("Flyability score")
            ax.set_ylabel("Precursor best ppm (log scale)")
            ax.set_yscale("log")
            ax.scatter(
                g0["fly_score"],
                g0["alb_precursor_mz_best_ppm"],
                s=10,
                alpha=0.5,
                label="non-hit",
            )
            ax.scatter(
                g1["fly_score"],
                g1["alb_precursor_mz_best_ppm"],
                s=10,
                alpha=0.5,
                label="hit",
            )
            ax.legend()
            _apply_grid(ax)
            fig.savefig(
                os.path.join(args.outdir, "scatter_fly_vs_albumin_ppm.png"),
                bbox_inches="tight",
            )
            plt.close(fig)
    except Exception as e:
        print(f"[wiqd][WARN] Extra plots failed: {e}")

    # Assumptions report
    write_assumptions_report(args.outdir, df, feat, args, albumin)

    # README
    with open(os.path.join(args.outdir, "README.txt"), "w") as fh:
        fh.write(
            "\n".join(
                [
                    "# WIQD summary",
                    f"Samples: n(non-hit)={n_nonhit}, n(hit)={n_hit}",
                    f"Testing: numeric_test={args.numeric_test} (auto uses perm_U if discrete/tie_fraction>{args.tie_thresh})",
                    "Files: features.csv | stats_summary.csv | plots/*.png | summary_feature_significance.png | ASSUMPTIONS.txt",
                    "",
                    "## Albumin metrics policy",
                    "- Counts only. No albumin sequence matching and no *_frac fields.",
                    "- Ungated and RT‑gated variants (suffix '_rt').",
                    f"- RT gating uses ±{args.rt_tolerance_min:g} min on a {args.gradient_min:g}-min gradient.",
                    "- Fragment metrics use ANY contiguous subsequences (not b/y‑only); matches in m/z space within ±ppm_tol at charges --charges.",
                    "- Optionally, fragment pools are restricted to albumin windows that also match PRECURSOR m/z (see --require_precursor_match_for_frag).",
                    "- Best precursor metrics retained (best ppm/Da/charge/len) + a ppm→score convenience scalar.",
                    "",
                    "## Proline & synthesis flags",
                    "- Proline-related presence flags collapsed to: frag_dbl_proline_after (X–P), frag_dbl_proline_before (P–K/R).",
                    "- hard_to_synthesize_residues = total count of specified synthesis-risk doublets.",
                    "",
                    "## Summary Top‑N gating",
                    f"- Features must satisfy nnz_nonhit ≥ {args.min_nnz_topn} and nnz_hit ≥ {args.min_nnz_topn} to appear in the Top‑N bar.",
                    "",
                    "## Flyability parity",
                    "- Flyability components/weights centralized in wiqd_fly_score.",
                ]
            )
        )


def main():
    ap = argparse.ArgumentParser(
        description="WIQD — hit vs non-hit peptide feature analysis (+albumin counts & flyability; no peptide modification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--in",
        dest="input_csv",
        required=True,
        help="Input CSV with peptide,is_hit (or provide --min_score and a score column)",
    )
    ap.add_argument("--outdir", default="wiqd_out", help="Output directory")
    # Stats
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    ap.add_argument(
        "--numeric_test",
        choices=["auto", "mw", "perm_U"],
        default="auto",
        help="Numeric test: auto (perm_U if discrete/tie-heavy), mw, or perm_U",
    )
    ap.add_argument(
        "--permutes", type=int, default=5000, help="Permutation iterations for perm_U"
    )
    ap.add_argument(
        "--tie_thresh",
        type=float,
        default=0.25,
        help="Tie-fraction threshold to trigger perm_U in auto mode",
    )
    ap.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="If set, derive is_hit = 1[score >= min_score]",
    )
    ap.add_argument(
        "--min_nnz_topn",
        type=int,
        default=4,
        help="Minimum nnz per group (non-zero counts in hits and non-hits) required to qualify for Top‑N summary",
    )
    # Fragment confusability (independent of albumin)
    ap.add_argument(
        "--frag_tol_da",
        type=float,
        default=0.02,
        help="Neutral mass tolerance (Da) for fragment confusability (indep.)",
    )
    ap.add_argument(
        "--frag_tol_ppm",
        type=float,
        default=None,
        help="Neutral mass tolerance (ppm) for fragment confusability (overrides Da if set)",
    )
    # Housekeeping
    ap.add_argument("--k", type=int, default=6, help="k for homology")
    ap.add_argument(
        "--collapse",
        choices=["none", "xle", "xle_de_qn", "xle+deqn"],
        default="xle",
        help="Collapsed alphabet for homology/albumin sequence checks",
    )
    ap.add_argument(
        "--housekeeping_fasta",
        default=None,
        help="Path to housekeeping FASTA (optional)",
    )
    ap.add_argument(
        "--download_housekeeping",
        action="store_true",
        help="Download housekeeping proteins from UniProt (requires internet)",
    )
    # Albumin
    ap.add_argument(
        "--albumin_source",
        choices=["embedded", "file", "fetch", "auto", "none"],
        default="embedded",
        help="Albumin source",
    )
    ap.add_argument(
        "--albumin_fasta", default=None, help="Albumin FASTA (if file/auto)"
    )
    ap.add_argument(
        "--albumin_acc",
        default="P02768",
        help="UniProt accession(s) for albumin fetch (comma-separated)",
    )
    ap.add_argument(
        "--albumin_expected",
        choices=["prepro", "mature", "either"],
        default="either",
        help="Expected albumin length check",
    )
    ap.add_argument(
        "--albumin_use",
        choices=["prepro", "mature", "auto"],
        default="mature",
        help="Which form to use",
    )
    # Albumin matching controls
    ap.add_argument(
        "--ppm_tol", type=float, default=30.0, help="PPM tolerance for albumin matching"
    )
    ap.add_argument(
        "--full_mz_len_min",
        type=int,
        default=5,
        help="Albumin window min length for PRECURSOR m/z",
    )
    ap.add_argument(
        "--full_mz_len_max",
        type=int,
        default=15,
        help="Albumin window max length for PRECURSOR m/z",
    )
    ap.add_argument(
        "--by_mz_len_min",
        type=int,
        default=2,
        help="Peptide fragment k-min for m/z (ANY contiguous subseq)",
    )
    ap.add_argument(
        "--by_mz_len_max",
        type=int,
        default=7,
        help="Peptide fragment k-max for m/z (ANY contiguous subseq)",
    )
    ap.add_argument(
        "--by_seq_len_min", type=int, default=2, help="(Retained for index bounds only)"
    )
    ap.add_argument(
        "--by_seq_len_max", type=int, default=7, help="(Retained for index bounds only)"
    )
    # RT gating
    ap.add_argument(
        "--rt_tolerance_min",
        type=float,
        default=1.0,
        help="RT co‑elution tolerance in minutes for RT‑gated albumin metrics",
    )
    ap.add_argument(
        "--gradient_min",
        type=float,
        default=20.0,
        help="Assumed gradient length in minutes used by RT predictor",
    )
    ap.add_argument(
        "--require_precursor_match_for_frag",
        action="store_true",
        help="For fragment metrics, only albumin windows that also match PRECURSOR m/z within ppm tolerance are considered (ungated and RT‑gated)",
    )
    # Chemistry
    ap.add_argument(
        "--charges",
        default="2,3",
        help="Charges to evaluate for albumin/fragment m/z metrics (e.g., 2,3)",
    )
    ap.add_argument(
        "--cys_mod",
        choices=["none", "carbamidomethyl"],
        default="carbamidomethyl",
        help="Fixed mod on Cys for albumin mass calc",
    )
    # Flyability weights
    ap.add_argument(
        "--fly_weights",
        default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05",
        help="Weights for flyability mix (sum normalized)",
    )
    ap.add_argument(
        "--summary_top_n",
        type=int,
        default=20,
        help="How many features to show in summary plot",
    )
    args = ap.parse_args()
    banner(args=args)
    run_analysis(args)


if __name__ == "__main__":
    main()
