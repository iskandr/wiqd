#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ref_endog_confusability.py

Reference–Endogenous confusability scoring for QqQ SRM/MRM method design.

Changes in this revision
------------------------
- Imports chemistry constants from `wiqd_constants` using *your* names:
  AA, KD, EISEN, HOPP, MONO, H2O, PROTON_MASS, PKA, hydrophobic_set,
  HBD_SC, HBA_SC, RES_ELEM, H2O_ELEM, C13_C12_DELTA, ADDUCT_MASS,
  POLYMER_REPEAT_MASS, POLYMER_ENDGROUP.
- All m/z math now uses PROTON_MASS exactly as defined in wiqd_constants.
- NEW: Reference-only modification *uncertainty* enumeration with averaging:
  * allow_heavy_first_residue (bool): enumerate none, 13C-first, 13C+15N-first.
  * allow_mtraq (bool): enumerate N-term mTRAQ Δ0 and Δ8.
  For each scenario we recompute reference precursor/fragment m/z and average
  the composite score across scenarios. Endogenous/protein windows remain
  unmodified. Polymer comparison also averages across scenarios.

What this module models
-----------------------
- Q1/Q3 as Gaussian passbands (FWHM in Da/ppm; optional hard gates).
- Fragment matching in Daltons (mirrors QqQ windows).
- Transition selection from the reference peptide; symmetric mode available.
- Protein-subsequence window index + search (with optional PNG for best hit).
- Polymer/chemical series (PEG/PPG/PTMEG/PDMS) enumeration and Q1-only match.
- Literature-informed fragment weighting (Proline effect, Asp/Glu bias) + optional
  AA-content multipliers. Heuristic but grounded.
- NEW: Uncertain reference-only first-residue heavy labeling and N-term mTRAQ.

Dependencies
------------
- Required: Python 3.x
- Optional: tqdm (progress bars), matplotlib (for plotting)
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

from wiqd_constants import (
    AA,
    KD,
    EISEN,
    HOPP,
    MONO,
    H2O,
    PROTON_MASS,
    PKA,
    hydrophobic_set,
    HBD_SC,
    HBA_SC,
    RES_ELEM,
    H2O_ELEM,
    C13_C12_DELTA,
    ADDUCT_MASS,
    POLYMER_REPEAT_MASS,
    POLYMER_ENDGROUP,
    MTRAQ_DELTA0_NTERM,
    MTRAQ_DELTA8_DIFF,
    MTRAQ_DELTA8_NTERM,
    N15_N14_DELTA,
)
from wiqd_features import mass_neutral as pep_mass, mz_from_mass

# ----------------------------- Literature-informed defaults -------------------
SERIES_BIAS_LIT = {
    "y": 1.00,
    "b": 0.60,
}  # y > b in low-energy CID/HCD (heuristic baseline)

MOTIF_BOOSTS_LIT: Dict[str, Tuple[float, float]] = {
    "GP": (2.4, 1.3),  # glycine–proline
    "AP": (1.8, 1.2),  # alanine–proline
}
PRO_PRO_SUPPRESSION = 0.30  # rare cleavage at Pro|Pro

ACID_EFFECT_MULT = {  # extra boost when protons are non-mobile
    "D": (1.6, 1.3),
    "E": (1.3, 1.15),
}

AA_CONTENT_WEIGHTS_DEFAULT: Dict[str, float] = {
    **{aa: 1.00 for aa in MONO.keys()},
    "P": 1.15,
    "K": 1.08,
    "R": 1.08,
    "D": 1.04,
    "E": 1.04,
    "H": 1.03,
}


# ----------------------------- Utilities & tqdm wrapper -----------------------
def _tqdm_wrap(iterable, desc=None, total=None, disable=False):
    """Wrap an iterable with tqdm if available; otherwise return the iterable."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        return iterable


def _validate_seq(seq: str) -> None:
    if not seq:
        raise ValueError("Sequence is empty.")
    bad = [ch for ch in seq if ch not in AA]
    if bad:
        raise ValueError(f"Unknown amino acid(s): {''.join(sorted(set(bad)))}")


def _fwhm_da(fwhm: Tuple[str, float], mz_ref: float) -> float:
    unit, val = fwhm
    u = unit.lower()
    if u == "da":
        return float(val)
    if u == "ppm":
        return float(val) * 1e-6 * mz_ref
    raise ValueError("FWHM unit must be 'da' or 'ppm'")


def _gate_allows(
    delta_da: float, mz_ref: float, tol: Optional[Tuple[str, float]]
) -> bool:
    if tol is None:
        return True
    unit, val = tol
    u = unit.lower()
    if u == "da":
        return abs(delta_da) <= float(val)
    if u == "ppm":
        return abs(delta_da) <= float(val) * 1e-6 * mz_ref
    raise ValueError("Tolerance unit must be 'da' or 'ppm'")


def _gauss(delta_da: float, mz_ref: float, fwhm: Tuple[str, float]) -> float:
    """Gaussian passband weight: exp(-0.5*(Δ/σ)^2), with σ=FWHM/2.355 at mz_ref."""
    f_da = _fwhm_da(fwhm, mz_ref)
    if f_da <= 0:
        return 0.0
    sigma = f_da / 2.355
    return math.exp(-0.5 * (delta_da / sigma) ** 2)


# ----------------------------- Modified reference modeling --------------------
def _heavy_delta_for_first_residue(first_res: str, mode: str) -> float:
    """
    Return mass delta for heavy labeling of the *first residue*:
      mode
           'C13'  -> #C * Δ(13C-12C)
           'C13N15' -> #C * Δ(13C-12C) + #N * Δ(15N-14N)
    """

    tup = RES_ELEM.get(first_res)
    if tup is None:
        raise ValueError(f"Unknown amino acid '{first_res}'")
    # RES_ELEM is (C,H,N,O,S)
    c, _, n, _, _ = tup
    c = int(c)
    n = int(n)
    if mode == "C13":
        return c * C13_C12_DELTA
    if mode == "C13N15":
        return c * C13_C12_DELTA + n * N15_N14_DELTA
    else:
        raise ValueError(f"Unknown heavy mode '{mode}'")


def _enumerate_reference_mod_scenarios(
    seq: str,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
) -> List[Dict[str, object]]:
    """
    Build a list of modification scenarios for the *reference peptide only*.
    Each scenario dict has:
      {'label','res_mono','nterm_extra','M_ref'}
    where:
      res_mono : per-position residue monoisotopic masses (may include heavy at pos0)
      nterm_extra : N-terminal mass addition (e.g., mTRAQ Δ0/Δ8)
      M_ref : neutral precursor mass (sum(res_mono) + H2O + nterm_extra)
    """
    # heavy modes
    heavy_modes: List[Optional[str]] = [None]
    if allow_heavy_first_residue:
        heavy_modes += ["C13", "C13N15"]

    # mTRAQ options
    mtraq_adds: List[float] = [0.0]
    if allow_mtraq:
        mtraq_adds = [MTRAQ_DELTA0_NTERM, MTRAQ_DELTA8_NTERM]

    scenarios: List[Dict[str, object]] = []
    base_res = [MONO[a] for a in seq]
    for hm in heavy_modes:
        res = list(base_res)
        if hm:
            res[0] = res[0] + _heavy_delta_for_first_residue(seq[0], hm)
        for mt in mtraq_adds:
            M = sum(res) + H2O + float(mt)
            label = f"heavy={hm or 'none'};mTRAQ={'Δ0' if mt==MTRAQ_DELTA0_NTERM else ('Δ8' if mt==MTRAQ_DELTA8_NTERM else 'none')}"
            scenarios.append(
                {
                    "label": label,
                    "res_mono": list(res),
                    "nterm_extra": float(mt),
                    "M_ref": float(M),
                }
            )
    return scenarios


# ----------------------------- Fragment generation (b/y) ----------------------
def _fragments(
    seq: str,
    *,
    series: Sequence[str] = ("y", "b"),
    frag_charges: Sequence[int] = (1,),
    include_y1: bool = False,
    include_y2: bool = True,
    include_b1: bool = False,
    include_b2: bool = True,
    min_mz: float = 200.0,
    max_mz: float = 1500.0,
) -> List[Dict]:
    """
    Generate fragment list: {'name','series','index','charge','mz'} for y/b series.
    Practical defaults: keep y2/b2; exclude y1/b1; m/z in [200,1500].
    """
    _validate_seq(seq)
    n = len(seq)
    # prefix[i] = sum of first i residue masses
    prefix = [0.0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + MONO[seq[i - 1]]
    Mtot = prefix[n] + H2O

    out: List[Dict] = []
    if "b" in series:
        for i in range(1, n):
            if i == 1 and not include_b1:
                continue
            if i == 2 and not include_b2:
                continue
            neutral = prefix[i]
            for z in frag_charges:
                mz = mz_from_mass(neutral, z)
                if min_mz <= mz <= max_mz:
                    out.append(
                        {
                            "name": f"b{i}^{z}+",
                            "series": "b",
                            "index": i,
                            "charge": z,
                            "mz": mz,
                        }
                    )
    if "y" in series:
        for i in range(1, n):
            if i == 1 and not include_y1:
                continue
            if i == 2 and not include_y2:
                continue
            neutral = Mtot - prefix[n - i]  # tail + H2O
            for z in frag_charges:
                mz = mz_from_mass(neutral, z)
                if min_mz <= mz <= max_mz:
                    out.append(
                        {
                            "name": f"y{i}^{z}+",
                            "series": "y",
                            "index": i,
                            "charge": z,
                            "mz": mz,
                        }
                    )
    return out


def _fragments_from_res_mono(
    res_mono: Sequence[float],
    *,
    series: Sequence[str] = ("y", "b"),
    frag_charges: Sequence[int] = (1,),
    include_y1: bool = False,
    include_y2: bool = True,
    include_b1: bool = False,
    include_b2: bool = True,
    min_mz: float = 200.0,
    max_mz: float = 1500.0,
    nterm_extra: float = 0.0,
) -> List[Dict]:
    """
    Generate fragments from a pre-modified monomer mass list and optional N-term extra mass.
    b-ions include nterm_extra; y-ions do not.
    """
    n = len(res_mono)
    prefix = [0.0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + float(res_mono[i - 1])
    Mtot = prefix[n] + H2O + float(nterm_extra)

    out: List[Dict] = []
    if "b" in series:
        for i in range(1, n):
            if i == 1 and not include_b1:
                continue
            if i == 2 and not include_b2:
                continue
            neutral = prefix[i] + float(nterm_extra)
            for z in frag_charges:
                mz = mz_from_mass(neutral, z)
                if min_mz <= mz <= max_mz:
                    out.append(
                        {
                            "name": f"b{i}^{z}+",
                            "series": "b",
                            "index": i,
                            "charge": z,
                            "mz": mz,
                        }
                    )
    if "y" in series:
        for i in range(1, n):
            if i == 1 and not include_y1:
                continue
            if i == 2 and not include_y2:
                continue
            # y_i contains the C-terminal i residues; no N-term addition
            neutral = (prefix[n] - prefix[n - i]) + H2O
            for z in frag_charges:
                mz = mz_from_mass(neutral, z)
                if min_mz <= mz <= max_mz:
                    out.append(
                        {
                            "name": f"y{i}^{z}+",
                            "series": "y",
                            "index": i,
                            "charge": z,
                            "mz": mz,
                        }
                    )
    return out


# ----------------------------- Fragment importance models ---------------------
def _mobile_proton_index(seq: str, z_precursor: int) -> float:
    basic = seq.count("K") + seq.count("R")
    denom = max(0.5, basic + 0.5)
    return float(z_precursor) / float(denom)


def _apply_motif_and_context(
    seq: str,
    frags: List[Dict],
    base_y: float,
    base_b: float,
    motif_boosts: Optional[Dict[str, Tuple[float, float]]],
    aa_content_weights: Optional[Dict[str, float]],
    apply_aa_content: bool,
    pro_pro_supp: float,
    acid_effect_mult: Dict[str, Tuple[float, float]],
    assume_precursor_charge: int,
) -> List[float]:
    n = len(seq)
    dipep_at_cleave = {i: f"{seq[i-1]}{seq[i]}" for i in range(1, n)}
    boosts = motif_boosts or {}
    aa_w = aa_content_weights or AA_CONTENT_WEIGHTS_DEFAULT
    nonmobile = _mobile_proton_index(seq, assume_precursor_charge) < 1.0

    weights: List[float] = []
    for f in frags:
        base = base_y if f["series"] == "y" else base_b
        i = f["index"] if f["series"] == "b" else (n - f["index"])
        if not (1 <= i < n):
            weights.append(max(0.0, base))
            continue
        di = dipep_at_cleave[i]
        left, right = di[0], di[1]

        if right == "P":
            if di in boosts:
                y_mul, b_mul = boosts[di]
            else:
                y_mul, b_mul = 2.2, 1.2
            if nonmobile and left not in ("D", "E"):
                y_mul *= 0.6
                b_mul *= 0.8
            base *= y_mul if f["series"] == "y" else b_mul

        if left == "P" and right == "P":
            base *= float(pro_pro_supp)

        if nonmobile and left in acid_effect_mult:
            y_mul, b_mul = acid_effect_mult[left]
            base *= y_mul if f["series"] == "y" else b_mul

        if apply_aa_content:
            base *= aa_w.get(left, 1.0) * aa_w.get(right, 1.0)

        weights.append(max(0.0, base))
    return weights


def _select_transitions(
    frags: List[Dict],
    weights: List[float],
    *,
    mode: str = "topk",
    k: int = 4,
    prefer_series_order: Tuple[str, ...] = ("y", "b"),
    prefer_high_mz: bool = True,
) -> List[int]:
    if mode == "all":
        return list(range(len(frags)))
    series_rank = {s: r for r, s in enumerate(prefer_series_order)}
    order = list(range(len(frags)))

    def key(i: int):
        f = frags[i]
        return (
            -weights[i],
            series_rank.get(f["series"], 999),
            -(f["mz"] if prefer_high_mz else -f["mz"]),
        )

    order.sort(key=key)
    return order[: max(0, k)]


# ----------------------------- Matching & Q1/Q3 scores ------------------------
def _nearest_fragment_delta(target_mz: float, pool: List[Dict]) -> Tuple[int, float]:
    if not pool:
        return -1, float("inf")
    best_idx, best_d = -1, float("inf")
    for j, f in enumerate(pool):
        d = abs(f["mz"] - target_mz)
        if d < best_d:
            best_idx, best_d = j, d
    return best_idx, best_d


def _best_precursor_overlap(
    M_ref: float,
    M_endog: float,
    charges: Iterable[int],
    q1_fwhm: Tuple[str, float],
    q1_tol: Optional[Tuple[str, float]],
    max_isotope_shift_q1: int = 0,
) -> Tuple[float, int, float]:
    best = (0.0, 0, float("inf"))
    for z in charges:
        mzr = mz_from_mass(M_ref, z)
        for k in range(0, max(0, int(max_isotope_shift_q1)) + 1):
            mze = mz_from_mass(M_endog + k * C13_C12_DELTA, z)
            d = abs(mzr - mze)
            w = _gauss(d, mzr, q1_fwhm) if _gate_allows(d, mzr, q1_tol) else 0.0
            if w > best[0]:
                best = (w, z, d)
    return best


def _rt_weight(
    delta_min: Optional[float], rt_inner_min: float, rt_fwhm_min: float
) -> float:
    if delta_min is None:
        return 1.0
    if delta_min <= max(0.0, rt_inner_min):
        return 1.0
    if rt_fwhm_min <= 0:
        return 0.0
    sigma = rt_fwhm_min / 2.355
    return math.exp(-0.5 * ((delta_min - rt_inner_min) / sigma) ** 2)


def _q3_overlap_mean_weight(
    trans_ref: List[Dict],
    frags_endog: List[Dict],
    q3_fwhm: Tuple[str, float],
    q3_tol: Optional[Tuple[str, float]],
    sel_weights: List[float],
) -> Tuple[float, int, List[Dict]]:
    if not trans_ref:
        return 0.0, 0, []
    details = []
    accum = 0.0
    matched = 0
    for w_sel, t in zip(sel_weights, trans_ref):
        j, d = _nearest_fragment_delta(t["mz"], frags_endog)
        allowed = _gate_allows(d, t["mz"], q3_tol)
        w_q3 = _gauss(d, t["mz"], q3_fwhm) if allowed else 0.0
        accum += w_sel * w_q3
        matched += int(allowed and w_q3 > 0.0)
        details.append(
            {
                "transition": t["name"],
                "target_mz": t["mz"],
                "nearest_name": (
                    frags_endog[j]["name"] if 0 <= j < len(frags_endog) else None
                ),
                "nearest_mz": (
                    frags_endog[j]["mz"] if 0 <= j < len(frags_endog) else None
                ),
                "delta_da": d,
                "w_q3": w_q3,
                "sel_weight": w_sel,
            }
        )
    return accum, matched, details


# ----------------------------- Public entry points (ref/endog) ----------------
def score_ref_endog_reference(
    ref_seq: str,
    endog_seq: str,
    *,
    # Mass filters
    q1_fwhm: Tuple[str, float] = ("da", 0.7),
    q3_fwhm: Tuple[str, float] = ("da", 0.7),
    q1_tol: Optional[Tuple[str, float]] = None,
    q3_tol: Optional[Tuple[str, float]] = None,
    charges: Sequence[int] = (1, 2, 3),
    max_isotope_shift_q1: int = 0,
    # Fragments
    series: Sequence[str] = ("y", "b"),
    frag_charges: Sequence[int] = (1,),
    include_y1: bool = False,
    include_y2: bool = True,
    include_b1: bool = False,
    include_b2: bool = True,
    min_frag_mz: float = 200.0,
    max_frag_mz: float = 1500.0,
    # Fragment models
    frag_model: str = "lit_weighted",
    transitions_k: int = 4,
    y_bias: float = SERIES_BIAS_LIT["y"],
    b_bias: float = SERIES_BIAS_LIT["b"],
    motif_boosts: Optional[Dict[str, Tuple[float, float]]] = MOTIF_BOOSTS_LIT,
    aa_content_weights: Optional[Dict[str, float]] = AA_CONTENT_WEIGHTS_DEFAULT,
    apply_aa_content: bool = False,
    prefer_series_order: Tuple[str, ...] = ("y", "b"),
    prefer_high_mz: bool = True,
    # RT
    rt_delta_min: Optional[float] = None,
    rt_inner_min: float = 1.0,
    rt_fwhm_min: float = 5.0,
    # Output
    return_fragment_maps: bool = True,
    # Context for acid/proline effects
    assume_precursor_charge_for_rules: Optional[int] = None,
    # NEW: reference-only uncertain modifications
    allow_heavy_first_residue: bool = False,
    allow_mtraq: bool = False,
) -> Dict[str, object]:
    """
    Reference-mode confusability with uncertain *reference-only* modifications.
    We enumerate reference scenarios (none / 13C-first / 13C+15N-first) × (mTRAQ Δ0/Δ8 when enabled),
    recompute Q1/Q3 overlaps per scenario, and return the **mean** composite score.
    Endogenous is never modified here.
    """
    _validate_seq(ref_seq)
    _validate_seq(endog_seq)

    # Unmodified endogenous
    M_endog = pep_mass(endog_seq)
    fr_end = _fragments(
        endog_seq,
        series=series,
        frag_charges=frag_charges,
        include_y1=include_y1,
        include_y2=include_y2,
        include_b1=include_b1,
        include_b2=include_b2,
        min_mz=min_frag_mz,
        max_mz=max_frag_mz,
    )

    # Resolve intensity model
    model = frag_model.lower()
    if model == "lit_weighted":
        model = "motif_weighted"
    assume_z = assume_precursor_charge_for_rules or (charges[0] if charges else 2)

    scenarios = _enumerate_reference_mod_scenarios(
        ref_seq,
        allow_heavy_first_residue=allow_heavy_first_residue,
        allow_mtraq=allow_mtraq,
    )
    if not scenarios:
        scenarios = [
            {
                "label": "baseline",
                "res_mono": [MONO[a] for a in ref_seq],
                "nterm_extra": 0.0,
                "M_ref": pep_mass(ref_seq),
            }
        ]

    per_s = []
    for s in scenarios:
        M_ref_s = float(s["M_ref"])
        # Q1
        w_q1, best_z, d_q1 = _best_precursor_overlap(
            M_ref_s, M_endog, charges, q1_fwhm, q1_tol, max_isotope_shift_q1
        )
        # Fragments for reference scenario
        fr_ref = _fragments_from_res_mono(
            s["res_mono"],
            series=series,
            frag_charges=frag_charges,
            include_y1=include_y1,
            include_y2=include_y2,
            include_b1=include_b1,
            include_b2=include_b2,
            min_mz=min_frag_mz,
            max_mz=max_frag_mz,
            nterm_extra=float(s["nterm_extra"]),
        )
        imp_ref = (
            _apply_motif_and_context(
                ref_seq,
                fr_ref,
                base_y=(y_bias if model in {"weighted", "motif_weighted"} else 1.0),
                base_b=(b_bias if model in {"weighted", "motif_weighted"} else 1.0),
                motif_boosts=(motif_boosts if model == "motif_weighted" else None),
                aa_content_weights=aa_content_weights,
                apply_aa_content=apply_aa_content,
                pro_pro_supp=PRO_PRO_SUPPRESSION,
                acid_effect_mult=ACID_EFFECT_MULT,
                assume_precursor_charge=assume_z,
            )
            if model in {"weighted", "motif_weighted"}
            else [1.0] * len(fr_ref)
        )
        sel_idx = _select_transitions(
            fr_ref,
            imp_ref,
            mode=("topk" if model in {"topk", "weighted", "motif_weighted"} else "all"),
            k=transitions_k,
            prefer_series_order=prefer_series_order,
            prefer_high_mz=prefer_high_mz,
        )
        trans_ref = [fr_ref[i] for i in sel_idx]
        sel_w = [max(imp_ref[i], 0.0) for i in sel_idx]
        norm = sum(sel_w) or 1.0
        sel_w = [w / norm for w in sel_w]

        mean_q3, matched, details = _q3_overlap_mean_weight(
            trans_ref, fr_end, q3_fwhm, q3_tol, sel_w
        )
        rt_w = _rt_weight(rt_delta_min, rt_inner_min, rt_fwhm_min)
        score = w_q1 * mean_q3 * rt_w
        per_s.append(
            {
                "scenario": s["label"],
                "score": score,
                "precursor": {"best_z": best_z, "delta_da": d_q1, "weight": w_q1},
                "fragments": {
                    "mean_weight": mean_q3,
                    "matched": matched,
                    "n_selected": len(sel_idx),
                    "selected_from_ref": trans_ref if return_fragment_maps else None,
                    "details": (details if return_fragment_maps else None),
                },
                "rt_weight": rt_w,
            }
        )

    # Aggregate
    scores = [x["score"] for x in per_s]
    score_mean = float(sum(scores) / len(scores)) if scores else 0.0
    score_max = max(scores) if scores else 0.0
    # For compatibility: keep the same keys; pick the scenario with highest score for representative fields
    best_idx = max(range(len(per_s)), key=lambda i: per_s[i]["score"]) if per_s else -1
    best = per_s[best_idx] if best_idx >= 0 else None

    summary = (
        f"Confusability(mean over {len(per_s)} mod scenarios)={score_mean:.3f} "
        f"| max={score_max:.3f}"
    )
    if best:
        summary += (
            f" | best: Q1 z={best['precursor']['best_z']}, Δ={best['precursor']['delta_da']:.3f} Da, "
            f"w_Q1={best['precursor']['weight']:.3f} | Q3={best['fragments']['mean_weight']:.3f} "
            + (f"| RTw={best['rt_weight']:.3f}" if rt_delta_min is not None else "")
        )

    return {
        "score": score_mean,  # mean over scenarios
        "summary": summary,
        "score_max": score_max,  # extra, non-breaking
        "precursor": (
            best["precursor"]
            if best
            else {"best_z": 0, "delta_da": float("inf"), "weight": 0.0}
        ),
        "fragments": (
            best["fragments"]
            if best
            else {
                "mean_weight": 0.0,
                "matched": 0,
                "n_selected": 0,
                "selected_from_ref": None,
                "details": None,
            }
        ),
        "rt_weight": (best["rt_weight"] if best else 1.0),
        "params": {
            "q1_fwhm": q1_fwhm,
            "q3_fwhm": q3_fwhm,
            "q1_tol": q1_tol,
            "q3_tol": q3_tol,
            "charges": list(charges),
            "max_isotope_shift_q1": max_isotope_shift_q1,
            "series": list(series),
            "frag_charges": list(frag_charges),
            "include_y1": include_y1,
            "include_y2": include_y2,
            "include_b1": include_b1,
            "include_b2": include_b2,
            "min_frag_mz": min_frag_mz,
            "max_frag_mz": max_frag_mz,
            "frag_model": frag_model,
            "transitions_k": transitions_k,
            "y_bias": y_bias,
            "b_bias": b_bias,
            "motif_boosts": motif_boosts,
            "aa_content_weights": aa_content_weights,
            "apply_aa_content": apply_aa_content,
            "prefer_series_order": prefer_series_order,
            "prefer_high_mz": prefer_high_mz,
            "rt_delta_min": rt_delta_min,
            "rt_inner_min": rt_inner_min,
            "rt_fwhm_min": rt_fwhm_min,
            "assume_precursor_charge_for_rules": assume_precursor_charge_for_rules,
            # NEW flags echoed back
            "allow_heavy_first_residue": allow_heavy_first_residue,
            "allow_mtraq": allow_mtraq,
        },
        "mod_scenarios": per_s,  # detailed per-scenario breakdown (additive, non-breaking)
    }


def score_ref_endog_symmetric(
    ref_seq: str, endog_seq: str, **kwargs
) -> Dict[str, object]:
    """
    Symmetric comparison: compute ref→endog and endog→ref (each picks its own top-K),
    and **each direction** applies reference-only modification enumeration/averaging
    to its designated reference peptide. Returns per-direction results + mean/max.
    """
    a2b = score_ref_endog_reference(ref_seq, endog_seq, **kwargs)
    b2a = score_ref_endog_reference(endog_seq, ref_seq, **kwargs)
    return {
        "ref_to_endog": a2b,
        "endog_to_ref": b2a,
        "score_mean": 0.5 * (a2b["score"] + b2a["score"]),
        "score_max": max(a2b["score"], b2a["score"]),
        "summary": f"Symmetric: mean={0.5*(a2b['score']+b2a['score']):.3f}, max={max(a2b['score'], b2a['score']):.3f}",
    }


# ----------------------------- Protein subsequence index & search -------------
def build_protein_index(
    proteins: Dict[str, str] | Sequence[Tuple[str, str]],
    *,
    min_len: int = 7,
    max_len: int = 25,
    step: int = 1,
    charges: Sequence[int] = (2,),  # store these z values per window
    series: Sequence[str] = ("y", "b"),
    frag_charges: Sequence[int] = (1,),
    include_y1: bool = False,
    include_y2: bool = True,
    include_b1: bool = False,
    include_b2: bool = True,
    min_frag_mz: float = 200.0,
    max_frag_mz: float = 1500.0,
    show_progress: bool = True,
) -> Dict[str, object]:
    """
    Build an index of windows. Each entry has:
      {'protein','start','end','subseq','precursor_mzs':{z:m/z}, 'fragments':[...]}
    Windows are unmodified; uncertain mods are applied to the reference at query time.
    """
    items = proteins.items() if isinstance(proteins, dict) else proteins
    entries: List[Dict] = []
    for pid, seq in _tqdm_wrap(
        list(items), desc="index:proteins", disable=not show_progress
    ):
        _validate_seq(seq)
        n = len(seq)
        for L in range(min_len, max_len + 1):
            for i in range(0, max(0, n - L + 1), step):
                sub = seq[i : i + L]
                M = pep_mass(sub)
                prec = {z: mz_from_mass(M, z) for z in charges}
                fr = _fragments(
                    sub,
                    series=series,
                    frag_charges=frag_charges,
                    include_y1=include_y1,
                    include_y2=include_y2,
                    include_b1=include_b1,
                    include_b2=include_b2,
                    min_mz=min_frag_mz,
                    max_mz=max_frag_mz,
                )
                entries.append(
                    {
                        "protein": pid,
                        "start": i,
                        "end": i + L,
                        "subseq": sub,
                        "precursor_mzs": prec,
                        "fragments": fr,
                    }
                )
    return {
        "meta": {
            "charges": list(charges),
            "series": list(series),
            "frag_charges": list(frag_charges),
            "include_y1": include_y1,
            "include_y2": include_y2,
            "include_b1": include_b1,
            "include_b2": include_b2,
            "min_frag_mz": min_frag_mz,
            "max_frag_mz": max_frag_mz,
            "min_len": min_len,
            "max_len": max_len,
            "step": step,
        },
        "entries": entries,
    }


def search_index_with_reference(
    ref_seq: str,
    index: Dict[str, object],
    *,
    top_n: int = 20,
    min_score: float = 0.10,
    q1_fwhm=("da", 0.7),
    q3_fwhm=("da", 0.7),
    q1_tol=None,
    q3_tol=None,
    charges=(1, 2, 3),
    max_isotope_shift_q1=0,
    frag_model="lit_weighted",
    transitions_k: int = 4,
    y_bias: float = SERIES_BIAS_LIT["y"],
    b_bias: float = SERIES_BIAS_LIT["b"],
    motif_boosts=MOTIF_BOOSTS_LIT,
    aa_content_weights=AA_CONTENT_WEIGHTS_DEFAULT,
    apply_aa_content: bool = False,
    prefer_series_order=("y", "b"),
    prefer_high_mz: bool = True,
    rt_delta_min=None,
    rt_inner_min=1.0,
    rt_fwhm_min=5.0,
    plot_png_path: Optional[str] = None,
    show_progress: bool = True,
    assume_precursor_charge_for_rules: Optional[int] = None,
    # NEW: reference-only uncertain modifications
    allow_heavy_first_residue: bool = False,
    allow_mtraq: bool = False,
) -> Dict[str, object]:
    """
    Score every window in the index against the (possibly modified) reference and return top matches.
    Protein windows are unmodified; we enumerate reference scenarios and average scores per window.
    """
    meta = index["meta"]
    model = frag_model.lower()
    if model == "lit_weighted":
        model = "motif_weighted"
    assume_z = assume_precursor_charge_for_rules or (charges[0] if charges else 2)

    scenarios = _enumerate_reference_mod_scenarios(
        ref_seq,
        allow_heavy_first_residue=allow_heavy_first_residue,
        allow_mtraq=allow_mtraq,
    )
    if not scenarios:
        scenarios = [
            {
                "label": "baseline",
                "res_mono": [MONO[a] for a in ref_seq],
                "nterm_extra": 0.0,
                "M_ref": pep_mass(ref_seq),
            }
        ]

    # Precompute scenario-specific transitions from reference
    scen_transitions = []
    for s in scenarios:
        fr_ref = _fragments_from_res_mono(
            s["res_mono"],
            series=tuple(meta["series"]),
            frag_charges=tuple(meta["frag_charges"]),
            include_y1=bool(meta["include_y1"]),
            include_y2=bool(meta["include_y2"]),
            include_b1=bool(meta["include_b1"]),
            include_b2=bool(meta["include_b2"]),
            min_mz=float(meta["min_frag_mz"]),
            max_mz=float(meta["max_frag_mz"]),
            nterm_extra=float(s["nterm_extra"]),
        )
        imp_ref = (
            _apply_motif_and_context(
                ref_seq,
                fr_ref,
                base_y=(y_bias if model in {"weighted", "motif_weighted"} else 1.0),
                base_b=(b_bias if model in {"weighted", "motif_weighted"} else 1.0),
                motif_boosts=(motif_boosts if model == "motif_weighted" else None),
                aa_content_weights=aa_content_weights,
                apply_aa_content=apply_aa_content,
                pro_pro_supp=PRO_PRO_SUPPRESSION,
                acid_effect_mult=ACID_EFFECT_MULT,
                assume_precursor_charge=assume_z,
            )
            if model in {"weighted", "motif_weighted"}
            else [1.0] * len(fr_ref)
        )
        sel_idx = _select_transitions(
            fr_ref,
            imp_ref,
            mode=("topk" if model in {"topk", "weighted", "motif_weighted"} else "all"),
            k=transitions_k,
            prefer_series_order=prefer_series_order,
            prefer_high_mz=prefer_high_mz,
        )
        trans_ref = [fr_ref[i] for i in sel_idx]
        sel_w = [max(imp_ref[i], 0.0) for i in sel_idx]
        norm = sum(sel_w) or 1.0
        sel_w = [w / norm for w in sel_w]
        scen_transitions.append({"scenario": s, "trans": trans_ref, "sel_w": sel_w})

    hits: List[Dict] = []
    for e in _tqdm_wrap(
        index["entries"],
        desc="search:index",
        total=len(index["entries"]),
        disable=not show_progress,
    ):
        per_scores = []
        per_details = []
        for st in scen_transitions:
            s = st["scenario"]
            # Q1 vs stored window precursors (like-for-like charges only)
            best_q1 = (0.0, 0, float("inf"))
            for z in charges:
                mzr = mz_from_mass(s["M_ref"], z)
                mze = e["precursor_mzs"].get(z)
                if mze is None:
                    continue
                d = abs(mzr - mze)
                w = _gauss(d, mzr, q1_fwhm) if _gate_allows(d, mzr, q1_tol) else 0.0
                if w > best_q1[0]:
                    best_q1 = (w, z, d)
            w_q1, zbest, d_q1 = best_q1
            mean_q3, matched, details = _q3_overlap_mean_weight(
                st["trans"], e["fragments"], q3_fwhm, q3_tol, st["sel_w"]
            )
            rt_w = _rt_weight(rt_delta_min, rt_inner_min, rt_fwhm_min)
            score = w_q1 * mean_q3 * rt_w
            per_scores.append(score)
            per_details.append(
                {
                    "scenario": s["label"],
                    "z": zbest,
                    "delta_da": d_q1,
                    "w_q1": w_q1,
                    "q3": mean_q3,
                    "rtw": rt_w,
                    "details": details,
                }
            )

        if not per_scores:
            continue
        s_mean = float(sum(per_scores) / len(per_scores))
        if s_mean >= min_score:
            # choose scenario with max score for detail
            jbest = int(max(range(len(per_scores)), key=lambda j: per_scores[j]))
            best = per_details[jbest]
            hits.append(
                {
                    "score": s_mean,
                    "protein": e["protein"],
                    "start": e["start"],
                    "end": e["end"],
                    "subseq": e["subseq"],
                    "summary": f"{e['protein']}:{e['start']}-{e['end']} score_mean={s_mean:.3f} Q1z={best['z']} Δ={best['delta_da']:.3f} Q3={best['q3']:.3f}",
                    "q1": {
                        "z": best["z"],
                        "delta_da": best["delta_da"],
                        "w_q1": best["w_q1"],
                    },
                    "q3": {"mean_weight": best["q3"], "matched": None},
                    "details": best["details"],
                }
            )

    hits.sort(key=lambda h: -h["score"])
    best_plot = None
    if hits and plot_png_path:
        try:
            # Use the top hit; pick the best scenario for this top hit to display
            # Plot remains unmodified for the window (endogenous); for reference, display the scenario with max score
            from matplotlib import pyplot as plt  # local import safe

            # Build scenario for plotting: choose the first scenario (as example) or the highest-scoring one is not tracked here;
            # we reuse baseline plot for simplicity; advanced: user can pass overrides in next API
            plot_fragment_mz_distributions(
                ref_seq,
                hits[0]["subseq"],
                series=tuple(meta["series"]),
                frag_charges=tuple(meta["frag_charges"]),
                include_y1=bool(meta["include_y1"]),
                include_y2=bool(meta["include_y2"]),
                include_b1=bool(meta["include_b1"]),
                include_b2=bool(meta["include_b2"]),
                min_frag_mz=float(meta["min_frag_mz"]),
                max_frag_mz=float(meta["max_frag_mz"]),
                save_path=plot_png_path,
                # Optional overrides for ref can be exposed later if needed
            )
            best_plot = plot_png_path
        except Exception as ex:
            best_plot = f"Plot failed: {ex!r}"

    return {
        "ref": ref_seq,
        "top_hits": hits[:top_n],
        "n_scored": len(index["entries"]),
        "plot_file": best_plot,
    }


# ----------------------------- Polymer/chemical series (e.g., PEG) ------------
def enumerate_polymer_series(
    *,
    families=("PEG", "PPG", "PTMEG", "PDMS"),
    endgroups=("auto",),
    adducts=("H", "Na", "K", "NH4"),
    zset=(1, 2, 3),
    n_min=3,
    n_max=200,
    mz_min=200.0,
    mz_max=1500.0,
) -> List[Dict]:
    """
    Enumerate polymer candidates: neutral mass = n * repeat_mass + end_group_mass.
    Each candidate yields m/z for chosen adduct multisets and charges.
    """
    from itertools import combinations_with_replacement

    out: List[Dict] = []
    for fam in families:
        valid_egs = (
            tuple(POLYMER_ENDGROUP.get(fam, {}))
            if ("auto" in endgroups)
            else tuple(e for e in endgroups if e in POLYMER_ENDGROUP.get(fam, {}))
        )
        for end in valid_egs:
            for n in range(n_min, n_max + 1):
                base = POLYMER_ENDGROUP[fam][end] + n * POLYMER_REPEAT_MASS[fam]
                for z in zset:
                    adlabs = [a for a in adducts if a in ADDUCT_MASS]
                    for ads in combinations_with_replacement(adlabs, z):
                        mztot = (base + sum(ADDUCT_MASS[a] for a in ads)) / z
                        if mz_min <= mztot <= mz_max:
                            out.append(
                                {
                                    "family": fam,
                                    "endgroup": end,
                                    "n": n,
                                    "adducts": "+".join(ads),
                                    "z": z,
                                    "mz": mztot,
                                }
                            )
    return out


def best_polymer_match_for_reference(
    ref_seq: str,
    *,
    candidates: Optional[List[Dict]] = None,
    peptide_z: Sequence[int] = (1, 2, 3),
    q1_fwhm=("da", 0.7),
    q1_tol=None,
    top_n: int = 10,
    # NEW: reference-only uncertain modifications
    allow_heavy_first_residue: bool = False,
    allow_mtraq: bool = False,
) -> Dict[str, object]:
    """
    Q1-only comparison of a peptide to a polymer/chemical series.
    If uncertain reference-only mods are enabled, average the Q1 weights
    across enumerated scenarios for each candidate.
    """
    cands = candidates or enumerate_polymer_series()
    scenarios = _enumerate_reference_mod_scenarios(
        ref_seq,
        allow_heavy_first_residue=allow_heavy_first_residue,
        allow_mtraq=allow_mtraq,
    )
    if not scenarios:
        scenarios = [
            {
                "label": "baseline",
                "res_mono": [MONO[a] for a in ref_seq],
                "nterm_extra": 0.0,
                "M_ref": pep_mass(ref_seq),
            }
        ]

    results_map: Dict[int, Dict] = {}
    for idx, c in enumerate(cands):
        # We will aggregate weight and delta across scenarios
        w_accum = 0.0
        d_accum = 0.0
        n_eff = 0
        for s in scenarios:
            for z in peptide_z:
                if z != c["z"]:
                    continue
                mzr = mz_from_mass(s["M_ref"], z)
                d = abs(mzr - c["mz"])
                w = _gauss(d, mzr, q1_fwhm) if _gate_allows(d, mzr, q1_tol) else 0.0
                w_accum += w
                d_accum += d
                n_eff += 1
        if n_eff > 0 and w_accum > 0:
            results_map[idx] = {
                **c,
                "weight": w_accum / n_eff,
                "delta_da": d_accum / n_eff,
            }

    results = list(results_map.values())
    results.sort(key=lambda r: -r["weight"])
    return {"ref": ref_seq, "top": results[:top_n], "n_tested": len(cands)}


# ----------------------------- Plotting helper --------------------------------
def plot_fragment_mz_distributions(
    ref_seq: str,
    endog_seq: str,
    *,
    series=("y", "b"),
    frag_charges=(1,),
    include_y1=False,
    include_y2=True,
    include_b1=False,
    include_b2=True,
    min_frag_mz=200.0,
    max_frag_mz=1500.0,
    save_path: Optional[str] = None,
):
    """
    Plot two vertical-stick distributions of fragment m/z for reference (taller) and endogenous (shorter).
    (This convenience plot uses *unmodified* reference; for mod-specific plots, construct
     fragments with `_fragments_from_res_mono` and call matplotlib directly.)
    """
    import matplotlib.pyplot as plt

    fr_ref = _fragments(
        ref_seq,
        series=series,
        frag_charges=frag_charges,
        include_y1=include_y1,
        include_y2=include_y2,
        include_b1=include_b1,
        include_b2=include_b2,
        min_mz=min_frag_mz,
        max_mz=max_frag_mz,
    )
    fr_end = _fragments(
        endog_seq,
        series=series,
        frag_charges=frag_charges,
        include_y1=include_y1,
        include_y2=include_y2,
        include_b1=include_b1,
        include_b2=include_b2,
        min_mz=min_frag_mz,
        max_mz=max_frag_mz,
    )
    xs_ref = [f["mz"] for f in fr_ref]
    xs_end = [f["mz"] for f in fr_end]
    fig, ax = plt.subplots()
    for x in xs_ref:
        ax.vlines(x, 0, 1.0, linewidth=1)
    for x in xs_end:
        ax.vlines(x, 0, 0.6, linewidth=1)
    ax.set_xlabel("m/z")
    ax.set_ylabel("relative (arb.)")
    ax.set_title(f"Fragments: reference ({ref_seq}) vs endogenous ({endog_seq})")
    ax.set_xlim(min_frag_mz - 20, max_frag_mz + 20)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


# ----------------------------- Convenience defaults ---------------------------
def reference_defaults() -> Dict[str, object]:
    """Return a dict of sensible defaults for reference-mode scoring."""
    return dict(
        q1_fwhm=("da", 0.7),
        q3_fwhm=("da", 0.7),
        q1_tol=("da", 1.2),
        q3_tol=("da", 1.2),
        charges=(1, 2, 3),
        max_isotope_shift_q1=0,
        series=("y", "b"),
        frag_charges=(1,),
        include_y1=False,
        include_y2=True,
        include_b1=False,
        include_b2=True,
        min_frag_mz=200.0,
        max_frag_mz=1500.0,
        frag_model="lit_weighted",
        transitions_k=4,
        y_bias=SERIES_BIAS_LIT["y"],
        b_bias=SERIES_BIAS_LIT["b"],
        motif_boosts=MOTIF_BOOSTS_LIT,
        aa_content_weights=AA_CONTENT_WEIGHTS_DEFAULT,
        apply_aa_content=False,
        prefer_series_order=("y", "b"),
        prefer_high_mz=True,
        rt_delta_min=None,
        rt_inner_min=1.0,
        rt_fwhm_min=5.0,
        return_fragment_maps=True,
        assume_precursor_charge_for_rules=None,
        # NEW flags default off for backward compatibility
        allow_heavy_first_residue=False,
        allow_mtraq=False,
    )
