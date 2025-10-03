from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ChargeabilityWeights:
    # Proton-accepting side chains (gas-phase basicity-ish ordering)
    arg: float = 1.00
    lys: float = 0.90
    his: float = 0.40  # context dependent; can be lower if you prefer

    # N-terminus contributions (only if N-term is "free")
    nterm_free_base: float = 0.60
    nterm_is_pro_penalty: float = (
        -0.20
    )  # N-terminal Pro is a secondary amine (less basic)
    nterm_blocked_penalty: float = (
        -0.60
    )  # acetyl, pyro-Glu, etc. (no N-term proton site)

    # Acidic/acidifying features (mild penalties; do not drive score < 0 by themselves)
    asp: float = -0.15
    glu: float = -0.15
    free_cterm_penalty: float = -0.30  # free COOH is acidic; amidation removes this
    phosphate_penalty: float = -0.50  # per phospho-group (S/T/Y), strong local acidity
    acidic_cluster_bonus_penalty: float = -0.20  # extra penalty if ≥2 acidic in a row

    # Minor backbone/context terms
    pro_internal_penalty: float = -0.05  # many Pro can reduce proton mobility a bit

    # Floor the total score at zero?
    floor_at_zero: bool = True


def es_chargeability_stats(
    seq: str,
    *,
    nterm_state: str = "free",  # "free", "acetyl", "pyroglutamate", "blocked"
    cterm_state: str = "free",  # "free", "amidated"
    phospho_sites: Optional[
        List[int]
    ] = None,  # 0-based indices of phospho S/T/Y (if known)
    weights: ChargeabilityWeights = ChargeabilityWeights(),
) -> Dict[str, float]:
    """
    Estimate how easily a peptide acquires positive charge in ESI.

    Returns (total_score, breakdown_dict).

    Parameters
    ----------
    seq : str
        Peptide sequence (one-letter AA codes).
    nterm_state : {"free","acetyl","pyroglutamate","blocked"}
        If not "free", the N-terminus doesn't contribute a protonation site.
    cterm_state : {"free","amidated"}
        Free C-terminus is acidic; amidation removes that penalty.
    phospho_sites : list of 0-based indices
        Known phosphorylation sites (S/T/Y). If unknown, leave None.
    weights : ChargeabilityWeights
        Tunable weights.

    Notes
    -----
    - Arg > Lys >> His contribute most; free N-terminus adds a site unless blocked.
    - Acidic residues (D/E), free C-terminus, and phosphate groups apply mild penalties
      (they can sequester charge / reduce proton mobility).
    - N-terminal Pro contributes less than a primary amine N-terminus.
    - Score is heuristic; intended for ranking/screening, not an absolute pA.
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return 0.0, {"empty": 0.0}

    brk: Dict[str, float] = {}

    # --- Basic side chains ---
    nR = seq.count("R")
    nK = seq.count("K")
    nH = seq.count("H")
    brk["Arg_sidechains"] = nR * weights.arg
    brk["Lys_sidechains"] = nK * weights.lys
    brk["His_sidechains"] = nH * weights.his

    # --- N-terminus ---
    nterm_contrib = 0.0
    if nterm_state == "free":
        nterm_contrib += weights.nterm_free_base
        if seq[0] == "P":
            nterm_contrib += weights.nterm_is_pro_penalty  # secondary amine at N-term
    else:
        nterm_contrib += (
            weights.nterm_blocked_penalty
        )  # blocked/acetyl/pyroGlu, no N-term proton
    brk["Nterm"] = nterm_contrib

    # --- Acidic residues & clusters ---
    nD = seq.count("D")
    nE = seq.count("E")
    brk["Asp"] = nD * weights.asp
    brk["Glu"] = nE * weights.glu

    # Extra penalty for contiguous acidic clusters (≥2 in a row)
    acidic_cluster_penalty = 0.0
    run = 0
    for aa in seq:
        if aa in ("D", "E"):
            run += 1
            if run == 2:
                acidic_cluster_penalty += weights.acidic_cluster_bonus_penalty
        else:
            run = 0
    brk["Acidic_clusters"] = acidic_cluster_penalty

    # --- Phosphorylation (if known) ---
    phospho_pen = 0.0
    if phospho_sites:
        for i in phospho_sites:
            if 0 <= i < len(seq) and seq[i] in ("S", "T", "Y"):
                phospho_pen += weights.phosphate_penalty
    brk["Phosphate_groups"] = phospho_pen

    # --- C-terminus ---
    brk["Cterm"] = weights.free_cterm_penalty if cterm_state == "free" else 0.0

    # --- Proline (internal) ---
    nP_internal = seq.count("P") - (1 if seq and seq[0] == "P" else 0)
    brk["Pro_internal"] = nP_internal * weights.pro_internal_penalty

    total = sum(brk.values())
    if weights.floor_at_zero and total < 0:
        total = 0.0

    brk["total"] = total
    return brk


def es_chargeability_score(
    seq: str,
    *,
    nterm_state: str = "free",  # "free", "acetyl", "pyroglutamate", "blocked"
    cterm_state: str = "free",  # "free", "amidated"
    phospho_sites: Optional[
        List[int]
    ] = None,  # 0-based indices of phospho S/T/Y (if known)
    weights: ChargeabilityWeights = ChargeabilityWeights(),
) -> float:

    brk = es_chargeability_stats(
        seq,
        nterm_state=nterm_state,
        cterm_state=cterm_state,
        phospho_sites=phospho_sites,
        weights=weights,
    )
    return brk["total"]
