import math
from typing import Dict

from wiqd_constants import KD as KD_HYDRO, AA, PROTON_MASS as PROTON

# =================== Flyability ===================


def triangular_len_score(
    L: int, zero_min: int = 7, peak_min: int = 9, peak_max: int = 11, zero_max: int = 20
) -> float:
    if L <= zero_min:
        return 0.0
    if L >= zero_max:
        return 0.0
    if L < peak_min:
        return (L - zero_min) / max(1, (peak_min - zero_min))
    if L <= peak_max:
        return 1.0
    return max(0.0, (zero_max - L) / max(1, (zero_max - peak_max)))


def compute_flyability_components(seq: str) -> Dict[str, float]:
    L = len(seq)
    if L == 0:
        return {
            "fly_charge_norm": 0.0,
            "fly_surface_norm": 0.0,
            "fly_aromatic_norm": 0.0,
            "fly_len_norm": 0.0,
        }
    counts = {aa: 0 for aa in KD_HYDRO.keys()}
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
    R = counts.get("R", 0)
    K = counts.get("K", 0)
    H = counts.get("H", 0)
    D = counts.get("D", 0)
    E = counts.get("E", 0)
    charge_sites = 1.0 + R + K + 0.7 * H
    acid_penalty = 0.1 * (D + E)
    charge_norm = 1.0 - math.exp(-(max(0.0, charge_sites - acid_penalty)) / 2.0)
    gravy = sum(KD_HYDRO.get(a, 0.0) for a in seq) / L
    surface_norm = min(1.0, max(0.0, (gravy + 4.5) / 9.0))
    W = counts.get("W", 0)
    Y = counts.get("Y", 0)
    F = counts.get("F", 0)
    aromatic_weighted = 1.0 * W + 0.7 * Y + 0.5 * F
    aromatic_norm = min(1.0, max(0.0, aromatic_weighted / max(1.0, L / 2.0)))
    len_norm = triangular_len_score(L, 7, 9, 11, 20)
    return {
        "fly_charge_norm": charge_norm,
        "fly_surface_norm": surface_norm,
        "fly_aromatic_norm": aromatic_norm,
        "fly_len_norm": len_norm,
    }


def combine_flyability_score(
    components: Dict[str, float], fly_weights: Dict[str, float]
) -> float:
    w = {
        "charge": fly_weights.get("charge", 0.5),
        "surface": fly_weights.get("surface", 0.35),
        "len": fly_weights.get("len", 0.1),
        "aromatic": fly_weights.get("aromatic", 0.05),
    }
    Wsum = max(1e-9, sum(w.values()))
    for k in list(w.keys()):
        w[k] /= Wsum
    c = components
    fly = (
        w["charge"] * c["fly_charge_norm"]
        + w["surface"] * c["fly_surface_norm"]
        + w["len"] * c["fly_len_norm"]
        + w["aromatic"] * c["fly_aromatic_norm"]
    )
    return float(min(1.0, max(0.0, fly)))
