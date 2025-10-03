from wiqd_features import (
    kd_gravy,
    basicity_proxy,
)


def length_weight(L: int, L_min=7, L_lo=8, L_hi=11, L_max=13) -> float:
    """
    Trapezoid on [L_min, L_max] with plateau [L_lo, L_hi].
    = 0 outside [L_min, L_max]
    = linear up from L_min->L_lo
    = 1 on [L_lo, L_hi]
    = linear down from L_hi->L_max
    """
    if L <= L_min or L >= L_max:
        return 0.0
    if L < L_lo:
        return (L - L_min) / float(L_lo - L_min)
    if L <= L_hi:
        return 1.0
    return (L_max - L) / float(L_max - L_hi)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def predict_rt_min(
    seq: str,
    gradient_min: float = 20.0,
    *,
    # Length prior defaults are MHC-I friendly; adjust for tryptic/SRM if needed
    L_min: int = 7,
    L_lo: int = 8,
    L_hi: int = 11,
    L_max: int = 13,
    # Coefficients (kept compatible with your earlier code)
    k_len: float = 0.03,  # minutes per aa beyond 8 (scaled to 20-min gradient)
    k_basic: float = -0.15  # minutes per "basicity unit" (scaled to 20-min gradient)
) -> float:
    """
    Fast retention-time surrogate on [0, gradient_min].

    Components
    ----------
    base:  hydrophobicity anchor from KD GRAVY (scaled to gradient span)
    + len: small positive shift for length > 8 aa
    + basic: earlier-pulling penalty for basic residues, weighted by a
             trapezoidal length prior (short/long peptides penalize more)

    Depends on existing helpers:
      - kd_gravy(seq)
      - basicity_proxy(seq): 1.0*R + 0.8*K + 0.3*H + 0.2
    """
    if not seq:
        return float("nan")

    # --- Hydrophobicity anchor (KD GRAVY mapped into 0..gradient_min)
    gravy = kd_gravy(seq)  # helper
    frac = _clip01((gravy + 4.5) / 9.0)  # rough GRAVY span [-4.5, +4.5]
    base = 0.5 + (gradient_min - 1.0) * frac  # keep a small floor

    # --- Length adjustment (same scale behavior as your prior version)
    L = len(seq)
    length_adj = k_len * max(0, L - 8) * (gradient_min / 20.0)

    # --- Basicity penalty (use the proxy; drop its constant bias)
    #     Units ~ R + 0.8*K + 0.3*H
    basic_units = max(0.0, basicity_proxy(seq) - 0.2)
    wL = length_weight(L, L_min=L_min, L_lo=L_lo, L_hi=L_hi, L_max=L_max)
    basic_adj = k_basic * basic_units * wL * (gradient_min / 20.0)

    rt = base + length_adj + basic_adj
    # Clamp to valid gradient window
    if rt < 0.0:
        return 0.0
    if rt > gradient_min:
        return float(gradient_min)
    return float(rt)
