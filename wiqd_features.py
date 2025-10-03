from wiqd_constants import (
    KD,
    MONO,
    PROTON_MASS as PROTON,
    H2O,
    hydrophobic_set as HYDRO_SET,
    AA_HYDRO_CTERM,
    H2O_ELEM,
    RES_ELEM,
    PKA,
    HBA_SC,
    HBD_SC,
)


def mass_neutral(seq: str, cys_fixed_mod: float = 0.0) -> float:
    return sum(MONO[a] for a in seq) + H2O + cys_fixed_mod * seq.count("C")


def mz_from_mass(m: float, z: int) -> float:
    if z <= 0:
        raise ValueError("Charge must be positive.")
    return (m + z * PROTON) / z


def mz_from_sequence(seq: str, z: int, cys_fixed_mod: float = 0.0) -> float:
    return mz_from_mass(mass_neutral(seq, cys_fixed_mod), z)


def kd_gravy(seq: str) -> float:
    return sum(KD[a] for a in seq) / len(seq) if seq else float("nan")


def hydrophobic_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    return sum(1 for a in seq if a in HYDRO_SET) / len(seq)


def mean_scale(p, scale):
    return sum(scale[a] for a in p) / len(p) if p else float("nan")


def kd_stdev(p):
    if not p:
        return float("nan")
    vals = [KD[a] for a in p]
    m = sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def max_hydro_streak(p):
    best = cur = 0
    for a in p:
        if a in HYDRO_SET:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def mass_monoisotopic(p):  # unmodified (Cys fixed mod applied only in albumin metrics)
    return mass_neutral(p, 0.0)


def isoelectric_point(p):
    def net(ph):
        q = 1 / (1 + 10 ** (ph - PKA["N_term"])) - 1 / (1 + 10 ** (PKA["C_term"] - ph))
        for a in p:
            if a == "K":
                q += 1 / (1 + 10 ** (ph - PKA["K"]))
            elif a == "R":
                q += 1 / (1 + 10 ** (ph - PKA["R"]))
            elif a == "H":
                q += 1 / (1 + 10 ** (ph - PKA["H"]))
            elif a == "D":
                q -= 1 / (1 + 10 ** (PKA["D"] - ph))
            elif a == "E":
                q -= 1 / (1 + 10 ** (PKA["E"] - ph))
            elif a == "C":
                q -= 1 / (1 + 10 ** (PKA["C"] - ph))
            elif a == "Y":
                q -= 1 / (1 + 10 ** (PKA["Y"] - ph))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def net_charge_at_pH(p, ph=7.4):
    def frac_pos(pka):
        return 1 / (1 + 10 ** (ph - pka))

    def frac_neg(pka):
        return 1 / (1 + 10 ** (pka - ph))

    q = frac_pos(PKA["N_term"]) - frac_neg(PKA["C_term"])
    for a in p:
        if a == "K":
            q += frac_pos(PKA["K"])
        elif a == "R":
            q += frac_pos(PKA["R"])
        elif a == "H":
            q += frac_pos(PKA["H"])
        elif a == "D":
            q -= frac_neg(PKA["D"])
        elif a == "E":
            q -= frac_neg(PKA["E"])
        elif a == "C":
            q -= frac_neg(PKA["C"])
        elif a == "Y":
            q -= frac_neg(PKA["Y"])
    return q


def aliphatic_index(p):
    if not p:
        return float("nan")
    L = len(p)
    A = p.count("A") / L
    V = p.count("V") / L
    I = p.count("I") / L
    Lc = p.count("L") / L
    return 100.0 * (A + 2.9 * V + 3.9 * (I + Lc))


def aromaticity(p):
    return (p.count("F") + p.count("Y") + p.count("W")) / len(p) if p else float("nan")


def cterm_hydrophobic(p):
    return 1 if p and p[-1] in AA_HYDRO_CTERM else 0


def tryptic_end(p):
    return 1 if p and p[-1] in {"K", "R"} else 0


def has_RP_or_KP(p):
    return 1 if ("RP" in p or "KP" in p) else 0


def xle_fraction(p):
    return (p.count("I") + p.count("L")) / len(p) if p else float("nan")


def aa_fraction(p, aa):
    return p.count(aa) / len(p) if p else float("nan")


def hbd_hba_counts(p):
    return (sum(1 for a in p if a in HBD_SC), sum(1 for a in p if a in HBA_SC))


def elemental_counts(p):
    C = H = N = O = S = 0
    for a in p:
        c, h, n, o, s = RES_ELEM[a]
        C += c
        H += h
        N += n
        O += o
        S += s
    C += H2O_ELEM[0]
    H += H2O_ELEM[1]
    O += H2O_ELEM[3]
    return C, H, N, O, S


def approx_M1_rel_abundance(C, H, N, O, S):
    # crude linearized estimate of relative M+1 abundance from elemental counts
    return 0.0107 * C + 0.000155 * H + 0.00364 * N + 0.00038 * O + 0.0075 * S


def count_basic(p):
    return sum(1 for a in p if a in "KRH")


def count_acidic(p):
    return sum(1 for a in p if a in "DE")


def basicity_proxy(p):
    return 1.0 * p.count("R") + 0.8 * p.count("K") + 0.3 * p.count("H") + 0.2
