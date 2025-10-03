# =================== Chemistry & constants ===================
AA = set("ACDEFGHIKLMNPQRSTVWY")
KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
EISEN = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}
HOPP = {
    "A": -0.5,
    "R": 3.0,
    "N": 0.2,
    "D": 3.0,
    "C": -1.0,
    "Q": 0.2,
    "E": 3.0,
    "G": 0.0,
    "H": -0.5,
    "I": -1.8,
    "L": -1.8,
    "K": 3.0,
    "M": -1.3,
    "F": -2.5,
    "P": 0.0,
    "S": 0.3,
    "T": -0.4,
    "W": -3.4,
    "Y": -2.3,
    "V": -1.5,
}
MONO = {
    "A": 71.037113805,
    "R": 156.10111105,
    "N": 114.04292747,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.05857754,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.09496305,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.07931298,
    "Y": 163.063328575,
    "V": 99.068413945,
}
H2O = 18.010564684
PROTON_MASS = 1.007276466812
PKA = {
    "N_term": 9.69,
    "C_term": 2.34,
    "K": 10.5,
    "R": 12.5,
    "H": 6.0,
    "D": 3.9,
    "E": 4.1,
    "C": 8.3,
    "Y": 10.1,
}

HYDRO_SET = hydrophobic_set = AA_HYDRO_CORE = {"A", "V", "L", "I", "M", "F", "W"}
AA_HYDRO_CORE_PLUS_Y = AA_HYDRO_CORE.union(set("Y"))
AA_HYDRO_LENIENT = AA_HYDRO_CORE_PLUS_Y.union(set("C"))
# C-terminal hydrophobic residues
AA_HYDRO_CTERM = {"F", "I", "L", "V", "W", "Y", "M"}

HBD_SC = set("KRHSTYNQWC")
HBA_SC = set("DENQHSTY")
RES_ELEM = {
    "A": (3, 5, 1, 1, 0),
    "R": (6, 12, 4, 1, 0),
    "N": (4, 6, 2, 2, 0),
    "D": (4, 5, 1, 3, 0),
    "C": (3, 5, 1, 1, 1),
    "E": (5, 7, 1, 3, 0),
    "Q": (5, 8, 2, 2, 0),
    "G": (2, 3, 1, 1, 0),
    "H": (6, 7, 3, 1, 0),
    "I": (6, 11, 1, 1, 0),
    "L": (6, 11, 1, 1, 0),
    "K": (6, 12, 2, 1, 0),
    "M": (5, 9, 1, 1, 1),
    "F": (9, 9, 1, 1, 0),
    "P": (5, 7, 1, 1, 0),
    "S": (3, 5, 1, 2, 0),
    "T": (4, 7, 1, 2, 0),
    "W": (11, 10, 2, 1, 0),
    "Y": (9, 9, 1, 2, 0),
    "V": (5, 9, 1, 1, 0),
}
H2O_ELEM = (0, 2, 0, 1, 0)


C13_C12_DELTA = 1.00335483507  # 13C - 12C, for isotopologue shifts

# Adduct masses & polymer family data (not in wiqd_constants; retained locally)
ADDUCT_MASS = {"H": PROTON_MASS, "Na": 22.989218, "K": 38.963158, "NH4": 18.033823}
POLYMER_REPEAT_MASS = {
    "PEG": 44.026214747,
    "PPG": 58.041865,
    "PTMEG": 72.057515,
    "PDMS": 74.018792,
}
POLYMER_ENDGROUP = {
    "PEG": {"diol": H2O, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PPG": {"diol": H2O, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PTMEG": {"diol": H2O, "monoMe": 32.026214748, "diMe": 46.041864812},
    "PDMS": {"cyclic": 0.0},
}

# --- Dimethyl (per labeled amine; N-terminus or Lys ε-amine) ---
# Light:        +28.031300
# D4:           +32.056407  (d4-formaldehyde)
# 13C2D6:       +36.075670  (13C2D6-formaldehyde)
DIMETHYL_LIGHT = 28.031300
DIMETHYL_D4 = 32.056407
DIMETHYL_13C2D6 = 36.075670

# Reductive dimethylation channels (amine-reactive)
DIMETHYL_ML_DIFF = DIMETHYL_D4 - DIMETHYL_LIGHT  # 4.025107 per site
DIMETHYL_HL_DIFF = DIMETHYL_13C2D6 - DIMETHYL_LIGHT  # 8.044371 per site


# --- SILAC (per residue) ---
# K(+8.014199) = 13C6 15N2 Lys
# R(+10.008269) = 13C6 15N4 Arg
SILAC_K8 = 8.014199
SILAC_R10 = 10.008269

# ----------------------------- Local isotopic & tag constants -----------------
# 15N–14N monoisotopic mass difference (Da).
N15_N14_DELTA = 0.997034886

# --- mTRAQ NHS tags (per labeled amine; N-terminus and Lys ε-amine) ---
# Commonly used channels; Δ8 - Δ0 ≈ +8.015
# Verify exact monoisotopic masses against your reagent documentation.
MTRAQ_D0 = MTRAQ_DELTA0_NTERM = 140.094963
MTRAQ_D8 = MTRAQ_DELTA8_NTERM = 148.110413
MTRAQ_DELTA8_DIFF = MTRAQ_D8 - MTRAQ_D0  # 8.015450 per site


# 18O exchange at C-terminus
O18_DIFF = 2.004246  # per 18O atom

# 13C-12C mass difference (for isotope spacing checks)
C13_C12_DELTA = 1.003354835

# PTM masses (for Δm/z proximity checks)
COMMON_PTM_MASS = {
    "Oxidation": 15.994915,
    "Deamidation": 0.984016,
    "Acetyl": 42.010565,
    "Methyl": 14.015650,
    "Phospho": 79.966331,
}
