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
"""

import os, sys, math, argparse, datetime, random, re, bisect
from typing import List, Dict, Optional, Iterable
from dataclasses import dataclass

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== tqdm ===================
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# =================== Banner ===================
def banner(name: str = "WIQD", title: str = "What Is Qing Doing?", args=None):
    width = 68
    inner = width - 2
    bar = "+" + "=" * inner + "+"

    def center_line(s: str) -> str:
        return f"|{s:^{inner}}|"

    name_line = f"({name})"
    print("\n".join([bar, center_line(title), center_line(name_line), bar]))
    if args:
        shown = [
            "input_csv","outdir","alpha","numeric_test","permutes","tie_thresh",
            "min_score","min_nnz_topn",
            "k","collapse","housekeeping_fasta","download_housekeeping",
            "frag_tol_da","frag_tol_ppm",
            # Albumin + RT
            "albumin_source","albumin_fasta","albumin_acc","albumin_expected","albumin_use",
            "ppm_tol","full_mz_len_min","full_mz_len_max",
            "by_mz_len_min","by_mz_len_max","by_seq_len_min","by_seq_len_max",
            "rt_tolerance_min","gradient_min","require_precursor_match_for_frag",
            # Chemistry
            "charges","cys_mod",
            # Fly
            "fly_weights",
            # Summary
            "summary_top_n",
        ]
        print("Args:", " | ".join(f"{k}={getattr(args,k)}" for k in shown if hasattr(args, k)))
    print()

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

def _apply_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# =================== Chemistry & constants ===================
AA = set("ACDEFGHIKLMNPQRSTVWY")
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
EISEN = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74,
    "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19,
    "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
HOPP = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0,
    "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3,
    "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
}
MONO = {
    "A": 71.037113805, "R": 156.10111105, "N": 114.04292747, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.05857754, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.09496305,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.07931298, "Y": 163.063328575, "V": 99.068413945,
}
H2O = 18.010564684
PROTON_MASS = 1.007276466812
PKA = {
    "N_term": 9.69, "C_term": 2.34, "K": 10.5, "R": 12.5, "H": 6.0,
    "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1,
}
hydrophobic_set = set("AVILMFWYC")
HBD_SC = set("KRHSTYNQWC")
HBA_SC = set("DENQHSTYWC")
RES_ELEM = {
    "A": (3, 5, 1, 1, 0), "R": (6, 12, 4, 1, 0), "N": (4, 6, 2, 2, 0), "D": (4, 5, 1, 3, 0),
    "C": (3, 5, 1, 1, 1), "E": (5, 7, 1, 3, 0), "Q": (5, 8, 2, 2, 0), "G": (2, 3, 1, 1, 0),
    "H": (6, 7, 3, 1, 0), "I": (6, 11, 1, 1, 0), "L": (6, 11, 1, 1, 0), "K": (6, 12, 2, 1, 0),
    "M": (5, 9, 1, 1, 1), "F": (9, 9, 1, 1, 0), "P": (5, 7, 1, 1, 0), "S": (3, 5, 1, 2, 0),
    "T": (4, 7, 1, 2, 0), "W": (11, 10, 2, 1, 0), "Y": (9, 9, 1, 2, 0), "V": (5, 9, 1, 1, 0),
}
H2O_ELEM = (0, 2, 0, 1, 0)

def clean_peptide(p):
    return "".join(ch for ch in str(p).strip().upper() if ch in AA)

def kd_gravy(p):
    return sum(KD[a] for a in p) / len(p) if p else float("nan")

def mean_scale(p, scale):
    return sum(scale[a] for a in p) / len(p) if p else float("nan")

def kd_stdev(p):
    if not p: return float("nan")
    vals = [KD[a] for a in p]
    m = sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5

def max_hydro_streak(p):
    best = cur = 0
    for a in p:
        if a in hydrophobic_set:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best

def mass_monoisotopic(p):
    # Peptide masses are unmodified by design (albumin metrics apply Cys fixed mod internally)
    return sum(MONO[a] for a in p) + H2O

def mz_from_mass(m, z):
    return (m + z * PROTON_MASS) / z

def isoelectric_point(p):
    def net(ph):
        q = 1 / (1 + 10 ** (ph - PKA["N_term"])) - 1 / (1 + 10 ** (PKA["C_term"] - ph))
        for a in p:
            if a == "K": q += 1 / (1 + 10 ** (ph - PKA["K"]))
            elif a == "R": q += 1 / (1 + 10 ** (ph - PKA["R"]))
            elif a == "H": q += 1 / (1 + 10 ** (ph - PKA["H"]))
            elif a == "D": q -= 1 / (1 + 10 ** (PKA["D"] - ph))
            elif a == "E": q -= 1 / (1 + 10 ** (PKA["E"] - ph))
            elif a == "C": q -= 1 / (1 + 10 ** (PKA["C"] - ph))
            elif a == "Y": q -= 1 / (1 + 10 ** (PKA["Y"] - ph))
        return q
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if net(mid) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0

def net_charge_at_pH(p, ph=7.4):
    def frac_pos(pka): return 1 / (1 + 10 ** (ph - pka))
    def frac_neg(pka): return 1 / (1 + 10 ** (pka - ph))
    q = frac_pos(PKA["N_term"]) - frac_neg(PKA["C_term"])
    for a in p:
        if a == "K": q += frac_pos(PKA["K"])
        elif a == "R": q += frac_pos(PKA["R"])
        elif a == "H": q += frac_pos(PKA["H"])
        elif a == "D": q -= frac_neg(PKA["D"])
        elif a == "E": q -= frac_neg(PKA["E"])
        elif a == "C": q -= frac_neg(PKA["C"])
        elif a == "Y": q -= frac_neg(PKA["Y"])
    return q

def aliphatic_index(p):
    if not p: return float("nan")
    L = len(p)
    A = p.count("A") / L; V = p.count("V") / L; I = p.count("I") / L; Lc = p.count("L") / L
    return 100.0 * (A + 2.9 * V + 3.9 * (I + Lc))

def aromaticity(p):
    return (p.count("F") + p.count("Y") + p.count("W")) / len(p) if p else float("nan")

def cterm_hydrophobic(p):
    return 1 if p and p[-1] in hydrophobic_set else 0

def tryptic_end(p):
    return 1 if p and p[-1] in {"K", "R"} else 0

def has_RP_or_KP(p):
    return 1 if ("RP" in p or "KP" in p) else 0

def xle_fraction(p):
    return (p.count("I") + p.count("L")) / len(p) if p else float("nan")

def aa_fraction(p, aa):
    return p.count(aa) / len(p) if p else float("nan")

def count_n_to_proline_bonds(p):
    return sum(1 for i in range(len(p) - 1) if p[i + 1] == "P")

def count_c_to_acidic_bonds(p):
    return sum(1 for i in range(len(p) - 1) if p[i] in ("D", "E"))

def hbd_hba_counts(p):
    return (sum(1 for a in p if a in HBD_SC), sum(1 for a in p if a in HBA_SC))

def elemental_counts(p):
    C = H = N = O = S = 0
    for a in p:
        c, h, n, o, s = RES_ELEM[a]
        C += c; H += h; N += n; O += o; S += s
    C += H2O_ELEM[0]; H += H2O_ELEM[1]; O += H2O_ELEM[3]
    return C, H, N, O, S

def approx_M1_rel_abundance(C, H, N, O, S):
    return 0.0107 * C + 0.000155 * H + 0.00364 * N + 0.00038 * O + 0.0075 * S

def count_basic(p):  return sum(1 for a in p if a in "KRH")
def count_acidic(p): return sum(1 for a in p if a in "DE")

def basicity_proxy(p):
    return 1.0 * p.count("R") + 0.8 * p.count("K") + 0.3 * p.count("H") + 0.2

# =================== Flyability ===================
KD_HYDRO = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

def triangular_len_score(L: int, zero_min: int = 7, peak_min: int = 9, peak_max: int = 11, zero_max: int = 20) -> float:
    if L <= zero_min: return 0.0
    if L >= zero_max: return 0.0
    if L < peak_min: return (L - zero_min) / max(1, (peak_min - zero_min))
    if L <= peak_max: return 1.0
    return max(0.0, (zero_max - L) / max(1, (zero_max - peak_max)))

def compute_flyability_components(seq: str) -> Dict[str, float]:
    L = len(seq)
    if L == 0:
        return {"fly_charge_norm": 0.0, "fly_surface_norm": 0.0, "fly_aromatic_norm": 0.0, "fly_len_norm": 0.0}
    counts = {aa: 0 for aa in KD_HYDRO.keys()}
    for aa in seq:
        if aa in counts: counts[aa] += 1
    R = counts.get("R", 0); K = counts.get("K", 0); H = counts.get("H", 0)
    D = counts.get("D", 0); E = counts.get("E", 0)
    charge_sites = 1.0 + R + K + 0.7 * H
    acid_penalty = 0.1 * (D + E)
    charge_norm = 1.0 - math.exp(-(max(0.0, charge_sites - acid_penalty)) / 2.0)
    gravy = sum(KD_HYDRO.get(a, 0.0) for a in seq) / L
    surface_norm = min(1.0, max(0.0, (gravy + 4.5) / 9.0))
    W = counts.get("W", 0); Y = counts.get("Y", 0); F = counts.get("F", 0)
    aromatic_weighted = 1.0 * W + 0.7 * Y + 0.5 * F
    aromatic_norm = min(1.0, max(0.0, aromatic_weighted / max(1.0, L / 2.0)))
    len_norm = triangular_len_score(L, 7, 9, 11, 20)
    return {
        "fly_charge_norm": charge_norm,
        "fly_surface_norm": surface_norm,
        "fly_aromatic_norm": aromatic_norm,
        "fly_len_norm": len_norm,
    }

def combine_flyability_score(components: Dict[str, float], fly_weights: Dict[str, float]) -> float:
    w = {
        "charge": fly_weights.get("charge", 0.5),
        "surface": fly_weights.get("surface", 0.35),
        "len": fly_weights.get("len", 0.1),
        "aromatic": fly_weights.get("aromatic", 0.05),
    }
    Wsum = max(1e-9, sum(w.values()))
    for k in list(w.keys()): w[k] /= Wsum
    c = components
    fly = (w["charge"] * c["fly_charge_norm"] +
           w["surface"] * c["fly_surface_norm"] +
           w["len"] * c["fly_len_norm"] +
           w["aromatic"] * c["fly_aromatic_norm"])
    return float(min(1.0, max(0.0, fly)))

# =================== Albumin acquisition ===================
ALBUMIN_P02768_MATURE = (
    "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFA"
    "KTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMC"
    "TAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRL"
    "KCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSIS"
    "SKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLL"
    "LRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVST"
    "PTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVD"
    "ETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCF"
    "AEEGKKLVAASQAALGL"
)
assert len(ALBUMIN_P02768_MATURE) == 585

def _parse_fasta_text_to_seq(text: str) -> str:
    text = "".join([ln.strip() for ln in text.splitlines() if not ln.startswith(">")])
    seq = re.sub(r"[^A-Z]", "", text.upper())
    return seq

def fetch_albumin_fasta_from_uniprot(acc: str, timeout: float = 10.0) -> str:
    import urllib.request
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        return _parse_fasta_text_to_seq(data)
    except Exception as e:
        print(f"[wiqd][WARN] UniProt fetch failed for {acc}: {e}")
        return ""

def load_albumin_sequence(args) -> str:
    seq = ""
    source_used = None
    if args.albumin_source in ("file", "auto") and args.albumin_fasta:
        try:
            with open(args.albumin_fasta, "r", encoding="utf-8") as fh:
                seq = _parse_fasta_text_to_seq(fh.read()); source_used = f"file:{args.albumin_fasta}"
        except Exception as e:
            print(f"[wiqd][WARN] albumin FASTA read failed: {e}")
    if not seq and args.albumin_source in ("fetch", "auto"):
        for acc in args.albumin_acc.split(","):
            acc = acc.strip()
            if not acc: continue
            seq = fetch_albumin_fasta_from_uniprot(acc)
            if seq:
                source_used = f"uniprot:{acc}"; break
    if not seq:
        seq = ALBUMIN_P02768_MATURE
        source_used = source_used or "embedded:ALBUMIN_P02768_MATURE"
    L = len(seq)
    is_prepro = L == 609
    is_mature = L == 585
    ok_len = (
        (args.albumin_expected == "prepro" and is_prepro) or
        (args.albumin_expected == "mature" and is_mature) or
        (args.albumin_expected == "either" and (is_prepro or is_mature))
    )
    if not ok_len:
        print(f"[wiqd][WARN] albumin length {L} differs from expected (609 prepro or 585 mature). Proceeding.")
    if args.albumin_use == "mature" and L == 609:
        seq = seq[24:]; print("[wiqd] Trimmed albumin to mature 585 aa.")
    elif (args.albumin_use == "auto" and args.albumin_expected in ("mature", "either") and L == 609):
        seq = seq[24:]; print("[wiqd] Auto-trimmed albumin to mature 585 aa.")
    print(f"[wiqd] Albumin source: {source_used}; final length={len(seq)}")
    return seq

# =================== Housekeeping seqs & homology ===================
def parse_fasta(text: str):
    seqs = {}
    hdr = None; buf = []
    for line in text.splitlines():
        if not line: continue
        if line[0] == ">":
            if hdr:
                seqs[hdr] = "".join(buf).replace(" ", "").upper()
            hdr = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    if hdr:
        seqs[hdr] = "".join(buf).replace(" ", "").upper()
    return seqs

HOUSEKEEPING_ACCS = ["P60709","P04406","P68104","P13639","P06733","P68371","P07437","P62937","P06748","P11021","P27824","P07237","P02768","P01857","P01834"]

def load_housekeeping_sequences(housekeeping_fasta: str = None, download: bool = False):
    seqs = {}
    if housekeeping_fasta and os.path.isfile(housekeeping_fasta):
        with open(housekeeping_fasta, "r") as fh:
            seqs = parse_fasta(fh.read())
    elif download:
        try:
            import requests
            url = "https://rest.uniprot.org/uniprotkb/stream"
            query = " OR ".join(f"accession:{acc}" for acc in HOUSEKEEPING_ACCS)
            params = {"query": query, "format": "fasta", "includeIsoform": "false"}
            r = requests.get(url, params=params, timeout=45); r.raise_for_status()
            seqs = parse_fasta(r.text)
        except Exception as e:
            print(f"[wiqd] WARNING: download_housekeeping failed ({e}); using built-in fallback set.")
            seqs = {}
    if not seqs:
        seqs = {
            "sp|P60709|ACTB_HUMAN": "MEEEIAALVIDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLK",
            "sp|P04406|G3P_HUMAN": "MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGK",
            "sp|P11021|HSPA5_HUMAN": "MKWVLALVLLYLLLLGQAEAKDVKDLVVLVGGSTRIPKIQKLLQDFFNGKELNVLENEKEKK",
            "sp|P07237|PDIA1_HUMAN": "MDSKGSSQKGKGGQKKDGKDEDKDLPGKKPKVLVLVGIWGAALLLQGAKELKDEL",
            "sp|P01857|IGHG1_HUMAN": "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAMS",
            "sp|P27824|CANX_HUMAN": "MRTLQLLLLLAVPLLLGSAQQADKHTHTNTKDNKGKKK",
        }
    return seqs

def collapse_seq_mode(seq: str, collapse: str) -> str:
    s = seq.upper()
    if collapse in ("xle", "xle_de_qn", "xle+deqn"):
        s = s.replace("L", "J").replace("I", "J")
        if collapse in ("xle_de_qn", "xle+deqn"):
            s = s.replace("D","B").replace("E","B").replace("Q","Z").replace("N","Z")
    return s

def build_kmer_bank(seqs: Dict[str, str], k: int = 6, collapse: str = "xle"):
    bank = set()
    for name, seq in tqdm(seqs.items(), desc=f"[4/8] Building hk k-mer bank (k={k}, collapse={collapse})"):
        s = collapse_seq_mode(seq, collapse)
        for i in range(0, len(s) - k + 1):
            bank.add(s[i : i + k])
    return bank

def homology_features_with_params(peptide: str, kmer_bank: set, k: int, collapse: str = "xle"):
    if len(peptide) < k or not kmer_bank:
        return {"hk_kmer_hits": 0, "hk_kmer_frac": 0.0}
    collapsed = collapse_seq_mode(peptide, collapse)
    kms = [collapsed[i : i + k] for i in range(0, len(collapsed) - k + 1)]
    hits = sum(1 for km in kms if km in kmer_bank)
    return {"hk_kmer_hits": hits, "hk_kmer_frac": hits / max(1, len(kms))}

# =================== Doublets affecting fragmentation or synthesis ===================
FRAG_DOUBLETS = ["AP","SP","TP","GP","DP","EP","KP","RP","PP","PK","PR"]
SYN_RISK_DOUBLETS = [
    "DG","DS","DT","DN","DP",
    "NG","NS","NT","NN","QG","QS","QT","QN",
    "PP","KK","RR","KR","RK","CC","MM","MW","WM","WW","YY","FF","II","LL","VV",
]

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
        # remove per-dipeptide _present fields
    out["frag_doublet_total_count"] = float(frag_total)
    out["frag_doublet_total_frac"] = float(frag_total) / bonds
    # Proline aggregation (binary presence)
    # proline_after: any X–P among AP,SP,TP,GP,DP,EP,KP,RP,PP
    proline_after_set = {"AP","SP","TP","GP","DP","EP","KP","RP","PP"}
    proline_before_set = {"PK","PR","PP"}  # PP counts for both
    out["frag_dbl_proline_after"] = 1 if any(counts.get(di,0)>0 for di in proline_after_set) else 0
    out["frag_dbl_proline_before"] = 1 if any(counts.get(di,0)>0 for di in proline_before_set) else 0

    # Synthesis-risk doublets (keep totals; replace many *_present with one numeric)
    syn_total = 0
    for di in SYN_RISK_DOUBLETS:
        c = counts.get(di, 0)
        syn_total += c
        out[f"synth_dbl_{di}_count"] = float(c)
        out[f"synth_dbl_{di}_frac"] = float(c) / bonds
        # remove per-dipeptide _present fields
    out["synth_doublet_total_count"] = float(syn_total)
    out["synth_doublet_total_frac"] = float(syn_total) / bonds
    # One consolidated hardness metric
    out["hard_to_synthesize_residues"] = float(syn_total)
    return out

# =================== Ambiguity classes & confusability (independent of albumin) ===================
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

def fragment_confusability_features(pep: str, frag_tol_da: float = 0.02, frag_tol_ppm: float = None):
    L = len(pep)
    if L < 2:
        return {"confusable_bion_frac": float("nan"),"confusable_yion_frac": float("nan"),"confusable_ion_frac": float("nan")}
    prefix = [0.0] * (L + 1)
    for i, ch in enumerate(pep, start=1):
        prefix[i] = prefix[i - 1] + MONO[ch]
    def tol_at(mass): return ((frag_tol_ppm * mass * 1e-6) if (frag_tol_ppm is not None) else frag_tol_da)
    present = {key for key, (members, _) in AMBIG_CLASSES.items() if any((aa in members) for aa in pep)}
    if not present:
        return {"confusable_bion_frac": 0.0,"confusable_yion_frac": 0.0,"confusable_ion_frac": 0.0}
    conf_b = conf_y = 0; total = L - 1
    for i in range(1, L):
        b_mass = prefix[i]; y_mass = prefix[L] - prefix[i]
        informative_b = informative_y = False
        for key in present:
            members, delta = AMBIG_CLASSES[key]
            if any((aa in members) for aa in pep[:i]):
                if delta > tol_at(b_mass): informative_b = True
            if any((aa in members) for aa in pep[i:]):
                if delta > tol_at(y_mass): informative_y = True
            if informative_b and informative_y: break
        if not informative_b: conf_b += 1
        if not informative_y: conf_y += 1
    frac_b = conf_b / total; frac_y = conf_y / total
    return {"confusable_bion_frac": frac_b,"confusable_yion_frac": frac_y,"confusable_ion_frac": 0.5 * (frac_b + frac_y)}

# =================== Predicted retention time ===================
def predict_rt_min(seq: str, gradient_min: float = 20.0) -> float:
    if not seq: return float("nan")
    gravy = kd_gravy(seq)
    frac = (gravy + 4.5) / 9.0
    base = 0.5 + max(0.0, gradient_min - 1.0) * min(1.0, max(0.0, frac))
    length_adj = 0.03 * max(0, len(seq) - 8) * (gradient_min / 20.0)
    basic_adj = -0.15 * count_basic(seq) * (gradient_min / 20.0)
    rt = base + length_adj + basic_adj
    return float(min(max(0.0, rt), gradient_min))

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
    return sum(MONO[a] for a in seq) + H2O + (cys_fixed_mod * seq.count("C"))

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
            mz_by_z = {z: (m + z * PROTON_MASS) / z for z in charges}
            rt = predict_rt_min(sub, gradient_min=gradient_min)
            windows.append(
                AlbWindow(
                    seq=sub, start0=s, end0=s + L, length=L, mass=m,
                    mz_by_z=mz_by_z, rt_min=rt, collapsed=collapse_seq_mode(sub, collapse),
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
            if wmz is None: continue
            if _ppm(pmz, wmz) <= ppm_tol:
                ok = True; break
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
            if wmz is None: continue
            if _ppm(pmz, wmz) <= ppm_tol:
                ok = True; break
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
    kmin = max(1, int(kmin)); kmax = min(int(kmax), max(0, Lp))
    mz_lists_by_z: Dict[int, List[float]] = {int(z): [] for z in charges}
    for k in range(kmin, kmax + 1):
        for i in range(0, Lp - k + 1):
            sub = pep[i:i+k]
            m = sum(MONO[a] for a in sub) + H2O + (cys_fixed_mod * sub.count("C"))
            for z in mz_lists_by_z.keys():
                mz_lists_by_z[z].append((m + z * PROTON_MASS) / z)
    for z in mz_lists_by_z.keys():
        mz_lists_by_z[z].sort()
    return {"mz_lists_by_z": mz_lists_by_z}

def _any_within_ppm(x: float, sorted_list: List[float], ppm_tol: float) -> bool:
    if not sorted_list:
        return False
    tol = x * ppm_tol * 1e-6
    i = bisect.bisect_left(sorted_list, x)
    if i < len(sorted_list) and abs(sorted_list[i] - x) <= tol: return True
    if i > 0 and abs(sorted_list[i - 1] - x) <= tol: return True
    if i + 1 < len(sorted_list) and abs(sorted_list[i + 1] - x) <= tol: return True
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
        # we assume albumin window lengths already constrained when pool is built
        for k in range(1, Lw + 1):
            for i in range(0, Lw - k + 1):
                sub = w.seq[i:i+k]
                cand += 1
                m = sum(MONO[a] for a in sub) + H2O + (cys_fixed_mod * sub.count("C"))
                hit = False
                for z in mz_lists_by_z.keys():
                    mz = (m + z * PROTON_MASS) / z
                    if _any_within_ppm(mz, mz_lists_by_z[z], ppm_tol):
                        hit = True; break
                if hit:
                    mz_count += 1
    return {"cand": cand, "mz_count": mz_count}

# =================== Stats helpers (MW / permutation U; BH FDR) ===================
def mannwhitney_u_p(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    combined = x + y
    if min(combined) == max(combined):
        return (float("nan"), float("nan"))
    data = [(v, 0) for v in x] + [(v, 1) for v in y]
    data.sort(key=lambda t: t[0])
    R = [0.0] * (n1 + n2)
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            R[k] = rank
        i = j
    R1 = sum(R[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    i = 0
    T = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        t = j - i
        if t > 1:
            T += t * (t * t - 1)
        i = j
    mu = n1 * n2 / 2.0
    sigma2 = (
        n1 * n2 * (n1 + n2 + 1) / 12.0
        - (n1 * n2 * T) / (12.0 * (n1 + n2) * (n1 + n2 - 1))
        if (n1 + n2) > 1
        else 0.0
    )
    sigma = (sigma2**0.5) if sigma2 > 0 else float("nan")
    if not (sigma == sigma) or sigma <= 0:
        return (U, float("nan"))
    z = (U - mu + 0.5) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / (2.0**0.5))))
    return (U, max(0.0, min(1.0, p)))

def ranks_avg(values):
    enumerated = list(enumerate(values))
    enumerated.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(enumerated):
        j = i
        while j < len(enumerated) and enumerated[j][1] == enumerated[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[enumerated[k][0]] = avg_rank
        i = j
    return ranks

def mannwhitney_U_only(x, y):
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = x + y
    ranks = ranks_avg(values)
    n1 = len(x)
    R1 = sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * len(y) - U1
    return min(U1, U2)

def perm_pvalue_U(x, y, iters=5000, rng_seed=123, progress_desc=None):
    import random
    try:
        import numpy as np
        use_np = True
    except Exception:
        use_np = False
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = x + y
    U_obs = mannwhitney_U_only(x, y)
    U_obs_min = min(U_obs, n1 * n2 - U_obs)
    cnt = 0
    iterable = range(iters)
    try:
        from tqdm.auto import tqdm as _tqdm
        iterable = _tqdm(iterable, desc=progress_desc) if progress_desc else iterable
    except Exception:
        pass
    if use_np:
        import numpy as np
        arr = np.array(pooled, dtype=float)
        for _ in iterable:
            np.random.shuffle(arr)
            xp = arr[:n1].tolist(); yp = arr[n1:].tolist()
            U_perm = mannwhitney_U_only(xp, yp)
            if min(U_perm, n1 * n2 - U_perm) <= U_obs_min:
                cnt += 1
    else:
        rng = random.Random(rng_seed)
        for _ in iterable:
            rng.shuffle(pooled)
            xp = pooled[:n1]; yp = pooled[n1:]
            U_perm = mannwhitney_U_only(xp, yp)
            if min(U_perm, n1 * n2 - U_perm) <= U_obs_min:
                cnt += 1
    return (cnt + 1) / (iters + 1)

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    gt = 0; lt = 0
    for xi in x:
        for yj in y:
            if xi > yj: gt += 1
            elif xi < yj: lt += 1
    n = n1 * n2
    return (gt - lt) / n if n else float("nan")

def cohens_d_and_g(x, y):
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return (float("nan"), float("nan"))
    mx = sum(x) / n1; my = sum(y) / n2
    sx = (sum((v - mx) ** 2 for v in x) / (n1 - 1)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y) / (n2 - 1)) ** 0.5
    sp2 = (((n1 - 1) * sx * sx + (n2 - 1) * sy * sy) / (n1 + n2 - 2)
           if (n1 + n2 - 2) > 0 else float("nan"))
    if not (sp2 == sp2) or sp2 <= 0:
        return (float("nan"), float("nan"))
    d = (my - mx) / (sp2**0.5)
    J = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
    g = J * d
    return d, g

def fisher_exact(a, b, c, d):
    from math import comb
    n = a + b + c + d
    if n == 0:
        return float("nan")
    row1 = a + b; col1 = a + c
    if row1 == 0 or col1 == 0 or row1 == n or col1 == n:
        return float("nan")
    def pmf(x): return comb(col1, x) * comb(n - col1, row1 - x) / comb(n, row1)
    obs = pmf(a); p = 0.0
    lo = max(0, row1 - (n - col1)); hi = min(row1, col1)
    for x in range(lo, hi + 1):
        px = pmf(x)
        if px <= obs + 1e-15:
            p += px
    return max(0.0, min(1.0, p))

def bh_fdr(pvals: Dict[str, float]) -> Dict[str, float]:
    valid = [(k, v) for k, v in pvals.items() if isinstance(v, float) and v == v]
    m = len(valid)
    if m == 0:
        return {k: float("nan") for k in pvals.keys()}
    valid.sort(key=lambda kv: kv[1])
    qs = [0.0] * m
    for i, (_, p) in enumerate(valid):
        qs[i] = p * m / (i + 1)
    for i in range(m - 2, -1, -1):
        qs[i] = min(qs[i], qs[i + 1])
    out = {}
    for i, (k, _) in enumerate(valid):
        out[k] = min(qs[i], 1.0)
    for k in pvals.keys():
        if k not in out:
            out[k] = float("nan")
    return out

def fmt_p(p: float) -> str:
    if not (isinstance(p, float) and p == p): return "NA"
    if p <= 0: return "<1e-300"
    if p < 1e-12: return "<1e-12"
    if p < 1e-3: return f"{p:.2e}"
    return f"{p:.3g}"

def fmt_q(q: float) -> str:
    return fmt_p(q)

# =================== Plot helpers ===================
def pretty_feature_name(feat: str) -> str:
    if feat.startswith("frag_dbl_") and feat.endswith("_count"):
        core = feat[len("frag_dbl_"):-len("_count")]
        return f"Fragmentation doublet {core} (count)"
    if feat.startswith("frag_dbl_") and feat.endswith("_frac"):
        core = feat[len("frag_dbl_"):-len("_frac")]
        return f"Fragmentation doublet {core} (fraction of bonds)"
    if feat == "frag_doublet_total_count": return "Fragmentation doublets (total count)"
    if feat == "frag_doublet_total_frac":  return "Fragmentation doublets (fraction of bonds)"
    if feat == "frag_dbl_proline_after":   return "Proline after (X–P) present"
    if feat == "frag_dbl_proline_before":  return "Proline before (P–K/R) present"

    if feat.startswith("synth_dbl_") and feat.endswith("_count"):
        core = feat[len("synth_dbl_"):-len("_count")]
        return f"Synthesis-risk doublet {core} (count)"
    if feat.startswith("synth_dbl_") and feat.endswith("_frac"):
        core = feat[len("synth_dbl_"):-len("_frac")]
        return f"Synthesis-risk doublet {core} (fraction of bonds)"
    if feat == "synth_doublet_total_count": return "Synthesis-risk doublets (total count)"
    if feat == "synth_doublet_total_frac":  return "Synthesis-risk doublets (fraction of bonds)"
    if feat == "hard_to_synthesize_residues": return "Hard-to-synthesize doublets (count)"

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
        "n_to_proline_bonds": "# potential N→Pro cleavages",
        "c_to_acidic_bonds": "# potential C→D/E cleavages",
        "HBD_sidechain": "# H-bond donors (side-chain)",
        "HBA_sidechain": "# H-bond acceptors (side-chain)",
        "elem_C": "C atoms","elem_H": "H atoms","elem_N": "N atoms","elem_O": "O atoms","elem_S": "S atoms",
        "approx_Mplus1_rel": "Approx. M+1 relative abundance",
        "mz_z1": "m/z (z=1)","mz_z2": "m/z (z=2)","mz_z3": "m/z (z=3)",
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
    if feat in special: return special[feat]
    if feat.startswith("frac_"): return f"Fraction of {feat.split('_',1)[1]}"
    return feat.replace("_", " ")

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)

def save_boxplot(x, y, feature, pval, qval, outpath):
    fig, ax = plt.subplots(constrained_layout=True)
    # Matplotlib 3.9 deprecation: use tick_labels
    bp = ax.boxplot([x, y], tick_labels=["non-hit", "hit"], showmeans=True, meanline=True)
    jitter = 0.08
    xs0 = [1 + (random.random() - 0.5) * 2 * jitter for _ in x]
    xs1 = [2 + (random.random() - 0.5) * 2 * jitter for _ in y]
    ax.scatter(xs0, x, alpha=0.4, s=12)
    ax.scatter(xs1, y, alpha=0.4, s=12)
    ax.set_title(pretty_feature_name(feature))
    ax.set_ylabel(pretty_feature_name(feature))
    n0, n1 = len(x), len(y)
    nnz0 = sum(1 for v in x if (isinstance(v, (int,float)) and v != 0))
    nnz1 = sum(1 for v in y if (isinstance(v, (int,float)) and v != 0))
    m0 = (sum(x)/n0) if n0 else float("nan")
    m1 = (sum(y)/n1) if n1 else float("nan")
    text = f"MW/perm(U) p={fmt_p(pval)}, q={fmt_q(qval)} | n={n0}/{n1} | nnz={nnz0}/{nnz1} | μ: {m0:.3g} vs {m1:.3g}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    try:
        for med in bp.get("medians", []): med.set_color("C1")
        for mean in bp.get("means", []): mean.set_color("C2")
        handles = ([bp["medians"][0], bp["means"][0]] if bp.get("medians") and bp.get("means") else [])
        labels = ["Median", "Mean"] if handles else []
        if handles: ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=9)
    except Exception:
        pass
    _apply_grid(ax)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def save_barplot_props(feature, a, b, c, d, pval, qval, outpath):
    n_hit = a + b; n_non = c + d
    p_hit = a / max(1, n_hit); p_non = c / max(1, n_non)
    fig, ax = plt.subplots(constrained_layout=True)
    xs = [0, 1]
    ax.bar(xs, [p_non, p_hit])
    jitter = 0.08
    pts_non = [1] * c + [0] * d
    pts_hit = [1] * a + [0] * b
    ax.scatter([0 + (random.random() - 0.5) * 0.16 for _ in pts_non], pts_non, alpha=0.4, s=12)
    ax.scatter([1 + (random.random() - 0.5) * 0.16 for _ in pts_hit], pts_hit, alpha=0.4, s=12)
    ax.set_xticks(xs, ["non-hit", "hit"])
    ax.set_ylabel(f"Proportion with {pretty_feature_name(feature)}")
    ax.set_title(pretty_feature_name(feature))
    text = f"Fisher p={fmt_p(pval)}, q={fmt_q(qval)} | counts: {c}/{n_non} vs {a}/{n_hit} | nnz={c}/{a}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.set_ylim(0, 1.05)
    _apply_grid(ax)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def save_grouped_bar(labels, vals_nonhit, vals_hit, title, ylabel, outpath, sig_mask=None):
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.45), 4.4))
    idx = list(range(len(labels))); width = 0.4
    ax.bar([i - width / 2 for i in idx], vals_nonhit, width=width, label="non-hit")
    ax.bar([i + width / 2 for i in idx], vals_hit, width=width, label="hit")
    ax.set_xticks(idx, labels, rotation=0)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
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

def save_summary_plot(stats_df: pd.DataFrame, alpha: float, outpath: str,
                      n_nonhit=None, n_hit=None, top_n: int = 20, min_nnz_topn: int = 4):
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
        lambda r: (neglog10(r["q_value"]) if (isinstance(r["q_value"], float) and r["q_value"] > 0)
                   else neglog10(r["p_value"])),
        axis=1,
    )
    df = df[df["score"] > 0].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.set_title("Feature summary — no non-zero signal scores"); ax.set_ylabel("Signal score")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "No detectable differences\n(all q≥α or insufficient nnz)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig); return
    top = df.sort_values("score", ascending=False).head(int(top_n) if isinstance(top_n, (int, float)) else 20)
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.32), 5))
    ax.bar(range(len(top)), top["score"].tolist())
    labels = []
    for _, r in top.iterrows():
        if r["type"] == "numeric":
            mh = r.get("mean_hit", float("nan")); mn = r.get("mean_nonhit", float("nan"))
            arrow = ("↑" if (isinstance(mh, float) and isinstance(mn, float) and mh > mn)
                     else ("↓" if (isinstance(mh, float) and isinstance(mn, float) and mh < mn) else ""))
        else:
            ph = r.get("prop_hit", float("nan")); pn = r.get("prop_nonhit", float("nan"))
            arrow = ("↑" if (isinstance(ph, float) and isinstance(pn, float) and ph > pn)
                     else ("↓" if (isinstance(ph, float) and isinstance(pn, float) and ph < pn) else ""))
        labels.append(f"{r['feature']}{arrow}")
    ax.set_xticks(range(len(top)), labels, rotation=90)
    ax.set_ylabel("Signal score")
    title = "Feature significance summary"
    if n_nonhit is not None and n_hit is not None:
        title += f" | n(non-hit)={n_nonhit}, n(hit)={n_hit} | nnz≥{min_nnz_topn}"
    ax.set_title(title)
    if alpha and alpha > 0:
        cutoff = -math.log10(alpha)
        ax.axhline(cutoff, linestyle="--", linewidth=1.0); ax.text(0.0, cutoff * 1.02, f"q={alpha:g}", va="bottom")
    _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

# =================== Core feature computation ===================
def compute_features(peptides: List[str], fly_weights: Dict[str, float], args) -> pd.DataFrame:
    rows = []
    for pep in tqdm(peptides, desc="[1/8] Computing core & flyability ..."):
        if not pep: continue
        Lp = len(pep)
        mass = mass_monoisotopic(pep)
        C, H, N, O, S = elemental_counts(pep)
        m1 = approx_M1_rel_abundance(C, H, N, O, S)
        hyd_frac = sum(1 for a in pep if a in hydrophobic_set) / Lp
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
        row = {
            "peptide": pep, "length": Lp, "mass_mono": mass,
            "gravy": kd_gravy(pep), "eisenberg_hydro": mean_scale(pep, EISEN),
            "hopp_woods_hydrophilicity": mean_scale(pep, HOPP), "kd_stdev": kd_std,
            "hydrophobic_fraction": hyd_frac, "max_hydrophobic_run": hydro_run,
            "pI": isoelectric_point(pep),
            "charge_pH2_0": net_charge_at_pH(pep, 2.0),
            "charge_pH2_7": net_charge_at_pH(pep, 2.7),
            "charge_pH7_4": net_charge_at_pH(pep, 7.4),
            "charge_pH10_0": net_charge_at_pH(pep, 10.0),
            "aliphatic_index": aliphatic_index(pep), "aromaticity": aromaticity(pep),
            "cterm_hydrophobic": 1 if pep and pep[-1] in hydrophobic_set else 0,
            "tryptic_end": 1 if pep and pep[-1] in {"K","R"} else 0,
            "has_RP_or_KP": has_RP_or_KP(pep),
            "has_C": 1 if "C" in pep else 0, "has_M": 1 if "M" in pep else 0,
            "has_W": 1 if "W" in pep else 0, "has_Y": 1 if "Y" in pep else 0,
            "xle_fraction": xle_fraction(pep),
            "basic_count": count_basic(pep), "acidic_count": count_acidic(pep),
            "basicity_proxy": basicity_proxy(pep),
            "proline_count": pep.count("P"),
            "proline_internal": 1 if (("P" in pep[1:-1]) if Lp > 2 else False) else 0,
            "n_to_proline_bonds": count_n_to_proline_bonds(pep),
            "c_to_acidic_bonds": count_c_to_acidic_bonds(pep),
            "HBD_sidechain": hbd_hba_counts(pep)[0],
            "HBA_sidechain": hbd_hba_counts(pep)[1],
            "elem_C": C, "elem_H": H, "elem_N": N, "elem_O": O, "elem_S": S,
            "approx_Mplus1_rel": m1,
            "mz_z1": mz_from_mass(mass, 1), "mz_z2": mz_from_mass(mass, 2), "mz_z3": mz_from_mass(mass, 3),
            "mass_defect": mass - int(mass),
            "frac_L_or_I": frac_L_or_I, "frac_K_or_Q": frac_K_or_Q, "frac_F_or_Mox": frac_F_or_Mox,
            "frac_N_or_D": frac_N_or_D, "frac_Q_or_E": frac_Q_or_E, "frac_ambiguous_union": frac_ambiguous_union,
            "fly_charge_norm": fcomp["fly_charge_norm"],
            "fly_surface_norm": fcomp["fly_surface_norm"],
            "fly_aromatic_norm": fcomp["fly_aromatic_norm"],
            "fly_len_norm": fcomp["fly_len_norm"],
            "fly_score": fly_score,
        }
        for a in sorted(AA):
            row[f"frac_{a}"] = aa_fraction(pep, a)
        # Doublet features (with proline collapse + hardness)
        row.update(doublet_features(pep))
        # Predicted RT (min)
        row["rt_pred_min"] = predict_rt_min(pep, gradient_min=args.gradient_min)
        rows.append(row)
    return pd.DataFrame(rows)

# =================== Assumption checks ===================
def write_assumptions_report(outdir: str, df_in: pd.DataFrame, feat: pd.DataFrame, args, albumin_seq: Optional[str]):
    lines = []
    lines.append("# WIQD Assumptions & Sanity Checks")
    lines.append(f"Input peptides: {len(df_in)} | unique: {df_in['peptide'].nunique() if 'peptide' in df_in.columns else 'NA'}")
    if "is_hit" in df_in.columns:
        uniq = sorted(pd.Series(df_in["is_hit"]).dropna().unique().tolist())
        lines.append(f"is_hit unique values: {uniq}")
        if set(uniq) - {0, 1}:
            lines.append("WARN: is_hit has values outside {0,1}. They were coerced if possible.")
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
    lines.append(f"RT gating applied where *_rt present: ±{args.rt_tolerance_min} min on a {args.gradient_min}‑min gradient.")
    if args.require_precursor_match_for_frag:
        lines.append("Fragment metrics are restricted to albumin windows that match precursor m/z within ppm tolerance (ungated and RT‑gated).")
    with open(os.path.join(outdir, "ASSUMPTIONS.txt"), "w") as fh:
        fh.write("\n".join(lines))

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
            if c.lower() == "score": score_col = c; break
        if score_col is None:
            raise ValueError("--min_score provided but no 'score' column found.")
        df["is_hit"] = (pd.to_numeric(df[score_col], errors="coerce") >= float(args.min_score)).astype(int)
    assert "is_hit" in df.columns, "Input CSV must include peptide,is_hit (or provide --min_score)."

    # Clean
    df["peptide"] = df["peptide"].map(clean_peptide)
    df = df[df["peptide"].str.len() > 0].copy()
    df["is_hit"] = df["is_hit"].astype(int)
    bad_is_hit = set(df["is_hit"].unique()) - {0,1}
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
        fragment_confusability_features(p, frag_tol_da=args.frag_tol_da, frag_tol_ppm=args.frag_tol_ppm)
        for p in tqdm(feat["peptide"], desc="[3/8] Fragment confusability ...")
    ]
    feat = pd.concat([feat, pd.DataFrame(conf_rows)], axis=1)

    # Housekeeping homology
    seqs = load_housekeeping_sequences(args.housekeeping_fasta, args.download_housekeeping)
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
        assert len(alb_windows_all) > 0, "Albumin windows index is empty; check albumin sequence"

        # Pre-filter pools for precursor length range (for counts and optional fragment gating)
        precursor_len_pool_all = [w for w in alb_windows_all if args.full_mz_len_min <= w.length <= args.full_mz_len_max]

        for _, r in tqdm(feat.iterrows(), total=len(feat), desc="[5/8] Albumin metrics ..."):
            pep = r["peptide"]
            pep_m = sum(MONO[a] for a in pep) + H2O + (cys_fixed_mod * pep.count("C"))
            pep_mz_by_z = {z: (pep_m + z * PROTON_MASS) / z for z in charges}
            pep_rt = predict_rt_min(pep, gradient_min=args.gradient_min)

            # Precompute peptide ANY-fragment m/z cache (across configured by_mz range on peptide)
            pep_frag_mz = _build_peptide_anyfrag_mz_cache(pep, charges, cys_fixed_mod, args.by_mz_len_min, args.by_mz_len_max)

            # RT pools
            rt_pool_precursor_len = _filter_rt(precursor_len_pool_all, pep_rt, args.rt_tolerance_min)

            # Count precursor matches (ungated and RT-gated) over the length-limited pools
            prec_match_count_all = _count_precursor_mz_matches(precursor_len_pool_all, pep_mz_by_z, args.ppm_tol)
            prec_match_count_rt  = _count_precursor_mz_matches(rt_pool_precursor_len,  pep_mz_by_z, args.ppm_tol)

            # Fragment pools
            frag_pool_all = precursor_len_pool_all
            frag_pool_rt  = rt_pool_precursor_len
            if args.require_precursor_match_for_frag:
                frag_pool_all = _filter_precursor_mz(precursor_len_pool_all, pep_mz_by_z, args.ppm_tol)
                frag_pool_rt  = _filter_precursor_mz(rt_pool_precursor_len,          pep_mz_by_z, args.ppm_tol)

            # Best precursor metrics (context)
            def precursor_mm_over(windows: List[AlbWindow]):
                best = {"ppm": float("inf"), "da": float("inf"), "z": None, "L": None}
                for w in windows:
                    if w.length < args.full_mz_len_min or w.length > args.full_mz_len_max: continue
                    for z, pmz in pep_mz_by_z.items():
                        wmz = w.mz_by_z.get(z)
                        if wmz is None: continue
                        da = abs(wmz - pmz)
                        ppmv = (da / pmz) * 1e6
                        if ppmv < best["ppm"]:
                            best.update({"ppm": ppmv, "da": da, "z": z, "L": w.length})
                return best
            pre_all = precursor_mm_over(alb_windows_all)
            cand_rt_all = _filter_rt(alb_windows_all, pep_rt, args.rt_tolerance_min)
            pre_rt  = precursor_mm_over(cand_rt_all)

            # ANY-fragment m/z counts (ungated + RT-gated)
            counts_mz_all = _fragment_any_mz_match_counts_over(frag_pool_all, pep_frag_mz, charges, cys_fixed_mod, args.ppm_tol)
            counts_mz_rt  = _fragment_any_mz_match_counts_over(frag_pool_rt,  pep_frag_mz, charges, cys_fixed_mod, args.ppm_tol)

            # Score (ppm→[0,1]) for best precursor m/z
            conf_prec_all = ppm_to_score(pre_all["ppm"], args.ppm_tol) if pre_all["ppm"] == pre_all["ppm"] else 0.0
            conf_prec_rt  = ppm_to_score(pre_rt["ppm"],  args.ppm_tol) if pre_rt["ppm"]  == pre_rt["ppm"]  else 0.0

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
    g0 = feat[feat["is_hit"] == 0]; g1 = feat[feat["is_hit"] == 1]
    n_nonhit = len(g0); n_hit = len(g1)

    numeric_feats = [
        "length","mass_mono","gravy","eisenberg_hydro","hopp_woods_hydrophilicity","kd_stdev",
        "hydrophobic_fraction","max_hydrophobic_run",
        "frag_doublet_total_count","frag_doublet_total_frac",
        "synth_doublet_total_count","synth_doublet_total_frac",
        "hard_to_synthesize_residues",
        "pI","charge_pH2_0","charge_pH2_7","charge_pH7_4","charge_pH10_0",
        "aliphatic_index","aromaticity","xle_fraction","basic_count","acidic_count","basicity_proxy",
        "n_to_proline_bonds","c_to_acidic_bonds","HBD_sidechain","HBA_sidechain",
        "elem_C","elem_H","elem_N","elem_O","elem_S",
        "approx_Mplus1_rel","mz_z1","mz_z2","mz_z3","mass_defect",
        "hk_kmer_hits","hk_kmer_frac",
        "frac_L_or_I","frac_K_or_Q","frac_F_or_Mox","frac_N_or_D","frac_Q_or_E","frac_ambiguous_union",
        "confusable_bion_frac","confusable_yion_frac","confusable_ion_frac",
        "fly_charge_norm","fly_surface_norm","fly_aromatic_norm","fly_len_norm","fly_score",
        # RT + albumin (counts only)
        "rt_pred_min",
        "alb_candidates_total","alb_candidates_rt",
        "alb_precursor_mz_match_count","alb_precursor_mz_match_count_rt",
        "alb_frag_candidates_mz","alb_frag_candidates_mz_rt",
        "alb_frag_mz_match_count","alb_frag_mz_match_count_rt",
        "alb_precursor_mz_best_ppm","alb_precursor_mz_best_da","alb_precursor_mz_best_len",
        "alb_confusability_precursor_mz",
        "alb_precursor_mz_best_ppm_rt","alb_precursor_mz_best_da_rt","alb_precursor_mz_best_len_rt",
        "alb_confusability_precursor_mz_rt",
    ]
    if albumin == "":
        numeric_feats = [f for f in numeric_feats if not f.startswith("alb_")]

    # Append per-motif doublet numeric features (still useful)
    for di in FRAG_DOUBLETS:
        for suffix in ("count", "frac"):
            col = f"frag_dbl_{di}_{suffix}"
            if col in feat.columns: numeric_feats.append(col)
    for di in SYN_RISK_DOUBLETS:
        for suffix in ("count", "frac"):
            col = f"synth_dbl_{di}_{suffix}"
            if col in feat.columns: numeric_feats.append(col)

    # Ensure dedup & existence
    seen = set(); numeric_feats = [f for f in numeric_feats if (f not in seen and not seen.add(f) and f in feat.columns)]

    # Binary feats
    binary_feats = [
        "cterm_hydrophobic","tryptic_end","has_RP_or_KP","has_C","has_M","has_W","has_Y","proline_internal",
        "has_L_or_I","has_K_or_Q","has_F_or_Mox","has_N_or_D","has_Q_or_E","has_ambiguous_union",
        "frag_dbl_proline_after","frag_dbl_proline_before",
    ]

    # Stats
    pmap = {}
    rows = []
    now = datetime.datetime.now().isoformat(timespec="seconds")

    # Numeric features
    for f in tqdm(numeric_feats, desc="[6/8] Testing numeric ..."):
        x = g0[f].dropna().tolist(); y = g1[f].dropna().tolist()
        med0 = float(pd.Series(x).median()) if x else float("nan")
        med1 = float(pd.Series(y).median()) if y else float("nan")
        mean0 = float(pd.Series(x).mean()) if x else float("nan")
        mean1 = float(pd.Series(y).mean()) if y else float("nan")
        uniq0 = len(set(x)); uniq1 = len(set(y))
        nnz0 = sum(1 for v in x if (isinstance(v, (int,float)) and v != 0))
        nnz1 = sum(1 for v in y if (isinstance(v, (int,float)) and v != 0))
        degenerate = False; reason = ""
        U = float("nan"); p_param = float("nan"); p_perm = float("nan")
        cliffs = float("nan"); d = float("nan"); g_ = float("nan")
        test_used = "none"
        if len(x) == 0 or len(y) == 0:
            degenerate = True; reason = "empty_group"; tie_fraction = float("nan"); is_discrete = 0
        else:
            combined = x + y
            if min(combined) == max(combined):
                degenerate = True; reason = "all_values_identical"; tie_fraction = float("nan"); is_discrete = 0
            else:
                from collections import Counter
                cnt = Counter(combined); ties = sum(c for c in cnt.values() if c > 1)
                tie_fraction = (ties / len(combined)) if combined else 0.0
                is_discrete = 1 if all(abs(v - round(v)) < 1e-12 for v in combined) else 0
                use_permU = (args.numeric_test == "perm_U") or (
                    args.numeric_test == "auto" and (is_discrete or tie_fraction > args.tie_thresh)
                )
                if use_permU and args.numeric_test != "mw":
                    U = mannwhitney_U_only(x, y)
                    p_perm = perm_pvalue_U(x, y, iters=args.permutes, rng_seed=123, progress_desc=f"[wiqd] permU {f}")
                    p_val = p_perm; test_used = "perm_U"
                else:
                    U, p_param = mannwhitney_u_p(x, y)
                    p_val = p_param; test_used = "mannwhitney_u"
                cliffs = cliffs_delta(x, y)
                d, g_ = cohens_d_and_g(x, y)
        pmap[f] = p_val if not degenerate else float("nan")
        rows.append(
            {
                "feature": f, "type": "numeric", "test_used": test_used,
                "n_nonhit": len(x), "n_hit": len(y),
                "unique_nonhit": uniq0, "unique_hit": uniq1,
                "median_nonhit": med0, "median_hit": med1,
                "mean_nonhit": mean0, "mean_hit": mean1,
                "sd_nonhit": float("nan"), "sd_hit": float("nan"),
                "var_nonhit": float("nan"), "var_hit": float("nan"),
                "U": U, "p_parametric": p_param, "p_perm_U": p_perm,
                "p_value": p_val if not degenerate else float("nan"), "q_value": float("nan"),
                "effect_cliffs_delta": cliffs, "effect_cohens_d": d, "effect_hedges_g": g_,
                "tie_fraction": (tie_fraction if not degenerate else float("nan")),
                "is_discrete": is_discrete if not degenerate else 0,
                "counts": "", "prop_nonhit": float("nan"), "prop_hit": float("nan"),
                "degenerate": 1 if degenerate else 0, "degenerate_reason": reason, "computed_at": now,
                "nnz_nonhit": nnz0, "nnz_hit": nnz1,
            }
        )

    # Binary features
    for f in tqdm(binary_feats, desc="[7/8] Testing binary ..."):
        a = int((g1[f] == 1).sum()); b = int((g1[f] == 0).sum())
        c = int((g0[f] == 1).sum()); d = int((g0[f] == 0).sum())
        n_hit = a + b; n_non = c + d
        prop_hit = a / max(1, n_hit); prop_non = c / max(1, n_non)
        degenerate = False; reason = ""
        p_val = float("nan"); test_used = "fisher_exact"
        if n_hit == 0 or n_non == 0:
            degenerate = True; reason = "empty_group"
        elif (a + b + c + d) == 0:
            degenerate = True; reason = "no_data"
        elif (a == 0 and c == 0) or (b == 0 and d == 0):
            degenerate = True; reason = "constant_feature"
        else:
            p_val = fisher_exact(a, b, c, d)
        pmap[f] = p_val if not degenerate else float("nan")
        rows.append(
            {
                "feature": f, "type": "binary", "test_used": test_used,
                "n_nonhit": n_non, "n_hit": n_hit,
                "unique_nonhit": float("nan"), "unique_hit": float("nan"),
                "median_nonhit": float("nan"), "median_hit": float("nan"),
                "mean_nonhit": prop_non, "mean_hit": prop_hit,
                "sd_nonhit": float("nan"), "sd_hit": float("nan"),
                "var_nonhit": float("nan"), "var_hit": float("nan"),
                "U": float("nan"), "p_parametric": p_val, "p_perm_U": float("nan"),
                "p_value": p_val if not degenerate else float("nan"), "q_value": float("nan"),
                "effect_cliffs_delta": float("nan"), "effect_cohens_d": float("nan"), "effect_hedges_g": float("nan"),
                "tie_fraction": float("nan"), "is_discrete": 0,
                "counts": f"hit: {a}/{n_hit}; non-hit: {c}/{n_non}",
                "prop_nonhit": prop_non, "prop_hit": prop_hit,
                "degenerate": 1 if degenerate else 0, "degenerate_reason": reason, "computed_at": now,
                "nnz_nonhit": c, "nnz_hit": a,
            }
        )

    # FDR
    qmap = bh_fdr(pmap)
    for r in rows: r["q_value"] = qmap[r["feature"]]
    stats_df = pd.DataFrame(rows)

    # Order & save stats
    order = [
        "feature","type","test_used","n_nonhit","n_hit","unique_nonhit","unique_hit",
        "median_nonhit","median_hit","mean_nonhit","mean_hit","sd_nonhit","sd_hit",
        "var_nonhit","var_hit","U","p_parametric","p_perm_U","p_value","q_value",
        "effect_cliffs_delta","effect_cohens_d","effect_hedges_g",
        "tie_fraction","is_discrete","counts","prop_nonhit","prop_hit",
        "degenerate","degenerate_reason","computed_at",
        "nnz_nonhit","nnz_hit",
    ]
    for col in order:
        if col not in stats_df.columns:
            stats_df[col] = ("" if col in ("feature","type","test_used","degenerate_reason","counts","computed_at") else float("nan"))
    stats_df = stats_df[order]
    stats_df.to_csv(os.path.join(args.outdir, "stats_summary.csv"), index=False)

    # Per-feature plots
    for f in tqdm(numeric_feats, desc="[8/8] Plot numeric ..."):
        x = g0[f].dropna().tolist(); y = g1[f].dropna().tolist()
        if len(x) == 0 and len(y) == 0: continue
        pv = stats_df.loc[stats_df["feature"] == f, "p_value"].values[0]
        qv = stats_df.loc[stats_df["feature"] == f, "q_value"].values[0]
        sig = isinstance(qv, float) and qv == qv and qv < args.alpha
        save_boxplot(x, y, f, pv, qv, os.path.join(plots_dir, f"{safe_name(f)}__box{'.SIG' if sig else ''}.png"))
    for f in tqdm(binary_feats, desc="[8/8] Plot binary ..."):
        a = int((g1[f] == 1).sum()); b = int((g1[f] == 0).sum())
        c = int((g0[f] == 1).sum()); d = int((g0[f] == 0).sum())
        pv = stats_df.loc[stats_df["feature"] == f, "p_value"].values[0]
        qv = stats_df.loc[stats_df["feature"] == f, "q_value"].values[0]
        sig = isinstance(qv, float) and qv == qv and qv < args.alpha
        save_barplot_props(f, a, b, c, d, pv, qv, os.path.join(plots_dir, f"{safe_name(f)}__bar{'.SIG' if sig else ''}.png"))

    # AA composition (original)
    aa_labels = sorted(AA)
    mean_non = [float(g0[f"frac_{a}"].mean()) if not g0.empty else float("nan") for a in aa_labels]
    mean_hit = [float(g1[f"frac_{a}"].mean()) if not g1.empty else float("nan") for a in aa_labels]
    sig_mask = [
        ((stats_df.loc[stats_df["feature"] == f"frac_{a}", "q_value"].values[0] < args.alpha)
         if not stats_df.loc[stats_df["feature"] == f"frac_{a}", "q_value"].empty else False)
        for a in aa_labels
    ]
    save_grouped_bar(
        aa_labels, mean_non, mean_hit,
        "AA composition (mean per peptide) | '*' = q<α", "mean fraction",
        os.path.join(plots_dir, "aa_composition.SIG.png" if any(sig_mask) else "aa_composition.png"),
        sig_mask=sig_mask,
    )

    # Summary plot (with nnz filter)
    save_summary_plot(
        stats_df, args.alpha, os.path.join(args.outdir, "summary_feature_significance.png"),
        n_nonhit=n_nonhit, n_hit=n_hit, top_n=args.summary_top_n, min_nnz_topn=args.min_nnz_topn,
    )

    # Extra comparative scatter (keep safe)
    try:
        if albumin and "alb_precursor_mz_best_ppm" in feat.columns:
            fig, ax = plt.subplots()
            ax.set_title("Flyability vs Albumin precursor best ppm")
            ax.set_xlabel("Flyability score")
            ax.set_ylabel("Precursor best ppm (log scale)")
            ax.set_yscale("log")
            ax.scatter(g0["fly_score"], g0["alb_precursor_mz_best_ppm"], s=10, alpha=0.5, label="non-hit")
            ax.scatter(g1["fly_score"], g1["alb_precursor_mz_best_ppm"], s=10, alpha=0.5, label="hit")
            ax.legend(); _apply_grid(ax)
            fig.savefig(os.path.join(args.outdir, "scatter_fly_vs_albumin_ppm.png"), bbox_inches="tight"); plt.close(fig)
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
                    "- Flyability components/weights unchanged; small clamp bug fixed.",
                ]
            )
        )

def main():
    ap = argparse.ArgumentParser(
        description="WIQD — hit vs non-hit peptide feature analysis (+albumin counts & flyability; no peptide modification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--in", dest="input_csv", required=True, help="Input CSV with peptide,is_hit (or provide --min_score and a score column)")
    ap.add_argument("--outdir", default="wiqd_out", help="Output directory")
    # Stats
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    ap.add_argument(
        "--numeric_test", choices=["auto", "mw", "perm_U"], default="auto",
        help="Numeric test: auto (perm_U if discrete/tie-heavy), mw, or perm_U",
    )
    ap.add_argument("--permutes", type=int, default=5000, help="Permutation iterations for perm_U")
    ap.add_argument("--tie_thresh", type=float, default=0.25, help="Tie-fraction threshold to trigger perm_U in auto mode")
    ap.add_argument("--min_score", type=float, default=None, help="If set, derive is_hit = 1[score >= min_score]")
    ap.add_argument("--min_nnz_topn", type=int, default=4, help="Minimum nnz per group (non-zero counts in hits and non-hits) required to qualify for Top‑N summary")
    # Fragment confusability (independent of albumin)
    ap.add_argument("--frag_tol_da", type=float, default=0.02, help="Neutral mass tolerance (Da) for fragment confusability (indep.)")
    ap.add_argument("--frag_tol_ppm", type=float, default=None, help="Neutral mass tolerance (ppm) for fragment confusability (overrides Da if set)")
    # Housekeeping
    ap.add_argument("--k", type=int, default=6, help="k for homology")
    ap.add_argument("--collapse", choices=["none", "xle", "xle_de_qn", "xle+deqn"], default="xle", help="Collapsed alphabet for homology/albumin sequence checks")
    ap.add_argument("--housekeeping_fasta", default=None, help="Path to housekeeping FASTA (optional)")
    ap.add_argument("--download_housekeeping", action="store_true", help="Download housekeeping proteins from UniProt (requires internet)")
    # Albumin
    ap.add_argument("--albumin_source", choices=["embedded", "file", "fetch", "auto", "none"], default="embedded", help="Albumin source")
    ap.add_argument("--albumin_fasta", default=None, help="Albumin FASTA (if file/auto)")
    ap.add_argument("--albumin_acc", default="P02768", help="UniProt accession(s) for albumin fetch (comma-separated)")
    ap.add_argument("--albumin_expected", choices=["prepro", "mature", "either"], default="either", help="Expected albumin length check")
    ap.add_argument("--albumin_use", choices=["prepro", "mature", "auto"], default="mature", help="Which form to use")
    # Albumin matching controls
    ap.add_argument("--ppm_tol", type=float, default=30.0, help="PPM tolerance for albumin matching")
    ap.add_argument("--full_mz_len_min", type=int, default=5, help="Albumin window min length for PRECURSOR m/z")
    ap.add_argument("--full_mz_len_max", type=int, default=15, help="Albumin window max length for PRECURSOR m/z")
    ap.add_argument("--by_mz_len_min", type=int, default=2, help="Peptide fragment k-min for m/z (ANY contiguous subseq)")
    ap.add_argument("--by_mz_len_max", type=int, default=7, help="Peptide fragment k-max for m/z (ANY contiguous subseq)")
    ap.add_argument("--by_seq_len_min", type=int, default=2, help="(Retained for index bounds only)")
    ap.add_argument("--by_seq_len_max", type=int, default=7, help="(Retained for index bounds only)")
    # RT gating
    ap.add_argument("--rt_tolerance_min", type=float, default=1.0, help="RT co‑elution tolerance in minutes for RT‑gated albumin metrics")
    ap.add_argument("--gradient_min", type=float, default=20.0, help="Assumed gradient length in minutes used by RT predictor")
    ap.add_argument("--require_precursor_match_for_frag", action="store_true",
                    help="For fragment metrics, only albumin windows that also match PRECURSOR m/z within ppm tolerance are considered (ungated and RT‑gated)")
    # Chemistry
    ap.add_argument("--charges", default="2,3", help="Charges to evaluate for albumin/fragment m/z metrics (e.g., 2,3)")
    ap.add_argument("--cys_mod", choices=["none", "carbamidomethyl"], default="carbamidomethyl", help="Fixed mod on Cys for albumin mass calc")
    # Flyability weights
    ap.add_argument("--fly_weights", default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05", help="Weights for flyability mix (sum normalized)")
    ap.add_argument("--summary_top_n", type=int, default=20, help="How many features to show in summary plot")
    args = ap.parse_args()
    banner(args=args)
    run_analysis(args)

if __name__ == "__main__":
    main()

