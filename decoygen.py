#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoygen.py — Cys+Pro+Met decoy generator with albumin m/z+RT chaining

What this script does:
  1) Parse/validate peptide rows; support mutation parsing from a "Gene_AA_Change" column.
  2) Base filters (strict as requested):
       - original peptide has NO Met, NO Cys, NO Proline
       - WT != 'C' and MUT != 'C'
       - peptide ends with a hydrophobic residue (A I L M F W Y V)
       - mutation is NOT at the last residue (within peptide)
     Prints remaining counts after each filter stage.
  3) Construct variants:
       - Replace mutant residue with Cys at the resolved mutation index.
       - Enumerate replacements of two additional residues: one to **Proline** and one to **Methionine**, excluding the C-terminus.
         **Constraint:** Proline position must lie **between** the Cys site and the Methionine site along the sequence
         (either C < P < M or M < P < C). Keep C at the mutation site unchanged; do not alter the last residue.
       - Compute a "fly" score for each candidate and keep the **highest-fly** candidate for that peptide.
  4) Albumin confusability (focused on albumin only):
       - Build albumin subsequences (k=5..15) with monoisotopic masses and a KD-based RT surrogate, map both windows
         and peptides to a 0..RT_total_min min scale (default 20 min).
       - For each peptide, compute:
            * N_precursor_mz       — #windows matched at precursor m/z (±ppm, z∈{1,2,3})
            * N_precursor_mz_rt    — #windows matched at precursor m/z AND |ΔRT| ≤ rt_tol (default 1.0 min)
            * N_fragment_mz_given_precursor         — #fragment types (b/y, k=2..11, z∈{1,2,3}) matched within ±ppm among the windows that matched precursor m/z
            * N_fragment_mz_given_precursor_rt      — same, but restricted to windows that matched precursor m/z **and** RT
            * Frac_* versions (divide by #fragment types considered for that peptide)
         "Chained" RT→precursor→fragment evidence = N_fragment_mz_given_precursor_rt.
       - "Confusability_simple" (bounded [0,1]) as in your pasted script:
            mean( Norm_N_precursor_mz,
                  Norm_N_precursor_mz_rt,
                  Frac_fragment_mz_given_precursor,
                  Frac_fragment_mz_given_precursor_rt ),
         where Norm_* are capped at the cohort 95th percentile and then scaled to [0,1].
  5) Rank final **Cys+Pro+Met** decoys by Confusability_simple (descending) and emit outputs:
       - *_decoy_features.csv (everything)
       - *_selected_top_N.csv
       - *_all_with_stage_labels.csv (includes dropped rows with reasons)
       - *_stage_counts.csv
       - *_decoys_cysPM.fasta (headers propagate Gene_AA_Change, HLA, rank, Type when present)
       - plots/ (sequence transitions; exclusion bars; fly vs confusability; top-fly scatter)

Author: ChatGPT (GPT-5 Pro)
"""

from __future__ import annotations

import argparse, re, math, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

# tqdm with fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textwrap import wrap
from pathlib import Path

# ---------------- Albumin sequences ----------------

ALBUMIN_P02768_PREPRO_609 = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF"
    "EDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEP"
    "ERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLF"
    "FAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAV"
    "ARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLK"
    "ECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYAR"
    "RHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFE"
    "QLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVV"
    "LNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTL"
    "SEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLV"
    "AASQAALGL"
)
ALBUMIN_P02768_MATURE_585 = (
    "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPF"
    "EDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEP"
    "ERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLF"
    "FAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAV"
    "ARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLK"
    "ECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYAR"
    "RHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFE"
    "QLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVV"
    "LNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTL"
    "SEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLV"
    "AASQAALGL"
)

# ---------------- Mass / m/z ----------------

AA_MONO = {
    "A": 71.037113805, "R": 156.10111105, "N": 114.04292747, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.05857754, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.09496305,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.07931298, "Y": 163.063328575, "V": 99.068413945,
}
WATER = 18.010564684
PROTON = 1.007276466812

def calc_mass(seq: str, cys_fixed_mod: float = 0.0) -> float:
    m = 0.0
    for a in seq:
        if a not in AA_MONO:
            raise ValueError(f"Unknown residue '{a}' in '{seq}'")
        m += AA_MONO[a]
        if a == "C" and cys_fixed_mod:
            m += cys_fixed_mod
    return m + WATER

def mz_from_mass(M: float, z: int) -> float:
    return (M + z*PROTON) / z

# ---------------- RT surrogate ----------------

KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4,
    "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
HYDRO = set("AILMFWYV")

def rt_raw(seq: str) -> float:
    if not seq: return 0.0
    vals = [KD.get(a, 0.0) for a in seq]
    return float(np.mean(vals) + 0.5 * (sum(a in HYDRO for a in seq)/max(1,len(seq))))

def map_to_minutes(raw: np.ndarray, rt_total_min: float) -> np.ndarray:
    if raw.size == 0: return raw
    vmin = float(np.min(raw)); vmax = float(np.max(raw))
    if math.isclose(vmin, vmax): return np.full_like(raw, rt_total_min/2.0)
    return (raw - vmin) / (vmax - vmin) * rt_total_min

# ---------------- Flyability ----------------

def triangular_len_score(L: int, zero_min: int = 7, peak_min: int = 9, peak_max: int = 11, zero_max: int = 20) -> float:
    if L <= zero_min: return 0.0
    if L >= zero_max: return 0.0
    if L < peak_min: return (L - zero_min) / max(1, (peak_min - zero_min))
    if L <= peak_max: return 1.0
    return max(0.0, (zero_max - L) / max(1, (zero_max - peak_max)))

def fly_components(seq: str) -> Dict[str, float]:
    L = len(seq)
    if L == 0: return {"charge":0.0,"surface":0.0,"aromatic":0.0,"len":0.0}
    counts = {a: seq.count(a) for a in AA_MONO}
    R,K,H,D,E = counts["R"], counts["K"], counts["H"], counts["D"], counts["E"]
    charge_sites = 1.0 + R + K + 0.7*H
    acid_penalty = 0.1*(D+E)
    charge = 1.0 - math.exp(-max(0.0, charge_sites - acid_penalty)/2.0)
    gravy = np.mean([KD.get(a,0.0) for a in seq])
    surface = float(np.clip((gravy + 4.5)/9.0, 0.0, 1.0))
    W,Y,F = counts["W"], counts["Y"], counts["F"]
    aromatic = float(np.clip((1.0*W + 0.7*Y + 0.5*F)/max(1.0, L/2.0), 0.0, 1.0))
    length = triangular_len_score(L)
    return {"charge":float(charge), "surface":surface, "aromatic":aromatic, "len":float(length)}

def fly_score(seq: str, weights: Dict[str, float]) -> float:
    c = fly_components(seq)
    w = {"charge":weights.get("charge",0.5), "surface":weights.get("surface",0.35),
         "len":weights.get("len",0.1), "aromatic":weights.get("aromatic",0.05)}
    s = w["charge"]*c["charge"] + w["surface"]*c["surface"] + w["len"]*c["len"] + w["aromatic"]*c["aromatic"]
    return float(np.clip(s/(sum(w.values()) or 1.0), 0.0, 1.0))

# ---------------- Parsing and indexing ----------------

def parse_mutation_notation(s: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Parse strings like 'GENE_A123B' to (wt, pos, mut, gene). If not parseable, returns (None,None,None,None).
    """
    if s is None: return (None,None,None,None)
    s = str(s).strip()
    if not s: return (None,None,None,None)
    m = re.search(r'([A-Za-z0-9_.-]+)[_:\-]([A-Za-z])(\d+)([A-Za-z])$', s)
    if m:
        return (m.group(2).upper(), int(m.group(3)), m.group(4).upper(), m.group(1))
    m2 = re.search(r'([A-Za-z])(\d+)([A-Za-z])$', s)
    if m2:
        return (m2.group(1).upper(), int(m2.group(2)), m2.group(3).upper(), s[:m2.start()].strip("_:-"))
    return (None,None,None,None)

def resolve_mut_idx(peptide: str, pos: int, mode: str) -> int:
    L = len(peptide)
    if mode in ("N0","0"): return pos
    if mode in ("N1","1"): return pos - 1
    if mode == "C1": return L - pos         # 1-based from C-terminus
    if mode == "C0": return L - 1 - pos     # 0-based from C-terminus
    raise ValueError(f"Unknown indexing mode {mode}")

def detect_indexing(peptide: str, pos: int, wt: str, mut: str) -> Tuple[int, str]:
    modes = ["N1","N0","C1","C0"]
    for m in modes:
        idx = resolve_mut_idx(peptide, pos, m)
        if 0 <= idx < len(peptide):
            if peptide[idx] in (wt, mut):
                return idx, m
    # fallback: N1 clamped
    idx = resolve_mut_idx(peptide, pos, "N1")
    if not (0 <= idx < len(peptide)):
        raise ValueError(f"Position {pos} out of range for peptide of length {len(peptide)}")
    return idx, "N1"

# ---------------- Albumin windows ----------------

def albumin_windows(seq: str, kmin: int, kmax: int) -> List[Tuple[int,str,float,float]]:
    """
    Returns list of (start0, window, mass, rt_raw).
    """
    out = []
    N = len(seq)
    for k in range(kmin, kmax+1):
        if N < k: break
        for s in range(0, N-k+1):
            w = seq[s:s+k]
            out.append((s, w, calc_mass(w), rt_raw(w)))
    return out

# ---------------- Fragment m/z for a given sequence ----------------

def fragment_mz_list(seq: str, kmin: int, kmax: int, charges: List[int]) -> List[Tuple[float,str,int,int]]:
    """
    Returns list of (mz, ion, k, z) for b/y fragments at lengths kmin..kmax and charges in charges.
    """
    out = []
    L = len(seq)
    kmax_eff = min(kmax, max(kmin, L-1))
    for k in range(kmin, kmax_eff+1):
        bseq = seq[:k]
        yseq = seq[-k:]
        Mb = calc_mass(bseq) - WATER
        My = calc_mass(yseq)
        for z in charges:
            out.append(((Mb + z*PROTON)/z, "b", k, z))
            out.append(((My + z*PROTON)/z, "y", k, z))
    return out

# ---------------- Matching helpers ----------------

def ppm(a: float, b: float) -> float:
    return abs(a-b)/b*1e6 if b != 0 else float("inf")

def any_within_ppm(query_mz: float, target_mz_list: List[float], tol_ppm: float) -> bool:
    d = query_mz * tol_ppm * 1e-6
    # Since target list is small in chained stage, linear scan is OK
    for x in target_mz_list:
        if abs(x - query_mz) <= d:
            return True
    return False

# ---------------- Plot helper ----------------

def plot_sequence_transitions(rows_df: "pd.DataFrame", out_path: Path, max_peptides: int = 25) -> None:
    """
    Three-line rows: ORIG (mutant in orange) → CYS (C in red) → CYS+P+M (C in red, P in blue, M in green).
    """
    import matplotlib.pyplot as plt
    if rows_df is None or len(rows_df)==0: return
    df = rows_df.head(max_peptides).copy()

    plt.figure(figsize=(14, max(6, 0.6*len(df))))
    ax = plt.gca(); ax.set_axis_off()
    y_gap = 1.0; y0 = 0.95; ax.set_xlim(0, 120); ax.set_ylim(0, 1)

    col_mut = "#d95f02"; col_cys = "#d62728"; col_pro = "#1f77b4"; col_met = "#2ca02c"; default_col = "#000000"

    def draw(seq, y, color_idx_map, x_start=2):
        for i, ch in enumerate(seq):
            ax.text(x_start + i*0.5, y, ch, fontsize=11, family="DejaVu Sans Mono",
                    ha="left", va="center", color=color_idx_map.get(i, default_col))

    for r in df.itertuples(index=False):
        pep = getattr(r, "peptide")
        mut_idx0 = int(getattr(r, "mut_idx0"))
        cys_seq = getattr(r, "cys_seq")
        decoy = getattr(r, "decoy_seq")
        p_idx = getattr(r, "p_index0")
        m_idx = getattr(r, "m_index0")

        base_y = y0 - df.index.get_loc(r.Index if hasattr(r,'Index') else 0) * y_gap
        y1, y2, y3 = base_y, base_y-0.3, base_y-0.6

        draw(pep, y1, {mut_idx0: col_mut})
        draw(cys_seq, y2, {mut_idx0: col_cys})

        # In decoy, C may shift? We did replacements, not insertions — index preserved
        hi = {}
        hi[mut_idx0] = col_cys
        if isinstance(p_idx, (int,np.integer)): hi[int(p_idx)] = col_pro
        if isinstance(m_idx, (int,np.integer)): hi[int(m_idx)] = col_met
        draw(decoy, y3, hi)

        ax.text(0.5, y1, "ORIG", fontsize=10, ha="left", va="center")
        ax.text(0.5, y2, "CYS", fontsize=10, ha="left", va="center")
        ax.text(0.5, y3, "CYS+P+M", fontsize=10, ha="left", va="center")

        x0 = 2 + mut_idx0*0.5
        ax.annotate("", xy=(x0, y2+0.04), xytext=(x0, y1-0.04), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))
        if isinstance(p_idx,(int,np.integer)):
            xp = 2 + int(p_idx)*0.5
            ax.annotate("", xy=(xp, y3+0.04), xytext=(x0, y2-0.04), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))

    ax.set_title("Sequence transitions: mutant (orange) → cysteine (red) → proline (blue) + methionine (green)")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ---------------- Main ----------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Cys+Pro+Met decoy generator with albumin m/z+RT chaining")
    # IO/columns
    ap.add_argument("--input", required=True, help="Input CSV/TSV")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--pos-col", default="mutation_position")
    ap.add_argument("--wt-col", default="wt")
    ap.add_argument("--mut-col", default="mut")
    ap.add_argument("--mut-notation-col", default=None, help="e.g., Gene_AA_Change like AHNAK_I4716M")
    ap.add_argument("--indexing", choices=["auto","N0","N1","C0","C1","0","1"], default="auto")
    ap.add_argument("--output-prefix", default="decoygen")
    ap.add_argument("--outdir", default="decoygen_out")

    # Chemistry / mass
    ap.add_argument("--charges", default="1,2,3")
    ap.add_argument("--cys-mod", choices=["none","carbamidomethyl"], default="none")

    # Albumin RT+m/z chaining
    ap.add_argument("--alb-kmin", type=int, default=5)
    ap.add_argument("--alb-kmax", type=int, default=15)
    ap.add_argument("--ppm", type=float, default=30.0)
    ap.add_argument("--frag-kmin", type=int, default=2)
    ap.add_argument("--frag-kmax", type=int, default=11)
    ap.add_argument("--rt-total-min", type=float, default=20.0)
    ap.add_argument("--rt-tol", type=float, default=1.0)

    # Fly weights
    ap.add_argument("--fly-weights", default="charge:0.5,surface:0.35,len:0.1,aromatic:0.05")

    # Selection
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--strict-mutation-check", action="store_true", default=True)
    ap.add_argument("--no-strict-mutation-check", action="store_false", dest="strict_mutation_check")

    args = ap.parse_args(argv)

    outdir = Path(args.outdir); plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    # Albumin sequence
    albumin = ALBUMIN_P02768_MATURE_585
    if len(albumin) not in (585,609):
        raise RuntimeError(f"Albumin length unexpected: {len(albumin)}")

    # Load input
    if args.input.lower().endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t", dtype=str)
    else:
        df = pd.read_csv(args.input, dtype=str)
    df["_row_id"] = np.arange(len(df))

    # Column mapping and cleaning
    if args.peptide_col not in df.columns or args.pos_col not in df.columns:
        raise KeyError(f"Missing required columns: peptide-col={args.peptide_col}, pos-col={args.pos_col}")
    df["_peptide"] = df[args.peptide_col].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    def safe_int(x):
        try: return int(float(str(x).strip()))
        except Exception: return np.nan
    df["_pos_int"] = df[args.pos_col].apply(safe_int)
    if args.wt_col in df.columns:
        df["_wt"] = df[args.wt_col].astype(str).str.upper().str.strip()
    else:
        df["_wt"] = np.nan
    if args.mut_col in df.columns:
        df["_mut"] = df[args.mut_col].astype(str).str.upper().str.strip()
    else:
        df["_mut"] = np.nan

    if args.mut_notation_col and args.mut_notation_col in df.columns:
        parsed = df[args.mut_notation_col].apply(parse_mutation_notation).apply(pd.Series)
        parsed.columns = ["_wt_parsed","_pos_parsed","_mut_parsed","_gene_parsed"]
        df = pd.concat([df, parsed], axis=1)
        if df["_wt"].isna().all(): df["_wt"] = df["_wt_parsed"]
        if df["_mut"].isna().all(): df["_mut"] = df["_mut_parsed"]
        # Only use parsed pos when pos-col is missing/NaN
        df["_pos_int"] = np.where(pd.isna(df["_pos_int"]), df["_pos_parsed"], df["_pos_int"])

    # Ensure ints
    if df["_pos_int"].isna().any():
        bad = df[df["_pos_int"].isna()]
        raise ValueError(f"Non-integer mutation positions encountered:\n{bad[[args.pos_col]].to_string(index=False)}")

    # Charges list
    charges = sorted({int(x.strip()) for x in args.charges.split(",") if x.strip()})
    if not charges or any(z<=0 for z in charges):
        raise ValueError("Provide positive integer charge(s) via --charges.")
    cys_fixed_mod = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0

    # ---- Stage A: validate mutation letter and resolve index (with requested novel indexing) ----
    stage_labels = []
    records = []

    # Counters for filter reporting
    n_total = len(df)
    n_drop_wtmutC = 0
    n_drop_containsP = 0
    n_drop_containsC = 0
    n_drop_containsM = 0
    n_drop_nonhydro = 0
    n_drop_mut_last = 0
    n_drop_mut_mismatch = 0

    survivors = []  # dict rows

    for row in tqdm(df.to_dict("records"), total=len(df), desc="Parsing & base filtering"):
        pep = row["_peptide"]; wt = str(row.get("_wt")) if pd.notna(row.get("_wt")) else None
        mut = str(row.get("_mut")) if pd.notna(row.get("_mut")) else None
        pos = int(row["_pos_int"])

        # WT/MUT != C
        if (wt == "C") or (mut == "C"):
            n_drop_wtmutC += 1
            row["stage_label"] = "dropped_base_wt_or_mut_C"; stage_labels.append(row); continue

        # Base content: no C, no P, no M
        hasP = "P" in pep; hasC = "C" in pep; hasM = "M" in pep
        if hasP: n_drop_containsP += 1; row["stage_label"] = "dropped_base_has_proline"; stage_labels.append(row); continue
        if hasC: n_drop_containsC += 1; row["stage_label"] = "dropped_base_has_cysteine"; stage_labels.append(row); continue
        if hasM: n_drop_containsM += 1; row["stage_label"] = "dropped_base_has_methionine"; stage_labels.append(row); continue

        # Hydrophobic C-term
        if not pep or pep[-1] not in HYDRO:
            n_drop_nonhydro += 1; row["stage_label"] = "dropped_base_nonhydrophobic_Cterm"; stage_labels.append(row); continue

        # Resolve mutation index (auto or explicit)
        if args.indexing == "auto":
            mut_idx0, mode_used = detect_indexing(pep, pos, wt or "", mut or "")
        else:
            mut_idx0 = resolve_mut_idx(pep, pos, args.indexing); mode_used = args.indexing
        if not (0 <= mut_idx0 < len(pep)):
            n_drop_mut_mismatch += 1; row["stage_label"] = "dropped_bad_index"; stage_labels.append(row); continue

        # Mutation not last
        if mut_idx0 == len(pep)-1:
            n_drop_mut_last += 1; row["stage_label"] = "dropped_mutation_at_last"; stage_labels.append(row); continue

        # Validate letter at index matches WT or MUT (user explicitly requested this check)
        letter = pep[mut_idx0]
        match_type = "none"
        if wt and letter == wt:
            match_type = "equals_wt"
        if mut and letter == mut:
            match_type = "equals_mut" if match_type == "none" else "equals_both"
        ok = match_type in ("equals_wt","equals_mut","equals_both")
        row["_mut_idx0"] = mut_idx0; row["_indexing_used"] = mode_used; row["_mut_letter"] = letter; row["_mut_match"] = match_type
        if not ok:
            n_drop_mut_mismatch += 1
            row["stage_label"] = "dropped_mutation_letter_mismatch"
            stage_labels.append(row)
            if args.strict_mutation_check:
                continue  # drop
            # else keep, but flagged
        survivors.append(row)

    print("[INFO] Base filters summary:")
    print(f"  total={n_total}  kept={len(survivors)}  dropped={n_total-len(survivors)}")
    print(f"    • WT/MUT==C: {n_drop_wtmutC}")
    print(f"    • contains P: {n_drop_containsP}")
    print(f"    • contains C: {n_drop_containsC}")
    print(f"    • contains M: {n_drop_containsM}")
    print(f"    • non-hydrophobic C-term: {n_drop_nonhydro}")
    print(f"    • mutation at last: {n_drop_mut_last}")
    print(f"    • mutation letter mismatch: {n_drop_mut_mismatch}")

    # ---- Stage B: build Cys+Pro+Met candidates and keep highest-fly ----
    fly_w = {k: float(v) for k,v in (x.split(":") for x in args.fly_weights.split(","))}
    decoy_rows = []
    for row in tqdm(survivors, total=len(survivors), desc="Building Cys+P+M decoys"):
        pep = row["_peptide"]; L = len(pep); mut_idx0 = row["_mut_idx0"]
        # Replace mutant with Cys
        cys_chars = list(pep); cys_chars[mut_idx0] = "C"; cys_seq = "".join(cys_chars)

        # Enumerate P/M positions: exclude last residue; exclude mut_idx0; ensure P lies between C and M
        positions = [i for i in range(0, L-1) if i != mut_idx0]  # exclude last index
        candidates = []
        for p_idx in positions:
            for m_idx in positions:
                if m_idx == p_idx: continue
                # P between C and M
                if not ((mut_idx0 < p_idx < m_idx) or (m_idx < p_idx < mut_idx0)):
                    continue
                # Build candidate by replacement
                chars = list(cys_seq)
                chars[p_idx] = "P"
                chars[m_idx] = "M"
                cand = "".join(chars)
                # Keep hydrophobic last residue unchanged implicitly (we did not touch last)
                fs = fly_score(cand, fly_w)
                candidates.append((fs, cand, p_idx, m_idx))

        if not candidates:
            row["stage_label"] = "dropped_no_valid_PM_pair"
            stage_labels.append(row); continue

        # Select highest fly
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_fs, best_seq, p_idx, m_idx = candidates[0]

        out = dict(row)  # copy original columns
        out.update({
            "peptide": pep,
            "mut_idx0": mut_idx0,
            "indexing_used": row["_indexing_used"],
            "match_check": row["_mut_match"],
            "cys_seq": cys_seq,
            "decoy_seq": best_seq,
            "p_index0": int(p_idx),
            "m_index0": int(m_idx),
            "decoy_fly_score": float(best_fs),
        })
        decoy_rows.append(out)

    if not decoy_rows:
        raise RuntimeError("No decoys built after Cys+P+M step.")

    out_df = pd.DataFrame(decoy_rows)

    # ---- Stage C: Albumin RT + m/z chaining ----
    alb_kmin, alb_kmax = int(args.alb_kmin), int(args.alb_kmax)
    frag_kmin, frag_kmax = int(args.frag_kmin), int(args.frag_kmax)
    ppm_tol = float(args.ppm)
    rt_total = float(args.rt_total_min); rt_tol = float(args.rt_tol)

    # Precompute albumin windows
    alb_wins = albumin_windows(albumin, alb_kmin, alb_kmax)
    alb_df = pd.DataFrame(alb_wins, columns=["start0","window","mass","rt_raw"])
    # RT mapping on combined (peptides decoy_seq + albumin windows)
    pep_rt_raw = out_df["decoy_seq"].apply(rt_raw).values
    all_raw = np.concatenate([pep_rt_raw, alb_df["rt_raw"].values])
    all_min = map_to_minutes(all_raw, rt_total)
    out_df["rt_min"] = all_min[:len(out_df)]
    alb_df["rt_min"] = all_min[len(out_df):]

    # Precompute masses / m/z for decoys
    cys_mod = 57.021464 if args.cys_mod == "carbamidomethyl" else 0.0
    out_df["decoy_mass"] = out_df["decoy_seq"].apply(lambda s: calc_mass(s, cys_mod))
    for z in charges:
        out_df[f"decoy_mz_z{z}"] = out_df["decoy_mass"].apply(lambda M: mz_from_mass(M, z))

    # Helper: precursor match to albumin windows for a peptide
    def precursor_match_windows(row) -> Tuple[set, set]:
        mzs = [row[f"decoy_mz_z{z}"] for z in charges]
        matched = set()
        matched_rt = set()
        for i, wr in alb_df.iterrows():
            # RT gate first (within ±1 min)
            if abs(float(wr["rt_min"]) - float(row["rt_min"])) <= rt_tol:
                # precursor m/z check
                for z in charges:
                    # window mass -> m/z at same charge z
                    wmz = mz_from_mass(float(wr["mass"]), z)
                    if ppm(wmz, mzs[charges.index(z)]) <= ppm_tol:
                        matched_rt.add(int(i))
                        matched.add(int(i))  # rt subset also included in matched
                        break
            else:
                # allow precursor without RT for N_precursor_mz
                for z in charges:
                    wmz = mz_from_mass(float(wr["mass"]), z)
                    if ppm(wmz, mzs[charges.index(z)]) <= ppm_tol:
                        matched.add(int(i)); break
        return matched, matched_rt

    # Fragment m/z lists per decoy
    decoy_frags = {}
    for i, r in out_df.iterrows():
        key = int(i)
        decoy_frags[key] = fragment_mz_list(r["decoy_seq"], frag_kmin, frag_kmax, charges)

    # Compute chained counts
    N_precursor_mz_list = []
    N_precursor_mz_rt_list = []
    N_frag_any_list = []
    Frac_frag_any_list = []
    N_frag_rt_list = []
    Frac_frag_rt_list = []
    N_frag_prec_list = []
    Frac_frag_prec_list = []
    N_frag_prec_rt_list = []
    Frac_frag_prec_rt_list = []

    for i, r in tqdm(out_df.iterrows(), total=len(out_df), desc="Albumin m/z+RT chaining"):
        matched, matched_rt = precursor_match_windows(r)
        N_precursor_mz = float(len(matched))
        N_precursor_mz_rt = float(len(matched_rt))

        # Build window fragment m/z for matched sets on-the-fly
        def build_fr_mz_for_windows(idxs: set) -> List[float]:
            mzs = []
            for idx in idxs:
                wseq = alb_df.at[idx, "window"]
                frs = fragment_mz_list(wseq, frag_kmin, frag_kmax, charges)
                mzs.extend([mz for (mz,ion,k,z) in frs])  # ion-type-agnostic
            return mzs

        # Peptide fragments (typed)
        frs_pep = decoy_frags[int(i)]
        n_types = float(len(frs_pep))

        # ANY window (global) — optional for diagnostics
        all_win_mz = build_fr_mz_for_windows(set(alb_df.index))
        matched_types_any = set()
        for (mz,ion,k,z) in frs_pep:
            if any_within_ppm(mz, all_win_mz, ppm_tol):
                matched_types_any.add((ion,k,z))
        N_frag_any = float(len(matched_types_any)); Frac_frag_any = N_frag_any / max(1.0, n_types)

        # RT-gated ANY (diagnostic)
        rt_win_mz = build_fr_mz_for_windows(matched_rt)
        matched_types_rt = set()
        for (mz,ion,k,z) in frs_pep:
            if any_within_ppm(mz, rt_win_mz, ppm_tol):
                matched_types_rt.add((ion,k,z))
        N_frag_rt = float(len(matched_types_rt)); Frac_frag_rt = N_frag_rt / max(1.0, n_types)

        # Chained to precursor windows
        prec_win_mz = build_fr_mz_for_windows(matched)
        matched_types_prec = set()
        for (mz,ion,k,z) in frs_pep:
            if any_within_ppm(mz, prec_win_mz, ppm_tol):
                matched_types_prec.add((ion,k,z))
        N_frag_prec = float(len(matched_types_prec)); Frac_frag_prec = N_frag_prec / max(1.0, n_types)

        # Chained to precursor+RT windows (the requested metric)
        prec_rt_win_mz = build_fr_mz_for_windows(matched_rt)
        matched_types_prec_rt = set()
        for (mz,ion,k,z) in frs_pep:
            if any_within_ppm(mz, prec_rt_win_mz, ppm_tol):
                matched_types_prec_rt.add((ion,k,z))
        N_frag_prec_rt = float(len(matched_types_prec_rt)); Frac_frag_prec_rt = N_frag_prec_rt / max(1.0, n_types)

        N_precursor_mz_list.append(N_precursor_mz)
        N_precursor_mz_rt_list.append(N_precursor_mz_rt)
        N_frag_any_list.append(N_frag_any); Frac_frag_any_list.append(Frac_frag_any)
        N_frag_rt_list.append(N_frag_rt);   Frac_frag_rt_list.append(Frac_frag_rt)
        N_frag_prec_list.append(N_frag_prec); Frac_frag_prec_list.append(Frac_frag_prec)
        N_frag_prec_rt_list.append(N_frag_prec_rt); Frac_frag_prec_rt_list.append(Frac_frag_prec_rt)

    # Attach counts
    out_df["N_precursor_mz"] = N_precursor_mz_list
    out_df["N_precursor_mz_rt"] = N_precursor_mz_rt_list
    out_df["N_fragment_mz_any"] = N_frag_any_list
    out_df["Frac_fragment_mz_any"] = Frac_frag_any_list
    out_df["N_fragment_mz_rt"] = N_frag_rt_list
    out_df["Frac_fragment_mz_rt"] = Frac_frag_rt_list
    out_df["N_fragment_mz_given_precursor"] = N_frag_prec_list
    out_df["Frac_fragment_mz_given_precursor"] = Frac_frag_prec_list
    out_df["N_fragment_mz_given_precursor_rt"] = N_frag_prec_rt_list
    out_df["Frac_fragment_mz_given_precursor_rt"] = Frac_frag_prec_rt_list

    # Confusability_simple (albumin-only)
    def cap95(series: pd.Series) -> float:
        p = float(pd.to_numeric(series, errors="coerce").quantile(0.95))
        return p if p > 0 else 1.0
    c1 = cap95(out_df["N_precursor_mz"])
    c2 = cap95(out_df["N_precursor_mz_rt"])
    norm1 = (out_df["N_precursor_mz"] / c1).clip(0,1)
    norm2 = (out_df["N_precursor_mz_rt"] / c2).clip(0,1)
    conf_simple = (norm1 + norm2 + out_df["Frac_fragment_mz_given_precursor"].clip(0,1)
                   + out_df["Frac_fragment_mz_given_precursor_rt"].clip(0,1)) / 4.0
    out_df["Confusability_simple_albumin"] = conf_simple.astype(float)

    # For continuity, also compute the legacy albumin_confusability-like score using ppm to 0..1 (optional)
    # Here we proxy with normalized counts as above; you can re-attach your exact legacy metric if needed.

    # ---- Final selection: sort by Confusability_simple_albumin and take top N ----
    s2 = out_df.sort_values(by=["Confusability_simple_albumin","decoy_fly_score"],
                            ascending=[False, False], kind="mergesort")
    N = int(args.N)
    sel = s2.head(N).copy()

    # ---- Stage labels table ----
    dropped_df = pd.DataFrame(stage_labels) if stage_labels else pd.DataFrame(columns=df.columns.tolist()+["stage_label"])
    dropped_df = dropped_df.assign(peptide=dropped_df.get("_peptide"))
    for col in set(out_df.columns) - set(dropped_df.columns):
        dropped_df[col] = np.nan
    all_with = pd.concat([out_df.assign(stage_label="selected").loc[sel.index],
                          out_df.drop(sel.index).assign(stage_label="not_selected"),
                          dropped_df], ignore_index=True, sort=False)

    # ---- Write outputs ----
    prefix = args.output_prefix
    outpath = Path(args.outdir)
    out_df.to_csv(outpath / f"{prefix}_decoy_features.csv", index=False)
    sel.to_csv(outpath / f"{prefix}_selected_top_N.csv", index=False)
    all_with.to_csv(outpath / f"{prefix}_all_with_stage_labels.csv", index=False)

    counts = all_with["stage_label"].value_counts().rename_axis("stage").reset_index(name="count")
    counts.to_csv(outpath / f"{prefix}_stage_counts.csv", index=False)

    # FASTA
    fasta = outpath / f"{prefix}_decoys_cysPM.fasta"
    with fasta.open("w") as fh:
        for i, r in sel.reset_index(drop=True).iterrows():
            extras = []
            for k in ["Gene_AA_Change","HLA","rank","Type"]:
                if k in r and pd.notna(r[k]):
                    extras.append(f"{k}={r[k]}")
            if "_gene_parsed" in r and pd.notna(r["_gene_parsed"]):
                extras.append(f"gene={r['_gene_parsed']}")
            hdr = (f">decoy_{i+1}|orig={r['peptide']}|mut_idx0={int(r['mut_idx0'])}|"
                   f"wt={str(r.get('_wt'))}|mut={str(r.get('_mut'))}|"
                   f"p_idx0={int(r['p_index0'])}|m_idx0={int(r['m_index0'])}|"
                   f"indexing={r['indexing_used']}|" + "|".join(extras))
            fh.write(hdr + "\n")
            for line in wrap(str(r["decoy_seq"]), 60):
                fh.write(line + "\n")

    # ---- Plots ----
    # Sequence transitions for selected
    try:
        plot_sequence_transitions(sel, outpath / "plots" / f"{prefix}_sequence_transitions_selected.png", max_peptides=25)
    except Exception as e:
        print(f"[WARN] sequence transitions plot failed: {e}")

    # Exclusion bars
    try:
        cats = ["WT/MUT==C","contains P","contains C","contains M","non-hydrophobic C-term","mutation at last","mutation letter mismatch","no valid P/M pair"]
        vals = [
            int((all_with["stage_label"]=="dropped_base_wt_or_mut_C").sum()),
            int((all_with["stage_label"]=="dropped_base_has_proline").sum()),
            int((all_with["stage_label"]=="dropped_base_has_cysteine").sum()),
            int((all_with["stage_label"]=="dropped_base_has_methionine").sum()),
            int((all_with["stage_label"]=="dropped_base_nonhydrophobic_Cterm").sum()),
            int((all_with["stage_label"]=="dropped_mutation_at_last").sum()),
            int((all_with["stage_label"]=="dropped_mutation_letter_mismatch").sum()),
            int((all_with["stage_label"]=="dropped_no_valid_PM_pair").sum()),
        ]
        plt.figure()
        plt.bar(cats, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count"); plt.title("Exclusions at each stage")
        plt.tight_layout(); plt.savefig(outpath / "plots" / f"{prefix}_bar_exclusions.png"); plt.close()
    except Exception as e:
        print(f"[WARN] exclusion bar plot failed: {e}")

    # Fly vs Confusability (selected colored)
    try:
        plt.figure()
        sel_mask = all_with["stage_label"]=="selected"
        plt.scatter(all_with.loc[~sel_mask, "decoy_fly_score"], all_with.loc[~sel_mask, "Confusability_simple_albumin"], s=12, label="not selected", alpha=0.7)
        plt.scatter(all_with.loc[sel_mask, "decoy_fly_score"], all_with.loc[sel_mask, "Confusability_simple_albumin"], s=16, label="selected")
        plt.xlabel("decoy_fly_score"); plt.ylabel("Confusability_simple_albumin (0..1)")
        plt.title("Flyability vs Albumin confusability (simple)")
        plt.legend(); plt.tight_layout()
        plt.savefig(outpath / "plots" / f"{prefix}_scatter_fly_vs_confusability_simple.png"); plt.close()
    except Exception as e:
        print(f"[WARN] fly vs confusability scatter failed: {e}")

    # Among top-fly: precursor counts vs overall confusability
    try:
        thr = out_df["decoy_fly_score"].quantile(0.75)
        top_fly = out_df[out_df["decoy_fly_score"] >= thr].copy()
        plt.figure()
        plt.scatter(top_fly["N_precursor_mz_rt"], top_fly["Confusability_simple_albumin"], s=14, label="top-fly")
        mask_sel_top = top_fly.index.isin(sel.index)
        if mask_sel_top.any():
            plt.scatter(top_fly.loc[mask_sel_top, "N_precursor_mz_rt"], top_fly.loc[mask_sel_top, "Confusability_simple_albumin"], s=16, label="selected (top-fly)")
        plt.xlabel("# RT+precursor windows"); plt.ylabel("Confusability_simple_albumin (0..1)")
        plt.title("Top-fly: RT+precursor window count vs confusability")
        plt.legend(); plt.tight_layout()
        plt.savefig(outpath / "plots" / f"{prefix}_scatter_topfly_precursor_rt_vs_confusability.png"); plt.close()
    except Exception as e:
        print(f"[WARN] top-fly scatter failed: {e}")

    print(f"[INFO] Output directory: {outdir.resolve()}")
    print("[INFO] Wrote: features, selected, all_with_stage_labels, stage_counts, FASTA, and plots.")

if __name__ == "__main__":
    main()
