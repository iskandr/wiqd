
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiqd_mzscreen.py — m/z-subsequence screen against common proteins
Screens peptides (optionally only hits) for subsequence windows in a ~100-protein common panel
that share precursor m/z (within tolerance, given charges).
Outputs CSVs and readable PNG plots.
"""
import os, math, argparse, re, datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

# ---------- ASCII UI ----------
def banner(script_name: str, args: argparse.Namespace = None):
    title = "What Is Qing Doing?"
    bar = "+" + "="*66 + "+"
    lines = [
        bar,
        "|{:^66}|".format(title),
        "|{:^66}|".format(f"({script_name})"),
        bar,
    ]
    print("\n".join(lines))
    print(f"Start: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if args:
        arg_summary = [
            f"input={getattr(args, 'input_csv', None)}",
            f"outdir={getattr(args, 'outdir', None)}",
            f"k={getattr(args, 'kmin', None)}..{getattr(args, 'kmax', None)}",
            f"charges={getattr(args, 'charges', None)} tol={getattr(args, 'tol', None)} {getattr(args, 'unit', None)}",
            f"prot_fasta={getattr(args, 'prot_fasta', None)} download_common={getattr(args, 'download_common', None)}",
        ]
        print("Args: " + " | ".join(arg_summary))
    print()

# ---- Readability defaults for plots ----
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.figsize": (7.2, 4.6),
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---- tqdm optional ----
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# ---- Mass constants (monoisotopic) ----
PROTON_MASS = 1.007276466812
WATER_MASS = 18.010564684
MASS = {
    "A": 71.037113805, "R": 156.101111050, "N": 114.042927470, "D": 115.026943065,
    "C": 103.009184505, "E": 129.042593135, "Q": 128.058577540, "G": 57.021463735,
    "H": 137.058911875, "I": 113.084064015, "L": 113.084064015, "K": 128.094963050,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875, "S": 87.032028435,
    "T": 101.047678505, "W": 186.079312980, "Y": 163.063328575, "V": 99.068413945,
    "U": 150.953633405
}
AA = set(MASS.keys())

COMMON_ACCESSIONS = [
    "P60709","P04406","P07437","P11142","P08238","P68104","P05388","P62805",
    "P11021","P14625","P27824","P27797","P07237","P30101","Q15084","Q15061","Q9BS26",
    "P02768","P01857","P01859","P01860","P01861","P01876","P01877","P01871","P04229",
    "P01854","P01834","P0DOY2"
]

def parse_fasta(text: str) -> Dict[str, str]:
    seqs = {}
    hdr=None; buf=[]
    for line in text.splitlines():
        if not line: continue
        if line[0] == ">":
            if hdr:
                seqs[hdr] = "".join(buf).replace(" ", "").upper()
            hdr = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if hdr:
        seqs[hdr] = "".join(buf).replace(" ", "").upper()
    return seqs

def open_maybe_gzip(path: str):
    import gzip
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def load_common_proteins(prot_fasta: Optional[str], download_common: bool, accessions_path: Optional[str]) -> Dict[str, str]:
    seqs = {}
    if prot_fasta and os.path.isfile(prot_fasta):
        with open_maybe_gzip(prot_fasta) as fh:
            seqs = parse_fasta(fh.read())
            return seqs

    # Build accession list
    accs = []
    if accessions_path and os.path.isfile(accessions_path):
        with open(accessions_path, "r") as f:
            for line in f:
                t=line.strip()
                if t and not t.startswith("#"): accs.append(t.split()[0])
    else:
        accs = list(dict.fromkeys(COMMON_ACCESSIONS))

    if download_common:
        try:
            import requests
            url = "https://rest.uniprot.org/uniprotkb/stream"
            query = " OR ".join(f"accession:{a}" for a in accs)
            params = {"query": query, "format": "fasta", "includeIsoform": "false"}
            print("[mzscreen] downloading common proteins via UniProt REST...")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            seqs = parse_fasta(r.text)
            if len(seqs) == 0:
                print("[mzscreen] WARNING: UniProt returned no sequences; falling back to embedded minimal set.")
        except Exception as e:
            print(f"[mzscreen] WARNING: download_common failed ({e}); falling back to embedded minimal set.")

    if len(seqs) == 0:
        # Minimal embedded fallback (HSPA8)
        seqs = {"sp|P11142|HSP7C_HUMAN": (
            "MSKGPAVGIDLGTTYSCVGVFQHGKVEIIANDQGNRTTPSYVAFTDTERLIGDAAKNQVA"
            "MNPTNTVFDAKRLIGRRFDDAVVQSDMKHWPFMVVNDAGRPKVQVEYKGETKSFYPEEVS"
            "SMVLTKMKEIAEAYLGKTVTNAVVTVPAYFNDSQRQATKDAGTIAGLNVLRIINEPTAAA"
            "IAYGLDKKVGAERNVLIFDLGGGTFDVSILTIEDGIFEVKSTAGDTHLGGEDFDNRMVNH"
            "FIAEFKRKHKKDISENKRAVRRLRTACERAKRTLSSSTQASIEIDSLYEGIDFYTSITRA"
            "RFEELNADLFRGTLDPVEKALRDAKLDKSQIHDIVLVGGSTRIPKIQKLLQDFFNGKELN"
            "KSINPDEAVAYGAAVQAAILSGDKSENVQDLLLLDVTPLSLGIETAGGVMTVLIKRNTTI"
            "PTKQTQTFTTYSDNQPGVLIQVYEGERAMTKDNNLLGKFELTGIPPAPRGVPQIEVTFDI"
            "DANGILNVSAVDKSTGKENKITITNDKGRLSKEDIERMVQEAEKYKAEDEKQRDKVSSKN"
            "SLESYAFNMKATVEDEKLQGKINDEDKQKILDKCNEIINWLDKNQTAEKEEFEHQQKELE"
            "KVCNPIITKLYQSAGGMPGGMPGGFPGGGAPPSGGASSGPTIEEVD"
        )}
    return seqs

def calc_mass(seq: str) -> float:
    m = 0.0
    for a in seq:
        if a not in MASS: return float("nan")
        m += MASS[a]
    return m + WATER_MASS

def mz_from_mass(neutral_mass: float, z: int) -> float:
    return (neutral_mass + z*PROTON_MASS)/z

def mz_window_to_mass_range(center_mz: float, z: int, tol: float, unit: str) -> Tuple[float,float]:
    if unit.lower()=="ppm":
        delta_mz = center_mz * tol * 1e-6
    else:
        delta_mz = tol
    mzmin, mzmax = center_mz - delta_mz, center_mz + delta_mz
    mmin = z*mzmin - z*PROTON_MASS
    mmax = z*mzmax - z*PROTON_MASS
    return (min(mmin,mmax), max(mmin,mmax))

def collapse_xle(s: str) -> str:
    return "".join("J" if c in ("L","I") else c for c in s)

def build_window_index(seqs: Dict[str,str], kmin: int, kmax: int) -> pd.DataFrame:
    rows = []
    for hdr, seq in tqdm(seqs.items(), desc="[mzscreen] indexing protein windows"):
        clean = "".join(c for c in seq if c.isalpha()).upper()
        L = len(clean)
        for k in range(kmin, kmax+1):
            if L < k: continue
            for start in range(0, L-k+1):
                window = clean[start:start+k]
                if any(ch not in MASS for ch in window):
                    continue
                mass = calc_mass(window)
                rows.append({
                    "protein": hdr,
                    "start": start,
                    "length": k,
                    "window": window,
                    "mass": mass,
                    "window_xle": collapse_xle(window),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("mass", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def binary_search_mass(df: pd.DataFrame, mmin: float, mmax: float) -> Tuple[int,int]:
    import bisect
    masses = df["mass"].values
    left = bisect.bisect_left(masses, mmin)
    right = bisect.bisect_right(masses, mmax)
    return left, right

def analyze_hits(input_csv: str, outdir: str, kmin: int, kmax: int, charges: List[int], tol: float, unit: str,
                 prot_fasta: Optional[str], download_common: bool, accessions_path: Optional[str], only_hits: bool):
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("[mzscreen] reading input CSV...")
    df = pd.read_csv(input_csv)
    if "peptide" not in df.columns:
        raise ValueError("Input CSV must include at least 'peptide' column.")
    if only_hits:
        if "is_hit" not in df.columns:
            raise ValueError("--only_hits specified but 'is_hit' column is missing.")
        df = df[df["is_hit"].astype(int)==1].copy()

    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df = df[df["peptide"].str.len()>0].copy()
    df["mass"] = df["peptide"].apply(calc_mass)

    seqs = load_common_proteins(prot_fasta, download_common, accessions_path)
    win_df = build_window_index(seqs, kmin, kmax)
    if win_df.empty:
        raise RuntimeError("No windows were indexed from the provided/common proteins. Check FASTA or download options.")

    def short_name(hdr: str) -> str:
        m = re.match(r"^\w+\|([^|]+)\|([^ ]+)", hdr)
        if m:
            acc, name = m.groups()
            return name
        return hdr.split()[0]

    win_df["protein_short"] = win_df["protein"].apply(short_name)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="[mzscreen] screening peptides"):
        pep = r["peptide"]
        q_mass = r["mass"]
        for z in charges:
            q_mz = mz_from_mass(q_mass, z)
            mmin, mmax = mz_window_to_mass_range(q_mz, z, tol, unit)
            left, right = binary_search_mass(win_df, mmin, mmax)
            if right <= left: 
                continue
            sub = win_df.iloc[left:right]
            pep_xle = collapse_xle(pep)
            for _, wr in sub.iterrows():
                hit_mz = mz_from_mass(wr["mass"], z)
                da_err = hit_mz - q_mz
                ppm_err = (da_err / q_mz) * 1e6
                rows.append({
                    "query_peptide": pep,
                    "charge": z,
                    "query_mz": q_mz,
                    "window_peptide": wr["window"],
                    "window_len": wr["length"],
                    "window_start": int(wr["start"]),
                    "protein": wr["protein"],
                    "protein_short": wr["protein_short"],
                    "window_mz": hit_mz,
                    "ppm_error": ppm_err,
                    "da_error": da_err,
                    "identical_seq": int(pep == wr["window"]),
                    "xle_equal": int(pep_xle == wr["window_xle"]),
                })

    match_df = pd.DataFrame(rows).sort_values(["query_peptide","charge","ppm_error","window_len"])
    match_path = os.path.join(outdir, "mzscreen_matches.csv")
    match_df.to_csv(match_path, index=False)

    sum_query = (match_df.groupby(["query_peptide"])
                 .agg(n_matches=("window_peptide","size"),
                      n_xle_equal=("xle_equal","sum"),
                      n_identical=("identical_seq","sum"))
                 .reset_index())
    sum_query_path = os.path.join(outdir, "mzscreen_summary_per_query.csv")
    sum_query.to_csv(sum_query_path, index=False)

    sum_protein = (match_df.groupby(["protein_short"])
                   .agg(n_matches=("window_peptide","size"),
                        mean_abs_ppm=("ppm_error", lambda x: float(pd.Series(x).abs().mean())))
                   .reset_index()
                   .sort_values("n_matches", ascending=False))
    sum_protein_path = os.path.join(outdir, "mzscreen_summary_per_protein.csv")
    sum_protein.to_csv(sum_protein_path, index=False)

    # ---- Plots ----
    def add_grid(ax):
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig, ax = plt.subplots()
    ax.hist(sum_query["n_matches"].values, bins=20)
    ax.set_xlabel("m/z-sharing windows in common proteins (per query peptide)")
    ax.set_ylabel("count of query peptides")
    ax.set_title("Distribution of m/z-sharing subsequences (all queries)")
    add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "hist_matches_per_query.png"))
    plt.close(fig)

    topN = min(20, len(sum_protein))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(range(topN), sum_protein["n_matches"].values[:topN])
    ax.set_xticks(range(topN), sum_protein["protein_short"].values[:topN], rotation=90)
    ax.set_ylabel("# m/z-sharing windows")
    ax.set_title("Top proteins contributing m/z-sharing subsequences")
    add_grid(ax); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bar_top_proteins.png"))
    plt.close(fig)

    if "is_hit" in df.columns:
        sum_query = sum_query.merge(df[["peptide","is_hit"]].rename(columns={"peptide":"query_peptide"}),
                                    on="query_peptide", how="left")
        fig, ax = plt.subplots()
        non = sum_query[sum_query["is_hit"]==0]["n_matches"].values.tolist()
        hit = sum_query[sum_query["is_hit"]==1]["n_matches"].values.tolist()
        ax.boxplot([non, hit], labels=["non-hit","hit"], showmeans=True, meanline=True)
        ax.set_ylabel("# m/z-sharing windows in common proteins")
        ax.set_title("Subsequence m/z-sharing burden by group")
        add_grid(ax); fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "box_matches_by_group.png"))
        plt.close(fig)

    lines = []
    lines.append("# m/z-subsequence screen against common proteins\n")
    lines.append(f"- Input peptides: {len(df)}")
    lines.append(f"- Proteins considered: {len(seqs)}")
    lines.append(f"- Total windows indexed: {len(win_df)} (k={kmin}..{kmax})")
    lines.append(f"- Charge states: {','.join(map(str,charges))}; tolerance: {tol} {unit}")
    lines.append("\nOutputs:")
    lines.append("- mzscreen_matches.csv")
    lines.append("- mzscreen_summary_per_query.csv")
    lines.append("- mzscreen_summary_per_protein.csv")
    lines.append("- plots/*.png")
    with open(os.path.join(outdir, "README.txt"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"[mzscreen] Done. Wrote: {match_path} | {sum_query_path} | {sum_protein_path} | {os.path.join(plots_dir, '[pngs]')}")

def main():
    ap = argparse.ArgumentParser(description="Screen peptides for subsequence matches in common proteins that share precursor m/z.")
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with peptide[, is_hit]")
    ap.add_argument("--outdir", default="mzscreen_out", help="Output directory")
    ap.add_argument("--kmin", type=int, default=7, help="Minimum window k (default 7)")
    ap.add_argument("--kmax", type=int, default=12, help="Maximum window k (default 12)")
    ap.add_argument("--charges", type=str, default="2,3", help="Comma-separated charge states to test (default '2,3')")
    ap.add_argument("--tol", type=float, default=10.0, help="Tolerance value (default 10)")
    ap.add_argument("--unit", type=str, choices=["ppm","da"], default="ppm", help="Tolerance unit (ppm or da)")
    ap.add_argument("--prot_fasta", type=str, default=None, help="FASTA of common proteins (optional)")
    ap.add_argument("--download_common", action="store_true", help="Download a small panel of common proteins from UniProt (requires internet)")
    ap.add_argument("--accessions", type=str, default=None, help="Text file with UniProt accessions to download (optional)")
    ap.add_argument("--only_hits", action="store_true", help="Restrict analysis to rows with is_hit==1")
    args = ap.parse_args()
    banner("WIQD m/z screen", args)
    charges = [int(x) for x in args.charges.split(",") if x.strip()]
    analyze_hits(args.input_csv, args.outdir, args.kmin, args.kmax, charges, args.tol, args.unit,
                 args.prot_fasta, args.download_common, args.accessions, args.only_hits)

if __name__ == "__main__":
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()
