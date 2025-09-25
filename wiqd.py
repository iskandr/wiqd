
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiqd.py — What Is Quantitatively Different
Visualize physicochemical properties & AA composition of candidate peptides, grouped by hit vs non-hit,
and compute approximate homology to housekeeping proteins using collapsed k-mers (L/I collapse by default).

Inputs: CSV with columns: peptide, is_hit (0/1).
Outputs: features.csv, stats_summary.csv, plots/*.png (significant plots prefixed with SIG_), README.txt, summary_feature_significance.png
"""
import os, sys, math, argparse, datetime
from typing import List, Dict
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
        # Summarize key args
        arg_summary = [
            f"input={getattr(args, 'input_csv', None)}",
            f"outdir={getattr(args, 'outdir', None)}",
            f"k={getattr(args, 'k', None)} collapse={getattr(args, 'collapse', None)} alpha={getattr(args, 'alpha', None)}",
            f"housekeeping_fasta={getattr(args, 'housekeeping_fasta', None)} download_housekeeping={getattr(args, 'download_housekeeping', None)}",
        ]
        print("Args: " + " | ".join(arg_summary))
    print()

# ---------- Matplotlib defaults for readability ----------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.figsize": (6.8, 4.4),
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---------- tqdm (optional) ----------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# ---------- Chemistry utilities ----------
AA = set("ACDEFGHIKLMNPQRSTVWY")
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}
MONO = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694, 'C': 103.00919,
    'E': 129.04259, 'Q': 128.05858, 'G': 57.02146, 'H': 137.05891, 'I': 113.08406,
    'L': 113.08406, 'K': 128.09496, 'M': 131.04049, 'F': 147.06841, 'P': 97.05276,
    'S': 87.03203, 'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841
}
H2O = 18.01056

PKA = {"N_term": 9.69, "C_term": 2.34, "K": 10.5, "R": 12.5, "H": 6.0, "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
hydrophobic_set = set("AVILMFWY")

HOUSEKEEPING_UNIPROT = ["P60709","P04406","P07437","P11142","P08238","P68104"]

def clean_peptide(p): return "".join(ch for ch in str(p).strip().upper() if ch in AA)
def kd_gravy(p): return sum(KD[a] for a in p)/len(p) if p else float("nan")
def mass_monoisotopic(p): return sum(MONO[a] for a in p) + H2O

def isoelectric_point(p):
    def net_charge(ph):
        q = 1.0/(1.0+10**(ph-PKA["N_term"])) - 1.0/(1.0+10**(PKA["C_term"]-ph))
        for a in p:
            if a=="K": q += 1.0/(1.0+10**(ph-PKA["K"]))
            elif a=="R": q += 1.0/(1.0+10**(ph-PKA["R"]))
            elif a=="H": q += 1.0/(1.0+10**(ph-PKA["H"]))
            elif a=="D": q -= 1.0/(1.0+10**(PKA["D"]-ph))
            elif a=="E": q -= 1.0/(1.0+10**(PKA["E"]-ph))
            elif a=="C": q -= 1.0/(1.0+10**(PKA["C"]-ph))
            elif a=="Y": q -= 1.0/(1.0+10**(PKA["Y"]-ph))
        return q
    lo,hi=0.0,14.0
    for _ in range(60):
        mid=(lo+hi)/2.0
        if net_charge(mid)>0: lo=mid
        else: hi=mid
    return (lo+hi)/2.0

def net_charge_at_pH(p, ph=7.4):
    def frac_pos(pka): return 1.0/(1.0+10**(ph-pka))
    def frac_neg(pka): return 1.0/(1.0+10**(pka-ph))
    q = frac_pos(PKA["N_term"]) - frac_neg(PKA["C_term"])
    for a in p:
        if a=="K": q += frac_pos(PKA["K"])
        elif a=="R": q += frac_pos(PKA["R"])
        elif a=="H": q += frac_pos(PKA["H"])
        elif a=="D": q -= frac_neg(PKA["D"])
        elif a=="E": q -= frac_neg(PKA["E"])
        elif a=="C": q -= frac_neg(PKA["C"])
        elif a=="Y": q -= frac_neg(PKA["Y"])
    return q

def aliphatic_index(p):
    if not p: return float("nan")
    L=len(p); A=p.count("A")/L; V=p.count("V")/L; I=p.count("I")/L; Lc=p.count("L")/L
    return 100.0*(A + 2.9*V + 3.9*(I+Lc))

def aromaticity(p): return (p.count("F")+p.count("Y")+p.count("W"))/len(p) if p else float("nan")
def cterm_hydrophobic(p): return 1 if p and p[-1] in hydrophobic_set else 0
def tryptic_end(p): return 1 if p and p[-1] in {"K","R"} else 0
def has_RP_or_KP(p): return 1 if ("RP" in p or "KP" in p) else 0
def xle_fraction(p): return (p.count("I")+p.count("L"))/len(p) if p else float("nan")
def aa_fraction(p,aa): return p.count(aa)/len(p) if p else float("nan")

def compute_features(peptides: List[str]) -> pd.DataFrame:
    rows=[]
    for pep in tqdm(peptides, desc="[wiqd] computing features"):
        if not pep: continue
        row={"peptide":pep,"length":len(pep),"mass_mono":mass_monoisotopic(pep),"gravy":kd_gravy(pep),
             "pI":isoelectric_point(pep),"charge_pH7_4":net_charge_at_pH(pep,7.4),"aliphatic_index":aliphatic_index(pep),
             "aromaticity":aromaticity(pep),"cterm_hydrophobic":cterm_hydrophobic(pep),"tryptic_end":tryptic_end(pep),
             "has_RP_or_KP":has_RP_or_KP(pep),"has_C":1 if "C" in pep else 0,"xle_fraction":xle_fraction(pep)}
        for a in sorted(AA):
            row[f"frac_{a}"] = aa_fraction(pep,a)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- Housekeeping k-mer homology ----------
def collapse_seq(seq: str, mode: str = "xle") -> str:
    m=[]
    for ch in seq:
        if mode.startswith("xle") and ch in ("L","I"): m.append("J")
        elif mode.endswith("+deqn") and ch in ("D","E"): m.append("B")
        elif mode.endswith("+deqn") and ch in ("Q","N"): m.append("Z")
        else: m.append(ch)
    return "".join(m)

def iter_kmers(seq: str, k: int):
    for i in range(0, len(seq)-k+1):
        yield seq[i:i+k]

def parse_fasta(text: str):
    seqs={}; hdr=None; buf=[]
    for line in text.splitlines():
        if not line: continue
        if line[0]==">":
            if hdr: seqs[hdr] = "".join(buf).replace(" ","").upper()
            hdr=line[1:].split()[0]; buf=[]
        else:
            buf.append(line.strip())
    if hdr: seqs[hdr] = "".join(buf).replace(" ","").upper()
    return seqs

def load_housekeeping_sequences(fasta_path: str = None, download: bool = False):
    seqs={}
    if fasta_path and os.path.isfile(fasta_path):
        with open(fasta_path,"r") as fh: seqs=parse_fasta(fh.read())
    elif download:
        try:
            import requests
            url="https://rest.uniprot.org/uniprotkb/stream"
            query = " OR ".join(f"accession:{acc}" for acc in HOUSEKEEPING_UNIPROT)
            params={"query":query,"format":"fasta","includeIsoform":"false"}
            r=requests.get(url, params=params, timeout=30); r.raise_for_status()
            seqs=parse_fasta(r.text)
        except Exception as e:
            print(f"[wiqd] WARNING: download failed ({e}); proceeding without housekeeping sequences.")
            seqs={}
    else:
        seqs={"sp|P11142|HSP7C_HUMAN":(
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

def build_housekeeping_kmer_bank(seqs: Dict[str,str], k: int = 6, collapse: str = "xle"):
    bank=set()
    for name, seq in tqdm(seqs.items(), desc="[wiqd] building housekeeping k-mer bank"):
        s=collapse_seq(seq, "xle" if collapse=="xle" else ("xle+deqn" if collapse=="xle_de_qn" else "none"))
        for km in iter_kmers(s, k): bank.add(km)
    return bank

def homology_features(peptide: str, kmer_bank: set, k: int, collapse: str = "xle"):
    if len(peptide)<k or not kmer_bank: return {"hk_kmer_hits":0,"hk_kmer_frac":0.0}
    collapsed = collapse_seq(peptide, "xle" if collapse=="xle" else ("xle+deqn" if collapse=="xle_de_qn" else "none"))
    kms = list(iter_kmers(collapsed, k))
    hits = sum(1 for km in kms if km in kmer_bank)
    return {"hk_kmer_hits":hits, "hk_kmer_frac": hits/max(1,len(kms))}

# ---------- Stats ----------
def mannwhitney_u_p(x, y):
    n1,n2=len(x),len(y)
    if n1==0 or n2==0: return float("nan"), float("nan")
    data=[(v,0) for v in x]+[(v,1) for v in y]; data.sort(key=lambda t:t[0])
    R=[0]*(n1+n2); i=0
    while i<len(data):
        j=i
        while j<len(data) and data[j][0]==data[i][0]: j+=1
        rank=(i+j+1)/2.0
        for k in range(i,j): R[k]=rank
        i=j
    R1=sum(R[:n1]); U1=R1 - n1*(n1+1)/2.0; U2=n1*n2-U1; U=min(U1,U2)
    tie_counts=[]; i=0
    while i<len(data):
        j=i
        while j<len(data) and data[j][0]==data[i][0]: j+=1
        t=j-i
        if t>1: tie_counts.append(t)
        i=j
    T=sum(t*(t*t-1) for t in tie_counts)
    mu=n1*n2/2.0
    sigma2 = n1*n2*(n1+n2+1)/12.0 - (n1*n2*T)/(12.0*(n1+n2)*(n1+n2-1)) if (n1+n2)>1 else 0.0
    sigma = (sigma2**0.5) if sigma2>0 else 1e-12
    z=(U - mu + 0.5)/sigma
    import math
    p=2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/(2.0**0.5))))
    return U, max(0.0, min(1.0, p))

def cliffs_delta(x,y):
    gt=0; lt=0
    for xi in x:
        for yj in y:
            if xi>yj: gt+=1
            elif xi<yj: lt+=1
    n=len(x)*len(y)
    return (gt-lt)/n if n else float("nan")

def fisher_exact(a,b,c,d):
    from math import comb
    n=a+b+c+d; row1=a+b; col1=a+c
    def pmf(x): return comb(col1,x)*comb(n-col1,row1-x)/comb(n,row1)
    obs=pmf(a); p=0.0
    lo=max(0, row1-(n-col1)); hi=min(row1, col1)
    for x in range(lo, hi+1):
        px=pmf(x)
        if px<=obs+1e-15: p+=px
    return max(0.0, min(1.0, p))

def bh_fdr(pvals: Dict[str,float]) -> Dict[str,float]:
    items=[(k,v) for k,v in pvals.items() if not (v is None or (isinstance(v,float) and math.isnan(v)))]
    items.sort(key=lambda kv: kv[1]); m=len(items); out={}; prev=1.0
    for i,(k,p) in enumerate(items, start=1):
        q=min(prev, p*m/i); out[k]=q; prev=q
    for k in pvals:
        if k not in out: out[k]=float("nan")
    return out

# ---------- Plotting helpers ----------
def _apply_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

def save_boxplot(x, y, title, ylabel, outpath):
    fig, ax = plt.subplots()
    ax.boxplot([x,y], labels=["non-hit","hit"], showmeans=True, meanline=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_barplot_props(p_nonhit, p_hit, title, ylabel, outpath):
    fig, ax = plt.subplots()
    xs=[0,1]; heights=[p_nonhit, p_hit]
    ax.bar(xs, heights)
    ax.set_xticks(xs, ["non-hit","hit"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    _apply_grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_grouped_bar(labels, vals_nonhit, vals_hit, title, ylabel, outpath, sig_mask=None):
    fig, ax = plt.subplots(figsize=(max(7, len(labels)*0.45), 4.4))
    idx=list(range(len(labels))); width=0.4
    ax.bar([i - width/2 for i in idx], vals_nonhit, width=width, label="non-hit")
    ax.bar([i + width/2 for i in idx], vals_hit,  width=width, label="hit")
    ax.set_xticks(idx, labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if sig_mask:
        for i, sig in enumerate(sig_mask):
            if sig:
                y = max(vals_nonhit[i] if not math.isnan(vals_nonhit[i]) else 0.0,
                        vals_hit[i] if not math.isnan(vals_hit[i]) else 0.0)
                ax.text(i, y*1.02 + 1e-6, "*", ha="center", va="bottom")
    _apply_grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_summary_plot(stats_df: pd.DataFrame, alpha: float, outpath: str):
    df = stats_df.copy()
    df["neglog10_q"] = df["q_value"].apply(lambda q: -math.log10(q) if isinstance(q,float) and q>0 else 0.0)
    def effect_dir(row):
        if row["type"] == "numeric":
            d = row.get("effect_cliffs_delta", float("nan"))
            return "hit↑" if isinstance(d,float) and d>0 else ("hit↓" if isinstance(d,float) and d<0 else "")
        else:
            try:
                ph = float(row.get("prop_hit", float("nan"))); pn = float(row.get("prop_nonhit", float("nan")))
                if not math.isnan(ph) and not math.isnan(pn):
                    return "hit↑" if (ph - pn) > 0 else ("hit↓" if (ph - pn) < 0 else "")
            except Exception:
                pass
            return ""
    df["dir"] = df.apply(effect_dir, axis=1)
    df = df.sort_values("neglog10_q", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(df)*0.32), 5))
    ax.bar(range(len(df)), df["neglog10_q"].tolist())
    ax.set_xticks(range(len(df)), df["feature"].tolist(), rotation=90)
    ax.set_ylabel("-log10(q)")
    ax.set_title("Feature significance summary (BH–FDR)")
    if alpha and alpha>0:
        cutoff = -math.log10(alpha)
        ax.axhline(cutoff, linestyle="--", linewidth=1.0)
        ax.text(0.0, cutoff*1.02, f"q={alpha:g}", va="bottom")
    for i, (h, d, q) in enumerate(zip(df["neglog10_q"], df["dir"], df["q_value"])):
        if isinstance(q,float) and q < alpha and h>0:
            ax.text(i, h*1.02, d, ha="center", va="bottom", rotation=90)
    _apply_grid(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

# ---------- Main analysis ----------
def run_analysis(input_csv: str, outdir: str, k: int = 6, collapse: str = "xle", alpha: float = 0.05,
                 housekeeping_fasta: str = None, download_housekeeping: bool = False):
    os.makedirs(outdir, exist_ok=True); plots_dir=os.path.join(outdir,"plots"); os.makedirs(plots_dir, exist_ok=True)

    print("[wiqd] reading input CSV...")
    df=pd.read_csv(input_csv)
    if "peptide" not in df.columns or "is_hit" not in df.columns:
        raise ValueError("Input CSV must have columns: peptide, is_hit (0/1)")

    df["peptide"] = df["peptide"].map(clean_peptide)
    df = df[df["peptide"].str.len()>0].copy()
    df["is_hit"] = df["is_hit"].astype(int)

    # Compute features
    feat = compute_features(df["peptide"].tolist())

    # Housekeeping sequences / k-mer bank
    seqs = load_housekeeping_sequences(housekeeping_fasta, download=download_housekeeping)
    bank = build_housekeeping_kmer_bank(seqs, k=k, collapse=collapse) if seqs else set()

    # Homology features
    print("[wiqd] computing housekeeping homology...")
    hk_rows=[]
    for p in tqdm(feat["peptide"], desc="[wiqd] homology per peptide"):
        hk_rows.append(homology_features(p, bank, k=k, collapse=collapse))
    hk_df=pd.DataFrame(hk_rows)
    feat=pd.concat([feat, hk_df], axis=1)

    # Merge labels
    feat=feat.merge(df[["peptide","is_hit"]], on="peptide", how="left")

    # Save features
    feat_path = os.path.join(outdir,"features.csv")
    feat.to_csv(feat_path, index=False)

    # Split
    g0=feat[feat["is_hit"]==0]; g1=feat[feat["is_hit"]==1]

    numeric_feats=["length","mass_mono","gravy","pI","charge_pH7_4","aliphatic_index","aromaticity",
                   "xle_fraction","hk_kmer_hits","hk_kmer_frac"] + [f"frac_{a}" for a in sorted(AA)]
    binary_feats=["cterm_hydrophobic","tryptic_end","has_RP_or_KP","has_C"]

    pvals={}; stats_rows=[]

    # Numeric
    for f in tqdm(numeric_feats, desc="[wiqd] testing numeric features"):
        x=g0[f].dropna().tolist(); y=g1[f].dropna().tolist()
        if len(x)==0 or len(y)==0: U=float("nan"); p=float("nan"); delta=float("nan")
        else:
            U,p=mannwhitney_u_p(x,y); delta=cliffs_delta(x,y)
        pvals[f]=p
        stats_rows.append({"feature":f,"type":"numeric","U":U,"p_value":p,"effect_cliffs_delta":delta,
                           "median_nonhit": float(pd.Series(x).median()) if x else float("nan"),
                           "median_hit": float(pd.Series(y).median()) if y else float("nan")})

    # Binary
    for f in tqdm(binary_feats, desc="[wiqd] testing binary features"):
        a=int((g1[f]==1).sum()); b=int((g1[f]==0).sum()); c=int((g0[f]==1).sum()); d=int((g0[f]==0).sum())
        if (a+b)==0 or (c+d)==0: p=float("nan")
        else: p=fisher_exact(a,b,c,d)
        pvals[f]=p
        prop_hit=a/max(1,(a+b)); prop_nonhit=c/max(1,(c+d))
        stats_rows.append({"feature":f,"type":"binary","p_value":p,"prop_nonhit":prop_nonhit,"prop_hit":prop_hit,
                           "counts":f"hit: {a}/{a+b}; non-hit: {c}/{c+d}"})

    qvals = bh_fdr(pvals)
    for r in stats_rows: r["q_value"]=qvals[r["feature"]]

    stats_df = pd.DataFrame(stats_rows).sort_values(["type","q_value","p_value"])
    stats_path = os.path.join(outdir,"stats_summary.csv")
    stats_df.to_csv(stats_path, index=False)

    # ---------- PLOTS ----------
    for f in tqdm(numeric_feats, desc="[wiqd] plotting numeric features"):
        x=g0[f].dropna().tolist(); y=g1[f].dropna().tolist()
        title=f"{f} | U-test p={pvals[f]:.3g}, q={qvals[f]:.3g}"
        fname=f"SIG_box_{f}.png" if (not math.isnan(qvals[f]) and qvals[f]<alpha) else f"box_{f}.png"
        save_boxplot(x,y,title,f, os.path.join(plots_dir,fname))

    for f in tqdm(binary_feats, desc="[wiqd] plotting binary features"):
        a=int((g1[f]==1).sum()); b=int((g1[f]==0).sum()); c=int((g0[f]==1).sum()); d=int((g0[f]==0).sum())
        prop_hit=a/max(1,(a+b)); prop_nonhit=c/max(1,(c+d))
        title=f"{f} | Fisher p={pvals[f]:.3g}, q={qvals[f]:.3g}"
        fname=f"SIG_bar_{f}.png" if (not math.isnan(qvals[f]) and qvals[f]<alpha) else f"bar_{f}.png"
        save_barplot_props(prop_nonhit, prop_hit, title, "proportion with feature", os.path.join(plots_dir,fname))

    aa_labels=sorted(AA)
    mean_nonhit=[float(g0[f"frac_{a}"].mean()) if not g0.empty else float("nan") for a in aa_labels]
    mean_hit=[float(g1[f"frac_{a}"].mean()) if not g1.empty else float("nan") for a in aa_labels]
    sig_mask=[ (not math.isnan(qvals[f"frac_{a}"])) and qvals[f"frac_{a}"]<alpha for a in aa_labels ]
    fname = "SIG_aa_composition.png" if any(sig_mask) else "aa_composition.png"
    title = "AA composition (mean per peptide) | '*' marks q<%.3g" % alpha
    save_grouped_bar(aa_labels, mean_nonhit, mean_hit, title, "mean fraction", os.path.join(plots_dir,fname), sig_mask=sig_mask)

    summary_path = os.path.join(outdir, "summary_feature_significance.png")
    save_summary_plot(stats_df, alpha, summary_path)

    # README
    sigs = stats_df[(~stats_df["q_value"].isna()) & (stats_df["q_value"] < alpha)]
    lines = []
    lines.append("# Summary of significant differences (q < %.3g)\n" % alpha)
    if sigs.empty:
        lines.append("No features reached significance at FDR %.3g." % alpha)
    else:
        for _, r in sigs.iterrows():
            if r["type"] == "numeric":
                lines.append(f"- {r['feature']}: median non-hit={r['median_nonhit']:.3g}, "
                             f"median hit={r['median_hit']:.3g}, p={r['p_value']:.3g}, q={r['q_value']:.3g}, "
                             f"Cliff's δ={r['effect_cliffs_delta']:.3g}")
            else:
                lines.append(f"- {r['feature']}: proportion non-hit={r.get('prop_nonhit', float('nan')):.3g}, "
                             f"proportion hit={r.get('prop_hit', float('nan')):.3g}, p={r['p_value']:.3g}, q={r['q_value']:.3g} "
                             f"({r.get('counts','')})")
    lines.append("\nFiles:\n- features.csv\n- stats_summary.csv\n- plots/*.png (SIG_*)\n- summary_feature_significance.png")
    lines.append("\nHousekeeping protein sequences used: %d" % (len(seqs)))
    if len(seqs) == 0:
        lines.append("NOTE: No housekeeping sequences were available. Provide --housekeeping_fasta or --download_housekeeping to enable homology features.")
    with open(os.path.join(outdir, "README.txt"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"[wiqd] Done. Wrote: {feat_path} | {stats_path} | {summary_path} | {os.path.join(plots_dir, '[pngs]')}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="Input CSV with peptide,is_hit")
    ap.add_argument("--outdir", default="wiqd_out", help="Output directory")
    ap.add_argument("--k", type=int, default=6, help="k-mer length for housekeeping homology (default 6)")
    ap.add_argument("--collapse", choices=["none","xle","xle_de_qn"], default="xle",
                    help="Collapsed alphabet for homology: none | xle (L/I) | xle_de_qn (L/I + D/E and Q/N)")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold for significance (default 0.05)")
    ap.add_argument("--housekeeping_fasta", default=None, help="Path to FASTA of housekeeping proteins (optional)")
    ap.add_argument("--download_housekeeping", action="store_true", help="Download canonical sequences for a small housekeeping panel from UniProt (requires internet)")
    args=ap.parse_args()
    banner("WIQD", args)
    run_analysis(args.input_csv, args.outdir, k=args.k, collapse=args.collapse, alpha=args.alpha,
                 housekeeping_fasta=args.housekeeping_fasta, download_housekeeping=args.download_housekeeping)

if __name__ == "__main__":
    # tqdm import
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()
