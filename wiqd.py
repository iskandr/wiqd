
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, argparse, datetime, re, random
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

# ---------- ASCII UI ----------
def banner(script_name: str, args: argparse.Namespace = None):
    title = "What Is Qing Doing?"
    bar = "+" + "="*66 + "+"
    lines = [bar, "|{:^66}|".format(title), "|{:^66}|".format(f"({script_name})"), bar]
    print("\n".join(lines))
    print(f"Start: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if args:
        keys = ["input_csv","outdir","k","collapse","alpha","housekeeping_fasta","download_housekeeping"]
        arg_summary = []
        for k in keys:
            if hasattr(args,k): arg_summary.append(f"{k}={getattr(args,k)}")
        print("Args: " + " | ".join(arg_summary))
    print()

# ---------- Matplotlib defaults ----------
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

# ---------- Chemistry constants ----------
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
hydrophobic_set = set("AVILMFWYC")

# Additional scales
EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 'Q': -0.85,
    'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50,
    'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18, 'T': -0.05, 'W': 0.81,
    'Y': 0.26, 'V': 1.08
}
HOPP_WOODS = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2,
    'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8, 'L': -1.8, 'K': 3.0,
    'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3, 'T': -0.4, 'W': -3.4,
    'Y': -2.3, 'V': -1.5
}

# Side-chain donors/acceptors (rough)
HBD_SC = set("KRHSTYNQWC")
HBA_SC = set("DENQHSTYWC")

# Elemental composition (residue = AA minus H2O; then add H2O once)
RES_ELEM = {
    'A': (3,5,1,1,0), 'R': (6,12,4,1,0), 'N': (4,6,2,2,0), 'D': (4,5,1,3,0),
    'C': (3,5,1,1,1), 'E': (5,7,1,3,0), 'Q': (5,8,2,2,0), 'G': (2,3,1,1,0),
    'H': (6,7,3,1,0), 'I': (6,11,1,1,0), 'L': (6,11,1,1,0), 'K': (6,12,2,1,0),
    'M': (5,9,1,1,1), 'F': (9,9,1,1,0), 'P': (5,7,1,1,0), 'S': (3,5,1,2,0),
    'T': (4,7,1,2,0), 'W': (11,10,2,1,0), 'Y': (9,9,1,2,0), 'V': (5,9,1,1,0)
}
H2O_ELEM = (0,2,0,1,0)

# ---------- Utility / feature functions ----------
def clean_peptide(p): return "".join(ch for ch in str(p).strip().upper() if ch in AA)
def kd_gravy(p): return sum(KD[a] for a in p)/len(p) if p else float("nan")
def mean_scale(p, scale): return sum(scale[a] for a in p)/len(p) if p else float("nan")
def kd_stdev(p):
    if not p: return float("nan")
    vals = [KD[a] for a in p]
    m = sum(vals)/len(vals)
    return (sum((x-m)**2 for x in vals)/len(vals))**0.5
def max_hydro_streak(p, hydro_set=hydrophobic_set):
    best=cur=0
    for a in p:
        if a in hydro_set: cur+=1; best=max(best,cur)
        else: cur=0
    return best
def mass_monoisotopic(p): return sum(MONO[a] for a in p) + H2O
def isoelectric_point(p):
    def net_charge(ph):
        q = 1/(1+10**(ph-PKA["N_term"])) - 1/(1+10**(PKA["C_term"]-ph))
        for a in p:
            if a=="K": q += 1/(1+10**(ph-PKA["K"]))
            elif a=="R": q += 1/(1+10**(ph-PKA["R"]))
            elif a=="H": q += 1/(1+10**(ph-PKA["H"]))
            elif a=="D": q -= 1/(1+10**(PKA["D"]-ph))
            elif a=="E": q -= 1/(1+10**(PKA["E"]-ph))
            elif a=="C": q -= 1/(1+10**(PKA["C"]-ph))
            elif a=="Y": q -= 1/(1+10**(PKA["Y"]-ph))
        return q
    lo,hi=0.0,14.0
    for _ in range(60):
        mid=(lo+hi)/2.0
        if net_charge(mid)>0: lo=mid
        else: hi=mid
    return (lo+hi)/2.0
def net_charge_at_pH(p, ph=7.4):
    def frac_pos(pka): return 1/(1+10**(ph-pka))
    def frac_neg(pka): return 1/(1+10**(pka-ph))
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
def count_n_to_proline_bonds(p): return sum(1 for i in range(len(p)-1) if p[i+1]=='P')
def count_c_to_acidic_bonds(p): return sum(1 for i in range(len(p)-1) if p[i] in ('D','E'))
def hbd_hba_counts(p): return (sum(1 for a in p if a in HBD_SC), sum(1 for a in p if a in HBA_SC))
def elemental_counts(p):
    C=H=N=O=S=0
    for a in p:
        if a in RES_ELEM:
            c,h,n,o,s = RES_ELEM[a]; C+=c; H+=h; N+=n; O+=o; S+=s
    C+=H2O_ELEM[0]; H+=H2O_ELEM[1]; O+=H2O_ELEM[3]
    return C,H,N,O,S
def approx_M1_rel_abundance(C,H,N,O,S): return 0.0107*C + 0.000155*H + 0.00364*N + 0.00038*O + 0.0075*S
def count_basic(p): return sum(1 for a in p if a in "KRH")
def count_acidic(p): return sum(1 for a in p if a in "DE")
def basicity_proxy(p): return (1.0*p.count('R') + 0.8*p.count('K') + 0.3*p.count('H') + 0.2)
PROTON_MASS = 1.007276466812
def mz_from_mass(neutral_mass: float, z: int) -> float: return (neutral_mass + z*PROTON_MASS)/z
def mobile_proton(z:int, p:str) -> int:
    sites = p.count('R') + p.count('K') + p.count('H') + 1  # +1 N-term
    return 1 if z > sites else 0

# ---------- Housekeeping collapsed k-mer homology ----------
HOUSEKEEPING_UNIPROT = ["P60709","P04406","P07437","P11142","P08238","P68104"]
def collapse_seq(seq: str, mode: str = "xle") -> str:
    m=[]
    for ch in seq:
        if mode.startswith("xle") and ch in ("L","I"): m.append("J")
        elif mode.endswith("+deqn") and ch in ("D","E"): m.append("B")
        elif mode.endswith("+deqn") and ch in ("Q","N"): m.append("Z")
        else: m.append(ch)
    return "".join(m)
def iter_kmers(seq: str, k: int):
    for i in range(0, len(seq)-k+1): yield seq[i:i+k]
def parse_fasta(text: str):
    seqs={}; hdr=None; buf=[]
    for line in text.splitlines():
        if not line: continue
        if line[0]==">":
            if hdr: seqs[hdr] = "".join(buf).replace(" ","").upper()
            hdr=line[1:].split()[0]; buf=[]
        else: buf.append(line.strip())
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

# ---------- Label helpers ----------
def pretty_feature_name(feat: str) -> str:
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
        "elem_C": "C atoms", "elem_H": "H atoms", "elem_N": "N atoms", "elem_O": "O atoms", "elem_S": "S atoms",
        "approx_Mplus1_rel": "Approx. M+1 relative abundance",
        "mz_z1": "m/z (z=1)", "mz_z2": "m/z (z=2)", "mz_z3": "m/z (z=3)",
        "mass_defect": "Mass defect",
        "hk_kmer_hits": "Housekeeping k-mer hits", "hk_kmer_frac": "Housekeeping k-mer fraction",
        "mass_mono": "Monoisotopic mass (Da)",
        "length": "Peptide length",
    }
    if feat in special: return special[feat]
    if feat.startswith("frac_"): return f"Fraction of {feat.split('_',1)[1]}"
    return feat.replace("_"," ")
def safe_name(s: str) -> str: return re.sub(r'[^A-Za-z0-9._+-]+', '_', s)

# ---------- Plotting helpers ----------
def _apply_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

def save_boxplot(x, y, feature, test_text, outpath):
    fig, ax = plt.subplots()
    ax.boxplot([x,y], labels=["non-hit","hit"], showmeans=True, meanline=True)
    jitter = 0.08
    xs0 = [1 + (random.random()-0.5)*2*jitter for _ in x]
    xs1 = [2 + (random.random()-0.5)*2*jitter for _ in y]
    ax.scatter(xs0, x, alpha=0.5, s=12)
    ax.scatter(xs1, y, alpha=0.5, s=12)
    ax.set_ylabel(pretty_feature_name(feature))
    n0, n1 = len(x), len(y)
    ax.set_title(f"{pretty_feature_name(feature)} — {test_text} | n(non-hit)={n0}, n(hit)={n1}")
    _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def save_barplot_props(feature, a, b, c, d, test_text, outpath):
    n_hit = a + b; n_nonhit = c + d
    p_hit = a/max(1,(n_hit)); p_nonhit = c/max(1,(n_nonhit))
    fig, ax = plt.subplots()
    xs=[0,1]; heights=[p_nonhit, p_hit]
    ax.bar(xs, heights)
    jitter = 0.08
    pts_non = [1]*c + [0]*d
    pts_hit = [1]*a + [0]*b
    xs_non = [0 + (random.random()-0.5)*2*jitter for _ in pts_non]
    xs_hit = [1 + (random.random()-0.5)*2*jitter for _ in pts_hit]
    ax.scatter(xs_non, pts_non, alpha=0.5, s=12)
    ax.scatter(xs_hit, pts_hit, alpha=0.5, s=12)
    ax.set_xticks(xs, ["non-hit","hit"])
    ax.set_ylabel(f"Proportion with {pretty_feature_name(feature)}")
    ax.set_title(f"{pretty_feature_name(feature)} — {test_text} | n(non-hit)={n_nonhit}, n(hit)={n_hit}")
    ax.set_ylim(0, 1.05)
    ax.text(0, heights[0]+0.02, f"{c}/{n_nonhit}", ha="center", va="bottom")
    ax.text(1, heights[1]+0.02, f"{a}/{n_hit}", ha="center", va="bottom")
    _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def save_grouped_bar(labels, vals_nonhit, vals_hit, title, ylabel, outpath, sig_mask=None):
    fig, ax = plt.subplots(figsize=(max(7, len(labels)*0.45), 4.4))
    idx=list(range(len(labels))); width=0.4
    ax.bar([i - width/2 for i in idx], vals_nonhit, width=width, label="non-hit")
    ax.bar([i + width/2 for i in idx], vals_hit,  width=width, label="hit")
    ax.set_xticks(idx, labels, rotation=0)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    if sig_mask:
        for i, sig in enumerate(sig_mask):
            if sig:
                y = max(vals_nonhit[i] if not math.isnan(vals_nonhit[i]) else 0.0,
                        vals_hit[i] if not math.isnan(vals_hit[i]) else 0.0)
                ax.text(i, y*1.02 + 1e-6, "*", ha="center", va="bottom")
    _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

def save_summary_plot(stats_df: pd.DataFrame, alpha: float, outpath: str, n_nonhit: int = None, n_hit: int = None):
    def neglog10(x):
        try:
            x = float(x)
            if 0.0 < x <= 1.0:
                return -math.log10(x)
        except Exception:
            pass
        return 0.0
    df = stats_df.copy()
    df["neglog10_q"] = df["q_value"].apply(neglog10)
    df["neglog10_p"] = df["p_value"].apply(neglog10)
    df["score"] = df.apply(lambda r: r["neglog10_q"] if r["neglog10_q"] > 0 else r["neglog10_p"], axis=1)
    df = df[df["score"] > 0].copy()
    def effect_dir(row):
        if row["type"] == "numeric":
            d = row.get("effect_cliffs_delta", float("nan"))
            try:
                d = float(d)
                if d > 0:  return "hit↑"
                if d < 0:  return "hit↓"
            except Exception:
                return ""
        else:
            try:
                ph = float(row.get("prop_hit", float("nan"))); pn = float(row.get("prop_nonhit", float("nan")))
                if not (math.isnan(ph) or math.isnan(pn)):
                    return "hit↑" if (ph - pn) > 0 else ("hit↓" if (ph - pn) < 0 else "")
            except Exception:
                return ""
        return ""
    df["dir"] = df.apply(effect_dir, axis=1)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.set_title("Feature summary — no non-zero signal scores")
        ax.set_ylabel("Signal score"); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "No detectable differences\n(all q≥α and p≥0.05 or insufficient data)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig); return
    top = df.sort_values("score", ascending=False).head(60)
    fig, ax = plt.subplots(figsize=(max(8, len(top)*0.32), 5))
    ax.bar(range(len(top)), top["score"].tolist())
    ax.set_xticks(range(len(top)), top["feature"].tolist(), rotation=90)
    ax.set_ylabel("Signal score")
    title = "Feature significance summary"
    if n_nonhit is not None and n_hit is not None:
        title += f" | n(non-hit)={n_nonhit}, n(hit)={n_hit}"
    ax.set_title(title)
    if alpha and alpha > 0:
        cutoff = -math.log10(alpha); ax.axhline(cutoff, linestyle="--", linewidth=1.0); ax.text(0.0, cutoff*1.02, f"q={alpha:g}", va="bottom")
    for i, (h, d, q) in enumerate(zip(top["score"], top["dir"], top["q_value"])):
        try: qf = float(q)
        except Exception: qf = float("nan")
        if (isinstance(qf, float) and not math.isnan(qf) and qf < alpha and h > 0):
            ax.text(i, h*1.02, d, ha="center", va="bottom", rotation=90)
    _apply_grid(ax); plt.tight_layout(); fig.savefig(outpath); plt.close(fig)

# ---------- Exemplars ----------
def summarize_top_feature_exemplars(feat_df: pd.DataFrame, stats_df: pd.DataFrame, outdir: str, top_n: int = 10) -> str:
    def neglog10(x):
        try:
            x = float(x)
            if 0.0 < x <= 1.0: return -math.log10(x)
        except Exception:
            pass
        return 0.0
    df = stats_df.copy()
    df["neglog10_q"] = df["q_value"].apply(neglog10)
    df["neglog10_p"] = df["p_value"].apply(neglog10)
    df["score"] = df.apply(lambda r: r["neglog10_q"] if r["neglog10_q"] > 0 else r["neglog10_p"], axis=1)
    df = df[df["score"] > 0].copy().sort_values("score", ascending=False)
    out_path = os.path.join(outdir, "top_features_exemplars.csv")
    if df.empty:
        pd.DataFrame(columns=["feature","pretty_name","type","score","p_value","q_value","direction",
                              "median_nonhit","median_hit","prop_nonhit","prop_hit",
                              "exemplars_hit","exemplars_nonhit"]).to_csv(out_path, index=False)
        return "No features with non-zero signal score to summarize.\n"
    def direction(row):
        if row["type"] == "numeric":
            d = row.get("effect_cliffs_delta", float("nan"))
            try:
                d = float(d)
                if d > 0:  return "hit↑"
                if d < 0:  return "hit↓"
            except Exception:
                return ""
        else:
            ph = row.get("prop_hit", float("nan")); pn = row.get("prop_nonhit", float("nan"))
            try:
                ph = float(ph); pn = float(pn)
                if ph - pn > 0: return "hit↑"
                if ph - pn < 0: return "hit↓"
            except Exception:
                return ""
        return ""
    rows=[]
    top = df.head(top_n)
    for _, r in top.iterrows():
        feat = r["feature"]; pretty = pretty_feature_name(feat); dirn = direction(r); typ = r["type"]
        ex_hit=[]; ex_non=[]
        if typ == "numeric":
            sub = feat_df[["peptide","is_hit",feat]].dropna().copy()
            if dirn == "hit↑":
                ex_hit = sub[sub["is_hit"]==1].sort_values(feat, ascending=False)["peptide"].head(5).tolist()
                ex_non = sub[sub["is_hit"]==0].sort_values(feat, ascending=True)["peptide"].head(5).tolist()
            elif dirn == "hit↓":
                ex_hit = sub[sub["is_hit"]==1].sort_values(feat, ascending=True)["peptide"].head(5).tolist()
                ex_non = sub[sub["is_hit"]==0].sort_values(feat, ascending=False)["peptide"].head(5).tolist()
            else:
                ex_hit = sub.sort_values(feat, ascending=False)["peptide"].head(5).tolist()
                ex_non = sub.sort_values(feat, ascending=True)["peptide"].head(5).tolist()
        else:
            sub = feat_df[["peptide","is_hit",feat]].copy()
            ex_hit = sub[(sub["is_hit"]==1) & (sub[feat]==1)]["peptide"].head(5).tolist()
            ex_non = sub[(sub["is_hit"]==0) & (sub[feat]==1)]["peptide"].head(5).tolist()
        rows.append({
            "feature": feat, "pretty_name": pretty, "type": typ, "score": r["score"],
            "p_value": r.get("p_value", float("nan")), "q_value": r.get("q_value", float("nan")),
            "direction": dirn,
            "median_nonhit": r.get("median_nonhit", float("nan")) if "median_nonhit" in r else float("nan"),
            "median_hit": r.get("median_hit", float("nan")) if "median_hit" in r else float("nan"),
            "prop_nonhit": r.get("prop_nonhit", float("nan")) if "prop_nonhit" in r else float("nan"),
            "prop_hit": r.get("prop_hit", float("nan")) if "prop_hit" in r else float("nan"),
            "exemplars_hit": ";".join(ex_hit), "exemplars_nonhit": ";".join(ex_non),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    lines = ["Top features by Signal score:"]
    for i, r in enumerate(rows, start=1):
        lines.append(f"{i}. {r['pretty_name']} ({r['feature']}) — score={r['score']:.2f}, p={r['p_value']:.2g}, q={r['q_value']:.2g}, dir={r['direction']}")
    return "\n".join(lines) + "\n"

# ---------- Core feature computation ----------
def compute_features(peptides: List[str]) -> pd.DataFrame:
    rows=[]
    for pep in tqdm(peptides, desc="[wiqd] computing features"):
        if not pep: continue
        mass = mass_monoisotopic(pep)
        C,H,N,O,S = elemental_counts(pep); m1 = approx_M1_rel_abundance(C,H,N,O,S)
        hyd_frac = sum(1 for a in pep if a in hydrophobic_set)/len(pep) if pep else float("nan")
        kd_std = kd_stdev(pep); hydro_run = max_hydro_streak(pep)
        charge_pH2_0 = net_charge_at_pH(pep, 2.0); charge_pH2_7 = net_charge_at_pH(pep, 2.7); charge_pH10_0 = net_charge_at_pH(pep, 10.0)
        pro_count = pep.count('P'); n_to_pro = count_n_to_proline_bonds(pep); c_to_acid = count_c_to_acidic_bonds(pep)
        hbd_sc, hba_sc = hbd_hba_counts(pep)
        bcnt = count_basic(pep); acnt = count_acidic(pep); gb = basicity_proxy(pep)
        mz1 = mz_from_mass(mass, 1); mz2 = mz_from_mass(mass, 2); mz3 = mz_from_mass(mass, 3); mdef = mass - int(mass)
        row={"peptide":pep,"length":len(pep),"mass_mono":mass,"gravy":kd_gravy(pep),
             "eisenberg_hydro":mean_scale(pep,EISENBERG),"hopp_woods_hydrophilicity":mean_scale(pep,HOPP_WOODS),
             "kd_stdev":kd_std,"hydrophobic_fraction":hyd_frac,"max_hydrophobic_run":hydro_run,
             "pI":isoelectric_point(pep),"charge_pH2_0":charge_pH2_0,"charge_pH2_7":charge_pH2_7,"charge_pH7_4":net_charge_at_pH(pep,7.4),
             "charge_pH10_0":charge_pH10_0,"aliphatic_index":aliphatic_index(pep),"aromaticity":aromaticity(pep),
             "cterm_hydrophobic":cterm_hydrophobic(pep),"tryptic_end":tryptic_end(pep),"has_RP_or_KP":has_RP_or_KP(pep),
             "has_C":1 if "C" in pep else 0,"has_M":1 if "M" in pep else 0,"has_W":1 if "W" in pep else 0,"has_Y":1 if "Y" in pep else 0,
             "xle_fraction":xle_fraction(pep),"basic_count":bcnt,"acidic_count":acnt,"basicity_proxy":gb,
             "mobile_proton_z2":mobile_proton(2,pep),"mobile_proton_z3":mobile_proton(3,pep),
             "proline_count":pro_count,"proline_internal": 1 if (pro_count>0 and (('P' in pep[1:-1]) if len(pep)>2 else False)) else 0,
             "n_to_proline_bonds":n_to_pro,"c_to_acidic_bonds":c_to_acid,
             "HBD_sidechain":hbd_sc,"HBA_sidechain":hba_sc,
             "elem_C":C,"elem_H":H,"elem_N":N,"elem_O":O,"elem_S":S,
             "approx_Mplus1_rel":m1,"mz_z1":mz1,"mz_z2":mz2,"mz_z3":mz3,"mass_defect":mdef}
        for a in sorted(AA): row[f"frac_{a}"] = aa_fraction(pep,a)
        rows.append(row)
    return pd.DataFrame(rows)

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

    feat = compute_features(df["peptide"].tolist())

    seqs = load_housekeeping_sequences(housekeeping_fasta, download=download_housekeeping)
    bank = build_housekeeping_kmer_bank(seqs, k=k, collapse=collapse) if seqs else set()

    print("[wiqd] computing housekeeping homology...")
    hk_rows=[homology_features(p, bank, k=k, collapse=collapse) for p in tqdm(feat["peptide"], desc="[wiqd] homology per peptide")]
    hk_df=pd.DataFrame(hk_rows)
    feat=pd.concat([feat, hk_df], axis=1)

    feat=feat.merge(df[["peptide","is_hit"]], on="peptide", how="left")
    feat_path = os.path.join(outdir,"features.csv"); feat.to_csv(feat_path, index=False)

    g0=feat[feat["is_hit"]==0]; g1=feat[feat["is_hit"]==1]

    numeric_feats=["length","mass_mono","gravy","eisenberg_hydro","hopp_woods_hydrophilicity","kd_stdev",
                   "hydrophobic_fraction","max_hydrophobic_run","pI","charge_pH2_0","charge_pH2_7","charge_pH7_4","charge_pH10_0",
                   "aliphatic_index","aromaticity","xle_fraction","basic_count","acidic_count","basicity_proxy",
                   "n_to_proline_bonds","c_to_acidic_bonds","HBD_sidechain","HBA_sidechain",
                   "elem_C","elem_H","elem_N","elem_O","elem_S","approx_Mplus1_rel",
                   "mz_z1","mz_z2","mz_z3","mass_defect","hk_kmer_hits","hk_kmer_frac"] + [f"frac_{a}" for a in sorted(AA)]
    binary_feats=["cterm_hydrophobic","tryptic_end","has_RP_or_KP","has_C","has_M","has_W","has_Y","proline_internal","mobile_proton_z2","mobile_proton_z3"]

    pvals={}; stats_rows=[]

    for f in tqdm(numeric_feats, desc="[wiqd] testing numeric features"):
        x=g0[f].dropna().tolist(); y=g1[f].dropna().tolist()
        if len(x)==0 or len(y)==0: U=float("nan"); p=float("nan"); delta=float("nan")
        else: U,p=mannwhitney_u_p(x,y); delta=cliffs_delta(x,y)
        pvals[f]=p
        stats_rows.append({"feature":f,"type":"numeric","U":U,"p_value":p,"effect_cliffs_delta":delta,
                           "median_nonhit": float(pd.Series(x).median()) if x else float("nan"),
                           "median_hit": float(pd.Series(y).median()) if y else float("nan")})

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
    stats_path = os.path.join(outdir,"stats_summary.csv"); stats_df.to_csv(stats_path, index=False)

    for f in tqdm(numeric_feats, desc="[wiqd] plotting numeric features"):
        x=g0[f].dropna().tolist(); y=g1[f].dropna().tolist()
        test_text=f"U-test p={pvals[f]:.3g}, q={qvals[f]:.3g}"
        sig = (not math.isnan(qvals[f]) and qvals[f] < alpha)
        base = f"{safe_name(f)}__box" + (".SIG" if sig else "")
        fname = f"{base}.png"
        save_boxplot(x, y, f, test_text, os.path.join(plots_dir, fname))

    for f in tqdm(binary_feats, desc="[wiqd] plotting binary features"):
        a=int((g1[f]==1).sum()); b=int((g1[f]==0).sum()); c=int((g0[f]==1).sum()); d=int((g0[f]==0).sum())
        test_text=f"Fisher p={pvals[f]:.3g}, q={qvals[f]:.3g}"
        sig = (not math.isnan(qvals[f]) and qvals[f] < alpha)
        base = f"{safe_name(f)}__bar" + (".SIG" if sig else "")
        fname = f"{base}.png"
        save_barplot_props(f, a, b, c, d, test_text, os.path.join(plots_dir, fname))

    aa_labels=sorted(AA)
    mean_nonhit=[float(g0[f"frac_{a}"].mean()) if not g0.empty else float("nan") for a in aa_labels]
    mean_hit=[float(g1[f"frac_{a}"].mean()) if not g1.empty else float("nan") for a in aa_labels]
    sig_mask=[ (not math.isnan(qvals[f"frac_{a}"])) and qvals[f"frac_{a}"]<alpha for a in aa_labels ]
    fname = "aa_composition.SIG.png" if any(sig_mask) else "aa_composition.png"
    title = "AA composition (mean per peptide) | '*' marks q<%.3g" % alpha
    save_grouped_bar(aa_labels, mean_nonhit, mean_hit, title, "mean fraction", os.path.join(plots_dir,fname), sig_mask=sig_mask)

    summary_path = os.path.join(outdir, "summary_feature_significance.png")
    save_summary_plot(stats_df, alpha, summary_path, n_nonhit=len(g0), n_hit=len(g1))

    # Top-feature exemplars
    exemplars_text = summarize_top_feature_exemplars(feat, stats_df, outdir, top_n=10)

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
    lines.append("\nFiles:\n- features.csv\n- stats_summary.csv\n- plots/*.png (feature-first names, '.SIG' for FDR-significant)\n- summary_feature_significance.png")
    lines.append("\nHousekeeping protein sequences used: %d" % (len(seqs)))
    if len(seqs) == 0:
        lines.append("NOTE: No housekeeping sequences were available. Provide --housekeeping_fasta or --download_housekeeping to enable homology features.")
    lines.append("\n---\n# Feature explanations\n")
    lines.append("- HBD_sidechain: Count of **side-chain hydrogen-bond donors** (rough proxy; includes K, R, H, S, T, Y, N, Q, W, C).")
    lines.append("- basic_count: Count of **basic residues** (K, R, H). Higher values tend to increase positive charge at low pH and can affect precursor charge states/selection.")
    lines.append("- n_to_proline_bonds: Number of potential **N→Proline cleavage sites** (i.e., 'XP' motifs). Proline often promotes cleavage N-terminal to P in CID/HCD and can shape fragmentation patterns.")
    lines.append("\nA CSV with exemplar peptides for the top features is written to: top_features_exemplars.csv\n")
    lines.append(exemplars_text)

    with open(os.path.join(outdir, "README.txt"), "w") as fh: fh.write("\n".join(lines))
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
    try:
        from tqdm.auto import tqdm  # noqa: F401
    except Exception:
        pass
    main()
