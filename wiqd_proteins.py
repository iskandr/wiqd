import os
import re

import requests
import urllib.request


# ===== Reference sequences / accession sets =====
HOUSEKEEPING_ACCS = [
    "P60709",
    "P04406",
    "P68104",
    "P13639",
    "P06733",
    "P68371",
    "P07437",
    "P62937",
    "P06748",
    "P11021",
    "P27824",
    "P07237",
    "P02768",
    "P01857",
    "P01834",
]

# UniProt P02768 mature albumin (fallback; 585 aa)
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


def load_albumin_sequence(args) -> str:
    seq = ""
    source_used = None
    if args.albumin_source in ("file", "auto") and args.albumin_fasta:
        try:
            with open(args.albumin_fasta, "r", encoding="utf-8") as fh:
                seq = _parse_fasta_text_to_seq(fh.read())
                source_used = f"file:{args.albumin_fasta}"
        except Exception as e:
            print(f"[wiqd][WARN] albumin FASTA read failed: {e}")
    if not seq and args.albumin_source in ("fetch", "auto"):
        for acc in args.albumin_acc.split(","):
            acc = acc.strip()
            if not acc:
                continue
            seq = fetch_albumin_fasta_from_uniprot(acc)
            if seq:
                source_used = f"uniprot:{acc}"
                break
    if not seq:
        seq = ALBUMIN_P02768_MATURE
        source_used = source_used or "embedded:ALBUMIN_P02768_MATURE"
    L = len(seq)
    is_prepro = L == 609
    is_mature = L == 585
    ok_len = (
        (args.albumin_expected == "prepro" and is_prepro)
        or (args.albumin_expected == "mature" and is_mature)
        or (args.albumin_expected == "either" and (is_prepro or is_mature))
    )
    if not ok_len:
        print(
            f"[wiqd][WARN] albumin length {L} differs from expected (609 prepro or 585 mature). Proceeding."
        )
    if args.albumin_use == "mature" and L == 609:
        seq = seq[24:]
        print("[wiqd] Trimmed albumin to mature 585 aa.")
    elif (
        args.albumin_use == "auto"
        and args.albumin_expected in ("mature", "either")
        and L == 609
    ):
        seq = seq[24:]
        print("[wiqd] Auto-trimmed albumin to mature 585 aa.")
    print(f"[wiqd] Albumin source: {source_used}; final length={len(seq)}")
    return seq


# =================== Housekeeping & homology ===================
def parse_fasta(text: str):
    seqs = {}
    hdr = None
    buf = []
    for line in text.splitlines():
        if not line:
            continue
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


def load_housekeeping_sequences(housekeeping_fasta: str = None, download: bool = False):
    seqs = {}
    if housekeeping_fasta and os.path.isfile(housekeeping_fasta):
        with open(housekeeping_fasta, "r") as fh:
            seqs = parse_fasta(fh.read())
    elif download:
        try:

            url = "https://rest.uniprot.org/uniprotkb/stream"
            query = " OR ".join(f"accession:{acc}" for acc in HOUSEKEEPING_ACCS)
            params = {"query": query, "format": "fasta", "includeIsoform": "false"}
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            seqs = parse_fasta(r.text)
        except Exception as e:
            print(
                f"[wiqd] WARNING: download_housekeeping failed ({e}); using built-in fallback set."
            )
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


def _parse_fasta_text_to_seq(text: str) -> str:
    text = "".join([ln.strip() for ln in text.splitlines() if not ln.startswith(">")])
    seq = re.sub(r"[^A-Z]", "", text.upper())
    return seq


def fetch_albumin_fasta_from_uniprot(acc: str, timeout: float = 10.0) -> str:

    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        return _parse_fasta_text_to_seq(data)
    except Exception as e:
        print(f"[wiqd][WARN] UniProt fetch failed for {acc}: {e}")
        return ""
