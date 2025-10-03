#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mzscreen.py — Fast WIQD m/z screening CLI with grid mode (RT × Q1 × Q3)

Key improvements vs legacy:
- Build protein window index ONCE (reused across peptides)
- LRU cache for RT predictions (pairwise ΔRT)
- Optional grid mode for:
    RT ∈ {off, 3min, 1min} × Q1 ∈ {0.4Da, 100ppm} × Q3 ∈ {0.1Da, 30ppm, 10ppm}
- Optional prefilter for grid mode:
    1 pass with lenient config → keep top-K windows → rescore only those
- Adds Q3=10ppm preset (strict)

The scoring logic remains delegated to wiqd_peptide_similarity.score_ref_endog_reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- Helpers (no import guards, per request) ----
from wiqd_sequence_helpers import clean_pep
from wiqd_rt_proxy import predict_rt_min
from wiqd_peptide_similarity import (
    build_protein_index,
    score_ref_endog_reference,
    reference_defaults,
    plot_fragment_mz_distributions,
)
from wiqd_features import mass_neutral as pep_mass, mz_from_mass
from wiqd_proteins import parse_fasta, fetch_albumin_fasta_from_uniprot

# Progress
from tqdm.auto import tqdm

# Plots (non-interactive)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Preset mapping ----------------


def _resolve_q_fwhm(which: str, kind: str) -> Tuple[str, float]:
    """
    Returns (unit, fwhm) where unit ∈ {"da","ppm"}.
    """
    key = (kind.lower(), (which or "").lower())
    if key == ("q1", "0.4da"):
        return ("da", 0.4)
    if key == ("q1", "100ppm"):
        return ("ppm", 100.0)
    if key == ("q3", "30ppm"):
        return ("ppm", 30.0)
    if key == ("q3", "10ppm"):  # NEW: strict Q3
        return ("ppm", 10.0)
    if key == ("q3", "0.1da"):
        return ("da", 0.1)
    raise ValueError(f"Unknown {kind} preset '{which}'")


def _resolve_rt_inner(preset: str) -> Optional[float]:
    p = (preset or "off").lower()
    if p == "off":
        return None
    if p == "1min":
        return 1.0
    if p == "3min":
        return 3.0
    raise ValueError(f"Unknown RT preset '{preset}'")


# ---------------- Protein set loaders (JSON → {header: seq}) ----------------


def _fetch_uniprot_accessions(accs: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for acc in accs:
        acc = (acc or "").strip()
        if not acc:
            continue
        seq = fetch_albumin_fasta_from_uniprot(acc)  # generic fetcher
        if seq:
            hdr = f"sp|{acc}|{acc}"
            out[hdr] = seq
    return out


def _load_sets_from_json(
    json_path: str, names: Sequence[str], download: bool
) -> Dict[str, str]:
    """
    JSON schema supported:
      {
        "albumin": ["P02768", ...],
        "blood_serum": { "accessions": ["P02768", ...] },
        "custom": { "fasta_urls": ["https://.../x.fasta", ...] }
      }
    Returns merged {header: sequence}.
    """
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"--sets_accessions_json not found: {json_path}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    proteins: Dict[str, str] = {}
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name not in data:
            print(
                f"[mzscreen] WARNING: set '{name}' not found in {json_path} (skipping)."
            )
            continue
        spec = data[name]
        accs: List[str] = []
        fasta_urls: List[str] = []
        if isinstance(spec, list):
            accs = [str(x) for x in spec]
        elif isinstance(spec, dict):
            accs = [str(x) for x in spec.get("accessions", [])]
            fasta_urls = [str(u) for u in spec.get("fasta_urls", [])]
        else:
            print(
                f"[mzscreen] WARNING: set '{name}' has unsupported spec type {type(spec)} (skipping)."
            )
            continue

        # Accessions
        if accs and download:
            fetched = _fetch_uniprot_accessions(accs)
            proteins.update(fetched)

        # FASTA URLs
        for url in fasta_urls:
            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=45) as resp:
                    fasta_text = resp.read().decode("utf-8", errors="ignore")
                proteins.update(parse_fasta(fasta_text))
            except Exception as ex:
                print(f"[mzscreen] WARNING: FASTA fetch failed for {url}: {ex}")

    if not proteins:
        print(
            "[mzscreen] WARNING: No sequences loaded from sets; result will be empty unless you also provide --fasta."
        )
    return proteins


# ---------------- Caches ----------------


@lru_cache(maxsize=200_000)
def _rt_cached(seq: str, gradient_min: float) -> float:
    return predict_rt_min(seq, gradient_min=gradient_min)


@lru_cache(maxsize=200_000)
def _mass_cached(seq: str) -> float:
    return pep_mass(seq)


# ---------------- RT helper (pairwise ΔRT) ----------------


def _pair_rt_delta_min(ref_seq: str, endog_seq: str, gradient_min: float) -> float:
    a = _rt_cached(ref_seq, gradient_min)
    b = _rt_cached(endog_seq, gradient_min)
    if not (a == a) or not (b == b):  # NaN guard
        return float("inf")
    return abs(a - b)


# ---------------- CSV I/O ----------------

_PEPTIDE_COL_CANDIDATES = (
    "peptide",
    "pept",
    "sequence",
    "seq",
    "ref",
    "ref_peptide",
    "pep",
    "pepseq",
)


def _read_peptides_from_csv(path: str) -> List[str]:
    peptides: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        sniffer = csv.Sniffer()
        sample = fh.read(2048)
        fh.seek(0)
        try:
            has_header = sniffer.has_header(sample)
        except Exception:
            has_header = True
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        return peptides

    if has_header:
        header = [h.strip().lower() for h in rows[0]]
        idx = None
        for cname in _PEPTIDE_COL_CANDIDATES:
            if cname in header:
                idx = header.index(cname)
                break
        if idx is None:
            idx = 0
        for r in rows[1:]:
            if not r or idx >= len(r):
                continue
            seq = clean_pep(r[idx])
            if seq:
                peptides.append(seq)
    else:
        for r in rows:
            if not r:
                continue
            seq = clean_pep(r[0])
            if seq:
                peptides.append(seq)

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for s in peptides:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _write_hits_csv(path: str, rows: List[Dict[str, object]]) -> None:
    cols = [
        "ref_peptide",
        "protein",
        "start",
        "end",
        "subseq",
        "score",
        "rt_delta_min",
        "q1_z",
        "q1_delta_da",
        "q1_ppm_approx",
        "q1_weight",
        "q3_mean_weight",
        "matched_frags",
        "n_transitions",
        "summary",
        "precursor_ppm_good",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in cols])


# ---------------- Plotting (per peptide) ----------------


def _plot_score_hist(rows: List[Dict[str, object]], out_png: Path, title: str):
    if not rows:
        return
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    if not scores:
        return
    fig = plt.figure()
    plt.hist(scores, bins=30, alpha=0.8)
    plt.xlabel("Composite score (RT×Q1×Q3)")
    plt.ylabel("Count")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_scatter_ppm_vs_q3(rows: List[Dict[str, object]], out_png: Path, title: str):
    if not rows:
        return
    xs, ys = [], []
    for r in rows:
        ppm = r.get("q1_ppm_approx")
        wq3 = r.get("q3_mean_weight")
        if ppm is None or wq3 is None or ppm == "":
            continue
        try:
            xs.append(abs(float(ppm)))
            ys.append(float(wq3))
        except Exception:
            continue
    if not xs:
        return
    fig = plt.figure()
    plt.scatter(xs, ys, s=12, alpha=0.6)
    plt.xlabel("|Δppm| (approx from unmod ref @ best z)")
    plt.ylabel("mean Q3 weight")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_top_hit_sticks(ref_seq: str, hit_seq: str, out_png: Path):
    try:
        plot_fragment_mz_distributions(
            ref_seq,
            hit_seq,
            series=("y", "b"),
            frag_charges=(1,),
            include_y1=False,
            include_y2=True,
            include_b1=False,
            include_b2=True,
            min_frag_mz=200.0,
            max_frag_mz=1500.0,
            save_path=str(out_png),
        )
    except Exception as ex:
        print(f"[plots] WARNING: stick plot failed: {ex}")


# ---------------- Screening core ----------------


def _make_base_knobs(
    *,
    q1_fwhm: Tuple[str, float],
    q3_fwhm: Tuple[str, float],
    charges: Tuple[int, ...],
    frag_charges: Tuple[int, ...],
    include_y1: bool,
    include_y2: bool,
    include_b1: bool,
    include_b2: bool,
    min_frag_mz: float,
    max_frag_mz: float,
    frag_model: str,
    transitions_k: int,
    prefer_series_order: Tuple[str, ...],
    prefer_high_mz: bool,
    rt_inner_min: Optional[float],
    rt_fwhm_min: float,
    allow_heavy_first_residue: bool,
    allow_mtraq: bool,
) -> Dict[str, object]:
    base_knobs = reference_defaults()
    base_knobs.update(
        dict(
            q1_fwhm=q1_fwhm,
            q3_fwhm=q3_fwhm,
            q1_tol=None,
            q3_tol=None,
            charges=charges,
            max_isotope_shift_q1=0,
            series=("y", "b"),
            frag_charges=frag_charges,
            include_y1=include_y1,
            include_y2=include_y2,
            include_b1=include_b1,
            include_b2=include_b2,
            min_frag_mz=min_frag_mz,
            max_frag_mz=max_frag_mz,
            frag_model=frag_model,
            transitions_k=transitions_k,
            prefer_series_order=prefer_series_order,
            prefer_high_mz=prefer_high_mz,
            rt_inner_min=(rt_inner_min if rt_inner_min is not None else 0.0),
            rt_fwhm_min=rt_fwhm_min,
            allow_heavy_first_residue=allow_heavy_first_residue,
            allow_mtraq=allow_mtraq,
            return_fragment_maps=True,
        )
    )
    # Remove any stray rt_delta_min; pair-specific assignment happens per window
    if "rt_delta_min" in base_knobs:
        del base_knobs["rt_delta_min"]
    return base_knobs


def _score_entries_for_ref(
    ref_seq: str,
    entries: List[Dict[str, object]],
    *,
    base_knobs: Dict[str, object],
    rt_inner_min: Optional[float],
    gradient_min: float,
    min_matched_frags: int,
    show_progress: bool,
    precursor_ppm_good_thresh: Optional[float],
) -> List[Dict[str, object]]:
    """Scores the provided entry list against `ref_seq` and returns ALL passing rows (no Top-N cut)."""
    rows_out: List[Dict[str, object]] = []
    pbar = tqdm(
        entries, disable=not show_progress, desc=f"Scoring windows for {ref_seq}"
    )
    ref_mass = _mass_cached(ref_seq)
    for e in pbar:
        sub = e["subseq"]
        # Pair-wise ΔRT (minutes) if RT is enabled
        delta_rt = (
            _pair_rt_delta_min(ref_seq, sub, gradient_min)
            if rt_inner_min is not None
            else None
        )

        # Per-pair knobs (assign rt_delta_min ONCE here)
        knobs = dict(base_knobs)
        if delta_rt is not None:
            knobs["rt_delta_min"] = float(delta_rt)

        # Score with helper
        res = score_ref_endog_reference(ref_seq, sub, **knobs)

        matched = int(res["fragments"]["matched"])
        if matched < int(min_matched_frags):
            continue

        # Q1 ppm approx (unmodified ref, best z)
        zbest = int(res["precursor"]["best_z"])
        if zbest > 0:
            mz_ref_approx = mz_from_mass(ref_mass, zbest)
            q1_da = float(res["precursor"]["delta_da"])
            q1_ppm_approx = (
                (q1_da / mz_ref_approx) * 1e6 if mz_ref_approx > 0 else float("nan")
            )
        else:
            q1_ppm_approx = float("nan")

        row = {
            "ref_peptide": ref_seq,
            "protein": e["protein"],
            "start": int(e["start"]),
            "end": int(e["end"]),
            "subseq": sub,
            "score": float(res["score"]),
            "rt_delta_min": (float(delta_rt) if delta_rt is not None else ""),
            "q1_z": zbest,
            "q1_delta_da": float(res["precursor"]["delta_da"]),
            "q1_ppm_approx": q1_ppm_approx,
            "q1_weight": float(res["precursor"]["weight"]),
            "q3_mean_weight": float(res["fragments"]["mean_weight"]),
            "matched_frags": matched,
            "n_transitions": int(res["fragments"]["n_selected"]),
            "summary": res["summary"],
            "precursor_ppm_good": (
                (
                    1
                    if (
                        isinstance(precursor_ppm_good_thresh, (int, float))
                        and q1_ppm_approx == q1_ppm_approx
                        and abs(q1_ppm_approx) <= float(precursor_ppm_good_thresh)
                    )
                    else 0
                )
                if precursor_ppm_good_thresh is not None
                else ""
            ),
        }
        rows_out.append(row)
    return rows_out


def _top_n(rows: List[Dict[str, object]], n: int) -> List[Dict[str, object]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: -float(r["score"]))
    return rows_sorted[: max(0, int(n))]


# ---------------- CLI ----------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WIQD m/z screen over protein sets (RT×Q1×Q3), optimized with grid mode."
    )
    # Legacy / existing
    p.add_argument(
        "--in",
        dest="infile",
        required=False,
        help="Input CSV with peptides (legacy flag).",
    )
    p.add_argument("--outdir", required=False, help="Output directory (legacy flag).")
    p.add_argument(
        "--sets_accessions_json", default=None, help="JSON describing protein sets."
    )
    p.add_argument("--sets", default=None, help="Comma-separated set names to include.")
    p.add_argument(
        "--download_sets",
        action="store_true",
        help="Fetch sequences from UniProt for accessions.",
    )
    p.add_argument(
        "--ppm_tol", type=float, default=None, help="Fragment matching FWHM (ppm) → Q3."
    )
    p.add_argument(
        "--score_prec_ppm_good",
        type=float,
        default=None,
        help="Annotate 1 if |precursor Δppm| ≤ this; doesn’t change scoring.",
    )
    p.add_argument("--full_mz_len_min", type=int, default=7, help="Min window length.")
    p.add_argument("--full_mz_len_max", type=int, default=25, help="Max window length.")
    p.add_argument(
        "--fragment_kmin",
        type=int,
        default=1,
        help="Minimum number of matched fragments required (legacy gate).",
    )
    p.add_argument(
        "--fragment_kmax", type=int, default=4, help="Top-K transitions to select."
    )
    p.add_argument(
        "--score_frag_types_req",
        type=int,
        default=None,
        help="(legacy) required series count; ignored.",
    )
    p.add_argument(
        "--cys_mod", default=None, help="(legacy) fixed Cys; ignored for mass filters."
    )
    p.add_argument(
        "--xle_collapse",
        action="store_true",
        help="(legacy) I/L collapse; ignored for mass filters.",
    )
    p.add_argument(
        "--require_precursor_match_for_fragment",
        action="store_true",
        help="Implicit in SCORE = Q1×Q3×RT; no extra gate needed.",
    )
    p.add_argument(
        "--min-matched-frags",
        type=int,
        default=None,
        help="Minimum matched fragments (overrides --fragment_kmin if provided).",
    )

    # New knobs (single-run)
    p.add_argument(
        "--rt",
        choices=["off", "1min", "3min"],
        default="off",
        help="RT preset (default: off).",
    )
    p.add_argument(
        "--q1",
        choices=["0.4Da", "100ppm"],
        default="0.4Da",
        help="Q1 FWHM (default: 0.4Da).",
    )
    p.add_argument(
        "--q3",
        choices=["30ppm", "10ppm", "0.1Da"],  # include 10ppm
        default=None,
        help="Q3 FWHM preset (ignored if --ppm_tol is set).",
    )
    p.add_argument(
        "--gradient-min",
        type=float,
        default=20.0,
        help="Gradient length (min) for RT proxy.",
    )
    p.add_argument(
        "--rt-fwhm-min",
        type=float,
        default=5.0,
        help="RT Gaussian FWHM beyond inner window.",
    )
    p.add_argument("--charges", default="1,2,3", help="Peptide charges, CSV.")
    p.add_argument("--frag-charges", default="1", help="Fragment charges, CSV.")
    p.add_argument("--include-y1", type=int, default=0)
    p.add_argument("--include-y2", type=int, default=1)
    p.add_argument("--include-b1", type=int, default=0)
    p.add_argument("--include-b2", type=int, default=1)
    p.add_argument("--min-frag-mz", type=float, default=200.0)
    p.add_argument("--max-frag-mz", type=float, default=1500.0)
    p.add_argument(
        "--frag-model",
        default="lit_weighted",
        choices=["lit_weighted", "weighted", "topk", "all"],
    )
    p.add_argument(
        "--prefer-series",
        default="y,b",
        help="Series order preference, CSV (default y,b).",
    )
    p.add_argument(
        "--top-n", type=int, default=25, help="Top results per peptide to keep."
    )
    p.add_argument("--allow-heavy-first-residue", type=int, default=0)
    p.add_argument("--allow-mtraq", type=int, default=0)
    p.add_argument(
        "--fasta", default=None, help="Optional FASTA to include into the protein set."
    )
    p.add_argument("--step", type=int, default=1, help="Window step.")
    p.add_argument(
        "--plots-top-k",
        type=int,
        default=1,
        help="Stick plots for top-K hits per peptide.",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--ref", default=None, help="Single reference peptide (overrides --in)."
    )

    # Grid mode
    p.add_argument(
        "--run-grid",
        action="store_true",
        help="Run all RT×Q1×Q3 combos: {off,3min,1min}×{0.4Da,100ppm}×{0.1Da,30ppm,10ppm}.",
    )
    p.add_argument(
        "--grid-prefilter-k",
        type=int,
        default=0,
        help="If >0, prefilter windows per peptide using a lenient pass and keep only Top-K for rescoring across the remaining combos.",
    )
    return p.parse_args()


def _csv_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(s).split(",") if x.strip())


def _csv_series(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def _combo_label(
    rt_inner: Optional[float], q1: Tuple[str, float], q3: Tuple[str, float]
) -> str:
    def _fmt_q(t):
        unit, val = t
        return f"{val:g}{unit}"

    rt_label = "off" if rt_inner is None else f"{int(rt_inner)}min"
    return f"RT_{rt_label}__Q1_{_fmt_q(q1)}__Q3_{_fmt_q(q3)}"


def main():
    args = _parse_args()
    print("\n=== WIQD m/z screen (optimized) ===")
    print(f"[mzscreen] Args: {args}")

    # Resolve presets for single-run
    q1_fwhm = _resolve_q_fwhm(args.q1, "q1")
    if args.ppm_tol is not None:
        q3_fwhm = ("ppm", float(args.ppm_tol))
    else:
        q3_fwhm = _resolve_q_fwhm(args.q3 or "30ppm", "q3")
    rt_inner = _resolve_rt_inner(args.rt)

    charges = _csv_ints(args.charges) or (1, 2, 3)
    frag_charges = _csv_ints(args.frag_charges) or (1,)
    prefer_series = _csv_series(args.prefer_series) or ("y", "b")

    # Protein set
    proteins: Dict[str, str] = {}
    if args.fasta:
        with open(args.fasta, "r", encoding="utf-8") as fh:
            proteins.update(parse_fasta(fh.read()))
    if args.sets_accessions_json and args.sets:
        print(f"[mzscreen] Loading protein sets {args.sets} …")
        names = [x.strip() for x in args.sets.split(",") if x.strip()]
        proteins.update(
            _load_sets_from_json(args.sets_accessions_json, names, args.download_sets)
        )
    if not proteins:
        print("[mzscreen] WARNING: protein set is empty; nothing to score.")

    # Peptides
    if args.ref:
        peptides = [clean_pep(args.ref)]
    elif args.infile:
        peptides = [clean_pep(p) for p in _read_peptides_from_csv(args.infile)]
    else:
        raise SystemExit("You must provide either --ref or --in (CSV of peptides).")

    outdir = Path(args.outdir or ".").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    min_matched_frags = (
        int(args.min_matched_frags)
        if args.min_matched_frags is not None
        else int(args.fragment_kmin)
    )

    print(f"Peptides: {len(peptides)} | Proteins: {len(proteins)}")
    print(
        f"Single-run presets (if not --run-grid): RT={args.rt} (inner {rt_inner if rt_inner is not None else 'OFF'} min), "
        f"Q1={args.q1} → {q1_fwhm}, Q3={q3_fwhm}"
    )
    print(
        f"Windows: L={args.full_mz_len_min}..{args.full_mz_len_max} step={args.step} | "
        f"Fragments: Top-K={args.fragment_kmax}, min-matched={min_matched_frags}"
    )
    print(f"Output dir: {outdir}\n")

    # Build protein index ONCE (reused for all peptides)
    proteins_clean = {k: clean_pep(v) for k, v in proteins.items() if clean_pep(v)}
    print(
        f"[mzscreen] Building window index for {len(proteins_clean)} proteins "
        f"(L={args.full_mz_len_min}..{args.full_mz_len_max}, step={args.step}) …"
    )
    index = build_protein_index(
        proteins_clean,
        min_len=int(args.full_mz_len_min),
        max_len=int(args.full_mz_len_max),
        step=int(args.step),
        charges=charges,  # store precursor z per window
        series=("y", "b"),
        frag_charges=frag_charges,
        include_y1=bool(args.include_y1),
        include_y2=bool(args.include_y2),
        include_b1=bool(args.include_b1),
        include_b2=bool(args.include_b2),
        min_frag_mz=float(args.min_frag_mz),
        max_frag_mz=float(args.max_frag_mz),
        show_progress=(not args.no_progress),
    )
    all_entries: List[Dict[str, object]] = list(index.get("entries", []))

    # Convenience closures
    def make_base(
        rt_inner_min: Optional[float], q1: Tuple[str, float], q3: Tuple[str, float]
    ) -> Dict[str, object]:
        return _make_base_knobs(
            q1_fwhm=q1,
            q3_fwhm=q3,
            charges=charges,
            frag_charges=frag_charges,
            include_y1=bool(args.include_y1),
            include_y2=bool(args.include_y2),
            include_b1=bool(args.include_b1),
            include_b2=bool(args.include_b2),
            min_frag_mz=float(args.min_frag_mz),
            max_frag_mz=float(args.max_frag_mz),
            frag_model=args.frag_model,
            transitions_k=int(args.fragment_kmax),
            prefer_series_order=prefer_series,
            prefer_high_mz=True,
            rt_inner_min=rt_inner_min,
            rt_fwhm_min=float(args.rt_fwhm_min),
            allow_heavy_first_residue=bool(args.allow_heavy_first_residue),
            allow_mtraq=bool(args.allow_mtraq),
        )

    # Runtime options
    show_progress = not args.no_progress
    precursor_ppm_good_thresh = (
        float(args.score_prec_ppm_good)
        if args.score_prec_ppm_good is not None
        else None
    )

    # Runner: single config OR grid
    if not args.run_grid:
        # ----- Single-run mode -----
        base_knobs = make_base(rt_inner, q1_fwhm, q3_fwhm)
        all_rows: List[Dict[str, object]] = []
        pep_bar = tqdm(peptides, disable=not show_progress, desc="Peptides")
        for ref in pep_bar:
            pep_bar.set_postfix_str(ref)
            rows_all = _score_entries_for_ref(
                ref,
                all_entries,
                base_knobs=base_knobs,
                rt_inner_min=rt_inner,
                gradient_min=float(args.gradient_min),
                min_matched_frags=min_matched_frags,
                show_progress=show_progress,
                precursor_ppm_good_thresh=precursor_ppm_good_thresh,
            )
            rows = _top_n(rows_all, int(args.top_n))
            all_rows.extend(rows)

            # Outputs & plots
            pep_dir = outdir / "per_peptide" / ref
            pep_dir.mkdir(parents=True, exist_ok=True)
            _write_hits_csv(str(pep_dir / "hits.csv"), rows)
            if rows:
                _plot_score_hist(
                    rows,
                    pep_dir / "score_hist.png",
                    f"{ref}: score distribution (Top {len(rows)})",
                )
                _plot_scatter_ppm_vs_q3(
                    rows,
                    pep_dir / "scatter_absppm_vs_q3.png",
                    f"{ref}: |Δppm| vs mean Q3 weight",
                )
                k = max(0, min(int(args.plots_top_k), len(rows)))
                for i in range(k):
                    _plot_top_hit_sticks(
                        ref, rows[i]["subseq"], pep_dir / f"stick_top{i+1}.png"
                    )

        out_csv = outdir / "mzscreen_hits.csv"
        _write_hits_csv(str(out_csv), all_rows)
        print(f"\n[mzscreen] Wrote {len(all_rows)} rows → {out_csv}")
    else:
        # ----- Grid mode -----
        RT_PRESETS: Tuple[Optional[float], ...] = (None, 3.0, 1.0)  # off, 3min, 1min
        Q1_PRESETS: Tuple[Tuple[str, float], ...] = (("da", 0.4), ("ppm", 100.0))
        Q3_PRESETS: Tuple[Tuple[str, float], ...] = (
            ("da", 0.1),
            ("ppm", 30.0),
            ("ppm", 10.0),
        )

        # Prefilter config (lenient / inclusive): RT off, Q1 0.4Da, Q3 0.1Da
        prefilter_enabled = int(args.grid_prefilter_k) > 0
        prefilter_k = int(args.grid_prefilter_k)
        if prefilter_enabled:
            print(
                f"[grid] Prefilter enabled: lenient pass → keep Top-{prefilter_k} windows per peptide for rescoring."
            )

        # Storage of global rows per combo
        global_rows_per_combo: Dict[str, List[Dict[str, object]]] = {}

        pep_bar = tqdm(peptides, disable=not show_progress, desc="Peptides (grid)")
        for ref in pep_bar:
            pep_bar.set_postfix_str(ref)

            # Optional prefilter step per peptide
            candidate_entries = all_entries
            if prefilter_enabled:
                base_pref = make_base(rt_inner_min=None, q1=("da", 0.4), q3=("da", 0.1))
                pre_rows_all = _score_entries_for_ref(
                    ref,
                    all_entries,
                    base_knobs=base_pref,
                    rt_inner_min=None,
                    gradient_min=float(args.gradient_min),
                    min_matched_frags=min_matched_frags,
                    show_progress=False,  # inner loop; keep outer progress readable
                    precursor_ppm_good_thresh=precursor_ppm_good_thresh,
                )
                # Keep Top-K windows by score (more inclusive than using matched_frags only)
                pre_rows_sorted = sorted(pre_rows_all, key=lambda r: -float(r["score"]))
                # Build a set of (protein,start,end) tuples to select candidate entries quickly
                selected_keys = set()
                for r in pre_rows_sorted[:prefilter_k]:
                    selected_keys.add((r["protein"], int(r["start"]), int(r["end"])))
                # Filter entries
                candidate_entries = [
                    e
                    for e in all_entries
                    if (e["protein"], int(e["start"]), int(e["end"])) in selected_keys
                ]
                print(
                    f"[grid] {ref}: prefilter retained {len(candidate_entries)} / {len(all_entries)} windows "
                    f"(Top-{prefilter_k} by lenient score)."
                )

            # Now sweep the combos
            for rt_i in RT_PRESETS:
                for q1_i in Q1_PRESETS:
                    for q3_i in Q3_PRESETS:
                        combo = _combo_label(rt_i, q1_i, q3_i)
                        base_knobs = make_base(rt_i, q1_i, q3_i)

                        rows_all = _score_entries_for_ref(
                            ref,
                            candidate_entries,
                            base_knobs=base_knobs,
                            rt_inner_min=rt_i,
                            gradient_min=float(args.gradient_min),
                            min_matched_frags=min_matched_frags,
                            show_progress=False,  # quieter inner loops
                            precursor_ppm_good_thresh=precursor_ppm_good_thresh,
                        )
                        rows = _top_n(rows_all, int(args.top_n))

                        # Store global per-combo
                        global_rows_per_combo.setdefault(combo, []).extend(rows)

                        # Per-peptide outputs
                        pep_dir = outdir / "grid" / combo / "per_peptide" / ref
                        pep_dir.mkdir(parents=True, exist_ok=True)
                        _write_hits_csv(str(pep_dir / "hits.csv"), rows)
                        if rows:
                            _plot_score_hist(
                                rows,
                                pep_dir / "score_hist.png",
                                f"{ref} [{combo}]: score distribution (Top {len(rows)})",
                            )
                            _plot_scatter_ppm_vs_q3(
                                rows,
                                pep_dir / "scatter_absppm_vs_q3.png",
                                f"{ref} [{combo}]: |Δppm| vs mean Q3 weight",
                            )
                            k = max(0, min(int(args.plots_top_k), len(rows)))
                            for i in range(k):
                                _plot_top_hit_sticks(
                                    ref,
                                    rows[i]["subseq"],
                                    pep_dir / f"stick_top{i+1}.png",
                                )

        # Write per-combo global CSVs
        for combo, rows in global_rows_per_combo.items():
            out_csv = outdir / "grid" / combo / "mzscreen_hits.csv"
            _write_hits_csv(str(out_csv), rows)
            print(f"[grid] {combo}: wrote {len(rows)} rows → {out_csv}")

    # Friendly notes about legacy flags
    if args.cys_mod:
        print(
            f"[mzscreen] NOTE: --cys_mod '{args.cys_mod}' is ignored for this mass-based screen."
        )
    if args.xle_collapse:
        print(
            "[mzscreen] NOTE: --xle_collapse is ignored (I/L collapse not applied to mass filters)."
        )
    if args.score_frag_types_req is not None:
        print(
            "[mzscreen] NOTE: --score_frag_types_req is ignored; transition selection is handled by the scorer."
        )
    if args.require_precursor_match_for_fragment:
        print(
            "[mzscreen] NOTE: --require_precursor_match_for_fragment is implicit (score = Q1×Q3×RT)."
        )

    print("[mzscreen] Done.")


if __name__ == "__main__":
    main()
