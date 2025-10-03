# =================== Banner ===================
def banner(name: str = "WIQD", title: str = "What Is Q Doing?", args=None):
    width = 68
    inner = width - 2
    bar = "+" + "=" * inner + "+"

    def center_line(s: str) -> str:
        return f"|{s:^{inner}}|"

    name_line = f"({name})"
    print("\n".join([bar, center_line(title), center_line(name_line), bar]))
    if args:
        shown = [
            "input_csv",
            "outdir",
            "alpha",
            "numeric_test",
            "permutes",
            "tie_thresh",
            "min_score",
            "min_nnz_topn",
            "k",
            "collapse",
            "housekeeping_fasta",
            "download_housekeeping",
            "frag_tol_da",
            "frag_tol_ppm",
            # Albumin + RT
            "albumin_source",
            "albumin_fasta",
            "albumin_acc",
            "albumin_expected",
            "albumin_use",
            "ppm_tol",
            "full_mz_len_min",
            "full_mz_len_max",
            "by_mz_len_min",
            "by_mz_len_max",
            "by_seq_len_min",
            "by_seq_len_max",
            "rt_tolerance_min",
            "gradient_min",
            "require_precursor_match_for_frag",
            # Chemistry
            "charges",
            "cys_mod",
            # Fly
            "fly_weights",
            # Summary
            "summary_top_n",
        ]
        print(
            "Args:",
            " | ".join(f"{k}={getattr(args,k)}" for k in shown if hasattr(args, k)),
        )
    print()


def fmt_p(p: float) -> str:
    if not (isinstance(p, float) and p == p):
        return "NA"
    if p <= 0:
        return "<1e-300"
    if p < 1e-12:
        return "<1e-12"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


fmt_q = fmt_p
