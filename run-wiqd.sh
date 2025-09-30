for infile in pfo010-co.csv pfo002-2023.csv pfo002-tissue-repeat.csv combined-pfo002-repeat-pfo010.csv; do
  outdir="wiqd2-out-${infile%.csv}"
  python wiqd.py \
    --in "$infile" \
    --outdir "$outdir" \
    --albumin_source embedded --albumin_use mature \
    --ppm_tol 30 \
    --full_mz_len_min 5 --full_mz_len_max 15 \
    --by_mz_len_min 2 --by_mz_len_max 11 \
    --charges 1,2,3 \
    --collapse xle \
    --cys_mod none \
    --rt_tolerance_min 1 \
    --k 4 \
    --require_precursor_match_for_frag \
    --summary_top_n 30
done

