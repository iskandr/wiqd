for infile in combined-pfo002-repeat-pfo010.csv; do
  outdir="out-mzscreen-${infile%.csv}"
  python mzscreen.py \
    --in "$infile" \
    --outdir "$outdir" \
    --charges 1,2,3 \
    --score_prec_ppm_good 10 \
    --ppm_tol 30  \
    --full_mz_len_min 5 \
    --full_mz_len_max 21 \
    --fragment_kmin 2 \
    --fragment_kmax 7 \
    --score_frag_types_req 2 \
    --cys_mod carbamidomethyl \
    --xle_collapse  \
    --require_precursor_match_for_fragment \
    --sets_accessions_json mzscreen-protein-sets.json \
    --sets "cytoskeleton_keratins,proteases,antibody_capture,blood_serum,blocking_reagents,streptavidin_avidin_birA,mhc_hardware,albumin" \
    --download_sets \
    --rt off \
    --q1 0.4Da \
    --plots-top-k 1 \
    --run-grid  \
     --top-n 25

done
