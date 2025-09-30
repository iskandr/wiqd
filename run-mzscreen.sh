for infile in pfo010-co.csv pfo002-ev.csv pfo002-tissue-repeat.csv combined-pfo002-repeat-pfo010.csv; do
  outdir="out-mzscreen-${infile%.csv}"
python mzscreen.py \
  --in "$infile" \
  --outdir "$outdir" \
  --charges 1,2,3 \
  --require_precursor_match_for_fragment \
  --ppm_tol 30  \
  --full_mz_len_min 5 \
  --full_mz_len_max 15 \
  --fragment_kmin 2 \
  --fragment_kmax 7 \
  --cys_mod none \
  --xle_collapse  \
  --sets_accessions_json mzscreen-protein-sets.json \
  --sets "cytoskeleton_keratins,proteases,antibody_capture,blood_serum,blocking_reagents,streptavidin_avidin_birA,mhc_hardware,albumin" \
  --download_sets
done
