for infile in pfo010-co.csv pfo002-2023.csv; do
  outdir="mzscreen-out-${infile%.csv}"
python mzscreen.py \
  --in "$infile" \
  --outdir "$outdir" \
  --kmin 5 --kmax 15 \
  --charges 1,2,3 \
  --prec_ppm 30 \
  --frag_ppm 30 \
  --download_common \
  --sets albumin,antibody,keratin,serum,cytoskeleton,chaperone \
  --set_mode both \
  --effect_threshold 0.147 --prevalence_threshold 0.10 --min_group_n 10
done
