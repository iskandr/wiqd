python wiqd_mzscreen_sets.py \
  --in pfo010-co.csv \
  --outdir out_sets \
  --kmin 5 --kmax 15 \
  --charges 1,2,3 \
  --tol 30 --unit ppm \
  --download_common \
  --sets albumin,antibody,keratin,serum,cytoskeleton,chaperone \
  --set_mode both \
  --effect_threshold 0.147 --prevalence_threshold 0.10 --min_group_n 10

