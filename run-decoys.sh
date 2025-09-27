# Typical run
python decoygen.py \
  --input pfo017-candidates.csv \
  --outdir decoy1 \
  --peptide-col epitope \
  --pos-col MT_pos \
  --mut-notation-col Gene_AA_Change \
  --indexing C1 \
  --albumin-source auto --albumin-acc P02768 --albumin-use prepro \
  --frag-len-min 5 --frag-len-max 12 \
  --by-frag-len-min 2 --by-frag-len-max 11 \
  --N 10



