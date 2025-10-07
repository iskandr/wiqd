for infile in pfo004-candidates.csv pfo017-candidates-v2.csv; do
  outdir="out-decoys-${infile%.csv}"
  python decoygen.py \
    --in "$infile" \
    --outdir "$outdir" \
    --peptide-col Epitope,MT_epitope,epitope,mt_epitope,peptide \
    --pos-col MT_pos_base1,Mutation_Position \
    --mut-notation-col Gene_AA_Change \
    --indexing 1 \
    --require-mut-idx0 1 \
    --download-sets 1 \
    --sets albumin,keratins,proteases,mhc_hardware \
    --nterm-good RYLFDM \
    --nterm-bad-strong KP \
    --nterm-bad-weak STVA \
    --nterm-targets RYLFDM \
    --bad-internal Q \
    --rule-flank-gap-min 2 \
    --rank-weights "hydro:0.4,conf_mz:0.2,hk:0.25,polymer:0.15" \
    --overall-rule-weight 0.5 \
    --overall-cont-weight 0.5 \
    --polymer-families PEG,PPG,PTMEG,PDMS --polymer-endgroups auto --polymer-adducts H,Na,K,NH4 --polymer-z 1,2,3 \
    --polymer-ppm-strict 10 \
    --N 10 \
    --require-contam-match 1 \
    --min-confusability-ratio 0.90 \
    --download-housekeeping \
    --dedup-k 5 \
    --verbose
done
