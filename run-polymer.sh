
for infile in  combined-pfo002-repeat-pfo010.csv; do
  outdir="out-polymer-${infile%.csv}";
python polymer.py \
    --in "$infile" \
    --out "$outdir" \
  --pepcol peptide \
  --hitcol is_hit 
done

