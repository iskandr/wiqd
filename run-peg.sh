
for infile in pfo010-co.csv pfo002-ev.csv pfo002-tissue-repeat.csv combined-pfo002-repeat-pfo010.csv; do
  outdir="out-peg-${infile%.csv}";
python peg.py \
    --in "$infile" \
    --out "$outdir" \
  --pepcol peptide \
  --hitcol is_hit 
done

