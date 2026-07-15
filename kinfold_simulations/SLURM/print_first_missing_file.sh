i=1; while [ -f $(printf "fpt_results/fpts_%06d.txt" $i) ]; do ((i++)); done; echo "First missing file: $(printf "fpts_results/fpts_%06d.txt" $i)"
