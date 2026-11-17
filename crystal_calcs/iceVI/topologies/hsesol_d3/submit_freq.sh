#!/bin/bash
# Submit only the frequency jobs (optimisations already finished).
# A freq job is submitted only if opt/<name>.out reports a converged
# optimisation and opt/<name>.gui exists.
# Usage:  ./submit_freq.sh            (all)
#         ./submit_freq.sh 122 3      (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(1 4 7 10 13 19 20 22 24 27 33 35 36 39 40 41 44 45); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "vi_c%05d" "$id")
  if ! grep -q "OPT END - CONVERGED" "$tag/opt/$name.out" 2>/dev/null || [ ! -f "$tag/opt/$name.gui" ]; then
    echo "$tag: optimisation not converged / no .gui - skipped"; continue
  fi
  fid=$(cd "$tag/freq" && sbatch --parsable "$name.sjob")
  echo "$tag  freq job $fid"
done
