#!/bin/bash
# Submit only the frequency jobs (optimisations already finished).
# A freq job is submitted only if opt/<name>.out reports a converged
# optimisation and opt/<name>.gui exists.
# Usage:  ./submit_freq.sh            (all)
#         ./submit_freq.sh 122 3      (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(3 36 40 72 73 78 83 93 96 100 106 108 114 117 118 122 128 130 132 133 135 137 138 139); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "ih_c%05d" "$id")
  if ! grep -q "OPT END - CONVERGED" "$tag/opt/$name.out" 2>/dev/null || [ ! -f "$tag/opt/$name.gui" ]; then
    echo "$tag: optimisation not converged / no .gui - skipped"; continue
  fi
  fid=$(cd "$tag/freq" && sbatch --parsable "$name.sjob")
  echo "$tag  freq job $fid"
done
