#!/bin/bash
# Submit only the frequency jobs (optimisations already finished).
# A freq job is submitted only if opt/<name>.out reports a converged
# optimisation and opt/<name>.gui exists.
# Usage:  ./submit_freq.sh            (all)
#         ./submit_freq.sh 122 3      (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(6 20 93 101 110 161 165 170 177 180 189 207 213 217 223 225 228 236 245 254 258 264 289 293 325 330 332 335 338 340 343 347 348 349 350); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "v_c%05d" "$id")
  if ! grep -q "OPT END - CONVERGED" "$tag/opt/$name.out" 2>/dev/null || [ ! -f "$tag/opt/$name.gui" ]; then
    echo "$tag: optimisation not converged / no .gui - skipped"; continue
  fi
  fid=$(cd "$tag/freq" && sbatch --parsable "$name.sjob")
  echo "$tag  freq job $fid"
done
