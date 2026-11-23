#!/bin/bash
# Ice V, 35 Cc topologies of the primitive cell (14 H2O), 30 cores per job.
# Step 1 (this script): only the OPTIMISATIONS (energy ranking), at most NLANES at the same time.
#   Each lane is a chain opt -> opt -> ... (--dependency=afterany: a failed job does not block its lane).
# Step 2: frequencies for selected topologies (lowest energies) with ./submit_freq.sh ID ID ...
#   or all opt+freq pairs of selected ids with ./submit_all.sh ID ID ...
#
# Usage:  ./submit_lanes.sh        (6 lanes x 30 cores = 180 cores)
#         ./submit_lanes.sh 10     (10 lanes = 300 cores)
cd "$(dirname "$0")"
NL=${1:-6}
ids=(6 20 93 101 110 161 165 170 177 180 189 207 213 217 223 225 228 236 245 254 258 264 289 293 325 330 332 335 338 340 343 347 348 349 350)
declare -a last
for k in "${!ids[@]}"; do
  id=${ids[$k]}
  lane=$((k % NL))
  tag=$(printf "config_%05d" "$id")
  name=$(printf "v_c%05d" "$id")
  if grep -q "OPT END - CONVERGED" "$tag/opt/$name.out" 2>/dev/null; then echo "$tag: already converged - skipped"; continue; fi
  dep=""
  [ -n "${last[$lane]}" ] && dep="--dependency=afterany:${last[$lane]}"
  jid=$(cd "$tag/opt" && sbatch --parsable $dep "$name.sjob") || { echo "submit failed: $tag"; exit 1; }
  last[$lane]=$jid
  echo "lane $((lane+1)): $tag  opt $jid"
done
