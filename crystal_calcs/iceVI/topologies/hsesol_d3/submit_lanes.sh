#!/bin/bash
# Ice VI, 18 symmetric topologies, 30 cores per job, at most NLANES jobs at the same time
# (default 10 lanes x 30 cores = 300 cores).
#
# Each lane is a chain:  opt -> freq -> opt of the next topology -> freq -> ...
# Every job waits for the previous job of its lane (--dependency=afterany), so a failed
# job does not block its lane; a freq job skips itself if its optimisation did not converge.
#
# Usage:  ./submit_lanes.sh            (10 lanes)
#         ./submit_lanes.sh 8          (8 lanes = 240 cores)
cd "$(dirname "$0")"
NL=${1:-10}
# order: ice XV candidates (P-1) first, then restarted optimisations (most advanced first), then fresh ones
ids=(7 36 40 10 13 24 27 19 22 1 20 35 33 4 39 41 44 45)
declare -a last
for k in "${!ids[@]}"; do
  id=${ids[$k]}
  lane=$((k % NL))
  tag=$(printf "config_%05d" "$id")
  name=$(printf "vi_c%05d" "$id")
  dep=""
  [ -n "${last[$lane]}" ] && dep="--dependency=afterany:${last[$lane]}"
  jid=$(cd "$tag/opt" && sbatch --parsable $dep "$name.sjob") || { echo "submit failed: $tag opt"; exit 1; }
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterany:$jid "$name.sjob") || { echo "submit failed: $tag freq"; exit 1; }
  last[$lane]=$fid
  echo "lane $((lane+1)): $tag  opt $jid  freq $fid"
done
