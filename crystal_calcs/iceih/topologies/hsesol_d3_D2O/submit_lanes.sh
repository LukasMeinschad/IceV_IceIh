#!/bin/bash
# Frequency jobs with H mass 2.0141 (D2O), 10 cores each, at most NLANES at a time.
# Usage: ./submit_lanes.sh [NLANES]     (default 6 -> 60 cores)
cd "$(dirname "$0")"
NL=${1:-6}
ids=(122 106 114 118 130 135 138 36 40 72 73 78 83 93 96 100 108 117 128 132 133 137 139)
declare -a last
for k in "${!ids[@]}"; do
  id=${ids[$k]}
  lane=$((k % NL))
  tag=$(printf "config_%05d" "$id")
  name=$(ls "$tag/freq"/*.sjob | head -1)
  name=$(basename "$name" .sjob)
  dep=""
  [ -n "${last[$lane]}" ] && dep="--dependency=afterany:${last[$lane]}"
  jid=$(cd "$tag/freq" && sbatch --parsable $dep "$name.sjob") || { echo "submit failed: $tag"; exit 1; }
  last[$lane]=$jid
  echo "lane $((lane+1)): $tag  job $jid"
done
