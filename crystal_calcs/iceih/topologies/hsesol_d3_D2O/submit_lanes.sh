#!/bin/bash
# Frequency jobs with H mass 2.0141 (D2O), 20 cores each.
# config_00003, 00036 and 00122 were already submitted by hand and are skipped.
#   ./submit_lanes.sh        submit all remaining jobs at once (SLURM queues them)
#   ./submit_lanes.sh N      at most N jobs at a time (N chains linked with --dependency=afterany)
cd "$(dirname "$0")"
NL=${1:-0}
ids=(106 114 118 130 135 138 40 72 73 78 83 93 96 100 108 117 128 132 133 137 139)
declare -a last
for k in "${!ids[@]}"; do
  id=${ids[$k]}
  tag=$(printf "config_%05d" "$id")
  name=$(basename "$(ls "$tag/freq"/*.sjob | head -1)" .sjob)
  if [ -e "$tag/freq/$name.out" ]; then echo "$tag: output exists - skipped"; continue; fi
  dep=""
  if [ "$NL" -gt 0 ]; then
    lane=$((k % NL))
    [ -n "${last[$lane]}" ] && dep="--dependency=afterany:${last[$lane]}"
  fi
  jid=$(cd "$tag/freq" && sbatch --parsable $dep "$name.sjob") || { echo "submit failed: $tag"; exit 1; }
  [ "$NL" -gt 0 ] && last[$lane]=$jid
  echo "$tag  job $jid ${dep}"
done
