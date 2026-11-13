#!/bin/bash
# Submit all optimisations; each frequency job starts after its optimisation
# finished successfully.  Usage:  ./submit_all.sh            (all)
#                                 ./submit_all.sh 1 5 17     (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(3 36 40 72 73 78 83 93 96 100 106 108 114 117 118 122 128 130 132 133 135 137 138 139); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "ih_c%05d" "$id")
  [ -d "$tag" ] || { echo "no $tag"; continue; }
  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")
  echo "$tag  opt job $jid  freq job $fid"
done
