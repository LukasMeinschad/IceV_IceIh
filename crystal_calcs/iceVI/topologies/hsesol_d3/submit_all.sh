#!/bin/bash
# Submit all optimisations; each frequency job starts after its optimisation
# finished successfully.  Usage:  ./submit_all.sh            (all)
#                                 ./submit_all.sh 1 5 17     (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(1 4 7 10 13 19 20 22 24 27 33 35 36 39 40 41 44 45); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "vi_c%05d" "$id")
  [ -d "$tag" ] || { echo "no $tag"; continue; }
  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")
  echo "$tag  opt job $jid  freq job $fid"
done
