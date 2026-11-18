#!/bin/bash
# Submit all optimisations; each frequency job starts after its optimisation
# finished successfully.  Usage:  ./submit_all.sh            (all)
#                                 ./submit_all.sh 1 5 17     (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(1 2 3); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "xiv_c%05d" "$id")
  [ -d "$tag" ] || { echo "no $tag"; continue; }
  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")
  echo "$tag  opt job $jid  freq job $fid"
done
