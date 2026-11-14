#!/bin/bash
# Submit all optimisations; each frequency job starts after its optimisation
# finished successfully.  Usage:  ./submit_all.sh            (all)
#                                 ./submit_all.sh 1 5 17     (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "vi_c%05d" "$id")
  [ -d "$tag" ] || { echo "no $tag"; continue; }
  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")
  echo "$tag  opt job $jid  freq job $fid"
done
