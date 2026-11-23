#!/bin/bash
# Submit all optimisations; each frequency job starts after its optimisation
# finished successfully.  Usage:  ./submit_all.sh            (all)
#                                 ./submit_all.sh 1 5 17     (selected ids)
cd "$(dirname "$0")"
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then ids=(6 20 93 101 110 161 165 170 177 180 189 207 213 217 223 225 228 236 245 254 258 264 289 293 325 330 332 335 338 340 343 347 348 349 350); fi
for id in "${ids[@]}"; do
  tag=$(printf "config_%05d" "$id")
  name=$(printf "v_c%05d" "$id")
  [ -d "$tag" ] || { echo "no $tag"; continue; }
  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")
  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")
  echo "$tag  opt job $jid  freq job $fid"
done
