#!/bin/bash
# ice XIII: optimisation, then the frequency job once the optimisation has finished successfully
cd "$(dirname "$0")"
jid=$(cd opt && sbatch --parsable xiii.sjob) || exit 1
fid=$(cd freq && sbatch --parsable --dependency=afterok:$jid xiii.sjob) || exit 1
echo "opt job $jid, freq job $fid"
