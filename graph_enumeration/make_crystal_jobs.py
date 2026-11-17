#!/usr/bin/env python3
"""Turn the symmetry-adapted geometries written by enumerate_ice.py into
ready-to-submit CRYSTAL optimisation + frequency jobs.

For every topology a directory  <out>/config_XXXXX/{opt,freq}/  is created:

  opt/<name>.d12    geometry of config_XXXXX_sym.d12 (space group + asym. unit)
                    + everything after the atom list of --opt-template
  opt/<name>.sjob   SLURM script made from --sjob-template
  freq/<name>.d12   EXTERNAL (reads the optimised structure incl. symmetry from
                    <name>.gui = fort.34 written by the opt run)
                    + everything after the atom list of --freq-template
  freq/<name>.sjob  copies ../opt/<name>.gui before running CRYSTAL

plus <out>/submit_all.sh, which submits every opt job and the corresponding
freq job with  sbatch --dependency=afterok:<opt job>.

    python make_crystal_jobs.py results/Ih_ortho2x1x1 \
        --opt-template  ../crystal_calcs/iceih/disp_cor/opt/iceih.d12 \
        --freq-template ../crystal_calcs/iceih/disp_cor/freq/iceih.d12 \
        --sjob-template ../crystal_calcs/iceih/disp_cor/opt/iceih.sjob \
        --prefix ih --out ../crystal_calcs/iceih/topologies/hsesol_d3
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iceenum.writers import read_d12_template   # noqa: E402


def geometry_block(path):
    """Title ... atom list of a CRYSTAL input (without the following keywords)."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().upper() == "CRYSTAL")
    n = int(lines[i + 4].split()[0])
    return lines[:i + 5 + n]


def make_sjob(template, jobname, inp, ncores, pre=""):
    s = open(template).read()
    s = re.sub(r"(#SBATCH\s+-J\s+)\S+", r"\g<1>" + jobname, s)
    if ncores:
        s = re.sub(r"(#SBATCH\s+--ntasks-per-node=)\d+", rf"\g<1>{ncores}", s)
    s = re.sub(r"^(\s*runmpi\S*)\s+\d+\s+\S+", lambda m: f"{m.group(1)} {ncores or ''} {inp}".replace("  ", " "),
               s, flags=re.M)
    if pre:
        s = re.sub(r"^(\s*runmpi\S*.*)$", pre + r"\n\1", s, count=1, flags=re.M)
    return s


def add_raman(tail):
    """Insert INTRAMAN + INTCPHF into the FREQCALC block (INTENS is kept/added)."""
    lines = tail.splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().upper() == "FREQCALC")
    j = next(k for k, l in enumerate(lines) if l.strip().upper() == "ENDFREQ")
    body = [l for l in lines[i + 1:j] if l.strip().upper() not in ("INTENS", "INTRAMAN")]
    if any(l.strip().upper() == "INTCPHF" for l in body):
        raise ValueError("template already contains INTCPHF")
    new = ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "END"] + body + ["ENDFREQ"]
    return "\n".join(lines[:i] + new + lines[j + 1:]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="output directory of enumerate_ice.py (contains d12/ and summary.csv)")
    p.add_argument("--opt-template", required=True)
    p.add_argument("--freq-template", required=True)
    p.add_argument("--sjob-template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="ice", help="input-file prefix, e.g. ih -> ih_c00001.d12")
    p.add_argument("--jobname", default=None, help="SLURM job-name prefix (default: prefix)")
    p.add_argument("--ncores", type=int, default=None, help="override ntasks-per-node / runmpi cores")
    p.add_argument("--ids", type=int, nargs="+", help="only these topology ids")
    p.add_argument("--raman", action="store_true",
                   help="IR + Raman intensities in the frequency run via CPHF/CPKS "
                        "(INTENS, INTRAMAN, INTCPHF ... END inside FREQCALC)")
    a = p.parse_args()

    opt_tail = read_d12_template(a.opt_template)
    freq_tail = read_d12_template(a.freq_template)
    if a.raman:
        freq_tail = add_raman(freq_tail)
    m = re.search(r"--ntasks-per-node=(\d+)", open(a.sjob_template).read())
    ncores = a.ncores or (int(m.group(1)) if m else None)
    jn = a.jobname or a.prefix

    summary = {}
    sfile = os.path.join(a.results, "summary.csv")
    if os.path.exists(sfile):
        for row in csv.DictReader(open(sfile)):
            summary[int(row["id"])] = row

    files = sorted(glob.glob(os.path.join(a.results, "d12", "config_*_sym.d12")))
    if not files:
        sys.exit("no config_*_sym.d12 found - run enumerate_ice.py with --write-all / --select")
    os.makedirs(a.out, exist_ok=True)
    done = []
    for f in files:
        cid = int(re.search(r"config_(\d+)_sym", f).group(1))
        if a.ids and cid not in a.ids:
            continue
        tag = f"config_{cid:05d}"
        name = f"{a.prefix}_c{cid:05d}"
        geo = geometry_block(f)
        dopt = os.path.join(a.out, tag, "opt")
        dfrq = os.path.join(a.out, tag, "freq")
        os.makedirs(dopt, exist_ok=True)
        os.makedirs(dfrq, exist_ok=True)
        open(os.path.join(dopt, name + ".d12"), "w").write("\n".join(geo) + "\n" + opt_tail)
        # runmpi17 exits 0 even if CRYSTAL failed: report failure to SLURM so that the
        # dependent freq job (--dependency=afterok) is not started
        post = (f'grep -q "OPT END - CONVERGED" {name}.out || '
                f'{{ echo "optimisation not converged / CRYSTAL failed"; exit 1; }}')
        open(os.path.join(dopt, name + ".sjob"), "w").write(
            make_sjob(a.sjob_template, f"{jn}{cid}_opt", name, ncores).rstrip("\n") + "\n" + post + "\n")
        open(os.path.join(dfrq, name + ".d12"), "w").write(f"{geo[0]} - frequencies\nEXTERNAL\n" + freq_tail)
        pre = (f"# run only if the optimisation converged; its geometry (with symmetry) -> fort.34\n"
               f'grep -q "OPT END - CONVERGED" ../opt/{name}.out 2>/dev/null || '
               f"{{ echo 'optimisation not converged - frequencies skipped'; exit 1; }}\n"
               f"cp ../opt/{name}.gui ./{name}.gui || {{ echo 'opt geometry missing'; exit 1; }}")
        open(os.path.join(dfrq, name + ".sjob"), "w").write(
            make_sjob(a.sjob_template, f"{jn}{cid}_freq", name, ncores, pre))
        p1 = f.replace("_sym.d12", "_P1.d12")
        if os.path.exists(p1):     # keep the P1 geometry for reference
            shutil.copy(p1, os.path.join(a.out, tag, f"{name}_P1_geometry.d12"))
        done.append((cid, tag, name))

    sub = ["#!/bin/bash",
           "# Submit all optimisations; each frequency job starts after its optimisation",
           "# finished successfully.  Usage:  ./submit_all.sh            (all)",
           "#                                 ./submit_all.sh 1 5 17     (selected ids)",
           'cd "$(dirname "$0")"',
           'ids=("$@")',
           'if [ ${#ids[@]} -eq 0 ]; then ids=(' + " ".join(str(c) for c, _, _ in done) + '); fi',
           'for id in "${ids[@]}"; do',
           '  tag=$(printf "config_%05d" "$id")',
           f'  name=$(printf "{a.prefix}_c%05d" "$id")',
           '  [ -d "$tag" ] || { echo "no $tag"; continue; }',
           '  jid=$(cd "$tag/opt" && sbatch --parsable "$name.sjob")',
           '  fid=$(cd "$tag/freq" && sbatch --parsable --dependency=afterok:$jid "$name.sjob")',
           '  echo "$tag  opt job $jid  freq job $fid"',
           "done", ""]
    sp = os.path.join(a.out, "submit_all.sh")
    open(sp, "w").write("\n".join(sub))
    os.chmod(sp, 0o755)

    # frequency jobs only, for topologies whose optimisation already converged
    frq = ["#!/bin/bash",
           "# Submit only the frequency jobs (optimisations already finished).",
           "# A freq job is submitted only if opt/<name>.out reports a converged",
           "# optimisation and opt/<name>.gui exists.",
           "# Usage:  ./submit_freq.sh            (all)",
           "#         ./submit_freq.sh 122 3      (selected ids)",
           'cd "$(dirname "$0")"',
           'ids=("$@")',
           'if [ ${#ids[@]} -eq 0 ]; then ids=(' + " ".join(str(c) for c, _, _ in done) + '); fi',
           'for id in "${ids[@]}"; do',
           '  tag=$(printf "config_%05d" "$id")',
           f'  name=$(printf "{a.prefix}_c%05d" "$id")',
           '  if ! grep -q "OPT END - CONVERGED" "$tag/opt/$name.out" 2>/dev/null || [ ! -f "$tag/opt/$name.gui" ]; then',
           '    echo "$tag: optimisation not converged / no .gui - skipped"; continue',
           '  fi',
           '  fid=$(cd "$tag/freq" && sbatch --parsable "$name.sjob")',
           '  echo "$tag  freq job $fid"',
           "done", ""]
    fp = os.path.join(a.out, "submit_freq.sh")
    open(fp, "w").write("\n".join(frq))
    os.chmod(fp, 0o755)

    with open(os.path.join(a.out, "jobs.csv"), "w", newline="") as fh:
        cols = ["id", "directory", "input", "space_group", "sg_number", "n_symops",
                "atoms_asym_unit", "multiplicity", "weight", "|P|"]
        w = csv.writer(fh)
        w.writerow(cols)
        for cid, tag, name in done:
            r = summary.get(cid, {})
            w.writerow([cid, tag, name] + [r.get(c, "") for c in cols[3:]])
    print(f"{len(done)} opt + freq job pairs written to {a.out}")


if __name__ == "__main__":
    main()
