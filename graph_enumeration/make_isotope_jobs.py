#!/usr/bin/env python3
"""Frequency jobs with isotopic substitution (e.g. D2O) for an existing topology set.

The optimised geometry and the force constants do not depend on the isotope
(Born-Oppenheimer), so only the frequency calculation is repeated: each new job
reads the converged optimisation of the original set (opt/<name>.gui, EXTERNAL)
and runs FREQCALC with ISOTOPES, setting the mass of every atom of the chosen
element.  The atom numbers are taken from the table
"ATOMS ISOTOPIC MASS (AMU) FOR FREQUENCY CALCULATION" of the original frequency
output (same geometry -> same numbering); the new output prints the same table
with the new masses, which is the check that the substitution worked.

    python make_isotope_jobs.py ../crystal_calcs/iceih/topologies/hsesol_d3 \
        --out ../crystal_calcs/iceih/topologies/hsesol_d3_D2O --ncores 10 --lanes 6

Writes <out>/config_XXXXX/freq/<name>_d2o.{d12,sjob}, jobs.csv and submit_lanes.sh.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_crystal_jobs import make_sjob   # noqa: E402

MASS_RE = re.compile(r"ATOMS ISOTOPIC MASS \(AMU\) FOR FREQUENCY CALCULATION\s*\n\s*\n(.*?)\n\s*\n", re.S)


def atom_table(path):
    txt = open(path, errors="replace").read()
    m = MASS_RE.search(txt)
    if not m:
        return None
    return [(int(n), el, float(ms)) for n, el, ms in re.findall(r"(\d+)\s+([A-Z][a-z]?)\s+([\d.]+)", m.group(1))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", help="original job directory (jobs.csv, config_XXXXX/{opt,freq})")
    p.add_argument("--out", required=True)
    p.add_argument("--element", default="H")
    p.add_argument("--mass", type=float, default=2.0141, help="new mass in amu (D: 2.0141)")
    p.add_argument("--suffix", default="d2o")
    p.add_argument("--ncores", type=int, default=10)
    p.add_argument("--lanes", type=int, default=6, help="jobs running at the same time in submit_lanes.sh")
    p.add_argument("--ids", type=int, nargs="+")
    a = p.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.jobs, "jobs.csv"))))
    os.makedirs(a.out, exist_ok=True)
    done = []
    for r in rows:
        cid = int(r["id"])
        if a.ids and cid not in a.ids:
            continue
        name = r["input"]
        src = os.path.join(a.jobs, r["directory"])
        fin = os.path.join(src, "freq", name + ".d12")
        fout = os.path.join(src, "freq", name + ".out")
        sjob = os.path.join(src, "freq", name + ".sjob")
        if not os.path.exists(os.path.join(src, "opt", name + ".out")):
            print(f"{cid}: no optimisation output - skipped")
            continue
        tab = atom_table(fout) if os.path.exists(fout) else None
        if tab is None:
            print(f"{cid}: no mass table in {fout} - skipped (run the original frequency job first)")
            continue
        sub = [(n, el) for n, el, ms in tab if el == a.element]
        lines = open(fin).read().splitlines()
        i = next(k for k, l in enumerate(lines) if l.strip().upper() == "FREQCALC")
        iso = ["ISOTOPES", str(len(sub))] + [f"{n} {a.mass:.4f}" for n, el in sub]
        title = lines[0].replace(" - frequencies", "") + f" - frequencies, {a.element} mass {a.mass:.4f} ({a.suffix.upper()})"
        new = [title] + lines[1:i + 1] + iso + lines[i + 1:]
        nname = f"{name}_{a.suffix}"
        d = os.path.join(a.out, r["directory"], "freq")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, nname + ".d12"), "w").write("\n".join(new) + "\n")
        rel = os.path.relpath(os.path.join(src, "opt"), d)
        pre = (f"# geometry: converged optimisation of the original ({a.element} mass 1.008) run -> fort.34\n"
               f'grep -q "OPT END - CONVERGED" {rel}/{name}.out 2>/dev/null || '
               f"{{ echo 'original optimisation not converged'; exit 1; }}\n"
               f"cp {rel}/{name}.gui ./{nname}.gui || {{ echo 'opt geometry missing'; exit 1; }}")
        tmpl = open(sjob).read()
        # drop the old pre-run block of the original freq sjob, keep header / modules
        tmpl = re.sub(r"# (run only if|optimised geometry).*?\n(?=\s*runmpi)", "", tmpl, flags=re.S)
        tmp = os.path.join(d, ".tmpl")
        open(tmp, "w").write(tmpl)
        s = make_sjob(tmp, f"Ih{cid}_{a.suffix}", nname, a.ncores, pre)
        os.remove(tmp)
        open(os.path.join(d, nname + ".sjob"), "w").write(s)
        done.append((cid, r["directory"], nname, r["space_group"], r["multiplicity"], len(sub)))

    with open(os.path.join(a.out, "jobs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "directory", "input", "space_group", "multiplicity", f"n_{a.element}_substituted"])
        w.writerows(done)
    # smallest cells first (ice XI etc. are cheap and confirm early that the substitution works)
    ids = " ".join(str(c) for c, *_ in sorted(done, key=lambda x: (x[5], x[0])))
    sub = f"""#!/bin/bash
# Frequency jobs with {a.element} mass {a.mass:.4f} ({a.suffix.upper()}), {a.ncores} cores each, at most NLANES at a time.
# Usage: ./submit_lanes.sh [NLANES]     (default {a.lanes} -> {a.lanes * a.ncores} cores)
cd "$(dirname "$0")"
NL=${{1:-{a.lanes}}}
ids=({ids})
declare -a last
for k in "${{!ids[@]}}"; do
  id=${{ids[$k]}}
  lane=$((k % NL))
  tag=$(printf "config_%05d" "$id")
  name=$(ls "$tag/freq"/*.sjob | head -1)
  name=$(basename "$name" .sjob)
  dep=""
  [ -n "${{last[$lane]}}" ] && dep="--dependency=afterany:${{last[$lane]}}"
  jid=$(cd "$tag/freq" && sbatch --parsable $dep "$name.sjob") || {{ echo "submit failed: $tag"; exit 1; }}
  last[$lane]=$jid
  echo "lane $((lane+1)): $tag  job $jid"
done
"""
    sp = os.path.join(a.out, "submit_lanes.sh")
    open(sp, "w").write(sub)
    os.chmod(sp, 0o755)
    print(f"{len(done)} isotope frequency jobs written to {a.out}")


if __name__ == "__main__":
    main()
