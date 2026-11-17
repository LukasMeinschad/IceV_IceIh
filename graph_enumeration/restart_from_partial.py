#!/usr/bin/env python3
"""Restart CRYSTAL ATOMONLY optimisations from the last geometry of a cancelled run.

For every config_XXXXX/opt/<name>.out that contains at least one completed
optimisation point, the last printed geometry (all atoms of CRYSTAL's
primitive cell + its lattice parameters) is read, re-standardised with spglib
(same space group, same cell -- checked), and written as the geometry block of
<name>.d12 (keywords after the atom list are kept).  The old outputs are moved
to config_XXXXX/opt/cancelled_run1/ (and freq/cancelled_run1/).

    python restart_from_partial.py ../crystal_calcs/iceVI/topologies/hsesol_d3 --ids 1 10 13
"""
import argparse
import csv
import glob
import os
import re
import shutil

import numpy as np

from iceenum.crystal import cell_parameters, lattice_vectors
from iceenum import writers as W

BLOCK = re.compile(
    r"(?:COORDINATE OPTIMIZATION - POINT\s+(\d+)).*?"
    r"A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([-\d.\s]+?)\n.*?"
    r"ATOMS IN THE ASYMMETRIC UNIT\s+\d+ - ATOMS IN THE UNIT CELL:\s+(\d+)\s*\n.*?\*{20,}\s*\n(.*?)(?:\n\s*\n|\n \*)",
    re.S)
ATOM = re.compile(r"^\s*\d+\s+[TF]\s+(\d+)\s+\w+\s+(-?[\d.]+E[+-]\d+)\s+(-?[\d.]+E[+-]\d+)\s+(-?[\d.]+E[+-]\d+)", re.M)


def last_geometry(path):
    txt = open(path, errors="replace").read()
    blocks = list(BLOCK.finditer(txt))
    if not blocks:
        return None
    b = blocks[-1]
    point = int(b.group(1))
    params = [float(x) for x in b.group(2).split()]
    nat = int(b.group(3))
    atoms = ATOM.findall(b.group(4))
    if len(params) != 6 or len(atoms) != nat:
        raise RuntimeError(f"{path}: could not read the geometry of point {point}")
    z = np.array([int(a[0]) for a in atoms])
    frac = np.array([[float(a[1]), float(a[2]), float(a[3])] for a in atoms]) % 1.0
    return point, lattice_vectors(*params), frac, z


def geometry_of_d12(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().upper() == "CRYSTAL")
    sg = int(lines[i + 2].split()[0])
    par = [float(x) for x in lines[i + 3].split()]
    n = int(lines[i + 4].split()[0])
    return lines, i, sg, par, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--ids", type=int, nargs="+", help="only these ids (default: all in jobs.csv)")
    p.add_argument("--min-points", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.jobs, "jobs.csv"))))
    for r in rows:
        cid = int(r["id"])
        if a.ids and cid not in a.ids:
            continue
        dopt = os.path.join(a.jobs, r["directory"], "opt")
        out = os.path.join(dopt, r["input"] + ".out")
        d12 = os.path.join(dopt, r["input"] + ".d12")
        if not os.path.exists(out):
            print(f"{cid:4d}: no previous output - fresh start")
            continue
        g = last_geometry(out)
        if g is None or g[0] < a.min_points:
            print(f"{cid:4d}: no completed optimisation point - fresh start")
            archive(a, r, dopt)
            continue
        point, L, frac, z = g
        sym = W.symmetry_info(L, frac, z, symprec=1e-3)
        lines, i, sg0, par0, n0 = geometry_of_d12(d12)
        # the new standard cell must be the old one (ATOMONLY keeps the cell)
        par_new = W._crystal_params(sym["number"], sym["std_lattice"])
        ok = sym["number"] == sg0 and len(par_new) == len(par0) and np.allclose(par_new, par0, atol=2e-4) \
            and len(sym["asym"]) == n0
        if not ok:
            print(f"{cid:4d}: CHECK FAILED (SG {sym['number']} vs {sg0}, cell {np.round(par_new, 4)} vs {par0}, "
                  f"asym {len(sym['asym'])} vs {n0}) - keeping fresh start")
            continue
        print(f"{cid:4d}: restart from point {point:3d}  ({sym['symbol']}, {len(sym['asym'])} asym. atoms, cell unchanged)")
        if a.dry_run:
            continue
        archive(a, r, dopt)
        tail = W.read_d12_template(d12)
        title = lines[0].split(" - restarted")[0] + f" - restarted from point {point} of cancelled run 1"
        W.write_d12(d12, title, None, None, None, sym, tail)


def archive(a, r, dopt):
    if a.dry_run:
        return
    for d in (dopt, os.path.join(os.path.dirname(dopt), "freq")):
        old = [f for f in glob.glob(os.path.join(d, "*")) if os.path.isfile(f)
               and (f.endswith((".out", ".gui", ".xyz", ".born")) or os.path.basename(f).startswith("slurm-"))]
        if old:
            dst = os.path.join(d, "cancelled_run1")
            os.makedirs(dst, exist_ok=True)
            for f in old:
                shutil.move(f, os.path.join(dst, os.path.basename(f)))


if __name__ == "__main__":
    main()
