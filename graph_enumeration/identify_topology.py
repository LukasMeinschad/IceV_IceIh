#!/usr/bin/env python3
"""Identify which enumerated H-bond topology a given ice structure is.

Takes any proton-ordered or -"disordered" snapshot (e.g. a GenIce output or an
old P1 CRYSTAL input) whose cell is a cell of our enumeration, maps its oxygens
onto our oxygen lattice (any axis order / origin, small distortions allowed),
reads the H-bond directions from the H positions, and returns the topology id
of enumerate_ice.py (same numbering as crystal_calcs/.../enumeration and the
CRYSTAL jobs), its space group and multiplicity.

    python identify_topology.py Ih --cell ortho2x1x1 ../crystal_calcs/iceih/opt/atompos/iceih.d12
    python identify_topology.py VI ice6_genice.cif

Formats: CRYSTAL .d12 (P1 geometry), CIF with P1 fractional coordinates,
extended xyz with Lattice="..." (GenIce: -f exyz).
"""
import argparse
import itertools
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iceenum.structures import STRUCTURES                                        # noqa: E402
from iceenum.crystal import IceGraph, lattice_vectors                            # noqa: E402
from iceenum.enumerate import enumerate_configurations, merge_crystal_equivalent, _keys  # noqa: E402
from iceenum import writers as W                                                 # noqa: E402


# ----------------------------------------------------------------- readers
def read_d12(path):
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.strip().upper() == "CRYSTAL")
    if int(L[i + 2].split()[0]) != 1:
        raise SystemExit("only P1 .d12 geometries are supported")
    lat = lattice_vectors(*map(float, L[i + 3].split()[:6]))
    n = int(L[i + 4].split()[0])
    at = [l.split() for l in L[i + 5:i + 5 + n]]
    z = np.array([int(a[0]) % 100 for a in at])
    f = np.array([[float(x) for x in a[1:4]] for a in at])
    return lat, f, z


def read_cif(path):
    txt = open(path).read()
    g = lambda k: float(re.search(rf"{k}\s+([-\d.]+)", txt).group(1))
    lat = lattice_vectors(g("_cell_length_a"), g("_cell_length_b"), g("_cell_length_c"),
                          g("_cell_angle_alpha"), g("_cell_angle_beta"), g("_cell_angle_gamma"))
    lines = txt.splitlines()
    k = max(i for i, l in enumerate(lines) if l.strip().startswith("_atom_site_fract_z"))
    hdr = []
    j = k
    while j >= 0 and lines[j].strip().startswith("_atom_site"):
        hdr.insert(0, lines[j].strip())
        j -= 1
    ix, iy, iz = (hdr.index(f"_atom_site_fract_{c}") for c in "xyz")
    it = hdr.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in hdr else hdr.index("_atom_site_label")
    f, z = [], []
    for l in lines[k + 1:]:
        p = l.split()
        if len(p) < len(hdr) or l.strip().startswith(("_", "loop_", "#")):
            if f:
                break
            continue
        el = re.match(r"[A-Za-z]+", p[it]).group(0).capitalize()
        el = {"Ow": "O", "Hw": "H", "D": "H"}.get(el, el)
        if el not in ("O", "H"):
            continue
        z.append(8 if el == "O" else 1)
        f.append([float(re.sub(r"\(.*\)", "", p[c])) for c in (ix, iy, iz)])
    return lat, np.array(f), np.array(z)


def read_exyz(path):
    L = open(path).read().splitlines()
    n = int(L[0])
    lat = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', L[1]).group(1).split()]).reshape(3, 3)
    el = [l.split()[0] for l in L[2:2 + n]]
    cart = np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2 + n]])
    z = np.array([8 if e.upper().startswith("O") else 1 for e in el])
    return lat, cart @ np.linalg.inv(lat), z


def read_structure(path):
    ext = path.lower().rsplit(".", 1)[-1]
    return {"d12": read_d12, "cif": read_cif, "xyz": read_exyz, "exyz": read_exyz}[ext](path)


# ------------------------------------------------------------- alignment
def align(g, lat_u, fO, tol_cell=0.12, tol_pos=0.8):
    """Integer matrix T (lat_u ~ T @ lat_ours up to a rotation) and origin shift that map
    the user's oxygens onto our sites.  Returns (T, shift, site index of every user O)."""
    Gu = lat_u @ lat_u.T
    Go = g.lattice @ g.lattice.T
    scale = np.sqrt(np.diag(Go))
    best = None
    for vals in itertools.product((-1, 0, 1), repeat=9):
        T = np.array(vals, float).reshape(3, 3)
        if abs(abs(np.linalg.det(T)) - 1) > 1e-6:
            continue
        Gt = T @ Go @ T.T
        if np.max(np.abs(Gt - Gu) / np.outer(np.sqrt(np.diag(Gu)), np.sqrt(np.diag(Gu)))) > tol_cell:
            continue
        x = fO @ T                                   # user O in our fractional coordinates
        for s0 in range(g.n):
            shift = g.frac[s0] - x[0]
            y = x + shift
            idx, dmax = [], 0.0
            for p in y:
                d = g.frac - p
                d -= np.round(d)
                r = np.linalg.norm(d @ g.lattice, axis=1)
                k = int(np.argmin(r))
                idx.append(k)
                dmax = max(dmax, r[k])
            if len(set(idx)) == g.n and dmax < tol_pos and (best is None or dmax < best[3]):
                best = (T, shift, idx, dmax)
    if best is None:
        raise ValueError("could not map the oxygens onto the enumerated cell "
                         "(different supercell or too distorted; try another --cell)")
    return best


def bond_variables(g, T, shift, fH):
    """b_r for every bond of g from the H positions (each H belongs to its nearest O)."""
    xH = fH @ T + shift
    L = g.lattice
    b = np.zeros(g.nb, np.int8)
    for h in xH:
        d = g.frac - h
        d -= np.round(d)
        r = np.linalg.norm(d @ L, axis=1)
        o = int(np.argmin(r))
        oh = (h - g.frac[o])
        oh -= np.round(oh)
        oh = oh @ L
        # bond of o whose direction is closest to the O-H vector
        bestc, bk, bs = -2, None, 0
        for k, bd in enumerate(g.bonds):
            if o not in (bd.tail, bd.head):
                continue
            v = (g.frac[bd.head] + bd.image - g.frac[bd.tail]) @ L
            s = 1 if bd.tail == o else -1
            v = v * s
            c = (v @ oh) / np.linalg.norm(v) / np.linalg.norm(oh)
            if c > bestc:
                bestc, bk, bs = c, k, s
        if b[bk] != 0:
            raise ValueError("two H atoms on the same O...O bond - not an ice-rule structure")
        b[bk] = bs
    if np.any(b == 0):
        raise ValueError(f"{int((b == 0).sum())} bonds without H - not an ice-rule structure")
    if np.any(b.astype(int) @ g.A.astype(int) != 0):
        raise ValueError("structure violates the ice rules")
    return b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=sorted(STRUCTURES))
    p.add_argument("structure", nargs="+")
    p.add_argument("--cell", help="cell preset of the enumeration (default as in enumerate_ice.py)")
    p.add_argument("--jobs", help="job directory (jobs.csv) to report whether the topology is calculated")
    a = p.parse_args()
    from enumerate_ice import DEFAULT_CELL
    st = STRUCTURES[a.phase]
    cname = a.cell or DEFAULT_CELL.get(a.phase)
    g = IceGraph(st, st.presets[cname])
    B, m = enumerate_configurations(g, verbose=False)
    Bm, mm, cls = merge_crystal_equivalent(g, B, m, verbose=False)
    index = {k: i for i, k in enumerate(_keys(B))}
    jobs = {}
    if a.jobs and os.path.exists(os.path.join(a.jobs, "jobs.csv")):
        import csv
        jobs = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(a.jobs, "jobs.csv")))}
    print(f"{st.name}, cell '{cname}': {g.n} H2O, {len(Bm)} topologies, {int(mm.sum())} configurations")
    for path in a.structure:
        lat, f, z = read_structure(path)
        fO, fH = f[z == 8], f[z == 1]
        if len(fO) != g.n or len(fH) != 2 * g.n:
            print(f"{path}: {len(fO)} O / {len(fH)} H, but the cell has {g.n} H2O - use another --cell")
            continue
        try:
            T, shift, idx, dmax = align(g, lat, fO)
            b = bond_variables(g, T, shift, fH)
        except ValueError as e:
            print(f"{path}: {e}")
            continue
        # canonical form under the cell group
        best = _keys(b[None, :])[0]
        for pm, sg in zip(g.perms, g.signs):
            t = np.empty_like(b)
            t[pm] = b * sg
            k = _keys(t[None, :])[0]
            best = min(best, k)
        row = index[best]
        tid = int(cls[row]) + 1
        fr, nm = W.build_atoms(g, Bm[tid - 1])
        sym = W.symmetry_info(g.lattice, fr, nm, full=False)
        P = W.dipole(g, b)
        calc = ""
        if jobs:
            calc = "  -> calculated, see config_%05d" % tid if tid in jobs else "  -> not in the calculated set"
        print(f"{path}:\n    topology id {tid}  ({sym['symbol']}, multiplicity {int(mm[tid - 1])}, "
              f"|P| = {np.linalg.norm(P):.2f})  max O displacement from the lattice {dmax:.2f} A{calc}")


if __name__ == "__main__":
    main()
