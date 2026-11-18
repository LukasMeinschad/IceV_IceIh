#!/usr/bin/env python3
"""Ordered models of a partially proton-ordered ice from a CIF with H/D occupancies.

Example: ice XIV (P2_12_12_1) where four D sites are fully occupied and two
pairs of sites at the two ends of O1-O2 bonds are occupied ~0.6/0.4.

* reads cell, symmetry operations and atom sites (with occupancies) from the CIF
* builds the oxygen lattice / H-bond graph of a (super)cell
* assigns every H/D site to (bond, donor end); sites with occupancy 1 fix the
  bond direction, the others leave it free
* enumerates all ice-rule configurations of the free bonds, groups them into
  symmetry-distinct structures (space group of the CIF + supercell translations)
  and reports their multiplicity and the occupancy of every partial site
* writes, per structure, P1 and symmetry-adapted CIF / CRYSTAL geometries using
  the experimental O and H/D positions of the occupied sites, plus summary.csv
  (compatible with make_crystal_jobs.py)

    python enumerate_partial_order.py ice14.cif -o ../crystal_calcs/iceXIV/ordered_models
    python enumerate_partial_order.py ice14.cif --supercell 1 1 2 -o ...
"""
import argparse
import collections
import csv
import itertools
import os
import re
import sys

import numpy as np
import spglib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iceenum.structures import IceStructure          # noqa: E402
from iceenum.crystal import IceGraph, lattice_vectors   # noqa: E402
from iceenum import writers as W                      # noqa: E402


def num(s):
    return float(re.sub(r"\(.*?\)", "", s))


def parse_cif(path):
    txt = open(path).read()
    cell = [num(re.search(rf"{k}\s+(\S+)", txt).group(1)) for k in
            ("_cell_length_a", "_cell_length_b", "_cell_length_c",
             "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")]
    lines = txt.splitlines()
    ops = []
    for i, l in enumerate(lines):
        if l.strip().startswith(("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")):
            for m in lines[i + 1:]:
                m = m.strip()
                if not m or m.startswith(("_", "loop_")):
                    break
                ops.append(m.strip("'\"").split("'")[0].strip())
    # atom loop
    k = next(i for i, l in enumerate(lines) if l.strip().startswith("_atom_site_fract_x"))
    s = k
    while lines[s - 1].strip().startswith("_atom_site"):
        s -= 1
    hdr = []
    j = s
    while lines[j].strip().startswith("_atom_site"):
        hdr.append(lines[j].strip())
        j += 1
    atoms = []
    for l in lines[j:]:
        p = l.split()
        if len(p) < len(hdr) or l.strip().startswith(("_", "loop_", "#")):
            if atoms:
                break
            continue
        rec = dict(zip(hdr, p))
        el = rec.get("_atom_site_type_symbol", re.match(r"[A-Za-z]+", rec["_atom_site_label"]).group(0))
        el = "H" if el.upper() in ("D", "H") else el.capitalize()
        occ = num(rec.get("_atom_site_occupancy", "1"))
        x = [num(rec[f"_atom_site_fract_{c}"]) for c in "xyz"]
        atoms.append((rec["_atom_site_label"], el, np.array(x), occ))
    return cell, ops, atoms


def parse_op(op):
    R = np.zeros((3, 3), int)
    t = np.zeros(3)
    for i, e in enumerate(op.replace(" ", "").lower().split(",")):
        e = e.replace("-", "+-")
        for term in e.split("+"):
            if not term:
                continue
            for k, ax in enumerate("xyz"):
                if term.endswith(ax):
                    R[i, k] = -1 if term.startswith("-") else 1
                    break
            else:
                t[i] = eval(term)
    return R, t % 1


def hall_from_ops(ops):
    target = {(tuple(R.ravel()), tuple(np.round(t, 4))) for R, t in map(parse_op, ops)}
    for h in range(1, 531):
        s = spglib.get_symmetry_from_database(h)
        cand = {(tuple(R.ravel()), tuple(np.round(t % 1, 4))) for R, t in zip(s["rotations"], s["translations"])}
        if cand == target:
            return h
    raise SystemExit("symmetry operations of the CIF not found in the spglib database")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cif")
    p.add_argument("--supercell", type=int, nargs=3, default=[1, 1, 1])
    p.add_argument("--name", default="ice")
    p.add_argument("--oo-range", type=float, nargs=2, default=[2.5, 3.13])
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()

    cell, ops, atoms = parse_cif(a.cif)
    hall = hall_from_ops(ops)
    sgt = spglib.get_spacegroup_type(hall)
    O = [x for lab, el, x, occ in atoms if el == "O"]
    Hs = [(lab, x, occ) for lab, el, x, occ in atoms if el == "H"]
    st = IceStructure(name=a.name, space_group=sgt.international_short, sg_number=sgt.number, hall_number=hall,
                      cell=tuple(cell), oxygens=[tuple(x) for x in O], reference=os.path.basename(a.cif),
                      oo_range=tuple(a.oo_range))
    P = np.diag(np.array(a.supercell, float))
    g = IceGraph(st, P)
    L = g.lattice
    Pinv = np.linalg.inv(P)
    print(g.summary())

    # --- assign every H/D site image to (bond, donor end)
    sym = spglib.get_symmetry_from_database(hall)
    site = {}
    for lab, x, occ in Hs:
        imgs = []
        for R, t in zip(sym["rotations"], sym["translations"]):
            y = R @ x + t
            for n in itertools.product(*[range(s) for s in a.supercell]):
                imgs.append(((y + np.array(n)) @ Pinv) % 1)
        for hpos in imgs:
            best = None
            for k, b in enumerate(g.bonds):
                ta = g.frac[b.tail]
                he = g.frac[b.head] + b.image
                for s, (don, acc) in ((1, (ta, he)), (-1, (he, ta))):
                    d = hpos - don
                    d -= np.round(d)
                    w = d @ L
                    if np.linalg.norm(w) > 1.25:
                        continue
                    v = (acc - don) @ L
                    ang = np.degrees(np.arccos(np.clip(v @ w / np.linalg.norm(v) / np.linalg.norm(w), -1, 1)))
                    if best is None or ang < best[0]:
                        best = (ang, k, s, don + d)
            if best is None or best[0] > 30:
                raise SystemExit(f"H site {lab} could not be assigned to an H bond")
            key = (best[1], best[2])
            if key in site and site[key][0] != lab:
                raise SystemExit(f"two H sites ({lab}, {site[key][0]}) on the same bond end")
            site[key] = (lab, best[3], occ)
    fixed = {k: s for (k, s), (lab, h, occ) in site.items() if occ > 0.999}
    free = sorted({k for (k, s), (lab, h, occ) in site.items() if occ <= 0.999})
    for k in range(g.nb):
        if k not in fixed and k not in free:
            raise SystemExit(f"bond {k} has no H site in the CIF")
    print(f"{len(fixed)} bonds with fully occupied H sites (fixed), {len(free)} partially occupied (free)")
    if len(free) > 24:
        raise SystemExit("too many free bonds for brute-force enumeration")

    configs = []
    for vals in itertools.product((1, -1), repeat=len(free)):
        b = np.zeros(g.nb, int)
        for k, s in fixed.items():
            b[k] = s
        for k, v in zip(free, vals):
            b[k] = v
        if all((k, b[k]) in site for k in free) and np.all(b @ g.A.astype(int) == 0):
            configs.append(b.astype(np.int8))
    classes = collections.OrderedDict()
    for b in configs:
        imgs = []
        for pm, sg in zip(g.perms, g.signs):
            t = np.empty_like(b)
            t[pm] = b * sg
            imgs.append(t.tobytes())
        classes.setdefault(min(imgs), []).append(b)
    print(f"{len(configs)} ice-rule configurations -> {len(classes)} symmetry-distinct ordered structures")

    partial_labels = sorted({lab for (k, s), (lab, h, occ) in site.items() if occ <= 0.999})
    exp_occ = {lab: occ for lab, x, occ in Hs}
    os.makedirs(os.path.join(a.out, "d12"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "cif"), exist_ok=True)
    rows = []
    for cid, (key, members) in enumerate(classes.items(), start=1):
        b = members[0]
        frac = [g.frac[i] for i in range(g.n)] + [site[(k, b[k])][1] % 1 for k in range(g.nb)]
        z = [8] * g.n + [1] * g.nb
        frac = np.array(frac)
        info = W.symmetry_info(L, frac, z, symprec=1e-3)
        occ = {}
        for lab in partial_labels:
            ends = [(k, s) for (k, s), v in site.items() if v[0] == lab]
            occ[lab] = np.mean([1.0 if b[k] == s else 0.0 for k, s in ends])
        tag = f"config_{cid:05d}"
        title = (f"{a.name} ordered model {cid}: {info['symbol']} (No. {info['number']}), multiplicity "
                 f"{len(members)}, occ " + " ".join(f"{l}={occ[l]:.2f}" for l in partial_labels))
        W.write_cif(os.path.join(a.out, "cif", f"{tag}_P1.cif"), L, frac, np.array(z), tag)
        W.write_cif(os.path.join(a.out, "cif", f"{tag}_sym.cif"), L, frac, np.array(z), tag, info)
        W.write_d12(os.path.join(a.out, "d12", f"{tag}_P1.d12"), title, L, frac, np.array(z), None)
        W.write_d12(os.path.join(a.out, "d12", f"{tag}_sym.d12"), title, L, frac, np.array(z), info)
        Pnet = W.dipole(g, b)
        row = {"id": cid, "multiplicity": len(members), "weight": len(members) / len(configs),
               "space_group": info["symbol"], "sg_number": info["number"], "n_symops": info["n_ops"],
               "atoms_prim_cell": info["n_atoms_prim"], "atoms_asym_unit": len(info["asym"]),
               "Px": round(Pnet[0], 4) + 0.0, "Py": round(Pnet[1], 4) + 0.0, "Pz": round(Pnet[2], 4) + 0.0,
               "|P|": round(float(np.linalg.norm(Pnet)), 4)}
        for lab in partial_labels:
            row[f"occ_{lab}"] = round(occ[lab], 4)
        rows.append(row)
        print(f"  {tag}: {info['symbol']:11s} multiplicity {len(members):3d}  "
              + "  ".join(f"{l} {occ[l]:.2f} (exp {exp_occ[l]:.3f})" for l in partial_labels)
              + f"  |P| {row['|P|']:.2f}")
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
