#!/usr/bin/env python3
"""Enumerate the symmetry-distinct H-bond topologies of ice Ih / ice V cells.

Python re-implementation of the MkInvar + GrEnum workflow (Knight, Beck &
Singer) built on the literature space group and lattice parameters:

    python enumerate_ice.py Ih --cell ortho              # 8 H2O, 16 topologies
    python enumerate_ice.py Ih --cell ortho2x1x1          # 16 H2O (cell of the P1 runs)
    python enumerate_ice.py V                             # 28 H2O, A2/a cell
    python enumerate_ice.py V --cell primitive            # 14 H2O
    python enumerate_ice.py V --write-all --d12-template ../crystal_calcs/iceV/opt/full/iceV.d12

Output directory (default results/<phase>_<cell>/):
    structure.txt     oxygen sites, bond list (tail, head, lattice image)
    configs.dat       one line per distinct topology: bond variables +-1
    summary.csv       id, multiplicity, weight, space group of the ordered
                      structure, net polarisation, graph invariants
    cif/ xyz/ d12/    structures (P1 and space-group adapted) per topology
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iceenum.structures import STRUCTURES                          # noqa: E402
from iceenum.crystal import IceGraph, cell_parameters               # noqa: E402
from iceenum.enumerate import enumerate_configurations, merge_crystal_equivalent  # noqa: E402
from iceenum import invariants as inv                               # noqa: E402
from iceenum import writers as W                                    # noqa: E402

DEFAULT_CELL = {"Ih": "ortho2x1x1", "V": "conventional", "VI": "unit", "III": "unit"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=sorted(STRUCTURES), help="ice phase")
    p.add_argument("--cell", help="named cell preset of the phase (see --list)")
    p.add_argument("--supercell", type=float, nargs="+", metavar="N",
                   help="3 numbers (diagonal) or 9 numbers (rows = new cell vectors in units of "
                        "the conventional cell vectors); overrides --cell")
    p.add_argument("--lattice", type=float, nargs=6, metavar=("a", "b", "c", "al", "be", "ga"),
                   help="override the literature lattice parameters (A, deg)")
    p.add_argument("--scale", type=float, default=1.0, help="scale the lattice constants")
    p.add_argument("--doh", type=float, default=1.0, help="O-H distance for placing H [A]")
    p.add_argument("--no-merge", action="store_true",
                   help="only use the symmetry operations of the simulation cell (GrEnum convention); "
                        "by default configurations related by any operation of the space group are merged")
    p.add_argument("--invariants", type=int, choices=(0, 1, 2), default=2,
                   help="highest order of graph invariants written to summary.csv")
    p.add_argument("--cutoff", type=float, default=4.5,
                   help="bond-midpoint distance cutoff for 2nd-order invariants [A]")
    p.add_argument("--max-write", type=int, default=300,
                   help="write structure files only if there are at most this many topologies")
    p.add_argument("--write-all", action="store_true", help="write structure files for all topologies")
    p.add_argument("--select", type=int, nargs="+", help="write structure files only for these ids")
    p.add_argument("--min-symops", type=int, default=0,
                   help="also write structure files of every topology whose ordered structure has at "
                        "least this many space-group operations (spglib count, conventional cell)")
    p.add_argument("--space-groups", nargs="+", metavar="SG",
                   help="also write files of topologies with these space groups (symbols or numbers)")
    p.add_argument("--formats", default="cif,d12,xyz", help="comma list of cif,d12,xyz")
    p.add_argument("--d12-template", help="existing CRYSTAL .d12: everything after its atom list "
                                          "(OPTGEOM, BASISSET, DFT, SHRINK, ...) is appended")
    p.add_argument("--symprec", type=float, default=1e-3, help="spglib tolerance [A]")
    p.add_argument("--no-symmetry", action="store_true", help="skip the space-group analysis")
    p.add_argument("-o", "--out", help="output directory")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--list", action="store_true", help="list cell presets and exit")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    st = STRUCTURES[a.phase]
    if a.list:
        print(f"{st.name}  {st.space_group}  cell {st.cell}\n  ref: {st.reference}")
        for k, v in st.presets.items():
            print(f"  --cell {k:12s} {v}")
        return
    if a.lattice:
        st.cell = tuple(a.lattice)
    if a.scale != 1.0:
        st.cell = tuple(x * a.scale for x in st.cell[:3]) + tuple(st.cell[3:])

    if a.supercell:
        sc = a.supercell
        if len(sc) == 3:
            P, cname = np.diag(sc), "x".join(f"{x:g}" for x in sc)
        elif len(sc) == 9:
            P, cname = np.array(sc).reshape(3, 3), "custom"
        else:
            sys.exit("--supercell needs 3 or 9 numbers")
    else:
        cname = a.cell or DEFAULT_CELL.get(a.phase, next(iter(st.presets)))
        if cname not in st.presets:
            sys.exit(f"unknown cell '{cname}', choose from {list(st.presets)}")
        P = st.presets[cname]

    out = a.out or os.path.join("results", f"{a.phase}_{cname}")
    os.makedirs(out, exist_ok=True)
    log = not a.quiet

    graph = IceGraph(st, P)
    print(graph.summary())
    print(f"  reference: {st.reference}")
    B, mult = enumerate_configurations(graph, verbose=log)
    n_cell = len(B)
    if not a.no_merge:
        B, mult, _ = merge_crystal_equivalent(graph, B, mult, verbose=log)
    total = int(mult.sum())
    print(f"==> {len(B)} symmetry-distinct H-bond topologies "
          f"({n_cell} under the cell group), {total} ice-rule configurations in the cell; "
          f"Pauling estimate (3/2)^N = {1.5 ** graph.n:.0f}")

    # ---------------------------------------------------------- structure
    with open(os.path.join(out, "structure.txt"), "w") as f:
        f.write(f"# {graph.summary()}\n# reference: {st.reference}\n")
        f.write("# lattice vectors (A, rows)\n")
        for v in graph.lattice:
            f.write(f"{v[0]:14.8f}{v[1]:14.8f}{v[2]:14.8f}\n")
        f.write(f"{graph.n} oxygen sites (fractional)\n")
        for i, x in enumerate(graph.frac):
            f.write(f"{i + 1:5d}{x[0]:14.8f}{x[1]:14.8f}{x[2]:14.8f}\n")
        f.write(f"{graph.nb} H-bonds: id tail head image(a b c) length;  b=+1 means tail -> head\n")
        for k, b in enumerate(graph.bonds):
            f.write(f"{k + 1:5d}{b.tail + 1:5d}{b.head + 1:5d}  {b.image[0]:3d}{b.image[1]:3d}{b.image[2]:3d}"
                    f"{b.length:10.4f}\n")
    np.savetxt(os.path.join(out, "configs.dat"), B, fmt="%2d",
               header=f"{len(B)} topologies x {graph.nb} bond variables (bond order as in structure.txt)")

    # ---------------------------------------------------------- invariants
    first = inv.first_order(graph) if a.invariants >= 1 else []
    second = inv.second_order(graph, a.cutoff) if a.invariants >= 2 else []
    I, inames = inv.evaluate(B, first, second)
    if log and a.invariants:
        print(f"  graph invariants: {len(first)} first order, {len(second)} second order")

    # ---------------------------------------------------------- per config
    if a.select:
        write_ids = set(a.select)
    elif a.write_all or len(B) <= a.max_write:
        write_ids = set(range(1, len(B) + 1))
    else:
        write_ids = set()
        print(f"  {len(B)} > --max-write {a.max_write}: structure files only for --select / "
              f"--min-symops / --space-groups matches (or use --write-all)")
    fmts = set(a.formats.split(","))
    tail = W.read_d12_template(a.d12_template) if a.d12_template else None

    unit_vecs = W.bond_unit_vectors(graph)
    n_written = 0
    rows, sg_count = [], {}
    t0 = time.time()
    for k, (cfg, m) in enumerate(zip(B, mult), start=1):
        frac, num = W.build_atoms(graph, cfg, a.doh)
        P_net = W.dipole(graph, cfg, unit_vecs)
        sym = None
        if not a.no_symmetry:
            sym = W.symmetry_info(graph.lattice, frac, num, a.symprec, full=False)
            sg_count[sym["symbol"]] = sg_count.get(sym["symbol"], 0) + 1
        row = {"id": k, "multiplicity": int(m), "weight": m / total,
               "space_group": sym["symbol"] if sym else "", "sg_number": sym["number"] if sym else "",
               "n_symops": sym["n_ops"] if sym else "", "atoms_prim_cell": sym["n_atoms_prim"] if sym else "",
               "atoms_asym_unit": sym["n_asym"] if sym else "",
               "Px": round(P_net[0], 4) + 0.0, "Py": round(P_net[1], 4) + 0.0, "Pz": round(P_net[2], 4) + 0.0,
               "|P|": round(float(np.linalg.norm(P_net)), 4)}
        for name, val in zip(inames, I[k - 1]):
            row[name] = int(val)
        rows.append(row)

        write = k in write_ids
        if sym and a.min_symops and sym["n_ops"] >= a.min_symops:
            write = True
        if sym and a.space_groups and (sym["symbol"] in a.space_groups or str(sym["number"]) in a.space_groups):
            write = True
        if write:
            if sym:
                sym = W.symmetry_info(graph.lattice, frac, num, a.symprec, full=True)
            for sub in fmts & {"cif", "xyz", "d12"}:
                os.makedirs(os.path.join(out, sub), exist_ok=True)
            n_written += 1
            tag = f"config_{k:05d}"
            title = (f"{st.name} {tag} H-bond topology, {sym['symbol'] if sym else 'P1'} "
                     f"(No. {sym['number'] if sym else 1}), multiplicity {m}")
            if "cif" in fmts:
                W.write_cif(os.path.join(out, "cif", f"{tag}_P1.cif"), graph.lattice, frac, num, tag)
                if sym:
                    W.write_cif(os.path.join(out, "cif", f"{tag}_sym.cif"), graph.lattice, frac, num, tag, sym)
            if "xyz" in fmts:
                W.write_xyz(os.path.join(out, "xyz", f"{tag}.xyz"), graph.lattice, frac, num, title)
            if "d12" in fmts:
                W.write_d12(os.path.join(out, "d12", f"{tag}_P1.d12"), title, graph.lattice, frac, num, None, tail)
                if sym:
                    W.write_d12(os.path.join(out, "d12", f"{tag}_sym.d12"), title, graph.lattice, frac, num,
                                sym, tail)
                    if sym["number"] in W.TWO_ORIGINS:
                        print(f"  warning: {tag} space group {sym['number']} has two origin choices; "
                              f"check the CRYSTAL origin setting (IFSO) of {tag}_sym.d12")
        if log and k % 5000 == 0:
            print(f"  analysed {k}/{len(B)} ({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  structure files written for {n_written} topologies")
    if sg_count:
        print("  space groups of the ordered topologies:")
        for s, c in sorted(sg_count.items(), key=lambda x: -x[1]):
            print(f"    {s:12s} {c}")
    print(f"  output written to {out}/")


if __name__ == "__main__":
    main()
