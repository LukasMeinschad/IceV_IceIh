#!/usr/bin/env python3
"""Relate our ice Ih topologies to the 16 structures of Hirsch & Ojamae (2004).

* classifies every H bond as A (h-cis), B (h-trans), C (c-cis) or D (c-trans)
  (Wong & Whalley / Hirsch & Ojamae notation): A/B are bonds within the
  puckered layers, C/D bonds along c; cis/trans from the dihedral between the
  donor's second O-H and the acceptor's O-H bonds;
* maps their structures 1-16 (8-H2O cell) onto our topology ids via the
  (A,B,C,D) fingerprint + space group;
* compares relative energies and fits their linear B/D model to our energies.

    python compare_hirsch_ojamae.py ../crystal_calcs/iceih/topologies/hsesol_d3 --cell ortho2x1x1
"""
import argparse
import collections
import csv
import os

import numpy as np

from iceenum.structures import STRUCTURES
from iceenum.crystal import IceGraph
from iceenum.enumerate import enumerate_configurations, merge_crystal_equivalent
from iceenum import writers as W

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference", "hirsch_ojamae_2004_iceIh.csv")
KCAL = 4.184
SGMAP = {"Pbn2_1": "Pna2_1", "C1c1": "Cc", "P1c1": "Pc", "P12_11": "P2_1"}


def dimer_types(g, cfg):
    L = g.lattice
    inc = collections.defaultdict(list)
    for k, (b, v) in enumerate(zip(g.bonds, cfg)):
        d = (g.frac[b.head] + b.image - g.frac[b.tail]) @ L
        inc[b.tail].append((k, d, v > 0))
        inc[b.head].append((k, -d, v < 0))
    cnt = collections.Counter()
    for k, (b, v) in enumerate(zip(g.bonds, cfg)):
        d = (g.frac[b.head] + b.image - g.frac[b.tail]) @ L
        don, acc, u = (b.tail, b.head, d) if v > 0 else (b.head, b.tail, -d)
        u = u / np.linalg.norm(u)

        def perp(x):
            x = x - (x @ u) * u
            return x / np.linalg.norm(x)
        vd = perp(next(w for kk, w, dn in inc[don] if kk != k and dn))
        angs = [np.degrees(np.arccos(np.clip(vd @ perp(w), -1, 1))) for kk, w, dn in inc[acc] if dn]
        cis = min(angs) < 90 and max(angs) < 150
        along_c = abs(u[2]) > 0.9
        cnt[("C" if cis else "D") if along_c else ("A" if cis else "B")] += 1
    return tuple(cnt[t] for t in "ABCD")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--cell", default="ortho2x1x1")
    a = p.parse_args()
    ref = list(csv.DictReader(l for l in open(REF) if not l.startswith("#")))
    st = STRUCTURES["Ih"]
    g = IceGraph(st, st.presets[a.cell])
    nw = g.n
    B, m = enumerate_configurations(g, verbose=False)
    B, m, _ = merge_crystal_equivalent(g, B, m, verbose=False)
    jobs = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(a.jobs, "jobs.csv")))}
    ef = os.path.join(a.jobs, "analysis", "energies.csv")
    en = {int(r["id"]): float(r["dE_kJmol_per_H2O"]) for r in csv.DictReader(open(ef))} if os.path.exists(ef) else {}

    info, fps = {}, {}
    for cid, cfg in enumerate(B, 1):
        f, n = W.build_atoms(g, cfg)
        info[cid] = W.symmetry_info(g.lattice, f, n, full=False)
        fps[cid] = dimer_types(g, cfg)
    # map the 8-H2O structures
    scale = nw // 8
    mapping = {}
    for r in ref:
        fp8 = tuple(int(r[t]) for t in "ABCD")
        sg = SGMAP.get(r["space_group"], r["space_group"])
        mapping[int(r["no"])] = [cid for cid in fps if info[cid]["n_atoms_prim"] <= 24
                                 and tuple(x // scale for x in fps[cid]) == fp8 and info[cid]["symbol"] == sg]
    out = os.path.join(a.jobs, "analysis")
    os.makedirs(out, exist_ok=True)
    e1 = {k: float(ref[0][k]) for k in ("castep_coo", "castep_coocel", "blyp_coo")}
    with open(os.path.join(out, "hirsch_ojamae_mapping.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["HO_no", "HO_space_group", "our_ids", "calculated", "ours_dE_kJmol",
                    "HO_castep_coo_dE_kJmol", "HO_blyp_coo_dE_kJmol", "HO_castep_coocel_dE_kJmol"])
        print(f"{'HO':>3} {'space group':12} {'our id':>8} {'ours':>7} {'CASTEP coo':>11} {'BLYP coo':>9} {'CASTEP cell':>12}  (kJ/mol per H2O)")
        for r in ref:
            no = int(r["no"])
            ids = mapping[no]
            calc = [i for i in ids if i in en]
            d = [(float(r[k]) - e1[k]) * KCAL for k in ("castep_coo", "blyp_coo", "castep_coocel")]
            ours = en[calc[0]] if len(calc) == 1 else None
            w.writerow([no, r["space_group"], " ".join(map(str, ids)), bool(calc),
                        "" if ours is None else round(ours, 4)] + [round(x, 3) for x in d])
            print(f"{no:3d} {r['space_group']:12} {' '.join(map(str, ids)):>8} "
                  f"{('%.3f' % ours) if ours is not None else '-':>7} {d[0]:11.2f} {d[1]:9.2f} {d[2]:12.2f}")
    pairs = [(en[mapping[int(r['no'])][0]], int(r["no"])) for r in ref
             if len(mapping[int(r["no"])]) == 1 and mapping[int(r["no"])][0] in en]
    if len(pairs) >= 3:
        o = np.array([x[0] for x in pairs])
        for k, nm in (("castep_coo", "CASTEP/PW91 coords only"), ("blyp_coo", "DMol3/BLYP coords only"),
                      ("castep_coocel", "CASTEP/PW91 coords+cell")):
            h = np.array([(float(ref[n - 1][k]) - e1[k]) * KCAL for _, n in pairs])
            print(f"  vs {nm:26s}: r = {np.corrcoef(o, h)[0, 1]:.2f}, ours/theirs = {(o @ h) / (h @ h):.2f}"
                  f"  ({len(pairs)} common structures)")
    # B/D model
    ids = [c for c in jobs if c in en]
    if len(ids) >= 4:
        X = np.array([[1, fps[c][1], fps[c][3]] for c in ids], float)
        y = np.array([en[c] for c in ids])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ c
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\n  B/D model on our {len(ids)} energies: E/H2O = {c[0]:.4f} + {c[1]:.4f} n_B {c[2]:+.4f} n_D "
              f"(kJ/mol, counts per {nw}-H2O cell); r^2 = {r2:.3f}, rms {np.sqrt(((y - pred) ** 2).mean()):.4f}")
        print(f"  -> per dimer: B {c[1] * nw:+.3f} kJ/mol, D {c[2] * nw:+.3f} kJ/mol "
              f"(Hirsch & Ojamae CASTEP optimised: B +0.33, D -0.66; ideal: B +0.31, D -0.17)")
        with open(os.path.join(out, "dimer_types.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "space_group", "A", "B", "C", "D", "dE_kJmol", "BD_model_kJmol"])
            for cid, pr in zip(ids, pred):
                w.writerow([cid, info[cid]["symbol"], *fps[cid], round(en[cid], 4), round(pr, 4)])
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
