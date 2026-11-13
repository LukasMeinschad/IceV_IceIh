"""Validation of the enumeration.

    python -m pytest tests        (or: python tests/test_counts.py)

* ice III, 12 H2O cell: 102 distinct configurations (GrEnum reference run)
* ice Ih, 8 H2O orthohexagonal cell: 114 configurations, 16 symmetry-distinct
  (Hirsch & Ojamae, J. Phys. Chem. B 108, 15856 (2004))
* totals and orbit counts are cross-checked against a brute-force depth-first
  enumeration without symmetry
* symmetry-adapted output: expanding the asymmetric unit with the standard
  setting of the detected space group reproduces the full structure.
"""
import os
import sys

import numpy as np
import spglib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from iceenum.structures import STRUCTURES                      # noqa: E402
from iceenum.crystal import IceGraph                            # noqa: E402
from iceenum.enumerate import enumerate_configurations, merge_crystal_equivalent  # noqa: E402
from iceenum import writers as W                                # noqa: E402


def brute_force(graph):
    """All ice-rule configurations by plain DFS (no symmetry)."""
    out = np.zeros(graph.n, int)
    inn = np.zeros(graph.n, int)
    b = np.zeros(graph.nb, np.int8)
    sols = []

    def rec(k):
        if k == graph.nb:
            sols.append(b.copy())
            return
        u, v = graph.bonds[k].tail, graph.bonds[k].head
        for val in (1, -1):
            d, a = (u, v) if val > 0 else (v, u)
            out[d] += 1
            inn[a] += 1
            if out[d] <= 2 and inn[a] <= 2:
                b[k] = val
                rec(k + 1)
            out[d] -= 1
            inn[a] -= 1
    rec(0)
    return np.array(sols)


def n_orbits(graph, configs):
    seen, n = set(), 0
    for c in configs:
        if c.tobytes() in seen:
            continue
        n += 1
        for p, s in zip(graph.perms, graph.signs):
            t = np.empty_like(c)
            t[p] = c * s
            seen.add(t.tobytes())
    return n


def run(phase, cell):
    g = IceGraph(STRUCTURES[phase], STRUCTURES[phase].presets[cell])
    B, m = enumerate_configurations(g, verbose=False)
    B2, m2, _ = merge_crystal_equivalent(g, B, m, verbose=False)
    return g, B, m, B2, m2


def test_ice_III_grenum():
    g, B, m, B2, m2 = run("III", "unit")
    assert len(B) == 102 and len(B2) == 102


def test_ice_Ih_ortho_hirsch_ojamae():
    g, B, m, B2, m2 = run("Ih", "ortho")
    assert m.sum() == 114 and len(B2) == 16


def test_against_brute_force():
    for phase, cell in [("Ih", "hex"), ("Ih", "ortho"), ("III", "unit"), ("V", "primitive")]:
        g, B, m, B2, m2 = run(phase, cell)
        allc = brute_force(g)
        assert len(allc) == m.sum() == m2.sum(), (phase, cell)
        assert n_orbits(g, allc) == len(B), (phase, cell)
        # every enumerated configuration obeys the ice rules
        net = B.astype(int) @ g.A.astype(int)
        assert np.all(net == 0)


def _default_halls():
    d = {}
    for h in range(1, 531):
        n = spglib.get_spacegroup_type(h).number
        d.setdefault(n, h)
    return d


def test_symmetry_adapted_output():
    halls = _default_halls()
    g, B, m, B2, m2 = run("Ih", "ortho")
    for cfg in B2:
        frac, num = W.build_atoms(g, cfg)
        sym = W.symmetry_info(g.lattice, frac, num)
        ops = spglib.get_symmetry_from_database(halls[sym["number"]])
        pos = []
        for i in sym["asym"]:
            for R, t in zip(ops["rotations"], ops["translations"]):
                x = (R @ sym["std_frac"][i] + t) % 1.0
                if not any(np.allclose(((x - y + 0.5) % 1.0) - 0.5, 0, atol=1e-4) for y in pos):
                    pos.append(x)
        assert len(pos) == sym["n_atoms_std"]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
