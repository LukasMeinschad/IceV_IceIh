"""Graph invariants (Kuo, Singer et al.) for H-bond configurations.

First order:   I_r  = sum_g s_g(r) b_{g(r)}
Second order:  I_rs = sum_g s_g(r) s_g(s) b_{g(r)} b_{g(s)}

(projection of b_r and b_r b_s onto the totally symmetric representation of
the cell's space group).  They take identical values for symmetry-related
configurations and are the basis functions of the expansion
E = E0 + sum a_r I_r + sum a_rs I_rs + ...   used to fit DFT energies of the
H-bond isomers.  Second-order invariants are generated from bond pairs whose
mid-points are closer than `cutoff`.
"""
import numpy as np


def _min_image(dfrac):
    return dfrac - np.round(dfrac)


def first_order(graph):
    invs, seen = [], set()
    for r in range(graph.nb):
        c = np.zeros(graph.nb, np.int64)
        np.add.at(c, graph.perms[:, r], graph.signs[:, r])
        if not c.any():
            continue
        key = c.tobytes()
        if key in seen or (-c).tobytes() in seen:
            continue
        seen.add(key)
        invs.append({"bonds": (r,), "coef": c})
    return invs


def second_order(graph, cutoff=4.5):
    mids = []
    for b in graph.bonds:
        mids.append(graph.frac[b.tail] + 0.5 * (graph.frac[b.head] + b.image - graph.frac[b.tail]))
    mids = np.array(mids)
    invs, seen = [], set()
    nb = graph.nb
    for r in range(nb):
        for s in range(r + 1, nb):
            d = np.linalg.norm(_min_image(mids[s] - mids[r]) @ graph.lattice)
            if d > cutoff:
                continue
            terms = {}
            for p, sg in zip(graph.perms, graph.signs):
                a, b = p[r], p[s]
                if a > b:
                    a, b = b, a
                terms[(a, b)] = terms.get((a, b), 0) + int(sg[r]) * int(sg[s])
            terms = {k: v for k, v in terms.items() if v}
            if not terms:
                continue
            key = tuple(sorted(terms.items()))
            if key in seen:
                continue
            seen.add(key)
            invs.append({"bonds": (r, s), "distance": d, "terms": terms})
    return invs


def evaluate(B, first, second):
    """Values of all invariants for configurations B (rows of +-1)."""
    cols, names = [], []
    Bf = B.astype(np.int64)
    for k, inv in enumerate(first):
        cols.append(Bf @ inv["coef"])
        names.append(f"I1_{k + 1}(b{inv['bonds'][0] + 1})")
    for k, inv in enumerate(second):
        v = np.zeros(B.shape[0], np.int64)
        for (a, b), c in inv["terms"].items():
            v += c * Bf[:, a] * Bf[:, b]
        cols.append(v)
        r, s = inv["bonds"]
        names.append(f"I2_{k + 1}(b{r + 1},b{s + 1};{inv['distance']:.2f}A)")
    if not cols:
        return np.zeros((B.shape[0], 0), np.int64), []
    return np.stack(cols, axis=1), names
