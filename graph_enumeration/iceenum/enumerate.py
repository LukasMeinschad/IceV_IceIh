"""Enumeration of symmetry-distinct H-bond topologies obeying the ice rules.

Algorithm (same strategy as GrEnum, Knight & Singer):

* The bonds are added one at a time in an order that closes vertices early
  (breadth-first over the oxygen graph).  All partial configurations of the
  first k bonds are kept as rows of an int8 array.
* After adding bond k every partial configuration that violates the ice rules
  (more than two donated or two accepted H-bonds at any oxygen) is dropped.
* Whenever the set {b_1..b_k} of assigned bonds is mapped onto itself by a
  nontrivial subgroup H_k of the cell's space group, symmetry-related partial
  configurations are merged: each is replaced by the lexicographically
  smallest member of its H_k orbit and duplicates are removed.  Because
  every h in H_k maps the remaining (unassigned) bonds onto themselves, the
  completions of equivalent partial configurations are equivalent too, so no
  distinct topology is lost.
* At the last bond H = G (full group): what is left is exactly one
  representative per symmetry-distinct topology.  For each one the size of
  its stabiliser is counted, giving its multiplicity |G|/|Stab| (number of
  symmetry-equivalent copies in the cell); the sum of all multiplicities is
  the total number of ice-rule configurations of the cell.
"""
import time
from collections import deque

import numpy as np


def bfs_bond_order(graph):
    """Order bonds so that vertices are completed as early as possible."""
    adj = [[] for _ in range(graph.n)]
    for k, b in enumerate(graph.bonds):
        adj[b.tail].append(k)
        adj[b.head].append(k)
    order, used, seen = [], np.zeros(graph.nb, bool), np.zeros(graph.n, bool)
    for start in range(graph.n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        while q:
            v = q.popleft()
            for k in adj[v]:
                if not used[k]:
                    used[k] = True
                    order.append(k)
                b = graph.bonds[k]
                w = b.head if b.tail == v else b.tail
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
    return order


def _keys(B):
    """Byte strings that order rows of a +-1 matrix lexicographically."""
    packed = np.packbits(B > 0, axis=1)
    return packed.view(f"S{packed.shape[1]}").ravel()


def _canonicalise(B, perms, signs):
    """Replace each row by the lexicographic minimum over its orbit."""
    k = B.shape[1]
    best = B.copy()
    best_key = _keys(best)
    for p, s in zip(perms, signs):
        T = np.empty_like(B)
        T[:, p[:k]] = B * s[:k]
        tk = _keys(T)
        better = tk < best_key
        if better.any():
            best[better] = T[better]
            best_key[better] = tk[better]
    return best, best_key


def enumerate_configurations(graph, verbose=True, max_configs=50_000_000):
    """Return (configs [n_distinct x nb] int8 with +-1, multiplicities)."""
    order = bfs_bond_order(graph)
    graph.reorder_bonds(order)
    nb, nv = graph.nb, graph.n
    perms, signs = graph.perms, graph.signs
    # op g stabilises {0..k-1} iff max(perm[:k]) == k-1
    cummax = np.maximum.accumulate(perms, axis=1)
    tails = np.array([b.tail for b in graph.bonds])
    heads = np.array([b.head for b in graph.bonds])

    B = np.zeros((1, 0), np.int8)
    out = np.zeros((1, nv), np.int8)
    inn = np.zeros((1, nv), np.int8)
    t0 = time.time()
    for k in range(nb):
        M = B.shape[0]
        B = np.concatenate([np.hstack([B, np.ones((M, 1), np.int8)]),
                            np.hstack([B, -np.ones((M, 1), np.int8)])])
        out = np.concatenate([out, out])
        inn = np.concatenate([inn, inn])
        u, v = tails[k], heads[k]
        plus = np.arange(2 * M) < M
        out[plus, u] += 1
        inn[plus, v] += 1
        inn[~plus, u] += 1
        out[~plus, v] += 1
        ok = (out[:, u] <= 2) & (inn[:, u] <= 2) & (out[:, v] <= 2) & (inn[:, v] <= 2)
        B, out, inn = B[ok], out[ok], inn[ok]
        n_before = B.shape[0]

        stab = np.where(cummax[:, k] == k)[0]
        if len(stab) > 1 and B.shape[0] > 1:
            B, keys = _canonicalise(B, perms[stab], signs[stab])
            _, idx = np.unique(keys, return_index=True)
            idx.sort()
            B = B[idx]
            Ak = graph.A[:k + 1].astype(np.int16)
            deg = (Ak != 0).sum(axis=0)
            net = B.astype(np.int16) @ Ak            # out - in
            out = ((deg + net) // 2).astype(np.int8)
            inn = ((deg - net) // 2).astype(np.int8)
        if verbose:
            print(f"  bond {k + 1:4d}/{nb}: {n_before:10d} partial configs"
                  + (f" -> {B.shape[0]:10d} after symmetry (|H|={len(stab)})" if len(stab) > 1 else ""),
                  flush=True)
        if B.shape[0] > max_configs:
            raise MemoryError(f"more than {max_configs} partial configurations")
        if B.shape[0] == 0:
            raise RuntimeError("no configuration satisfies the ice rules")

    assert np.all(out == 2) and np.all(inn == 2)
    # multiplicities: |G| / |stabiliser|
    keys = _keys(B)
    nstab = np.zeros(B.shape[0], int)
    for p, s in zip(perms, signs):
        T = np.empty_like(B)
        T[:, p] = B * s
        nstab += (_keys(T) == keys)
    mult = graph.order // nstab
    if verbose:
        print(f"  done in {time.time() - t0:.1f} s: {B.shape[0]} symmetry-distinct "
              f"configurations, {mult.sum()} ice-rule configurations in total")
    return B, mult


def merge_crystal_equivalent(graph, B, mult, verbose=True):
    """Merge configurations that are the same infinite crystal.

    Operations of the parent space group that do not preserve the supercell
    (e.g. the 6-fold axis of ice Ih for an orthohexagonal cell) still map a
    configuration onto a configuration of the same cell when the
    configuration has extra translational symmetry.  Such pairs describe the
    same crystal structure and are merged here (union-find); multiplicities
    are added.  Returns (B, mult, class_of_each_input_row).
    """
    n = B.shape[0]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if graph.extra_ops:
        from .crystal import IceGraph
        D = graph.covering_cell()
        big = IceGraph(graph.structure, D, with_group=False)
        kmap, smap = graph.map_bonds_from(big)
        # representative big bond of every small bond
        rep = np.full(graph.nb, -1)
        for jb, k in enumerate(kmap):
            if rep[k] < 0:
                rep[k] = jb
        Bbig = B[:, kmap] * smap
        index = {k: i for i, k in enumerate(_keys(B))}
        for R, t in graph.extra_ops:
            DTinv = np.linalg.inv(D).T
            p, s = big._bond_action(np.round(DTinv @ R @ D.T), DTinv @ t)
            T = np.empty_like(Bbig)
            T[:, p] = Bbig * s
            T = T * smap                       # values in small-bond orientation
            Tsmall = T[:, rep]
            ok = np.all(T == Tsmall[:, kmap], axis=1)   # image periodic in the small cell?
            rows = np.where(ok)[0]
            if len(rows) == 0:
                continue
            _, keys = _canonicalise(Tsmall[rows], graph.perms, graph.signs)
            for r, k in zip(rows, keys):
                j = index.get(k)
                if j is None:
                    raise RuntimeError("image of a configuration not found - inconsistent symmetry")
                a, b = find(r), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n)])
    reps, cls = np.unique(roots, return_inverse=True)
    new_mult = np.bincount(cls, weights=mult).astype(int)
    if verbose and len(reps) != n:
        print(f"  {n} -> {len(reps)} configurations after merging those related by "
              f"space-group operations that do not preserve the supercell")
    return B[reps], new_mult, cls
