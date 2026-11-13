"""Oxygen lattice, H-bond graph and space-group action on the bonds.

This replaces MkInvar: starting from the space group + Wyckoff positions of
the oxygens it builds the (super)cell, finds all H-bonds (O...O within a
distance window, periodic boundary conditions) and represents every symmetry
operation of the (super)cell as a *signed permutation* of the bonds,

    g :  b_r  ->  s_g(r) * b_{p_g(r)} ,

where b_r = +1 if H-bond r points from its first vertex (tail) to its second
vertex (head) and -1 otherwise.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import spglib

TOL = 1e-4


def lattice_vectors(a, b, c, alpha, beta, gamma):
    """Rows are the lattice vectors (a along x, b in the xy plane)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    v1 = [a, 0.0, 0.0]
    v2 = [b * np.cos(ga), b * np.sin(ga), 0.0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    v = np.array([v1, v2, [cx, cy, cz]])
    v[np.abs(v) < 1e-12] = 0.0
    return v


def cell_parameters(L):
    a, b, c = np.linalg.norm(L, axis=1)

    def ang(u, v):
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    return a, b, c, ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1])


def _wrap(x):
    x = np.asarray(x, float) % 1.0
    x[np.abs(x - 1.0) < TOL] = 0.0
    x[np.abs(x) < TOL] = 0.0
    return x


def _find(pos, x):
    """Index of fractional position x in pos (periodic), or -1."""
    d = pos - x
    d -= np.round(d)
    hit = np.where(np.all(np.abs(d) < 1e-3, axis=1))[0]
    return int(hit[0]) if len(hit) else -1


def _unique_positions(xs):
    out = []
    for x in xs:
        x = _wrap(x)
        if not out or _find(np.array(out), x) < 0:
            out.append(x)
    return np.array(out)


@dataclass
class Bond:
    tail: int
    head: int
    image: tuple          # lattice image of the head (in supercell units)
    length: float


class IceGraph:
    """Periodic H-bond network of a (super)cell of an ice structure."""

    def __init__(self, structure, supercell=None, with_group=True):
        self.structure = structure
        P = np.eye(3) if supercell is None else np.array(supercell, float)
        if P.shape == (3,):
            P = np.diag(P)
        if abs(np.linalg.det(P)) < 1e-8:
            raise ValueError("singular supercell matrix")
        self.P = P
        Pinv = np.linalg.inv(P)

        self.parent_lattice = lattice_vectors(*structure.cell)
        self.lattice = P @ self.parent_lattice

        sym = spglib.get_symmetry_from_database(structure.hall_number)
        R_all, t_all = sym["rotations"], sym["translations"]
        self.parent_ops = list(zip(R_all, t_all))
        sgtype = spglib.get_spacegroup_type(structure.hall_number)
        if sgtype.number != structure.sg_number:
            raise ValueError("Hall number does not match space-group number")

        # --- oxygen sites of the conventional (parent) cell -------------
        xs = [R @ np.array(o) + t for o in structure.oxygens for R, t in zip(R_all, t_all)]
        parent_sites = _unique_positions(xs)
        self.n_parent_sites = len(parent_sites)

        # --- sites of the supercell --------------------------------------
        m = int(np.ceil(np.abs(P).sum(axis=0).max())) + 1
        rng = range(-m, m + 1)
        cand = []
        for n in itertools.product(rng, rng, rng):
            for x in parent_sites:
                y = (x + np.array(n)) @ Pinv
                if np.all(y > -TOL) and np.all(y < 1 - TOL):
                    cand.append(y)
        self.frac = _unique_positions(cand)
        expected = self.n_parent_sites * abs(np.linalg.det(P))
        if abs(len(self.frac) - expected) > 1e-6:
            raise ValueError(f"supercell has {len(self.frac)} sites, expected {expected:g}")
        self.n = len(self.frac)
        self.cart = self.frac @ self.lattice

        self._check_min_distance()
        self._build_bonds()

        if not with_group:
            return
        # --- symmetry operations of the supercell as bond permutations ----
        centring = [t for R, t in zip(R_all, t_all) if np.array_equal(R, np.eye(3, dtype=int))]
        trans = [(np.array(c) + np.array(n)) @ Pinv for c in centring for n in itertools.product(rng, rng, rng)]
        trans = _unique_positions(trans)
        self.translations = trans

        ops, seen = [], set()
        self.extra_ops = []        # parent ops (parent basis) that do not preserve the supercell
        PT, PTinv = P.T, Pinv.T
        for R, t in zip(R_all, t_all):
            Rn = PTinv @ R @ PT
            if not np.allclose(Rn, np.round(Rn), atol=1e-6):
                self.extra_ops.append((R, t))
                continue
            Rn = np.round(Rn)
            tn = PTinv @ t
            for tau in trans:
                perm, sign = self._bond_action(Rn, tn + tau)
                key = (perm.tobytes(), sign.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                ops.append((Rn, _wrap(tn + tau), perm, sign))
        self.ops_rot = np.array([o[0] for o in ops])
        self.ops_trans = np.array([o[1] for o in ops])
        self.perms = np.array([o[2] for o in ops])
        self.signs = np.array([o[3] for o in ops])
        self.n_parent_ops = len(R_all)

    # ------------------------------------------------------------------
    def _check_min_distance(self):
        d = self.frac[None, :, :] - self.frac[:, None, :]
        d -= np.round(d)
        r = np.linalg.norm(d @ self.lattice, axis=2) + 10 * np.eye(self.n)
        if r.min() < 1.0:
            raise ValueError("two oxygens closer than 1 A - check coordinates")

    def _build_bonds(self):
        lo, hi = self.structure.oo_range
        rng = range(-2, 3)
        bonds, self._lookup = [], {}
        for n in itertools.product(rng, rng, rng):
            d = (self.frac[None, :, :] + np.array(n) - self.frac[:, None, :]) @ self.lattice
            r = np.linalg.norm(d, axis=2)
            ii, jj = np.where((r >= lo) & (r <= hi))
            for i, j in zip(ii, jj):
                if i < j or (i == j and n > (0, 0, 0)):
                    bonds.append(Bond(int(i), int(j), tuple(n), float(r[i, j])))
        bonds.sort(key=lambda b: (b.tail, b.head, b.image))
        self.bonds = bonds
        self.nb = len(bonds)
        for k, b in enumerate(bonds):
            D = self.frac[b.head] + b.image - self.frac[b.tail]
            self._lookup[(b.tail, b.head, self._dkey(b.tail, b.head, D))] = (k, 1)
            self._lookup[(b.head, b.tail, self._dkey(b.head, b.tail, -D))] = (k, -1)
        deg = np.zeros(self.n, int)
        for b in bonds:
            deg[b.tail] += 1
            deg[b.head] += 1
        if not np.all(deg == 4):
            raise ValueError(f"oxygen coordination numbers {sorted(set(deg))} != 4; "
                             f"adjust oo_range {self.structure.oo_range}")
        # incidence matrix A[r, v] = +1 tail, -1 head
        self.A = np.zeros((self.nb, self.n), np.int8)
        for k, b in enumerate(bonds):
            self.A[k, b.tail] += 1
            self.A[k, b.head] -= 1

    def _dkey(self, i, j, D):
        """Integer lattice image of a bond i->j with fractional vector D."""
        n = np.asarray(D) - (self.frac[j] - self.frac[i])
        if not np.allclose(n, np.round(n), atol=1e-3):
            raise RuntimeError("bond vector inconsistent with the oxygen positions")
        return tuple(int(x) for x in np.round(n))

    def _bond_action(self, R, t, shift=(0, 0, 0)):
        """Signed bond permutation of x -> R x + t applied to the bonds whose
        tail lies in the cell translated by `shift` (supercell units).  For an
        operation that does not preserve the supercell the result depends on
        `shift`; see ``crystal_equivalence_maps``."""
        shift = np.asarray(shift, float)
        img = np.array([_find(self.frac, _wrap(R @ (x + shift) + t)) for x in self.frac])
        if np.any(img < 0) or len(set(img)) != self.n:
            raise RuntimeError("symmetry operation does not map the oxygen lattice onto itself")
        perm = np.empty(self.nb, np.int64)
        sign = np.empty(self.nb, np.int8)
        for k, b in enumerate(self.bonds):
            D = R @ (self.frac[b.head] + b.image - self.frac[b.tail])
            i2, j2 = img[b.tail], img[b.head]
            k2, s = self._lookup[(i2, j2, self._dkey(i2, j2, D))]
            perm[k], sign[k] = k2, s
        return perm, sign

    def covering_cell(self):
        """Supercell (in parent units) that contains this cell and is mapped
        onto itself by every operation of the parent space group."""
        Pinv = np.linalg.inv(self.P)
        n = []
        for i in range(3):
            for k in range(1, 1000):
                if np.allclose(k * Pinv[i], np.round(k * Pinv[i]), atol=1e-6):
                    n.append(k)
                    break
        # axes mixed by the point-group operations must get the same factor
        n = np.array(n)
        mixed = np.zeros((3, 3), bool)
        for R, _ in self.parent_ops:
            mixed |= (R != 0)
        for _ in range(3):
            for i in range(3):
                for j in range(3):
                    if mixed[i, j]:
                        n[i] = n[j] = np.lcm(n[i], n[j])
        return np.diag(n).astype(float)

    def map_bonds_from(self, other):
        """For each bond of `other` (a larger cell of the same crystal) the
        index and sign of the corresponding bond of this cell."""
        T = other.P @ np.linalg.inv(self.P)          # other frac -> self frac
        kmap = np.empty(other.nb, np.int64)
        smap = np.empty(other.nb, np.int8)
        for k, b in enumerate(other.bonds):
            i = _find(self.frac, _wrap(other.frac[b.tail] @ T))
            j = _find(self.frac, _wrap(other.frac[b.head] @ T))
            D = (other.frac[b.head] + b.image - other.frac[b.tail]) @ T
            kmap[k], smap[k] = self._lookup[(i, j, self._dkey(i, j, D))]
        return kmap, smap

    # ------------------------------------------------------------------
    @property
    def order(self):
        return len(self.perms)

    def bond_vector(self, k):
        b = self.bonds[k]
        return (self.frac[b.head] + b.image - self.frac[b.tail]) @ self.lattice

    def reorder_bonds(self, order):
        """Relabel bonds so that new bond i is old bond order[i]."""
        order = np.asarray(order)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        self.bonds = [self.bonds[k] for k in order]
        self.A = self.A[order]
        # g maps old r -> perm[r];  in new labels: new i = inv[old]
        self.perms = inv[self.perms[:, order]]
        self.signs = self.signs[:, order]
        self._lookup = {k: (inv[v[0]], v[1]) for k, v in self._lookup.items()}

    def summary(self):
        s = self.structure
        a, b, c, al, be, ga = cell_parameters(self.lattice)
        return (f"{s.name}: space group {s.space_group} (No. {s.sg_number}), "
                f"parent cell {s.cell}\n"
                f"  simulation cell  a={a:.4f} b={b:.4f} c={c:.4f} "
                f"alpha={al:.2f} beta={be:.2f} gamma={ga:.2f}\n"
                f"  {self.n} H2O, {self.nb} H-bonds, "
                f"{self.order} symmetry operations of the cell act on the bonds"
                + (f"\n  ({len(self.extra_ops)} of the {self.n_parent_ops} space-group operations "
                   f"do not preserve this supercell)" if self.extra_ops else ""))
