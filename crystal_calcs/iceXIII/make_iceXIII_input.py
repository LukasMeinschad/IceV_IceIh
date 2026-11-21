#!/usr/bin/env python3
"""Ordered H2O ice XIII input from the D2O structure of Salzmann et al., J. Chem. Phys. 154, 134504 (2021)
(COD 1567273, P2_1/a, 80 K): all O, and of every D pair the majority site (occupancy > 0.5, 14 of 28);
D is written as H (H2O calculation; masses only matter in FREQCALC, CRYSTAL uses 1.008 for element 1).
Transformed to the standard setting P 1 21/c 1 (CRYSTAL space group 14) with spglib.

    python make_iceXIII_input.py 1567273.cif > geometry.txt
"""
import re
import sys

import numpy as np
import spglib


def num(s):
    return float(re.sub(r"\(.*?\)", "", s))


txt = open(sys.argv[1]).read()
cell = [num(re.search(rf"_cell_{k}\s+(\S+)", txt).group(1)) for k in
        ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")]
ops = re.findall(r"^'([^']+)'\s*$", txt.split("_space_group_symop_operation_xyz")[1].split("loop_")[0], re.M)
sites = []
for l in txt.splitlines():
    p = l.split()
    if len(p) >= 8 and re.match(r"^[OD]\(\d+\)$", p[0]):
        if p[-1] == "O" or num(p[4]) > 0.5:
            sites.append((8 if p[-1] == "O" else 1, np.array([num(x) for x in p[1:4]]), p[0], num(p[4])))

a, b, c, al, be, ga = cell
be = np.radians(be)
L = np.array([[a, 0, 0], [0, b, 0], [c * np.cos(be), 0, c * np.sin(be)]])
pos, z, lab = [], [], []
for Z, f, name, occ in sites:
    for op in ops:
        x, y, zz = f
        q = np.array([eval(e, {"x": x, "y": y, "z": zz}) for e in op.replace(" ", "").split(",")]) % 1
        if not any(Z == z[i] and np.linalg.norm(((q - pos[i] + .5) % 1 - .5) @ L) < 0.05 for i in range(len(pos))):
            pos.append(q), z.append(Z), lab.append(name)
pos, z = np.array(pos), np.array(z)
nO, nH = (z == 8).sum(), (z == 1).sum()
# ice rules: every O has exactly 2 H within 1.2 A, every H exactly 1 O
cart = lambda d: ((d + .5) % 1 - .5) @ L
nh = [sum(np.linalg.norm(cart(pos[j] - pos[i])) < 1.2 for j in np.where(z == 1)[0]) for i in np.where(z == 8)[0]]
oh = [np.linalg.norm(cart(pos[j] - pos[i])) for i in np.where(z == 8)[0] for j in np.where(z == 1)[0]
      if np.linalg.norm(cart(pos[j] - pos[i])) < 1.2]
sys.stderr.write(f"{nO} O, {nH} H; H per O: {sorted(set(nh))}; O-H {min(oh):.3f}-{max(oh):.3f} A; "
                 f"majority D occupancies {min(s[3] for s in sites if s[0] == 1):.3f}-"
                 f"{max(s[3] for s in sites if s[0] == 1):.3f}\n")
assert nO == 28 and nH == 56 and set(nh) == {2}
ds = spglib.get_symmetry_dataset((L, pos, z), symprec=1e-3, hall_number=81)   # P 1 21/c 1
sys.stderr.write(f"spglib: {ds.international} (No. {ds.number}), Hall {ds.hall}\n")
P, t = ds.transformation_matrix, ds.origin_shift
# standardised (non-idealised) lattice: rows of std_lattice from spglib refine without idealisation
Lstd = (L.T @ np.linalg.inv(P)).T
fstd = (pos @ P.T + t) % 1
pa, pb, pc = (np.linalg.norm(v) for v in Lstd)
beta = np.degrees(np.arccos(Lstd[0] @ Lstd[2] / pa / pc))
eq = spglib.get_symmetry_dataset((Lstd, fstd, z), symprec=1e-3, hall_number=81)
assert eq.number == 14 and eq.hall_number == 81
asym = sorted(set(eq.equivalent_atoms), key=lambda i: (-z[i], i))
print(f"{pa:.6f}  {pb:.6f}  {pc:.6f}  {beta:.6f}")
print(len(asym))
for i in asym:
    print(f"{z[i]:<3d} {fstd[i][0]:14.10f} {fstd[i][1]:14.10f} {fstd[i][2]:14.10f}")
sys.stderr.write(f"standard cell P 1 21/c 1: a={pa:.5f} b={pb:.5f} c={pc:.5f} beta={beta:.4f}; "
                 f"{len(asym)} atoms in the asymmetric unit\n")
