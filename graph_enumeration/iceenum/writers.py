"""Atomic structures of H-bond configurations and file writers.

* hydrogens are put on the O...O line at distance d_OH from the donor oxygen
* every configuration is analysed with spglib: its space group (the symmetry
  that survives proton ordering) and the standardised cell / asymmetric unit
  are used to write symmetry-adapted CRYSTAL inputs and CIF files.
"""
import numpy as np
import spglib

from .crystal import cell_parameters

# space groups with two origin choices in the International Tables
TWO_ORIGINS = {48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134,
               137, 138, 141, 142, 201, 203, 222, 224, 227, 228}


def build_atoms(graph, config, d_oh=1.0):
    """Fractional coordinates and atomic numbers (O first, then H)."""
    L = graph.lattice
    frac = [graph.frac[i] for i in range(graph.n)]
    for b, v in zip(graph.bonds, config):
        tail = graph.frac[b.tail]
        head = graph.frac[b.head] + b.image
        donor, acc = (tail, head) if v > 0 else (head, tail)
        d = (acc - donor) @ L
        h = donor @ L + d_oh * d / np.linalg.norm(d)
        frac.append((h @ np.linalg.inv(L)) % 1.0)
    numbers = [8] * graph.n + [1] * graph.nb
    return np.array(frac), np.array(numbers)


def bond_unit_vectors(graph):
    return np.array([graph.bond_vector(k) / np.linalg.norm(graph.bond_vector(k)) for k in range(graph.nb)])


def dipole(graph, config, unit_vecs=None):
    """Sum of b_r * (unit O->O vector): net H-bond polarisation (Cartesian)."""
    v = bond_unit_vectors(graph) if unit_vecs is None else unit_vecs
    return (config[:, None] * v).sum(axis=0)


def symmetry_info(lattice, frac, numbers, symprec=1e-3, full=True):
    """Space group of a structure; with full=True also the standardised cell
    and asymmetric unit needed for writing symmetry-adapted files."""
    ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=symprec)
    info = {"number": ds.number, "symbol": ds.international, "hall": ds.hall,
            "n_ops": len(ds.rotations),
            "n_asym": len(set(ds.equivalent_atoms)),
            "n_atoms_prim": len(set(ds.mapping_to_primitive))}
    if not full:
        return info
    std_L, std_x, std_n = spglib.standardize_cell((lattice, frac, numbers), to_primitive=False,
                                                  no_idealize=False, symprec=symprec)
    ds2 = spglib.get_symmetry_dataset((std_L, std_x, std_n), symprec=symprec)
    asym = sorted(set(ds2.equivalent_atoms), key=lambda i: (-std_n[i], i))
    info.update({
        "std_lattice": std_L,
        "std_frac": std_x,
        "std_numbers": std_n,
        "asym": asym,
        "wyckoffs": [ds2.wyckoffs[i] for i in asym],
        "rotations": ds2.rotations,
        "translations": ds2.translations,
        "n_atoms_std": len(std_n),
    })
    return info


# --------------------------------------------------------------------- CIF
def _symop_xyz(R, t):
    out = []
    for i in range(3):
        s = ""
        for j, ax in enumerate("xyz"):
            if R[i, j] == 1:
                s += "+" + ax
            elif R[i, j] == -1:
                s += "-" + ax
        tt = t[i] % 1.0
        for num, den in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 6), (5, 6)]:
            if abs(tt - num / den) < 1e-4:
                s += f"+{num}/{den}"
                break
        else:
            if abs(tt) > 1e-4 and abs(tt - 1) > 1e-4:
                s += f"+{tt:.6f}"
        out.append(s.lstrip("+"))
    return ",".join(out)


def write_cif(path, lattice, frac, numbers, title, sym=None):
    """P1 CIF if sym is None, otherwise symmetry-adapted CIF (asym. unit)."""
    if sym is not None:
        lattice, frac, numbers = sym["std_lattice"], sym["std_frac"], sym["std_numbers"]
        idx = sym["asym"]
        hm, num = sym["symbol"], sym["number"]
        ops = [_symop_xyz(R, t) for R, t in zip(sym["rotations"], sym["translations"])]
    else:
        idx = range(len(numbers))
        hm, num, ops = "P 1", 1, ["x,y,z"]
    a, b, c, al, be, ga = cell_parameters(lattice)
    el = {8: "O", 1: "H"}
    lines = [f"data_{title}", f"_symmetry_space_group_name_H-M '{hm}'",
             f"_space_group_IT_number {num}",
             f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}", f"_cell_length_c {c:.6f}",
             f"_cell_angle_alpha {al:.6f}", f"_cell_angle_beta {be:.6f}", f"_cell_angle_gamma {ga:.6f}",
             "loop_", "_space_group_symop_operation_xyz"] + [f"'{o}'" for o in ops] + [
             "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    cnt = {}
    for i in idx:
        e = el[int(numbers[i])]
        cnt[e] = cnt.get(e, 0) + 1
        x = np.where(np.abs(frac[i]) < 1e-10, 0.0, frac[i])
        lines.append(f"{e}{cnt[e]} {e} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}")
    open(path, "w").write("\n".join(lines) + "\n")


def write_xyz(path, lattice, frac, numbers, comment=""):
    cart = frac @ lattice
    el = {8: "O", 1: "H"}
    Ls = " ".join(f"{v:.6f}" for v in lattice.ravel())
    lines = [str(len(numbers)), f'Lattice="{Ls}" Properties=species:S:1:pos:R:3 {comment}']
    lines += [f"{el[int(z)]} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}" for z, p in zip(numbers, cart)]
    open(path, "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ CRYSTAL
def _crystal_params(number, lattice):
    a, b, c, al, be, ga = cell_parameters(lattice)
    if number <= 2:
        return [a, b, c, al, be, ga]
    if number <= 15:
        return [a, b, c, be]          # unique axis b (standard setting)
    if number <= 74:
        return [a, b, c]
    if number <= 194:
        return [a, c]                 # tetragonal / trigonal (hex. axes) / hexagonal
    return [a]


def read_d12_template(path):
    """Return the part of a CRYSTAL input that follows the atom list."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().upper() in ("CRYSTAL", "SLAB", "MOLECULE"))
    n = int(lines[i + 4].split()[0])
    return "\n".join(lines[i + 5 + n:]) + "\n"


def write_d12(path, title, lattice, frac, numbers, sym=None, tail=None):
    """CRYSTAL geometry input; space-group adapted if sym is given."""
    if sym is not None:
        number = sym["number"]
        L, x, z = sym["std_lattice"], sym["std_frac"], sym["std_numbers"]
        idx = sym["asym"]
    else:
        number, L, x, z, idx = 1, lattice, frac, numbers, range(len(numbers))
    out = [title, "CRYSTAL", "0 0 0", str(number),
           "  ".join(f"{p:.6f}" for p in _crystal_params(number, L)), str(len(idx))]
    for i in idx:
        f = np.where(np.abs(x[i]) < 1e-10, 0.0, x[i])
        out.append(f"{int(z[i]):<4d}{f[0]:16.10f}{f[1]:16.10f}{f[2]:16.10f}")
    text = "\n".join(out) + "\n"
    if tail:
        text += tail
    open(path, "w").write(text)
