# H-bond topology enumeration for ice Ih and ice V

A Python version of the **MkInvar + GrEnum** workflow (C. Knight, C. Beck,
S. J. Singer). It enumerates every symmetry-distinct proton (H-bond)
arrangement that obeys the Bernal–Fowler ice rules in a periodic cell. The cell
is built from the **literature space group and lattice parameters**, not from a
P1 cell.

For every topology you get:

* its multiplicity (the number of symmetry-equivalent copies in the cell) and its statistical weight,
* the **space group of the proton-ordered structure** (from spglib),
* CRYSTAL `.d12` inputs, both P1 and **space-group adapted** (asymmetric unit
  in the standard setting). You can run each configuration with its full
  symmetry instead of in P1,
* CIF (P1 and symmetrised) and extended-xyz files,
* the net H-bond polarisation and the first- and second-order graph invariants.
  These are the basis functions for fitting `E = E0 + Σ a_r I_r + Σ a_rs I_rs`
  to the DFT energies.

## Requirements

```
pip install numpy spglib
```

## Usage

```
python enumerate_ice.py Ih --list                    # show cell presets + reference
python enumerate_ice.py Ih --cell ortho              # 8 H2O, a x √3a x c
python enumerate_ice.py Ih                           # 16 H2O (default, same cell as the P1 runs)
python enumerate_ice.py V                            # 28 H2O, A2/a cell (default)
python enumerate_ice.py V --cell primitive           # 14 H2O
```

Useful options:

| option | meaning |
|---|---|
| `--supercell 2 2 1` or 9 numbers | any supercell of the conventional cell |
| `--lattice a b c al be ga`, `--scale s` | other lattice parameters (e.g. your DFT cell or another temperature) |
| `--d12-template file.d12` | append everything after the atom list of an existing CRYSTAL input (OPTGEOM/FREQCALC, basis, DFT, SHRINK …) |
| `--space-groups P2_1/c Cc` | write structure files for topologies with these space groups |
| `--min-symops 4` | write structure files for topologies with ≥ 4 symmetry operations |
| `--select 1 5 17`, `--write-all` | write files for chosen ids / all of them (written by default only if ≤ `--max-write`=300) |
| `--doh 1.0` | O–H distance used to place the protons on the O···O line |
| `--no-merge` | use only the operations of the simulation cell (GrEnum convention, see below) |
| `--invariants 0/1/2`, `--cutoff 4.5` | graph invariants written to `summary.csv` |

Example: the symmetric ice V topologies as ready-to-run CRYSTAL optimisations

```
python enumerate_ice.py V --space-groups P2_1/c Cc P-1 \
       --d12-template ../crystal_calcs/iceV/opt/full/iceV.d12
```

Output goes to `results/<phase>_<cell>/`:

```
structure.txt   lattice, oxygen sites, bond list (tail head image length); b=+1: tail -> head
configs.dat     one row of bond variables (+-1) per distinct topology (like GrEnum's XFile)
summary.csv     id, multiplicity, weight, space group, #ops, atoms in prim. cell / asym. unit,
                net polarisation, graph invariants
d12/  config_XXXXX_P1.d12   config_XXXXX_sym.d12
cif/  config_XXXXX_P1.cif   config_XXXXX_sym.cif
xyz/  config_XXXXX.xyz
```

## Structures (`iceenum/structures.py`)

| | space group | cell (Å, °) | O sites | reference |
|---|---|---|---|---|
| ice Ih | P6₃/mmc (194) | a = 4.4973, c = 7.3240 | 4f (1/3, 2/3, 0.0625) | Röttger et al., Acta Cryst. B50, 644 (1994), H₂O 10 K |
| ice V | A2/a (15, `A 1 2/a 1`) | a = 9.22, b = 7.54, c = 10.35, β = 109.2 | 4e + 3 × 8f | Kamb, Prakash, Knobler, Acta Cryst. 22, 706 (1967) |

* **Ice V oxygen coordinates.** I took them from the oxygen positions of the P1
  input in `crystal_calcs/iceV` (the Kamb cell scaled by 0.9978),
  symmetrised them to C2/c with spglib and transformed them to the A2/a setting.
  If you have the coordinate table of Kamb 1967 or Lobban et al. 2000, enter it
  there. Only the H-bond network matters for the enumeration, so the counts do
  not change.
* **Ice Ih.** z = 1/16 is the ideal tetrahedral value. The lattice constants
  are for 10 K. Use `--lattice` or `--scale` to choose another temperature.
* Adding another phase only needs a new entry: a Hall number (spglib), a cell,
  the oxygen Wyckoff positions and an O···O window (`ice III` is included as a
  test case).

## Algorithm

1. **Lattice and graph (MkInvar).** The space-group operations (spglib
   database) generate all oxygens of the conventional cell. The cell is then
   expanded to the requested supercell. H-bonds are the O···O pairs within
   2.5–3.13 Å under periodic boundary conditions, and every oxygen must be
   4-coordinated. Each operation that preserves the supercell, combined with
   every lattice translation inside it, is stored as a *signed permutation*
   of the bond variables, `b_r → s_g(r) b_{g(r)}`.
2. **Enumeration (GrEnum).** Bonds are added one at a time in breadth-first
   order, so vertices close early. Partial configurations with more than 2
   donors or 2 acceptors at an oxygen are discarded. Whenever a subgroup H_k
   maps the set of assigned bonds onto itself, each partial configuration is
   replaced by the lexicographic minimum of its H_k orbit and duplicates are
   removed. At the last bond H = G, which leaves one canonical representative
   per orbit. The multiplicity is |G| / |stabiliser|.
3. **Merging (optional, on by default).** Some space-group operations do not
   preserve a supercell, for example the 6-fold axis of ice Ih in an
   orthohexagonal cell. They can still map one configuration of the cell onto
   another when that configuration has extra translational symmetry. Such
   configurations are the same crystal, so they are merged. This uses a
   covering cell that the whole group preserves, the same idea as GrEnum's
   covering cells. `--no-merge` gives the counts under the cell group only.
4. **Structures.** H is placed on the O···O line at `d_OH` from the donor.
   spglib then determines the space group and builds the standardised cell and
   asymmetric unit. CRYSTAL gets the minimal lattice-parameter set of the
   crystal system (monoclinic: unique axis b, a b c β).

## Validation (`python tests/test_counts.py`)

| system | distinct | total | check |
|---|---|---|---|
| ice III, 12 H₂O | 102 | 600 | = GrEnum reference run (`GrEnum/IceIII_Files/Log`) |
| ice Ih, 8 H₂O ortho | 16 | 114 | = Hirsch & Ojamäe, J. Phys. Chem. B 108, 15856 (2004) |
| ice Ih, 4 H₂O hex | 2 | 18 | brute force |
| ice V, 14 H₂O primitive | 350 | 1330 | brute force |

All totals and orbit counts are also checked against a brute-force
enumeration without symmetry. Writing the asymmetric unit with the standard
setting of the detected space group reproduces the full structure.

Results for the cells used in this project:

| cell | distinct topologies | ice-rule configs | space groups of the ordered structures |
|---|---|---|---|
| Ih 16 H₂O (`ortho2x1x1`) | 139 (141 under cell group) | 2970 | P1 65, P2₁ 30, Pc 19, Pca2₁ 10, P2₁2₁2₁ 6, Pna2₁ 6, Pmc2₁ 1, Cc 1, **Cmc2₁ 1 (ice XI)** |
| Ih 8 H₂O (`ortho`) | 16 | 114 | P2₁ 4, Pc 3, Pna2₁ 3, P2₁2₁2₁ 2, Cc 1, P1 1, Pca2₁ 1, Cmc2₁ 1 |
| V 28 H₂O (A2/a cell) | 69 380 | 547 612 | P1 67 873, Pc 813, P2₁ 315, P-1 309, **P2₁/c 35 (ice XIII type)**, Cc 35 |

The 28-molecule ice V run takes about 5 s to enumerate and about 2 min for the
spglib analysis of all topologies.

## Notes

* A configuration and its fully reversed counterpart (all b → −b) are not
  related by a space-group operation, so they count as distinct (same as GrEnum).
* For space groups with two origin choices the script prints a warning. Check
  the CRYSTAL `IFSO` flag for those inputs (not expected for ordered Ih / V).
* The protons are placed on straight O···O lines, so the structures need a
  geometry optimisation (`atompos` or `full`) before you compare energies.
