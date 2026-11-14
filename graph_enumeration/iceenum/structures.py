"""Literature crystal structures (oxygen sublattice only) of the H-disordered ices.

Each entry defines the space group (by spglib Hall number, i.e. a fully
specified setting/origin), the lattice parameters and the Wyckoff positions of
the oxygen atoms of the asymmetric unit.  The hydrogen positions are *not*
given: the H-bond topology is exactly what is enumerated.

Edit / add entries here if you want other lattice parameters (e.g. a
different temperature or your DFT-optimised cell) or another ice phase.
"""
from dataclasses import dataclass, field


@dataclass
class IceStructure:
    name: str
    space_group: str            # Hermann-Mauguin symbol (for information)
    sg_number: int
    hall_number: int            # spglib Hall number -> exact setting & origin
    cell: tuple                 # a, b, c [Angstrom], alpha, beta, gamma [deg]
    oxygens: list               # asymmetric unit, fractional coordinates
    reference: str
    oo_range: tuple = (2.5, 3.13)   # O...O distance window for H-bonds [A]
    notes: str = ""
    presets: dict = field(default_factory=dict)   # named supercell matrices


STRUCTURES = {
    # ------------------------------------------------------------------ Ih
    "Ih": IceStructure(
        name="ice Ih",
        space_group="P6_3/mmc",
        sg_number=194,
        hall_number=488,
        # Rottger et al., Acta Cryst. B50, 644 (1994), H2O, 10 K
        # (a = 4.49728, c = 7.32403 A as tabulated by later ice XI work).
        cell=(4.4973, 4.4973, 7.3240, 90.0, 90.0, 120.0),
        # O on 4f (1/3, 2/3, z); z = 1/16 = 0.0625 is the ideal tetrahedral
        # value, experiment gives z = 0.062-0.0625.
        oxygens=[(1 / 3, 2 / 3, 0.0625)],
        reference=("K. Rottger, A. Endriss, J. Ihringer, S. Doyle, W.F. Kuhs, "
                   "Acta Cryst. B50, 644 (1994); O at 4f (1/3,2/3,z)"),
        notes="4 H2O in the hexagonal cell (Z=4).",
        presets={
            # 4 H2O, primitive hexagonal cell
            "hex": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            # 8 H2O orthohexagonal cell a x sqrt(3)a x c (Hirsch & Ojamae)
            "ortho": [[1, 0, 0], [1, 2, 0], [0, 0, 1]],
            # 16 H2O: 2 x orthohexagonal along a (the cell used in the P1 runs:
            # 7.35 x 7.82 x 9.04 A is c x sqrt(3)a x 2a)
            "ortho2x1x1": [[2, 0, 0], [1, 2, 0], [0, 0, 1]],
            # 12 H2O
            "ortho1x1x2": [[1, 0, 0], [1, 2, 0], [0, 0, 2]],
        },
    ),
    # ------------------------------------------------------------------- V
    "V": IceStructure(
        name="ice V",
        space_group="A2/a",
        sg_number=15,
        hall_number=93,          # A 1 2/a 1 (unique axis b, Kamb setting)
        # B. Kamb, A. Prakash, C. Knobler, Acta Cryst. 22, 706 (1967)
        cell=(9.22, 7.54, 10.35, 90.0, 109.2, 90.0),
        # 4 symmetry-independent O: O1 on 4e, O2-O4 on 8f  (28 H2O / cell).
        # Coordinates: oxygen positions of the P1 input used in the CRYSTAL
        # runs of this repository (Kamb cell scaled by 0.9978) symmetrised
        # to A2/a with spglib and transformed to the A 1 2/a 1 setting.
        oxygens=[
            (0.7500, 0.6846, 0.5000),   # O1 4e
            (0.5987, 0.1294, 0.5072),   # O2 8f
            (0.7318, 0.8414, 0.2440),   # O3 8f
            (0.5368, 0.9372, 0.8404),   # O4 8f
        ],
        reference=("B. Kamb, A. Prakash, C. Knobler, Acta Cryst. 22, 706 (1967); "
                   "see also Lobban, Finney, Kuhs, J. Chem. Phys. 112, 7169 (2000)"),
        notes="28 H2O in the A-centred cell, 14 H2O in the primitive cell.",
        presets={
            "conventional": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            # primitive cell of the A-centred lattice (14 H2O)
            "primitive": [[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]],
        },
    ),
    # ------------------------------------------------------------------ VI
    "VI": IceStructure(
        name="ice VI",
        space_group="P4_2/nmc (origin choice 1)",
        sg_number=137,
        hall_number=420,
        # B. Kamb, Science 150, 205 (1965); CIF: AMCSD 0015417
        # (crystal_calcs/iceVI/ice_vi.cif)
        cell=(6.27, 6.27, 5.79, 90.0, 90.0, 90.0),
        # 10 H2O per cell, two interpenetrating, mutually non-bonded networks
        oxygens=[
            (0.0, 0.276, 0.382),        # O1 8g
            (0.0, 0.0, 0.0),            # O2 2a
        ],
        reference="B. Kamb, Science 150, 205 (1965); AMCSD 0015417",
        notes="10 H2O per cell; the ordered counterpart is ice XV (P-1).",
        presets={
            "unit": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],              # 10 H2O
            "1x1x2": [[1, 0, 0], [0, 1, 0], [0, 0, 2]],             # 20 H2O
            "sqrt2": [[1, 1, 0], [-1, 1, 0], [0, 0, 1]],            # 20 H2O
        },
    ),
    # ---------------------------------------- validation against GrEnum runs
    "III": IceStructure(
        name="ice III (GrEnum test case)",
        space_group="P4_12_12",
        sg_number=92,
        hall_number=369,
        cell=(6.666, 6.666, 6.936, 90.0, 90.0, 90.0),
        # approximate O positions (8b, 4a); the idealised ones of the manual give
        # an ambiguous distance criterion. Only the network topology matters here.
        oxygens=[(0.3807, 0.1924, 0.2955), (0.1101, 0.1101, 0.0)],
        reference="MkInvar/GrEnum manual (Knight, Beck, Singer 2010), ice III example",
        notes="GrEnum finds 102 symmetry-distinct configurations for the 12 H2O cell.",
        presets={"unit": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    ),
}
