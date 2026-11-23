# Ice V: H-bond topologies, HSEsol-D3 / POB-TZVP (CRYSTAL 17)

Cell and O positions: Kamb, Prakash & Knobler, Acta Cryst. 22, 706 (1967) (`../Ice_V.cif`, A2/a,
a = 9.22, b = 7.54, c = 10.35 A, beta = 109.2 deg). Enumeration in the primitive cell (14 H2O,
`graph_enumeration/enumerate_ice.py V --cell primitive`): 1330 ice-rule configurations,
350 symmetry-distinct topologies, 315 P1 + 35 Cc. `enumeration/` holds the 35 Cc structures
(each multiplicity 2, 21 atoms in the asymmetric unit, written in the C-centred setting
10.35 x 7.54 x 9.22 A, beta 109.2). The ordered phase ice XIII (P2_1/a) needs the conventional
28-H2O cell and is calculated separately (`crystal_calcs/iceXIII`).

`hsesol_d3/`: same settings as ice Ih / VI (ATOMONLY optimisation in the literature cell, then
FREQCALC + INTENS), SHRINK 4 4, 30 cores per job.

1. `./submit_lanes.sh [NLANES]` - all 35 optimisations (default 6 lanes = 180 cores)
2. after the energy ranking (`analyze_energies.py`): `./submit_freq.sh ID ID ...` for the
   lowest-energy topologies (+ a few higher ones)
