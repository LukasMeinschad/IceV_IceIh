# Ice XIII (ordered ice V): reference calculation, HSEsol-D3 / POB-TZVP (CRYSTAL 17)

Source: `1567273.cif` (COD 1567273), D2O ice XIII, Salzmann, Rosu-Finsen, Sharif, Radaelli, Finney,
J. Chem. Phys. 154, 134504 (2021): P2_1/a, a = 9.24033, b = 7.46235, c = 10.29166 A,
beta = 109.7675 deg, 80 K, Z = 28. 7 O and 28 D sites; every D pair has one majority site
(occupancy 0.885-0.998).

`make_iceXIII_input.py` builds the fully ordered H2O model: all O plus the 14 majority D sites,
written as H (H2O; CRYSTAL uses mass 1.008 for element 1, so no ISOTOPES needed), checked for the
ice rules (2 H per O, O-H 0.94-1.01 A, H-bond O...O 2.71-2.88 A, zero net dipole), and transformed
to the standard setting P 1 21/c 1 (CRYSTAL space group 14: a and c swapped,
a = 10.29166, b = 7.46235, c = 9.24033 A, beta = 109.7675), 21 atoms in the asymmetric unit.

`hsesol_d3/`: same settings as the ice Ih / VI topologies (ATOMONLY optimisation in the experimental
cell, then FREQCALC + INTENS on the optimised geometry), SHRINK 4 4, 30 cores.

    cd crystal_calcs/iceXIII/hsesol_d3 && ./submit.sh
