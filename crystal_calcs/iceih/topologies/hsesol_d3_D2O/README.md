# Ice Ih, D2O: frequency calculations with deuterium masses

Same 24 topologies and settings as `../hsesol_d3/` (HSEsol-D3, POB-TZVP, SHRINK 4 4,
TOLINTEG 10 10 10 10 18, FREQCALC + INTENS, IR intensities via Berry phase), but every H gets the
mass of D (2.0141 amu, FREQCALC keyword `ISOTOPES`).

* **No new optimisation.** Geometry and force constants do not depend on the isotope
  (Born-Oppenheimer). Each job takes the converged optimised structure of the H2O run
  (`../hsesol_d3/config_XXXXX/opt/ih_cXXXXX.gui`, read with `EXTERNAL`, symmetry included) and
  recomputes the full frequency calculation with D masses.
* The H atom numbers in `ISOTOPES` are taken from the table
  "ATOMS ISOTOPIC MASS (AMU) FOR FREQUENCY CALCULATION" of the H2O frequency output (same geometry,
  same numbering). **Check:** the D2O output prints the same table, now with 2.0141 for every H.
* The same lattice constants as for H2O are used (the D2O cell is only ~0.1 % larger).
* Weights (multiplicities) are the same as for H2O.

Generated with `graph_enumeration/make_isotope_jobs.py`.

## Running

```bash
cd crystal_calcs/iceih/topologies/hsesol_d3_D2O
./submit_lanes.sh        # remaining 21 jobs (3, 36, 122 submitted by hand), 20 cores each, all at once; ./submit_lanes.sh 5 -> at most 5 at a time (100 cores)
```

The jobs are ordered by size: ice XI (#122) first, then the 8-molecule cells, then the 16-molecule
ones. Check the mass table in `config_00122/freq/ih_c00122_d2o.out` once it has started.

Reference for D2O ice Ih: Bertie & Whalley, J. Chem. Phys. 40, 1637 (1964), Table I:
O-D stretch 2545 sh, 2485 sh, 2425 s, 2395-2360 sh, 2332 s; nu2 1210; librations 640 (s), 675/425 sh.
