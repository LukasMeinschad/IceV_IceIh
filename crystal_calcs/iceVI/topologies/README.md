# Ice VI H-bond topologies: HSEsol-D3 / POB-TZVP

These are all **45 symmetry-distinct proton arrangements** of the 10-H2O unit
cell of ice VI (576 ice-rule configurations in total). They are built on Kamb's
structure (`../ice_vi.cif`, B. Kamb, Science 150, 205 (1965), AMCSD 0015417):

* space group P4_2/nmc (No. 137), origin choice 1, the same setting as the CIF
* a = 6.27 Å, c = 5.79 Å
* O on 8g (0, 0.276, 0.382) and 2a (0, 0, 0)
* two interpenetrating H-bond networks that are not bonded to each other;
  O···O = 2.80–2.81 Å

I generated them with `graph_enumeration/enumerate_ice.py VI` and
`make_crystal_jobs.py`.

| space group | count | ids |
|---|---|---|
| P-1 (ice XV type) | 3 | 7 36 40 |
| P2_1 | 6 | 1 10 13 20 39 41 |
| Cc | 6 | 22 27 33 35 44 45 |
| Pc | 3 | 4 19 24 |
| P1 | 27 | the rest |

The ordered phase of ice VI, ice XV, is P-1 with 10 H2O per cell
(Salzmann et al., PRL 103, 105701 (2009)).

## Layout

The layout is the same as for ice Ih:

```
enumeration/            structure.txt, configs.dat, summary.csv, cif/ xyz/ d12/
hsesol_d3/
  jobs.csv              id, directory, input, space group, #symops, asym. atoms, multiplicity, weight
  submit_all.sh         opt jobs + dependent freq jobs (all or selected ids)
  submit_freq.sh        freq jobs only, for converged optimisations
  config_XXXXX/opt/     vi_cXXXXX.d12 (space group + asym. unit, OPTGEOM ATOMONLY), .sjob
  config_XXXXX/freq/    vi_cXXXXX.d12 (EXTERNAL + FREQCALC INTENS), .sjob
  config_XXXXX/         vi_cXXXXX_sym.cif, vi_cXXXXX_P1.cif, vi_cXXXXX_P1_geometry.d12
```

The settings are copied from `iceVI/disp_cor/opt` and `iceVI/disp_cor/freq`:
HSEsol-D3, POB-TZVP, SHRINK 4 4, TOLINTEG 10 10 10 10 18, FMIXING 70, DIIS. The
optimisation is ATOMONLY with TOLDEG 1e-5, TOLDEX 4e-5, TOLDEE 10 and MAXCYCLE
200, and the frequency run computes IR intensities only (INTENS, Berry phase).
Each job runs on 10 cores on CPU_rune with CRYSTAL 17.

The Cc structures are written in their standard C-centred cell (20 H2O).
CRYSTAL reduces it to the 10-H2O primitive cell.

## Running

```bash
cd crystal_calcs/iceVI/topologies/hsesol_d3
./submit_all.sh               # all 45 (90 jobs)
./submit_all.sh 7 36 40       # e.g. only the P-1 (ice XV type) topologies
```

When the optimisations are done:

```bash
python ../../../../graph_enumeration/analyze_energies.py . --temps 72 108
```

## Cell parameters

Kamb's cell (6.27 / 5.79 Å) is larger than the Kuhs et al. (1984) cell used in
the earlier P1 runs in `iceVI/opt` and `iceVI/disp_cor` (6.181 / 5.698 Å, D2O,
225 K, 1.1 GPa). To use that cell instead, regenerate with
`enumerate_ice.py VI --lattice 6.1811 6.1811 5.6981 90 90 90 ...`.
