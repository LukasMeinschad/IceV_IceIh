# Ice Ih H-bond topologies: HSEsol-D3 / POB-TZVP

These are all **139 symmetry-distinct proton arrangements** of the 16-H2O
orthohexagonal ice Ih cell (2a x sqrt(3)a x c, the same size as the earlier P1
runs). They are built on the literature structure: P6_3/mmc, a = 4.4973 Å,
c = 7.3240 Å (Röttger et al. 1994, 10 K), O at 4f. I generated them with
`graph_enumeration/enumerate_ice.py` and `graph_enumeration/make_crystal_jobs.py`.

**Jobs were set up only for the 24 topologies whose ordered structure has an
orthorhombic subgroup of P6_3/mmc:** Cmc2_1 (ice XI) 1, Pca2_1 10, Pna2_1 6,
P2_12_12_1 6, Pmc2_1 1 (ids 3 36 40 72 73 78 83 93 96 100 106 108 114 117 118
122 128 130 132 133 135 137 138 139). No proton-ordered structure can keep
P6_3/mmc itself: that symmetry only exists for the H-disordered average. The
other 115 (P1, P2_1, Pc, Cc) are in `enumeration/` and can be added with
`make_crystal_jobs.py --ids ...`.

## Layout

```
enumeration/            structure.txt, configs.dat (bond variables), summary.csv
                        (multiplicity, weight, space group, polarisation, graph invariants),
                        cif/ xyz/ d12/ (P1 and symmetry-adapted geometries)
hsesol_d3/
  jobs.csv              id, directory, input name, space group, #symops, asym. atoms, weight
  iceIh_orthorhombic_topologies.cif
                        all 24 structures in one CIF (one data block each, symmetry-adapted;
                        ids 106 114 118 122 130 135 138 have an 8-H2O unit cell)
  submit_all.sh         submits all (or selected) opt jobs + dependent freq jobs
  config_XXXXX/
    opt/ih_cXXXXX.d12   space group + asymmetric unit, OPTGEOM ATOMONLY (literature cell kept)
    opt/ih_cXXXXX.sjob
    freq/ih_cXXXXX.d12  EXTERNAL (optimised geometry incl. symmetry) + FREQCALC with
                        IR and Raman intensities (INTENS, INTRAMAN, INTCPHF ... END)
    freq/ih_cXXXXX.sjob copies ../opt/ih_cXXXXX.gui (fort.34) and runs CRYSTAL
    ih_cXXXXX_P1_geometry.d12   same structure in P1 (reference only)
    ih_cXXXXX_sym.cif / _P1.cif  CIF of the starting structure (symmetry-adapted / P1)
```

The settings are copied from `iceih/disp_cor/opt` and `iceih/disp_cor/freq`:
HSEsol-D3, POB-TZVP, SHRINK 4 4, TOLINTEG 10 10 10 10 18, FMIXING 70, DIIS.
The ATOMONLY optimisation uses TOLDEG 1e-5, TOLDEX 4e-5, TOLDEE 10 and
MAXCYCLE 200. The SLURM settings are copied from the same sjob (CPU_rune, CRYSTAL 17,
scratch /scratch/lme), but each job uses **10 cores**.

### IR and Raman intensities

The frequency runs compute IR **and** Raman intensities with the analytical
CPHF/CPKS scheme:

```
FREQCALC
INTENS
INTRAMAN
INTCPHF
END
ENDFREQ
```

With INTCPHF the IR intensities also come from CPHF (analytical Born charges),
not from the Berry phase. The CPHF/CPKS step makes the frequency run noticeably
more expensive. Two things to check:

* Test one job first (e.g. id 122, ice XI). Some CRYSTAL versions do not
  support CPKS with screened hybrids such as HSEsol. If the job stops in the
  CPHF step for that reason, keep only `INTENS` (IR, Berry phase) or use a
  global hybrid for the Raman part.
* The intensities are printed in the `.out` file. For spectra with
  experimental conditions you can add `RAMANEXP` (temperature in K, laser
  wavelength in nm) and the `IRSPEC` / `RAMSPEC` blocks inside FREQCALC.

## Running on the cluster

```bash
cd crystal_calcs/iceih/topologies/hsesol_d3
./submit_all.sh             # all 24 orthorhombic topologies (48 jobs)
./submit_all.sh 122 3 73    # or selected ids (see jobs.csv; 122 = ice XI)
```

Each freq job waits for its opt job (`--dependency=afterok`). It then takes the
optimised structure from `opt/ih_cXXXXX.gui`, which `runmpi17` passes to CRYSTAL
as fort.34, so the frequency run keeps the space group.

For Boltzmann/statistical averages, weight each topology by its
`multiplicity` (or `weight`) from jobs.csv.
