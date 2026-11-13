# Ice Ih H-bond topologies: HSEsol-D3 / POB-TZVP

These are all **139 symmetry-distinct proton arrangements** of the 16-H2O
orthohexagonal ice Ih cell (2a x sqrt(3)a x c, the same size as the earlier P1
runs). They are built on the literature structure: P6_3/mmc, a = 4.4973 Å,
c = 7.3240 Å (Röttger et al. 1994, 10 K), O at 4f. I generated them with
`graph_enumeration/enumerate_ice.py` and `graph_enumeration/make_crystal_jobs.py`.

## Layout

```
enumeration/            structure.txt, configs.dat (bond variables), summary.csv
                        (multiplicity, weight, space group, polarisation, graph invariants),
                        cif/ xyz/ d12/ (P1 and symmetry-adapted geometries)
hsesol_d3/
  jobs.csv              id, directory, input name, space group, #symops, asym. atoms, weight
  submit_all.sh         submits all (or selected) opt jobs + dependent freq jobs
  config_XXXXX/
    opt/ih_cXXXXX.d12   space group + asymmetric unit, OPTGEOM ATOMONLY (literature cell kept)
    opt/ih_cXXXXX.sjob
    freq/ih_cXXXXX.d12  EXTERNAL (optimised geometry incl. symmetry) + FREQCALC INTENS
    freq/ih_cXXXXX.sjob copies ../opt/ih_cXXXXX.gui (fort.34) and runs CRYSTAL
    ih_cXXXXX_P1_geometry.d12   same structure in P1 (reference only)
```

The settings are copied from `iceih/disp_cor/opt` and `iceih/disp_cor/freq`:
HSEsol-D3, POB-TZVP, SHRINK 4 4, TOLINTEG 10 10 10 10 18, FMIXING 70, DIIS.
The ATOMONLY optimisation uses TOLDEG 1e-5, TOLDEX 4e-5, TOLDEE 10 and
MAXCYCLE 200. The SLURM settings are copied from the same sjob: CPU_rune,
50 cores, CRYSTAL 17, scratch /scratch/lme.

## Running on the cluster

```bash
cd crystal_calcs/iceih/topologies/hsesol_d3
./submit_all.sh             # all 139 topologies (278 jobs)
./submit_all.sh 3 52 87     # or selected ids (see jobs.csv)
```

Each freq job waits for its opt job (`--dependency=afterok`). It then takes the
optimised structure from `opt/ih_cXXXXX.gui`, which `runmpi17` passes to CRYSTAL
as fort.34, so the frequency run keeps the space group.

Space groups of the topologies: P1 65, P2_1 30, Pc 19, Pca2_1 10,
P2_12_12_1 6, Pna2_1 6, Pmc2_1 1, Cc 1, Cmc2_1 1 (ice XI).
The symmetric ones are much cheaper than P1. To start with those:
`awk -F, 'NR>1 && $6>1 {print $1}' jobs.csv | xargs ./submit_all.sh`.

For Boltzmann/statistical averages, weight each topology by its
`multiplicity` (or `weight`) from jobs.csv.
