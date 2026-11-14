#!/usr/bin/env python3
"""Relative energies of the optimised H-bond topologies.

Reads <jobs>/jobs.csv and every <jobs>/config_XXXXX/opt/<name>.out, takes the
final energy of the converged optimisation (CRYSTAL prints it including the
D3 dispersion), normalises it per H2O using the number of atoms in CRYSTAL's
primitive cell and writes

    <jobs>/analysis/energies.csv   one row per topology, sorted by energy
    <jobs>/analysis/energies.png   relative energy per topology

    python analyze_energies.py ../crystal_calcs/iceih/topologies/hsesol_d3 --temps 72 108
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

HARTREE_MEV = 27211.386246
HARTREE_KJMOL = 2625.499639
KB_MEV = 0.08617333262

# categorical palette (fixed order, validated) + marker per space group as
# secondary encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]


def parse_opt(path):
    txt = open(path, errors="replace").read()
    conv = re.findall(r"OPT END - CONVERGED \* E\(AU\):\s*(-?\d+\.\d+E[+-]\d+)\s+POINTS\s+(\d+)", txt)
    natoms = re.search(r"N\. OF ATOMS PER CELL\s+(\d+)", txt)
    sg = re.search(r"SPACE GROUP \([^)]*\)\s*:\s*(.+)", txt)
    vol = re.findall(r"PRIMITIVE CELL - CENTRING CODE [^ ]+ VOLUME=\s*([\d.]+)", txt)
    d3 = re.findall(r"D3 DISPERSION ENERGY \(AU\)\s+(-?\d+\.\d+E[+-]\d+)", txt)
    return {
        "converged": bool(conv),
        "E_au": float(conv[-1][0]) if conv else None,
        "opt_points": int(conv[-1][1]) if conv else None,
        "natoms": int(natoms.group(1)) if natoms else None,
        "sg_crystal": sg.group(1).strip() if sg else "",
        "volume": float(vol[-1]) if vol else None,
        "E_D3_au": float(d3[-1]) if d3 else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", help="directory with jobs.csv and config_XXXXX/ (from make_crystal_jobs.py)")
    p.add_argument("--temps", type=float, nargs="*", default=[72.0, 108.0],
                   help="temperatures [K] for Boltzmann populations (multiplicity-weighted)")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    jobs = list(csv.DictReader(open(os.path.join(a.jobs, "jobs.csv"))))
    rows = []
    for j in jobs:
        outs = glob.glob(os.path.join(a.jobs, j["directory"], "opt", j["input"] + ".out"))
        if not outs:
            print(f"{j['directory']}: no output")
            continue
        r = parse_opt(outs[0])
        if not r["converged"]:
            print(f"{j['directory']}: optimisation not converged - skipped")
            continue
        nw = r["natoms"] / 3
        rows.append({"id": int(j["id"]), "input": j["input"], "space_group": j["space_group"],
                     "multiplicity": int(j["multiplicity"]), "|P|": float(j["|P|"] or 0),
                     "n_H2O_prim": int(nw), "E_au_cell": r["E_au"], "E_au_per_H2O": r["E_au"] / nw,
                     "E_D3_au_per_H2O": (r["E_D3_au"] / nw) if r["E_D3_au"] is not None else "",
                     "volume_per_H2O_A3": r["volume"] / nw if r["volume"] else "",
                     "opt_points": r["opt_points"], "sg_crystal": r["sg_crystal"]})
    if not rows:
        raise SystemExit("no converged optimisations found")

    E = np.array([r["E_au_per_H2O"] for r in rows])
    dE = (E - E.min())
    g = np.array([r["multiplicity"] for r in rows], float)
    for r, de in zip(rows, dE):
        r["dE_meV_per_H2O"] = de * HARTREE_MEV
        r["dE_kJmol_per_H2O"] = de * HARTREE_KJMOL
    for T in a.temps:
        # energy of a whole 16-H2O cell configuration enters the Boltzmann factor
        x = -(dE * HARTREE_MEV * 16) / (KB_MEV * T)
        w = g * np.exp(x - x.max())
        w /= w.sum()
        for r, wi in zip(rows, w):
            r[f"population_{T:g}K"] = wi
    rows.sort(key=lambda r: r["dE_meV_per_H2O"])

    out = os.path.join(a.jobs, "analysis")
    os.makedirs(out, exist_ok=True)
    cols = ["id", "input", "space_group", "multiplicity", "|P|", "dE_meV_per_H2O", "dE_kJmol_per_H2O"] + \
           [f"population_{T:g}K" for T in a.temps] + \
           ["E_au_per_H2O", "E_au_cell", "n_H2O_prim", "E_D3_au_per_H2O", "volume_per_H2O_A3",
            "opt_points", "sg_crystal"]
    with open(os.path.join(out, "energies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items() if k in cols})

    print(f"{'rank':>4} {'id':>4} {'space group':12} {'mult':>4} {'|P|':>7} {'dE meV/H2O':>11} {'kJ/mol':>8}"
          + "".join(f" {'pop ' + format(T, 'g') + 'K':>9}" for T in a.temps))
    for k, r in enumerate(rows, 1):
        print(f"{k:4d} {r['id']:4d} {r['space_group']:12} {r['multiplicity']:4d} {r['|P|']:7.2f} "
              f"{r['dE_meV_per_H2O']:11.3f} {r['dE_kJmol_per_H2O']:8.4f}"
              + "".join(f" {r[f'population_{T:g}K']:9.3f}" for T in a.temps))

    if not a.no_plot:
        plot(rows, os.path.join(out, "energies.png"))
    print(f"written to {out}/")


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = []
    for r in rows:
        if r["space_group"] not in groups:
            groups.append(r["space_group"])
    # fixed order: by first appearance in energy order
    style = {sg: (COLORS[i % 8], MARKERS[i % 8]) for i, sg in enumerate(groups)}

    ink, muted, grid = "#2b2b2a", "#6b6a64", "#e4e3de"
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(1, len(rows) + 1)
    for sg in groups:
        idx = [i for i, r in enumerate(rows) if r["space_group"] == sg]
        c, m = style[sg]
        ax.scatter(x[idx], [rows[i]["dE_meV_per_H2O"] for i in idx], s=46, color=c, marker=m,
                   edgecolors="#fcfcfb", linewidths=1.5, zorder=3, label=sg)
    for i, r in enumerate(rows):
        if i < 3 or r["space_group"] == "Cmc2_1":
            ax.annotate(f"#{r['id']}", (x[i], r["dE_meV_per_H2O"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=ink)
    ax.set_xlabel("topology, ranked by energy", color=muted, fontsize=9)
    ax.set_ylabel("ΔE (meV / H$_2$O)", color=muted, fontsize=9)
    ax.set_title("Ice Ih H-bond topologies: HSEsol-D3 relative energies (literature cell, ATOMONLY)",
                 color=ink, fontsize=10, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["id"]) for r in rows], fontsize=7, color=muted, rotation=90)
    ax.tick_params(axis="y", labelsize=8, colors=muted)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=len(groups), loc="upper left", labelcolor=ink,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_xlim(0.3, len(rows) + 0.7)
    ax.set_ylim(bottom=min(-0.05 * max(r["dE_meV_per_H2O"] for r in rows), -0.1))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
