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
    p.add_argument("--unit", choices=("kJmol", "meV"), default="kJmol", help="energy unit of the plot")
    p.add_argument("--title", default="Ice Ih", help="phase name for the plot title")
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

    ncell = 16
    st = os.path.join(a.jobs, "..", "enumeration", "structure.txt")
    if os.path.exists(st):
        m = re.search(r"^(\d+) oxygen sites", open(st).read(), re.M)
        if m:
            ncell = int(m.group(1))
    E = np.array([r["E_au_per_H2O"] for r in rows])
    dE = (E - E.min())
    g = np.array([r["multiplicity"] for r in rows], float)
    for r, de in zip(rows, dE):
        r["dE_meV_per_H2O"] = de * HARTREE_MEV
        r["dE_kJmol_per_H2O"] = de * HARTREE_KJMOL
    for T in a.temps:
        # energy of a whole cell configuration (enumeration cell) enters the Boltzmann factor
        x = -(dE * HARTREE_MEV * ncell) / (KB_MEV * T)
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
        plot(rows, os.path.join(out, "energies.png"), a.unit, a.title)
    print(f"written to {out}/")


def _sg_label(sg):
    """Hermann-Mauguin symbol in mathtext, e.g. P2_12_12_1 -> $P2_12_12_1$."""
    return f"${sg}$" if "_" in sg else sg


def plot(rows, path, unit="kJmol", title="Ice Ih"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key, ylab = (("dE_kJmol_per_H2O", r"$\Delta E$ (kJ mol$^{-1}$ per H$_2$O)") if unit == "kJmol"
                 else ("dE_meV_per_H2O", r"$\Delta E$ (meV per H$_2$O)"))
    groups = []
    for r in rows:
        if r["space_group"] not in groups:
            groups.append(r["space_group"])
    style = {sg: (COLORS[i % 8], MARKERS[i % 8]) for i, sg in enumerate(groups)}

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    x = np.arange(1, len(rows) + 1)
    y = np.array([r[key] for r in rows])
    for sg in groups:
        idx = [i for i, r in enumerate(rows) if r["space_group"] == sg]
        c, m = style[sg]
        ax.scatter(x[idx], y[idx], s=50, color=c, marker=m, edgecolors="black",
                   linewidths=0.6, zorder=3, label=_sg_label(sg))
    for i, r in enumerate(rows):
        if r["space_group"] == "Cmc2_1" and title == "Ice Ih":
            ax.annotate("ice XI", (x[i], y[i]), textcoords="offset points",
                        xytext=(8, 2), ha="left", va="bottom", fontsize=9)
    ax.set_xlabel("topology id (ranked by energy)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{title} H-bond topologies, HSEsol-D3/POB-TZVP (literature cell, ATOMONLY)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["id"]) for r in rows], fontsize=8, rotation=90)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.5)
    ax.set_xlim(0.3, len(rows) + 0.7)
    pad = 0.06 * y.max()
    ax.set_ylim(-pad, y.max() + pad)
    ax.legend(title="space group", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=True, edgecolor="black", fancybox=False, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


if __name__ == "__main__":
    main()
