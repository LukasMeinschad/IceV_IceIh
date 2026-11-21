#!/usr/bin/env python3
"""Ice VI overview: Bertie, Labbe & Whalley (1968) vs. the calculated IR spectra of the finished
topologies (left: multiplicity-weighted and Boltzmann-weighted averages), and all individual
spectra stacked and sorted by energy (right).

    python plot_vi_overview.py ../crystal_calcs/iceVI/topologies/hsesol_d3
"""
import argparse
import csv
import os

import numpy as np

from analyze_spectra import broaden

HERE = os.path.dirname(os.path.abspath(__file__))
VI_REF = os.path.join(HERE, "reference", "bertie_1968_iceVI_fig1_H2O_propylene.csv")
COLORS = {"P2_1": "#2a78d6", "Pc": "#eb6834", "Cc": "#1baf7a", "P-1": "#6250d6"}
GRID = np.arange(300.0, 4100.0, 1.0)


def nrm(y, x=GRID):
    s = (x >= 400) & (x <= 4000)
    b = y[s].min()
    return (y - b) / (y[(x >= 2900) & (x <= 3700)].max() - b)


def band_max(x, y, lo, hi):
    s = (x >= lo) & (x <= hi)
    return x[s][np.argmax(y[s])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--fwhm", type=float, default=30.0)
    p.add_argument("--temp", type=float, default=108.0, help="temperature of the Boltzmann average [K]")
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()
    an = os.path.join(a.jobs, "analysis")
    out = a.out or os.path.join(an, "overview_bertie_individual.png")

    jobs = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(a.jobs, "jobs.csv")))}
    en = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(an, "energies.csv")))}
    modes = {}
    for r in csv.DictReader(open(os.path.join(an, "modes.csv"))):
        f = float(r["freq_cm-1"])
        if r["IR"] == "A" and f > 1.0:
            modes.setdefault(int(r["id"]), []).append((f, float(r["intensity_km/mol_H2O"])))
    ids = sorted(modes, key=lambda i: float(en[i]["dE_kJmol_per_H2O"]))
    spec = {i: broaden(np.array([m[0] for m in modes[i]]), np.array([m[1] for m in modes[i]]), GRID, a.fwhm)
            for i in ids}
    g = {i: int(jobs[i]["multiplicity"]) for i in ids}
    avg = sum(g[i] * spec[i] for i in ids) / sum(g.values())
    # Boltzmann weights over the finished topologies (energy per cell = n_H2O * dE)
    kT = 8.314462e-3 * a.temp
    bw = {i: g[i] * np.exp(-float(en[i]["dE_kJmol_per_H2O"]) * int(en[i]["n_H2O_prim"]) / kT) for i in ids}
    Z = sum(bw.values())
    bw = {i: v / Z for i, v in bw.items()}
    avgB = sum(bw[i] * spec[i] for i in ids)

    r = [l.split(",") for l in open(VI_REF) if l[0].isdigit()]
    bx = np.array([float(u) for u, v in r])
    by = np.array([float(v) for u, v in r])
    o = np.argsort(bx)
    bx, by = bx[o], by[o]

    print(f"{len(ids)} topologies; band maxima (cm-1):   stretch   bend   librations")
    for lab, x, y in (("Bertie 1968", bx, by), ("multiplicity-weighted", GRID, avg),
                      (f"Boltzmann {a.temp:g} K", GRID, avgB)):
        print(f"  {lab:24s} {band_max(x, y, 2900, 3700):7.0f} {band_max(x, y, 1550, 1800):6.0f} "
              f"{band_max(x, y, 450, 1150):9.0f}")
    print("  Boltzmann weights: " + ", ".join(f"{i}:{bw[i]:.2f}" for i in ids if bw[i] > 0.005))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    fig = plt.figure(figsize=(18, 10), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.12)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    axL.plot(bx, nrm(by, bx), color="#c0392b", lw=2.0,
             label="Bertie, Labbé & Whalley 1968, H$_2$O ice VI (propylene mull, digitised)")
    axL.plot(GRID, nrm(avg), color="#2a78d6", lw=1.7,
             label=f"calculated, {len(ids)} topologies, multiplicity-weighted")
    axL.plot(GRID, nrm(avgB), color="black", lw=1.4, ls="--",
             label=f"calculated, Boltzmann-weighted ({a.temp:g} K, finished topologies)")
    for lo, hi in ((2900, 3700), (1550, 1800), (450, 1150)):
        axL.axvline(band_max(bx, by, lo, hi), color="#c0392b", ls=":", lw=0.8, zorder=0)
    axL.set_xlim(4000, 400)
    axL.set_ylim(-0.04, 1.15)
    axL.minorticks_on()
    axL.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axL.set_ylabel("normalised absorbance")
    axL.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9.5)
    axL.set_title(f"Ice VI (Kamb cell): Bertie et al. (1968) vs. HSEsol-D3 harmonic IR, "
                  f"Lorentzian FWHM {a.fwhm:g} cm$^{{-1}}$", fontsize=12, loc="left")

    step = 0.75 * max(s[GRID > 2900].max() for s in spec.values())
    n = len(ids)
    for k, i in enumerate(ids):
        off = (n - 1 - k) * step
        sg = jobs[i]["space_group"]
        c = COLORS.get(sg, "#888888")
        axR.plot(GRID, spec[i] + off, color=c, lw=1.0)
        axR.fill_between(GRID, off, spec[i] + off, color=c, alpha=0.12, lw=0)
        axR.text(2650, off + 0.18 * step, f"#{i}  ${sg}$  ΔE={float(en[i]['dE_kJmol_per_H2O']):.3f}  "
                 f"w$_B$={bw[i]:.2f}", fontsize=8.5, color="#2b2b2a", va="bottom")
    offavg = n * step
    axR.plot(GRID, avg + offavg, color="black", lw=1.6)
    axR.text(2650, offavg + 0.18 * step, "multiplicity-weighted average", fontsize=8.8, fontweight="bold",
             va="bottom")
    axR.set_xlim(4000, 400)
    axR.set_ylim(-0.3 * step, offavg + 1.5 * step)
    axR.set_yticks([])
    axR.minorticks_on()
    axR.tick_params(axis="y", which="both", left=False, right=False)
    axR.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axR.set_title("individual topologies, sorted by ΔE (kJ mol$^{-1}$ per H$_2$O)", fontsize=11, loc="left")
    sgs = [s for s in COLORS if any(jobs[i]["space_group"] == s for i in ids)]
    axR.legend(handles=[plt.Line2D([], [], color=COLORS[s], lw=2.5, label=f"${s}$") for s in sgs],
               title="space group", loc="upper left", frameon=True, edgecolor="black", fancybox=False,
               fontsize=9, title_fontsize=9)
    fig.text(0.01, 0.005, "All topologies have multiplicity 8, so the multiplicity-weighted average is the plain mean. "
             "w$_B$: Boltzmann weight among the finished topologies. Dotted: experimental band maxima.",
             fontsize=8.5, color="#3d3d3a")
    fig.savefig(out, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
