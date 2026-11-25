#!/usr/bin/env python3
"""Ice Ih and ice VI side by side: Boltzmann-weighted calculated IR spectra (HSEsol-D3, harmonic) of
all topologies against Bertie & Whalley (1964, ice Ih) and Bertie, Labbe & Whalley (1968, ice VI).

top    : experiment, Boltzmann-weighted average, multiplicity-weighted average (= full Pauling
         disorder, infinite temperature) for reference
bottom : contributions w_i * spectrum_i of the individual topologies (they add up to the average)

Weights: population_<T>K of analysis/energies.csv (analyze_energies.py: g_i exp(-dE_i N_cell / kT)).

    python plot_ih_vi_boltzmann.py --ih ../crystal_calcs/iceih/topologies/hsesol_d3 \
        --vi ../crystal_calcs/iceVI/topologies/hsesol_d3 --temp 108
"""
import argparse
import csv
import os

import numpy as np

from analyze_spectra import broaden
from plot_atr_vs_bertie import read_bertie, BERTIE

HERE = os.path.dirname(os.path.abspath(__file__))
VI_REF = os.path.join(HERE, "reference", "bertie_1968_iceVI_fig1_H2O_propylene.csv")
GRID = np.arange(300.0, 4100.0, 1.0)
REG = {"O-H stretch": (2900, 3700), "bend": (1550, 1800), "librations": (450, 1150)}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def load(jobs, fwhm, temp):
    an = os.path.join(jobs, "analysis")
    en = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(an, "energies.csv")))}
    modes = {}
    for r in csv.DictReader(open(os.path.join(an, "modes.csv"))):
        f = float(r["freq_cm-1"])
        if r["IR"] == "A" and f > 1:
            modes.setdefault(int(r["id"]), []).append((f, float(r["intensity_km/mol_H2O"])))
    ids = [i for i in modes if i in en]
    spec = {i: broaden(np.array([a for a, b in modes[i]]), np.array([b for a, b in modes[i]]), GRID, fwhm)
            for i in ids}
    wb = {i: float(en[i][f"population_{temp:g}K"]) for i in ids}
    wb = {i: v / sum(wb.values()) for i, v in wb.items()}       # renormalise over topologies with spectra
    g = {i: int(en[i]["multiplicity"]) for i in ids}
    wm = {i: g[i] / sum(g.values()) for i in ids}
    avgB = sum(wb[i] * spec[i] for i in ids)
    avgM = sum(wm[i] * spec[i] for i in ids)
    return ids, spec, wb, avgB, avgM, en


def norm(x, y):
    s = (x >= 400) & (x <= 4000)
    b = y[s].min()
    return (y - b) / (y[(x >= 2900) & (x <= 3700)].max() - b)


def bmax(x, y, lo, hi):
    s = (x >= lo) & (x <= hi)
    return x[s][np.argmax(y[s])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ih", required=True)
    p.add_argument("--vi", required=True)
    p.add_argument("--temp", type=float, default=108.0, help="temperature of the population column (72 or 108)")
    p.add_argument("--fwhm", type=float, default=30.0)
    p.add_argument("--nshow", type=int, default=6, help="topologies drawn individually in the bottom row")
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(a.vi, "analysis", f"ih_vs_vi_boltzmann_{a.temp:g}K.png")

    ih = load(a.ih, a.fwhm, a.temp)
    vi = load(a.vi, a.fwhm, a.temp)
    segs = read_bertie(BERTIE)
    bix = np.concatenate([[q[0] for q in s] for s in segs])
    biy = np.concatenate([[q[1] for q in s] for s in segs])
    o = np.argsort(bix)
    bix, biy = bix[o], biy[o]
    r = [l.split(",") for l in open(VI_REF) if l[0].isdigit()]
    bvx = np.array([float(u) for u, v in r])
    bvy = np.array([float(v) for u, v in r])
    o = np.argsort(bvx)
    bvx, bvy = bvx[o], bvy[o]
    exp_ih_tab = {"O-H stretch": 3220.0, "bend": 1650.0, "librations": 840.0}   # Bertie & Whalley Table I

    rows = []
    for name, (lo, hi) in REG.items():
        rows.append((name, exp_ih_tab[name], bmax(GRID, ih[3], lo, hi), bmax(bvx, bvy, lo, hi), bmax(GRID, vi[3], lo, hi)))
    print(f"Boltzmann {a.temp:g} K, FWHM {a.fwhm:g} cm-1; band maxima (cm-1)")
    print(f"{'band':12s} {'exp Ih':>7} {'calc Ih':>8} {'exp VI':>7} {'calc VI':>8} | {'shift exp':>9} {'shift calc':>10}")
    for n, ei, ci, ev, cv in rows:
        print(f"{n:12s} {ei:7.0f} {ci:8.0f} {ev:7.0f} {cv:8.0f} | {ev - ei:+9.0f} {cv - ci:+10.0f}")
    for tag, d in (("Ih", ih), ("VI", vi)):
        top = sorted(d[0], key=lambda i: -d[2][i])[:6]
        print(f"  {tag} weights: " + ", ".join(f"{i}:{d[2][i]:.2f}" for i in top))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5), dpi=200, sharex=True,
                           gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.06, "wspace": 0.08})
    panels = [(ih, bix, biy, "ice Ih", "Bertie & Whalley 1964, H$_2$O ice Ih (digitised)", "#eb6834"),
              (vi, bvx, bvy, "ice VI", "Bertie, Labbé & Whalley 1968, H$_2$O ice VI (digitised)", "#c0392b")]
    for j, ((ids, spec, wb, avgB, avgM, en), ex, ey, ph, elab, ecol) in enumerate(panels):
        x = ax[0, j]
        if j == 0:   # Bertie & Whalley: draw the digitised segments separately, uncertain parts dashed
            sb = (bix >= 400) & (bix <= 4000)
            b0, b1 = biy[sb].min(), biy[(bix >= 2900) & (bix <= 3700)].max()
            for k, sg in enumerate(segs):
                sx = np.array([q[0] for q in sg])
                sy = (np.array([q[1] for q in sg]) - b0) / (b1 - b0)
                unc = np.array([q[2] != "" for q in sg])
                x.plot(np.where(~unc, sx, np.nan), np.where(~unc, sy, np.nan), color=ecol, lw=2.0,
                       label=elab if k == 0 else None)
                x.plot(np.where(unc, sx, np.nan), np.where(unc, sy, np.nan), color=ecol, lw=1.5, ls="--")
        else:
            x.plot(ex, norm(ex, ey), color=ecol, lw=2.0, label=elab)
        x.plot(GRID, norm(GRID, avgB), color="black", lw=1.7,
               label=f"calculated, Boltzmann-weighted ({a.temp:g} K), {len(ids)} topologies")
        x.plot(GRID, norm(GRID, avgM), color="#2a78d6", lw=1.1, ls="--",
               label="calculated, multiplicity-weighted (full disorder)")
        for n, ei, ci, ev, cv in rows:
            x.axvline(ei if j == 0 else ev, color=ecol, ls=":", lw=0.8, zorder=0)
        x.set_ylim(-0.04, 1.18)
        x.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=8.8)
        x.set_title(f"({'ab'[j]}) {ph}: experiment vs. HSEsol-D3 harmonic IR, FWHM {a.fwhm:g} cm$^{{-1}}$",
                    fontsize=11, loc="left")
        # contributions
        y = ax[1, j]
        scale = 1.0 / avgB[(GRID >= 2900) & (GRID <= 3700)].max()
        order = sorted(ids, key=lambda i: -wb[i])
        base = np.zeros_like(GRID)
        for k, i in enumerate(order):
            c = wb[i] * spec[i] * scale
            if k < a.nshow:
                y.fill_between(GRID, base, base + c, color=COLORS[k % 8], alpha=0.55, lw=0,
                               label=f"#{i} ${en[i]['space_group']}$  w={wb[i]:.2f}  ΔE={float(en[i]['dE_kJmol_per_H2O']):.3f}")
            else:
                y.fill_between(GRID, base, base + c, color="#b4b2a9", alpha=0.6, lw=0,
                               label=f"{len(order) - a.nshow} others (Σw={sum(wb[q] for q in order[a.nshow:]):.2f})"
                               if k == a.nshow else None)
            base = base + c
        y.plot(GRID, base, color="black", lw=1.0)
        y.set_ylim(0, 1.12)
        y.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=8.3,
                 title="contributions w$_i$·spectrum$_i$ (stacked)", title_fontsize=8.5)
        y.set_xlabel(r"wavenumber (cm$^{-1}$)")
    for x in ax.flat:
        x.set_xlim(4000, 400)
        x.minorticks_on()
    ax[0, 0].set_ylabel("normalised absorbance")
    ax[1, 0].set_ylabel("IR intensity (rel. to the stretch maximum)")
    for x in ax[:, 1]:
        x.tick_params(labelleft=False)
    txt = "band maxima (cm$^{-1}$)   exp Ih → VI  |  calc Ih → VI\n" + "\n".join(
        f"{n:12s} {ei:5.0f} → {ev:4.0f} ({ev - ei:+4.0f}) | {ci:5.0f} → {cv:4.0f} ({cv - ci:+4.0f})"
        for n, ei, ci, ev, cv in rows)
    ax[0, 1].text(0.985, 0.55, txt, transform=ax[0, 1].transAxes, fontsize=8, family="monospace", ha="right",
                  va="top", bbox=dict(facecolor="white", edgecolor="#9a9a94", lw=0.6))
    fig.text(0.01, 0.005, "Normalised to the O-H stretch maximum after subtracting the minimum. Dotted: experimental band "
             "maxima (ice Ih from Bertie & Whalley Table I); dashed: regions marked uncertain in the paper. Ice VI: Kamb cell.", fontsize=8, color="#3d3d3a")
    fig.savefig(out, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
