#!/usr/bin/env python3
"""Overview figure: Bertie & Whalley (1964) vs. calculated IR (left) and all
individual calculated spectra (right, stacked, sorted by energy).

    python plot_ih_overview.py ../crystal_calcs/iceih/topologies/hsesol_d3 \
        -o ../crystal_calcs/iceih/topologies/hsesol_d3/analysis/overview_bertie_individual.png
"""
import argparse
import csv
import os

import numpy as np

from analyze_spectra import broaden
from plot_atr_vs_bertie import read_bertie, BERTIE, TABLE

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--fwhm", type=float, default=30.0, help="Lorentzian FWHM of the individual spectra")
    p.add_argument("--fwhm-broad", type=float, default=100.0, help="Gaussian FWHM of the 'disorder' average")
    p.add_argument("--scale", type=float, default=1.0, help="scale factor for calculated wavenumbers")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--extra", action="append", default=[], metavar="OUT[:LABEL]",
                   help="(with --single) additional CRYSTAL frequency output to overlay, e.g. a GenIce structure")
    p.add_argument("--title-single", default=None)
    p.add_argument("--single", action="store_true",
                   help="only one panel: Bertie & Whalley vs. the weighted average with --fwhm")
    a = p.parse_args()
    out = a.out or os.path.join(a.jobs, "analysis", "overview_bertie_individual.png")
    an = os.path.join(a.jobs, "analysis")

    jobs = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(a.jobs, "jobs.csv")))}
    en = {int(r["id"]): float(r["dE_kJmol_per_H2O"]) for r in csv.DictReader(open(os.path.join(an, "energies.csv")))}
    modes = {}
    for r in csv.DictReader(open(os.path.join(an, "modes.csv"))):
        f = float(r["freq_cm-1"])
        if r["IR"] == "A" and f > 1.0:
            modes.setdefault(int(r["id"]), []).append((f * a.scale, float(r["intensity_km/mol_H2O"])))
    ids = sorted(modes, key=lambda i: en.get(i, 9))
    grid = np.arange(300.0, 4100.0, 1.0)
    spec = {i: broaden(np.array([m[0] for m in modes[i]]), np.array([m[1] for m in modes[i]]), grid, a.fwhm)
            for i in ids}
    specb = {i: broaden(np.array([m[0] for m in modes[i]]), np.array([m[1] for m in modes[i]]), grid,
                        a.fwhm_broad, "gauss") for i in ids}
    g = {i: int(jobs[i]["multiplicity"]) for i in ids}
    G = sum(g.values())
    avg = sum(g[i] * spec[i] for i in ids) / G
    avgb = sum(g[i] * specb[i] for i in ids) / G

    sgs = []
    for i in ids:
        if jobs[i]["space_group"] not in sgs:
            sgs.append(jobs[i]["space_group"])
    col = {s: COLORS[k % 8] for k, s in enumerate(sgs)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    if a.single:
        single(a, grid, avg, ids, plt, out if a.out else os.path.join(an, f"bertie_vs_calc_fwhm{a.fwhm:g}.png"))
        return
    fig = plt.figure(figsize=(18, 11), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1], wspace=0.12, hspace=0.08)
    axL1 = fig.add_subplot(gs[0, 0])
    axL2 = fig.add_subplot(gs[1, 0], sharex=axL1)
    axR = fig.add_subplot(gs[:, 1])

    # ---------------- left: Bertie vs calculated average ------------------
    segs = read_bertie(BERTIE)
    bx = np.concatenate([[q[0] for q in s] for s in segs])
    by = np.concatenate([[q[1] for q in s] for s in segs])
    bmin, bmax = by.min(), by[(bx > 2900) & (bx < 3700)].max()
    table = [l.split(",") for l in open(TABLE) if l[0].isdigit()]

    def bertie(ax, label=True):
        first = True
        for s in segs:
            x = np.array([q[0] for q in s])
            y = (np.array([q[1] for q in s]) - bmin) / (bmax - bmin)
            unc = np.array([q[2] != "" for q in s])
            ax.plot(np.where(~unc, x, np.nan), np.where(~unc, y, np.nan), color="#eb6834", lw=2.0,
                    label="Bertie & Whalley 1964, H$_2$O ice Ih, ~110 K" if (first and label) else None)
            ax.plot(np.where(unc, x, np.nan), np.where(unc, y, np.nan), color="#eb6834", lw=1.6, ls="--")
            first = False

    def marks(ax):
        for t in table:
            ax.axvline(float(t[0]), color="#9a9a94", ls=":", lw=0.8, zorder=0)

    sc = f" (×{a.scale:g})" if a.scale != 1 else ""
    bertie(axL1)
    axL1.plot(grid, avg / avg[(grid > 2900) & (grid < 3700)].max(), color="#2a78d6", lw=1.6,
              label=f"calculated{sc}, weighted average, FWHM {a.fwhm:g} cm$^{{-1}}$")
    axL1.set_title(f"Ice Ih: Bertie & Whalley (1964) vs. HSEsol-D3 harmonic IR "
                   f"({len(ids)} topologies, multiplicity-weighted)", fontsize=12, loc="left", pad=34)
    bertie(axL2, label=False)
    axL2.plot(grid, avgb / avgb[(grid > 2900) & (grid < 3700)].max(), color="#1c5cab", lw=2.0,
              label=f"calculated{sc}, weighted average, Gaussian FWHM {a.fwhm_broad:g} cm$^{{-1}}$ "
                    f"(disorder broadening)")
    axL2.plot([], [], color="#eb6834", lw=2.0, label="Bertie & Whalley 1964")
    for ax in (axL1, axL2):
        marks(ax)
        ax.set_ylabel("normalised absorbance")
        ax.set_ylim(-0.04, 1.15)
        ax.minorticks_on()
        ax.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9.5)
    for t in table:
        if t[3].strip() in ("s", "sh s", "vb w"):
            axL1.text(float(t[0]), 1.01, f"{float(t[0]):.0f}", transform=axL1.get_xaxis_transform(),
                      rotation=90, ha="center", va="bottom", fontsize=8, color="#3d3d3a")
    plt.setp(axL1.get_xticklabels(), visible=False)
    axL2.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axL1.set_xlim(4000, 400)

    # ---------------- right: individual spectra ---------------------------
    step = 0.75 * max(s[(grid > 2900)].max() for s in spec.values())
    n = len(ids)
    for k, i in enumerate(ids):
        off = (n - 1 - k) * step
        sg = jobs[i]["space_group"]
        axR.plot(grid, spec[i] + off, color=col[sg], lw=1.0)
        axR.fill_between(grid, off, spec[i] + off, color=col[sg], alpha=0.12, lw=0)
        lab = f"#{i}  ${sg}$  g={g[i]}  ΔE={en.get(i, float('nan')):.2f}"
        if sg == "Cmc2_1":
            lab += "  (ice XI)"
        axR.text(2650, off + 0.18 * step, lab, fontsize=8.2, color="#2b2b2a", va="bottom")
    offavg = n * step
    axR.plot(grid, avg + offavg, color="black", lw=1.6)
    axR.text(2650, offavg + 0.18 * step, "weighted average", fontsize=8.6, fontweight="bold", va="bottom")
    marks(axR)
    axR.set_xlim(4000, 400)
    axR.set_ylim(-0.3 * step, offavg + 1.5 * step)
    axR.set_yticks([])
    axR.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axR.set_title(f"individual topologies (sorted by ΔE, kJ mol$^{{-1}}$ per H$_2$O; "
                  f"Lorentzian FWHM {a.fwhm:g} cm$^{{-1}}$)", fontsize=11, loc="left", pad=34)
    axR.minorticks_on()
    axR.tick_params(axis="y", which="both", left=False, right=False)
    # scale bar
    axR.plot([560, 560], [offavg + 0.15 * step, offavg + 0.15 * step + 10], color="black", lw=1.5)
    axR.text(590, offavg + 0.15 * step + 5, "10 km mol$^{-1}$ cm per H$_2$O", fontsize=8, va="center", ha="right")
    handles = [plt.Line2D([], [], color=col[s], lw=2.5, label=f"${s}$") for s in sgs]
    axR.legend(handles=handles, title="space group", loc="upper left", frameon=True, edgecolor="black",
               fancybox=False, fontsize=9, title_fontsize=9)
    fig.text(0.01, 0.005, "dotted: band positions from Bertie & Whalley Table I; dashed orange: region marked "
             "uncertain in the paper. g = multiplicity (number of ice-rule configurations of the 16-H$_2$O cell).",
             fontsize=8.5, color="#3d3d3a")
    fig.savefig(out, bbox_inches="tight")
    print("written", out)


def single(a, grid, avg, ids, plt, out):
    segs = read_bertie(BERTIE)
    bx = np.concatenate([[q[0] for q in s] for s in segs])
    by = np.concatenate([[q[1] for q in s] for s in segs])
    bmin, bmax = by.min(), by[(bx > 2900) & (bx < 3700)].max()
    table = [l.split(",") for l in open(TABLE) if l[0].isdigit()]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    first = True
    for s in segs:
        x = np.array([q[0] for q in s])
        y = (np.array([q[1] for q in s]) - bmin) / (bmax - bmin)
        unc = np.array([q[2] != "" for q in s])
        ax.plot(np.where(~unc, x, np.nan), np.where(~unc, y, np.nan), color="#eb6834", lw=2.0,
                label="Bertie & Whalley 1964, H$_2$O ice Ih, ~110 K (digitised)" if first else None)
        ax.plot(np.where(unc, x, np.nan), np.where(unc, y, np.nan), color="#eb6834", lw=1.6, ls="--")
        first = False
    sc = f" (×{a.scale:g})" if a.scale != 1 else ""
    ax.plot(grid, avg / avg[(grid > 2900) & (grid < 3700)].max(), color="#2a78d6", lw=1.7,
            label=f"calculated{sc}, HSEsol-D3 harmonic, {len(ids)} topologies,\n"
                  f"multiplicity-weighted, Lorentzian FWHM {a.fwhm:g} cm$^{{-1}}$")
    from analyze_spectra import parse_freq
    xcol = ["#1baf7a", "#6250d6", "#e87ba4", "#008300"]
    sel = (grid > 2900) & (grid < 3700)

    def centres(y):
        out = {}
        for nm, lo, hi in (("stretch", 2800, 3800), ("bend", 1500, 1800), ("libration", 400, 1300)):
            ss = (grid >= lo) & (grid <= hi)
            out[nm] = ((grid[ss] * y[ss]).sum() / y[ss].sum(), grid[ss][np.argmax(y[ss])])
        return out
    print(f"{'spectrum':42s} {'stretch centre/max':>20} {'bend centre/max':>18} {'libration centre/max':>22}")
    c = centres(avg)
    print(f"{'weighted average (%d topologies)' % len(ids):42s} {c['stretch'][0]:9.0f} /{c['stretch'][1]:6.0f}   "
          f"{c['bend'][0]:7.0f} /{c['bend'][1]:6.0f}   {c['libration'][0]:9.0f} /{c['libration'][1]:6.0f}")
    for k, ex in enumerate(a.extra):
        path, _, lab = ex.partition(":")
        r = parse_freq(path)
        if r is None:
            print(f"{path}: not finished")
            continue
        fr = np.array([mm["freq"] for mm in r["modes"]]) * a.scale
        ii = np.array([mm["intensity"] for mm in r["modes"]]) / r["n_H2O"]
        nim = int((fr < -1).sum())
        ok = fr > 1
        y = broaden(fr[ok], ii[ok], grid, a.fwhm)
        ax.plot(grid, y / y[sel].max(), color=xcol[k % 4], lw=1.4, ls="-",
                label=(lab or os.path.basename(path)) + f", FWHM {a.fwhm:g} cm$^{{-1}}$")
        c = centres(y)
        print(f"{(lab or os.path.basename(path))[:42]:42s} {c['stretch'][0]:9.0f} /{c['stretch'][1]:6.0f}   "
              f"{c['bend'][0]:7.0f} /{c['bend'][1]:6.0f}   {c['libration'][0]:9.0f} /{c['libration'][1]:6.0f}"
              f"   ({r['n_H2O']} H2O, {nim} imaginary modes, lowest {fr[np.abs(fr) > 1].min():.0f} cm-1)")
    for t in table:
        ax.axvline(float(t[0]), color="#9a9a94", ls=":", lw=0.8, zorder=0)
        if t[3].strip() in ("s", "sh s", "vb w"):
            ax.text(float(t[0]), 1.01, f"{float(t[0]):.0f}", transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=8, color="#3d3d3a")
    ax.set_xlim(4000, 400)
    ax.set_ylim(-0.04, 1.15)
    ax.minorticks_on()
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax.set_ylabel("normalised absorbance")
    ax.set_title(a.title_single or "Ice Ih: Bertie & Whalley (1964) vs. calculated IR", fontsize=12, pad=34)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True,
              edgecolor="black", fancybox=False, fontsize=9)
    fig.text(0.01, -0.02, "dotted: band positions from Bertie & Whalley Table I; dashed orange: region marked "
             "uncertain in the paper (mulling agent). All spectra normalised to their O-H stretch maximum.",
             fontsize=8, color="#3d3d3a")
    fig.savefig(out, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
