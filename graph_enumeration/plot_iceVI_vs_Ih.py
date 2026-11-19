#!/usr/bin/env python3
"""Ice VI vs. ice Ih: calculated IR spectra against Bertie's spectra of both phases.

(a) ice VI: Bertie, Labbe, Whalley (1968), H2O propylene mull, vs. the multiplicity-weighted
    calculated spectrum of the finished ice VI topologies (+ optional extra outputs, e.g. a run
    in another cell)
(b) ice Ih: Bertie & Whalley (1964) vs. the calculated ice Ih spectrum
and a table of band positions / Ih -> VI shifts (experiment vs. calculation).

    python plot_iceVI_vs_Ih.py --vi ../crystal_calcs/iceVI/topologies/hsesol_d3 \
        --ih ../crystal_calcs/iceih/topologies/hsesol_d3 \
        --extra "../crystal_calcs/iceVI/disp_cor/freq/icevi.out:topology 35, Kuhs cell" \
        -o ../crystal_calcs/iceVI/topologies/hsesol_d3/analysis/iceVI_vs_Ih_bertie.png
"""
import argparse
import csv
import os

import numpy as np

from analyze_spectra import broaden, parse_freq
from plot_atr_vs_bertie import read_bertie, BERTIE

HERE = os.path.dirname(os.path.abspath(__file__))
VI_REF = os.path.join(HERE, "reference", "bertie_1968_iceVI_fig1_H2O_propylene.csv")
GRID = np.arange(300.0, 4100.0, 1.0)
REG = {"O-H stretch": (2900, 3700), "bend": (1550, 1800), "librations": (450, 1150)}


def calc_avg(jobs, fwhm):
    mult = {r["id"]: int(r["multiplicity"]) for r in csv.DictReader(open(os.path.join(jobs, "jobs.csv")))}
    modes = {}
    for r in csv.DictReader(open(os.path.join(jobs, "analysis", "modes.csv"))):
        f = float(r["freq_cm-1"])
        if r["IR"] == "A" and f > 1:
            modes.setdefault(r["id"], []).append((f, float(r["intensity_km/mol_H2O"])))
    y = sum(mult[i] * broaden(np.array([a for a, b in v]), np.array([b for a, b in v]), GRID, fwhm)
            for i, v in modes.items()) / sum(mult[i] for i in modes)
    return y, sorted(modes, key=int)


def single(path, fwhm):
    r = parse_freq(path)
    fr = np.array([m["freq"] for m in r["modes"]])
    ii = np.array([m["intensity"] for m in r["modes"]]) / r["n_H2O"]
    ok = fr > 1
    return broaden(fr[ok], ii[ok], GRID, fwhm)


def norm(x, y):
    s = (x >= 400) & (x <= 4000)
    b = y[s].min()
    m = y[(x >= 2900) & (x <= 3700)].max()
    return (y - b) / (m - b)


def band_max(x, y, lo, hi):
    s = (x >= lo) & (x <= hi)
    return x[s][np.argmax(y[s])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vi", required=True)
    p.add_argument("--ih", required=True)
    p.add_argument("--extra", action="append", default=[], metavar="OUT:LABEL")
    p.add_argument("--fwhm", type=float, default=30.0)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()

    yvi, idsvi = calc_avg(a.vi, a.fwhm)
    yih, idsih = calc_avg(a.ih, a.fwhm)
    r = [l.split(",") for l in open(VI_REF) if l[0].isdigit()]
    bvx = np.array([float(u) for u, v in r])
    bvy = np.array([float(v) for u, v in r])
    o = np.argsort(bvx)
    bvx, bvy = bvx[o], bvy[o]
    segs = read_bertie(BERTIE)
    bix = np.concatenate([[q[0] for q in s] for s in segs])
    biy = np.concatenate([[q[1] for q in s] for s in segs])
    o = np.argsort(bix)
    bix, biy = bix[o], biy[o]

    # band positions: ice Ih from Bertie & Whalley Table I (the digitised Ih curve is too flat
    # in the bend region for a reliable maximum), ice VI from the digitised curve
    TABLE_IH = {"O-H stretch": 3220.0, "bend": 1650.0, "librations": 840.0}
    rows = []
    for name, (lo, hi) in REG.items():
        e_ih, e_vi = TABLE_IH[name], band_max(bvx, bvy, lo, hi)
        c_ih, c_vi = band_max(GRID, yih, lo, hi), band_max(GRID, yvi, lo, hi)
        rows.append((name, e_ih, e_vi, c_ih, c_vi))
    print(f"{'band (maximum, cm-1)':22s} {'exp Ih':>7} {'exp VI':>7} {'shift':>6} | {'calc Ih':>7} {'calc VI':>7} {'shift':>6}")
    for name, e_ih, e_vi, c_ih, c_vi in rows:
        print(f"{name:22s} {e_ih:7.0f} {e_vi:7.0f} {e_vi - e_ih:+6.0f} | {c_ih:7.0f} {c_vi:7.0f} {c_vi - c_ih:+6.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=200, sharex=True)
    ax1.plot(bvx, norm(bvx, bvy), color="#c0392b", lw=2.0,
             label="Bertie, Labbé & Whalley 1968, H$_2$O ice VI (propylene mull, digitised)")
    ax1.plot(GRID, norm(GRID, yvi), color="#2a78d6", lw=1.6,
             label=f"calculated ice VI, Kamb cell, {len(idsvi)} topologies ({', '.join(idsvi)}), "
                   f"FWHM {a.fwhm:g} cm$^{{-1}}$")
    cols = ["#1baf7a", "#6250d6", "#eda100"]
    for k, ex in enumerate(a.extra):
        path, _, lab = ex.partition(":")
        y = single(path, a.fwhm)
        ax1.plot(GRID, norm(GRID, y), color=cols[k % 3], lw=1.3, ls="--", label=f"calculated ice VI, {lab}")
    ax1.set_title("Ice VI vs. ice Ih: calculated IR (HSEsol-D3, harmonic) against Bertie's spectra", fontsize=11)
    bmin, bmax = biy.min(), biy[(bix >= 2900) & (bix <= 3700)].max()
    for k, sg in enumerate(segs):
        x = np.array([q[0] for q in sg])
        y = (np.array([q[1] for q in sg]) - bmin) / (bmax - bmin)
        unc = np.array([q[2] != "" for q in sg])
        ax2.plot(np.where(~unc, x, np.nan), np.where(~unc, y, np.nan), color="#eb6834", lw=2.0,
                 label="Bertie & Whalley 1964, H$_2$O ice Ih (digitised)" if k == 0 else None)
        ax2.plot(np.where(unc, x, np.nan), np.where(unc, y, np.nan), color="#eb6834", lw=1.6, ls="--")
    ax2.plot(GRID, norm(GRID, yih), color="#2a78d6", lw=1.6,
             label=f"calculated ice Ih, {len(idsih)} topologies, FWHM {a.fwhm:g} cm$^{{-1}}$")
    for ax, tag in ((ax1, "(a) ice VI"), (ax2, "(b) ice Ih")):
        ax.set_ylabel("normalised absorbance")
        ax.set_ylim(-0.04, 1.15)
        ax.minorticks_on()
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True,
                  edgecolor="black", fancybox=False, fontsize=8.5)
        ax.text(0.01, 0.95, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    for name, e_ih, e_vi, c_ih, c_vi in rows:
        ax1.axvline(e_vi, color="#c0392b", ls=":", lw=0.8, zorder=0)
        ax2.axvline(e_ih, color="#eb6834", ls=":", lw=0.8, zorder=0)
    ax2.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax2.set_xlim(4000, 400)
    txt = "band maxima (cm$^{-1}$)   exp Ih → VI   |   calc Ih → VI\n(exp Ih: Bertie & Whalley Table I)\n" + "\n".join(
        f"{n:12s} {e_ih:5.0f} → {e_vi:5.0f} ({e_vi - e_ih:+4.0f})   |   {c_ih:5.0f} → {c_vi:5.0f} ({c_vi - c_ih:+4.0f})"
        for n, e_ih, e_vi, c_ih, c_vi in rows)
    fig.text(0.70, 0.30, txt, fontsize=8.5, family="monospace", va="top")
    fig.text(0.01, 0.005, "All spectra normalised to their O-H stretch maximum after subtracting their minimum. "
             "Dotted lines: experimental band maxima.", fontsize=8, color="#3d3d3a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
