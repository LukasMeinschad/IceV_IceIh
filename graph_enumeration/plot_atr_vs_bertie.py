#!/usr/bin/env python3
"""Measured ATR spectrum (and optionally the calculated IR spectrum) against
the digitised Bertie & Whalley (1964) spectrum of H2O ice Ih (~110 K).

All spectra are shown as absorbance normalised to their own O-H stretch
maximum after subtracting their minimum between 400 and 4000 cm-1, so band
positions and shapes can be compared directly (the absolute scales of ATR,
mull transmission and calculated intensities are not comparable).

    python plot_atr_vs_bertie.py --atr ../atr_data/Ih_-165_25_scans_1.csv \
        --calc ../crystal_calcs/iceih/topologies/hsesol_d3/analysis/ir_spectra.csv \
        -o ../crystal_calcs/iceih/topologies/hsesol_d3/analysis/atr_vs_bertie.png
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BERTIE = os.path.join(HERE, "reference", "bertie_whalley_1964_fig1_H2O_curve_a.csv")
TABLE = os.path.join(HERE, "reference", "bertie_whalley_1964_iceIh_H2O.csv")


def read_atr(path):
    x, y = [], []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        parts = line.strip().replace(",", ".").split(";")
        try:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    o = np.argsort(x)
    return np.array(x)[o], np.array(y)[o]


def read_bertie(path):
    rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))
    segs = {}
    for r in rows:
        segs.setdefault(int(r["segment"]), []).append(
            (float(r["wavenumber_cm-1"]), float(r["absorbance"]), r["flag"]))
    return [sorted(v) for _, v in sorted(segs.items())]


def norm(x, y, lo=400, hi=4000, smax=(2900, 3700), base=None):
    sel = (x >= lo) & (x <= hi)
    b = np.min(y[sel]) if base is None else base
    s = (x >= smax[0]) & (x <= smax[1])
    return (y - b) / (np.max(y[s]) - b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atr", required=True)
    p.add_argument("--calc", help="ir_spectra.csv from analyze_spectra.py (column weighted_average is used)")
    p.add_argument("--calc-scale", type=float, default=1.0, help="scale factor for calculated wavenumbers")
    p.add_argument("-o", "--out", default="atr_vs_bertie.png")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xa, ya = read_atr(a.atr)
    segs = read_bertie(BERTIE)
    bx = np.concatenate([[q[0] for q in s] for s in segs])
    by = np.concatenate([[q[1] for q in s] for s in segs])
    bmin = by[(bx >= 400) & (bx <= 4000)].min()
    bmax = by[(bx >= 2900) & (bx <= 3700)].max()
    table = [l.split(",") for l in open(TABLE) if l[0].isdigit()]

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    n = 2 if a.calc else 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.6 * n), dpi=200, sharex=True, squeeze=False)
    axes = axes[:, 0]

    def draw_bertie(ax):
        first = True
        for s in segs:
            x = np.array([q[0] for q in s])
            y = (np.array([q[1] for q in s]) - bmin) / (bmax - bmin)
            unc = np.array([q[2] != "" for q in s])
            ax.plot(np.where(~unc, x, np.nan), np.where(~unc, y, np.nan), color="#eb6834", linewidth=1.6,
                    label="Bertie & Whalley 1964, H$_2$O ice, ~110 K (digitised)" if first else None)
            ax.plot(np.where(unc, x, np.nan), np.where(unc, y, np.nan), color="#eb6834", linewidth=1.4,
                    linestyle="--")
            first = False

    def marks(ax):
        for t in table:
            ax.axvline(float(t[0]), color="#9a9a94", linestyle=":", linewidth=0.8, zorder=0)

    ax = axes[0]
    draw_bertie(ax)
    sel = (xa >= 400) & (xa <= 4000)
    ax.plot(xa[sel], norm(xa, ya)[sel], color="black", linewidth=1.2,
            label=f"ATR, {os.path.basename(a.atr)}")
    marks(ax)
    ax.set_ylabel("normalised absorbance")
    ax.set_title("Ice Ih: measured ATR vs. Bertie & Whalley (1964) transmission spectrum", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True,
              edgecolor="black", fancybox=False, fontsize=8)

    if a.calc:
        rows = list(csv.reader(open(a.calc)))
        hdr = rows[0]
        k = hdr.index("weighted_average")
        d = np.array([[float(r[0]), float(r[k])] for r in rows[1:]])
        xc, yc = d[:, 0] * a.calc_scale, d[:, 1]
        ax = axes[1]
        draw_bertie(ax)
        selc = (xc >= 400) & (xc <= 4000)
        ax.plot(xc[selc], norm(xc, yc, base=0.0)[selc], color="#2a78d6", linewidth=1.2,
                label="calculated, HSEsol-D3 harmonic"
                      + (f" (×{a.calc_scale:g})" if a.calc_scale != 1 else "") + ",\nweighted average of finished topologies")
        marks(ax)
        ax.set_ylabel("normalised absorbance")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True,
                  edgecolor="black", fancybox=False, fontsize=8)
    axes[-1].set_xlabel(r"wavenumber (cm$^{-1}$)")
    for ax in axes:
        ax.set_xlim(4000, 400)
        ax.set_ylim(-0.05, 1.12)
        ax.minorticks_on()
    fig.text(0.01, 0.005, "dotted: band positions from Bertie & Whalley Table I; dashed orange: region marked "
             "uncertain in the paper (mulling agent)", fontsize=7, color="#3d3d3a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
