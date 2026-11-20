#!/usr/bin/env python3
"""H2O vs. D2O: calculated IR spectra of one topology (same geometry, H mass 1.008 vs 2.0141)
against the band maxima of H2O and D2O ice Ih from Bertie & Whalley, J. Chem. Phys. 40, 1637 (1964),
Table I.  Prints band maxima, intensity-weighted band centres and H/D ratios, and plots both spectra.

    python plot_isotope_shift.py ../crystal_calcs/iceih/topologies/hsesol_d3/config_00122/freq/ih_c00122.out \
        ../crystal_calcs/iceih/topologies/hsesol_d3_D2O/config_00122/freq/ih_c00122_d2o.out \
        --label "topology 122 (ice XI, Cmc2_1)" -o ../crystal_calcs/iceih/topologies/hsesol_d3_D2O/analysis/H2O_vs_D2O_122.png
"""
import argparse
import os

import numpy as np

from analyze_spectra import broaden, parse_freq

GRID = np.arange(50.0, 4000.0, 1.0)
# band: (H2O window, D2O window, Bertie H2O, Bertie D2O) - Bertie & Whalley 1964, Table I (~100 K)
BANDS = {
    "O-H / O-D stretch": ((2900, 3700), (2100, 2750), 3220.0, 2425.0),
    "bend nu2": ((1550, 1800), (1150, 1320), 1650.0, 1210.0),
    "librations nu_R": ((450, 1200), (350, 900), 840.0, 640.0),
    "translations": ((50, 400), (50, 400), None, None),
}


def load(path, fwhm):
    r = parse_freq(path)
    f = np.array([m["freq"] for m in r["modes"]])
    i = np.array([m["intensity"] for m in r["modes"]]) / r["n_H2O"]
    ok = f > 1
    return f[ok], i[ok], broaden(f[ok], i[ok], GRID, fwhm)


def band(f, i, y, lo, hi):
    s = (f >= lo) & (f <= hi)
    g = (GRID >= lo) & (GRID <= hi)
    return GRID[g][np.argmax(y[g])], (f[s] * i[s]).sum() / i[s].sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h2o")
    p.add_argument("d2o")
    p.add_argument("--label", default="")
    p.add_argument("--fwhm", type=float, default=30.0)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()

    fh, ih, yh = load(a.h2o, a.fwhm)
    fd, idd, yd = load(a.d2o, a.fwhm)
    rows = []
    print(f"{'band':20s} {'calc H2O':>9} {'calc D2O':>9} {'ratio':>6} | {'exp H2O':>8} {'exp D2O':>8} {'ratio':>6}"
          f"   (maxima at FWHM {a.fwhm:g}; centre = intensity-weighted mean)")
    for name, (wh, wd, eh, ed) in BANDS.items():
        mh, ch = band(fh, ih, yh, *wh)
        md, cd = band(fd, idd, yd, *wd)
        rows.append((name, mh, md, ch, cd, eh, ed))
        e = f"{eh:8.0f} {ed:8.0f} {eh / ed:6.3f}" if eh else f"{'-':>8} {'-':>8} {'-':>6}"
        print(f"{name:20s} {mh:9.0f} {md:9.0f} {mh / md:6.3f} | {e}")
        print(f"{'   centre':20s} {ch:9.0f} {cd:9.0f} {ch / cd:6.3f}")
    # mode by mode (same ordering by frequency; fine as long as bands do not cross)
    rat = fh / fd
    parts = []
    for lab, (lo, hi) in (("stretch", (2900, 3700)), ("bend", (1550, 1800)), ("libration", (450, 1200)),
                          ("translation", (100, 400))):
        r = rat[(fh >= lo) & (fh <= hi)]
        parts.append(f"{lab} {r.min():.3f}-{r.max():.3f}")
    print("mode-by-mode nu(H)/nu(D): " + ", ".join(parts))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
    top = max(yh.max(), yd.max())
    ax.plot(GRID, yh / top, color="#2a78d6", lw=1.6, label="calculated H$_2$O")
    ax.plot(GRID, yd / top, color="#eb6834", lw=1.6, label="calculated D$_2$O")
    for k, (name, mh, md, ch, cd, eh, ed) in enumerate(rows):
        if eh:
            ax.axvline(eh, color="#2a78d6", ls=":", lw=1.0, zorder=0,
                       label="Bertie & Whalley 1964, H$_2$O (Table I)" if k == 0 else None)
            ax.axvline(ed, color="#eb6834", ls=":", lw=1.0, zorder=0,
                       label="Bertie & Whalley 1964, D$_2$O (Table I)" if k == 0 else None)
    ax.set_xlim(4000, 50)
    ax.set_ylim(0, 1.08)
    ax.minorticks_on()
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax.set_ylabel("IR intensity (common normalisation)")
    ax.set_title(f"Ice Ih {a.label}: H$_2$O vs. D$_2$O, HSEsol-D3 harmonic, FWHM {a.fwhm:g} cm$^{{-1}}$", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=True, edgecolor="black",
              fancybox=False, fontsize=8.5)
    txt = "band maxima (cm$^{-1}$)\n          calc H/D (ratio) | exp H/D (ratio)\n" + "\n".join(
        f"{n.split()[0][:12]:12s} {mh:4.0f}/{md:4.0f} ({mh / md:.3f}) | " +
        (f"{eh:4.0f}/{ed:4.0f} ({eh / ed:.3f})" if eh else "   -")
        for n, mh, md, ch, cd, eh, ed in rows)
    ax.text(0.40, 0.95, txt, transform=ax.transAxes, fontsize=8, family="monospace", va="top")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
