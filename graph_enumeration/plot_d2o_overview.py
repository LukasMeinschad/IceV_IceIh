#!/usr/bin/env python3
"""D2O vs. H2O ice Ih: multiplicity-weighted calculated IR spectra of the topologies finished in both
sets (same geometries, H mass 1.008 vs 2.0141) against the band maxima of Bertie & Whalley,
J. Chem. Phys. 40, 1637 (1964), Table I (H2O and D2O ice Ih, ~100 K).

left : weighted average H2O and D2O, experimental band maxima (dotted)
right: individual D2O spectra, sorted by energy
and a table of band maxima and H/D ratios (calculated average vs. experiment).

    python plot_d2o_overview.py ../crystal_calcs/iceih/topologies/hsesol_d3 ../crystal_calcs/iceih/topologies/hsesol_d3_D2O
"""
import argparse
import csv
import os

import numpy as np

from analyze_spectra import broaden

GRID = np.arange(100.0, 4000.0, 1.0)
# band: (H2O window, D2O window, Bertie H2O, Bertie D2O)
BANDS = {
    "stretch": ((2900, 3700), (2100, 2750), 3220.0, 2425.0),
    "bend": ((1550, 1800), (1150, 1320), 1650.0, 1210.0),
    "librations": ((450, 1200), (350, 950), 840.0, 640.0),
    "translations": ((150, 400), (150, 400), None, None),
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def modes(path):
    out = {}
    for r in csv.DictReader(open(os.path.join(path, "analysis", "modes.csv"))):
        f = float(r["freq_cm-1"])
        if r["IR"] == "A" and f > 1:
            out.setdefault(int(r["id"]), []).append((f, float(r["intensity_km/mol_H2O"])))
    return out


def bmax(y, lo, hi):
    """intensity-weighted band centre (robust for bands with several sub-peaks of similar height)"""
    s = (GRID >= lo) & (GRID <= hi)
    return (GRID[s] * y[s]).sum() / y[s].sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h2o")
    p.add_argument("d2o")
    p.add_argument("--fwhm", type=float, default=30.0)
    p.add_argument("-o", "--out")
    a = p.parse_args()
    out = a.out or os.path.join(a.d2o, "analysis", "D2O_vs_H2O_overview.png")

    mh, md = modes(a.h2o), modes(a.d2o)
    ids = sorted(set(mh) & set(md))
    jobs = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(a.d2o, "jobs.csv")))}
    en = {int(r["id"]): float(r["dE_kJmol_per_H2O"])
          for r in csv.DictReader(open(os.path.join(a.h2o, "analysis", "energies.csv")))}
    ids.sort(key=lambda i: en.get(i, 9))
    g = {i: int(jobs[i]["multiplicity"]) for i in ids}
    sp = lambda m, i: broaden(np.array([x for x, y in m[i]]), np.array([y for x, y in m[i]]), GRID, a.fwhm)
    sh, sd = {i: sp(mh, i) for i in ids}, {i: sp(md, i) for i in ids}
    G = sum(g.values())
    ah = sum(g[i] * sh[i] for i in ids) / G
    ad = sum(g[i] * sd[i] for i in ids) / G

    rows = []
    print(f"{len(ids)} topologies ({', '.join(map(str, ids))}), multiplicity-weighted, FWHM {a.fwhm:g} cm-1")
    print(f"{'band centre':20s} {'calc H2O':>8} {'calc D2O':>8} {'H/D':>6} | {'exp H2O':>7} {'exp D2O':>7} {'H/D':>6} |"
          f" {'calc-exp D2O':>12}   per-topology H/D range")
    for name, (wh, wd, eh, ed) in BANDS.items():
        xh, xd = bmax(ah, *wh), bmax(ad, *wd)
        per = [bmax(sh[i], *wh) / bmax(sd[i], *wd) for i in ids]
        rows.append((name, xh, xd, eh, ed))
        e = f"{eh:7.0f} {ed:7.0f} {eh / ed:6.3f} | {xd - ed:+12.0f}" if eh else f"{'-':>7} {'-':>7} {'-':>6} | {'-':>12}"
        print(f"{name:20s} {xh:8.0f} {xd:8.0f} {xh / xd:6.3f} | {e}   {min(per):.3f}-{max(per):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    fig = plt.figure(figsize=(18, 10), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.4, 1.0], wspace=0.12, hspace=0.25)
    axL = fig.add_subplot(gs[0, 0])
    axT = fig.add_subplot(gs[1, 0])
    axR = fig.add_subplot(gs[:, 1])
    top = max(ah.max(), ad.max())
    axL.plot(GRID, ah / top, color="#2a78d6", lw=1.7, label="calculated H$_2$O")
    axL.plot(GRID, ad / top, color="#eb6834", lw=1.7, label="calculated D$_2$O")
    for k, (name, xh, xd, eh, ed) in enumerate(rows):
        if eh:
            axL.axvline(eh, color="#2a78d6", ls=":", lw=1.0, zorder=0,
                        label="Bertie & Whalley 1964 H$_2$O (Table I)" if k == 0 else None)
            axL.axvline(ed, color="#eb6834", ls=":", lw=1.0, zorder=0,
                        label="Bertie & Whalley 1964 D$_2$O (Table I)" if k == 0 else None)
    axL.set_xlim(4000, 100)
    axL.set_ylim(0, 1.08)
    axL.minorticks_on()
    axL.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axL.set_ylabel("IR intensity (common normalisation)")
    axL.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=9.5)
    axL.set_title(f"Ice Ih H$_2$O vs. D$_2$O: HSEsol-D3 harmonic, {len(ids)} topologies, multiplicity-weighted, "
                  f"FWHM {a.fwhm:g} cm$^{{-1}}$", fontsize=11.5, loc="left")

    # table
    axT.axis("off")
    cells = [[n, f"{xh:.0f}", f"{xd:.0f}", f"{xh / xd:.3f}",
              f"{eh:.0f}" if eh else "–", f"{ed:.0f}" if ed else "–", f"{eh / ed:.3f}" if eh else "–",
              f"{xd - ed:+.0f}" if ed else "–"] for n, xh, xd, eh, ed in rows]
    t = axT.table(cellText=cells, colLabels=["band (cm$^{-1}$)", "calc H$_2$O", "calc D$_2$O", "calc H/D",
                                             "exp H$_2$O", "exp D$_2$O", "exp H/D", "calc − exp (D$_2$O)"],
                  loc="upper center", cellLoc="center", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.8)
    axT.text(0.5, -0.02, "calc: intensity-weighted band centres of the weighted average; exp: band maxima, "
             "Bertie & Whalley 1964 Table I", transform=axT.transAxes, ha="center", va="top", fontsize=9,
             color="#3d3d3a")

    step = 0.75 * max(s[GRID > 2100].max() for s in sd.values())
    sgs = []
    for i in ids:
        if jobs[i]["space_group"] not in sgs:
            sgs.append(jobs[i]["space_group"])
    col = {s: COLORS[k % 8] for k, s in enumerate(sgs)}
    n = len(ids)
    for k, i in enumerate(ids):
        off = (n - 1 - k) * step
        s = jobs[i]["space_group"]
        axR.plot(GRID, sd[i] + off, color=col[s], lw=1.0)
        axR.fill_between(GRID, off, sd[i] + off, color=col[s], alpha=0.12, lw=0)
        axR.text(1950, off + 0.18 * step, f"#{i}  ${s}$  g={g[i]}  ΔE={en.get(i, float('nan')):.2f}"
                 + ("  (ice XI)" if s == "Cmc2_1" else ""), fontsize=8.5, va="bottom")
    axR.plot(GRID, ad + n * step, color="black", lw=1.6)
    axR.text(1950, n * step + 0.18 * step, "weighted average D$_2$O", fontsize=9, fontweight="bold", va="bottom")
    axR.set_xlim(2900, 100)
    axR.set_ylim(-0.3 * step, (n + 1.5) * step)
    axR.set_yticks([])
    axR.minorticks_on()
    axR.tick_params(axis="y", which="both", left=False, right=False)
    axR.set_xlabel(r"wavenumber (cm$^{-1}$)")
    axR.set_title("individual D$_2$O topologies (sorted by ΔE, kJ mol$^{-1}$ per H$_2$O)", fontsize=11, loc="left")
    axR.legend(handles=[plt.Line2D([], [], color=col[s], lw=2.5, label=f"${s}$") for s in sgs],
               title="space group", loc="upper left", frameon=True, edgecolor="black", fancybox=False,
               fontsize=9, title_fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
