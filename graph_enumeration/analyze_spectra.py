#!/usr/bin/env python3
"""IR spectra of the H-bond topologies from CRYSTAL frequency runs.

Reads every finished <jobs>/config_XXXXX/freq/<name>.out (FREQCALC + INTENS),
and writes to <jobs>/analysis/:

    modes.csv          all Gamma-point modes: frequency, irrep, IR activity,
                       intensity (km/mol per cell and per H2O)
    freq_summary.csv   per topology: imaginary modes, band positions
                       (intensity-weighted centres of the main regions)
    ir_spectra.csv     broadened spectra (per H2O) of every topology + the
                       multiplicity-weighted average
    ir_spectra.png     calculated spectra vs. the measured ATR spectrum

    python analyze_spectra.py ../crystal_calcs/iceih/topologies/hsesol_d3 \
        --atr ../atr_data/Ih_-165_25_scans_1.csv --fwhm 30

Frequencies are harmonic and unscaled unless --scale is given.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

MODE_RE = re.compile(r"^\s*(\d+)-\s*(\d+)\s+(-?[\d.E+-]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\((\S+)\s*\)\s+"
                     r"([AI])\s+\(\s*([\d.]+)\)\s+([AI])")
REGIONS = {  # cm-1, for band centres
    "translations": (0, 400),
    "librations": (400, 1300),
    "bend": (1500, 1800),
    "OH_stretch": (2800, 3800),
}


def parse_freq(path):
    txt = open(path, errors="replace").read()
    if "EEEEEEEEEE TERMINATION" not in txt:
        return None
    natoms = int(re.findall(r"N\. OF ATOMS PER CELL\s+(\d+)", txt)[-1])
    block = txt[txt.rfind("MODES         EIGV          FREQUENCIES"):]
    modes = []
    for line in block.splitlines()[2:]:
        m = MODE_RE.match(line)
        if not m:
            if modes:
                break
            continue
        n1, n2 = int(m.group(1)), int(m.group(2))
        modes.append({"first": n1, "degeneracy": n2 - n1 + 1, "freq": float(m.group(4)),
                      "irrep": m.group(6), "ir": m.group(7), "intensity": float(m.group(8)),
                      "raman": m.group(9)})
    return {"natoms": natoms, "n_H2O": natoms // 3, "modes": modes}


def broaden(freqs, ints, grid, fwhm, shape="lorentz"):
    y = np.zeros_like(grid)
    if shape == "lorentz":
        g = fwhm / 2
        for f, i in zip(freqs, ints):
            y += i * (g / np.pi) / ((grid - f) ** 2 + g ** 2)
    else:
        s = fwhm / (2 * np.sqrt(2 * np.log(2)))
        for f, i in zip(freqs, ints):
            y += i * np.exp(-0.5 * ((grid - f) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return y          # km/mol per cm-1


def read_atr(path):
    x, y = [], []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        parts = line.strip().replace(",", ".").split(";")
        if len(parts) < 2:
            continue
        try:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        except ValueError:
            continue
    x, y = np.array(x), np.array(y)
    o = np.argsort(x)
    return x[o], y[o]


def read_ref(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or line.startswith("wavenumber") or not line.strip():
            continue
        w, lo, hi, inten, assign, label = [x.strip() for x in line.split(",", 5)]
        rows.append({"wavenumber": float(w), "low": float(lo), "high": float(hi),
                     "intensity": inten, "assignment": assign, "label": label})
    return rows


def compare_ref(summ, done, grid, avg, ref, a, path):
    """Nearest calculated feature for every reference band (weighted average spectrum)."""
    def local_maxima(y, lo, hi):
        s = (grid >= lo) & (grid <= hi)
        xs, ys = grid[s], y[s]
        if len(ys) < 3 or ys.max() <= 0:
            return []
        return [(xs[i], ys[i] / ys.max()) for i in range(1, len(ys) - 1)
                if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] > 0.03 * ys.max()]
    regions = [(2900, 3700), (2000, 2500), (1500, 1800), (400, 1300)]
    print(f"\n  comparison with {a.ref_name} (calculated: weighted average, FWHM {a.fwhm:g})")
    print(f"  {'reference':>22} {'assignment':>18} {'nearest calc. max':>18} {'rel. int.':>9} {'diff':>7}")
    rows = []
    for r in ref:
        reg = next(((lo, hi) for lo, hi in regions if lo <= r["wavenumber"] <= hi), None)
        mx = local_maxima(avg, *reg) if reg else []
        if mx:
            x, rel = min(mx, key=lambda t: abs(t[0] - r["wavenumber"]))
            d = x - r["wavenumber"]
        else:
            x = rel = d = None
        rows.append({"reference_cm-1": r["wavenumber"], "label": r["label"], "assignment": r["assignment"],
                     "calc_cm-1": x, "calc_rel_intensity": round(rel, 2) if rel is not None else "",
                     "diff_cm-1": d})
        print(f"  {r['label']:>22} {r['assignment']:>18} {('%.0f' % x) if x is not None else 'none':>18} "
              f"{('%.2f' % rel) if rel is not None else '':>9} {('%+.0f' % d) if d is not None else '':>7}")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--atr", help="measured ATR spectrum (cm-1;A, decimal commas allowed)")
    p.add_argument("--fwhm", type=float, default=30.0, help="line width [cm-1]")
    p.add_argument("--shape", choices=("lorentz", "gauss"), default="lorentz")
    p.add_argument("--scale", type=float, default=1.0, help="frequency scale factor (all modes)")
    p.add_argument("--scale-stretch", type=float, default=None,
                   help="separate scale factor for modes above 2500 cm-1 (O-H stretches)")
    p.add_argument("--xmin", type=float, default=400.0)
    p.add_argument("--xmax", type=float, default=4000.0)
    p.add_argument("--title", default=None)
    p.add_argument("--ref", help="reference band list (csv: wavenumber,low,high,intensity,assignment,label), "
                                 "e.g. reference/bertie_whalley_1964_iceIh_H2O.csv")
    p.add_argument("--ref-name", default="Bertie & Whalley 1964")
    a = p.parse_args()

    jobs = list(csv.DictReader(open(os.path.join(a.jobs, "jobs.csv"))))
    out = os.path.join(a.jobs, "analysis")
    os.makedirs(out, exist_ok=True)
    grid = np.arange(0.0, 4200.0, 1.0)

    done, mode_rows, summ = [], [], []
    for j in jobs:
        f = os.path.join(a.jobs, j["directory"], "freq", j["input"] + ".out")
        if not os.path.exists(f):
            continue
        r = parse_freq(f)
        if r is None or not r["modes"]:
            continue
        cid = int(j["id"])
        fr = np.array([m["freq"] for m in r["modes"]])
        fr_s = fr * a.scale
        if a.scale_stretch:
            fr_s = np.where(fr > 2500, fr * a.scale_stretch, fr_s)
        ii = np.array([m["intensity"] for m in r["modes"]]) / r["n_H2O"]
        for m, fs, ih in zip(r["modes"], fr_s, ii):
            mode_rows.append({"id": cid, "space_group": j["space_group"], "mode": m["first"],
                              "degeneracy": m["degeneracy"], "freq_cm-1": m["freq"], "freq_scaled": round(fs, 2),
                              "irrep": m["irrep"], "IR": m["ir"], "Raman": m["raman"],
                              "intensity_km/mol_cell": m["intensity"], "intensity_km/mol_H2O": round(ih, 3)})
        real = fr > 1.0
        imag = fr < -1.0
        s = {"id": cid, "space_group": j["space_group"], "multiplicity": int(j["multiplicity"]),
             "n_H2O": r["n_H2O"], "n_imaginary": int(imag.sum()),
             "lowest_cm-1": round(float(fr[~(np.abs(fr) <= 1.0)].min()), 1),
             "total_IR_km/mol_H2O": round(float(ii.sum()), 1)}
        for name, (lo, hi) in REGIONS.items():
            sel = real & (fr_s >= lo) & (fr_s < hi)
            w = ii[sel]
            s[f"{name}_centre"] = round(float((fr_s[sel] * w).sum() / w.sum()), 1) if w.sum() > 0 else ""
            s[f"{name}_IR_km/mol_H2O"] = round(float(w.sum()), 1)
        s["strongest_stretch_cm-1"] = round(float(fr_s[real & (fr_s > 2500)][np.argmax(ii[real & (fr_s > 2500)])]), 1) \
            if np.any(real & (fr_s > 2500)) else ""
        summ.append(s)
        spec = broaden(fr_s[real], ii[real], grid, a.fwhm, a.shape)
        done.append((cid, j["space_group"], int(j["multiplicity"]), spec))

    if not done:
        raise SystemExit("no finished frequency calculations found")

    with open(os.path.join(out, "modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mode_rows[0].keys()))
        w.writeheader()
        w.writerows(mode_rows)
    with open(os.path.join(out, "freq_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    g = np.array([d[2] for d in done], float)
    avg = sum(gi * d[3] for gi, d in zip(g, done)) / g.sum()
    with open(os.path.join(out, "ir_spectra.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wavenumber"] + [f"config_{d[0]}_{d[1]}" for d in done] + ["weighted_average"])
        for k, x in enumerate(grid):
            w.writerow([x] + [f"{d[3][k]:.5g}" for d in done] + [f"{avg[k]:.5g}"])

    print(f"{len(done)} finished frequency calculations, FWHM {a.fwhm} cm-1 ({a.shape}), scale {a.scale}"
          + (f", stretch scale {a.scale_stretch}" if a.scale_stretch else ""))
    hdr = ["id", "space_group", "n_imaginary", "lowest_cm-1", "librations_centre", "bend_centre",
           "OH_stretch_centre", "strongest_stretch_cm-1", "OH_stretch_IR_km/mol_H2O"]
    print(" ".join(f"{h:>14}" for h in hdr))
    for s in summ:
        print(" ".join(f"{str(s[h]):>14}" for h in hdr))

    atr = read_atr(a.atr) if a.atr else None
    if atr is not None:
        x, y = atr
        for name, (lo, hi) in [("librations", (500, 1100)), ("bend", (1500, 1800)),
                               ("combination", (2100, 2350)), ("OH_stretch", (2900, 3700))]:
            sel = (x >= lo) & (x <= hi)
            print(f"  ATR {name:12s} maximum at {x[sel][np.argmax(y[sel])]:7.1f} cm-1")
    ref = read_ref(a.ref) if a.ref else None
    if ref:
        compare_ref(summ, done, grid, avg, ref, a, os.path.join(out, "comparison_reference.csv"))
    plot(done, grid, avg, atr, a, os.path.join(out, "ir_spectra.png"), ref)
    print(f"written to {out}/")


def plot(done, grid, avg, atr, a, path, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    n = 2 if atr is not None else 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.4 * n), dpi=200, sharex=True, squeeze=False)
    axes = axes[:, 0]
    if atr is not None:
        x, y = atr
        sel = (x >= a.xmin) & (x <= a.xmax)
        axes[0].plot(x[sel], y[sel], color="black", linewidth=1.2)
        axes[0].set_ylabel("ATR absorbance")
        axes[0].set_title(a.title or "Ice Ih: measured ATR vs. calculated IR (HSEsol-D3, harmonic)",
                          fontsize=10, pad=22 if ref else 6)
        axes[0].text(0.01, 0.92, "measured, " + os.path.basename(a.atr), transform=axes[0].transAxes,
                     fontsize=8, va="top")
    ax = axes[-1]
    groups = []
    for d in done:
        if d[1] not in groups:
            groups.append(d[1])
    colour = {sg: COLORS[i % 8] for i, sg in enumerate(groups)}
    seen = set()
    for cid, sg, mult, spec in done:
        lab = f"${sg}$" if sg not in seen else None
        seen.add(sg)
        ax.plot(grid, spec, color=colour[sg], linewidth=0.8, alpha=0.8, label=lab)
    ax.plot(grid, avg, color="black", linewidth=1.6, label="weighted average")
    ax.set_ylabel(r"IR intensity (km mol$^{-1}$ cm per H$_2$O)")
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)" + (f", scaled ×{a.scale}" if a.scale != 1 else "")
                  + (f", stretch ×{a.scale_stretch}" if a.scale_stretch else ""))
    ax.text(0.01, 0.92, f"calculated, {len(done)} topologies, FWHM {a.fwhm:g} cm$^{{-1}}$",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(title="space group", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=True, edgecolor="black", fancybox=False, fontsize=9, title_fontsize=9)
    if ref:
        for axx in axes:
            for r in ref:
                if r["high"] > r["low"]:
                    axx.axvspan(r["low"], r["high"], color="#9a9a94", alpha=0.18, linewidth=0, zorder=0)
                axx.axvline(r["wavenumber"], color="#6b6a64", linestyle=":", linewidth=0.9, zorder=0)
        top = axes[0]
        for r in ref:
            if r["intensity"].strip() in ("s", "sh s", "vb w"):
                top.text(r["wavenumber"], 1.01, f"{r['wavenumber']:.0f}", transform=top.get_xaxis_transform(),
                         rotation=90, ha="center", va="bottom", fontsize=7, color="#3d3d3a")
        axes[-1].plot([], [], color="#6b6a64", linestyle=":", linewidth=0.9, label=a.ref_name)
        axes[-1].legend(title="space group", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                        frameon=True, edgecolor="black", fancybox=False, fontsize=9, title_fontsize=9)
    for axx in axes:
        axx.set_xlim(a.xmax, a.xmin)
        axx.minorticks_on()
        axx.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


if __name__ == "__main__":
    main()
