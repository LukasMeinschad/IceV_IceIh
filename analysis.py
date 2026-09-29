import re
from pathlib import Path



import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter, peak_widths, find_peaks
from pybaselines import Baseline
from scipy.optimize import curve_fit
import numpy as np

"""
General Settings
"""
# PATH Settings
HERE = Path(__file__).resolve().parent 
DATA = HERE / "data"
OUT = HERE / "out"



USE_KUBELKA_MUNK = False

ATR_CORRECTION = False  

IR_MIN_WAVENUMBER = 500


# Spectral Correction Plots
SAVGOL_WINDOW, SAVGOL_ORDER = 81, 5
BASELINE_LAM = 1e8
BASELINE_LAM_OVERRIDE = {-165: 1e8}
BASELINE_DIFF_ORDER = 3
BASELINE_DIFF_ORDER_OVERRIDE = {-165: 2} # stiffer baseline for ice Ih so it does not bend into the plateau
BASELINE_QUANTILE = 0.05

# Broadening for Simulated Spectra
BROADENING_FWHM = 100.0 #cm-1


MAIN_SPECTRA = {"ice V": -160, "ice Ih": -165} # Temperatures in Degree

# XRD Settings
XRD_WAVELENGTH = 1.5406 # Cu K-alpha wavelength in Angstroms
XRD_RANGE = (1.7, 3.4) # Angstrom
XRD_SIGMA = 0.001 # Width to draw simulated peaks 
XRD_MERGE_TOL = 0.05 # Reflections within this d-spacing tolerance are merged


XRD_LITERATURE = { 
    "Bertie et al. (1963)": [3.02, 2.82, 2.65, 2.43, 2.29, 2.19, 2.05, 1.96, 1.88, 1.78, 1.72, 1.63, 1.56, 1.49, 1.44, 1.38, 1.29],
    "Kamb et al. (1967)": [3.03, 2.93, 2.86, 2.75, 2.68, 2.52, 2.45, 2.40, 2.29, 2.05],
    "Salzmann et al. (2021)": [3.01, 2.98, 2.91, 2.84, 2.73, 2.65, 2.50, 2.43, 2.30, 1.57]
}

# Fit Regions for Gaussians
FIT_REGIONS = {"stretching": (3000, 3500), "bending": (1200, 1800), "libration": (550, 1000)}
FWHM_REGIONS = {"stretching": (2800, 4000), "combination": (2000, 2600),
                "bending": (1000, 1800), "libration": (500, 1000)}

GROUPS = [
    ("ice I$_\\mathrm{h}$, recooled", (-165, -165), None),
    ("ice V, 113$-$158 K", (-160, -115),
     ["#00001a", "#000080", "#0000CD", "#3366FF", "#66B2FF", "#B3E0FF"]),
    ("ice V, 163$-$193 K", (-110, -80),
     ["#002200", "#006400", "#228B22", "#4CAF50", "#8BC34A", "#C8E6A0"]),
]
SI_REGIONS = [("full", 500, 4000), ("stretching", 2800, 3600), ("combination", 2000, 2600),
              ("bending", 1000, 1800), ("libration", 500, 1000)]
 
C_MEAS, C_BERTIE, C_CALC = "black", "darkorange", "royalblue"
 
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42})


""" 
Helper Functions
"""
def kelvin(t_c):
    return f"{t_c + 273.15:.0f} K"

def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name} to {OUT}")

def save_table(df, name):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / name, index=False, float_format="%.6f")
    print(f"Saved {name} to {OUT}")

def normalize(y):
    """
    Shift minimum to zero and maximum to 100%
    """
    y = np.asarray(y, float) - np.nanmin(y)
    return 100 * y / np.nanmax(y)

def select(x,y,lo,hi):
    """  
    Select a certain region of the spectrum
    """
    x, y = np.asarray(x), np.asarray(y)
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]  

def y_label():
    if USE_KUBELKA_MUNK:
        return r"Normalized Kubelka$-$Munk function $F(R_\infty)$"
    else:
        return r"Normalized Absorbance (%)"

"""  
Data READ
"""
def read_atr_spectra():
    """Return {temperature_degC: (phase, wavenumber, absorbance)}."""
    spectra = {}
    for path in sorted((DATA / "atr").glob("*.csv")):
        m = re.match(r"^([A-Za-z]+)_(-?\d+)_\d+_scans_\d+\.csv$", path.name)
        if not m:
            print(f"  skipping {path.name} (unexpected name)")
            continue
        df = pd.read_csv(path, sep=";", decimal=",", skiprows=1, encoding="utf-8-sig")
        df = df.iloc[:, :2].sort_values(df.columns[0])
        spectra[int(m[2])] = (m[1], df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
    return dict(sorted(spectra.items()))

def read_bertie(phase):
    """Digitized mull spectrum (transmission %) -> wavenumber, signal."""
    name = {"ice V": "iceV", "ice Ih": "iceih"}[phase]
    df = pd.read_csv(DATA / "literature" / f"bertie1964_{name}_mull_transmission.csv", comment="#",
                     header=None, names=["wavenumber_cm-1", "transmission_pct"])
    df = df.sort_values("wavenumber_cm-1")
    t = df["transmission_pct"].to_numpy() / 100
    y = (1 - t) ** 2 / (2 * t) if USE_KUBELKA_MUNK else -np.log10(t)
    return df["wavenumber_cm-1"].to_numpy(), y

def read_crystal_modes(path):
    """Frequencies (cm^-1) and IR intensities (km/mol) from a CRYSTAL output."""
    num = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
    line_re = re.compile(rf"^\s*(\d+)-\s*(\d+)\s+({num})\s+({num})\s+({num})")
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = line_re.match(line)
        if m:
            paren = re.findall(rf"\(\s*({num})\s*\)", line)
            rows.append((int(m[1]), int(m[2]), float(m[4]), float(paren[-1]) if paren else 0.0))
    df = pd.DataFrame(rows, columns=["mode_first", "mode_last", "frequency_cm1", "ir_intensity"])
    starts = np.flatnonzero(df["mode_first"] == 1)
    if len(starts) > 1:                       # table printed twice -> keep the last one
        df = df.iloc[starts[-1]:]
    n_imag = (df["frequency_cm1"] < 0).sum()
    if n_imag:
        print(f"  {Path(path).name}: {n_imag} imaginary mode(s) omitted")
    return df[df["frequency_cm1"] >= 0].reset_index(drop=True)


"""  
Data Processing
"""

def process_atr(wavenumber, absorbance, t_c):
    """
    Follow the steps in the paper
    1. Remove low wavenumber region (below 500 cm^-1)
    2. Apply Savitzky-Golay filter to smooth the data
    3. Apply baseline correction using the irsqr method from pybaselines
    4. 
    """
    m = wavenumber > IR_MIN_WAVENUMBER
    x, y = wavenumber[m], absorbance[m]
    if USE_KUBELKA_MUNK:
        r = 10.0 ** (-y)
        y = (1 - r) ** 2 / (2 * r)
    y = savgol_filter(y, SAVGOL_WINDOW, SAVGOL_ORDER)
    lam = BASELINE_LAM_OVERRIDE.get(t_c, BASELINE_LAM)
    base, _ = Baseline(x_data=x).irsqr(y, lam=lam, quantile=BASELINE_QUANTILE,
                                       diff_order=BASELINE_DIFF_ORDER_OVERRIDE.get(t_c, BASELINE_DIFF_ORDER))
    y = y - base
    if ATR_CORRECTION and not USE_KUBELKA_MUNK:
        #TODO
        raise NotImplementedError("ATR correction not implemented yet")
    return x, y

def broaden(freqs, intensities, fwhm=BROADENING_FWHM):
    """
    Apply Gaussian Broadening
    """
    x = np.arange(0, 4000.2, 0.2)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = np.zeros_like(x)
    for f, a in zip(freqs, intensities):
        y += a * np.exp(-0.5 * ((x - f) / sigma) ** 2)
    return x, y

def pseudo_voigt(x, amp, cen, fwhm, eta, slope, offset):
    """Gaussian and Lorentzian with common FWHM, mixed by eta, on a linear background."""
    g = np.exp(-4 * np.log(2) * ((x - cen) / fwhm) ** 2)
    lor = 1 / (1 + (2 * (x - cen) / fwhm) ** 2)
    return amp * ((1 - eta) * g + eta * lor) + slope * x + offset

def fit_band(x, y, window):
    lo, hi = window
    xr, yr = select(x, y, lo, hi)
    ys = savgol_filter(yr, 25, 3)
    i0 = np.argmax(ys)
    p0 = [ys[i0] - yr.min(), xr[i0], (hi - lo) / 4, 0.3, 0, yr.min()]
    bounds = ([0, lo, 1, 0, -np.inf, -np.inf], [np.inf, hi, 2 * (hi - lo), 1, np.inf, np.inf])
    try:
        popt, _ = curve_fit(pseudo_voigt, xr, yr, p0=p0, bounds=bounds,
                            loss="soft_l1", max_nfev=10000)
    except (RuntimeError, ValueError):
        return None
    resid = yr - pseudo_voigt(xr, *popt)
    r2 = 1 - np.sum(resid ** 2) / np.sum((yr - yr.mean()) ** 2)
    return dict(popt=popt, center=popt[1], fwhm=popt[2], r2=r2, x=xr, y=yr)


def band_fwhm(x, y, label):
    """Peak positions and FWHM at half height in each FWHM region."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], normalize(np.asarray(y)[order]) / 100
    rows = []
    for region, (lo, hi) in FWHM_REGIONS.items():
        xr, yr = select(x, y, lo, hi)
        if len(xr) < 3:
            continue
        dist = max(1, int(20 / np.median(np.diff(xr))))
        pk, _ = find_peaks(yr, prominence=0.03, distance=dist)
        if len(pk) == 0:
            continue
        _, _, left, right = peak_widths(yr, pk, rel_height=0.5)
        idx = np.arange(len(xr))
        for p, lft, rgt in zip(pk, left, right):
            rows.append(dict(spectrum=label, region=region, position_cm1=xr[p],
                             height_rel=yr[p],
                             fwhm_cm1=np.interp(rgt, idx, xr) - np.interp(lft, idx, xr)))
    return rows 


def merge_reflections(d, intensity, tol=XRD_MERGE_TOL):
    """Sum reflections within tol (symmetry-equivalent reflections are listed separately in P1)."""
    order = np.argsort(d)[::-1]
    d, intensity = d[order], intensity[order]
    groups = [[0]]
    for i in range(1, len(d)):
        if d[groups[-1][0]] - d[i] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    rows = [dict(d_spacing_A=np.average(d[g], weights=intensity[g]) if intensity[g].sum() > 0
                 else d[g].mean(), intensity=intensity[g].sum(), n_reflections=len(g))
            for g in groups]
    df = pd.DataFrame(rows)
    df["intensity_rel"] = df["intensity"] / df["intensity"].max()
    return df


""" 
Figures
"""
def xrd_figure():
    print("XRD")
    lo, hi = XRD_RANGE
    raw = np.loadtxt(DATA / "xrd" / "xrd_iceV_measured.txt", skiprows=1)
    d = XRD_WAVELENGTH / (2 * np.sin(np.radians(raw[:, 0] / 2)))
    m = (d >= lo) & (d <= hi)
    d_meas, i_meas = d[m], raw[m, 1] / raw[m, 1].max()
 
    grid = np.linspace(lo, hi, 4000)
    calc, tables = {}, []
    for phase, fname in [("ice V", "xrd_pattern_iceV_hsesol.dat"),
                         ("ice Ih", "xrd_pattern_iceIh_hsesol.dat")]:
        ref = np.loadtxt(DATA / "xrd" / fname, skiprows=1, usecols=range(9))
        d_ref, i_ref = ref[:, 3], ref[:, 8]
        prof = sum(i * np.exp(-0.5 * ((grid - dr) / XRD_SIGMA) ** 2)
                   for dr, i in zip(d_ref, i_ref) if lo - 0.05 <= dr <= hi + 0.05)
        calc[phase] = prof / prof.max()
        sel = (d_ref >= lo) & (d_ref <= hi)
        t = merge_reflections(d_ref[sel], i_ref[sel])
        t.insert(0, "source", f"{phase} HSEsol/POB-TZVP")
        tables.append(t)
 
    pk, _ = find_peaks(i_meas, height=0.10, prominence=0.015, distance=8)
    tables.insert(0, pd.DataFrame(dict(source="measured ice V", d_spacing_A=d_meas[pk],
                                       intensity_rel=i_meas[pk])))
    lit = {s: np.array(v) for s, v in XRD_LITERATURE.items()}
    tables += [pd.DataFrame(dict(source=s, d_spacing_A=v)) for s, v in lit.items()]
    save_table(pd.concat(tables, ignore_index=True), "table_xrd_peaks.csv")
 
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.fill_between(d_meas, i_meas, color="k", alpha=0.07, lw=0)
    h_meas, = ax.plot(d_meas, i_meas, color="k", lw=1.0, label="Measured ice V")
    h_v, = ax.plot(grid, np.where(calc["ice V"] > 2e-3, calc["ice V"], np.nan),
                   color="orangered", ls="--", lw=1.1, label="ice V HSEsol/POB-TZVP")
    h_ih, = ax.plot(grid, np.where(calc["ice Ih"] > 2e-3, calc["ice Ih"], np.nan),
                    color="royalblue", ls=(0, (5, 2, 1, 2)), lw=1.1,
                    label=r"ice I$_\mathrm{h}$ HSEsol/POB-TZVP")
    ax.axhline(0, color="0.45", ls="--", lw=0.8)
    colors = ["green", "darkviolet", "goldenrod"]
    pitch, height = 0.075, 0.055
    h_lit = []
    for i, ((src, refl), c) in enumerate(zip(lit.items(), colors)):
        y0 = -pitch * (i + 1)
        h_lit.append(ax.vlines(refl, y0 - height / 2, y0 + height / 2, color=c, lw=1.6, label=src))
    ax.text(hi - 0.015, -pitch * (len(lit) + 1) / 2, "literature\nreflections",
            ha="right", va="center", color="0.35", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-pitch * (len(lit) + 0.7), 1.36)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"$d$-spacing (Å)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(color="0.5", alpha=0.15)
    kw = dict(frameon=False, alignment="left", title_fontproperties={"weight": "bold"})
    ax.add_artist(ax.legend([h_meas, *h_lit], [h.get_label() for h in [h_meas, *h_lit]],
                            title="Experimental", loc="upper left", ncol=2, **kw))
    ax.legend([h_v, h_ih], [h_v.get_label(), h_ih.get_label()], title="Calculated",
              loc="upper right", **kw)
    save(fig, "fig_xrd")


def dft_figures():
    print("DFT")
    modes, spectra = {}, {}
    for phase, tag in [("ice V", "iceV"), ("ice Ih", "iceIh")]:
        for opt, suffix in (("atomonly", "atomonly"), ("fullcell", "full")):
            df = read_crystal_modes(DATA / "calcs" / tag / f"{tag}_hsesol_{suffix}.out")
            modes[(phase, opt)] = df
            spectra[(phase, opt)] = broaden(df["frequency_cm1"], df["ir_intensity"])
            save_table(df, f"table_modes_{tag}_{opt}.csv")
 
    rows = []
    for (phase, opt), (x, y) in spectra.items():
        pk, _ = find_peaks(y / y.max(), prominence=1e-4, distance=50)
        rows += [dict(phase=phase, optimization=opt, wavenumber_cm1=x[k],
                      intensity_rel=y[k] / y.max()) for k in pk]
    save_table(pd.DataFrame(rows), "table_calc_maxima.csv")
 
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    for ax, phase in zip(axes, ["ice V", "ice Ih"]):
        for opt, c, lab in [("atomonly", "blue", "Atom-only"), ("fullcell", "red", "Full-cell")]:
            x, y = spectra[(phase, opt)]
            ax.plot(x, y / y.max(), color=c, lw=1.1, label=lab)
        ax.set_title(phase.replace("Ih", r"I$_\mathrm{h}$") + ": HSEsol/POB-TZVP")
        ax.set_ylabel("Normalized intensity")
        ax.set_xlim(4000, 0)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    save(fig, "figS_dft_atomonly_vs_fullcell")
    return modes, spectra


def ir_main_figure(processed, modes, spectra):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.8), sharex=True)
    fwhm_rows = []
    for ax, (phase, t_c) in zip(axes, MAIN_SPECTRA.items()):
        xm, ym = processed[t_c][1:]
        xb, yb = read_bertie(phase)
        xc, yc = spectra[(phase, "atomonly")]
        df = modes[(phase, "atomonly")]
 
        fwhm_rows += band_fwhm(xm, ym, f"{phase} this work {kelvin(t_c)}")
        fwhm_rows += band_fwhm(xb, yb, f"{phase} Bertie and Whalley")
        fwhm_rows += band_fwhm(xc, yc, f"{phase} HSEsol")
 
        axc = ax.twinx()
        xc, yc = select(xc, yc, 500, 4000)
        f, s = df["frequency_cm1"].to_numpy(), df["ir_intensity"].to_numpy()
        ok = (f >= 500) & (f <= 4000)
        axc.vlines(f[ok], 0, 100 * s[ok] / s[ok].max(), color=C_CALC, lw=0.6, alpha=0.45)
        h_c, = axc.plot(xc, 100 * yc / yc.max(), color=C_CALC, lw=1.2)
        axc.fill_between(xc, 0, 100 * yc / yc.max(), color=C_CALC, alpha=0.15)
        axc.set_ylim(0, 105)
        axc.tick_params(axis="y", labelcolor=C_CALC, color=C_CALC)
        axc.spines["right"].set_color(C_CALC)
 
        xb, yb = select(xb, yb, 500, 4000)
        xm, ym = select(xm, ym, 500, 4000)
        h_b, = ax.plot(xb, normalize(yb), color=C_BERTIE, ls="--", lw=1.2)
        h_m, = ax.plot(xm, normalize(ym), color=C_MEAS, lw=1.2)
        ax.set_xlim(4000, 500)
        ax.set_ylim(0, 105)
        ax.set_title(phase.replace("Ih", r"I$_\mathrm{h}$"))
        ax.grid(True, color="0.9")
        ax.set_zorder(axc.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.legend([h_m, h_b, h_c], [f"This work, {kelvin(t_c)}", "Bertie and Whalley (1964)",
                                    "HSEsol/POB-TZVP (harmonic)"],
                  loc="upper right", framealpha=0.9, edgecolor="0.8")
    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    fig.supylabel(y_label(), x=-0.005, fontsize=11)
    fig.text(1.0, 0.5, "Normalized IR intensity (%)", color=C_CALC, rotation=270,
             va="center", ha="left", fontsize=11)
    save(fig, "fig_ir_main")
    save_table(pd.DataFrame(fwhm_rows), "table_ir_fwhm.csv")


def ir_temperature_series(processed):
    for name, lo, hi in SI_REGIONS:
        fig, axes = plt.subplots(3, 1, figsize=(7, 7.8), sharex=True)
        for ax, (title, (t_lo, t_hi), colors) in zip(axes, GROUPS):
            temps = [t for t in processed if t_lo <= t <= t_hi]
            if colors:
                cmap = LinearSegmentedColormap.from_list(title, colors)
                cols = cmap(np.linspace(0, 1, len(temps)) if len(temps) > 1 else [0.5])
            else:
                cols = ["black"] * len(temps)
            top = 1
            for t, c in zip(temps, cols):
                _, x, y = processed[t]
                x, y = select(x, 100 * y / y.max(), lo, hi)   # normalized to the full spectrum
                ax.plot(x, y, color=c, lw=1.1, label=kelvin(t))
                top = max(top, y.max())
            ax.set_title(title)
            ax.set_xlim(hi, lo)
            ax.set_ylim(top=1.08 * top)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
        axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
        fig.supylabel(y_label(), fontsize=11)
        fig.tight_layout()
        save(fig, f"figS_ir_series_{name}")
 
 
def band_centre_figures(processed):
    rows, example = [], {}
    for t, (phase, x, y) in processed.items():
        y = 100 * y / y.max()
        for region, window in FIT_REGIONS.items():
            res = fit_band(x, y, window)
            rows.append(dict(temperature_C=t, temperature_K=t + 273.15, phase=phase,
                             region=region,
                             center_cm1=res["center"] if res else np.nan,
                             fwhm_fit_cm1=res["fwhm"] if res else np.nan,
                             r2=res["r2"] if res else np.nan))
            if t == MAIN_SPECTRA["ice V"]:
                example[region] = res
    centres = pd.DataFrame(rows)
    save_table(centres, "table_band_centres.csv")
 
    # band centre vs temperature (ice V only)
    colors = {"stretching": "#1f4e8c", "bending": "#c0392b", "libration": "#2e8b57"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    icev = centres[centres["temperature_C"] != MAIN_SPECTRA["ice Ih"]]
    for ax, (region, (lo, hi)) in zip(axes, FIT_REGIONS.items()):
        d = icev[icev["region"] == region].sort_values("temperature_K")
        ax.plot(d["temperature_K"], d["center_cm1"], "o--", color=colors[region], ms=4, lw=0.8)
        ax.set_title(f"{region.capitalize()} ({hi}$-${lo} cm$^{{-1}}$)")
        ax.set_xlabel("Temperature (K)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Band centre (cm$^{-1}$)")
    fig.tight_layout()
    save(fig, "fig_band_shifts")
 
    # fit example
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (region, res) in zip(axes, example.items()):
        lo, hi = FIT_REGIONS[region]
        if res is None:
            ax.set_title(f"{region}: fit failed")
            continue
        xf = np.linspace(lo, hi, 600)
        ax.plot(res["x"], res["y"], color="royalblue", lw=1.2,
                label=f"ATR, {kelvin(MAIN_SPECTRA['ice V'])}")
        ax.plot(xf, pseudo_voigt(xf, *res["popt"]), color="red", lw=1, label="pseudo-Voigt fit")
        ax.axvline(res["center"], color="green", ls="--", lw=0.8,
                   label=f"centre {res['center']:.1f} cm$^{{-1}}$")
        ax.set_xlim(hi, lo)
        ax.set_title(f"{region.capitalize()} ($R^2$ = {res['r2']:.3f})")
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(y_label())
    fig.tight_layout()
    save(fig, "figS_band_fit_example")
 
 
def baseline_check(raw):
    """Smoothed spectra with their baselines, to check the baseline parameters."""
    temps = list(raw)
    fig, axes = plt.subplots(int(np.ceil(len(temps) / 3)), 3,
                             figsize=(11, 2.2 * np.ceil(len(temps) / 3)), sharex=True)
    for ax, t in zip(axes.flat, temps):
        phase, x, a = raw[t]
        m = x > IR_MIN_WAVENUMBER
        x, y = x[m], a[m]
        if USE_KUBELKA_MUNK:
            y = (1 - 10.0 ** -y) ** 2 / (2 * 10.0 ** -y)
        y = savgol_filter(y, SAVGOL_WINDOW, SAVGOL_ORDER)
        base, _ = Baseline(x_data=x).irsqr(y, lam=BASELINE_LAM_OVERRIDE.get(t, BASELINE_LAM),
                                           quantile=BASELINE_QUANTILE,
                                           diff_order=BASELINE_DIFF_ORDER_OVERRIDE.get(t, BASELINE_DIFF_ORDER))
        ax.plot(x, y, "k", lw=0.8)
        ax.plot(x, base, "r", lw=0.8)
        ax.set_title(f"{phase} {kelvin(t)}", fontsize=9)
        ax.set_xlim(4000, IR_MIN_WAVENUMBER)
    for ax in axes.flat[len(temps):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "check_baselines")    


def main():
    print("Starting analysis...")
    xrd_figure()
    print(f"ATR Correction: {'ON' if ATR_CORRECTION else 'OFF'}")
    print(f"Using Kubelka-Munk: {'ON' if USE_KUBELKA_MUNK else 'OFF'}")
    
    modes, spectra = dft_figures()

    print("Processing ATR spectra...")
    raw = read_atr_spectra()
    processed = {t: (ph, *process_atr(x, a, t)) for t, (ph, x, a) in raw.items()}
    table = pd.DataFrame({"wavenumber_cm-1": processed[next(iter(processed))][1]})
    for t, (ph, x, y) in processed.items():
        table[f"{ph}_{kelvin(t).replace(' ', '')}"] = y
    save_table(table, "table_atr_processed.csv")

    ir_main_figure(processed, modes, spectra)
    ir_temperature_series(processed)
    band_centre_figures(processed)
    baseline_check(raw)
    print("Analysis complete.")
    
if __name__ == "__main__":
    main()