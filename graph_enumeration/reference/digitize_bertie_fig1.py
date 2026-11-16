#!/usr/bin/env python3
"""Digitise curve a (H2O ice Ih/Ic, ~110 K) of Fig. 1 (upper panel) of
J.E. Bertie, E. Whalley, J. Chem. Phys. 40, 1637 (1964).

The solid black curve is traced column by column from a start point,
following the nearest dark pixel run (grey dotted mulling-agent lines are
lighter and are skipped by the darkness threshold and the continuity window).
Pixel -> data calibration was read from the tick marks of the figure:

  upper panel, transmission: y = 16 px -> 0 %,  y = 241 px -> 100 %
  left part : x = 72 px -> 4000 cm-1, x = 322 px -> 1600 cm-1 (ticks every 400)
  right part: x = 368 px -> 1600 cm-1, x = 666 px -> 400 cm-1 (ticks every 200)

Regions where the paper marks the ice absorption as uncertain (dashed:
1500-1350 cm-1; mulling agent in 3000-2800 cm-1, interpolated by the
authors) are flagged.  Output: bertie_whalley_1964_fig1_H2O_curve_a.csv

    python digitize_bertie_fig1.py fig1.png [--check overlay.png]
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

Y0, Y100 = 16.0, 241.0
LEFT = (72.0, 4000.0, 322.0, 1600.0)
RIGHT = (368.0, 1600.0, 666.0, 400.0)

# (part, x_start, x_end, y_seed, x_seed, (ymin, ymax)) for each traced segment of curve a
SEGMENTS = [
    ("left", 73, 184, 152, 73, (18, 239)),      # 4000 -> ~2900 cm-1
    ("left", 186, 330, 162, 186, (156, 239)),   # ~2900 -> ~1520 cm-1, below curve b
    ("right", 343, 604, 187, 343, (18, 239)),   # ~1700 -> ~650 cm-1 (first detector)
    ("right", 592, 688, 171, 592, (18, 239)),   # ~690 -> ~310 cm-1 (second detector/grating)
]


def x_to_nu(x, part):
    x0, n0, x1, n1 = LEFT if part == "left" else RIGHT
    return n0 + (x - x0) * (n1 - n0) / (x1 - x0)


def y_to_T(y):
    return 100.0 * (y - Y0) / (Y100 - Y0)


def runs(col, thr):
    """centres of dark pixel runs in a column"""
    idx = np.where(col < thr)[0]
    if len(idx) == 0:
        return []
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 1:
            cur.append(i)
        else:
            out.append(np.mean(cur))
            cur = [i]
    out.append(np.mean(cur))
    return out


def trace(img, x0, x1, yseed, xseed, thr=90, win=5, ylim=(18, 239)):
    pts = {}
    y = yseed
    step = 1 if x1 >= x0 else -1
    miss = 0
    for x in range(xseed, x1 + step, step):
        cand = [c for c in runs(img[:, x], thr) if ylim[0] <= c <= ylim[1] and abs(c - y) <= win + 2 * miss]
        if not cand:
            # allow dashed parts / gaps: keep last y, widen the window
            miss += 1
            continue
        c = min(cand, key=lambda c: abs(c - y))
        pts[x] = c
        y = c
        miss = 0
    return pts


def despike(pts, half=6, tol=2.5):
    """drop points that jump away from the running median (e.g. onto a crossing curve)"""
    xs = sorted(pts)
    ys = np.array([pts[x] for x in xs])
    keep = {}
    for i, x in enumerate(xs):
        med = np.median(ys[max(0, i - half):i + half + 1])
        if abs(ys[i] - med) <= tol:
            keep[x] = ys[i]
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--check", help="write an overlay image to check the trace")
    p.add_argument("-o", "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "bertie_whalley_1964_fig1_H2O_curve_a.csv"))
    a = p.parse_args()
    img = np.array(Image.open(a.image).convert("L")).astype(float)

    rows = []
    allpts = []
    for k, (part, x0, x1, yseed, xseed, ylim) in enumerate(SEGMENTS):
        pts = trace(img, x0, x1, yseed, xseed, ylim=ylim)
        if k >= 1:          # not on segment 1: it contains the sharp O-H stretch tip
            pts = despike(pts)
        for x in sorted(pts):
            nu = x_to_nu(x, part)
            if part == "left" and nu < 1700:
                continue    # overlaps the right-hand part of the figure; keep that one
            T = y_to_T(pts[x])
            flag = ""
            if 1260 <= nu <= 1560 and part == "right":
                flag = "uncertain (dashed in paper)"
            if 2800 <= nu <= 3000:
                flag = "mulling-agent region"
            rows.append({"segment": k + 1, "wavenumber_cm-1": round(nu, 1), "transmission_%": round(T, 2),
                         "absorbance": round(-np.log10(max(T, 0.5) / 100.0), 4), "flag": flag})
            allpts.append((x, pts[x], k))
    with open(a.out, "w", newline="") as fh:
        fh.write("# Digitised from Fig. 1 (upper panel, curve a: H2O ice, ~110 K, mull, transmission) of\n"
                 "# J.E. Bertie, E. Whalley, J. Chem. Phys. 40, 1637 (1964). Approximate (~5 cm-1, ~2 % T).\n"
                 "# absorbance = -log10(T/100); the curve includes reflection/scattering background.\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} points written to {a.out}")

    if a.check:
        im = Image.open(a.image).convert("RGB")
        W, H = im.size
        im = im.resize((3 * W, 3 * H), Image.NEAREST)
        d = ImageDraw.Draw(im)
        col = [(230, 0, 0), (200, 0, 200), (0, 140, 255), (0, 170, 0)]
        for x, y, k in allpts:
            d.ellipse([3 * x - 2, 3 * y - 2, 3 * x + 2, 3 * y + 2], fill=col[k])
        im.crop((0, 0, 3 * W, 3 * 250)).save(a.check)


if __name__ == "__main__":
    main()
