#!/usr/bin/env python3
"""Digitise Fig. 1 (upper frame) of J.E. Bertie, H.J. Labbé, E. Whalley,
"Infrared spectrum of ice VI in the range 4000-50 cm-1", J. Chem. Phys. 49, 2141 (1968).

Upper frame, lower curve: H2O ice VI (propylene mull), 4000-360 cm-1.
Upper frame, upper-right curve: H2O ice VI (propane mull), ~1800-360 cm-1.
The y axis is already absorbance, log10(I0/I).

Calibration from the figure's tick marks (pixel coordinates of the 756 x 716 image):
  y = 306 px -> 0.0,  y = 21 px -> 1.0   (upper frame)
  x = 110 px -> 4000 cm-1, x = 673 px -> 0 cm-1 (ticks at 3000/2000/1000 checked)

    python digitize_bertie_iceVI.py bertie_1968_iceVI_fig1.png [--check overlay.png]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

Y0, Y1 = 306.0, 21.0
X4000, X0 = 110.0, 673.0

# (name, x_start, x_end, y_seed, (ymin, ymax))
SEGMENTS = [
    ("H2O_propylene", 112, 622, 146, (27, 300)),
    ("H2O_propane", 407, 622, 199, (27, 300)),
    ("H2O_propylene", 596, 622, 230, (215, 300)),     # tail after the near-vertical drop
]


def nu(x):
    return 4000.0 * (X0 - x) / (X0 - X4000)


def absorb(y):
    return (Y0 - y) / (Y0 - Y1)


def runs(col, thr):
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


def trace(img, x0, x1, yseed, ylim, thr=110, win=6, below=None):
    """follow the curve column by column; across gaps (dashed parts) the position is
    extrapolated with the slope of the last traced points.  'below': another trace this
    curve must stay above (smaller y)."""
    pts, xl, yl, slope = {}, x0 - 1, yseed, 0.0
    for x in range(x0, x1 + 1):
        miss = x - xl - 1
        ypred = yl + slope * (x - xl)
        w = min(win + 1.5 * miss, 14) + 3 * abs(slope)
        cand = [c for c in runs(img[:, x], thr) if ylim[0] <= c <= ylim[1] and abs(c - ypred) <= w
                and (below is None or x not in below or c < below[x] - 5)]
        if not cand:
            continue
        c = min(cand, key=lambda c: abs(c - ypred))
        pts[x] = c
        xs = [k for k in sorted(pts) if k >= x - 8]
        if len(xs) >= 3:
            slope = float(np.polyfit(xs, [pts[k] for k in xs], 1)[0])
        xl, yl = x, c
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--check")
    a = p.parse_args()
    img = np.array(Image.open(a.image).convert("L")).astype(float)
    here = os.path.dirname(os.path.abspath(__file__))
    allpts = []
    first = None
    merged = {}
    for k, (name, x0, x1, ys, ylim) in enumerate(SEGMENTS):
        pts = trace(img, x0, x1, ys, ylim, below=first if name != "H2O_propylene" else None)
        if first is None:
            first = pts
        allpts += [(x, y, min(k, 1)) for x, y in pts.items()]
        merged.setdefault(name, {}).update(pts)
    for name, pts in merged.items():
        out = os.path.join(here, f"bertie_1968_iceVI_fig1_{name}.csv")
        with open(out, "w") as fh:
            fh.write("# Digitised from Bertie, Labbe, Whalley, J. Chem. Phys. 49, 2141 (1968), Fig. 1 upper frame,\n"
                     f"# {name.replace('_', ' ')} mull, ice VI. Approximate (~10 cm-1, ~0.01 absorbance).\n")
            fh.write("wavenumber_cm-1,absorbance\n")
            for x in sorted(pts):
                fh.write(f"{nu(x):.1f},{absorb(pts[x]):.4f}\n")
        print(f"{name}: {len(pts)} points, {nu(max(pts)):.0f}-{nu(min(pts)):.0f} cm-1 -> {out}")
    if a.check:
        im = Image.open(a.image).convert("RGB")
        W, H = im.size
        im = im.resize((3 * W, 3 * H), Image.NEAREST)
        d = ImageDraw.Draw(im)
        col = [(230, 0, 0), (0, 140, 255)]
        for x, y, k in allpts:
            d.ellipse([3 * x - 2, 3 * y - 2, 3 * x + 2, 3 * y + 2], fill=col[k])
        im.crop((0, 0, 3 * W, 3 * 320)).save(a.check)


if __name__ == "__main__":
    main()
