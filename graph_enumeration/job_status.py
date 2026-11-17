#!/usr/bin/env python3
"""Progress and time estimates for the CRYSTAL opt + freq jobs of a topology set.

For every config_XXXXX/{opt,freq}/<name>.out it reports the state
(queued / running / converged / finished / failed), optimisation steps,
frequency displacements done of 3 x (irreducible atoms), elapsed wall time,
and an estimate of the remaining time from the job's own pace.

The estimate for jobs that have not reached a stage yet uses the median pace
of the jobs of the same set that did (seconds per optimisation step, seconds
per frequency displacement, number of optimisation steps).

    python job_status.py ../crystal_calcs/iceih/topologies/hsesol_d3
"""
import argparse
import csv
import datetime as dt
import glob
import os
import re
import statistics as st

TL = re.compile(r"T{10,} (\w+)\s+TELAPSE\s+([\d.]+)")


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="replace").read()
    d = {"text": txt}
    m = re.search(r"EEEEEEEEEE STARTING\s+DATE\s+(\d+) (\d+) (\d+) TIME (\d+):(\d+):(\d+)", txt)
    d["start"] = dt.datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)), int(m.group(4)),
                             int(m.group(5)), int(m.group(6))) if m else None
    tl = TL.findall(txt)
    d["elapsed"] = float(tl[-1][1]) if tl else 0.0
    d["terminated"] = "EEEEEEEEEE TERMINATION" in txt
    d["error"] = bool(re.search(r"^ ERROR \*\*\*\*", txt, re.M)) and not d["terminated"]
    err = re.findall(r"^ ERROR \*\*\*\*.*$", txt, re.M)
    d["error_msg"] = err[-1].strip() if err else ""
    slurm = "".join(open(x, errors="replace").read() for x in glob.glob(os.path.join(os.path.dirname(path), "slurm-*.out")))
    d["crashed"] = ("SCF abnormal end" in slurm or "lost connection" in slurm) and "EEEEEEEEEE STARTING" not in txt
    if "lost connection" in slurm and "EEEEEEEEEE STARTING" not in txt:
        d["error_msg"] = "never started (scratch copy failed, SSH)"
    ops = re.findall(r"NUMBER OF SYMMETRY OPERATORS\s*:\s*(\d+)", txt)
    d["ops"] = int(ops[0]) if ops else None
    irr = re.findall(r"ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)", txt) or \
        re.findall(r"NUMBER OF IRREDUCIBLE ATOMS IN THE CELL:\s*(\d+)", txt)
    d["irr"] = int(irr[0]) if irr else None
    return d


def opt_info(d):
    txt = d["text"]
    steps = len(re.findall(r"T{10,} OPTI\s+TELAPSE", txt))
    conv = "OPT END - CONVERGED" in txt
    state = "converged" if conv else ("failed" if d["terminated"] or d["error"] or d.get("crashed") else "running")
    return {"state": state, "steps": steps, "elapsed": d["elapsed"],
            "s_per_step": d["elapsed"] / max(steps, 1) if steps else None}


def freq_info(d):
    txt = d["text"]
    ndisp = len(re.findall(r"^\s+\d+ [A-Z][a-z]?\s+D[XYZ]\s+[\d.E+-]+\s+-?[\d.]+E", txt, re.M))
    total = 3 * d["irr"] if d["irr"] else None
    done = d["terminated"] and "FREQUENCIES" in txt
    state = "finished" if done else ("failed" if d["terminated"] or d["error"] or d.get("crashed") else "running")
    return {"state": state, "ndisp": ndisp, "total": total, "elapsed": d["elapsed"],
            "s_per_disp": d["elapsed"] / (ndisp + 1) if ndisp else None}


def fmt(sec):
    if sec is None:
        return "?"
    h = sec / 3600
    return f"{h:5.1f} h" if h < 48 else f"{h / 24:5.1f} d"


def git_time(path):
    """commit time of the newest commit touching path (= time the outputs were copied)"""
    import subprocess
    try:
        out = subprocess.run(["git", "log", "-1", "--no-merges", "--format=%ai", "--", "."], cwd=path,
                             capture_output=True, text=True).stdout.strip()
        return dt.datetime.strptime(out[:19], "%Y-%m-%d %H:%M:%S") if out else None
    except Exception:
        return None


def collect(jobs):
    rows = list(csv.DictReader(open(os.path.join(jobs, "jobs.csv"))))
    recs = []
    for r in rows:
        o = parse(os.path.join(jobs, r["directory"], "opt", r["input"] + ".out"))
        f = parse(os.path.join(jobs, r["directory"], "freq", r["input"] + ".out"))
        irr = (o or {}).get("irr") or (f or {}).get("irr") or int(r.get("atoms_asym_unit") or 0)
        ops = (o or {}).get("ops") or (f or {}).get("ops")
        recs.append({"row": r, "o": o, "f": f, "oi": opt_info(o) if o else None,
                     "fi": freq_info(f) if f else None, "irr": irr, "ops": ops})
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", nargs="+")
    p.add_argument("--now", help="snapshot time 'YYYY-MM-DD HH:MM' (default: time of the last git commit "
                                 "touching the job directory, i.e. when the outputs were copied)")
    a = p.parse_args()
    sets = {j: collect(j) for j in a.jobs}

    # calibration over all sets: freq displacement cost / (opt step cost x symmetry ops)
    cal = []
    for recs in sets.values():
        for x in recs:
            oi, fi = x["oi"], x["fi"]
            if oi and fi and fi["state"] == "finished" and oi["s_per_step"] and x["ops"]:
                cal.append(fi["elapsed"] / (fi["ndisp"] + 1) / oi["s_per_step"] / x["ops"])
    per_op = st.median(cal) if cal else 0.9

    for jobs, recs in sets.items():
        now = dt.datetime.strptime(a.now, "%Y-%m-%d %H:%M") if a.now else (git_time(jobs) or dt.datetime.now())
        conv = [x["oi"]["steps"] for x in recs if x["oi"] and x["oi"]["state"] == "converged"]
        nsteps = st.median(conv) if conv else 35
        pace = {}
        for x in recs:
            if x["oi"] and x["oi"]["s_per_step"] and x["oi"]["state"] != "failed":
                pace.setdefault(x["irr"], []).append(x["oi"]["s_per_step"])
        allp = [v for vs in pace.values() for v in vs]

        def step_time(x):
            if x["oi"] and x["oi"]["s_per_step"]:
                return x["oi"]["s_per_step"]
            v = pace.get(x["irr"]) or allp
            return st.median(v) if v else None

        def ops_of(x):
            if x["ops"]:
                return x["ops"]
            same = [y["ops"] for y in recs if y["irr"] == x["irr"] and y["ops"]]
            return st.median(same) if same else 1

        print(f"\n=== {jobs}   (snapshot {now.strftime('%d.%m.%Y %H:%M')})")
        print(f"    reference: {nsteps:.0f} optimisation steps (median of converged); one freq displacement "
              f"≈ {per_op:.2f} × (opt step) × (symmetry ops), since displaced geometries lose their symmetry")
        print(f"{'id':>4} {'SG':11} {'irr':>3} {'ops':>3}  {'optimisation':27}  {'frequencies':27}  "
              f"{'remaining':>9}  {'expected end':>12}")
        counts, ends = {}, []
        for x in recs:
            r, o, f, oi, fi, irr = x["row"], x["o"], x["f"], x["oi"], x["fi"], x["irr"]
            nops = ops_of(x)
            ts = step_time(x)
            disp = ts * per_op * nops if ts else None
            ndisp_tot = 3 * irr
            rem = 0.0
            # --- optimisation
            if oi is None:
                os_, state = "queued", "queued"
                rem += nsteps * ts if ts else 0
                rem += (ndisp_tot + 1) * disp if disp else 0
            elif oi["state"] == "failed":
                os_, state = "FAILED: " + (o["error_msg"] or "see output")[:19], "failed"
            elif oi["state"] == "running":
                run = (now - o["start"]).total_seconds() if o["start"] else oi["elapsed"]
                os_ = f"running  {oi['steps']:3d} st  {fmt(run)}"
                state = "opt running"
                if ts:
                    left = max(nsteps - oi["steps"], 3) * ts - max(run - oi["elapsed"], 0)
                    rem += max(left, 0.5 * ts) + (ndisp_tot + 1) * disp
            else:
                os_ = f"done     {oi['steps']:3d} st  {fmt(oi['elapsed'])}"
                state = "opt done"
            # --- frequencies
            if state in ("queued", "failed") or (oi and oi["state"] == "running"):
                fs = "-"
            elif fi is None:
                fs, state = "pending", "freq pending"
                rem += (ndisp_tot + 1) * disp if disp else 0
            elif fi["state"] == "finished":
                fs, state = f"done    {fi['ndisp']:3d}/{fi['total']}  {fmt(fi['elapsed'])}", "done"
            elif fi["state"] == "failed":
                fs, state = "FAILED: " + (f["error_msg"] or "see output")[:19], "failed"
            else:
                run = (now - f["start"]).total_seconds() if f["start"] else fi["elapsed"]
                fs, state = f"running {fi['ndisp']:3d}/{fi['total']}  {fmt(run)}", "freq running"
                rem += ((fi["total"] or ndisp_tot) - fi["ndisp"]) * run / (fi["ndisp"] + 1)
            counts[state] = counts.get(state, 0) + 1
            end = now + dt.timedelta(seconds=rem) if rem and state not in ("queued", "failed") else None
            if end:
                ends.append(end)
            rs = fmt(rem) if rem else "-"
            if state == "queued":
                rs = "+" + fmt(rem).strip()
            print(f"{r['id']:>4} {r['space_group']:11} {irr:3d} {nops:3.0f}  {os_:27}  {fs:27}  {rs:>9}  "
                  f"{end.strftime('%d.%m. %H:%M') if end else '':>12}")
        print("    summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        if ends:
            print(f"    last running job expected to finish: {max(ends).strftime('%d.%m.%Y %H:%M')}"
                  "  (queued jobs: '+' = run time once they start)")


if __name__ == "__main__":
    main()
