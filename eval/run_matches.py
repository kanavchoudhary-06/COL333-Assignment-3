#!/usr/bin/env python3
"""
Run a batch of evaluation matches with a3.py and write the results as JSON + CSV.

Runs on any Linux x86_64 box that has passed `python3 a3.py doctor`
(the cloud VMs set themselves up with eval/setup_vm.sh first).

    python3 eval/run_matches.py --matches 4 --opponents full,baseline
    python3 eval/run_matches.py --shard 0 --shards 3 --out eval/out/results_shard0.json
    python3 eval/run_matches.py --dry          # no server; fake rows, tests the pipeline

Opponent names:
    full       the HELIOS reference team used for grading
    baseline   the unmodified starter (class Baseline in src/student/agent.py)
    idle       Agent2, players that stand still
    <path>[:Class]  any other agent.py
"""
import argparse
import csv
import glob
import json
import os
import queue
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A3 = os.path.join(ROOT, "a3.py")
OPP_SPECS = {"full": "full", "baseline": ":Baseline", "idle": ":Agent2"}
RESULT = re.compile(r"result\s*:\s*(\S+)\s+(\d+)-(\d+)\s+(\S+)\s+\[(.*?)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MINE, OPP = "MINE", "OPP"


def schedule(matches, opponents, seed_base):
    """Every opponent gets `matches` games, alternating which side we play."""
    out = []
    for oi, opp in enumerate(opponents):
        for r in range(matches):
            out.append({"opponent": opp, "round": r,
                        "mine_left": r % 2 == 0,
                        "seed": seed_base + 1000 * oi + r})
    for i, m in enumerate(out):
        m["index"] = i
    return out


def timing(stamp):
    """Our team's CPU spend, cut-offs and forfeits from runlogs/timing."""
    d = os.path.join(ROOT, "runlogs", "timing", stamp)
    spent, killed, broke = 0.0, 0, False
    worst_wall = 0.0
    for p in glob.glob(os.path.join(d, f"{MINE}_*.json")):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        spent = max(spent, r.get("spent_ms", 0.0) or 0.0)
        killed += r.get("killed", 0) or 0
        broke = broke or bool(r.get("broke_at"))
        worst_wall = max([worst_wall] + (r.get("wall_ms") or [0.0]))
    forfeits = len(glob.glob(os.path.join(d, f"FORFEIT_{MINE}_*.json")))
    return {"cpu_spent_ms": round(spent, 1), "cutoffs": killed,
            "budget_exhausted": broke, "forfeits": forfeits,
            "worst_decision_wall_ms": round(worst_wall, 1)}


def play(m, agent_spec, port, half, timeout):
    opp_spec = OPP_SPECS.get(m["opponent"], m["opponent"])
    left, right = (MINE, OPP) if m["mine_left"] else (OPP, MINE)
    lspec, rspec = ((agent_spec, opp_spec) if m["mine_left"]
                    else (opp_spec, agent_spec))
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"_eval{m['index']:03d}"
    cmd = [sys.executable, os.path.join(ROOT, "eval", "a3_patient.py"), "match", left, right,
           "--left", lspec, "--right", rspec,
           "-p", str(port), "--no-trace", "--stamp", stamp]
    # rcssserver 19 has no server seed option; noise (kick_rand) differs per run
    if half:
        cmd += ["-t", str(half)]
    row = dict(m, stamp=stamp, port=port)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        text = ANSI.sub("", p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired as e:
        text = "TIMEOUT " + str(e)
    row["wall_s"] = round(time.time() - t0, 1)
    hit = RESULT.search(text)
    if hit:
        n1, s1, s2, n2, tag = hit.groups()
        score = {n1: int(s1), n2: int(s2)}
        row["gf"], row["ga"] = score.get(MINE, 0), score.get(OPP, 0)
        row["valid"] = tag.strip() == "valid"
        row["problem"] = "" if row["valid"] else tag.strip()
    else:
        row["gf"] = row["ga"] = None
        row["valid"] = False
        row["problem"] = "no result line: " + text.strip()[-300:]
    row.update(timing(stamp))
    return row


def fake(m):
    """--dry: a plausible row without running anything."""
    return dict(m, stamp="dry", port=0, wall_s=0.0, gf=1, ga=1, valid=True,
                problem="", dry=True, cpu_spent_ms=0.0, cutoffs=0,
                budget_exhausted=False, forfeits=0,
                worst_decision_wall_ms=0.0)


def summarize(rows):
    per = {}
    for r in rows:
        s = per.setdefault(r["opponent"], {
            "played": 0, "valid": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0,
            "cutoffs": 0, "forfeits": 0, "budget_exhausted": 0,
            "max_cpu_spent_ms": 0.0, "max_decision_wall_ms": 0.0})
        s["played"] += 1
        s["cutoffs"] += r.get("cutoffs", 0) or 0
        s["forfeits"] += r.get("forfeits", 0) or 0
        s["budget_exhausted"] += int(bool(r.get("budget_exhausted")))
        s["max_cpu_spent_ms"] = max(s["max_cpu_spent_ms"],
                                    r.get("cpu_spent_ms") or 0.0)
        s["max_decision_wall_ms"] = max(s["max_decision_wall_ms"],
                                        r.get("worst_decision_wall_ms") or 0.0)
        if not r.get("valid") or r.get("gf") is None:
            continue
        s["valid"] += 1
        s["GF"] += r["gf"]
        s["GA"] += r["ga"]
        s["W" if r["gf"] > r["ga"] else "D" if r["gf"] == r["ga"] else "L"] += 1
    for s in per.values():
        s["GD"] = s["GF"] - s["GA"]
        s["points_per_game"] = round((3 * s["W"] + s["D"]) / s["valid"], 2) \
            if s["valid"] else 0.0
    return per


def print_summary(per):
    print(f"{'opponent':<12}{'valid':>6}{'W':>4}{'D':>4}{'L':>4}"
          f"{'GF':>5}{'GA':>5}{'GD':>5}{'ppg':>6}{'cuts':>6}"
          f"{'forf':>6}{'maxCPU s':>10}")
    for opp, s in sorted(per.items()):
        print(f"{opp:<12}{s['valid']:>3}/{s['played']:<2}{s['W']:>4}{s['D']:>4}"
              f"{s['L']:>4}{s['GF']:>5}{s['GA']:>5}{s['GD']:>5}"
              f"{s['points_per_game']:>6}{s['cutoffs']:>6}{s['forfeits']:>6}"
              f"{s['max_cpu_spent_ms'] / 1000.0:>10.1f}")


def write(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    data = {"rows": rows, "summary": summarize(rows),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "job": os.environ.get("CDEPLOY_JOB", ""),
            "run_id": os.environ.get("CDEPLOY_RUN_ID", "")}
    with open(out + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(out + ".tmp", out)
    cols = ["index", "opponent", "round", "mine_left", "seed", "gf", "ga",
            "valid", "problem", "cpu_spent_ms", "cutoffs", "forfeits",
            "budget_exhausted", "worst_decision_wall_ms", "wall_s", "stamp"]
    with open(os.path.splitext(out)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--matches", type=int, default=4,
                    help="games per opponent (sides alternate)")
    ap.add_argument("--opponents", default="full,baseline")
    ap.add_argument("--agent", default=":Agent1",
                    help="our team spec for a3.py (default :Agent1)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=0,
                    help="matches at once (0 = cpu_count / 2)")
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--half", type=int, default=0,
                    help="override half length (0 = base.conf)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "out",
                                                  "results.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    opps = [o.strip() for o in a.opponents.split(",") if o.strip()]
    todo = [m for m in schedule(a.matches, opps, a.seed_base)
            if m["index"] % a.shards == a.shard]
    par = a.parallel or max(1, (os.cpu_count() or 2) // 2)
    print(f"shard {a.shard}/{a.shards}: {len(todo)} matches, {par} at a time")

    rows = []
    if a.dry:
        rows = [fake(m) for m in todo]
    else:
        ports = queue.Queue()
        for k in range(par):
            ports.put(6000 + 10 * k)          # each match uses port..port+2

        def job(m):
            port = ports.get()
            try:
                return play(m, a.agent, port, a.half, a.timeout)
            finally:
                ports.put(port)

        with ThreadPoolExecutor(max_workers=par) as ex:
            for r in ex.map(job, todo):
                rows.append(r)
                print(f"  #{r['index']:03d} vs {r['opponent']:<9} "
                      f"{r['gf']}-{r['ga']}  valid={r['valid']} "
                      f"cuts={r['cutoffs']} {r['wall_s']}s", flush=True)
                write(rows, a.out)          # partial results survive a crash
    write(rows, a.out)
    print_summary(summarize(rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
