#!/usr/bin/env python3
"""Run N matches (ours vs FULL) with P in parallel; print per-match and aggregate score.

usage: runmatches.py N P [--left SPEC] [--tag TAG] [--base-port 6000] [extra a3 args]
"""
import argparse, os, re, subprocess, sys, time, json
from concurrent.futures import ThreadPoolExecutor

REPO = "/home/user/repo"
PY = os.path.join(REPO, ".venv/bin/python")

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("p", type=int)
ap.add_argument("--left", default=None)
ap.add_argument("--right", default="full")
ap.add_argument("--tag", default="run")
ap.add_argument("--base-port", type=int, default=6000)
ap.add_argument("--trace", action="store_true")
ap.add_argument("--name", default="ME")
ap.add_argument("--opp", default="FULL")
ap.add_argument("--stagger", type=float, default=20.0)
ap.add_argument("--variant", default=None, help="NAME:ATTR=val,ATTR=val")
a, extra = ap.parse_known_args()

# freeze our agent for this batch so later edits do not leak in
if a.left is None:
    import shutil
    snap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots", a.tag)
    os.makedirs(snap, exist_ok=True)
    shutil.copy(os.path.join(REPO, "src/student/agent.py"), os.path.join(snap, "agent.py"))
    a.left = os.path.join(snap, "agent.py") + ":Agent1"
    if a.variant:
        vname, _, assigns = a.variant.partition(":")
        body = "".join(f"    {kv.split('=')[0]} = {kv.split('=', 1)[1]}\n" for kv in assigns.split(",") if kv)
        with open(os.path.join(snap, "agent.py"), "a") as fh:
            fh.write(f"\n\nclass {vname}(Agent1):\n{body or '    pass'}\n")
        a.left = os.path.join(snap, "agent.py") + ":" + vname

RES = re.compile(r"result\s*:\s*(\S+)\s+(\d+)-(\d+)\s+(\S+)\s+\[(.*)\]")
PLAN = re.compile(r"planning\s*:\s*(\S+)\s+(\d+)% of")

def one(i):
    port = a.base_port + 10 * (i % a.p)
    stamp = f"{a.tag}_{i:03d}_{time.strftime('%H%M%S')}"
    cmd = [PY, "a3.py", "match", a.name, a.opp, "-p", str(port), "--stamp", stamp]
    if a.left:
        cmd += ["--left", a.left]
    if a.right:
        cmd += ["--right", a.right]
    if not a.trace:
        cmd += ["--no-trace"]
    cmd += extra
    for attempt in range(3):
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout
        m = RES.search(out)
        if m and "NO_SHOW_DATA" not in out and "only" not in out.split("result")[0][-200:]:
            break
        time.sleep(5)
        stamp = stamp + "r"
        cmd[cmd.index("--stamp") + 1] = stamp
    if not m:
        return dict(i=i, stamp=stamp, ok=False, raw=out[-800:])
    l, ls, rs, r, tag = m.groups()
    sc = {l: int(ls), r: int(rs)}
    plans = {t: int(p) for t, p in PLAN.findall(out)}
    return dict(i=i, stamp=stamp, ok="valid" in tag and "INVALID" not in tag,
                tag=tag, us=sc.get(a.name, -1), them=sc.get(a.opp, -1), plan=plans)

results = []
t0 = time.time()
with ThreadPoolExecutor(a.p) as ex:
    futs = []
    for i in range(a.n):
        futs.append(ex.submit(one, i))
        if i < a.p:
            time.sleep(a.stagger)
    for f in futs:
        r = f.result()
        results.append(r)
        if r.get("us") is None:
            print("FAILED", r, flush=True)
        else:
            print(f"#{r['i']:02d} {a.name} {r['us']}-{r['them']} {a.opp}  {r['tag']}  plan={r['plan']}", flush=True)

good = [r for r in results if r.get("us", -1) >= 0]
w = sum(r["us"] > r["them"] for r in good)
d = sum(r["us"] == r["them"] for r in good)
l = sum(r["us"] < r["them"] for r in good)
gf = sum(r["us"] for r in good); ga = sum(r["them"] for r in good)
print(f"== {a.tag}: {len(good)} matches  W{w} D{d} L{l}  goals {gf}-{ga}  "
      f"({time.time()-t0:.0f}s)")
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
json.dump(results, open(os.path.join(os.path.dirname(__file__), "results", f"{a.tag}.json"), "w"), indent=1)
