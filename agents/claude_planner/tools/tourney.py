#!/usr/bin/env python3
"""Variant tournament. Freezes src/student/agent.py into one file, appends
variant subclasses + a Starter opponent, then plays the listed fixtures.

usage: tourney.py TAG P FIXTURES_JSON
FIXTURES_JSON: list of [left_class, right_spec, repeats]; right_spec is a
class name in the frozen file or "full".
"""
import json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

REPO = "/home/user/repo"
PY = os.path.join(REPO, ".venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
tag, P, fixtures = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
variants = json.loads(open(os.path.join(HERE, "variants.json")).read())

snap = os.path.join(HERE, "snapshots", tag)
os.makedirs(snap, exist_ok=True)
fpath = os.path.join(snap, "agent.py")
shutil.copy(os.path.join(REPO, "src/student/agent.py"), fpath)
with open(fpath, "a") as fh:
    fh.write("\n\nclass Starter(StarterAgent):\n    pass\n")
    for name, attrs in variants.items():
        body = "".join(f"    {k} = {v!r}\n" for k, v in attrs.items()) or "    pass\n"
        fh.write(f"\n\nclass {name}(Agent1):\n{body}")

RES = re.compile(r"result\s*:\s*(\S+)\s+(\d+)-(\d+)\s+(\S+)\s+\[(.*)\]")
jobs = []
for left, right, rep in fixtures:
    for r in range(rep):
        jobs.append((left, right, r))

def short(n):
    return n[:12]

def one(i, job):
    left, right, r = job
    port = 6000 + 10 * (i % P)
    ln = "L" + short(left)
    rn = "R" + short(right.split(":")[-1] if right != "full" else "FULL")
    for attempt in range(3):
        stamp = f"{tag}_{i:03d}_{left}_{right.split(':')[-1]}_{attempt}"
        cmd = [PY, "a3.py", "match", ln, rn, "-p", str(port), "--stamp", stamp,
               "--no-trace", "--left", f"{fpath}:{left}",
               "--right", "full" if right == "full" else (right if ":" in right else f"{fpath}:{right}")]
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout
        m = RES.search(out)
        if m and "NO_SHOW_DATA" not in out and "EXCEPTION" not in out:
            a, sa, sb, b, flag = m.groups()
            sc = {a: int(sa), b: int(sb)}
            return dict(left=left, right=right, gl=sc.get(ln, -1), gr=sc.get(rn, -1), flag=flag)
        time.sleep(5)
    return dict(left=left, right=right, gl=-1, gr=-1, flag="FAILED")

results = []
with ThreadPoolExecutor(P) as ex:
    futs = []
    for i, job in enumerate(jobs):
        futs.append(ex.submit(one, i, job))
        if i < P:
            time.sleep(20)
    for f in futs:
        r = f.result()
        results.append(r)
        print(f"{r['left']:>12} {r['gl']}-{r['gr']} {r['right']:<12} {r['flag']}", flush=True)
json.dump(results, open(os.path.join(HERE, "results", f"{tag}.json"), "w"), indent=1)
