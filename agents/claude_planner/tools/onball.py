#!/usr/bin/env python3
"""On-ball decision summary from traces. usage: onball.py TAGPREFIX [TEAM]"""
import json, glob, sys
from collections import Counter, defaultdict
tag = sys.argv[1]; team = sys.argv[2] if len(sys.argv) > 2 else "ME"
kinds = Counter(); zone = defaultdict(Counter); stuck = 0; cont = 0; n = 0
shot_seen = []
for d in sorted(glob.glob(f"/home/user/repo/runlogs/trace/{tag}*/{team}")):
    for f in glob.glob(d + "/*.jsonl"):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        run = []
        for r in rows:
            if not r.get("kickable") or not r.get("notes"):
                run = []; continue
            nt = r["notes"][0]; k = nt.get("note"); n += 1
            kinds[k] += 1
            x = r["pos"][0]
            z = "def" if x < -17 else ("mid" if x < 17 else "att")
            zone[z][k] += 1
            cont += nt.get("cont", 0) or 0
            if nt.get("ps") is not None and x > 25:
                shot_seen.append(nt.get("ps"))
            if run and abs(run[-1]["pos"][0] - r["pos"][0]) < 0.3 and abs(run[-1]["pos"][1] - r["pos"][1]) < 0.3 and run[-1]["c"] == r["c"] - 1:
                run.append(r)
                if len(run) == 5:
                    stuck += 1
            else:
                run = [r]
print(f"on-ball decisions {n}: {dict(kinds.most_common())}")
for z in ("def", "mid", "att"):
    print(f"  {z}: {dict(zone[z].most_common())}")
print(f"continuations {cont}, stuck runs (>=5 cycles in place) {stuck}")
if shot_seen:
    import statistics
    print(f"best shot p seen in att third: n={len(shot_seen)} mean={statistics.mean(shot_seen):.2f} >0.3: {sum(p>0.3 for p in shot_seen)} >0.5: {sum(p>0.5 for p in shot_seen)}")
