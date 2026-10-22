#!/usr/bin/env python3
"""Calibration of on-ball success probabilities. usage: calib.py TAG [TEAM]
For each traced kick decision (pass/dribble/shoot) find who touches the ball
next in the rcg: same team = success."""
import json, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcgstats import RE_SHOW, RE_TEAM, RE_PM, RE_P, RE_C
from collections import defaultdict

def touches(path, us):
    left = right = None; prev = {}; out = []; pm = None
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            m = RE_TEAM.match(line); left, right = m.group(1), m.group(2); continue
        if line.startswith("(playmode"):
            m = RE_PM.match(line); pm = m.group(2)
            out.append((int(m.group(1)), "mode", pm)); continue
        m = RE_SHOW.match(line.rstrip("\n"))
        if not m: continue
        cyc = int(m.group(1))
        for chunk in m.group(2).split("((")[1:]:
            mp = RE_P.match("((" + chunk)
            if not mp: continue
            side, unum = mp.group(1), int(mp.group(2))
            mc = RE_C.search(chunk)
            if not mc: continue
            k = (side, unum); kc, tc = int(mc.group(1)), int(mc.group(9))
            if k in prev and (kc > prev[k][0] or tc > prev[k][1]):
                team = left if side == "l" else right
                out.append((cyc, "us" if team == us else "them", unum))
            prev[k] = (kc, tc)
    return out

tag = sys.argv[1]; team = sys.argv[2] if len(sys.argv) > 2 else "ME"
bins = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for tdir in sorted(glob.glob(f"/home/user/repo/runlogs/trace/{tag}*")):
    stamp = os.path.basename(tdir)
    rcg = f"/home/user/repo/logs/{stamp}.rcg"
    if not os.path.exists(rcg): continue
    ev = touches(rcg, team)
    dec = []
    for f in glob.glob(f"{tdir}/{team}/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("kickable") and r.get("notes"):
                n = r["notes"][0]
                if n.get("note") in ("pass", "dribble", "shoot", "shoot1") and n.get("p") is not None:
                    dec.append((r["c"], r["unum"], n["note"], n["p"]))
    dec.sort()
    for c, unum, kind, p in dec:
        # next touch after c by someone other than this player (or same player for dribble)
        nxt = None
        for e in ev:
            if e[0] <= c: continue
            if e[1] == "mode":
                nxt = ("mode", e[2]); break
            if e[1] == "us" and e[2] == unum and kind != "dribble" and e[0] <= c + 3:
                continue  # multi-touch kick by the same player
            if e[1] == "us" and e[2] == unum and kind == "dribble" and e[0] <= c + 1:
                continue
            nxt = e; break
        if nxt is None: continue
        if kind.startswith("shoot"):
            ok = nxt[0] == "mode" and nxt[1] in ("goal_l", "goal_r")
        else:
            ok = nxt[1] == "us" if nxt[0] != "mode" else False
        b = min(4, int(p * 5))
        bins[kind][b][0] += ok; bins[kind][b][1] += 1
for kind in bins:
    print(kind)
    for b in sorted(bins[kind]):
        s, n = bins[kind][b]
        print(f"   p in [{b/5:.1f},{(b+1)/5:.1f}): {s}/{n} = {s/max(1,n):.2f}")
