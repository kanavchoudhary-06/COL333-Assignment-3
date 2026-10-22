#!/usr/bin/env python3
"""Print the kick sequence before each goal. usage: goals.py OURTEAM file.rcg..."""
import re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from rcgstats import RE_SHOW, RE_TEAM, RE_PM, RE_BALL, RE_P, RE_C

def run(path, us):
    left = right = None
    pm = None
    prev = {}
    kicks = []
    frames = {}
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            m = RE_TEAM.match(line)
            if m: left, right = m.group(1), m.group(2)
            continue
        if line.startswith("(playmode"):
            m = RE_PM.match(line)
            if m:
                pm = m.group(2)
                cyc = int(m.group(1))
                if pm in ("goal_l", "goal_r"):
                    side = pm[5]
                    scorer = left if side == "l" else right
                    tag = "WE SCORED" if scorer == us else "CONCEDED"
                    print(f"  -- {tag} at {cyc} --")
                    for k in [k for k in kicks if k[0] > cyc - 120][-8:]:
                        c, (sd, un), px, py, bx, by, kind = k
                        team = left if sd == "l" else right
                        who = "us " if team == us else "opp"
                        # flip to our POV: we attack +x
                        flip = -1 if (us == right) else 1
                        print(f"     {c:5d} {who}#{un} {kind:6s} at ({px*flip:6.1f},{py*flip:6.1f})")
            continue
        m = RE_SHOW.match(line.rstrip("\n"))
        if not m: continue
        cyc = int(m.group(1)); body = m.group(2)
        mb = RE_BALL.search(body)
        bx, by = float(mb.group(1)), float(mb.group(2))
        for chunk in body.split("((")[1:]:
            mp = RE_P.match("((" + chunk)
            if not mp: continue
            side, unum, state, x, y = mp.groups()[:5]
            mc = RE_C.search(chunk)
            if not mc: continue
            key = (side, int(unum))
            kc, tc = int(mc.group(1)), int(mc.group(9))
            if key in prev:
                if kc > prev[key][0]:
                    kicks.append((cyc, key, float(x), float(y), bx, by, "kick"))
                if tc > prev[key][1]:
                    kicks.append((cyc, key, float(x), float(y), bx, by, "tackle"))
            prev[key] = (kc, tc)

us = sys.argv[1]
for p in sys.argv[2:]:
    print(p.split('/')[-1])
    run(p, us)
