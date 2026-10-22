#!/usr/bin/env python3
"""Shot quality: for each shot (fast kick towards the goal mouth) report the
shooter's distance to goal and the closest defender (excluding goalie).
usage: shots.py TEAM_A file.rcg ...   (prints shots BY and AGAINST TEAM_A)"""
import re, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcgstats import RE_SHOW, RE_TEAM, RE_PM, RE_BALL, RE_P, RE_C
from collections import defaultdict

def run(path, us, agg):
    left = right = None; prev = {}; pm = None; pending = None
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            m = RE_TEAM.match(line); left, right = m.group(1), m.group(2); continue
        if line.startswith("(playmode"):
            m = RE_PM.match(line); pm = m.group(2)
            if pending and pm in ("goal_l", "goal_r"):
                pending["goal"] = True
            continue
        m = RE_SHOW.match(line.rstrip("\n"))
        if not m: continue
        cyc = int(m.group(1)); body = m.group(2)
        mb = RE_BALL.search(body)
        bx, by, bvx, bvy = map(float, mb.groups())
        pl = {}
        kicked = None
        for chunk in body.split("((")[1:]:
            mp = RE_P.match("((" + chunk)
            if not mp: continue
            side, unum, state, x, y = mp.groups()[:5]
            st = int(state, 16)
            pl[(side, int(unum))] = (float(x), float(y), bool(st & 0x8))
            mc = RE_C.search(chunk)
            if mc:
                k = (side, int(unum)); kc = int(mc.group(1))
                if k in prev and kc > prev[k]:
                    kicked = k
                prev[k] = kc
        if pm != "play_on" or kicked is None: continue
        sp = math.hypot(bvx, bvy)
        side = kicked[0]
        gx = 52.5 if side == "l" else -52.5
        if sp < 2.0 or (gx - bx) * bvx <= 0 or abs(gx - bx) > 35: continue
        t = (gx - bx) / bvx
        if abs(by + bvy * t) > 8.0: continue
        sx, sy, _ = pl[kicked]
        dgoal = math.hypot(gx - sx, sy)
        oth = "r" if side == "l" else "l"
        dd = min((math.hypot(x - sx, y - sy) for (s2, u), (x, y, g) in pl.items() if s2 == oth and not g), default=99)
        team = left if side == "l" else right
        rec = dict(dist=dgoal, def_d=dd, goal=False)
        if pending: agg[pending["team"]].append(pending)
        pending = dict(team="us" if team == us else "them", **rec)
    if pending: agg[pending["team"]].append(pending)

us = sys.argv[1]
agg = defaultdict(list)
for p in sys.argv[2:]:
    run(p, us, agg)
n = len(sys.argv) - 2
for who in ("us", "them"):
    L = agg[who]
    if not L: print(who, "no shots"); continue
    g = sum(r["goal"] for r in L)
    print(f"{who}: {len(L)/n:.1f} shots/match, {g/n:.1f} goals/match, conv {g/len(L):.2f}, "
          f"mean dist {sum(r['dist'] for r in L)/len(L):.1f}m, mean nearest defender {sum(min(r['def_d'],15) for r in L)/len(L):.1f}m")
    for lo, hi in ((0, 12), (12, 18), (18, 25), (25, 40)):
        S = [r for r in L if lo <= r["dist"] < hi]
        if S: print(f"    {lo:2d}-{hi:2d}m: {len(S):3d} shots, {sum(r['goal'] for r in S)} goals, defender {sum(min(r['def_d'],15) for r in S)/len(S):.1f}m")
