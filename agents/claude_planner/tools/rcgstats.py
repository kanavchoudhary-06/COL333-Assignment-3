#!/usr/bin/env python3
"""Summarise rcg logs: score, possession, territory, kicks, shots, tackles per team.

usage: rcgstats.py TEAM file.rcg [...]     (TEAM = our team name)
"""
import re, sys
from collections import defaultdict

RE_SHOW = re.compile(r"^\(show\s+(-?\d+)\s+(.*)\)\s*$")
RE_TEAM = re.compile(r'^\(team\s+-?\d+\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)')
RE_PM = re.compile(r"^\(playmode\s+(-?\d+)\s+(\S+?)\)")
RE_BALL = re.compile(r"\(\(b\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\)")
RE_P = re.compile(r"\(\((l|r)\s+(\d+)\)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
RE_C = re.compile(r"\(c\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
RE_S = re.compile(r"\(s\s+(\S+)")

def analyse(path, us):
    left = right = None
    score = (0, 0)
    pm = "before_kick_off"
    prev_kick = {}
    prev_tackle = {}
    st = defaultdict(lambda: defaultdict(float))
    last_kicker = None
    events = []
    goals = []
    kick_hist = []
    n_play = 0
    stamina_min = defaultdict(lambda: 9999)
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            m = RE_TEAM.match(line)
            if m:
                left, right = m.group(1), m.group(2)
                score = (int(m.group(3)), int(m.group(4)))
            continue
        if line.startswith("(playmode"):
            m = RE_PM.match(line)
            if m:
                pm = m.group(2)
                if pm in ("goal_l", "goal_r"):
                    side = pm[5]
                    goals.append((int(m.group(1)), side, list(kick_hist[-4:])))
            continue
        m = RE_SHOW.match(line.rstrip("\n"))
        if not m:
            continue
        cyc = int(m.group(1)); body = m.group(2)
        mb = RE_BALL.search(body)
        bx, by, bvx, bvy = map(float, mb.groups())
        players = {}
        for chunk in body.split("((")[1:]:
            mp = RE_P.match("((" + chunk)
            if not mp:
                continue
            side, unum, state, x, y, vx, vy, bd, nk = mp.groups()
            mc = RE_C.search(chunk)
            ms = RE_S.search(chunk)
            players[(side, int(unum))] = (float(x), float(y), int(mc.group(1)) if mc else 0,
                                          int(mc.group(9)) if mc else 0,
                                          float(ms.group(1)) if ms else 0)
        if pm != "play_on":
            for k, p in players.items():
                prev_kick[k] = p[2]; prev_tackle[k] = p[3]
            continue
        n_play += 1
        # territory from left POV
        st["l"]["ballx"] += bx
        st["r"]["ballx"] += -bx
        # nearest player to ball
        best = min(players.items(), key=lambda kv: (kv[1][0]-bx)**2 + (kv[1][1]-by)**2)
        d = ((best[1][0]-bx)**2 + (best[1][1]-by)**2) ** 0.5
        if d < 1.2:
            st[best[0][0]]["control"] += 1
        for k, p in players.items():
            stamina_min[k] = min(stamina_min[k], p[4])
            if k in prev_kick and p[2] > prev_kick[k]:
                st[k[0]]["kicks"] += 1
                # shot = ball heading to the opponent goal fast after the kick
                kick_hist.append((cyc, k, round(p[0], 1), round(p[1], 1)))
                if last_kicker and last_kicker != k:
                    if last_kicker[0] == k[0]:
                        st[k[0]]["passok"] += 1
                    else:
                        st[last_kicker[0]]["lost"] += 1
                last_kicker = k
            if k in prev_tackle and p[3] > prev_tackle[k]:
                st[k[0]]["tackles"] += 1
            prev_kick[k] = p[2]; prev_tackle[k] = p[3]
        # shots: ball fast and heading into goal mouth
        sp = (bvx**2 + bvy**2) ** 0.5
        if sp > 1.5 and last_kicker:
            for side, gx in (("l", 52.5), ("r", -52.5)):
                if (gx - bx) * bvx > 0 and abs(gx - bx) < 30:
                    t = (gx - bx) / bvx
                    yy = by + bvy * t
                    # does the ball still reach?
                    dist = abs(gx - bx) / max(1e-6, abs(bvx) / sp)
                    reach = sp / 0.06
                    if abs(yy) < 7.5 and reach > dist and last_kicker[0] == side:
                        key = (side, cyc // 15)
                        if key not in st[side].get("_shotkeys", set()):
                            st[side].setdefault("_shotkeys", set()).add(key)
    res = {}
    for side, name in (("l", left), ("r", right)):
        s = st[side]
        res[name] = dict(side=side, ballx=s["ballx"] / max(1, n_play), control=100*s["control"]/max(1,n_play),
                         kicks=int(s["kicks"]), tackles=int(s["tackles"]),
                         shots=len(s.get("_shotkeys", ())), passok=int(s["passok"]), lost=int(s["lost"]))
    sc = {left: score[0], right: score[1]}
    return sc, res, goals, left, right, stamina_min

if __name__ == "__main__":
    us = sys.argv[1]
    tot = defaultdict(float)
    for path in sys.argv[2:]:
        sc, res, goals, left, right, stmin = analyse(path, us)
        them = right if left == us else left
        r, o = res.get(us, {}), res.get(them, {})
        print(f"{path.split('/')[-1][:60]:60s} {us} {sc.get(us)}-{sc.get(them)} | ctrl {r.get('control',0):4.1f}/{o.get('control',0):4.1f} "
              f"ballx {r.get('ballx',0):5.1f} pass {r.get('passok')}/{o.get('passok')} lost {r.get('lost')}/{o.get('lost')} kicks {r.get('kicks')}/{o.get('kicks')} shots {r.get('shots')}/{o.get('shots')} tackles {r.get('tackles')}/{o.get('tackles')}")
        for k in ("control", "ballx", "kicks", "shots", "tackles", "passok", "lost"):
            tot["us_" + k] += r.get(k, 0); tot["them_" + k] += o.get(k, 0)
        tot["n"] += 1
        if "-v" in sys.argv[0:1]:
            pass
    n = max(1, tot["n"])
    print(f"AVG ctrl {tot['us_control']/n:.1f}/{tot['them_control']/n:.1f} ballx {tot['us_ballx']/n:.1f} "
          f"kicks {tot['us_kicks']/n:.0f}/{tot['them_kicks']/n:.0f} shots {tot['us_shots']/n:.1f}/{tot['them_shots']/n:.1f} "
          f"tackles {tot['us_tackles']/n:.1f}/{tot['them_tackles']/n:.1f} passok {tot['us_passok']/n:.1f}/{tot['them_passok']/n:.1f} lost {tot['us_lost']/n:.1f}/{tot['them_lost']/n:.1f}")
