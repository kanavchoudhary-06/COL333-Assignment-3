#!/usr/bin/env python3
"""League table over tournament result files. usage: table.py t1 t2 ..."""
import json, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
tab = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))  # W D L GF GA per opponent group
for tag in sys.argv[1:]:
    for r in json.load(open(os.path.join(HERE, "results", f"{tag}.json"))):
        if r["gl"] < 0:
            continue
        for me, opp, gf, ga in ((r["left"], r["right"], r["gl"], r["gr"]), (r["right"], r["left"], r["gr"], r["gl"])):
            if me in ("full", "Starter"):
                continue
            grp = "FULL" if opp == "full" else ("Starter" if opp == "Starter" else "peers")
            t = tab[me][grp]
            t[0] += gf > ga; t[1] += gf == ga; t[2] += gf < ga; t[3] += gf; t[4] += ga
def fmt(t):
    n = t[0] + t[1] + t[2]
    return f"{n:2d}g W{t[0]} D{t[1]} L{t[2]} {t[3]:2d}-{t[4]:<2d}" if n else "      -        "
rows = []
for v, g in tab.items():
    pts = sum(3 * t[0] + t[1] for t in g.values())
    gd = sum(t[3] - t[4] for t in g.values())
    rows.append((pts, gd, v))
print(f"{'variant':12s} {'vs peers':20s} {'vs FULL':20s} {'vs Starter':20s} pts  GD")
for pts, gd, v in sorted(rows, reverse=True):
    g = tab[v]
    print(f"{v:12s} {fmt(g['peers']):20s} {fmt(g['FULL']):20s} {fmt(g['Starter']):20s} {pts:3d} {gd:+3d}")
