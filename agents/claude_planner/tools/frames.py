#!/usr/bin/env python3
"""ASCII frames from an rcg. usage: frames.py OURTEAM file.rcg c1 c2 step
Our players = digits, opponents = letters (a=1..), ball = *. Our POV (attack +x)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from rcgstats import RE_SHOW, RE_TEAM, RE_BALL, RE_P
us, path, c1, c2, step = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
left = right = None
W, H = 106, 35
for line in open(path, errors="replace"):
    if line.startswith("(team"):
        m = RE_TEAM.match(line); left, right = m.group(1), m.group(2); continue
    m = RE_SHOW.match(line.rstrip("\n"))
    if not m: continue
    cyc = int(m.group(1))
    if cyc < c1 or cyc > c2 or (cyc - c1) % step: continue
    flip = -1 if us == right else 1
    body = m.group(2)
    grid = [[" "] * W for _ in range(H)]
    for x in range(W):
        grid[0][x] = grid[H-1][x] = "-"
    for y in range(H):
        grid[y][0] = grid[y][W-1] = "|"; grid[y][W//2] = ":"
    for y in range(int((34-7)/68*H), int((34+7)/68*H)+1):
        grid[y][0] = grid[y][W-1] = "#"
    def put(x, y, ch):
        gx = int(round((x * flip + 52.5) / 105 * (W - 1))); gy = int(round((y * flip + 34) / 68 * (H - 1)))
        if 0 <= gx < W and 0 <= gy < H: grid[gy][gx] = ch
    mb = RE_BALL.search(body)
    bx, by = float(mb.group(1)), float(mb.group(2))
    for chunk in body.split("((")[1:]:
        mp = RE_P.match("((" + chunk)
        if not mp: continue
        side, unum, state, x, y = mp.groups()[:5]
        if int(state, 16) == 0: continue
        team = left if side == "l" else right
        ch = unum if team == us else "abcdefghijk"[int(unum)-1]
        put(float(x), float(y), ch)
    put(bx, by, "*")
    print(f"cycle {cyc}  ball ({bx*flip:.1f},{by*flip:.1f})")
    print("\n".join("".join(r) for r in grid))
