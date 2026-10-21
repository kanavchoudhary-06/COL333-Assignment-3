# Agent1 - Gliders2d / Fractals2019-style team (separate agent).
#
# On ball : Gliders2d v1.3 "multi-point" action evaluator built on a
#           discrete VORONOI diagram (each grid cell owned by the nearest
#           player = multi-source nearest-site assignment, AIMA 3.4 style).
#           Passes may also go to the centroid of a teammate's Voronoi
#           region (a point near him and away from opponents). Gliders2d
#           v1.6 "risk level" lets passes through with a negative time
#           margin of up to RISK[zone] cycles.
# Off ball: Gliders2d v1.1 stamina rules and v1.2 zone-dependent pressing
#           level; v1.4 positioning on Voronoi-vertex-like points (largest
#           empty circles among opponents) and offside-line points. The
#           assignment of players to points is solved with Fractals2019
#           "Dynamic Constraint Annealing": a hill-climbing phase, then
#           SIMULATED ANNEALING (AIMA 4.1) inside the constraint-satisfying
#           region (spacing, onside, near the formation slot). The RNG is
#           seeded from the cycle, so every player computes the same plan.
#
# Plain Python only.

import heapq
import math
import random
import time

from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note

# ===================================================================== core
# Shared helpers (each agent file is self-contained, so this block is
# repeated in every agent).

def sig(z):
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class Geo:
    """Server constants (rcssserver defaults as fallback) + ball kinematics
    + the relaxed reach-time bound (straight dash at top speed, no turn)."""

    def __init__(self, sp):
        g = lambda n, d: float(getattr(sp, n, d) or d)
        self.decay = g("ball_decay", 0.94)
        self.vball = g("ball_speed_max", 3.0)
        self.vmax = g("player_speed_max", 1.05)
        self.ka = (g("kickable_margin", 0.7) + g("player_size", 0.3)
                   + g("ball_size", 0.085))
        self.catch = max(g("catchable_area", 1.2), self.ka)
        self.hl = g("pitch_half_length", 52.5)
        self.hw = g("pitch_half_width", 34.0)
        self.goal_w = g("goal_width", 14.02)
        self.pen_x = self.hl - g("penalty_area_length", 16.5)
        self.pen_hw = g("penalty_area_half_width", 20.16)
        self.S = [0.0]
        acc, p = 0.0, 1.0
        for _ in range(80):
            acc += p
            p *= self.decay
            self.S.append(acc)

    def steps_for(self, dist, v0):
        for t in range(1, len(self.S)):
            if v0 * self.S[t] >= dist - 1e-6:
                return t
        return None

    def pass_plan(self, dist, end_speed):
        """(cycles, first speed) so the ball arrives at about end_speed."""
        n = max(1, int(round(math.log(1.0 + dist * (1.0 - self.decay) / end_speed)
                             / -math.log(self.decay))))
        v0 = dist / self.S[n]
        if v0 > self.vball:
            n = self.steps_for(dist, self.vball)
            if n is None:
                return None, None
            v0 = dist / self.S[n]
        return n, v0

    def kick_path(self, bx, by, tx, ty, v0, n):
        d = math.hypot(tx - bx, ty - by) or 1e-6
        ux, uy = (tx - bx) / d, (ty - by) / d
        return [(bx + ux * v0 * self.S[t], by + uy * v0 * self.S[t])
                for t in range(1, n + 1)]

    def carry_path(self, bx, by, tx, ty, speed):
        d = math.hypot(tx - bx, ty - by)
        n = max(1, int(math.ceil(d / speed)))
        ux, uy = ((tx - bx) / d, (ty - by) / d) if d > 1e-6 else (0.0, 0.0)
        return [(bx + ux * min(d, speed * t), by + uy * min(d, speed * t))
                for t in range(1, n + 1)]

    def reach_t(self, ox, oy, px, py, goalie=False, react=1.0):
        r = self.catch if (goalie and px > self.pen_x
                           and abs(py) < self.pen_hw) else self.ka
        d = math.hypot(px - ox, py - oy) - r
        return d / self.vmax + react if d > 0.0 else 0.0

    def margin(self, opps, pts, react=1.0):
        """min over path cycles t of (earliest opponent reach time - t)."""
        m = 1e9
        for t, (px, py) in enumerate(pts, 1):
            for ox, oy, g in opps:
                v = self.reach_t(ox, oy, px, py, g, react) - t
                if v < m:
                    m = v
        return m

    def inside(self, x, y, pad=1.0):
        return abs(x) < self.hl - pad and abs(y) < self.hw - pad


class World:
    """Plain-tuple snapshot of the world model for fast planning."""
    __slots__ = ("bx", "by", "me", "holder", "mates", "opps", "offx")

    def __init__(self, agent):
        wm = agent.wm
        s = wm.self
        self.me = s.uniform_number
        self.mates = {s.uniform_number: (s.position.x(), s.position.y(), False)}
        for t in wm.teammates:
            if t.uniform_number > 0:
                self.mates[t.uniform_number] = (t.position.x(), t.position.y(),
                                                bool(t.is_goalie))
        self.opps = [(o.position.x() + o.velocity.x(),
                      o.position.y() + o.velocity.y(), bool(o.is_goalie))
                     for o in wm.opponents if o.uniform_number > 0]
        b = wm.ball.position
        self.bx, self.by = b.x(), b.y()
        self.holder = s.uniform_number
        self.offx = wm.offside_line_x


def field_rank(wm):
    """Our field players by (ball reach steps, unum): identical everywhere."""
    rows = [(int(wm.intercept_table.self_reach_steps), wm.self.uniform_number)]
    for t in wm.teammates:
        if t.uniform_number > 0 and not t.is_goalie:
            rows.append((int(t.ball_reach_steps), t.uniform_number))
    rows.sort()
    return [u for _, u in rows]


def budget_ms(agent, k, reserve=7500.0, cap=250.0, wall_safety=300.0):
    """CPU ms this decision may use: k x the fair share of what is left."""
    b = agent.budget
    rem = b.remaining_ms
    if rem == float("inf"):
        rem = 75000.0
    left = max(50, 1500 - int(agent.wm.cycle))
    ms = min(cap, max(2.0, k * (rem - reserve) / left))
    ms = min(ms, b.decision_remaining_ms - wall_safety)
    if rem < reserve:
        ms = min(ms, 2.0)
    return max(0.0, ms)


def try_tackle(agent, threshold=0.75):
    wm = agent.wm
    me = wm.self
    if me.tackle_probability < threshold or wm.ball.position.dist(me.position) > 2.0:
        return False
    goal = Vector2D(agent.server_params.pitch_half_length, 0.0)
    rel = (goal - me.position).th().degree() - me.body_direction
    rel = (rel + 180.0) % 360.0 - 180.0
    Action.tackle(agent, rel, foul=False)
    note(agent, "tackle", p=round(me.tackle_probability, 2))
    return True


def execute(agent, meta):
    """Send a planned macro action; always ends with a fallback that works."""
    kind = meta[0]
    if kind == "shoot":
        tgt = Vector2D(meta[1], meta[2])
        near = min((o.dist_from_self for o in agent.wm.opponents
                    if o.uniform_number > 0), default=99.0)
        if near < 2.5:
            Action.kick_one_step(agent, tgt)
        else:
            Action.kick(agent, tgt, 1)
    elif kind == "pass":
        Action.kick(agent, Vector2D(meta[1], meta[2]), meta[3], accept_late=3)
    elif kind == "dribble":
        Action.dribble(agent, Vector2D(meta[1], meta[2]))
        Action.hold(agent)
    elif kind == "hold":
        Action.hold(agent)
    Action.clear(agent)


def lane(opps, bx, by, x, y, cap=5.0):
    """Clearance of the segment ball->(x, y): closest opponent to it."""
    dx, dy = x - bx, y - by
    L2 = dx * dx + dy * dy
    best = cap
    if L2 > 1e-6:
        for o in opps:
            u = ((o[0] - bx) * dx + (o[1] - by) * dy) / L2
            if 0.0 < u < 1.0:
                best = min(best, math.hypot(bx + u * dx - o[0], by + u * dy - o[1]))
    return best


class SafeAgent(StarterAgent):
    """Wraps _on/_off: any exception falls back to the starter behaviour,
    so a bug can never print a traceback (which voids the match)."""

    def on_ball(self, agent) -> None:
        try:
            if self._on(agent):
                return
        except Exception:
            pass
        try:
            StarterAgent.on_ball(self, agent)
        except Exception:
            Action.clear(agent)

    def off_ball(self, agent) -> None:
        try:
            if self._off(agent):
                return
        except Exception:
            pass
        try:
            StarterAgent.off_ball(self, agent)
        except Exception:
            Action.go_to_point(agent, Tools.home_position(agent))

    def _on(self, agent):
        return False

    def _off(self, agent):
        return False

# ================================================================ end core


# ----------------------------------------------------------------- settings
# These are the "evolvable" parameters Gliders tuned offline; here they are
# hand-set start values for a GA / hill-climbing tuner (AIMA 4.1).
VOR_CELL = 3.0            # Voronoi grid for the on-ball evaluator (m)
POS_CELL = 5.0            # grid for positioning candidates (m)
VOR_W = 1.5               # weight of the Voronoi term
OPP_COUNT = 2             # Voronoi term used when >= this many opps within 10 m
RISK = {"def": 0.0, "mid": 0.5, "att": 1.5}   # v1.6 risk level (cycles)
PRESS = {"def": 5, "mid": 3, "att": 4}        # v1.2 pressing level (cycles)
SA_ITERS, SA_T0, SA_T1 = 250, 1.0, 0.01
SPACING, MAX_FROM_HOME, HARD = 8.0, 18.0, 50.0
REPLAN = 4                # cycles between positioning plans


def zone_of(x):
    return "def" if x < -17.5 else ("att" if x > 17.5 else "mid")


def voronoi_centroids(geo, w, cell, radius=15.0):
    """Owner of every grid cell = nearest player of either team; return the
    centroid of each teammate's cells within `radius` of him."""
    sites = [(x, y, "m", u) for u, (x, y, gk) in w.mates.items() if not gk]
    sites += [(x, y, "o", 0) for x, y, gk in w.opps]
    acc = {}
    nx, ny = int(2 * geo.hl / cell), int(2 * geo.hw / cell)
    for i in range(nx):
        cx = -geo.hl + (i + 0.5) * cell
        for j in range(ny):
            cy = -geo.hw + (j + 0.5) * cell
            best, owner = 1e9, None
            for sx, sy, team, u in sites:
                d = (sx - cx) ** 2 + (sy - cy) ** 2
                if d < best:
                    best, owner = d, (team, u)
            if owner[0] == "m":
                sx, sy, _ = w.mates[owner[1]]
                if (sx - cx) ** 2 + (sy - cy) ** 2 <= radius * radius:
                    a = acc.setdefault(owner[1], [0.0, 0.0, 0])
                    a[0] += cx
                    a[1] += cy
                    a[2] += 1
    return {u: (a[0] / a[2], a[1] / a[2]) for u, a in acc.items() if a[2]}


class Evaluator:
    def __init__(self, geo, w):
        self.g, self.w = geo, w
        near = sum(1 for x, y, _ in w.opps
                   if math.hypot(x - w.bx, y - w.by) < 10.0)
        self.use_vor = near >= OPP_COUNT

    def value(self, x, y, receiver=None):
        g, w = self.g, self.w
        v = (x + g.hl) + max(0.0, 40.0 - math.hypot(g.hl - x, y))
        if self.use_vor:
            d_opp = min((math.hypot(ox - x, oy - y) for ox, oy, _ in w.opps),
                        default=10.0)
            d_mate = min((math.hypot(mx - x, my - y)
                          for u, (mx, my, gk) in w.mates.items()
                          if u != w.holder and not gk), default=10.0)
            v += VOR_W * max(-10.0, min(10.0, d_opp - d_mate))
        return v


def choose_action(geo, w):
    """-> (meta, value). Highest-value feasible action."""
    g = geo
    bx, by = w.bx, w.by
    ev = Evaluator(g, w)
    risk = RISK[zone_of(bx)]
    best = (("clear",), -1e9)
    # shoot
    if math.hypot(g.hl - bx, by) < 30.0:
        half = g.goal_w / 2.0 - 0.7
        for i in range(5):
            ty = -half + half * i / 2.0
            dist = math.hypot(g.hl - bx, ty - by)
            k = g.steps_for(dist, g.vball)
            if k is None:
                continue
            m = g.margin(w.opps, g.kick_path(bx, by, g.hl, ty, g.vball, k))
            if m >= 0.0 and 1000.0 + m > best[1]:
                best = (("shoot", g.hl, ty, k), 1000.0 + m)
    if best[1] >= 1000.0:
        return best
    vor = voronoi_centroids(g, w, VOR_CELL) if ev.use_vor else {}
    for u, (rx, ry, gk) in w.mates.items():
        if u == w.holder or gk or (rx > w.offx and rx > 0.0):
            continue
        targets = [(rx, ry)]
        if u in vor:
            targets.append(vor[u])
        for tx, ty in targets:
            if not g.inside(tx, ty) or (tx > w.offx and tx > 0.0):
                continue
            dist = math.hypot(tx - bx, ty - by)
            if dist < 3.0 or dist > 40.0:
                continue
            k, v0 = g.pass_plan(dist, 1.3)
            if k is None:
                continue
            t_recv = math.hypot(tx - rx, ty - ry) / g.vmax + 1.0 if (tx, ty) != (rx, ry) else 0.0
            pts = g.kick_path(bx, by, tx, ty, v0, k)
            if t_recv > k:
                pts += [(tx, ty)] * int(math.ceil(t_recv - k))
            if g.margin(w.opps, pts) < -risk:
                continue
            v = ev.value(tx, ty, u)
            if v > best[1]:
                best = (("pass", tx, ty, k, u), v)
    base = math.atan2(-by, g.hl - bx)
    for i in range(8):
        a = base + math.pi * i / 4.0
        tx, ty = bx + 6.0 * math.cos(a), by + 6.0 * math.sin(a)
        if not g.inside(tx, ty):
            continue
        if g.margin(w.opps, g.carry_path(bx, by, tx, ty, 0.65)) < 1.0:
            continue
        v = ev.value(tx, ty) - 1.0
        if v > best[1]:
            best = (("dribble", tx, ty), v)
    near = min((math.hypot(ox - bx, oy - by) for ox, oy, _ in w.opps), default=99.0)
    if best[0][0] == "clear" and near > 3.0:
        best = (("hold",), ev.value(bx, by) - 5.0)
    return best


# ------------------------------------------- Dynamic Constraint Annealing
def dca_assign(players, cands, values, home, attack, offx, seed):
    """players: [(unum, x, y)], cands: [(x, y)], values: [float].
    -> {unum: candidate index}. Phase 1 hill climbing, phase 2 annealing."""
    n = len(players)
    if n == 0 or not cands:
        return {}

    def cost(assign):
        c = 0.0
        for i, j in enumerate(assign):
            u, px, py = players[i]
            cx, cy = cands[j]
            c += 0.15 * math.hypot(px - cx, py - cy) - values[j]
            hx, hy = home[u]
            if math.hypot(cx - hx, cy - hy) > MAX_FROM_HOME:
                c += HARD
            if attack and cx > offx - 0.5 and cx > 0.0:
                c += HARD
            for k in range(i + 1, n):
                qx, qy = cands[assign[k]]
                if math.hypot(cx - qx, cy - qy) < SPACING:
                    c += HARD * 0.5
        return c

    # phase 1: greedy start, then hill climbing on single moves and swaps
    assign, used = [], set()
    for i in range(n):
        u, px, py = players[i]
        free = [j for j in range(len(cands)) if j not in used]
        if not free:
            free = list(range(len(cands)))
        j = min(free, key=lambda j: (0.15 * math.hypot(px - cands[j][0], py - cands[j][1])
                                     - values[j], j))
        assign.append(j)
        used.add(j)
    cur = cost(assign)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(len(cands)):
                if j == assign[i]:
                    continue
                trial = list(assign)
                if j in trial:
                    k = trial.index(j)
                    trial[k] = assign[i]
                trial[i] = j
                c = cost(trial)
                if c < cur - 1e-9:
                    assign, cur, improved = trial, c, True
    # phase 2: simulated annealing around the hill-climbing optimum
    rng = random.Random(seed)
    best, best_c = list(assign), cur
    for it in range(SA_ITERS):
        T = SA_T0 * (SA_T1 / SA_T0) ** (it / float(SA_ITERS - 1))
        i = rng.randrange(n)
        j = rng.randrange(len(cands))
        if j == assign[i]:
            continue
        trial = list(assign)
        if j in trial:
            trial[trial.index(j)] = assign[i]
        trial[i] = j
        c = cost(trial)
        if c < cur or rng.random() < math.exp(-(c - cur) / T):
            assign, cur = trial, c
            if c < best_c:
                best, best_c = list(trial), c
    return {players[i][0]: best[i] for i in range(n)}


# -------------------------------------------------------------------- agent
class Agent1(SafeAgent):
    def __init__(self):
        super().__init__()
        self.geo = None
        self.plan = None          # (cycle block, {unum: Vector2D}, attack)

    def _g(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
        return self.geo

    def _on(self, agent):
        if budget_ms(agent, 2.0) <= 0.5:
            return False
        w = World(agent)
        meta, v = choose_action(self._g(agent), w)
        execute(agent, meta)
        note(agent, meta[0], v=round(v, 1))
        return True

    def _power(self, agent, pos, home):
        """Gliders2d v1.1: full dash in the situations that matter."""
        wm = agent.wm
        bx = wm.ball.position.x()
        if pos == 0:
            return 100.0
        if bx < -20.0 and home.x() < -10.0:
            return 100.0
        if bx > wm.offside_line_x - 15.0 and home.x() > 10.0:
            return 100.0
        return self.dash_power(agent)

    def _off(self, agent):
        wm = agent.wm
        it = wm.intercept_table
        rank = field_rank(wm)
        me = wm.self.uniform_number
        pos = rank.index(me) if me in rank else 9
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        if theirs and not ours and try_tackle(agent):
            return True
        if not ours:
            my_t = int(it.self_reach_steps)
            opp_t = int(it.first_opponent_reach_steps)
            level = PRESS[zone_of(wm.ball.position.x())]
            if pos == 0 or (pos == 1 and theirs and my_t <= opp_t + level):
                Action.intercept(agent)
                note(agent, "press" if theirs else "chase")
                return True
        home = Tools.home_position(agent)
        tgt = home
        if budget_ms(agent, 0.3) > 0.5:
            plan = self._positions(agent, rank)
            tgt = plan.get(me, home)
        Action.go_to_point(agent, tgt, max_dash_power=self._power(agent, pos, home))
        note(agent, "dca")
        return True

    def _positions(self, agent, rank):
        wm = agent.wm
        cyc = int(wm.cycle)
        attack = bool(wm.kickable_teammate_existance) or (
            not wm.kickable_opponent_existance
            and int(wm.intercept_table.first_opponent_reach_steps)
            > min(int(wm.intercept_table.self_reach_steps),
                  int(wm.intercept_table.first_teammate_reach_steps)))
        block = cyc // REPLAN      # plans are made on shared cycle blocks
        if (self.plan is not None and self.plan[0] == block
                and self.plan[2] == attack):
            return self.plan[1]
        g = self._g(agent)
        w = World(agent)
        hl = g.hl
        holder = None
        if wm.kickable_teammate_existance:
            mates = [t for t in wm.teammates if t.uniform_number > 0]
            mates.append(wm.self)
            holder = min(mates, key=lambda t: t.position.dist(wm.ball.position)).uniform_number
        movers = [u for u in rank[1:] if u != holder] if rank else []
        home = {}
        players = []
        for u in movers:
            x, y, _ = w.mates[u]
            players.append((u, x, y))
            home[u] = self._home_of(agent, u)
        cands, values = [], []
        if attack:
            x0, x1 = w.bx - 15.0, min(w.offx - 1.0, hl - 3.0)
            nx = int((2 * hl) / POS_CELL)
            ny = int((2 * g.hw) / POS_CELL)
            grid = {}
            for i in range(nx):
                cx = -hl + (i + 0.5) * POS_CELL
                if cx < x0 or cx > x1:
                    continue
                for j in range(ny):
                    cy = -g.hw + (j + 0.5) * POS_CELL
                    grid[(i, j)] = min((math.hypot(ox - cx, oy - cy)
                                        for ox, oy, _ in w.opps), default=20.0)
            for (i, j), d in grid.items():
                # local maximum of the free-space field ~ a Voronoi vertex
                if all(d >= grid.get((i + a, j + b), -1.0)
                       for a in (-1, 0, 1) for b in (-1, 0, 1)):
                    cands.append((-hl + (i + 0.5) * POS_CELL,
                                  -g.hw + (j + 0.5) * POS_CELL))
            for y in (-24.0, -12.0, 0.0, 12.0, 24.0):
                cands.append((min(w.offx - 1.5, hl - 3.0), y))
            for u in movers:
                cands.append(home[u])
            for cx, cy in cands:
                free = min((math.hypot(ox - cx, oy - cy) for ox, oy, _ in w.opps),
                           default=10.0)
                values.append(0.6 * min(free, 10.0)
                              + 1.2 * lane(w.opps, w.bx, w.by, cx, cy)
                              + 0.08 * cx)
        else:
            for ox, oy, gk in w.opps:
                if gk or math.hypot(ox - w.bx, oy - w.by) < 2.0:
                    continue
                dx, dy = -hl - ox, -oy
                d = math.hypot(dx, dy) or 1.0
                cands.append((ox + 1.5 * dx / d, oy + 1.5 * dy / d))
                values.append(10.0 * sig(0.1 - 0.1 * d) + 2.0)
            gx, gy = -hl - w.bx, -w.by
            gd = math.hypot(gx, gy) or 1.0
            cands.append((w.bx + gx / gd * min(8.0, gd * 0.4),
                          w.by + gy / gd * min(8.0, gd * 0.4)))
            values.append(6.0)
            for u in movers:
                cands.append(home[u])
                values.append(0.0)
        seed = block * 7919 + sum(movers)
        assign = dca_assign(players, cands, values, home, attack, w.offx, seed)
        plan = {u: Vector2D(cands[j][0], cands[j][1]) for u, j in assign.items()}
        self.plan = (block, plan, attack)
        return plan

    def _home_of(self, agent, unum):
        """Anchor for the 'stay near your slot' constraint. Teammates' slots
        are not visible, so EVERY player uses current positions here; that
        keeps the inputs, and so the annealed plan, identical on all players."""
        if unum == agent.wm.self.uniform_number:
            p = agent.wm.self.position
            return (p.x(), p.y())
        for t in agent.wm.teammates:
            if t.uniform_number == unum:
                return (t.position.x(), t.position.y())
        return (0.0, 0.0)


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
