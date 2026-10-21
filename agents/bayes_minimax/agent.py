# Agent1 - Bayesian opponent model + adversarial search team (separate agent).
#
# Opponent model:
#   * formation identification after Fukushima, Nakashima & Akiyama (2017):
#     opponents' positions -> 6x4 grid occupancy vector (independent of
#     uniform numbers) -> class in {wall, line, normal}. They clustered with
#     a GMM fitted by EM and classified with NN/SVM/RF; here the classes are
#     Gaussian prototypes, and the class is tracked with HMM FORWARD
#     FILTERING (AIMA 14.2) using a Gaussian naive-Bayes observation model.
#   * opponent holder behaviour: P(pass | dribble | shoot, zone) learned
#     online by MLE with Laplace smoothing (AIMA 20.2) from observed kicks,
#     as a cheap stand-in for CYRUS's kick-behaviour predictor.
# On ball : depth-limited EXPECTIMINIMAX with alpha-beta pruning (AIMA 5.2-5.5):
#           our macro action (MAX) -> its success (CHANCE) -> the opponents'
#           worst response, who sprints to the ball (MIN) -> our next action
#           (MAX) -> evaluation function. Iterative deepening keeps it
#           anytime. Action bonuses switch with the identified formation.
# Off ball: marking/doubling chosen from P(action | zone); support points
#           chosen to exploit the identified formation.
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
GX, GY = 6, 4
CLASSES = ("wall", "line", "normal")
SIGMA2 = 0.02            # naive-Bayes variance per grid cell
TEMPER = 0.1             # likelihood exponent (observations are correlated)
STAY = 0.98              # HMM: P(formation unchanged per cycle)
EMA = 0.95               # smoothing of the occupancy vector
OPP_ACTS = ("pass", "dribble", "shoot")
GAMMA = 0.98
H_B, H_D, H_R = -0.5, -0.10, 0.20
TOP_ROOT, TOP_INNER, N_RESP = 8, 6, 3
STYLE = {   # bonuses by identified formation
    "wall":   {"wide": 0.03, "through": 0.0, "dribble": 0.0, "shot_range": 32.0},
    "line":   {"wide": 0.0, "through": 0.04, "dribble": 0.01, "shot_range": 28.0},
    "normal": {"wide": 0.0, "through": 0.0, "dribble": 0.0, "shot_range": 28.0},
}


def prototypes():
    """Expected share of opponent field players per grid cell, per class."""
    def cell(i, j):
        return i * GY + j
    P = {c: [0.0] * (GX * GY) for c in CLASSES}
    for j in (1, 2):                      # wall: packed in front of their goal
        P["wall"][cell(5, j)] = 0.4
        P["wall"][cell(4, j)] = 0.1
    for i in (3, 4):                      # line: one vertical band, all lanes
        for j in range(GY):
            P["line"][cell(i, j)] = 0.125
    for i in (2, 3, 4):                   # normal: spread out
        for j in range(GY):
            P["normal"][cell(i, j)] = 1.0 / 12.0
    return P


class OpponentModel:
    def __init__(self):
        self.proto = prototypes()
        self.occ = [1.0 / (GX * GY)] * (GX * GY)
        self.belief = {c: 1.0 / len(CLASSES) for c in CLASSES}
        self.counts = [[0, 0, 0] for _ in range(3)]
        self.prev = None          # (opponent kickable?, ball vel, ball pos, holder pos)
        self.last_cycle = -1

    def zone(self, x):
        return 0 if x < -17.5 else (1 if x < 17.5 else 2)

    def p_act(self, x):
        c = self.counts[self.zone(x)]
        n = sum(c)
        return {a: (c[i] + 1.0) / (n + 3.0) for i, a in enumerate(OPP_ACTS)}

    def formation(self):
        return max(CLASSES, key=lambda c: (self.belief[c], c))

    def update(self, wm):
        cyc = int(wm.cycle)
        if cyc == self.last_cycle:
            return
        self.last_cycle = cyc
        # 1) occupancy -> HMM forward filtering over the formation class
        cur = [0.0] * (GX * GY)
        field = [o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie]
        for o in field:
            i = min(GX - 1, max(0, int((o.position.x() + 52.5) / (105.0 / GX))))
            j = min(GY - 1, max(0, int((o.position.y() + 34.0) / (68.0 / GY))))
            cur[i * GY + j] += 1.0 / max(1, len(field))
        self.occ = [EMA * a + (1.0 - EMA) * b for a, b in zip(self.occ, cur)]
        pred = {c: sum(self.belief[c2] * (STAY if c2 == c else (1.0 - STAY) / 2.0)
                       for c2 in CLASSES) for c in CLASSES}
        logl = {c: -sum((a - m) ** 2 for a, m in zip(self.occ, self.proto[c]))
                / (2.0 * SIGMA2) for c in CLASSES}
        top = max(logl.values())
        post = {c: pred[c] * math.exp(TEMPER * (logl[c] - top)) for c in CLASSES}
        z = sum(post.values()) or 1.0
        self.belief = {c: max(1e-6, post[c] / z) for c in CLASSES}
        # 2) opponent kick events -> counts for P(action | zone)
        b = wm.ball
        bv = (b.velocity.x(), b.velocity.y())
        bp = (b.position.x(), b.position.y())
        holder = None
        if wm.kickable_opponent_existance:
            holder = min((o for o in wm.opponents if o.uniform_number > 0),
                         key=lambda o: o.dist_from_ball, default=None)
        if self.prev is not None and self.prev[0]:
            pv, pp, hp = self.prev[1], self.prev[2], self.prev[3]
            dv = math.hypot(bv[0] - pv[0], bv[1] - pv[1])
            if dv > 0.5:
                speed = math.hypot(*bv)
                to_goal = math.atan2(-bp[1], -52.5 - bp[0])
                ang = math.atan2(bv[1], bv[0])
                da = abs((ang - to_goal + math.pi) % (2 * math.pi) - math.pi)
                if speed < 1.0:
                    a = 1
                elif da < math.radians(20.0) and bp[0] < -25.0:
                    a = 2
                else:
                    a = 0
                self.counts[self.zone(pp[0])][a] += 1
        hp = (holder.position.x(), holder.position.y()) if holder is not None else None
        self.prev = (holder is not None, bv, bp, hp)


class SearchTimeout(Exception):
    pass


class Searcher:
    def __init__(self, geo, offx, style, deadline):
        self.g, self.offx, self.style, self.deadline = geo, offx, style, deadline
        self.nodes = 0

    def tick(self):
        self.nodes += 1
        if self.nodes % 16 == 0 and time.perf_counter() > self.deadline:
            raise SearchTimeout()

    def H(self, bx, by, opps):
        free = min((math.hypot(ox - bx, oy - by) for ox, oy, _ in opps), default=10.0)
        return sig(H_B + H_D * math.hypot(self.g.hl - bx, by) + H_R * min(free, 6.0))

    def v_fail(self, x, y):
        return -sig(0.1 - 0.1 * math.hypot(x + self.g.hl, y))

    def actions(self, bx, by, holder, mates, opps, top):
        """-> [(P, cycles, (x, y, new holder) | None, meta, order key)]."""
        g, st = self.g, self.style
        out = []
        if math.hypot(g.hl - bx, by) < st["shot_range"]:
            half = g.goal_w / 2.0 - 0.7
            for i in range(3):
                ty = -half + half * i
                dist = math.hypot(g.hl - bx, ty - by)
                k = g.steps_for(dist, g.vball)
                if k is None:
                    continue
                m = g.margin(opps, g.kick_path(bx, by, g.hl, ty, g.vball, k))
                p = sig(1.2 * (m - 1.0)) * max(0.3, 1.0 - 0.012 * max(0.0, dist - 12.0))
                out.append((p, k, None, ("shoot", g.hl, ty, k), p))
        for u, (rx, ry, gk) in mates.items():
            if u == holder or gk or (rx > self.offx and rx > 0.0):
                continue
            tg = [(rx, ry, 0.0, 0.0)]
            tg.append((rx + 5.0, ry, 0.0, 0.0))
            if st["through"] > 0.0:              # balls in behind a line defence
                tg.append((min(self.offx + 6.0, g.hl - 3.0), ry, st["through"], 1.0))
            for tx, ty, bonus, through in tg:
                lead = (tx, ty) != (rx, ry)
                if not g.inside(tx, ty) or (lead and not through
                                            and tx > self.offx and tx > 0.0):
                    continue
                dist = math.hypot(tx - bx, ty - by)
                if dist < 3.0 or dist > 40.0:
                    continue
                k, v0 = g.pass_plan(dist, 1.3)
                if k is None:
                    continue
                t_recv = math.hypot(tx - rx, ty - ry) / g.vmax + 1.0 if lead else 0.0
                if t_recv > k + 3:
                    continue
                pts = g.kick_path(bx, by, tx, ty, v0, k)
                if t_recv > k:
                    pts += [(tx, ty)] * int(math.ceil(t_recv - k))
                p = sig(g.margin(opps, pts) - 0.5)
                wide = st["wide"] if abs(ty) > 12.0 else 0.0
                c = int(max(k, math.ceil(t_recv)))
                out.append((p, c, (tx, ty, u), ("pass", tx, ty, k, u),
                            p * self.H(tx, ty, opps) + bonus + wide))
        base = math.atan2(-by, g.hl - bx)
        for i in range(8):
            a = base + math.pi * i / 4.0
            tx, ty = bx + 5.0 * math.cos(a), by + 5.0 * math.sin(a)
            if not g.inside(tx, ty):
                continue
            pts = g.carry_path(bx, by, tx, ty, 0.65)
            p = sig(g.margin(opps, pts) - 1.0)
            out.append((p, len(pts), (tx, ty, holder), ("dribble", tx, ty),
                        p * self.H(tx, ty, opps) + st["dribble"]))
        out.sort(key=lambda a: -a[4])
        return out[:top]

    def responses(self, x, y, opps, cycles):
        """MIN's options: everybody drifts, or one of the 3 nearest sprints."""
        g = self.g
        order = sorted(range(len(opps)),
                       key=lambda i: (math.hypot(opps[i][0] - x, opps[i][1] - y), i))
        outs = []
        for who in [None] + order[:N_RESP]:
            moved = []
            for i, (ox, oy, gk) in enumerate(opps):
                frac = 1.0 if i == who else (0.8 if who is None else 0.5)
                step = g.vmax * cycles * frac
                d = math.hypot(x - ox, y - oy)
                mv = min(step, max(0.0, d - 0.8))
                if d > 1e-6:
                    ox, oy = ox + (x - ox) / d * mv, oy + (y - oy) / d * mv
                moved.append((ox, oy, gk))
            outs.append(moved)
        return outs

    def max_value(self, bx, by, holder, mates, opps, depth, alpha, beta, prev,
                  top=TOP_INNER):
        self.tick()
        acts = self.actions(bx, by, holder, mates, opps, top)
        if not acts:
            return self.H(bx, by, opps), None
        best, best_meta = -1e9, None
        for p, c, nxt, meta, _ in acts:
            if nxt is not None and meta[0] == "pass" and meta[4] == prev:
                continue
            disc = GAMMA ** c
            if meta[0] == "shoot":
                v = disc * (p + (1.0 - p) * -0.02)
            else:
                vf = self.v_fail((bx + nxt[0]) * 0.5, (by + nxt[1]) * 0.5)
                if depth <= 1:
                    v = disc * (p * self.H(nxt[0], nxt[1], opps) + (1.0 - p) * vf)
                else:
                    # chance node: bounds for the success branch
                    lo = (alpha / disc - (1.0 - p) * vf) / max(p, 1e-3)
                    hi = (beta / disc - (1.0 - p) * vf) / max(p, 1e-3)
                    mts = dict(mates)
                    mts[nxt[2]] = (nxt[0], nxt[1], mts[nxt[2]][2])
                    sv = self.min_value(nxt[0], nxt[1], nxt[2], mts, opps, c,
                                        depth - 1, lo, hi, holder)
                    v = disc * (p * sv + (1.0 - p) * vf)
            if v > best:
                best, best_meta = v, meta
            alpha = max(alpha, best)
            if best >= beta:
                break
        return best, best_meta

    def min_value(self, bx, by, holder, mates, opps, cycles, depth, alpha, beta, prev):
        self.tick()
        best = 1e9
        for moved in self.responses(bx, by, opps, cycles):
            v, _ = self.max_value(bx, by, holder, mates, moved, depth - 1,
                                  alpha, beta, prev)
            best = min(best, v)
            beta = min(beta, best)
            if best <= alpha:
                break
        return best


# -------------------------------------------------------------------- agent
class Agent1(SafeAgent):
    def __init__(self):
        super().__init__()
        self.geo = None
        self.om = OpponentModel()
        self.commit = None

    def _g(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
        return self.geo

    def on_ball(self, agent) -> None:
        try:
            self.om.update(agent.wm)
        except Exception:
            pass
        SafeAgent.on_ball(self, agent)

    def off_ball(self, agent) -> None:
        try:
            self.om.update(agent.wm)
        except Exception:
            pass
        SafeAgent.off_ball(self, agent)

    def _on(self, agent):
        ms = budget_ms(agent, 4.0)
        if ms <= 1.0:
            return False
        wm = agent.wm
        cyc = int(wm.cycle)
        if (self.commit and self.commit[0] == cyc - 1 and self.commit[2] < 2
                and self.commit[1][0] in ("pass", "shoot")):
            execute(agent, self.commit[1])
            self.commit = (cyc, self.commit[1], self.commit[2] + 1)
            note(agent, "continue")
            return True
        w = World(agent)
        form = self.om.formation()
        sr = Searcher(self._g(agent), w.offx, STYLE[form],
                      time.perf_counter() + ms / 1000.0)
        meta, val, depth_done = None, 0.0, 0
        for depth in (1, 3):                     # iterative deepening
            try:
                v, m = sr.max_value(w.bx, w.by, w.me, w.mates, w.opps, depth,
                                    -1e9, 1e9, None, TOP_ROOT)
            except SearchTimeout:
                break
            if m is not None:
                meta, val, depth_done = m, v, depth
        if meta is None:
            near = min((math.hypot(ox - w.bx, oy - w.by) for ox, oy, _ in w.opps),
                       default=99.0)
            meta = ("hold",) if near > 3.0 else ("clear",)
        elif val < -0.03 and w.bx < -17.5:
            meta = ("clear",)                    # every line loses near our goal
        self.commit = (cyc, meta, 0)
        execute(agent, meta)
        note(agent, meta[0], v=round(val, 3), d=depth_done, form=form)
        return True

    def _off(self, agent):
        wm = agent.wm
        rank = field_rank(wm)
        me = wm.self.uniform_number
        pos = rank.index(me) if me in rank else 9
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        home = Tools.home_position(agent)
        if theirs and not ours and try_tackle(agent):
            return True
        if not ours and pos == 0:
            Action.intercept(agent)
            note(agent, "intercept")
            return True
        if theirs:
            pa = self.om.p_act(wm.ball.position.x())
            double = pa["dribble"] > pa["pass"]
            if double and pos == 1:
                blk = self.block_point(agent)
                if blk is not None:
                    Action.go_to_point(agent, blk, max_dash_power=100.0)
                    note(agent, "double", p=round(pa["dribble"], 2))
                    return True
            tgt = self._mark(agent, rank, skip=2 if double else 1)
            if tgt is not None:
                Action.go_to_point(agent, tgt, max_dash_power=100.0)
                note(agent, "mark", p=round(pa["pass"], 2))
                return True
        if ours:
            Action.go_to_point(agent, self._support(agent, home),
                               max_dash_power=self.dash_power(agent))
            note(agent, "support", form=self.om.formation())
            return True
        Action.go_to_point(agent, home, max_dash_power=self.dash_power(agent))
        note(agent, "home")
        return True

    def _mark(self, agent, rank, skip):
        """Greedy danger-ordered marking over ranks[skip:], same everywhere."""
        wm = agent.wm
        hl = agent.server_params.pitch_half_length
        pos = {wm.self.uniform_number: wm.self.position}
        for t in wm.teammates:
            if t.uniform_number > 0:
                pos[t.uniform_number] = t.position
        opps = [o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie]
        if not opps:
            return None
        holder = min(opps, key=lambda o: o.dist_from_ball)
        targets = sorted((o for o in opps if o is not holder),
                         key=lambda o: (math.hypot(o.position.x() + hl, o.position.y()),
                                        o.uniform_number))
        free = list(rank[skip:])
        for o in targets:
            if not free:
                break
            u = min(free, key=lambda u: (pos[u].dist(o.position), u) if u in pos else (1e9, u))
            free.remove(u)
            if u == wm.self.uniform_number:
                to_goal = Vector2D(-hl, 0.0) - o.position
                return o.position + to_goal.set_length_vector(1.5)
        return None

    def _support(self, agent, home):
        wm = agent.wm
        g = self._g(agent)
        form = self.om.formation()
        bx, by = wm.ball.position.x(), wm.ball.position.y()
        offx = wm.offside_line_x
        opps = [(o.position.x(), o.position.y()) for o in wm.opponents
                if o.uniform_number > 0]
        hx, hy = home.x(), home.y()
        cands = [(hx, hy)] + [(hx + 5.0 * math.cos(k * math.pi / 4.0),
                               hy + 5.0 * math.sin(k * math.pi / 4.0)) for k in range(8)]
        if form == "wall":                      # stretch a packed defence wide
            side = 1.0 if hy >= 0.0 else -1.0
            cands.append((hx, side * min(g.hw - 4.0, abs(hy) + 8.0)))
        elif form == "line":                    # wait on the line to run behind
            cands.append((min(offx - 1.0, g.hl - 4.0), hy))
        best, best_s = (hx, hy), -1e9
        for x, y in cands:
            if not g.inside(x, y, 2.0) or (x > offx - 0.5 and x > 0.0):
                continue
            free = min((math.hypot(ox - x, oy - y) for ox, oy in opps), default=10.0)
            s = min(free, 10.0) + 1.5 * lane(opps, bx, by, x, y) + 0.05 * x \
                - 0.1 * math.hypot(x - hx, y - hy)
            if form == "wall":
                s += 0.1 * abs(y)
            if s > best_s:
                best, best_s = (x, y), s
        return Vector2D(best[0], best[1])


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
