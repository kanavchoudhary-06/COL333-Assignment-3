# Your agent goes here. The Agent1 class must hold your final submission.
#
# Agent1 (v2): one coherent planner built only from course methods.
#
#   EVALUATION   V(s) = possession value of the ball's spot from an abstract
#                MDP solved by VALUE ITERATION at start-up (AIMA 17.2),
#                interpolated over the pitch, combined with an expected-goal
#                term (distance + goal angle left open by opponents).
#                Losing the ball at (x, y) is worth -V of the opponent's
#                possession there (zero-sum).
#   OPPONENTS    each opponent's tendency to chase is learned online as a
#                Beta-Bernoulli posterior (AIMA 20.2); reach times use it, so
#                passive opponents are not treated as sprinters.
#   ON BALL      1. A* SEARCH (AIMA 3.5) plans a dribble route to the shooting
#                   zone around opponents; its first waypoint is a candidate.
#                2. Greedy BEST-FIRST search over macro-action chains
#                   (pass / lead or through pass / dribble / shoot / advance),
#                   chains scored by expected utility with chance nodes
#                   (WrightEagle 2013 without MAXQ).
#                3. MONTE CARLO TREE SEARCH (UCT, AIMA 5.4) over the best root
#                   actions: each simulation samples opponent speed and
#                   reaction from the learned posterior and execution noise,
#                   then plays one greedy follow-up. Visit-weighted values are
#                   blended with the chain values.
#                Holding the ball is a last resort and gets more expensive
#                every cycle it is repeated.
#   OFF BALL     deterministic roles computed identically by every player:
#                ranked trap point (HELIOS2023), cover, OMAM marking (CYRUS),
#                CYRUS2022 unmarking, YuShan2022 grid positioning.
#   TIME         per-decision CPU budget from the match budget left; A*,
#                chain search and MCTS each get a share and all are anytime.
#
# Plain Python only (math, time, heapq, random).

import heapq
import math
import random
import time

from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# ---------------------------------------------------------------- parameters
# Every number here is a starting value to tune offline (eval/sweep.py).
class P:
    GAMMA = 0.98            # per-cycle discount

    # ---- time manager (ms)
    RESERVE_MS = 7500.0     # CPU kept back for the end of the match
    K_ONBALL = 5.0          # on-ball decisions may use k x the fair share
    B_MIN_MS = 2.0
    B_MAX_MS = 300.0        # hard cap per decision
    WALL_SAFETY_MS = 300.0  # stop with this much left of the 1 s wall cap
    MATCH_CYCLES = 1500
    SHARE_ASTAR = 0.15      # budget fractions: A*, then chain search,
    SHARE_CHAIN = 0.55      # MCTS gets whatever is left

    # ---- chain search
    D_MAX = 3               # macro actions per chain
    N_MAX = 400             # node expansions (the deadline usually binds)
    CAP = 10                # children kept per expanded node
    CHECK_EVERY = 4

    # ---- MCTS refinement
    MCTS_ARMS = 6           # best root actions that get simulated
    MCTS_C = 0.05           # UCB exploration constant (values ~ 0..0.4)
    MCTS_PRIOR = 6.0        # chain value counts as this many simulations
    MCTS_MAX_SIMS = 600
    EXEC_NOISE = 0.8        # sd (cycles) of execution noise on margins
    FOLLOW_UPS = 3          # greedy follow-up actions scored per simulation

    # ---- opponent model (Beta prior on "chases the ball when near")
    OPP_PRIOR_A, OPP_PRIOR_B = 6.0, 1.0
    OPP_DECAY = 0.995       # forget old evidence slowly
    OPP_NEAR = 25.0         # learn only from the 2 opponents nearest the ball,
                            # within this distance, while their team does not
                            # have it (the moments they should be chasing)
    OPP_ACTIVE_SPEED = 0.4  # m/cycle that counts as "chasing"
    OPP_MIN_FACTOR = 1.0    # floor on the learned speed factor. 1.0 = learning
                            # off (assume every opponent sprints): in tests the
                            # "speed near the ball" signal could not tell full
                            # from idle and lower floors lost badly to full.
                            # Sweep 0.6-0.9 to re-test.
    OPP_REACT = 1.0         # cycles of reaction before an opponent moves

    # ---- success models: P = sigmoid(K * (margin - C)), margin in cycles
    PASS_K, PASS_C = 1.0, 0.3
    DRIB_K, DRIB_C = 1.0, 0.5
    SHOT_K, SHOT_C = 1.2, 0.8
    HOLD_K, HOLD_C = 1.0, 1.5
    DRIB_COMMIT = 4         # dribble risk only over the cycles we commit to

    # ---- evaluation
    XG_A, XG_D, XG_ANG = -1.0, -0.15, 2.5   # xG = sig(A + D*dist + ANG*open rad)
    XG_W = 0.7              # share of xG that counts on top of possession value
    XG_RANGE = 30.0
    LOSS_W = 1.0            # weight of the opponent's value after we lose it
    V_FAIL_SHOT = -0.02
    CLEAR_VALUE = -0.03
    HOLD_DISCOUNT = 0.75    # holding wins nothing
    HOLD_STREAK = 0.6       # each repeated hold multiplies its value by this
    HOLD_NEAR = 2.5         # hold only considered with an opponent this close
                            # or when nothing else is positive

    # ---- macro action geometry
    PASS_MIN, PASS_MAX = 3.0, 40.0
    PASS_END_SPEEDS = (1.5, 0.9)
    LEAD_DIST = (4.0, 7.0)
    THROUGH_DEPTH = 6.0     # through balls this far behind the offside line
    DRIB_DIRS = 8
    DRIB_DISTS = (4.0, 8.0, 12.0)
    DRIB_SPEED = 0.65
    SHOT_POINTS = 5
    SHOT_RANGE = 32.0
    ADVANCE_DIST = 30.0
    ADVANCE_PRESS = 4.0     # only offer advance with an opponent this close
    ADVANCE_TRUST = 0.7     # discount on its (optimistic) success estimate
    COMMIT_DELTA = 0.05     # keep last cycle's kick unless beaten by this

    # ---- A* dribble router
    ASTAR_CELL = 2.0
    ASTAR_NODES = 900
    ASTAR_RISK_W = 3.0      # cost (cycles) per cycle of missing safety margin
    ASTAR_SAFE = 2.0        # wanted margin (cycles) at every route cell
    ASTAR_WAYPOINT = 7.0    # metres along the route for the dribble target
    ASTAR_MIN_X = -25.0     # only plan routes from beyond this x
    ASTAR_HORIZON = 10.0    # risk is judged as if a cell is reached within
                            # this many cycles: the route is re-planned each
                            # cycle, so far cells are not committed to yet

    # ---- off-ball (YuShan2022-style ranker): dist, x, free|cover, lane
    GRID_W = {"aggressive":   (-0.25, 0.08, 1.0, 1.2),
              "conservative": (-0.35, 0.02, 0.8, 1.5),
              "defend":       (-0.30, -0.02, -0.6, 0.0)}


def _sig(z):
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------------------------------------------- physics
class Phys:
    """Constants read once from server params (rcssserver defaults)."""

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
        self.S = [0.0]                       # S[t] = (1 - d^t) / (1 - d)
        d, acc, p = self.decay, 0.0, 1.0
        for _ in range(80):
            acc += p
            p *= d
            self.S.append(acc)

    def steps_for(self, dist, v0):
        S = self.S
        for t in range(1, len(S)):
            if v0 * S[t] >= dist - 1e-6:
                return t
        return None

    def speed_for(self, dist, n):
        return dist / self.S[max(1, n)]

    def inside(self, x, y, pad=1.0):
        return abs(x) < self.hl - pad and abs(y) < self.hw - pad


# ------------------------------------------------- MDP-based zone values
NX, NY, NPR = 6, 3, 3
DANGER = (0.45, 0.20, 0.08, 0.03, 0.01, 0.005)


def _zone_model(s, a):
    """Abstract possession MDP (same model as agents/mdp_qlearning):
    -> list of (prob, next state | None, reward, cycles)."""
    xi, yi, pi = s // (NY * NPR), (s // NPR) % NY, s % NPR
    idx = lambda x, y, p: (x * NY + y) * NPR + p
    lost = -DANGER[xi]
    out = []
    if a == 0:                                              # shoot
        base = {5: 0.35, 4: 0.08}.get(xi, 0.0)
        pg = base * (1.0 if yi == 1 else 0.5) * (0.5, 0.8, 1.0)[pi]
        return [(pg, None, 1.0, 6), (1.0 - pg, None, lost * 0.3, 6)]
    if a in (1, 2, 3):                                      # fwd / safe / dribble
        press = (0.3, 0.4, 0.3)
        if a == 1:
            p, nx, k, ys = (0.45, 0.60, 0.75)[pi], [(min(NX - 1, xi + 1), 1.0)], 10, \
                [(0, 1 / 3.0), (1, 1 / 3.0), (2, 1 / 3.0)]
        elif a == 2:
            p, nx, k, ys = (0.70, 0.82, 0.90)[pi], [(xi, 1.0)], 8, \
                [(0, 1 / 3.0), (1, 1 / 3.0), (2, 1 / 3.0)]
            press = (0.2, 0.4, 0.4)
        else:
            p, k = (0.35, 0.70, 0.88)[pi], 8
            nx, ys = [(min(NX - 1, xi + 1), 0.4), (xi, 0.6)], [(yi, 1.0)]
            press = (0.4, 0.4, 0.2)
        for x2, px in nx:
            for y2, py in ys:
                for p2 in range(NPR):
                    out.append((p * px * py * press[p2], idx(x2, y2, p2), 0.0, k))
        out.append((1.0 - p, None, lost, k))
        return out
    if a == 4:                                              # hold
        p = (0.50, 0.85, 0.95)[pi]
        return [(p * 0.5, s, 0.0, 2), (p * 0.5, idx(xi, yi, max(0, pi - 1)), 0.0, 2),
                (1.0 - p, None, lost, 2)]
    return [(1.0, None, -0.35 * DANGER[xi] - 0.01, 1)]     # clear


def zone_values(sweeps=80):
    """Value iteration over the 54-state possession MDP."""
    n = NX * NY * NPR
    tables = [[_zone_model(s, a) for a in range(6)] for s in range(n)]
    V = [0.0] * n
    for _ in range(sweeps):
        delta = 0.0
        for s in range(n):
            best = -1e9
            for outs in tables[s]:
                q = 0.0
                for p, s2, r, k in outs:
                    q += p * (r + (0.0 if s2 is None else 0.99 ** k * V[s2]))
                best = max(best, q)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-7:
            break
    return V


# ------------------------------------------------------------ evaluation
class Evaluator:
    """V(s) for us; -V for the opponent where we would lose the ball."""

    def __init__(self, phys, zone_v):
        self.ph = phys
        self.Z = zone_v

    def _zone(self, x, y, free):
        """Bilinear interpolation of the zone values in x and lane, linear in
        pressure, so the value rises smoothly toward goal."""
        hl = self.ph.hl
        fx = max(0.0, min(NX - 1.0, (x + hl) / (2 * hl / NX) - 0.5))
        x0 = int(fx)
        x1 = min(NX - 1, x0 + 1)
        tx = fx - x0
        ay = abs(y)
        lane_c = max(0.0, min(1.0, (ay - 6.0) / 8.0))       # 0 centre .. 1 wing
        yw = 0 if y < 0 else 2
        fp = max(0.0, min(2.0, (free - 1.5) / 4.0))          # 1.5 m->0, 9.5 m->2
        p0 = int(fp)
        p1 = min(2, p0 + 1)
        tp = fp - p0
        Z = self.Z

        def at(xi, yi):
            a = Z[(xi * NY + yi) * NPR + p0]
            b = Z[(xi * NY + yi) * NPR + p1]
            return a + (b - a) * tp

        def lane(xi):
            c = at(xi, 1)
            w = at(xi, yw)
            return c + (w - c) * lane_c

        v0, v1 = lane(x0), lane(x1)
        return v0 + (v1 - v0) * tx

    def open_angle(self, bx, by, blockers):
        """Radians of the goal mouth not covered by blockers (incl. goalie)."""
        ph = self.ph
        gx, half = ph.hl, ph.goal_w / 2.0
        a_lo = math.atan2(-half - by, gx - bx)
        a_hi = math.atan2(half - by, gx - bx)
        if a_hi <= a_lo:
            return 0.0
        cover = []
        for ox, oy, gk, _v in blockers:
            if ox <= bx:
                continue
            d = math.hypot(ox - bx, oy - by)
            if d < 0.5:
                continue
            r = (1.5 if gk else 0.8) + 0.05 * d            # reach grows with time
            a = math.atan2(oy - by, ox - bx)
            w = math.atan(r / d)
            lo, hi = max(a_lo, a - w), min(a_hi, a + w)
            if hi > lo:
                cover.append((lo, hi))
        cover.sort()
        blocked, cur_lo, cur_hi = 0.0, None, None
        for lo, hi in cover:
            if cur_hi is None or lo > cur_hi:
                if cur_hi is not None:
                    blocked += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        if cur_hi is not None:
            blocked += cur_hi - cur_lo
        return max(0.0, (a_hi - a_lo) - blocked)

    def xg(self, bx, by, opps):
        d = math.hypot(self.ph.hl - bx, by)
        if d > P.XG_RANGE:
            return 0.0
        return _sig(P.XG_A + P.XG_D * d + P.XG_ANG * self.open_angle(bx, by, opps))

    def V(self, st):
        free = min((math.hypot(o[0] - st.bx, o[1] - st.by) for o in st.opps),
                   default=10.0)
        z = self._zone(st.bx, st.by, free)
        g = self.xg(st.bx, st.by, st.opps)
        return z + (1.0 - z) * P.XG_W * g if z >= 0.0 else z + P.XG_W * g

    def loss(self, x, y):
        """Value for us if the opponent gets the ball at (x, y)."""
        z = self._zone(-x, -y, 5.0)                 # their view, mid pressure
        return -P.LOSS_W * max(0.0, z)              # never a gain for us


# --------------------------------------------------- opponent behaviour
class OpponentModel:
    """Per-opponent Beta posterior over 'moves fast when near the ball'."""

    def __init__(self):
        self.ab = {}
        self.last_cycle = -1

    def update(self, wm):
        cyc = int(wm.cycle)
        if cyc == self.last_cycle:
            return
        self.last_cycle = cyc
        if wm.kickable_opponent_existance:
            return                       # they have it: nobody needs to chase
        b = wm.ball.position
        field = sorted((o.position.dist(b), o.uniform_number, o)
                       for o in wm.opponents
                       if o.uniform_number > 0 and not o.is_goalie)
        for d, u, o in field[:2]:
            if d > P.OPP_NEAR:
                break
            a, bb = self.ab.get(u, (P.OPP_PRIOR_A, P.OPP_PRIOR_B))
            a, bb = a * P.OPP_DECAY, bb * P.OPP_DECAY
            if o.velocity.r() >= P.OPP_ACTIVE_SPEED or d < 2.0:
                a += 1.0
            else:
                bb += 1.0
            self.ab[u] = (a, bb)

    def mean_factor(self):
        if not self.ab:
            return 1.0
        return sum(max(P.OPP_MIN_FACTOR, a / (a + b))
                   for a, b in self.ab.values()) / len(self.ab)

    def factor(self, unum, goalie=False):
        if goalie:
            return 1.0
        a, b = self.ab.get(unum, (P.OPP_PRIOR_A, P.OPP_PRIOR_B))
        return max(P.OPP_MIN_FACTOR, a / (a + b))

    def sample(self, unum, goalie, rng):
        if goalie:
            return 1.0
        a, b = self.ab.get(unum, (P.OPP_PRIOR_A, P.OPP_PRIOR_B))
        return max(P.OPP_MIN_FACTOR, rng.betavariate(a, b))


# ----------------------------------------------------------------- state
class State:
    """Ball at (bx, by) held by `holder`. mates: {unum: (x, y, goalie)};
    opps: [(x, y, goalie, speed factor)]."""
    __slots__ = ("bx", "by", "holder", "mates", "opps", "prev")

    def __init__(self, bx, by, holder, mates, opps, prev=None):
        self.bx, self.by, self.holder = bx, by, holder
        self.mates, self.opps, self.prev = mates, opps, prev


def snapshot(agent, model=None):
    wm = agent.wm
    me = wm.self
    mates = {me.uniform_number: (me.position.x(), me.position.y(), False)}
    for t in wm.teammates:
        if t.uniform_number > 0:
            mates[t.uniform_number] = (t.position.x(), t.position.y(),
                                       bool(t.is_goalie))
    opps = []
    for o in wm.opponents:
        if o.uniform_number > 0:
            gk = bool(o.is_goalie)
            f = model.factor(o.uniform_number, gk) if model else 1.0
            opps.append((o.position.x() + o.velocity.x(),
                         o.position.y() + o.velocity.y(), gk, f))
    b = wm.ball.position
    return State(b.x(), b.y(), me.uniform_number, mates, opps)


class Cand:
    """One root (or chain) action with what MCTS needs to re-simulate it."""
    __slots__ = ("p", "c", "child", "meta", "pts", "t_end", "fail_xy")

    def __init__(self, p, c, child, meta, pts=None, t_end=0, fail_xy=None):
        self.p, self.c, self.child, self.meta = p, c, child, meta
        self.pts, self.t_end, self.fail_xy = pts, t_end, fail_xy


# --------------------------------------------------------------- planner
class Planner:
    def __init__(self, phys, ev, offside_x):
        self.ph = phys
        self.ev = ev
        self.offside_x = offside_x

    # ---- relaxed reach time with the opponent's learned speed factor
    def reach(self, o, px, py, factor=None, react=None):
        ph = self.ph
        ox, oy, gk, f = o
        r = ph.catch if (gk and px > ph.pen_x and abs(py) < ph.pen_hw) else ph.ka
        d = math.hypot(px - ox, py - oy) - r
        if d <= 0.0:
            return 0.0
        v = ph.vmax * (f if factor is None else factor)
        return d / v + (P.OPP_REACT if react is None else react)

    def margin(self, opps, pts, t_end, factors=None, reacts=None):
        """min over path cycles t <= t_end of (opponent reach time - t)."""
        m = 1e9
        for t, (px, py) in enumerate(pts, 1):
            if t > t_end:
                break
            for j, o in enumerate(opps):
                v = self.reach(o, px, py,
                               None if factors is None else factors[j],
                               None if reacts is None else reacts[j]) - t
                if v < m:
                    m = v
        return m

    def _ball_pts(self, x0, y0, ux, uy, v0, n):
        S = self.ph.S
        return [(x0 + ux * v0 * S[t], y0 + uy * v0 * S[t]) for t in range(1, n + 1)]

    # ---- successor model
    def _move_opps(self, st, nx, ny, cycles):
        out = []
        for ox, oy, gk, f in st.opps:
            step = self.ph.vmax * f * cycles * 0.8
            dx, dy = nx - ox, ny - oy
            d = math.hypot(dx, dy)
            mv = min(step, max(0.0, d - 1.0))
            if d > 1e-6:
                ox, oy = ox + dx / d * mv, oy + dy / d * mv
            out.append((ox, oy, gk, f))
        return out

    def _child(self, st, nx, ny, holder, cycles):
        mates = dict(st.mates)
        mates[holder] = (nx, ny, mates[holder][2])
        return State(nx, ny, holder, mates, self._move_opps(st, nx, ny, cycles),
                     prev=st.holder)

    # ---- generators
    def shots(self, st):
        ph = self.ph
        bx, by = st.bx, st.by
        out = []
        if math.hypot(ph.hl - bx, by) >= P.SHOT_RANGE:
            return out
        half = ph.goal_w / 2.0 - 0.7
        for i in range(P.SHOT_POINTS):
            ty = -half + 2.0 * half * i / (P.SHOT_POINTS - 1)
            dist = math.hypot(ph.hl - bx, ty - by)
            n = ph.steps_for(dist, ph.vball)
            if n is None:
                continue
            pts = self._ball_pts(bx, by, (ph.hl - bx) / dist, (ty - by) / dist,
                                 ph.vball, n)
            m = self.margin(st.opps, pts, n)
            p = _sig(P.SHOT_K * (m - P.SHOT_C))
            p *= max(0.3, 1.0 - 0.012 * max(0.0, dist - 12.0))
            out.append(Cand(p, n, None, ("shoot", ph.hl, ty, n), pts, n))
        return out

    def pass_options(self, st, unum, rx, ry, tx, ty, lead):
        """Pass to (tx, ty) for teammate unum at (rx, ry); one Cand per pace.
        Offside is judged on the receiver now, so a through ball may land
        beyond the offside line."""
        ph = self.ph
        bx, by = st.bx, st.by
        out = []
        if not ph.inside(tx, ty):
            return out
        dist = math.hypot(tx - bx, ty - by)
        if dist < P.PASS_MIN or dist > P.PASS_MAX:
            return out
        ux, uy = (tx - bx) / dist, (ty - by) / dist
        t_recv = (max(0.0, (math.hypot(tx - rx, ty - ry) - ph.ka) / ph.vmax) + 1.0
                  if lead else 0.0)
        for e in (P.PASS_END_SPEEDS if not lead else P.PASS_END_SPEEDS[:1]):
            x = 1.0 + dist * (1.0 - ph.decay) / e
            n = max(1, int(round(math.log(x) / -math.log(ph.decay))))
            v0 = ph.speed_for(dist, n)
            if v0 > ph.vball:
                n = ph.steps_for(dist, ph.vball)
                if n is None:
                    continue
                v0 = ph.speed_for(dist, n)
            t_ctrl = max(float(n), t_recv)
            if lead and t_recv > n + 3:
                continue
            pts = self._ball_pts(bx, by, ux, uy, v0, n)
            t_end = int(math.ceil(t_ctrl))
            if t_end > n:
                pts = pts + [(tx, ty)] * (t_end - n)
            m = self.margin(st.opps, pts, t_end)
            p = _sig(P.PASS_K * (m - P.PASS_C))
            p *= max(0.5, 1.0 - 0.01 * max(0.0, dist - 20.0))
            out.append(Cand(p, t_end, self._child(st, tx, ty, unum, t_end),
                            ("pass", tx, ty, n, unum), pts, t_end,
                            ((bx + tx) * 0.5, (by + ty) * 0.5)))
        return out

    def passes(self, st, root):
        ph = self.ph
        out = []
        for unum, (rx, ry, gk) in st.mates.items():
            if unum == st.holder or gk:
                continue
            if not root and unum == st.prev:
                continue                       # no immediate give-and-go
            if rx > self.offside_x and rx > 0.0:
                continue                       # receiver offside now
            targets = [(rx, ry, False)]
            ang = math.atan2(-ry, ph.hl - rx)
            for ld in P.LEAD_DIST:
                targets.append((rx + ld * math.cos(ang), ry + ld * math.sin(ang), True))
                targets.append((rx + ld, ry, True))
            if rx > self.offside_x - 15.0:     # through ball into space
                targets.append((min(self.offside_x + P.THROUGH_DEPTH, ph.hl - 3.0),
                                ry * 0.7, True))
            for tx, ty, lead in targets:
                out.extend(self.pass_options(st, unum, rx, ry, tx, ty, lead))
        return out

    def dribble_to(self, st, tx, ty, tag="dribble"):
        """Dribble Cand: risk only over the committed first cycles, since the
        holder re-decides every cycle."""
        bx, by = st.bx, st.by
        L = math.hypot(tx - bx, ty - by)
        if L < 0.5:
            return None
        ca, sa = (tx - bx) / L, (ty - by) / L
        n = int(math.ceil(L / P.DRIB_SPEED))
        pts = [(bx + ca * min(P.DRIB_SPEED * t, L), by + sa * min(P.DRIB_SPEED * t, L))
               for t in range(1, n + 1)]
        t_end = min(n, P.DRIB_COMMIT + int(L / 6.0))
        m = self.margin(st.opps, pts, t_end)
        p = _sig(P.DRIB_K * (m - P.DRIB_C))
        # the end point must also not be owned by an opponent
        m_end = min((self.reach(o, tx, ty) for o in st.opps), default=99.0) - n
        p *= _sig(1.0 * (m_end + 2.0))
        return Cand(p, n, self._child(st, tx, ty, st.holder, n),
                    (tag, tx, ty, n), pts, t_end, (tx, ty))

    def dribbles(self, st):
        ph = self.ph
        out = []
        base = math.atan2(-st.by, ph.hl - st.bx)
        for k in range(P.DRIB_DIRS):
            a = base + 2.0 * math.pi * k / P.DRIB_DIRS
            for L in P.DRIB_DISTS:
                tx, ty = st.bx + L * math.cos(a), st.by + L * math.sin(a)
                if ph.inside(tx, ty):
                    c = self.dribble_to(st, tx, ty)
                    if c is not None:
                        out.append(c)
        return out

    def advance(self, st):
        """Action.advance_ball: ~30 m up the clearest forward lane; the ball
        is then contested at the landing point."""
        ph = self.ph
        bx, by = st.bx, st.by
        best = None
        for deg in (-40, -20, 0, 20, 40):
            a = math.radians(deg)
            tx = min(ph.hl - 2.0, bx + P.ADVANCE_DIST * math.cos(a))
            ty = max(-ph.hw + 2.0, min(ph.hw - 2.0, by + P.ADVANCE_DIST * math.sin(a)))
            lane = min((self._seg_dist(o, bx, by, tx, ty) for o in st.opps), default=9.0)
            if best is None or lane > best[0]:
                best = (lane, tx, ty)
        _, tx, ty = best
        dist = math.hypot(tx - bx, ty - by)
        n = ph.steps_for(dist, ph.vball) or 25
        t_us = min((math.hypot(mx - tx, my - ty) / ph.vmax + 1.0
                    for u, (mx, my, gk) in st.mates.items()
                    if u != st.holder and not gk), default=99.0)
        t_them = min((self.reach(o, tx, ty) for o in st.opps), default=99.0)
        p = P.ADVANCE_TRUST * _sig(0.8 * (t_them - max(t_us, n)))
        return Cand(p, int(max(t_us, n)), self._child(st, tx, ty, st.holder, n),
                    ("advance", tx, ty), None, 0, (tx, ty))

    @staticmethod
    def _seg_dist(o, bx, by, tx, ty):
        dx, dy = tx - bx, ty - by
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return math.hypot(o[0] - bx, o[1] - by)
        u = max(0.0, min(1.0, ((o[0] - bx) * dx + (o[1] - by) * dy) / L2))
        return math.hypot(bx + u * dx - o[0], by + u * dy - o[1])

    def actions(self, st, root, extra=()):
        out = self.shots(st) + self.passes(st, root) + self.dribbles(st)
        out.extend(extra)
        if root:
            near = min((self.reach(o, st.bx, st.by) for o in st.opps), default=20.0)
            p = _sig(P.HOLD_K * (near - P.HOLD_C))
            held = State(st.bx, st.by, st.holder, st.mates,
                         self._move_opps(st, st.bx, st.by, 3), prev=st.prev)
            out.append(Cand(p, 3, held, ("hold",), None, 0, (st.bx, st.by)))
            out.append(Cand(1.0, 0, None, ("clear",)))
            near_d = min((math.hypot(o[0] - st.bx, o[1] - st.by) for o in st.opps),
                         default=99.0)
            if st.bx > -40.0 and near_d < P.ADVANCE_PRESS:
                out.append(self.advance(st))
        return out

    # ---- expected utility of one step (chance node)
    def step_value(self, pc, csum, acc, cand, st, hold_mult=1.0):
        g = P.GAMMA ** csum
        kind = cand.meta[0]
        p, c, child = cand.p, cand.c, cand.child
        if kind == "clear":
            return acc + pc * g * P.CLEAR_VALUE, None
        if kind == "shoot":
            gs = g * P.GAMMA ** c
            return acc + pc * gs * (p + (1.0 - p) * P.V_FAIL_SHOT), None
        fx, fy = cand.fail_xy
        acc2 = acc + pc * g * (1.0 - p) * self.ev.loss(fx, fy)
        if kind in ("hold", "advance"):
            mult = P.HOLD_DISCOUNT * hold_mult if kind == "hold" else 1.0
            return acc2 + pc * g * p * P.GAMMA ** c * mult * self.ev.V(child), None
        pc2, c2 = pc * p, csum + c
        return acc2 + pc2 * (P.GAMMA ** c2) * self.ev.V(child), (pc2, c2, acc2)

    def pass_value(self, st, unum, tx, ty):
        best = -1e9
        for cand in self.pass_options(st, unum, tx, ty, tx, ty, False):
            e, _ = self.step_value(1.0, 0, 0.0, cand, st)
            best = max(best, e)
        return best

    # ---- greedy best-first over chains (anytime)
    def chain_search(self, root, cands, deadline, hold_mult):
        best = []
        heap, tie, expanded = [], 0, 0
        for i, cand in enumerate(cands):
            e, nxt = self.step_value(1.0, 0, 0.0, cand, root, hold_mult)
            best.append(e)
            if nxt is not None and P.D_MAX > 1:
                tie += 1
                heapq.heappush(heap, (-e, tie, 1, cand.child, nxt, i))
        clock = time.perf_counter
        while heap and expanded < P.N_MAX:
            if expanded % P.CHECK_EVERY == 0 and clock() > deadline:
                break
            _, _, depth, st, (pc, cs, acc), i = heapq.heappop(heap)
            expanded += 1
            kids = []
            for cand in self.actions(st, False):
                e, nxt = self.step_value(pc, cs, acc, cand, st)
                if e > best[i]:
                    best[i] = e
                if nxt is not None and depth + 1 < P.D_MAX:
                    kids.append((e, cand.child, nxt))
            if kids:
                kids.sort(key=lambda k: -k[0])
                for e, child, nxt in kids[:P.CAP]:
                    tie += 1
                    heapq.heappush(heap, (-e, tie, depth + 1, child, nxt, i))
        return best, expanded

    # ---- A* dribble route to the shooting zone
    def astar_route(self, st, deadline):
        ph = self.ph
        cell = P.ASTAR_CELL
        sx, sy = st.bx, st.by
        goal_c = (ph.hl - 11.0, 0.0)

        def to_ij(x, y):
            return int(round((x - sx) / cell)), int(round((y - sy) / cell))

        def xy(ij):
            return sx + ij[0] * cell, sy + ij[1] * cell

        def is_goal(x, y):
            return math.hypot(x - goal_c[0], y) < 7.0 or (
                x > ph.pen_x and abs(y) < 9.0)

        def h(x, y):
            return max(0.0, math.hypot(x - goal_c[0], y - goal_c[1]) - 7.0) / P.DRIB_SPEED

        reach_cache = {}

        def reach_at(ij):
            r = reach_cache.get(ij)
            if r is None:
                x, y = xy(ij)
                r = min((self.reach(o, x, y) for o in st.opps), default=99.0)
                reach_cache[ij] = r
            return r

        start = (0, 0)
        frontier = [(h(sx, sy), 0.0, 0.0, start)]
        came, g_cost = {start: None}, {start: 0.0}
        found, n = None, 0
        steps = [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, -1)]
        while frontier and n < P.ASTAR_NODES:
            if n % 32 == 0 and time.perf_counter() > deadline:
                break
            f, g, t, ij = heapq.heappop(frontier)
            if g > g_cost.get(ij, 1e18) + 1e-9:
                continue
            n += 1
            x, y = xy(ij)
            if ij != start and is_goal(x, y):
                found = ij
                break
            for di, dj in steps:
                nij = (ij[0] + di, ij[1] + dj)
                nx, ny = xy(nij)
                if not ph.inside(nx, ny, 1.5):
                    continue
                step = cell * math.hypot(di, dj)
                t2 = t + step / P.DRIB_SPEED
                m = reach_at(nij) - min(t2, P.ASTAR_HORIZON)
                if m < -1.0:
                    continue                    # an opponent owns this cell
                g2 = g + step / P.DRIB_SPEED + P.ASTAR_RISK_W * max(0.0, P.ASTAR_SAFE - m)
                if g2 < g_cost.get(nij, 1e18) - 1e-9:
                    g_cost[nij] = g2
                    came[nij] = ij
                    heapq.heappush(frontier, (g2 + h(nx, ny), g2, t2, nij))
        if found is None:
            return None, n
        path = []
        ij = found
        while ij is not None:
            path.append(xy(ij))
            ij = came[ij]
        path.reverse()
        for x, y in path[1:]:                   # first point ~7 m along
            if math.hypot(x - sx, y - sy) >= P.ASTAR_WAYPOINT:
                return (x, y), n
        return path[-1], n

    # ---- MCTS over the best root actions
    def mcts_refine(self, root, cands, chain_vals, deadline, model, rng, hold_mult):
        order = sorted(range(len(cands)), key=lambda i: -chain_vals[i])
        arms = [i for i in order if cands[i].meta[0] in
                ("pass", "dribble", "route", "shoot")][:P.MCTS_ARMS]
        if len(arms) < 2:
            return {}, 0
        follow = {}
        for i in arms:
            ch = cands[i].child
            if ch is None:
                follow[i] = []
                continue
            acts = self.actions(ch, False)
            scored = []
            for cand in acts:
                e, _ = self.step_value(1.0, 0, 0.0, cand, ch)
                scored.append((e, cand))
            scored.sort(key=lambda t: -t[0])
            follow[i] = [c for _, c in scored[:P.FOLLOW_UPS]]
            if time.perf_counter() > deadline:
                return {}, 0
        units = [o for o in root.opps]
        n = {i: 0 for i in arms}
        w = {i: 0.0 for i in arms}
        sims = 0
        while sims < P.MCTS_MAX_SIMS and time.perf_counter() < deadline:
            sims += 1
            total = sims
            a = None
            for i in arms:
                if n[i] == 0:
                    a = i
                    break
            if a is None:
                lnN = math.log(total)
                a = max(arms, key=lambda i: w[i] / n[i]
                        + P.MCTS_C * math.sqrt(lnN / n[i]))
            # opponent speed ~ Beta around its learned mean (concentration 8)
            factors = [1.0 if o[2] else max(P.OPP_MIN_FACTOR, rng.betavariate(
                max(0.5, o[3] * 8.0), max(0.5, (1.0 - o[3]) * 8.0)))
                for o in units]
            reacts = [rng.uniform(0.3, 1.7) for _ in units]
            v = self._simulate(root, cands[a], follow[a], factors, reacts, rng)
            n[a] += 1
            w[a] += v
        return {i: (w[i] / n[i], n[i]) for i in arms if n[i]}, sims

    def _simulate(self, root, cand, follow, factors, reacts, rng):
        """One sample: does the action succeed under sampled opponents and
        execution noise, and what is the best follow-up worth?"""
        kind = cand.meta[0]
        m = self.margin(root.opps, cand.pts, cand.t_end, factors, reacts) \
            + rng.gauss(0.0, P.EXEC_NOISE)
        if kind == "shoot":
            ok = m > P.SHOT_C - 0.5 and rng.random() < max(
                0.3, 1.0 - 0.012 * max(0.0, math.hypot(self.ph.hl - root.bx, root.by) - 12.0))
            return P.GAMMA ** cand.c * (1.0 if ok else P.V_FAIL_SHOT)
        if m < 0.0:
            fx, fy = cand.fail_xy
            return self.ev.loss(fx, fy)
        ch = cand.child
        here = self.ev.V(ch)
        best = here
        for f in follow:
            if f.meta[0] == "shoot" or f.pts is None:
                e, _ = self.step_value(1.0, 0, 0.0, f, ch)
            else:
                m2 = self.margin(ch.opps, f.pts, f.t_end) + rng.gauss(0.0, P.EXEC_NOISE)
                if m2 < 0.0:
                    e = self.ev.loss(*f.fail_xy)
                else:
                    e = P.GAMMA ** f.c * self.ev.V(f.child)
            best = max(best, e)
        return P.GAMMA ** cand.c * best


# ---------------------------------------------------------------- the agent
class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        self._phys = None
        self._ev = None
        self._zone_v = zone_values()     # value iteration once, at start-up
        self._model = OpponentModel()
        self._commit = None              # (cycle, meta)
        self._hold_streak = 0
        self._unmark_cache = None
        self._marks_cache = None
        self._rng = None

    # ---- time manager
    def _budget_ms(self, agent, k):
        b = agent.budget
        rem = b.remaining_ms
        if rem == float("inf"):
            rem = 75000.0
        left = max(50, P.MATCH_CYCLES - int(agent.wm.cycle))
        ms = min(P.B_MAX_MS, max(P.B_MIN_MS, k * (rem - P.RESERVE_MS) / left))
        ms = min(ms, b.decision_remaining_ms - P.WALL_SAFETY_MS)
        if rem < P.RESERVE_MS:
            ms = min(ms, P.B_MIN_MS)
        return max(0.0, ms)

    def _ph(self, agent):
        if self._phys is None:
            self._phys = Phys(agent.server_params)
            self._ev = Evaluator(self._phys, self._zone_v)
            self._rng = random.Random(7919 * agent.wm.self.uniform_number)
        return self._phys

    def _planner(self, agent):
        ph = self._ph(agent)
        return Planner(ph, self._ev, agent.wm.offside_line_x)

    # ---- on ball
    def on_ball(self, agent) -> None:
        try:
            self._model.update(agent.wm)
        except Exception:
            pass
        try:
            if self._plan(agent):
                return
        except Exception:
            pass
        try:
            super().on_ball(agent)       # reflex fallback (starter)
        except Exception:
            Action.clear(agent)

    def _plan(self, agent):
        wm = agent.wm
        ms = self._budget_ms(agent, P.K_ONBALL)
        if ms <= 0.5:
            return False
        t0 = time.perf_counter()
        end = t0 + ms / 1000.0
        pl = self._planner(agent)
        root = snapshot(agent, self._model)
        hold_mult = P.HOLD_STREAK ** self._hold_streak

        # 1) A* dribble route (its first waypoint becomes a candidate)
        extra, a_nodes = [], 0
        if root.bx > P.ASTAR_MIN_X:
            wp, a_nodes = pl.astar_route(root, t0 + ms * P.SHARE_ASTAR / 1000.0)
            if wp is not None:
                c = pl.dribble_to(root, wp[0], wp[1], tag="route")
                if c is not None:
                    extra.append(c)

        # 2) best-first chain search
        cands = pl.actions(root, True, extra)
        chain_vals, nexp = pl.chain_search(
            root, cands, t0 + ms * (P.SHARE_ASTAR + P.SHARE_CHAIN) / 1000.0, hold_mult)

        # 3) MCTS over the best root actions with the remaining time
        stats, sims = pl.mcts_refine(root, cands, chain_vals, end, self._model,
                                     self._rng, hold_mult)
        score = list(chain_vals)
        for i, (mean, n) in stats.items():
            score[i] = (n * mean + P.MCTS_PRIOR * chain_vals[i]) / (n + P.MCTS_PRIOR)

        # holding is a last resort: only when pressed or nothing is positive
        near = min((math.hypot(o[0] - root.bx, o[1] - root.by) for o in root.opps),
                   default=99.0)
        best_other = max((score[i] for i, c in enumerate(cands)
                          if c.meta[0] != "hold"), default=-1e9)
        for i, c in enumerate(cands):
            if c.meta[0] == "hold" and near > P.HOLD_NEAR and best_other > 0.0:
                score[i] = -1e9
        bi = max(range(len(cands)), key=lambda i: (score[i], -i))
        meta, val = cands[bi].meta, score[bi]

        # commitment: keep last cycle's kick if still close in value
        cyc = int(wm.cycle)
        if self._commit is not None and self._commit[0] == cyc - 1:
            old = self._commit[1]
            if old[0] in ("pass", "shoot"):
                for i, c in enumerate(cands):
                    m = c.meta
                    if (m[0] == old[0] and abs(m[1] - old[1]) < 0.6
                            and abs(m[2] - old[2]) < 0.6):
                        if score[i] + P.COMMIT_DELTA >= val:
                            meta, val = m, score[i]
                        break
        self._hold_streak = self._hold_streak + 1 if meta[0] == "hold" else 0
        self._commit = (cyc, meta)
        self._execute(agent, meta)
        note(agent, meta[0], v=round(val, 4), n=nexp, sims=sims, astar=a_nodes,
             ms=round(ms, 1), opp=round(self._model.mean_factor(), 2))
        return True

    def _execute(self, agent, meta):
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
            Action.kick(agent, Vector2D(meta[1], meta[2]), meta[3], accept_late=5)
        elif kind in ("dribble", "route"):
            Action.dribble(agent, Vector2D(meta[1], meta[2]))
            Action.hold(agent)
        elif kind == "advance":
            Action.advance_ball(agent)
        elif kind == "hold":
            Action.hold(agent)
        Action.clear(agent)              # last line: cannot fail

    # ---- off ball
    def off_ball(self, agent) -> None:
        try:
            self._model.update(agent.wm)
        except Exception:
            pass
        try:
            if self._off(agent):
                return
        except Exception:
            pass
        try:
            super().off_ball(agent)
        except Exception:
            Action.go_to_point(agent, Tools.home_position(agent))

    def _rank(self, agent):
        """Field players sorted by (reach steps, unum); same on every player."""
        wm = agent.wm
        rows = [(int(wm.intercept_table.self_reach_steps), wm.self.uniform_number)]
        for t in wm.teammates:
            if t.uniform_number > 0 and not t.is_goalie:
                rows.append((int(t.ball_reach_steps), t.uniform_number))
        rows.sort()
        return [u for _, u in rows]

    def _off(self, agent):
        wm = agent.wm
        me = wm.self
        my = me.uniform_number
        ball = wm.ball.position
        sp = agent.server_params
        it = wm.intercept_table
        rank = self._rank(agent)
        pos = rank.index(my) if my in rank else len(rank)
        power = self.dash_power(agent)
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        cheap = agent.budget.remaining_ms < P.RESERVE_MS
        home = Tools.home_position(agent)
        self._hold_streak = 0

        # opponent has the ball: tackle, press, OMAM marking, cover
        if theirs and not ours:
            if me.tackle_probability >= 0.75 and ball.dist(me.position) < 2.0:
                goal = Vector2D(sp.pitch_half_length, 0.0)
                rel = (goal - me.position).th().degree() - me.body_direction
                rel = (rel + 180.0) % 360.0 - 180.0
                Action.tackle(agent, rel, foul=False)
                note(agent, "tackle", p=round(me.tackle_probability, 2))
                return True
            if pos == 0:
                Action.intercept(agent)
                note(agent, "press")
                return True
            if not cheap:
                marks = self._omam_marks(agent, rank)
                if my in marks:
                    Action.go_to_point(agent, marks[my], max_dash_power=100.0)
                    note(agent, "mark")
                    return True
            if pos == 1:
                block = self.block_point(agent)
                if block is not None:
                    Action.go_to_point(agent, block, max_dash_power=100.0)
                    note(agent, "cover")
                    return True

        # loose ball: the fastest of us goes (ranked trap point), next covers
        if not ours and not theirs:
            if pos == 0:
                trap = None if cheap else self._trap_point(agent)
                if trap is not None:
                    Action.go_to_point(agent, trap, max_dash_power=100.0,
                                       distance_threshold=0.5)
                    Action.intercept(agent)
                    note(agent, "trap")
                else:
                    Action.intercept(agent)
                    note(agent, "chase")
                return True
            our_best = min(int(it.self_reach_steps), int(it.first_teammate_reach_steps))
            if pos == 1 and int(it.first_opponent_reach_steps) < our_best:
                block = self.block_point(agent)
                if block is not None:
                    Action.go_to_point(agent, block, max_dash_power=100.0)
                    note(agent, "cover")
                    return True

        if ours:
            tgt = home if cheap else self._unmark(agent, home)
            Action.go_to_point(agent, tgt, max_dash_power=power)
            note(agent, "unmark")
            return True
        if theirs:
            mode = "defend"
        elif ball.x() > 10.0 or wm.our_team_score < wm.their_team_score:
            mode = "aggressive"
        else:
            mode = "conservative"
        tgt = self._grid_position(agent, home, mode)
        Action.go_to_point(agent, tgt, max_dash_power=power)
        note(agent, "grid", mode=mode)
        return True

    # ---- CYRUS2022 unmarking: predicted-pass tree + reachable receive point
    def _unmark(self, agent, home):
        wm = agent.wm
        cyc = int(wm.cycle)
        if self._unmark_cache is not None and self._unmark_cache[0] >= cyc - 2:
            return self._unmark_cache[1]
        pl = self._planner(agent)
        ph = self._phys
        my = wm.self.uniform_number
        mates = [t for t in wm.teammates if t.uniform_number > 0]
        if not mates:
            return home
        holder = min(mates, key=lambda t: t.dist_from_ball)
        base = snapshot(agent, self._model)
        root = State(base.bx, base.by, holder.uniform_number, base.mates, base.opps)

        def predicted(st):
            """Receivers the holder's own planner would pick, best first."""
            best = {}
            for cand in pl.passes(st, False):
                e, _ = pl.step_value(1.0, 0, 0.0, cand, st)
                r = cand.meta[4]
                if r not in best or e > best[r][0]:
                    best[r] = (e, cand.child)
            return sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))

        passer_state = root
        first = predicted(root)
        if first and first[0][0] != my:
            for r, (e, child) in first[:2]:
                if r == my:
                    break
                second = predicted(child)
                if second and second[0][0] == my:
                    passer_state = child
                    break

        mx, myy = wm.self.position.x(), wm.self.position.y()
        cands = [(mx, myy)]
        for rad in (3.0, 6.0):
            for k in range(8):
                a = k * math.pi / 4.0
                cands.append((mx + rad * math.cos(a), myy + rad * math.sin(a)))
        offx = wm.offside_line_x
        best, best_v = (home.x(), home.y()), -1e9
        for cx, cy in cands:
            if not ph.inside(cx, cy, 2.0):
                continue
            if cx > offx - 0.5 and cx > 0.0:
                continue
            t_me = math.hypot(cx - mx, cy - myy) / ph.vmax + 1.0
            t_op = min((pl.reach(o, cx, cy) for o in passer_state.opps), default=99.0)
            if t_op <= t_me:
                continue                      # an opponent gets there first
            mates_c = dict(passer_state.mates)
            mates_c[my] = (cx, cy, False)
            st_c = State(passer_state.bx, passer_state.by, passer_state.holder,
                         mates_c, passer_state.opps)
            v = pl.pass_value(st_c, my, cx, cy)
            v -= 0.002 * math.hypot(cx - home.x(), cy - home.y())
            if v > best_v:
                best, best_v = (cx, cy), v
        tgt = Vector2D(best[0], best[1])
        self._unmark_cache = (cyc, tgt)
        return tgt

    # ---- CYRUS OMAM marking: assignment with top-3 tasks per defender
    def _omam_marks(self, agent, rank):
        wm = agent.wm
        cyc = int(wm.cycle)
        if self._marks_cache is not None and self._marks_cache[0] == cyc:
            return self._marks_cache[1]
        hl = agent.server_params.pitch_half_length
        pos = {wm.self.uniform_number: (wm.self.position.x(), wm.self.position.y())}
        for t in wm.teammates:
            if t.uniform_number > 0 and not t.is_goalie:
                pos[t.uniform_number] = (t.position.x(), t.position.y())
        presser = rank[0] if rank else None
        defenders = sorted(u for u in pos if u != presser)
        opps = [o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie]
        if not opps or not defenders:
            self._marks_cache = (cyc, {})
            return {}
        holder = min(opps, key=lambda o: o.dist_from_ball)
        tasks = []
        for o in opps:
            if o.uniform_number == holder.uniform_number:
                continue
            ox, oy = o.position.x(), o.position.y()
            dgoal = math.hypot(ox + hl, oy)
            if ox > 10.0 and dgoal > 35.0:
                continue
            dx, dy = -hl - ox, -oy
            d = math.hypot(dx, dy) or 1.0
            tasks.append((o.uniform_number, (ox + 1.5 * dx / d, oy + 1.5 * dy / d),
                          dgoal < 35.0))
        if not tasks:
            self._marks_cache = (cyc, {})
            return {}
        options = {}
        for u in defenders:
            x, y = pos[u]
            options[u] = sorted(tasks, key=lambda t: (math.hypot(t[1][0] - x,
                                                                t[1][1] - y), t[0]))[:3]
        order = sorted(defenders, key=lambda u: (len(options[u]), u))
        best = [(-1, 0.0), {}]

        def search(i, used, n_danger, cost, plan):
            if i == len(order):
                key = (n_danger, -cost)
                if key > best[0]:
                    best[0], best[1] = key, dict(plan)
                return
            u = order[i]
            x, y = pos[u]
            for unum, mark, danger in options[u]:
                if unum in used:
                    continue
                plan[u] = mark
                used.add(unum)
                search(i + 1, used, n_danger + int(danger),
                       cost + math.hypot(mark[0] - x, mark[1] - y), plan)
                used.discard(unum)
                del plan[u]
            search(i + 1, used, n_danger, cost + 8.0, plan)

        search(0, set(), 0, 0.0, {})
        marks = {u: Vector2D(m[0], m[1]) for u, m in best[1].items()}
        self._marks_cache = (cyc, marks)
        return marks

    # ---- HELIOS2023-style ranked ball trapping
    def _trap_point(self, agent):
        wm = agent.wm
        ph = self._ph(agent)
        v = wm.ball.velocity
        if v.r() < 0.4:
            return None
        x, y = wm.ball.position.x(), wm.ball.position.y()
        vx, vy = v.x(), v.y()
        mx, myy = wm.self.position.x(), wm.self.position.y()
        opps = [(o.position.x(), o.position.y(),
                 self._model.factor(o.uniform_number, bool(o.is_goalie)))
                for o in wm.opponents if o.uniform_number > 0]
        best = None
        for t in range(1, 40):
            x, y = x + vx, y + vy
            vx, vy = vx * ph.decay, vy * ph.decay
            if abs(x) > ph.hl or abs(y) > ph.hw:
                break
            t_me = max(0.0, (math.hypot(x - mx, y - myy) - ph.ka) / ph.vmax) + 1.0
            if t_me > t:
                continue
            t_op = 99.0
            for ox, oy, f in opps:
                d = math.hypot(x - ox, y - oy) - ph.ka
                t_op = min(t_op, d / (ph.vmax * f) + P.OPP_REACT if d > 0.0 else 0.0)
            safety = t_op - t
            if safety < 2.0:
                continue
            score = 0.1 * x + 0.3 * min(safety, 6.0) - 0.15 * t
            if best is None or score > best[0]:
                best = (score, x, y, t)
        if best is None or best[3] <= int(wm.intercept_table.self_reach_steps) + 1:
            return None
        return Vector2D(best[1], best[2])

    # ---- YuShan2022 grid of candidate targets around the formation slot
    def _grid_position(self, agent, home, mode):
        wm = agent.wm
        bx, by = wm.ball.position.x(), wm.ball.position.y()
        hl = agent.server_params.pitch_half_length
        offx = wm.offside_line_x
        opps = [(o.position.x(), o.position.y()) for o in wm.opponents
                if o.uniform_number > 0]
        hx, hy = home.x(), home.y()
        w = P.GRID_W[mode]
        gx, gy = -hl - bx, -by
        g2 = gx * gx + gy * gy
        best, best_s = (hx, hy), -1e9
        for i in range(-3, 4):
            for j in range(-3, 4):
                x, y = hx + 1.5 * i, hy + 1.5 * j
                if abs(x) > 51.0 or abs(y) > 32.0:
                    continue
                s = w[0] * math.hypot(x - hx, y - hy) + w[1] * x
                if mode != "defend":
                    if x > offx - 0.5 and x > 0.0:
                        continue
                    free = min((math.hypot(ox - x, oy - y) for ox, oy in opps),
                               default=10.0)
                    dx, dy = x - bx, y - by
                    L2 = dx * dx + dy * dy
                    lane = 5.0
                    if L2 > 1e-6:
                        for ox, oy in opps:
                            u = ((ox - bx) * dx + (oy - by) * dy) / L2
                            if 0.0 < u < 1.0:
                                lane = min(lane, math.hypot(bx + u * dx - ox,
                                                            by + u * dy - oy))
                    s += w[2] * min(free, 10.0) + w[3] * lane
                else:
                    u = 0.0
                    if g2 > 1e-6:
                        u = max(0.0, min(1.0, ((x - bx) * gx + (y - by) * gy) / g2))
                    s += w[2] * math.hypot(bx + u * gx - x, by + u * gy - y)
                if s > best_s:
                    best, best_s = (x, y), s
        return Vector2D(best[0], best[1])


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None


class Baseline(StarterAgent):
    """The unmodified starter, kept so evaluation can play against it."""
    pass

# STARTER CODE ENDS HERE #
