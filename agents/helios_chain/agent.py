# Agent1 - HELIOS-style action-chain team (separate agent).
#
# On ball : HELIOS "action chain" planning. Kick sequences (pass, through
#           pass, dribble, shoot) are generated with a STRICT feasibility
#           check (every opponent arrives later than the ball, plus a
#           buffer) and searched best-first (AIMA 3.5) up to depth 3. Only the
#           FINAL state of a chain is scored, by a HELIOS-like field
#           evaluator. Chains that break the intended tactics are pruned by a
#           linear classifier (HELIOS2017 used an SVM; here a perceptron-style
#           w.x + b > 0 rule with hand-set weights, AIMA/Goodfellow ch.6).
# Off ball: HELIOS basic move: fastest player intercepts, others follow the
#           formation (the framework's formation is the Delaunay-triangulation
#           model of HELIOS base). Defence: press, block the shooting line,
#           mark the nearest free attacker.
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
MAX_DEPTH = 3
MAX_NODES = 400
PASS_BUF = 1.0          # cycles every opponent must be later than the ball
DRIB_BUF = 1.5
SHOT_BUF = 0.5
THROUGH = (5.0, 9.0)    # through-pass lead distances
DRIB_DIRS, DRIB_LEN, DRIB_SPEED = 8, 5.0, 0.65
SHOOT_BONUS = 1000.0
# pruning classifier: features of a chain -> prune if W.x + B > 0
PRUNE_W = (1.5,   # backward passes made in the attacking half
           3.0,   # chain ends inside our own penalty area
           1.0,   # lateral/backward pass inside our third
           0.6)   # dribble that loses ground in our half
PRUNE_B = -1.2


class Node:
    __slots__ = ("bx", "by", "holder", "mates", "opps", "depth", "first",
                 "feat", "prev")

    def __init__(self, bx, by, holder, mates, opps, depth, first, feat, prev):
        self.bx, self.by, self.holder = bx, by, holder
        self.mates, self.opps = mates, opps
        self.depth, self.first, self.feat, self.prev = depth, first, feat, prev


class ChainPlanner:
    def __init__(self, geo, offx):
        self.g = geo
        self.offx = offx

    def evaluate(self, n):
        """HELIOS-like field evaluator: territory + closeness to goal."""
        g = self.g
        d = math.hypot(g.hl - n.bx, n.by)
        return (n.bx + g.hl) + max(0.0, 40.0 - d)

    def pruned(self, feat):
        z = PRUNE_B + sum(w * f for w, f in zip(PRUNE_W, feat))
        return z > 0.0

    def _moved(self, opps, x, y, cycles):
        step = self.g.vmax * cycles * 0.6
        out = []
        for ox, oy, gk in opps:
            d = math.hypot(x - ox, y - oy)
            mv = min(step, max(0.0, d - 1.0))
            if d > 1e-6:
                ox, oy = ox + (x - ox) / d * mv, oy + (y - oy) / d * mv
            out.append((ox, oy, gk))
        return out

    def _child(self, n, x, y, holder, cycles, meta, feat_add):
        mates = dict(n.mates)
        mates[holder] = (x, y, mates[holder][2])
        feat = tuple(a + b for a, b in zip(n.feat, feat_add))
        return Node(x, y, holder, mates, self._moved(n.opps, x, y, cycles),
                    n.depth + 1, n.first or meta, feat, n.holder)

    def expand(self, n):
        """-> (children, shot_meta or None). Strictly feasible actions only."""
        g = self.g
        bx, by = n.bx, n.by
        kids = []
        shot = None
        # shoot
        if math.hypot(g.hl - bx, by) < 30.0:
            half = g.goal_w / 2.0 - 0.7
            for i in range(5):
                ty = -half + half * i / 2.0
                dist = math.hypot(g.hl - bx, ty - by)
                k = g.steps_for(dist, g.vball)
                if k is None:
                    continue
                m = g.margin(n.opps, g.kick_path(bx, by, g.hl, ty, g.vball, k))
                if m >= SHOT_BUF and (shot is None or m > shot[0]):
                    shot = (m, ("shoot", g.hl, ty, k))
        # passes and through passes
        for u, (rx, ry, gk) in n.mates.items():
            if u == n.holder or gk or u == n.prev:
                continue
            if rx > self.offx and rx > 0.0:
                continue
            targets = [(rx, ry, 0.0)]
            for ld in THROUGH:
                targets.append((rx + ld, ry, ld))
            for tx, ty, ld in targets:
                if not g.inside(tx, ty) or (ld and tx > self.offx and tx > 0.0):
                    continue
                dist = math.hypot(tx - bx, ty - by)
                if dist < 3.0 or dist > 40.0:
                    continue
                k, v0 = g.pass_plan(dist, 1.3)
                if k is None:
                    continue
                t_recv = (math.hypot(tx - rx, ty - ry) / g.vmax + 1.0) if ld else 0.0
                if t_recv > k + 2:
                    continue
                pts = g.kick_path(bx, by, tx, ty, v0, k)
                if ld and t_recv > k:
                    pts += [(tx, ty)] * int(math.ceil(t_recv - k))
                if g.margin(n.opps, pts) < PASS_BUF:
                    continue
                attack = 1.0 if bx > 0.0 and tx < bx - 2.0 else 0.0
                own_box = 1.0 if (tx < -g.pen_x and abs(ty) < g.pen_hw) else 0.0
                lateral = 1.0 if (bx < -17.5 and tx <= bx + 1.0) else 0.0
                c = int(max(k, math.ceil(t_recv)))
                kids.append(self._child(n, tx, ty, u, c,
                                        ("pass", tx, ty, k, u),
                                        (attack, own_box, lateral, 0.0)))
        # dribble
        base = math.atan2(-by, g.hl - bx)
        for i in range(DRIB_DIRS):
            a = base + 2.0 * math.pi * i / DRIB_DIRS
            tx, ty = bx + DRIB_LEN * math.cos(a), by + DRIB_LEN * math.sin(a)
            if not g.inside(tx, ty):
                continue
            pts = g.carry_path(bx, by, tx, ty, DRIB_SPEED)
            if g.margin(n.opps, pts) < DRIB_BUF:
                continue
            back = 1.0 if (bx < 0.0 and tx < bx - 1.0) else 0.0
            own_box = 1.0 if (tx < -g.pen_x and abs(ty) < g.pen_hw) else 0.0
            kids.append(self._child(n, tx, ty, n.holder, len(pts),
                                    ("dribble", tx, ty),
                                    (0.0, own_box, 0.0, back)))
        return kids, (shot[1] if shot else None)

    def search(self, w, deadline):
        root = Node(w.bx, w.by, w.holder, w.mates, w.opps, 0, None,
                    (0.0, 0.0, 0.0, 0.0), None)
        best_val, best_first = -1e9, None
        heap, tie, expanded = [], 0, 0
        kids, shot = self.expand(root)
        if shot is not None:
            return shot, SHOOT_BONUS, 0
        for c in kids:
            tie += 1
            heapq.heappush(heap, (-self.evaluate(c), tie, c))
        while heap and expanded < MAX_NODES:
            if expanded % 8 == 0 and time.perf_counter() > deadline:
                break
            negv, _, n = heapq.heappop(heap)
            expanded += 1
            if self.pruned(n.feat):
                continue
            v = -negv
            if v > best_val:
                best_val, best_first = v, n.first
            if n.depth >= MAX_DEPTH:
                continue
            kids, shot = self.expand(n)
            if shot is not None:
                v = SHOOT_BONUS - n.depth          # a reachable shot ends it
                if v > best_val:
                    best_val, best_first = v, n.first
                continue
            for c in kids:
                tie += 1
                heapq.heappush(heap, (-self.evaluate(c), tie, c))
        return best_first, best_val, expanded


# -------------------------------------------------------------------- agent
class Agent1(SafeAgent):
    def __init__(self):
        super().__init__()
        self.geo = None

    def _g(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
        return self.geo

    def _on(self, agent):
        ms = budget_ms(agent, 4.0)
        if ms <= 0.5:
            return False
        g = self._g(agent)
        w = World(agent)
        pl = ChainPlanner(g, w.offx)
        first, val, nexp = pl.search(w, time.perf_counter() + ms / 1000.0)
        if first is None:
            near = min((math.hypot(ox - w.bx, oy - w.by) for ox, oy, _ in w.opps),
                       default=99.0)
            first = ("hold",) if near > 3.0 else ("clear",)
        execute(agent, first)
        note(agent, first[0], v=round(val, 1), n=nexp)
        return True

    def _off(self, agent):
        wm = agent.wm
        rank = field_rank(wm)
        me = wm.self.uniform_number
        pos = rank.index(me) if me in rank else 9
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        if theirs and not ours and try_tackle(agent):
            return True
        if not ours and pos == 0:
            Action.intercept(agent)
            note(agent, "intercept")
            return True
        home = Tools.home_position(agent)
        if theirs and pos == 1:
            blk = self.block_point(agent)
            if blk is not None:
                Action.go_to_point(agent, blk, max_dash_power=100.0)
                note(agent, "block")
                return True
        if theirs and wm.ball.position.x() < 0.0:
            tgt = self._mark(agent, rank, home)
            if tgt is not None:
                Action.go_to_point(agent, tgt, max_dash_power=100.0)
                note(agent, "mark")
                return True
        Action.go_to_point(agent, home, max_dash_power=self.dash_power(agent))
        note(agent, "formation")
        return True

    def _mark(self, agent, rank, home):
        """Greedy nearest-free-attacker marking, in rank order (shared)."""
        wm = agent.wm
        hl = agent.server_params.pitch_half_length
        pos = {wm.self.uniform_number: wm.self.position}
        for t in wm.teammates:
            if t.uniform_number > 0:
                pos[t.uniform_number] = t.position
        holder = min((o for o in wm.opponents if o.uniform_number > 0),
                     key=lambda o: o.dist_from_ball, default=None)
        targets = [o for o in wm.opponents if o.uniform_number > 0
                   and not o.is_goalie and o is not holder
                   and o.position.x() < 5.0]
        taken = set()
        for u in rank[2:]:
            if u not in pos or not targets:
                break
            free = [o for o in targets if o.uniform_number not in taken]
            if not free:
                break
            o = min(free, key=lambda o: (o.position.dist(pos[u]), o.uniform_number))
            taken.add(o.uniform_number)
            if u == wm.self.uniform_number:
                to_goal = Vector2D(-hl, 0.0) - o.position
                return o.position + to_goal.set_length_vector(1.5)
        return None


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
