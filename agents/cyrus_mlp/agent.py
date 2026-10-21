# Agent1 - CYRUS-style learned-evaluation team (separate agent).
#
# On ball : every candidate kick (pass, lead pass, dribble) is scored by a
#           small FEEDFORWARD NEURAL NETWORK (8 inputs -> 6 tanh -> 1 sigmoid,
#           AIMA ch.21 / Goodfellow ch.6), standing in for CYRUS's pass
#           prediction DNN over engineered features. The weights are
#           hand-initialised so each hidden unit encodes one engineered
#           feature group; they are meant to be refit by backpropagation +
#           mini-batch SGD on logged passes.
# Off ball: CYRUS2022 unmarking: the same network predicts which pass the
#           holder (and the predicted next holder) will make, a best-first
#           predicted-pass tree of depth 2 finds who will pass to me, and I
#           move to the reachable point that network rates highest.
#           Defence uses OMAM-style marking: an assignment search where each
#           defender keeps its top-3 tasks and solutions are ranked by
#           dangerous attackers covered, then total distance (backtracking
#           search with MRV ordering, AIMA ch.6).
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


# ------------------------------------------------------------ the network
# inputs: 0 margin/5   1 dist/30   2 progress/30   3 target-goal dist/50
#         4 free space/10   5 opponents in the lane cone/3
#         6 target x/52.5   7 lead-or-dribble flag
W1 = [[1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # safety
      [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0],   # closeness to goal
      [0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0],    # space at the target
      [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0],   # crowded lane
      [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # progress up the pitch
      [0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]]   # not too long
B1 = [0.0, 1.2, -0.4, 0.3, 0.0, 0.9]
W2 = [1.4, 0.8, 0.4, 0.5, 0.5, 0.3]
B2 = -0.6


def mlp(x):
    h = [math.tanh(sum(w * v for w, v in zip(row, x)) + b)
         for row, b in zip(W1, B1)]
    return sig(sum(w * v for w, v in zip(W2, h)) + B2)


def features(g, opps, bx, by, tx, ty, m, lead):
    dist = math.hypot(tx - bx, ty - by)
    ang = math.atan2(ty - by, tx - bx)
    cone = 0
    for ox, oy, _ in opps:
        d = math.hypot(ox - bx, oy - by)
        if 0.5 < d < dist + 3.0:
            da = abs((math.atan2(oy - by, ox - bx) - ang + math.pi)
                     % (2 * math.pi) - math.pi)
            if da < math.radians(15.0):
                cone += 1
    free = min((math.hypot(ox - tx, oy - ty) for ox, oy, _ in opps), default=10.0)
    return [max(-2.0, min(2.0, m / 5.0)), dist / 30.0, (tx - bx) / 30.0,
            math.hypot(g.hl - tx, ty) / 50.0, min(free, 10.0) / 10.0,
            cone / 3.0, tx / 52.5, 1.0 if lead else 0.0]


def pass_quality(g, opps, bx, by, tx, ty, rx, ry, lead):
    """-> (quality, cycles, first-kick n) or None if impossible."""
    dist = math.hypot(tx - bx, ty - by)
    if dist < 3.0 or dist > 40.0 or not g.inside(tx, ty):
        return None
    n, v0 = g.pass_plan(dist, 1.2 if not lead else 1.5)
    if n is None:
        return None
    t_recv = math.hypot(tx - rx, ty - ry) / g.vmax + 1.0 if lead else 0.0
    if t_recv > n + 3:
        return None
    pts = g.kick_path(bx, by, tx, ty, v0, n)
    if t_recv > n:
        pts += [(tx, ty)] * int(math.ceil(t_recv - n))
    m = g.margin(opps, pts)
    return mlp(features(g, opps, bx, by, tx, ty, m, lead)), n


def pass_options(g, opps, bx, by, holder, mates, offx):
    """All passes from the ball for `holder`: [(q, meta)]."""
    out = []
    for u, (rx, ry, gk) in mates.items():
        if u == holder or gk or (rx > offx and rx > 0.0):
            continue
        ang = math.atan2(-ry, g.hl - rx)
        tg = [(rx, ry, False), (rx + 5.0, ry, True),
              (rx + 5.0 * math.cos(ang), ry + 5.0 * math.sin(ang), True)]
        for tx, ty, lead in tg:
            if lead and tx > offx and tx > 0.0:
                continue
            r = pass_quality(g, opps, bx, by, tx, ty, rx, ry, lead)
            if r is not None:
                out.append((r[0], ("pass", tx, ty, r[1], u)))
    return out


# -------------------------------------------------------------------- agent
class Agent1(SafeAgent):
    def __init__(self):
        super().__init__()
        self.geo = None
        self.unmark = None        # (cycle, Vector2D)
        self.marks = None         # (cycle, {unum: Vector2D})
        self.commit = None        # (cycle, meta)

    def _g(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
        return self.geo

    # ---------------------------------------------------------- on ball
    def _on(self, agent):
        if budget_ms(agent, 2.0) <= 0.5:
            return False
        g = self._g(agent)
        w = World(agent)
        bx, by = w.bx, w.by
        cands = pass_options(g, w.opps, bx, by, w.me, w.mates, w.offx)
        # shoot: margin-based probability, offset so a good shot dominates
        if math.hypot(g.hl - bx, by) < 30.0:
            half = g.goal_w / 2.0 - 0.7
            for i in range(5):
                ty = -half + half * i / 2.0
                dist = math.hypot(g.hl - bx, ty - by)
                k = g.steps_for(dist, g.vball)
                if k is None:
                    continue
                m = g.margin(w.opps, g.kick_path(bx, by, g.hl, ty, g.vball, k))
                p = sig(1.2 * (m - 1.0)) * max(0.3, 1.0 - 0.012 * max(0.0, dist - 12.0))
                cands.append((0.3 + p, ("shoot", g.hl, ty, k)))
        # dribble, scored by the same network with the dribble flag
        base = math.atan2(-by, g.hl - bx)
        for i in range(8):
            a = base + math.pi * i / 4.0
            tx, ty = bx + 5.0 * math.cos(a), by + 5.0 * math.sin(a)
            if not g.inside(tx, ty):
                continue
            m = g.margin(w.opps, g.carry_path(bx, by, tx, ty, 0.65)) - 1.0
            cands.append((mlp(features(g, w.opps, bx, by, tx, ty, m, True)) - 0.05,
                          ("dribble", tx, ty)))
        near = min((math.hypot(ox - bx, oy - by) for ox, oy, _ in w.opps),
                   default=99.0)
        cands.append((0.35 if near > 4.0 else 0.1, ("hold",)))
        cands.append((0.15 if bx < -17.5 else 0.05, ("clear",)))
        q, meta = max(cands, key=lambda c: c[0])
        cyc = int(agent.wm.cycle)
        if self.commit and self.commit[0] == cyc - 1 and self.commit[1][0] == "pass":
            old = self.commit[1]
            for q2, m2 in cands:
                if (m2[0] == "pass" and abs(m2[1] - old[1]) < 0.6
                        and abs(m2[2] - old[2]) < 0.6 and q2 + 0.05 >= q):
                    q, meta = q2, m2
                    break
        self.commit = (cyc, meta)
        execute(agent, meta)
        note(agent, meta[0], q=round(q, 3))
        return True

    # ---------------------------------------------------------- off ball
    def _off(self, agent):
        wm = agent.wm
        rank = field_rank(wm)
        me = wm.self.uniform_number
        pos = rank.index(me) if me in rank else 9
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        home = Tools.home_position(agent)
        cheap = budget_ms(agent, 0.3) <= 0.5
        if theirs and not ours and try_tackle(agent):
            return True
        if not ours and pos == 0:
            Action.intercept(agent)
            note(agent, "intercept")
            return True
        if ours and not cheap:
            Action.go_to_point(agent, self._unmark(agent, home),
                               max_dash_power=self.dash_power(agent))
            note(agent, "unmark")
            return True
        if theirs and not cheap:
            marks = self._omam(agent, rank)
            if me in marks:
                Action.go_to_point(agent, marks[me], max_dash_power=100.0)
                note(agent, "mark")
                return True
            if pos == 1:
                blk = self.block_point(agent)
                if blk is not None:
                    Action.go_to_point(agent, blk, max_dash_power=100.0)
                    note(agent, "block")
                    return True
        Action.go_to_point(agent, home, max_dash_power=self.dash_power(agent))
        note(agent, "home")
        return True

    def _unmark(self, agent, home):
        wm = agent.wm
        cyc = int(wm.cycle)
        if self.unmark and self.unmark[0] >= cyc - 2:
            return self.unmark[1]
        g = self._g(agent)
        w = World(agent)
        me = w.me
        mates = [t for t in wm.teammates if t.uniform_number > 0]
        if not mates:
            return home
        holder = min(mates, key=lambda t: t.dist_from_ball).uniform_number

        def predict(h, bx, by, mts):
            opts = pass_options(g, w.opps, bx, by, h, mts, w.offx)
            best = {}
            for q, meta in opts:
                r = meta[4]
                if r not in best or q > best[r][0]:
                    best[r] = (q, meta)
            return sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))

        # predicted-pass tree, best first, depth 2
        passer, pbx, pby = holder, w.bx, w.by
        first = predict(holder, w.bx, w.by, w.mates)
        if first and first[0][0] != me:
            for r, (q, meta) in first[:2]:
                if r == me:
                    break
                mts = dict(w.mates)
                mts[r] = (meta[1], meta[2], False)
                second = predict(r, meta[1], meta[2], mts)
                if second and second[0][0] == me:
                    passer, pbx, pby = r, meta[1], meta[2]
                    break
        mx, my, _ = w.mates[me]
        best, best_v = (home.x(), home.y()), -1e9
        pts = [(mx, my)] + [(mx + r * math.cos(k * math.pi / 4.0),
                             my + r * math.sin(k * math.pi / 4.0))
                            for r in (3.0, 6.0) for k in range(8)]
        for cx, cy in pts:
            if not g.inside(cx, cy, 2.0) or (cx > w.offx - 0.5 and cx > 0.0):
                continue
            t_me = math.hypot(cx - mx, cy - my) / g.vmax + 1.0
            t_op = min((g.reach_t(ox, oy, cx, cy, gk) for ox, oy, gk in w.opps),
                       default=99.0)
            if t_op <= t_me:
                continue
            r = pass_quality(g, w.opps, pbx, pby, cx, cy, cx, cy, False)
            if r is None:
                continue
            v = r[0] - 0.01 * math.hypot(cx - home.x(), cy - home.y())
            if v > best_v:
                best, best_v = (cx, cy), v
        tgt = Vector2D(best[0], best[1])
        self.unmark = (cyc, tgt)
        return tgt

    def _omam(self, agent, rank):
        wm = agent.wm
        cyc = int(wm.cycle)
        if self.marks and self.marks[0] == cyc:
            return self.marks[1]
        hl = agent.server_params.pitch_half_length
        pos = {wm.self.uniform_number: (wm.self.position.x(), wm.self.position.y())}
        for t in wm.teammates:
            if t.uniform_number > 0 and not t.is_goalie:
                pos[t.uniform_number] = (t.position.x(), t.position.y())
        defenders = sorted(u for u in pos if not rank or u != rank[0])
        opps = [o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie]
        result = {}
        if opps and defenders:
            holder = min(opps, key=lambda o: o.dist_from_ball).uniform_number
            tasks = []
            for o in opps:
                if o.uniform_number == holder:
                    continue
                ox, oy = o.position.x(), o.position.y()
                dg = math.hypot(ox + hl, oy)
                if ox > 10.0 and dg > 35.0:
                    continue
                dx, dy = -hl - ox, -oy
                d = math.hypot(dx, dy) or 1.0
                tasks.append((o.uniform_number, (ox + 1.5 * dx / d, oy + 1.5 * dy / d),
                              dg < 35.0))
            opts = {u: sorted(tasks, key=lambda t: (math.hypot(t[1][0] - pos[u][0],
                                                               t[1][1] - pos[u][1]),
                                                    t[0]))[:3]
                    for u in defenders}
            order = sorted(defenders, key=lambda u: (len(opts[u]), u))
            best = [(-1, 0.0), {}]

            def bt(i, used, nd, cost, plan):
                if i == len(order):
                    if (nd, -cost) > best[0]:
                        best[0], best[1] = (nd, -cost), dict(plan)
                    return
                u = order[i]
                for unum, mk, dang in opts[u]:
                    if unum in used:
                        continue
                    used.add(unum)
                    plan[u] = mk
                    bt(i + 1, used, nd + int(dang),
                       cost + math.hypot(mk[0] - pos[u][0], mk[1] - pos[u][1]), plan)
                    used.discard(unum)
                    del plan[u]
                bt(i + 1, used, nd, cost + 8.0, plan)

            bt(0, set(), 0, 0.0, {})
            result = {u: Vector2D(m[0], m[1]) for u, m in best[1].items()}
        self.marks = (cyc, result)
        return result


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
