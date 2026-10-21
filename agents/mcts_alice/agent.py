# Agent1 - Monte Carlo Tree Search team (separate agent, after Alice2022).
#
# On ball : UCT / MONTE CARLO TREE SEARCH (AIMA 5.4) over macro actions.
#           Selection by UCB1, expansion of the top-K actions by prior,
#           simulation by sampling each action's success from its
#           probability (chance outcomes), with an evaluation function at
#           the leaf instead of a long random playout, and averaged backup.
#           The most-visited root action is played.
#           Shot success is a Monte Carlo estimate M/N over sampled kick
#           directions and speeds (Alice2022 used an SVM fed with sampled
#           ball speeds; here the sample count itself is the estimate).
# Off ball: game phase (offence / defence / transition) decides the role;
#           the presser picks its defensive action with a one-level MCTS
#           (UCB1 bandit) against sampled opponent responses, as Alice2022
#           used MCTS to predict defensive outcomes against the ball holder.
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
GAMMA = 0.98
UCB_C = 0.3
MAX_DEPTH = 7
TOP_K_ROOT, TOP_K = 10, 6
SHOT_SAMPLES = 12
H_B, H_D, H_R = -0.5, -0.10, 0.20
V_FAIL_SHOT, CLEAR_VALUE = -0.02, -0.03
DEF_ITERS = 40


class St:
    __slots__ = ("bx", "by", "holder", "mates", "opps", "prev")

    def __init__(self, bx, by, holder, mates, opps, prev=None):
        self.bx, self.by, self.holder = bx, by, holder
        self.mates, self.opps, self.prev = mates, opps, prev


class Model:
    """Generative model of macro actions: (P, cycles, next state, meta)."""

    def __init__(self, geo, offx, rng):
        self.g, self.offx, self.rng = geo, offx, rng

    def H(self, st):
        d = math.hypot(self.g.hl - st.bx, st.by)
        free = min((math.hypot(ox - st.bx, oy - st.by) for ox, oy, _ in st.opps),
                   default=10.0)
        return sig(H_B + H_D * d + H_R * min(free, 6.0))

    def v_fail(self, x, y):
        return -sig(0.1 - 0.1 * math.hypot(x + self.g.hl, y))

    def _child(self, st, x, y, holder, cycles):
        mates = dict(st.mates)
        mates[holder] = (x, y, mates[holder][2])
        step = self.g.vmax * cycles * 0.8
        opps = []
        for ox, oy, gk in st.opps:
            d = math.hypot(x - ox, y - oy)
            mv = min(step, max(0.0, d - 1.0))
            if d > 1e-6:
                ox, oy = ox + (x - ox) / d * mv, oy + (y - oy) / d * mv
            opps.append((ox, oy, gk))
        return St(x, y, holder, mates, opps, st.holder)

    def shot_prob(self, st, ty):
        """Monte Carlo estimate: share of sampled shots no one can stop."""
        g = self.g
        bx, by = st.bx, st.by
        base = math.atan2(ty - by, g.hl - bx)
        hits = 0
        for _ in range(SHOT_SAMPLES):
            a = base + self.rng.uniform(-0.05, 0.05)
            v0 = self.rng.uniform(2.5, g.vball)
            dist = (g.hl - bx) / max(0.05, math.cos(a))
            yend = by + dist * math.sin(a)
            if abs(yend) > g.goal_w / 2.0:
                continue
            k = g.steps_for(dist, v0)
            if k is None:
                continue
            if g.margin(st.opps, g.kick_path(bx, by, g.hl, yend, v0, k)) > 0.0:
                hits += 1
        return hits / float(SHOT_SAMPLES)

    def actions(self, st, root):
        g = self.g
        bx, by = st.bx, st.by
        out = []
        if math.hypot(g.hl - bx, by) < 30.0:
            half = g.goal_w / 2.0 - 0.7
            for i in range(3):
                ty = -half + half * i
                dist = math.hypot(g.hl - bx, ty - by)
                k = g.steps_for(dist, g.vball)
                if k is None:
                    continue
                if root:
                    p = self.shot_prob(st, ty)
                else:
                    m = g.margin(st.opps, g.kick_path(bx, by, g.hl, ty, g.vball, k))
                    p = sig(1.2 * (m - 1.0))
                out.append((p, k, None, ("shoot", g.hl, ty, k), p))
        for u, (rx, ry, gk) in st.mates.items():
            if u == st.holder or gk or (not root and u == st.prev):
                continue
            if rx > self.offx and rx > 0.0:
                continue
            for tx, ty, lead in ((rx, ry, False), (rx + 5.0, ry, True)):
                if not g.inside(tx, ty) or (lead and tx > self.offx and tx > 0.0):
                    continue
                dist = math.hypot(tx - bx, ty - by)
                if dist < 3.0 or dist > 40.0:
                    continue
                k, v0 = g.pass_plan(dist, 1.3)
                if k is None:
                    continue
                t_recv = math.hypot(tx - rx, ty - ry) / g.vmax + 1.0 if lead else 0.0
                pts = g.kick_path(bx, by, tx, ty, v0, k)
                if t_recv > k:
                    pts += [(tx, ty)] * int(math.ceil(t_recv - k))
                p = sig(g.margin(st.opps, pts) - 0.5)
                c = int(max(k, math.ceil(t_recv)))
                child = self._child(st, tx, ty, u, c)
                out.append((p, c, child, ("pass", tx, ty, k, u), p * self.H(child)))
        base = math.atan2(-by, g.hl - bx)
        for i in range(8):
            a = base + math.pi * i / 4.0
            tx, ty = bx + 5.0 * math.cos(a), by + 5.0 * math.sin(a)
            if not g.inside(tx, ty):
                continue
            pts = g.carry_path(bx, by, tx, ty, 0.65)
            p = sig(g.margin(st.opps, pts) - 1.0)
            child = self._child(st, tx, ty, st.holder, len(pts))
            out.append((p, len(pts), child, ("dribble", tx, ty), p * self.H(child)))
        out.sort(key=lambda a: -a[4])
        out = out[:TOP_K_ROOT if root else TOP_K]
        if root:
            out.append((1.0, 1, None, ("clear",), 0.0))
            out.append((0.9, 2, St(bx, by, st.holder, st.mates, st.opps, st.prev),
                        ("hold",), 0.0))
        return out


class MNode:
    __slots__ = ("st", "acts", "n", "w", "N", "kids")

    def __init__(self, st):
        self.st, self.acts, self.kids = st, None, {}
        self.n, self.w, self.N = [], [], 0


def mcts(model, root_st, deadline, max_iters=2000):
    root = MNode(root_st)
    root.acts = model.actions(root_st, True)
    root.n = [0] * len(root.acts)
    root.w = [0.0] * len(root.acts)
    rng = model.rng
    it = 0
    while it < max_iters and time.perf_counter() < deadline:
        it += 1
        node, depth, disc, path = root, 0, 1.0, []
        while True:
            if node.acts is None:
                node.acts = model.actions(node.st, False)
                node.n = [0] * len(node.acts)
                node.w = [0.0] * len(node.acts)
            if not node.acts:
                leaf_v, leaf_d = model.H(node.st), disc
                break
            # selection: untried arm in prior order, else UCB1
            a = next((i for i, c in enumerate(node.n) if c == 0), None)
            if a is None:
                lnN = math.log(node.N + 1)
                a = max(range(len(node.acts)),
                        key=lambda i: node.w[i] / node.n[i]
                        + UCB_C * math.sqrt(lnN / node.n[i]))
            p, c, child, meta, _ = node.acts[a]
            path.append((node, a, disc))
            fresh = node.n[a] == 0
            leaf_d = disc * GAMMA ** c
            kind = meta[0]
            ok = rng.random() < p                      # chance outcome
            if kind == "shoot":
                leaf_v = 1.0 if ok else V_FAIL_SHOT
                break
            if kind == "clear":
                leaf_v = CLEAR_VALUE
                break
            if not ok:
                leaf_v = model.v_fail((node.st.bx + child.bx) * 0.5,
                                      (node.st.by + child.by) * 0.5)
                break
            if kind == "hold" or fresh or depth + 1 >= MAX_DEPTH:
                leaf_v = model.H(child) * (0.85 if kind == "hold" else 1.0)
                if a not in node.kids and kind != "hold":
                    node.kids[a] = MNode(child)     # expansion
                break
            node = node.kids.setdefault(a, MNode(child))
            disc, depth = leaf_d, depth + 1
        # backup: return seen from each node on the path
        for nd, a, d in path:
            g = leaf_v * (leaf_d / d if d > 0 else 1.0)
            nd.n[a] += 1
            nd.w[a] += g
            nd.N += 1
    best = max(range(len(root.acts)),
               key=lambda i: (root.n[i], root.w[i] / root.n[i] if root.n[i] else -9.0, -i))
    q = root.w[best] / root.n[best] if root.n[best] else 0.0
    return root.acts[best][3], q, it


# -------------------------------------------------------------------- agent
class Agent1(SafeAgent):
    def __init__(self):
        super().__init__()
        self.geo = None
        self.commit = None

    def _g(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
        return self.geo

    def _on(self, agent):
        ms = budget_ms(agent, 4.0)
        if ms <= 1.0:
            return False
        wm = agent.wm
        cyc = int(wm.cycle)
        # finish a kick that started last cycle (multi-touch kicks)
        if (self.commit and self.commit[0] == cyc - 1 and self.commit[2] < 2
                and self.commit[1][0] in ("pass", "shoot")):
            execute(agent, self.commit[1])
            self.commit = (cyc, self.commit[1], self.commit[2] + 1)
            note(agent, "continue")
            return True
        w = World(agent)
        rng = random.Random(cyc * 31 + w.me)
        model = Model(self._g(agent), w.offx, rng)
        st = St(w.bx, w.by, w.me, w.mates, w.opps)
        meta, q, iters = mcts(model, st, time.perf_counter() + ms / 1000.0)
        self.commit = (cyc, meta, 0)
        execute(agent, meta)
        note(agent, meta[0], q=round(q, 3), it=iters)
        return True

    def _off(self, agent):
        wm = agent.wm
        rank = field_rank(wm)
        me = wm.self.uniform_number
        pos = rank.index(me) if me in rank else 9
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        home = Tools.home_position(agent)
        phase = "offence" if ours else ("defence" if theirs else "transition")
        if phase == "defence":
            if pos == 0 and budget_ms(agent, 0.5) > 1.0:
                return self._defend_mcts(agent)
            if try_tackle(agent):
                return True
            if pos == 1:
                blk = self.block_point(agent)
                if blk is not None:
                    Action.go_to_point(agent, blk, max_dash_power=100.0)
                    note(agent, "block")
                    return True
        if phase != "offence" and pos == 0:
            Action.intercept(agent)
            note(agent, "intercept")
            return True
        if phase == "offence":
            Action.go_to_point(agent, self._support(agent, home),
                               max_dash_power=self.dash_power(agent))
            note(agent, "support")
            return True
        Action.go_to_point(agent, home, max_dash_power=self.dash_power(agent))
        note(agent, "home", phase=phase)
        return True

    def _support(self, agent, home):
        wm = agent.wm
        g = self._g(agent)
        bx, by = wm.ball.position.x(), wm.ball.position.y()
        opps = [(o.position.x(), o.position.y()) for o in wm.opponents
                if o.uniform_number > 0]
        hx, hy = home.x(), home.y()
        best, best_s = (hx, hy), -1e9
        for r, k in [(0.0, 0)] + [(5.0, k) for k in range(8)]:
            x, y = hx + r * math.cos(k * math.pi / 4.0), hy + r * math.sin(k * math.pi / 4.0)
            if not g.inside(x, y, 2.0) or (x > wm.offside_line_x - 0.5 and x > 0.0):
                continue
            free = min((math.hypot(ox - x, oy - y) for ox, oy in opps), default=10.0)
            s = min(free, 10.0) + 1.5 * lane(opps, bx, by, x, y) + 0.05 * x \
                - 0.1 * math.hypot(x - hx, y - hy)
            if s > best_s:
                best, best_s = (x, y), s
        return Vector2D(best[0], best[1])

    def _defend_mcts(self, agent):
        """UCB1 over defensive arms; each sample draws the opponent holder's
        response from a softmax over its options and its success from P."""
        wm = agent.wm
        g = self._g(agent)
        me = wm.self
        mx, my = me.position.x(), me.position.y()
        bx, by = wm.ball.position.x(), wm.ball.position.y()
        hl = g.hl
        ours = [(t.position.x(), t.position.y(), False) for t in wm.teammates
                if t.uniform_number > 0]
        opp_list = [o for o in wm.opponents if o.uniform_number > 0]
        holder = min(opp_list, key=lambda o: o.dist_from_ball, default=None)
        if holder is None:
            return False
        receivers = [o for o in opp_list if o is not holder and not o.is_goalie]
        # arms: where my move takes me in ~3 cycles
        step = g.vmax * 3.0

        def toward(x, y):
            d = math.hypot(x - mx, y - my)
            if d < 1e-6:
                return mx, my
            s = min(1.0, step / d)
            return mx + (x - mx) * s, my + (y - my) * s

        arms = [("intercept", toward(bx, by))]
        blk = self.block_point(agent)
        if blk is not None:
            arms.append(("block", toward(blk.x(), blk.y())))
        if receivers:
            r = min(receivers, key=lambda o: math.hypot(o.position.x() + hl, o.position.y()))
            dx, dy = -hl - r.position.x(), -r.position.y()
            d = math.hypot(dx, dy) or 1.0
            arms.append(("mark", toward(r.position.x() + 1.5 * dx / d,
                                        r.position.y() + 1.5 * dy / d), r))
        if me.tackle_probability >= 0.5 and wm.ball.position.dist(me.position) < 2.0:
            arms.append(("tackle", (mx, my)))

        def h_opp(x, y, us):
            free = min((math.hypot(ux - x, uy - y) for ux, uy, _ in us), default=10.0)
            return sig(H_B + H_D * math.hypot(x + hl, y) + H_R * min(free, 6.0))

        # deterministic part of each arm: opponent options (weight, P, threat)
        tables = []
        for arm in arms:
            us = ours + [(arm[1][0], arm[1][1], False)]
            opts = []
            for o in receivers + [None]:
                if o is None:                       # dribble toward our goal
                    ang = math.atan2(-by, -hl - bx)
                    tx, ty = bx + 5.0 * math.cos(ang), by + 5.0 * math.sin(ang)
                    pts = g.carry_path(bx, by, tx, ty, 0.65)
                else:
                    tx, ty = o.position.x(), o.position.y()
                    dist = math.hypot(tx - bx, ty - by)
                    if dist < 2.0:
                        continue
                    k, v0 = g.pass_plan(dist, 1.3)
                    if k is None:
                        continue
                    pts = g.kick_path(bx, by, tx, ty, v0, k)
                p = sig(g.margin(us, pts) - 0.5)
                thr = h_opp(tx, ty, us)
                opts.append((math.exp(5.0 * p * thr), p, thr))
            z = sum(o[0] for o in opts) or 1.0
            tables.append([(o[0] / z, o[1], o[2]) for o in opts])
        rng = random.Random(int(wm.cycle) * 17 + me.uniform_number)
        n = [0] * len(arms)
        s = [0.0] * len(arms)
        for t in range(1, DEF_ITERS + 1):
            a = next((i for i, c in enumerate(n) if c == 0), None)
            if a is None:
                a = min(range(len(arms)),
                        key=lambda i: s[i] / n[i] - UCB_C * math.sqrt(math.log(t) / n[i]))
            threat = 0.0
            if arms[a][0] == "tackle" and rng.random() < me.tackle_probability:
                threat = 0.0
            elif tables[a]:
                r, acc = rng.random(), 0.0
                pick = tables[a][-1]
                for row in tables[a]:
                    acc += row[0]
                    if r <= acc:
                        pick = row
                        break
                threat = pick[2] if rng.random() < pick[1] else 0.0
            n[a] += 1
            s[a] += threat
        best = min(range(len(arms)), key=lambda i: (s[i] / max(1, n[i]), -n[i], i))
        kind = arms[best][0]
        if kind == "tackle":
            if not try_tackle(agent, 0.5):
                Action.intercept(agent)
        elif kind == "intercept":
            Action.intercept(agent)
        elif kind == "block":
            Action.go_to_point(agent, blk, max_dash_power=100.0)
            Action.intercept(agent)
        else:
            Action.go_to_point(agent, Vector2D(*arms[best][1]), max_dash_power=100.0)
            Action.intercept(agent)
        note(agent, "def_" + kind, threat=round(s[best] / max(1, n[best]), 3))
        return True


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
