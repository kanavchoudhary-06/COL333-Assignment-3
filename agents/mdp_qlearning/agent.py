# Agent1 - basic MDP + Q-learning team (separate from the main submission).
#
# On ball : an abstract MDP over (ball zone x 6, lane y 3, pressure 3) = 54
#           states and 6 macro actions (shoot, forward pass, safe pass,
#           dribble, hold, clear), as in Half Field Offense's high-level
#           actions. A hand-written transition/reward model is solved by
#           VALUE ITERATION when the player starts (AIMA 17.2); the resulting
#           Q* initialises a TABULAR Q-LEARNING table that keeps learning
#           online from the macro actions this player takes (AIMA 22.3),
#           with epsilon-greedy exploration that decays over the match.
# Off ball: the fastest player (same ranking on every player) intercepts;
#           everyone else solves a tiny 5x5 grid MDP around the formation
#           slot with value iteration and follows the extracted policy.
#
# Plain Python only (math, time, random).

import math
import random
import time

from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# ----------------------------------------------------------------- settings
GAMMA = 0.99                 # per-cycle discount
ALPHA = 0.10                 # Q-learning rate
EPS0, EPS1 = 0.05, 0.01      # epsilon at kick-off / at full time
VI_SWEEPS = 80
MATCH_CYCLES = 1500
ACTIONS = ("shoot", "pass_fwd", "pass_safe", "dribble", "hold", "clear")
NX, NY, NP = 6, 3, 3
N_STATES = NX * NY * NP
DANGER = (0.45, 0.20, 0.08, 0.03, 0.01, 0.005)   # cost of losing the ball, by x zone

GRID_N, GRID_STEP = 5, 4.0   # off-ball positioning MDP
GRID_GAMMA, GRID_SWEEPS, STEP_COST = 0.9, 15, 0.05


# ---------------------------------------------------------- abstract states
def zone_of(x, y):
    xi = min(NX - 1, max(0, int((x + 52.5) / (105.0 / NX))))
    yi = 0 if y < -10.0 else (1 if y <= 10.0 else 2)
    return xi, yi


def pressure_of(d):
    return 0 if d < 3.0 else (1 if d < 8.0 else 2)


def sidx(xi, yi, pi):
    return (xi * NY + yi) * NP + pi


def unpack(s):
    return s // (NY * NP), (s // NP) % NY, s % NP


# ------------------------------------------- hand-written model (the prior)
def model(s, a):
    """-> list of (prob, next_state | 'END', reward, cycles)."""
    xi, yi, pi = unpack(s)
    lost = -DANGER[xi]
    press = (0.3, 0.4, 0.3)
    out = []
    if a == "shoot":
        base = {5: 0.35, 4: 0.08}.get(xi, 0.0)
        pg = base * (1.0 if yi == 1 else 0.5) * (0.5, 0.8, 1.0)[pi]
        out = [(pg, "END", 1.0, 6), (1.0 - pg, "END", lost * 0.3, 6)]
    elif a in ("pass_fwd", "pass_safe", "dribble"):
        if a == "pass_fwd":
            p, nx, k = (0.45, 0.60, 0.75)[pi], [(min(NX - 1, xi + 1), 1.0)], 10
            ys = [(0, 1 / 3.0), (1, 1 / 3.0), (2, 1 / 3.0)]
        elif a == "pass_safe":
            p, nx, k = (0.70, 0.82, 0.90)[pi], [(xi, 1.0)], 8
            ys = [(0, 1 / 3.0), (1, 1 / 3.0), (2, 1 / 3.0)]
            press = (0.2, 0.4, 0.4)
        else:
            p, k = (0.35, 0.70, 0.88)[pi], 8
            nx = [(min(NX - 1, xi + 1), 0.4), (xi, 0.6)]
            ys = [(yi, 1.0)]
            press = (0.4, 0.4, 0.2)
        for x2, px in nx:
            for y2, py in ys:
                for p2 in range(NP):
                    out.append((p * px * py * press[p2], sidx(x2, y2, p2), 0.0, k))
        out.append((1.0 - p, "END", lost, k))
    elif a == "hold":
        p = (0.50, 0.85, 0.95)[pi]
        worse = sidx(xi, yi, max(0, pi - 1))
        out = [(p * 0.5, s, 0.0, 2), (p * 0.5, worse, 0.0, 2),
               (1.0 - p, "END", lost, 2)]
    elif a == "clear":
        out = [(1.0, "END", -0.35 * DANGER[xi] - 0.01, 1)]
    return out


def value_iteration():
    """Solve the abstract MDP; return Q*[s][a]."""
    V = [0.0] * N_STATES
    tables = [[model(s, a) for a in ACTIONS] for s in range(N_STATES)]
    for _ in range(VI_SWEEPS):
        delta = 0.0
        for s in range(N_STATES):
            best = -1e9
            for outs in tables[s]:
                q = 0.0
                for p, s2, r, k in outs:
                    q += p * (r + (0.0 if s2 == "END" else GAMMA ** k * V[s2]))
                if q > best:
                    best = q
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-7:
            break
    Q = []
    for s in range(N_STATES):
        row = []
        for outs in tables[s]:
            q = 0.0
            for p, s2, r, k in outs:
                q += p * (r + (0.0 if s2 == "END" else GAMMA ** k * V[s2]))
            row.append(q)
        Q.append(row)
    return Q


# --------------------------------------------- low-level option geometry
class Geo:
    def __init__(self, sp):
        g = lambda n, d: float(getattr(sp, n, d) or d)
        self.decay = g("ball_decay", 0.94)
        self.vball = g("ball_speed_max", 3.0)
        self.vmax = g("player_speed_max", 1.05)
        self.ka = (g("kickable_margin", 0.7) + g("player_size", 0.3)
                   + g("ball_size", 0.085))
        self.hl = g("pitch_half_length", 52.5)
        self.hw = g("pitch_half_width", 34.0)
        self.goal_w = g("goal_width", 14.02)
        self.S = [0.0]
        acc, p = 0.0, 1.0
        for _ in range(80):
            acc += p
            p *= self.decay
            self.S.append(acc)

    def pass_plan(self, bx, by, tx, ty, end_speed=1.2):
        """(cycles, first speed) to send the ball to (tx, ty)."""
        dist = math.hypot(tx - bx, ty - by)
        n = max(1, int(round(math.log(1.0 + dist * (1.0 - self.decay) / end_speed)
                             / -math.log(self.decay))))
        v0 = dist / self.S[n]
        while v0 > self.vball and n < 79:
            n += 1
            v0 = dist / self.S[n]
        return n, v0

    def margin(self, opps, bx, by, tx, ty, n, v0, path_speed=None):
        """min over path cycles of (relaxed opponent reach time - t)."""
        dist = math.hypot(tx - bx, ty - by)
        if dist < 1e-6:
            return 10.0
        ux, uy = (tx - bx) / dist, (ty - by) / dist
        m = 1e9
        for t in range(1, n + 1):
            if path_speed is None:
                s = v0 * self.S[t]
            else:
                s = min(dist, path_speed * t)
            px, py = bx + ux * s, by + uy * s
            for ox, oy in opps:
                d = math.hypot(px - ox, py - oy) - self.ka
                v = (d / self.vmax + 1.0 if d > 0 else 0.0) - t
                if v < m:
                    m = v
        return m


# ------------------------------------------------------------------- agent
class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        self.Q = value_iteration()           # MDP prior -> initial Q table
        self.geo = None
        self.rng = None
        self.pending = None                  # (s, a_index, cycle, kind, params)
        self.score = None                    # (ours, theirs) last seen
        self.grid_cache = None               # (cycle, target)
        self.updates = 0

    # ---------------------------------------------------------- helpers
    def _geo(self, agent):
        if self.geo is None:
            self.geo = Geo(agent.server_params)
            self.rng = random.Random(1000 + agent.wm.self.uniform_number)
        return self.geo

    def _opps(self, wm):
        return [(o.position.x(), o.position.y()) for o in wm.opponents
                if o.uniform_number > 0]

    def _state_at_ball(self, wm):
        b = wm.ball.position
        xi, yi = zone_of(b.x(), b.y())
        near = min((math.hypot(ox - b.x(), oy - b.y())
                    for ox, oy in self._opps(wm)), default=20.0)
        return sidx(xi, yi, pressure_of(near))

    # ----------------------------------------------------- Q-learning
    def _observe(self, agent):
        """Close the pending macro action if its outcome is now visible."""
        wm = agent.wm
        score = (int(wm.our_team_score), int(wm.their_team_score))
        if self.score is None:
            self.score = score
        if self.pending is None:
            self.score = score
            return
        s, ai, c0, kind, _ = self.pending
        k = max(1, int(wm.cycle) - c0)
        me = wm.self.uniform_number
        target = None
        if score[0] > self.score[0]:
            target = 1.0                                   # we scored
        elif score[1] > self.score[1]:
            target = -1.0                                  # we conceded
        elif wm.kickable_opponent_existance:
            xi, _ = zone_of(wm.ball.position.x(), wm.ball.position.y())
            target = -DANGER[xi]                           # lost the ball
        elif wm.kickable_teammate_existance or wm.self.is_kickable:
            holder_is_me = wm.self.is_kickable
            moved_on = (not holder_is_me) or kind in ("dribble", "hold")
            if moved_on and k >= 1 and kind not in ("shoot", "clear"):
                s2 = self._state_at_ball(wm)
                target = GAMMA ** k * max(self.Q[s2])      # bootstrap
        elif k > 60:
            self.pending = None                            # outcome unclear
        self.score = score
        if target is not None:
            q = self.Q[s][ai]
            self.Q[s][ai] = q + ALPHA * (target - q)
            self.updates += 1
            self.pending = None

    # ------------------------------------------------------------ on ball
    def on_ball(self, agent) -> None:
        try:
            if self._on(agent):
                return
        except Exception:
            pass
        try:
            super().on_ball(agent)
        except Exception:
            Action.clear(agent)

    def _options(self, agent):
        """Parameterise every macro action; None = not available now."""
        wm, g = agent.wm, self._geo(agent)
        b = wm.ball.position
        bx, by = b.x(), b.y()
        opps = self._opps(wm)
        opt = {"hold": ("hold",), "clear": ("clear",)}
        # shoot: corner away from the goalie
        if math.hypot(g.hl - bx, by) < 30.0:
            gk = [o for o in wm.opponents if o.is_goalie]
            gy = gk[0].position.y() if gk else 0.0
            ty = (g.goal_w / 2.0 - 0.8) * (-1.0 if gy > 0 else 1.0)
            opt["shoot"] = ("shoot", g.hl, ty)
        # passes: forward = furthest-up safe receiver, safe = biggest margin
        best_fwd, best_safe = None, None
        for t in wm.teammates:
            if t.uniform_number <= 0 or t.is_goalie:
                continue
            tx, ty = t.position.x(), t.position.y()
            if tx > wm.offside_line_x and tx > 0.0:
                continue
            d = math.hypot(tx - bx, ty - by)
            if d < 3.0 or d > 35.0:
                continue
            n, v0 = g.pass_plan(bx, by, tx, ty)
            m = g.margin(opps, bx, by, tx, ty, n, v0)
            if m >= 0.0 and tx > bx + 2.0 and (best_fwd is None or tx > best_fwd[1]):
                best_fwd = ("pass", tx, ty, n)
            if best_safe is None or m > best_safe[0]:
                best_safe = (m, ("pass", tx, ty, n))
        if best_fwd:
            opt["pass_fwd"] = best_fwd
        if best_safe and best_safe[0] > -1.0:
            opt["pass_safe"] = best_safe[1]
        # dribble: five headings toward goal, pick the widest margin
        base = math.atan2(-by, g.hl - bx)
        best_d = None
        for off in (-1.0, -0.5, 0.0, 0.5, 1.0):
            a = base + off
            tx, ty = bx + 6.0 * math.cos(a), by + 6.0 * math.sin(a)
            if abs(tx) > g.hl - 1.0 or abs(ty) > g.hw - 1.0:
                continue
            m = g.margin(opps, bx, by, tx, ty, 9, 0.0, path_speed=0.65)
            if best_d is None or m > best_d[0]:
                best_d = (m, ("dribble", tx, ty))
        if best_d:
            opt["dribble"] = best_d[1]
        return opt

    def _on(self, agent):
        wm = agent.wm
        self._observe(agent)
        cyc = int(wm.cycle)
        # a kick started last cycle may need more touches: keep it going
        if (self.pending is not None and self.pending[3] in ("pass_fwd", "pass_safe", "shoot")
                and cyc - self.pending[2] <= 3):
            self._run(agent, self.pending[4])
            note(agent, "continue", a=self.pending[3])
            return True
        s = self._state_at_ball(wm)
        opt = self._options(agent)
        legal = [i for i, a in enumerate(ACTIONS) if a in opt]
        eps = EPS0 + (EPS1 - EPS0) * min(1.0, cyc / float(MATCH_CYCLES))
        if self.rng.random() < eps:
            ai = self.rng.choice(legal)
        else:
            ai = max(legal, key=lambda i: (self.Q[s][i], -i))
        kind = ACTIONS[ai]
        params = opt[kind]
        self.pending = (s, ai, cyc, kind, params)
        self._run(agent, params)
        note(agent, kind, s=s, q=round(self.Q[s][ai], 3), upd=self.updates)
        return True

    def _run(self, agent, p):
        kind = p[0]
        if kind == "shoot":
            tgt = Vector2D(p[1], p[2])
            near = min((o.dist_from_self for o in agent.wm.opponents
                        if o.uniform_number > 0), default=99.0)
            if near < 2.5:
                Action.kick_one_step(agent, tgt)
            else:
                Action.kick(agent, tgt, 1)
        elif kind == "pass":
            Action.kick(agent, Vector2D(p[1], p[2]), p[3], accept_late=3)
        elif kind == "dribble":
            Action.dribble(agent, Vector2D(p[1], p[2]))
            Action.hold(agent)
        elif kind == "hold":
            Action.hold(agent)
        Action.clear(agent)

    # ----------------------------------------------------------- off ball
    def off_ball(self, agent) -> None:
        try:
            self._observe(agent)
            if self._off(agent):
                return
        except Exception:
            pass
        try:
            super().off_ball(agent)
        except Exception:
            Action.go_to_point(agent, Tools.home_position(agent))

    def _rank(self, wm):
        rows = [(int(wm.intercept_table.self_reach_steps), wm.self.uniform_number)]
        for t in wm.teammates:
            if t.uniform_number > 0 and not t.is_goalie:
                rows.append((int(t.ball_reach_steps), t.uniform_number))
        rows.sort()
        return [u for _, u in rows]

    def _off(self, agent):
        wm = agent.wm
        me = wm.self
        g = self._geo(agent)
        first = self._rank(wm)[0] == me.uniform_number
        ours = wm.kickable_teammate_existance
        theirs = wm.kickable_opponent_existance
        if theirs and not ours:
            if me.tackle_probability >= 0.75 and wm.ball.position.dist(me.position) < 2.0:
                goal = Vector2D(g.hl, 0.0)
                rel = (goal - me.position).th().degree() - me.body_direction
                rel = (rel + 180.0) % 360.0 - 180.0
                Action.tackle(agent, rel, foul=False)
                note(agent, "tackle")
                return True
        if not ours and first:
            Action.intercept(agent)
            note(agent, "intercept")
            return True
        if agent.budget.remaining_ms < 5000.0:
            Action.go_to_point(agent, Tools.home_position(agent),
                               max_dash_power=self.dash_power(agent))
            return True
        tgt = self._grid_target(agent, attacking=ours)
        Action.go_to_point(agent, tgt, max_dash_power=self.dash_power(agent))
        note(agent, "grid_mdp", att=ours)
        return True

    def _grid_target(self, agent, attacking):
        """5x5 grid MDP around home: value iteration, then follow the policy."""
        wm = agent.wm
        cyc = int(wm.cycle)
        if self.grid_cache and self.grid_cache[0] >= cyc - 1:
            return self.grid_cache[1]
        home = Tools.home_position(agent)
        hx, hy = home.x(), home.y()
        bx, by = wm.ball.position.x(), wm.ball.position.y()
        opps = self._opps(wm)
        offx = wm.offside_line_x
        half = GRID_N // 2
        cells = [(hx + (i - half) * GRID_STEP, hy + (j - half) * GRID_STEP)
                 for i in range(GRID_N) for j in range(GRID_N)]

        def reward(x, y):
            if abs(x) > 51.0 or abs(y) > 32.0:
                return -1.0
            r = -0.02 * math.hypot(x - hx, y - hy)
            if attacking:
                if x > offx - 0.5 and x > 0.0:
                    return -1.0
                free = min((math.hypot(ox - x, oy - y) for ox, oy in opps), default=10.0)
                r += 0.06 * min(free, 10.0) + 0.01 * x
                # passing lane from the ball
                dx, dy = x - bx, y - by
                L2 = dx * dx + dy * dy
                lane = 5.0
                if L2 > 1e-6:
                    for ox, oy in opps:
                        u = ((ox - bx) * dx + (oy - by) * dy) / L2
                        if 0.0 < u < 1.0:
                            lane = min(lane, math.hypot(bx + u * dx - ox, by + u * dy - oy))
                r += 0.04 * lane
            else:
                # stand between the ball and our goal
                gx, gy = -52.5, 0.0
                dx, dy = gx - bx, gy - by
                L2 = dx * dx + dy * dy
                u = max(0.0, min(1.0, ((x - bx) * dx + (y - by) * dy) / L2)) if L2 > 1e-6 else 0.0
                r -= 0.05 * math.hypot(bx + u * dx - x, by + u * dy - y)
            return r

        R = [reward(x, y) for x, y in cells]
        V = [0.0] * len(cells)

        def nbrs(k):
            i, j = divmod(k, GRID_N)
            out = [k]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < GRID_N and 0 <= b < GRID_N:
                    out.append(a * GRID_N + b)
            return out

        N = [nbrs(k) for k in range(len(cells))]
        for _ in range(GRID_SWEEPS):
            V = [R[k] + GRID_GAMMA * max(V[n] - (0.0 if n == k else STEP_COST)
                                         for n in N[k])
                 for k in range(len(cells))]
        # start from the nearest cell and follow the greedy policy
        mx, my = wm.self.position.x(), wm.self.position.y()
        k = min(range(len(cells)), key=lambda c: math.hypot(cells[c][0] - mx, cells[c][1] - my))
        for _ in range(2 * GRID_N):
            nk = max(N[k], key=lambda n: (V[n] - (0.0 if n == k else STEP_COST), -n))
            if nk == k:
                break
            k = nk
        tgt = Vector2D(cells[k][0], cells[k][1])
        self.grid_cache = (cyc, tgt)
        return tgt


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None
