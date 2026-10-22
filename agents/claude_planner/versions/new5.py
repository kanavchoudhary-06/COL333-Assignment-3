# Your agent goes here. The Agent1 class must hold your final submission.
"""
Agent1 - a hybrid planner built only from classic search / probabilistic
reasoning techniques.

  * A kinematic world model (ball flight + player reach time), evaluated in
    batch with numpy. It plays the role of the transition model in a
    model-based planner.
  * ON THE BALL: an expectimax-style action search. Shots, passes (direct,
    lead and through balls) and self-passes/dribbles are generated, every
    candidate is rolled out through the model and turned into a success
    probability (chance node), and scored as
        p * V(success state) + (1 - p) * V(failure state)
    with a hand-built heuristic state evaluation V. The best few are then
    expanded one more ply (beam search) to look at what the receiver can
    do next, while the CPU budget allows (anytime).
  * OFF THE BALL: every player computes the same team-wide allocation from
    the same full state, so no communication is needed.
      - who chases: minimum model reach time (ties by uniform number)
      - defence: marking is a small CSP (variables = our defenders,
        values = opponents to mark or 'cover'), solved with backtracking,
        MRV variable ordering, forward checking and branch & bound on the
        total cost.
      - attack: support spots chosen by greedy local search over a grid
        around a ball-relative anchor, scored for pass-lane openness,
        threat and spacing.
"""
import math
import time

import numpy as np
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


T_MAX = 40              # rollout horizon in cycles
NEVER = 10000
GOAL_VALUE = 200.0      # value of a goal on the evaluation scale
LOSS_WEIGHT = 1.0       # weight of the opponent's value when we lose the ball
# The goalie as seen by a shot: fitted to the observed conversion of shots
# in our own match logs (it reacts late and covers less than a field
# player's model would suggest).
GOALIE_REACT = 2.0
GOALIE_CTRL = 1.2


def _xy(v):
    return v.x(), v.y()


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _norm_deg(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def agent_budget_low(agent_obj):
    """True when this player's match budget is nearly gone (set each
    decision from agent.budget)."""
    return getattr(agent_obj, "budget_low", False)


class Clock:
    """Anytime deadline for one decision: CPU ms from the match budget and
    a wall-clock cap that keeps us well inside the 1 s per-decision limit."""

    def __init__(self, cpu_ms, wall_ms):
        self.c_end = time.process_time() + cpu_ms / 1000.0
        self.w_end = time.perf_counter() + wall_ms / 1000.0

    def left(self):
        return 1000.0 * min(self.c_end - time.process_time(),
                            self.w_end - time.perf_counter())


class Model:
    """Ball flight and player reach-time model, batched over candidates."""

    def __init__(self, sp):
        self.decay = sp.ball_decay or 0.94
        self.bmax = sp.ball_speed_max or 3.0
        self.accel_max = sp.ball_accel_max or 2.7
        self.max_power = sp.max_power or 100.0
        self.hl = sp.pitch_half_length or 52.5
        self.hw = sp.pitch_half_width or 34.0
        self.goal_hw = (sp.goal_width or 14.02) / 2.0
        self.pa_x = self.hl - (sp.penalty_area_length or 16.5)
        self.pa_hw = sp.penalty_area_half_width or 20.16
        self.kick_area = ((sp.kickable_margin or 0.7) + (sp.player_size or 0.3)
                          + (sp.ball_size or 0.085))
        self.catch_area = 1.35
        pdec = sp.player_decay or 0.4
        accel = (sp.max_dash_power or 100.0) * (sp.dash_power_rate or 0.006)
        vmax = sp.player_speed_max or 1.05

        t = np.arange(1, T_MAX + 1, dtype=float)
        self.t = t
        self.G = (1.0 - self.decay ** t) / (1.0 - self.decay)
        self.F = (1.0 - pdec ** t) / (1.0 - pdec)
        dash, v, d = [0.0], 0.0, 0.0
        for _ in range(T_MAX + 8):
            v = min(v + accel, vmax)
            d += v
            dash.append(d)
            v *= pdec
        self.dash = np.array(dash)
        self.dash_list = dash

    # ball ------------------------------------------------------------

    def first_speed(self, dist, n):
        return dist * (1.0 - self.decay) / (1.0 - self.decay ** n)

    def min_steps(self, dist, speed):
        """Cycles for a ball launched at `speed` to cover `dist` (or NEVER)."""
        r = 1.0 - dist * (1.0 - self.decay) / speed
        if r <= 0.0:
            return NEVER
        return max(1, int(math.ceil(math.log(r) / math.log(self.decay) - 1e-9)))

    def run_steps(self, dist):
        """Dashes needed from rest to cover dist."""
        for n, d in enumerate(self.dash_list):
            if d >= dist:
                return n
        return len(self.dash_list) + int(dist - self.dash_list[-1]) + 1

    # reach -----------------------------------------------------------

    def first_reach(self, BP, K, P, V, U, ctrl, react, turn_cos, self_col=-1):
        """First cycle (1-based) each player can control the ball.

        BP: (C,T,2) ball positions after 1..T cycles of flight
        K : (C,)   cycles before the ball starts flying (extra kicks)
        P, V, U: (N,2) player positions, velocities, body unit vectors
        ctrl: (N,) control radius, or (C,N,T)
        react: (N,) cycles lost before the player can start dashing
        returns (C,N) int array
        """
        C, T = BP.shape[0], BP.shape[1]
        N = P.shape[0]
        if N == 0:
            return np.full((C, 0), NEVER)
        Pi = P[:, None, :] + V[:, None, :] * self.F[None, :T, None]
        dx = BP[:, None, :, 0] - Pi[None, :, :, 0]
        dy = BP[:, None, :, 1] - Pi[None, :, :, 1]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-9
        cos = (dx * U[None, :, None, 0] + dy * U[None, :, None, 1]) / dist
        if np.ndim(ctrl) == 1:
            ctrl = ctrl[None, :, None]
        need = dist - ctrl
        turn = (cos < turn_cos) & (need > 0.6)
        delay = np.repeat(K[:, None], N, axis=1)
        if self_col >= 0:
            delay[:, self_col] = 0          # the kicker spends those cycles kicking
        avail = (self.t[None, None, :T] + delay[:, :, None]
                 - react[None, :, None] - turn)
        avail = np.clip(avail, 0, len(self.dash) - 1).astype(np.int64)
        slack = self.dash[avail] - need
        self.last_slack = slack            # (C,N,T): metres to spare
        ok = slack >= 0.0
        hit = ok.any(axis=2)
        return np.where(hit, ok.argmax(axis=2) + 1, NEVER)

    def shot_success(self, BP, slack, goalie_col, gdist):
        """Probability a shot gets in: smooth in how many metres the best
        opponent has to spare anywhere before the ball crosses the line,
        times an aiming-error factor. -> (p (C,), crossing y (C,))"""
        C, T = BP.shape[0], BP.shape[1]
        cross = BP[:, :, 0] > self.hl
        t_goal = np.where(cross.any(axis=1), cross.argmax(axis=1), T - 1)
        tmask = np.arange(T)[None, :] <= t_goal[:, None]
        sl = np.where(tmask[:, None, :], slack, -50.0).max(axis=2).max(axis=1)
        p = 1.0 / (1.0 + np.exp(sl / 0.6))
        y_at = BP[np.arange(C), t_goal, 1]
        err = 0.03 * gdist + 0.3
        p = p * np.clip((self.goal_hw - np.abs(y_at)) / err, 0.0, 1.0)
        p = np.where(cross.any(axis=1) & (np.abs(y_at) < self.goal_hw - 0.2), p, 0.0)
        return p, y_at


class Snapshot:
    """The world in plain floats/arrays, rebuilt once per cycle."""

    def __init__(self, agent, model):
        wm = agent.wm
        self.wm = wm
        self.cycle = wm.cycle
        self.bx, self.by = _xy(wm.ball.position)
        self.bvx, self.bvy = _xy(wm.ball.velocity)
        me = wm.self
        self.unum = me.uniform_number
        self.mx, self.my = _xy(me.position)
        self.mvx, self.mvy = _xy(me.velocity)
        self.mbody = me.body_direction
        self.stamina = me.stamina
        self.offside_x = wm.offside_line_x

        def pack(players):
            n = len(players)
            P = np.zeros((n, 2)); V = np.zeros((n, 2)); U = np.zeros((n, 2))
            G = np.zeros(n, dtype=bool); num = []
            for i, p in enumerate(players):
                P[i] = _xy(p.position)
                V[i] = _xy(p.velocity)
                a = math.radians(p.body_direction)
                U[i] = (math.cos(a), math.sin(a))
                G[i] = p.is_goalie
                num.append(p.uniform_number)
            return P, V, U, G, num

        mates = [p for p in wm.teammates if p.uniform_number > 0]
        opps = [p for p in wm.opponents if p.uniform_number > 0]
        self.mates, self.opps = mates, opps
        self.oP, self.oV, self.oU, self.oG, self.onum = pack(opps)
        # our field players INCLUDING me, goalie excluded; me first
        field = [p for p in mates if not p.is_goalie]
        self.fP, self.fV, self.fU, _, fnum = pack(field)
        a = math.radians(self.mbody)
        self.fP = np.vstack([[self.mx, self.my], self.fP]) if len(field) else np.array([[self.mx, self.my]])
        self.fV = np.vstack([[self.mvx, self.mvy], self.fV]) if len(field) else np.array([[self.mvx, self.mvy]])
        self.fU = np.vstack([[math.cos(a), math.sin(a)], self.fU]) if len(field) else np.array([[math.cos(a), math.sin(a)]])
        self.fnum = [self.unum] + fnum
        gk = [p for p in mates if p.is_goalie]
        self.gP = np.array([_xy(gk[0].position)]) if gk else np.zeros((0, 2))
        self.kickable_mate = wm.kickable_teammate_existance
        self.kickable_opp = wm.kickable_opponent_existance


class Agent1(StarterAgent):

    # design switches (kept as class attributes so variants can be A/B tested)
    PHI = "hybrid"        # state value: "v2" | "new" | "hybrid"
    FOLLOW = True         # do not re-kick a soft kick that is rolling as planned
    REFINE = True         # second search ply (receiver's follow-up shot)
    OFFBALL = "v2"        # "v2": single chaser + blocker + marking CSP; "jobs"
    STAMINA = "tiered"    # "tiered" (hysteresis recovery) | "v2"
    TACKLE = "v2"         # "v2": only at a held/arriving ball; "loose"
    SIM_SUPPORT = True    # support spots scored by simulated passes (anytime)
    MINIMAX_DEF = False   # defensive spots by minimax over the holder's options
    THREAT_CSP = True     # marking values from simulated 2-ply pass threats

    def __init__(self):
        super().__init__()
        self.model = None
        self.snap = None
        self.last_plan = None        # (cycle, kind, qx, qy, n, extra, speed)
        self.offside_flag_until = -1
        self.recovering = False
        self.total_ms = None
        self.match_cycles = 1500
        self.clk = None
        self.expanded = 0

    # ----------------------------------------------------------------
    # shared helpers

    TARGET_USE = 0.78     # fraction of the match CPU budget to spend
    WALL_MARGIN = 300.0   # ms of the 1 s decision cap never touched

    def _prepare(self, agent):
        if self.model is None:
            sp = agent.server_params
            self.model = Model(sp)
            ht, halves = sp.half_time or 0, sp.nr_normal_halfs or 2
            cyc = ht * halves
            if 0 < cyc < 1000:              # half_time given in seconds
                cyc = cyc * 1000 // (sp.simulator_step or 100)
            self.match_cycles = cyc if cyc > 0 else 1500
        if self.snap is None or self.snap.cycle != agent.wm.cycle:
            self.snap = Snapshot(agent, self.model)
            self._ball_cache = None
        return self.snap

    def clock(self, agent, on_ball):
        self.budget_low = agent.budget.remaining_ms < 4000.0
        """How long this decision may think. The match budget is paced so
        that about TARGET_USE of it is spent by the final whistle; on-ball
        decisions are rarer and get a bigger share."""
        b = agent.budget
        rem = b.remaining_ms
        wall_left = b.decision_remaining_ms - self.WALL_MARGIN
        if rem == float("inf"):
            allow = 150.0 if on_ball else 35.0
        else:
            if self.total_ms is None:
                self.total_ms = rem
            left = max(30, self.match_cycles - agent.wm.cycle)
            spend = rem - (1.0 - self.TARGET_USE) * self.total_ms
            per = spend / (left * 0.72)          # ~72% of cycles are play-on
            allow = _clip(per * (6.0 if on_ball else 0.85), 0.5,
                          420.0 if on_ball else 160.0)
        return Clock(allow, max(1.0, min(wall_left, 2.5 * allow + 20.0)))

    def phi(self, x, y):
        """Heuristic value of US holding the ball at (x, y). Works on arrays."""
        m = self.model
        dx = m.hl - x
        a1 = np.arctan2(y - m.goal_hw, dx)
        a2 = np.arctan2(y + m.goal_hw, dx)
        ang = np.abs(a2 - a1)
        dist = np.sqrt(dx * dx + y * y)
        base = 0.5 * (x + m.hl)
        s_old = np.clip(ang / 0.5, 0.0, 1.0) * np.exp(-np.maximum(dist - 8.0, 0.0) / 14.0)
        # shots from inside ~22 m go in most of the time in this league
        s_sig = np.clip(ang / 0.45, 0.0, 1.0) / (1.0 + np.exp((dist - 22.0) / 2.5))
        if self.PHI == "v2":
            return base + 50.0 * s_old
        if self.PHI == "new":
            return base + 95.0 * s_sig
        return base + 35.0 * s_old + 75.0 * s_sig - 0.15 * np.abs(y)

    def loss(self, x, y):
        """Value (negative) of the opponent holding the ball at (x, y)."""
        return -LOSS_WEIGHT * self.phi(-x, y)

    def ball_reach(self, s):
        """Reach time of every player to the ball's current free trajectory.
        -> (field reach (incl. me first), opp reach)"""
        if self._ball_cache is not None:
            return self._ball_cache
        m = self.model
        BP = np.empty((1, T_MAX, 2))
        BP[0, :, 0] = s.bx + s.bvx * m.G
        BP[0, :, 1] = s.by + s.bvy * m.G
        K = np.zeros(1, dtype=np.int64)
        nf = s.fP.shape[0]
        f = m.first_reach(BP, K, s.fP, s.fV, s.fU,
                          np.full(nf, m.kick_area), np.zeros(nf), 0.85)[0]
        no = s.oP.shape[0]
        octrl = np.where(s.oG, m.catch_area, m.kick_area)
        o = m.first_reach(BP, K, s.oP, s.oV, s.oU, octrl, np.zeros(no), 0.7)[0]
        # held balls: whoever holds it is at 0
        if s.kickable_opp and no:
            d = np.hypot(s.oP[:, 0] - s.bx, s.oP[:, 1] - s.by)
            o[np.argmin(d)] = 0
        self._ball_cache = (f, o, BP[0])
        return self._ball_cache

    # ----------------------------------------------------------------
    # ON THE BALL

    def on_ball(self, agent) -> None:
        s = self._prepare(agent)
        self.clk = self.clock(agent, True)
        try:
            if self.plan_with_ball(agent, s):
                return
        except Exception as e:          # never leave the ball without an order
            note(agent, "onball_error", err=str(e)[:80])
        Action.clear(agent)

    def _one_step_window(self, s, agent, ux, uy):
        """Fastest / slowest launch speed along unit (ux,uy) in one kick."""
        m = self.model
        A = min(m.max_power * agent.wm.self.kick_rate, m.accel_max)
        proj = s.bvx * ux + s.bvy * uy
        perp2 = s.bvx * s.bvx + s.bvy * s.bvy - proj * proj
        disc = A * A - perp2
        if disc < 0:
            return 0.0, 0.0
        r = math.sqrt(disc)
        return max(0.0, proj - r), min(m.bmax, proj + r)

    def follow_kick(self, agent, s):
        """A soft kick leaves the ball inside my kickable area for a cycle
        or two. If it is rolling the way last cycle's plan wanted, do not
        touch it again: run after it (dribble) or stand off (pass/shot)."""
        lp = self.last_plan
        if not lp or lp[0] != s.cycle - 1:
            return False
        _, kind, qx, qy, n, extra, speed = lp
        sp = math.hypot(s.bvx, s.bvy)
        # only once the kick has really been made (a multi-touch kick's
        # first touch just repositions the ball)
        if sp < 0.25 or sp < 0.8 * speed * self.model.decay:
            return False
        dx, dy = qx - s.bx, qy - s.by
        dist = math.hypot(dx, dy)
        if dist < 0.8:
            return False
        if (s.bvx * dx + s.bvy * dy) / (sp * dist) < 0.9:
            return False
        # an opponent about to take it: replan instead
        if len(s.oP):
            od = np.hypot(s.oP[:, 0] - s.bx, s.oP[:, 1] - s.by).min()
            if od < 2.0:
                return False
        self.last_plan = (s.cycle, kind, qx, qy, max(1, n - 1), extra, sp)
        if kind == "dribble":
            px = s.bx + s.bvx * (1.0 + self.model.decay)
            py = s.by + s.bvy * (1.0 + self.model.decay)
            note(agent, "follow", q=(round(px, 1), round(py, 1)))
            Action.go_to_point(agent, Vector2D(px, py),
                               max_dash_power=self.dash_power(agent, s, "max"),
                               distance_threshold=0.3)
            Action.turn_to_ball(agent)
        else:
            note(agent, "release")
            Action.turn_to_ball(agent)
        return True

    def plan_with_ball(self, agent, s):
        m = self.model
        wm = agent.wm
        if self.FOLLOW and self.follow_kick(agent, s):
            return True
        B = np.array([s.bx, s.by])
        f_reach, o_reach, _ = self.ball_reach(s)
        opp_now = int(o_reach.min()) if len(o_reach) else NEVER
        # distance-based pressure (o_reach is to the ball's free path)
        if len(s.oP):
            od = np.hypot(s.oP[:, 0] - s.bx, s.oP[:, 1] - s.by)
            nearest_opp = float(od.min())
            opp_now = min(opp_now, max(0, m.run_steps(max(0.0, nearest_opp - m.kick_area - 0.3))))
        else:
            nearest_opp = 99.0

        cands = []     # (kind, qx, qy, n, vx, vy, K, extra)
        # ball relative to my body: kicking "through" myself needs a
        # repositioning touch first
        rbx, rby = s.bx - s.mx, s.by - s.my
        rbn = math.hypot(rbx, rby) + 1e-6
        rbx, rby = rbx / rbn, rby / rbn

        def add(kind, qx, qy, n, speed=None, extra=None):
            dx, dy = qx - s.bx, qy - s.by
            L = math.hypot(dx, dy)
            if L < 0.5:
                return
            ux, uy = dx / L, dy / L
            sp_ = m.first_speed(L, n) if speed is None else speed
            if sp_ > m.bmax:
                sp_ = m.bmax
            if sp_ < (1.0 if kind == "pass" else 0.45):
                return          # a slow ball just sits at my feet
            lo, hi = self._one_step_window(s, agent, ux, uy)
            if lo - 0.05 <= sp_ <= hi + 0.05:
                K = 0
            elif sp_ <= 2.5:
                K = 1
            else:
                K = 2
            if K == 0 and rbx * ux + rby * uy < -0.3:
                K = 1
            cands.append((kind, qx, qy, n, sp_ * ux, sp_ * uy, K, extra))

        hl, hw = m.hl, m.hw
        # shots
        gdist = math.hypot(hl - s.bx, s.by)
        if gdist < 32.0:
            for ty in np.linspace(-m.goal_hw + 0.8, m.goal_hw - 0.8, 9):
                dx, dy = hl - s.bx, ty - s.by
                L = math.hypot(dx, dy)
                ux, uy = dx / L, dy / L
                lo, hi = self._one_step_window(s, agent, ux, uy)
                if rbx * ux + rby * uy < -0.3:
                    hi = 0.0          # the ball would run into me
                if hi > 1.8:
                    cands.append(("shoot1", hl + 1.0, float(ty), 1, hi * ux, hi * uy, 0, None))
                if hi < m.bmax - 0.15:
                    cands.append(("shoot", hl + 1.0, float(ty), 1, m.bmax * ux, m.bmax * uy,
                                  1 if hi > 1.2 else 2, None))

        # passes
        offx = max(s.offside_x, s.bx)
        dense = self.clk is not None and self.clk.left() > 120.0
        radii = (3.0, 6.0, 9.0, 13.0) if dense else (3.5, 7.0, 11.0)
        angs = (0.0, -0.5, 0.5, -1.0, 1.0, -1.57, 1.57) if dense else (0.0, -0.8, 0.8, -1.57, 1.57)
        for p in s.mates:
            if p.is_goalie:
                continue
            px, py = _xy(p.position)
            if px > offx + 0.3:           # offside now: cannot receive
                continue
            vx, vy = _xy(p.velocity)
            px, py = px + vx * 1.5, py + vy * 1.5
            gx, gy = hl - px, -py
            base = math.atan2(gy, gx)
            pts = [(px, py)]
            for r in radii:
                for da in angs:
                    a = base + da
                    pts.append((px + r * math.cos(a), py + r * math.sin(a)))
            for qx, qy in pts:
                if abs(qx) > hl - 1.0 or abs(qy) > hw - 1.0:
                    continue
                L = math.hypot(qx - s.bx, qy - s.by)
                if L < 3.0 or L > 42.0:
                    continue
                tr = m.run_steps(max(0.0, math.hypot(qx - px, qy - py) - 0.8)) + 1
                nmin = m.min_steps(L, m.bmax)
                if nmin >= NEVER:
                    continue
                ns = {max(nmin, 2), max(nmin, tr)}
                if tr > nmin + 4:
                    ns.add(max(nmin, int(0.6 * tr)))
                for n in ns:
                    add("pass", qx, qy, n, extra=p.uniform_number)

        # self passes / dribbles
        body = math.radians(s.mbody)
        ndir = 24 if dense else 16
        fwd_L = (2.5, 4.0, 6.0, 9.0, 12.0, 16.0) if dense else (2.5, 5.0, 9.0, 14.0)
        for k in range(ndir):
            a = k * 2.0 * math.pi / ndir
            ca, sa = math.cos(a), math.sin(a)
            back = ca < -0.3
            for L in ((2.5, 4.5) if back else fwd_L):
                qx, qy = s.bx + L * ca, s.by + L * sa
                if abs(qx) > hl - 0.8 or abs(qy) > hw - 0.8:
                    continue
                turn = 1 if math.cos(a - body) < 0.9 else 0
                n = m.run_steps(max(0.0, L - 0.6)) + 1 + turn
                add("dribble", qx, qy, n)

        # carry on with last cycle's kick (multi-touch kicks span cycles)
        cont = -1
        lp = self.last_plan
        if lp and lp[0] == s.cycle - 1:
            _, lk, lqx, lqy, ln, lextra, _ = lp
            before = len(cands)
            if lk in ("pass", "dribble"):
                add(lk, lqx, lqy, max(1, ln - 1), extra=lextra)
            elif lk in ("shoot", "shoot1"):
                dx, dy = lqx - s.bx, lqy - s.by
                L = math.hypot(dx, dy) + 1e-6
                cands.append(("shoot", lqx, lqy, 1, m.bmax * dx / L, m.bmax * dy / L, 1, None))
            if len(cands) > before:
                cont = len(cands) - 1

        if not cands:
            return False

        # ---- rollout through the model
        C = len(cands)
        V0 = np.array([[c[4], c[5]] for c in cands])
        K = np.array([c[6] for c in cands], dtype=np.int64)
        BP = B[None, None, :] + V0[:, None, :] * m.G[None, :, None]

        nf = s.fP.shape[0]
        fctrl = np.full(nf, m.kick_area - 0.15)
        freact = np.zeros(nf); freact[0] = 1.0
        # teammates in offside positions cannot receive
        offs = np.zeros(nf, dtype=bool)
        offs[1:] = s.fP[1:, 0] > offx + 0.3
        tm = m.first_reach(BP, K, s.fP, s.fV, s.fU, fctrl, freact, 0.9, self_col=0)
        tm[:, offs] = NEVER

        in_pa = (BP[:, :, 0] > m.pa_x) & (np.abs(BP[:, :, 1]) < m.pa_hw)
        octrl = np.where(s.oG[None, :, None] & in_pa[:, None, :],
                         m.catch_area + 0.15, m.kick_area + 0.35)
        to = m.first_reach(BP, K, s.oP, s.oV, s.oU, octrl,
                           np.zeros(len(s.onum)), 0.6)
        oslack = m.last_slack

        x = BP[:, :, 0]; y = BP[:, :, 1]
        outm = (np.abs(x) > hl) | (np.abs(y) > hw)
        t_out = np.where(outm.any(axis=1), outm.argmax(axis=1) + 1, NEVER)

        t_m = tm.min(axis=1)
        who = tm.argmin(axis=1)
        t_o = to.min(axis=1) if to.shape[1] else np.full(C, NEVER)
        # opponents can steal the ball while we wind up a multi-kick
        steal = K >= max(opp_now, 1)

        idx = np.arange(C)
        tm_i = np.clip(t_m, 1, T_MAX) - 1
        to_i = np.clip(np.minimum(t_o, t_out), 1, T_MAX) - 1
        Rx, Ry = x[idx, tm_i], y[idx, tm_i]
        Ix, Iy = x[idx, to_i], y[idx, to_i]

        v_succ = self.phi(Rx, Ry)
        v_fail = self.loss(Ix, Iy)
        # ball out of play: throw-in / goal kick for them
        out_first = t_out < t_o
        v_fail = np.where(out_first,
                          np.where(Ix > hl - 0.5, -8.0, 0.8 * self.loss(Ix, Iy)),
                          v_fail)

        margin = np.minimum(t_o, NEVER).astype(float) - t_m
        p = 1.0 / (1.0 + np.exp(-(margin - 1.0) * 1.3))
        p = np.where((t_m >= t_out) | (t_m >= NEVER), 0.0, p)
        p = np.where(steal, p * 0.25, p)

        kinds = [c[0] for c in cands]
        is_shot = np.array([k.startswith("shoot") for k in kinds])
        is_drib = np.array([k == "dribble" for k in kinds])
        speed_recv = np.hypot(V0[:, 0], V0[:, 1]) * m.decay ** np.clip(t_m - 1, 0, T_MAX)
        # hard to control a fast ball, and long balls are noisy
        p = np.where(~is_shot & (speed_recv > 2.0), p * 0.8, p)
        Ldist = np.hypot(np.array([c[1] for c in cands]) - s.bx,
                         np.array([c[2] for c in cands]) - s.by)
        p = np.where(~is_shot & ~is_drib, p * (1.0 - 0.004 * np.maximum(Ldist - 15.0, 0.0)), p)

        score = p * v_succ + (1.0 - p) * v_fail
        # a dribble keeps the ball with the same player; small time cost for passes
        score = np.where(is_drib, score - 1.0, score)

        if is_shot.any():
            si = np.nonzero(is_shot)[0]
            gctrl = np.where(s.oG, GOALIE_CTRL, m.kick_area + 0.35)
            greact = np.where(s.oG, GOALIE_REACT, 0.0)
            m.first_reach(BP[si], K[si], s.oP, s.oV, s.oU, gctrl, greact, 0.6)
            pg = np.zeros(C)
            pg[si], _ = m.shot_success(BP[si], m.last_slack, None, gdist)
            pg = np.where(steal, pg * 0.3, pg)
            v_shot = pg * GOAL_VALUE + (1.0 - pg) * self.loss(Ix, Iy)
            score = np.where(is_shot, v_shot, score)
            p = np.where(is_shot, pg, p)

        # hysteresis: a half-finished multi-touch kick is worth finishing
        if cont >= 0:
            score[cont] += 12.0
        best_shot_p = float(p[is_shot].max()) if is_shot.any() else 0.0

        # hold the ball
        p_hold = 1.0 / (1.0 + math.exp(-(opp_now - 2.0) * 1.2))
        v_hold = p_hold * float(self.phi(s.bx, s.by)) + (1 - p_hold) * float(self.loss(s.bx, s.by)) - 4.0

        best = int(np.argmax(score))
        # 2nd ply: refine the top few with what the receiver could do next
        best = self.refine(agent, s, cands, score, p, BP, t_m, who, best)

        if score[best] < v_hold:
            note(agent, "hold", v=round(v_hold, 1))
            Action.hold(agent)
            self.last_plan = None
            return True

        kind, qx, qy, n, vx, vy, Kb, extra = cands[best]
        self.last_plan = (s.cycle, kind, qx, qy, n, extra, math.hypot(vx, vy))
        tgt = Vector2D(qx, qy)
        note(agent, kind, to=extra, q=(round(qx, 1), round(qy, 1)), n=n, K=Kb, x2=self.expanded,
             p=round(float(p[best]), 2), v=round(float(score[best]), 1),
             ps=round(best_shot_p, 2), cont=int(best == cont))
        if kind == "shoot1":
            Action.kick_one_step(agent, tgt)
        elif kind == "shoot":
            Action.kick(agent, tgt, 1)
        elif kind == "pass":
            Action.kick(agent, tgt, n, accept_late=2)
        else:
            Action.kick(agent, tgt, n, accept_late=3)
        return True

    BEAM = 12

    def predict_opps(self, s, rx, ry, t):
        """Opponents after t cycles of running at the ball (goalie: towards
        the near post side of the goal mouth)."""
        m = self.model
        P = s.oP.copy()
        if not len(P):
            return P
        reach = m.dash[int(_clip(t, 0, len(m.dash) - 1))]
        gx = m.hl
        for j in range(len(P)):
            ox, oy = P[j]
            if s.oG[j]:
                dgx, dgy = rx - gx, ry
                dn = math.hypot(dgx, dgy) + 1e-6
                tx, ty = gx + dgx / dn * min(3.5, dn), dgy / dn * min(3.5, dn)
                stop = 0.0
            else:
                tx, ty, stop = rx, ry, 1.0
            dx, dy = tx - ox, ty - oy
            dd = math.hypot(dx, dy) + 1e-6
            mv = min(reach, max(0.0, dd - stop))
            P[j] = (ox + dx / dd * mv, oy + dy / dd * mv)
        return P

    def predict_players(self, P, G, rx, ry, t, goal_x):
        """Any side's players after t cycles of running at (rx, ry); a goalie
        steps across the mouth of the goal at goal_x instead."""
        m = self.model
        out = P.copy()
        reach = m.dash[int(_clip(t, 0, len(m.dash) - 1))]
        for j in range(len(P)):
            ox, oy = P[j]
            if G[j]:
                dgx, dgy = rx - goal_x, ry
                dn = math.hypot(dgx, dgy) + 1e-6
                tx, ty = goal_x + dgx / dn * min(3.5, dn), dgy / dn * min(3.5, dn)
                stop = 0.0
            else:
                tx, ty, stop = rx, ry, 1.0
            dx, dy = tx - ox, ty - oy
            dd = math.hypot(dx, dy) + 1e-6
            mv = min(reach, max(0.0, dd - stop))
            out[j] = (ox + dx / dd * mv, oy + dy / dd * mv)
        return out

    def shot_prob(self, rx, ry, oP, delay, speed=2.8, n=7):
        """Best goal probability of a shot from (rx, ry) after `delay`
        cycles, against opponents standing at oP (bodies facing the ball)."""
        m = self.model
        dist = math.hypot(m.hl - rx, ry)
        if dist > 30.0 or not len(oP):
            return 0.0
        ty = np.linspace(-m.goal_hw + 0.8, m.goal_hw - 0.8, n)
        dx = m.hl - rx
        dy = ty - ry
        L = np.hypot(dx, dy)
        V = np.stack([speed * dx / L, speed * dy / L], axis=1)
        BP = np.array([rx, ry])[None, None, :] + V[:, None, :] * m.G[None, :, None]
        K = np.full(n, delay, dtype=np.int64)
        U = np.array([[rx, ry]]) - oP
        U = U / (np.hypot(U[:, 0], U[:, 1])[:, None] + 1e-6)
        isg = np.zeros(len(oP), dtype=bool)
        isg[:len(self.snap.oG)] = self.snap.oG
        ctrl = np.where(isg, GOALIE_CTRL, m.kick_area + 0.35)
        m.first_reach(BP, K, oP, np.zeros_like(oP), U, ctrl,
                      np.where(isg, GOALIE_REACT, 0.0), 0.6)
        pg, _ = m.shot_success(BP, m.last_slack, None, dist)
        return float(pg.max())

    def options(self, hx, hy, delay, A, holder, D, spots=None, dense=1):
        """Expected value of every option open to the side holding the ball
        at (hx, hy), attacking +x: shots, passes (direct and lead) and
        dribbles, each a chance node p * V(success) + (1 - p) * V(loss).

        A, D: (P, V, U, G) arrays of the holder's side and of the defenders.
        spots: optional (S, 2) positions for ONE extra defender; the values
        then come back per spot, which is how minimax positioning scores many
        defensive spots in a single batch.
        -> (values (C, S), kinds (C,), BP, t_rec (C,), p (C, S),
            reception x, y (C,), receiver (C,)) or None"""
        m = self.model
        AP, AV, AU, AG = A
        DP, DV, DU, DG = D
        hl, hw = m.hl, m.hw
        xs = np.sort(DP[:, 0])[::-1] if len(DP) else np.array([hl, hl])
        offx = max(float(xs[1]) if len(xs) > 1 else hl, hx)
        rows = []                      # qx, qy, speed, kind (0 shot, 1 pass, 2 dribble)
        if math.hypot(hl - hx, hy) < 30.0:
            for ty in np.linspace(-m.goal_hw + 0.8, m.goal_hw - 0.8, 3 + 2 * dense):
                rows.append((hl + 1.0, float(ty), 2.8, 0))
        radii = (4.0, 8.0) if dense < 2 else (3.5, 7.0, 11.0)
        angs = (0.0, -0.8, 0.8) if dense < 2 else (0.0, -0.8, 0.8, -1.57, 1.57)
        for j in range(len(AP)):
            if j == holder or AG[j] or AP[j, 0] > offx + 0.3:
                continue
            px, py = AP[j, 0] + AV[j, 0] * 1.5, AP[j, 1] + AV[j, 1] * 1.5
            base = math.atan2(-py, hl - px)
            pts = [(px, py)] + [(px + r * math.cos(base + da), py + r * math.sin(base + da))
                                for r in radii for da in angs]
            for qx, qy in pts:
                if abs(qx) > hl - 1.0 or abs(qy) > hw - 1.0:
                    continue
                L = math.hypot(qx - hx, qy - hy)
                if L < 3.0 or L > 40.0:
                    continue
                nmin = m.min_steps(L, m.bmax)
                if nmin >= NEVER:
                    continue
                tr = m.run_steps(max(0.0, math.hypot(qx - px, qy - py) - 0.8)) + 1
                for n in {max(nmin, 2), max(nmin, tr)}:
                    sp_ = min(m.first_speed(L, n), m.bmax)
                    if sp_ >= 1.0:
                        rows.append((qx, qy, sp_, 1))
        nd = 8 * dense
        for k in range(nd):
            ang = 2.0 * math.pi * k / nd
            for L in (5.0, 10.0):
                qx, qy = hx + L * math.cos(ang), hy + L * math.sin(ang)
                if abs(qx) > hl - 0.8 or abs(qy) > hw - 0.8:
                    continue
                n = m.run_steps(L - 0.6) + 1
                sp_ = m.first_speed(L, n)
                if sp_ >= 0.45:
                    rows.append((qx, qy, sp_, 2))
        if not rows:
            return None
        Q = np.array(rows)
        C = len(Q)
        kind = Q[:, 3].astype(np.int64)
        ddx, ddy = Q[:, 0] - hx, Q[:, 1] - hy
        Ln = np.hypot(ddx, ddy) + 1e-9
        V0 = np.stack([Q[:, 2] * ddx / Ln, Q[:, 2] * ddy / Ln], axis=1)
        K = np.full(C, int(delay), dtype=np.int64)
        BP = np.array([hx, hy])[None, None, :] + V0[:, None, :] * m.G[None, :, None]

        nA = len(AP)
        areact = np.zeros(nA)
        areact[holder] = 1.0
        ta = m.first_reach(BP, K, AP, AV, AU, np.full(nA, m.kick_area - 0.15),
                           areact, 0.9, self_col=holder)
        bad = AG | (AP[:, 0] > offx + 0.3)
        bad[holder] = False
        ta[:, bad] = NEVER
        t_a = ta.min(axis=1)
        who = ta.argmin(axis=1)

        x = BP[:, :, 0]; y = BP[:, :, 1]
        in_pa = (x > m.pa_x) & (np.abs(y) < m.pa_hw)
        if len(DP):
            dctrl = np.where(DG[None, :, None] & in_pa[:, None, :],
                             m.catch_area + 0.15, m.kick_area + 0.35)
            t_d = m.first_reach(BP, K, DP, DV, DU, dctrl, np.zeros(len(DP)), 0.6).min(axis=1)
        else:
            t_d = np.full(C, NEVER)
        S = 1
        if spots is not None:
            S = len(spots)
            SU = np.array([hx, hy])[None, :] - spots
            SU = SU / (np.hypot(SU[:, 0], SU[:, 1])[:, None] + 1e-9)
            ts = m.first_reach(BP, K, spots, np.zeros_like(spots), SU,
                               np.full(S, m.kick_area + 0.35), np.zeros(S), 0.6)
            t_d = np.minimum(t_d[:, None], ts)
        else:
            t_d = t_d[:, None]

        outm = (np.abs(x) > hl) | (np.abs(y) > hw)
        t_out = np.where(outm.any(axis=1), outm.argmax(axis=1) + 1, NEVER)
        idx = np.arange(C)
        Rx = x[idx, np.clip(t_a, 1, T_MAX) - 1]
        Ry = y[idx, np.clip(t_a, 1, T_MAX) - 1]
        fi = np.clip(np.minimum(t_d, t_out[:, None]), 1, T_MAX) - 1
        Ix = x[idx[:, None], fi]
        Iy = y[idx[:, None], fi]
        margin = t_d - t_a[:, None]
        p = 1.0 / (1.0 + np.exp(-(margin - 1.0) * 1.3))
        dead = (t_a >= t_out) | (t_a >= NEVER)
        p = np.where(dead[:, None], 0.0, p)
        v_fail = np.where(t_out[:, None] < t_d,
                          np.where(Ix > hl - 0.5, -8.0, 0.8 * self.loss(Ix, Iy)),
                          self.loss(Ix, Iy))
        val = p * self.phi(Rx, Ry)[:, None] + (1.0 - p) * v_fail
        val = np.where((kind == 2)[:, None], val - 1.0, val)

        si = np.nonzero(kind == 0)[0]
        if len(si):
            BPs = BP[si]
            if len(DP):
                gctrl = np.where(DG, GOALIE_CTRL, m.kick_area + 0.35)
                m.first_reach(BPs, K[si], DP, DV, DU, gctrl,
                              np.where(DG, GOALIE_REACT, 0.0), 0.6)
                sl = m.last_slack.max(axis=1)[:, None, :]
            else:
                sl = np.full((len(si), 1, T_MAX), -50.0)
            if spots is not None:
                m.first_reach(BPs, K[si], spots, np.zeros_like(spots), SU,
                              np.full(S, m.kick_area + 0.35), np.zeros(S), 0.6)
                sl = np.maximum(sl, m.last_slack)
            cross = BPs[:, :, 0] > hl
            t_goal = np.where(cross.any(axis=1), cross.argmax(axis=1), T_MAX - 1)
            tmask = np.arange(T_MAX)[None, :] <= t_goal[:, None]
            slm = np.where(tmask[:, None, :], sl, -50.0).max(axis=2)
            y_at = BPs[np.arange(len(si)), t_goal, 1]
            err = 0.03 * math.hypot(hl - hx, hy) + 0.3
            post = np.clip((m.goal_hw - np.abs(y_at)) / err, 0.0, 1.0)
            ok = cross.any(axis=1) & (np.abs(y_at) < m.goal_hw - 0.2)
            pg = np.where(ok[:, None], post[:, None] / (1.0 + np.exp(slm / 0.6)), 0.0)
            val[si] = pg * GOAL_VALUE + (1.0 - pg) * self.loss(Ix[si], Iy[si])
            p[si] = pg
        return val, kind, BP, t_a, p, Rx, Ry, who

    def refine(self, agent, s, cands, score, p, BP, t_m, who, best):
        """Expectimax, second ply, anytime: expand candidates in order of
        their one-ply score with the receiver's own best option, in a world
        advanced to the reception (opponents have run at the ball), while
        this decision's CPU allowance lasts."""
        if not self.REFINE or self.clk is None:
            return best
        m = self.model
        AP = np.vstack([s.fP, s.gP]) if len(s.gP) else s.fP
        AV = np.vstack([s.fV, np.zeros_like(s.gP)]) if len(s.gP) else s.fV
        AG = np.zeros(len(AP), dtype=bool)
        AG[len(s.fP):] = True
        new = score.copy()
        done = 0
        for c in np.argsort(-score):
            if self.clk.left() < 3.0:
                break
            if cands[c][0].startswith("shoot") or p[c] < 0.15 or t_m[c] >= NEVER:
                continue
            tm = int(t_m[c])
            ti = min(tm, T_MAX) - 1
            rx, ry = float(BP[c, ti, 0]), float(BP[c, ti, 1])
            r = int(who[c])
            P2 = AP.copy()
            P2[r] = (rx, ry)
            V2 = AV.copy()
            V2[r] = 0.0
            U2 = np.tile([[1.0, 0.0]], (len(AP), 1))
            D2 = self.predict_opps(s, rx, ry, tm)
            DU = np.array([[rx, ry]]) - D2
            DU = DU / (np.hypot(DU[:, 0], DU[:, 1])[:, None] + 1e-9)
            res = self.options(rx, ry, 1, (P2, V2, U2, AG), r,
                               (D2, np.zeros_like(D2), DU, s.oG),
                               dense=2 if self.clk.left() > 60.0 else 1)
            done += 1
            if res is None:
                continue
            nxt = float(res[0].max())
            v_succ = float(self.phi(rx, ry))
            new[c] = score[c] + p[c] * (max(v_succ, 0.95 * nxt) - v_succ)
        self.expanded = done
        score[:] = new
        return int(np.argmax(score))

    # ----------------------------------------------------------------
    # OFF THE BALL

    def off_ball(self, agent) -> None:
        s = self._prepare(agent)
        self.clk = self.clock(agent, False)
        try:
            if self.plan_without_ball(agent, s):
                return
        except Exception as e:
            note(agent, "offball_error", err=str(e)[:80])
        Action.go_to_point(agent, Tools.home_position(agent))

    def plan_without_ball(self, agent, s):
        if self.OFFBALL == "v2":
            return self.plan_without_ball_v2(agent, s)
        m = self.model
        it = agent.wm.intercept_table
        self_min = it.self_reach_steps
        mate_min = it.first_teammate_reach_steps
        opp_min = it.first_opponent_reach_steps
        f_reach, o_reach, ball_path = self.ball_reach(s)

        # offside bookkeeping: if a mate is about to play the ball and I
        # stand offside, stay out of it until the ball changes hands
        if s.kickable_mate and s.mx > max(s.offside_x, s.bx) + 0.1:
            self.offside_flag_until = s.cycle + 25
        if s.kickable_opp:
            self.offside_flag_until = -1
        offside_now = s.cycle <= self.offside_flag_until

        # 1) tackle
        if self.try_tackle(agent, s, self_min, opp_min):
            return True

        # 2) go for the ball: anyone very close, or the fastest of us while
        #    the opponent is not clearly first
        if not s.kickable_mate and not offside_now:
            if self_min <= 3 or (self_min <= mate_min and self_min < opp_min + 3):
                note(agent, "chase", t=self_min, opp=opp_min)
                Action.intercept(agent)
                return True

        our_ball = s.kickable_mate or (not s.kickable_opp
                                       and min(self_min, mate_min) < opp_min)
        # ball position the formation is built around
        t_all = int(_clip(min(self_min, mate_min, opp_min), 0, T_MAX))
        if t_all == 0:
            fbx, fby = s.bx, s.by
        else:
            fbx, fby = float(ball_path[t_all - 1, 0]), float(ball_path[t_all - 1, 1])
        slots = self.formation(fbx, fby, s.offside_x)
        if our_ball:
            roles = self.assign_roles(s, slots)
            return self.support(agent, s, f_reach, ball_path, slots, roles)

        return self.defend(agent, s, ball_path, opp_min, slots)

    def try_tackle(self, agent, s, self_min, opp_min):
        wm = agent.wm
        prob = wm.self.tackle_probability
        if prob <= 0.0:
            return False
        if self.TACKLE == "loose":
            if not (s.kickable_opp or opp_min < self_min):
                return False
        elif not (s.kickable_opp or (opp_min <= 1 and opp_min < self_min)):
            return False
        m = self.model
        dgoal = math.hypot(s.bx + m.hl, s.by)
        thr = 0.8
        if dgoal < 30.0:
            thr = 0.7
        if dgoal < 18.0:
            thr = 0.55
        if prob < thr:
            return False
        # send it towards their goal, away from ours when deep
        if s.bx < -30.0:
            tx, ty = s.bx + 20.0, (25.0 if s.by >= 0 else -25.0)
        else:
            tx, ty = m.hl, 0.0
        ang = math.degrees(math.atan2(ty - s.by, tx - s.bx))
        rel = _norm_deg(ang - s.mbody)
        note(agent, "tackle", prob=round(prob, 2))
        Action.tackle(agent, rel, False)
        return True

    # --- formation and roles -------------------------------------------

    def formation(self, bx, by, offside_x):
        """Slots for the four outfield roles as a function of the ball:
        a least-squares fit (rmse 2-3.5 m) to the 5-a-side reference
        formation, so the shape is compact and goal-side of the ball."""
        m = self.model
        q = bx * bx / 50.0
        ay = abs(by)
        c = bx * by / 50.0
        raw = {
            2: (-12.4 + 0.537 * bx - 0.251 * q + 0.026 * ay, 0.36 * by),
            3: (-3.9 + 0.851 * bx + 0.08 * by + 0.049 * ay + 0.066 * c,
                -9.0 + 0.454 * by + 0.102 * q - 0.12 * ay),
            4: (-3.9 + 0.851 * bx - 0.08 * by + 0.049 * ay - 0.066 * c,
                9.0 + 0.454 * by - 0.102 * q + 0.12 * ay),
            5: (8.2 + 0.73 * bx - 0.019 * q + 0.155 * ay, 0.432 * by + 0.187 * c),
        }
        out = {}
        for r, (x, y) in raw.items():
            x = _clip(x, -m.hl + 3.0, min(m.hl - 3.0, offside_x - 0.5))
            y = _clip(y, -m.hw + 2.0, m.hw - 2.0)
            out[r] = (x, y)
        return out

    def assign_roles(self, s, slots):
        """Which player takes which slot: an assignment CSP (all-different),
        cost = distance to the slot + a small penalty for leaving one's
        own role, solved by backtracking with MRV + forward checking +
        branch & bound. Every player solves the same problem from the same
        state, so they agree without talking."""
        players = sorted(s.fnum)
        rlist = sorted(slots)
        pos = {u: s.fP[i] for i, u in enumerate(s.fnum)}

        def cost(u, r):
            px, py = pos[u]
            sx, sy = slots[r]
            return math.hypot(sx - px, sy - py) + (0.0 if u == r else 6.0)

        sol = self.csp_assign(players, rlist, cost)
        return {u: rlist[v] for u, v in sol.items()}

    def csp_assign(self, variables, values, cost_fn):
        """Generic all-different assignment: minimise the summed cost.
        Backtracking search, MRV variable ordering, least-cost value
        ordering, forward checking and a branch & bound cut."""
        nv = len(values)
        cost = {(u, v): cost_fn(u, values[v]) for u in variables for v in range(nv)}
        best = [float("inf"), {}]
        domains = {u: list(range(nv)) for u in variables}

        def bound(assigned, cur):
            lb = cur
            for u in variables:
                if u not in assigned and domains[u]:
                    lb += min(cost[(u, v)] for v in domains[u])
            return lb

        def rec(assigned, cur):
            if len(assigned) == len(variables) or len(assigned) == nv:
                if cur < best[0]:
                    best[0], best[1] = cur, dict(assigned)
                return
            if bound(assigned, cur) >= best[0]:
                return
            u = min((x for x in variables if x not in assigned),
                    key=lambda x: (len(domains[x]), x))           # MRV
            for v in sorted(domains[u], key=lambda v: cost[(u, v)]):
                pruned = []                                        # forward checking
                for w in variables:
                    if w not in assigned and w != u and v in domains[w]:
                        domains[w].remove(v)
                        pruned.append(w)
                if all(domains[w] for w in variables if w not in assigned and w != u) \
                        or len(assigned) + 1 >= nv:
                    assigned[u] = v
                    rec(assigned, cur + cost[(u, v)])
                    del assigned[u]
                for w in pruned:
                    domains[w].append(v)

        rec({}, 0.0)
        return best[1]

    # --- attack support --------------------------------------------------

    def support(self, agent, s, f_reach, ball_path, slots, roles):
        m = self.model
        if s.kickable_mate:
            hx, hy = s.bx, s.by
        else:
            t_us = int(f_reach.min())
            ti = int(_clip(t_us, 1, T_MAX)) - 1
            hx, hy = float(ball_path[ti, 0]), float(ball_path[ti, 1])
        holder = s.fnum[int(np.argmin(f_reach))]
        offx = max(s.offside_x, hx)
        idx_of = {u: i for i, u in enumerate(s.fnum)}
        # the most advanced opponent outfielder (their next counter-attack)
        fwd = None
        for j in range(len(s.onum)):
            if not s.oG[j] and (fwd is None or s.oP[j, 0] < fwd[0]):
                fwd = (float(s.oP[j, 0]), float(s.oP[j, 1]))

        chosen = []
        my_target = None
        order = sorted((u for u in s.fnum if u != holder),
                       key=lambda u: (roles.get(u, u) != 2, u))
        for u in order:
            r = roles.get(u, u)
            px, py = s.fP[idx_of[u]]
            ax, ay = slots[r]
            if r == 2 and fwd is not None:
                # the back stays goal-side of their most advanced player
                ax = max(min(ax, fwd[0] - 3.0), -m.hl + 8.0)
                ay = 0.5 * ay + 0.5 * fwd[1]
            q = self.best_spot(s, ax, ay, px, py, hx, hy, offx, chosen, tight=(r == 2))
            chosen.append(q)
            if u == s.unum:
                my_target = q
                break
        if my_target is None:
            return False
        tx, ty = my_target
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, "normal" if d > 6.0 else "low")
        note(agent, "support", q=(round(tx, 1), round(ty, 1)))
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=1.0)
        Action.turn_to_ball(agent)
        return True

    def best_spot(self, s, ax, ay, px, py, hx, hy, offx, chosen, tight=False):
        """Greedy local search over a grid round the formation slot: open
        passing lane from the holder, space, threat, spacing."""
        m = self.model
        g = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        QX = (ax + g[:, None] + 0 * g[None, :]).ravel()
        QY = (ay + 0 * g[:, None] + g[None, :]).ravel()
        QX = np.clip(QX, -m.hl + 3.0, min(m.hl - 3.0, offx - 0.8))
        QY = np.clip(QY, -m.hw + 2.0, m.hw - 2.0)
        val = 0.5 * self.phi(QX, QY)
        if len(s.oP):
            ox = s.oP[:, 0][None, :]; oy = s.oP[:, 1][None, :]
            vx = (QX - hx)[:, None]; vy = (QY - hy)[:, None]
            L2 = vx * vx + vy * vy + 1e-6
            tpar = np.clip(((ox - hx) * vx + (oy - hy) * vy) / L2, 0.0, 1.0)
            cx = hx + tpar * vx; cy = hy + tpar * vy
            dseg = np.hypot(ox - cx, oy - cy)
            along = tpar * np.sqrt(L2)
            lane = np.min(dseg - 0.15 * along, axis=1)
            space = np.min(np.hypot(ox - QX[:, None], oy - QY[:, None]), axis=1)
            val += 5.0 * np.clip(lane, -3.0, 4.0) + 2.0 * np.clip(space - 4.0, -4.0, 6.0)
        dh = np.hypot(QX - hx, QY - hy)
        val -= 3.0 * np.clip(7.0 - dh, 0.0, 7.0)
        val -= 1.0 * np.clip(dh - 30.0, 0.0, 20.0)
        for cx, cy in chosen:
            val -= 3.0 * np.clip(9.0 - np.hypot(QX - cx, QY - cy), 0.0, 9.0)
        val -= (2.0 if tight else 0.35) * np.hypot(QX - ax, QY - ay)
        val -= 0.15 * np.hypot(QX - px, QY - py)
        k = int(np.argmax(val))
        return float(QX[k]), float(QY[k])

    # --- defence -----------------------------------------------------------

    def defend(self, agent, s, ball_path, opp_min, slots):
        """Opponent ball. One blocker gets goal-side of the owner's path; the
        others share out jobs (goal-side marks weighted by how close the
        man is to our goal, one shot-lane cover, or a formation slot) by
        solving an assignment CSP that every player solves identically."""
        m = self.model
        gx, gy = -m.hl, 0.0
        ti = int(_clip(opp_min, 0, T_MAX))
        if ti == 0:
            hx, hy = s.bx, s.by
        else:
            hx, hy = float(ball_path[ti - 1, 0]), float(ball_path[ti - 1, 1])
        blocker, (Dx, Dy), k = self.block_plan(s, hx, hy, ti)
        if blocker == s.unum:
            return self.press(agent, s, hx, hy, Dx, Dy, k)

        owner = -1
        if len(s.oP):
            owner = int(np.argmin(np.hypot(s.oP[:, 0] - hx, s.oP[:, 1] - hy)))
        jobs = []                      # (tag, id, x, y, value)
        for j, onum in enumerate(s.onum):
            if j == owner or s.oG[j]:
                continue
            ox = s.oP[j, 0] + s.oV[j, 0] * 2.0
            oy = s.oP[j, 1] + s.oV[j, 1] * 2.0
            dgo = math.hypot(ox - gx, oy - gy) + 1e-6
            r = 1.0 + 0.04 * min(dgo, 40.0)
            ux, uy = (gx - ox) / dgo, (gy - oy) / dgo
            nb = math.hypot(hx - ox, hy - oy) + 1e-6
            dx_ = 0.75 * ux + 0.25 * (hx - ox) / nb
            dy_ = 0.75 * uy + 0.25 * (hy - oy) / nb
            nd = math.hypot(dx_, dy_) + 1e-6
            sx, sy = ox + r * dx_ / nd, oy + r * dy_ / nd
            danger = 75.0 * math.exp(-dgo / 20.0) + (8.0 if ox < hx else 0.0)
            jobs.append(("mark", onum, sx, sy, danger))
        # shot-lane cover on the owner's line to goal
        dH = math.hypot(hx - gx, hy - gy) + 1e-6
        dc = _clip(dH - 7.0, 6.0, 24.0)
        jobs.append(("cover", None, gx + (hx - gx) / dH * dc, gy + (hy - gy) / dH * dc,
                     15.0 + 80.0 * math.exp(-dH / 25.0)))
        for r, (fx, fy) in sorted(slots.items()):
            jobs.append(("slot", r, fx, fy, 3.0))

        defenders = [u for u in sorted(s.fnum) if u != blocker]
        pos = {u: s.fP[i] for i, u in enumerate(s.fnum)}

        def cost(u, job):
            px, py = pos[u]
            return math.hypot(job[2] - px, job[3] - py) - job[4]

        sol = self.csp_assign(defenders, jobs, cost)
        job = jobs[sol[s.unum]] if s.unum in sol else jobs[-1]
        tag, jid, tx, ty, _ = job
        d = math.hypot(tx - s.mx, ty - s.my)
        if tag != "slot" and (d > 3.0 or s.mx > tx + 1.0):
            level = "high"
        elif d > 3.0:
            level = "normal"
        else:
            level = "low"
        note(agent, tag, o=jid, q=(round(tx, 1), round(ty, 1)))
        Action.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self.dash_power(agent, s, level),
                           distance_threshold=0.7)
        Action.turn_to_ball(agent)
        return True

    DRIBBLE_SPEED = 1.0

    def block_plan(self, s, hx, hy, t0):
        """Where the ball owner will dribble (straight at our goal), and the
        earliest point on that path each of our field players can get to
        first. -> (blocker unum, (x, y) path arrays, path index)"""
        m = self.model
        gx, gy = -m.hl + 3.0, 0.0
        dx, dy = gx - hx, gy - hy
        dist = math.hypot(dx, dy) + 1e-6
        ux, uy = dx / dist, dy / dist
        tt = np.arange(0, T_MAX, dtype=float)
        step = np.minimum(tt * self.DRIBBLE_SPEED, max(dist - 2.0, 0.0))
        Dx = hx + ux * step
        Dy = hy + uy * step
        P, V, U = s.fP, s.fV, s.fU
        Pi = P[:, None, :] + V[:, None, :] * m.F[None, :1, None]
        ex = Dx[None, :] - Pi[:, :, 0]
        ey = Dy[None, :] - Pi[:, :, 1]
        d = np.hypot(ex, ey) + 1e-9
        cos = (ex * U[:, None, 0] + ey * U[:, None, 1]) / d
        turn = (cos < 0.85) & (d > 2.0)
        avail = np.clip(t0 + tt[None, :] - turn, 0, len(m.dash) - 1).astype(np.int64)
        ok = d - 1.0 <= m.dash[avail]
        first = np.where(ok.any(axis=1), ok.argmax(axis=1), NEVER)
        order = sorted(range(len(s.fnum)), key=lambda i: (first[i], d[i, 0], s.fnum[i]))
        bi = order[0]
        k = int(min(first[bi], T_MAX - 1)) if first[bi] < NEVER else 0
        return s.fnum[bi], (Dx, Dy), k

    def press(self, agent, s, hx, hy, Dx, Dy, k):
        """Get in front of the dribble path, then go for the ball."""
        m = self.model
        gx, gy = -m.hl, 0.0
        d = math.hypot(hx - gx, hy - gy) + 1e-6
        ux, uy = (gx - hx) / d, (gy - hy) / d
        dm = math.hypot(s.bx - s.mx, s.by - s.my)
        goal_side = (s.mx - s.bx) * ux + (s.my - s.by) * uy > 0.5
        if dm < 2.0 or (goal_side and dm < 5.0):
            note(agent, "press_ball")
            Action.intercept(agent)
            return True
        kk = int(min(k + 1, len(Dx) - 1))
        tx, ty = float(Dx[kk]), float(Dy[kk])
        note(agent, "block", q=(round(tx, 1), round(ty, 1)), k=k)
        Action.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self.dash_power(agent, s, "max"),
                           distance_threshold=0.5)
        Action.intercept(agent)
        return True

    def dash_power(self, agent, s, level):
        """Stamina policy. Below 45% we go into recovery (dash under the
        recovery rate) until back above 70%; only 'max' moves (ball
        winning, blocking) still sprint, and never below the effort floor."""
        stamina_max = agent.server_params.stamina_max or 8000.0
        r = s.stamina / stamina_max
        if self.STAMINA == "v2":
            if r < 0.32:
                return 30.0
            if level in ("max", "high"):
                return 100.0 if r > 0.45 else 70.0
            return 60.0 if r > 0.6 else 40.0
        if r < 0.45:
            self.recovering = True
        elif r > 0.7:
            self.recovering = False
        if level == "max":
            return 100.0 if r > 0.33 else 45.0
        if self.recovering:
            return 30.0
        return {"high": 100.0, "normal": 70.0}.get(level, 45.0)


    # ----------------------------------------------------------------
    # OFF THE BALL, "v2" design: one chaser from our own reach model, a
    # blocker on the owner's dribble path, a marking CSP for the rest and
    # role anchors + local search when we have the ball.

    def plan_without_ball_v2(self, agent, s):
        f_reach, o_reach, ball_path = self.ball_reach(s)
        t_me = int(f_reach[0])
        t_us = int(f_reach.min())
        t_opp = int(o_reach.min()) if len(o_reach) else NEVER
        order = sorted(range(len(s.fnum)), key=lambda i: (f_reach[i], s.fnum[i]))
        chaser = s.fnum[order[0]]

        if s.kickable_mate and s.mx > max(s.offside_x, s.bx) + 0.1:
            self.offside_flag_until = s.cycle + 25
        if s.kickable_opp:
            self.offside_flag_until = -1
        offside_now = s.cycle <= self.offside_flag_until

        if self.try_tackle(agent, s, t_me, t_opp):
            return True

        our_ball = s.kickable_mate or (not s.kickable_opp and t_us < t_opp)
        if not s.kickable_mate and not offside_now:
            if chaser == s.unum and (our_ball or t_me <= t_opp + 1):
                note(agent, "chase", t=t_me, opp=t_opp)
                Action.intercept(agent)
                return True

        if our_ball:
            return self.support_v2(agent, s, f_reach, ball_path, t_us)
        return self.defend_v2(agent, s, o_reach, ball_path, t_opp)

    ROLE_ANCHOR = {
        # unum: (dx from ball, x min, x max, base y, y follow)
        2: (-22.0, -42.0, 5.0, 0.0, 0.35),
        3: (2.0, -35.0, 40.0, -17.0, 0.35),
        4: (2.0, -35.0, 40.0, 17.0, 0.35),
        5: (13.0, -20.0, 44.0, 0.0, 0.3),
    }

    def anchor(self, unum, bx, by, offx):
        dx, xlo, xhi, y0, yf = self.ROLE_ANCHOR.get(unum, (0.0, -30.0, 30.0, 0.0, 0.3))
        ax = _clip(bx + dx, xlo, xhi)
        ay = _clip(y0 + by * yf, -self.model.hw + 4.0, self.model.hw - 4.0)
        return min(ax, offx - 1.0), ay

    def support_v2(self, agent, s, f_reach, ball_path, t_us):
        if s.kickable_mate:
            hx, hy = s.bx, s.by
        else:
            ti = int(_clip(t_us, 1, T_MAX)) - 1
            hx, hy = float(ball_path[ti, 0]), float(ball_path[ti, 1])
        holder = s.fnum[int(np.argmin(f_reach))]
        offx = max(s.offside_x, hx)
        chosen = []
        my_target = None
        idx_of = {u: i for i, u in enumerate(s.fnum)}
        for u in sorted(s.fnum):
            if u == holder:
                continue
            px, py = s.fP[idx_of[u]]
            ax, ay = self.anchor(u, hx, hy, offx)
            q = self.best_spot(s, ax, ay, px, py, hx, hy, offx, chosen)
            if u == s.unum and not self.SIM_SUPPORT:
                my_target = q
                break
            if u == s.unum:
                delay = 1 if s.kickable_mate else int(_clip(t_us, 1, T_MAX))
                my_target = self.sim_support_spot(s, q[0], q[1], ax, ay, hx, hy,
                                                  delay, offx, chosen)
                break
            chosen.append(q)
        if my_target is None:
            return False
        tx, ty = my_target
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, "high" if d > 8.0 else "normal")
        note(agent, "support", q=(round(tx, 1), round(ty, 1)))
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=1.0)
        Action.turn_to_ball(agent)
        return True

    def defend_v2(self, agent, s, o_reach, ball_path, t_opp):
        m = self.model
        gx, gy = -m.hl, 0.0
        ti = int(_clip(t_opp, 0, T_MAX))
        if ti == 0:
            hx, hy = s.bx, s.by
        else:
            hx, hy = float(ball_path[ti - 1, 0]), float(ball_path[ti - 1, 1])
        holder_i = int(np.argmin(o_reach)) if len(o_reach) else -1

        blocker, (Dx, Dy), k = self.block_plan(s, hx, hy, ti)
        if s.unum == blocker:
            return self.press(agent, s, hx, hy, Dx, Dy, k)
        k2 = int(min(k + 12, T_MAX - 1))
        cover = (max(float(Dx[k2]), -m.hl + 4.0), float(Dy[k2]))

        jobs = []
        for j, onum in enumerate(s.onum):
            if j == holder_i or s.oG[j]:
                continue
            ox = s.oP[j, 0] + s.oV[j, 0] * 2.0
            oy = s.oP[j, 1] + s.oV[j, 1] * 2.0
            dgo = math.hypot(ox - gx, oy - gy) + 1e-6
            r = 1.0 + 0.05 * min(dgo, 40.0)
            nb = math.hypot(hx - ox, hy - oy) + 1e-6
            dx_ = 0.7 * (gx - ox) / dgo + 0.3 * (hx - ox) / nb
            dy_ = 0.7 * (gy - oy) / dgo + 0.3 * (hy - oy) / nb
            nd = math.hypot(dx_, dy_) + 1e-6
            danger = 55.0 * math.exp(-dgo / 22.0) + (8.0 if ox < hx else 0.0)
            jobs.append(("mark", onum, ox + r * dx_ / nd, oy + r * dy_ / nd, danger))
        # the cover costs a little extra so it is only taken when worth it
        jobs.append(("cover", None, cover[0], cover[1], -6.0))
        if self.THREAT_CSP and holder_i >= 0:
            threat = self.receiver_threats(s, hx, hy, ti, holder_i)
            jobs = [(t, o, x_, y_, 0.5 * v + 0.35 * threat.get(o, 0.0)) if t == "mark"
                    else (t, o, x_, y_, v) for t, o, x_, y_, v in jobs]

        defenders = [u for u in sorted(s.fnum) if u != blocker]
        pos = {u: s.fP[i] for i, u in enumerate(s.fnum)}

        def cost(u, job):
            px, py = pos[u]
            d = math.hypot(job[2] - px, job[3] - py)
            return (0.8 * d if job[0] == "cover" else d) - job[4]

        sol = self.csp_assign(defenders, jobs, cost)
        job = jobs[sol[s.unum]] if s.unum in sol else jobs[-1]
        tag, jid, tx, ty, _ = job
        if self.MINIMAX_DEF:
            tx, ty = self.minimax_spot(s, tx, ty, hx, hy, ti, holder_i)
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, "high" if d > 3.0 else "normal")
        note(agent, tag, q=(round(tx, 1), round(ty, 1)), o=jid)
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=0.7)
        Action.turn_to_ball(agent)
        return True

    # ----------------------------------------------------------------
    # anytime positioning searches (they soak up the per-decision budget)

    def _grid(self, cx, cy, step, xmax, half=2):
        m = self.model
        g = np.arange(-half, half + 1) * step
        X = (cx + g[:, None] + 0.0 * g[None, :]).ravel()
        Y = (cy + 0.0 * g[:, None] + g[None, :]).ravel()
        X = np.clip(X, -m.hl + 2.0, min(m.hl - 2.0, xmax))
        Y = np.clip(Y, -m.hw + 1.5, m.hw - 1.5)
        return np.stack([X, Y], axis=1)

    def pass_spot_values(self, s, hx, hy, delay, spots):
        """Value for us of the holder passing to me standing at each spot:
        a fast and a measured pass are rolled out against the opponents and
        the better chance node is kept."""
        m = self.model
        S = len(spots)
        dx, dy = spots[:, 0] - hx, spots[:, 1] - hy
        L = np.hypot(dx, dy) + 1e-9
        rows, owner = [], []
        for k in range(S):
            nmin = m.min_steps(float(L[k]), m.bmax)
            if nmin >= NEVER or L[k] < 2.5:
                continue
            for n in {max(nmin, 2), max(nmin, int(math.ceil(L[k] / 1.6)))}:
                sp_ = min(m.first_speed(float(L[k]), n), m.bmax)
                rows.append((sp_ * dx[k] / L[k], sp_ * dy[k] / L[k]))
                owner.append(k)
        out = np.full(S, -60.0)
        if not rows:
            return out
        V0 = np.array(rows)
        owner = np.array(owner)
        C = len(rows)
        K = np.full(C, int(delay), dtype=np.int64)
        BP = np.array([hx, hy])[None, None, :] + V0[:, None, :] * m.G[None, :, None]
        SU = np.array([hx, hy])[None, :] - spots
        SU = SU / (np.hypot(SU[:, 0], SU[:, 1])[:, None] + 1e-9)
        tm_all = m.first_reach(BP, K, spots, np.zeros_like(spots), SU,
                               np.full(S, m.kick_area - 0.15), np.zeros(S), 0.9)
        t_m = tm_all[np.arange(C), owner]
        x = BP[:, :, 0]; y = BP[:, :, 1]
        if len(s.oP):
            in_pa = (x > m.pa_x) & (np.abs(y) < m.pa_hw)
            octrl = np.where(s.oG[None, :, None] & in_pa[:, None, :],
                             m.catch_area + 0.15, m.kick_area + 0.35)
            t_o = m.first_reach(BP, K, s.oP, s.oV, s.oU, octrl,
                                np.zeros(len(s.oP)), 0.6).min(axis=1)
        else:
            t_o = np.full(C, NEVER)
        idx = np.arange(C)
        Rx, Ry = x[idx, np.clip(t_m, 1, T_MAX) - 1], y[idx, np.clip(t_m, 1, T_MAX) - 1]
        fi = np.clip(t_o, 1, T_MAX) - 1
        p = 1.0 / (1.0 + np.exp(-((t_o - t_m) - 1.0) * 1.3))
        p = np.where(t_m >= NEVER, 0.0, p)
        v = p * self.phi(Rx, Ry) + (1.0 - p) * self.loss(x[idx, fi], y[idx, fi])
        np.maximum.at(out, owner, v)
        return out

    def sim_support_spot(self, s, qx, qy, ax, ay, hx, hy, delay, offx, chosen):
        """Anytime search for my support spot.
        1. coarse grid, each spot scored by simulating the holder's pass to it
        2. hill climbing round the best with a shrinking step
        3. second ply for the best spots in turn: what I could do with the
           ball there (my own shots / passes / dribbles against opponents
           advanced to the reception), while the clock lasts."""
        me = np.array([s.mx, s.my])

        def extra(sp):
            v = np.zeros(len(sp))
            for c_x, c_y in chosen:
                v -= 3.0 * np.clip(9.0 - np.hypot(sp[:, 0] - c_x, sp[:, 1] - c_y), 0.0, 9.0)
            v -= 0.3 * np.hypot(sp[:, 0] - ax, sp[:, 1] - ay)
            v -= 0.15 * np.hypot(sp[:, 0] - me[0], sp[:, 1] - me[1])
            return v

        seen_s, seen_v = [], []
        spots = self._grid(qx, qy, 2.5, offx - 0.8, half=3)
        step = 1.25
        while self.clk is not None and self.clk.left() > 3.0:
            v = self.pass_spot_values(s, hx, hy, delay, spots) + extra(spots)
            seen_s.append(spots); seen_v.append(v)
            if step < 0.3:
                break
            k = int(np.argmax(v))
            spots = self._grid(float(spots[k, 0]), float(spots[k, 1]), step, offx - 0.8)
            step *= 0.5
        if not seen_s:
            return qx, qy
        S = np.vstack(seen_s); V = np.concatenate(seen_v)
        best = int(np.argmax(V))
        # second ply over the best distinct spots
        A_P = np.vstack([s.fP, s.gP]) if len(s.gP) else s.fP
        A_G = np.zeros(len(A_P), dtype=bool); A_G[len(s.fP):] = True
        bonus = {}
        for k in np.argsort(-V)[:40]:
            if self.clk.left() < 4.0:
                break
            rx, ry = float(S[k, 0]), float(S[k, 1])
            if any(abs(rx - a) < 0.2 and abs(ry - b) < 0.2 for a, b in bonus):
                continue
            t = max(2, int(math.hypot(rx - hx, ry - hy) / 1.8) + delay)
            P2 = A_P.copy(); P2[0] = (rx, ry)
            D2 = self.predict_opps(s, rx, ry, t)
            DU = np.array([[rx, ry]]) - D2
            DU = DU / (np.hypot(DU[:, 0], DU[:, 1])[:, None] + 1e-9)
            res = self.options(rx, ry, 1, (P2, np.zeros_like(P2), np.tile([[1.0, 0.0]], (len(P2), 1)), A_G),
                               0, (D2, np.zeros_like(D2), DU, s.oG), dense=2)
            if res is None:
                continue
            nxt = float(res[0].max())
            ph = float(self.phi(rx, ry))
            bonus[(rx, ry)] = 0.5 * max(0.0, 0.95 * nxt - ph)
            V[k] += bonus[(rx, ry)]
        best = int(np.argmax(V))
        return float(S[best, 0]), float(S[best, 1])

    THREAT_TOP = 16

    def receiver_threats(self, s, hx, hy, t_hold, owner):
        """How dangerous each opponent is as a pass receiver right now: the
        value to them of the holder's best pass to him, backed up one more
        ply with that receiver's own best option (expectimax in their
        attacking frame). A FIXED amount of work, so that every one of our
        players computes identical numbers and the marking CSP agrees.
        -> {opponent unum: value}"""
        if agent_budget_low(self):
            return {}
        f = np.array([-1.0, 1.0])
        A = (s.oP * f, s.oV * f, s.oU * f, s.oG)
        DP = np.vstack([s.fP, s.gP]) if len(s.gP) else s.fP
        DV = np.vstack([s.fV, np.zeros_like(s.gP)]) if len(s.gP) else s.fV
        DU = np.vstack([s.fU, np.tile([[-1.0, 0.0]], (len(s.gP), 1))]) if len(s.gP) else s.fU
        DG = np.zeros(len(DP), dtype=bool)
        DG[len(s.fP):] = True
        D = (DP * f, DV * f, DU * f, DG)
        res = self.options(-hx, hy, max(0, int(t_hold)), A, owner, D, dense=2)
        if res is None:
            return {}
        val, kind, _, t_a, p, Rx, Ry, who = res
        val = val[:, 0]; p = p[:, 0]
        out = {}
        passes = [c for c in np.argsort(-val) if kind[c] == 1 and t_a[c] < NEVER]
        for c in passes:
            o = s.onum[int(who[c])]
            out[o] = max(out.get(o, 0.0), float(val[c]))
        hl = self.model.hl
        for c in passes[:self.THREAT_TOP]:
            if p[c] < 0.15:
                continue
            j = int(who[c])
            rx, ry, t = float(Rx[c]), float(Ry[c]), int(t_a[c])
            A2 = A[0].copy(); A2[j] = (rx, ry)
            D2 = self.predict_players(D[0], DG, rx, ry, t, hl)
            DU2 = np.array([[rx, ry]]) - D2
            DU2 = DU2 / (np.hypot(DU2[:, 0], DU2[:, 1])[:, None] + 1e-9)
            nxt = self.options(rx, ry, 1, (A2, np.zeros_like(A2), A[2], A[3]), j,
                               (D2, np.zeros_like(D2), DU2, DG), dense=2)
            if nxt is None:
                continue
            ph = float(self.phi(rx, ry))
            v2 = float(val[c]) + float(p[c]) * (max(ph, 0.95 * float(nxt[0].max())) - ph)
            o = s.onum[j]
            out[o] = max(out.get(o, 0.0), v2)
        return out

    def minimax_spot(self, s, jx, jy, hx, hy, t_hold, owner):
        """Minimax positioning for a defender: among spots round the job the
        CSP gave me, take the one that minimises the opponent holder's best
        option (their max over shots, passes and dribbles, evaluated in their
        attacking frame with me standing at the spot), plus travel and
        leave-your-job costs. Coarse-to-fine while the clock lasts."""
        if owner < 0 or self.clk is None or self.clk.left() < 3.0:
            return jx, jy
        f = np.array([-1.0, 1.0])                       # mirror into their frame
        A = (s.oP * f, s.oV * f, s.oU * f, s.oG)
        DP = np.vstack([s.fP[1:], s.gP]) if len(s.gP) else s.fP[1:]
        DV = np.vstack([s.fV[1:], np.zeros_like(s.gP)]) if len(s.gP) else s.fV[1:]
        DU = np.vstack([s.fU[1:], np.tile([[-1.0, 0.0]], (len(s.gP), 1))]) if len(s.gP) else s.fU[1:]
        DG = np.zeros(len(DP), dtype=bool)
        DG[len(s.fP) - 1:] = True
        D = (DP * f, DV * f, DU * f, DG)
        best, best_c = (jx, jy), 1e9
        seen = []
        cx, cy, step, half = jx, jy, 2.0, 4
        while self.clk.left() > 3.0 and step > 0.2:
            spots = self._grid(cx, cy, step, self.model.hl, half=half)
            half = 2
            res = self.options(-hx, hy, max(0, int(t_hold)), A, owner, D,
                               spots=spots * f, dense=2)
            if res is None:
                break
            c = res[0].max(axis=0)
            c += 0.5 * np.hypot(spots[:, 0] - s.mx, spots[:, 1] - s.my)
            c += 1.0 * np.hypot(spots[:, 0] - jx, spots[:, 1] - jy)
            k = int(np.argmin(c))
            if c[k] < best_c:
                best_c, best = float(c[k]), (float(spots[k, 0]), float(spots[k, 1]))
            seen.append((spots, c))
            cx, cy = best
            step *= 0.5
        if not seen:
            return best
        # second ply: for the best spots, back up the opponent's most
        # dangerous passes with what their receiver could do next (with our
        # defenders, me at the spot, closing on the reception point)
        S = np.vstack([a for a, _ in seen]); Cc = np.concatenate([b for _, b in seen])
        hl = self.model.hl
        tried = []
        for k in np.argsort(Cc):
            if self.clk.left() < 6.0 or len(tried) >= 12:
                break
            sx, sy = float(S[k, 0]), float(S[k, 1])
            if any(abs(sx - a) < 0.3 and abs(sy - b) < 0.3 for a, b, _ in tried):
                continue
            Dk = (np.vstack([DP, [[sx, sy]]]) * f, np.vstack([DV, [[0.0, 0.0]]]) * f,
                  np.vstack([DU, [[-1.0, 0.0]]]) * f, np.append(DG, False))
            res = self.options(-hx, hy, max(0, int(t_hold)), A, owner, Dk, dense=2)
            if res is None:
                break
            val, kind, _, t_a, p, Rx, Ry, who = res
            val = val[:, 0]; p = p[:, 0]
            worst = float(val.max())
            for c in np.argsort(-val)[:3]:
                if kind[c] != 1 or p[c] < 0.15 or t_a[c] >= NEVER or self.clk.left() < 4.0:
                    continue
                rx, ry, t = float(Rx[c]), float(Ry[c]), int(t_a[c])
                A2 = A[0].copy(); A2[who[c]] = (rx, ry)
                D2 = self.predict_players(Dk[0], Dk[3], rx, ry, t, hl)
                DU2 = np.array([[rx, ry]]) - D2
                DU2 = DU2 / (np.hypot(DU2[:, 0], DU2[:, 1])[:, None] + 1e-9)
                nxt = self.options(rx, ry, 1, (A2, np.zeros_like(A2), A[2], A[3]), int(who[c]),
                                   (D2, np.zeros_like(D2), DU2, Dk[3]))
                if nxt is None:
                    continue
                ph = float(self.phi(rx, ry))
                worst = max(worst, float(val[c]) + float(p[c]) * (max(ph, 0.95 * float(nxt[0].max())) - ph))
            cost = worst + 0.5 * math.hypot(sx - s.mx, sy - s.my) + 1.0 * math.hypot(sx - jx, sy - jy)
            tried.append((sx, sy, cost))
        if tried:
            sx, sy, _ = min(tried, key=lambda t: t[2])
            return sx, sy
        return best
