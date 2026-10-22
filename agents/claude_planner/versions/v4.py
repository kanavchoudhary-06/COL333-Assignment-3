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

import numpy as np
from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


T_MAX = 40              # rollout horizon in cycles
NEVER = 10000
GOAL_VALUE = 170.0      # value of a goal on the evaluation scale
LOSS_WEIGHT = 1.0       # weight of the opponent's value when we lose the ball


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
        ok = need <= self.dash[avail]
        hit = ok.any(axis=2)
        return np.where(hit, ok.argmax(axis=2) + 1, NEVER)


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
        self.kickable_mate = wm.kickable_teammate_existance
        self.kickable_opp = wm.kickable_opponent_existance


class Agent1(StarterAgent):

    def __init__(self):
        super().__init__()
        self.model = None
        self.snap = None
        self.last_plan = None        # (cycle, kind, qx, qy)
        self.offside_flag_until = -1

    # ----------------------------------------------------------------
    # shared helpers

    def _prepare(self, agent):
        if self.model is None:
            self.model = Model(agent.server_params)
        if self.snap is None or self.snap.cycle != agent.wm.cycle:
            self.snap = Snapshot(agent, self.model)
            self._ball_cache = None
        return self.snap

    def phi(self, x, y):
        """Heuristic value of US holding the ball at (x, y). Works on arrays."""
        m = self.model
        dx = m.hl - x
        a1 = np.arctan2(y - m.goal_hw, dx)
        a2 = np.arctan2(y + m.goal_hw, dx)
        ang = np.abs(a2 - a1)
        dist = np.sqrt(dx * dx + y * y)
        shot = np.clip(ang / 0.5, 0.0, 1.0) * np.exp(-np.maximum(dist - 8.0, 0.0) / 14.0)
        return 0.5 * (x + m.hl) + 50.0 * shot

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

    def plan_with_ball(self, agent, s):
        m = self.model
        wm = agent.wm
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
            if sp_ < 0.3:
                return
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
            for r in (3.5, 7.0, 11.0):
                for da in (0.0, -0.8, 0.8, -1.57, 1.57):
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
        for k in range(16):
            a = k * math.pi / 8.0
            ca, sa = math.cos(a), math.sin(a)
            back = ca < -0.3
            for L in ((2.5, 4.5) if back else (2.5, 5.0, 9.0, 14.0)):
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
            _, lk, lqx, lqy, ln, lextra = lp
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
            cross = x > hl
            t_goal = np.where(cross.any(axis=1), cross.argmax(axis=1) + 1, NEVER)
            gi = np.clip(t_goal, 1, T_MAX) - 1
            # y where it crosses the line (linear between samples)
            y_at = y[idx, gi]
            in_mouth = np.abs(y_at) < m.goal_hw - 0.2
            gm = t_o.astype(float) - t_goal
            pg = 1.0 / (1.0 + np.exp(-(gm - 0.3) * 0.9))
            # aiming error grows with distance; keep away from the posts
            err = 0.03 * gdist + 0.3
            post = (m.goal_hw - np.abs(y_at)) / err
            pg = pg * np.clip(post, 0.0, 1.0)
            pg = np.where(in_mouth & (t_goal < NEVER), pg, 0.0)
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
        self.last_plan = (s.cycle, kind, qx, qy, n, extra)
        tgt = Vector2D(qx, qy)
        note(agent, kind, to=extra, q=(round(qx, 1), round(qy, 1)), n=n, K=Kb,
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

    def shot_prob(self, rx, ry, oP, delay, speed=2.7, n=7):
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
        in_pa = (BP[:, :, 0] > m.pa_x) & (np.abs(BP[:, :, 1]) < m.pa_hw)
        isg = np.zeros(len(oP), dtype=bool)
        isg[:len(self.snap.oG)] = self.snap.oG
        ctrl = np.where(isg[None, :, None] & in_pa[:, None, :],
                        m.catch_area + 0.15, m.kick_area + 0.35)
        to = m.first_reach(BP, K, oP, np.zeros_like(oP), U, ctrl,
                           np.zeros(len(oP)), 0.6).min(axis=1)
        cross = BP[:, :, 0] > m.hl
        t_goal = np.where(cross.any(axis=1), cross.argmax(axis=1) + 1, NEVER)
        gm = to.astype(float) - (t_goal + delay)
        pg = 1.0 / (1.0 + np.exp(-(gm - 0.3) * 0.9))
        err = 0.03 * dist + 0.3
        pg = pg * np.clip((m.goal_hw - np.abs(ty)) / err, 0.0, 1.0)
        pg = np.where(t_goal < NEVER, pg, 0.0)
        return float(pg.max())

    def refine(self, agent, s, cands, score, p, BP, t_m, who, best):
        """Beam search, second ply: expand the top candidates with the
        receiver's best follow-up (a shot from the reception point against
        the opponents' predicted positions), anytime under the CPU budget."""
        if agent.budget.remaining_ms < 3000 or agent.budget.decision_remaining_ms < 300:
            return best
        m = self.model
        order = np.argsort(-score)
        new = score.copy()
        done = 0
        for c in order:
            if done >= self.BEAM:
                break
            kind = cands[c][0]
            if kind.startswith("shoot"):
                continue
            tm = int(t_m[c])
            if tm >= NEVER or p[c] < 0.2:
                continue
            ti = min(tm, T_MAX) - 1
            rx, ry = float(BP[c, ti, 0]), float(BP[c, ti, 1])
            if math.hypot(m.hl - rx, ry) > 28.0:
                continue
            done += 1
            oP = self.predict_opps(s, rx, ry, tm)
            pg = self.shot_prob(rx, ry, oP, 1)
            follow = pg * GOAL_VALUE * 0.9 + (1.0 - pg) * float(self.loss(m.hl - 2.0, ry))
            v_succ = float(self.phi(rx, ry))
            if follow > v_succ:
                new[c] = score[c] + p[c] * (follow - v_succ)
        b = int(np.argmax(new))
        score[:] = new
        return b

    # ----------------------------------------------------------------
    # OFF THE BALL

    def off_ball(self, agent) -> None:
        s = self._prepare(agent)
        try:
            if self.plan_without_ball(agent, s):
                return
        except Exception as e:
            note(agent, "offball_error", err=str(e)[:80])
        Action.go_to_point(agent, Tools.home_position(agent))

    def plan_without_ball(self, agent, s):
        m = self.model
        wm = agent.wm
        f_reach, o_reach, ball_path = self.ball_reach(s)
        t_me = int(f_reach[0])
        t_us = int(f_reach.min())
        t_opp = int(o_reach.min()) if len(o_reach) else NEVER
        order = sorted(range(len(s.fnum)), key=lambda i: (f_reach[i], s.fnum[i]))
        chaser = s.fnum[order[0]]
        rank = order.index(0)

        # offside bookkeeping: if a mate is about to play the ball and I
        # stand offside, stay out of it until the ball changes hands
        if s.kickable_mate and s.mx > max(s.offside_x, s.bx) + 0.1:
            self.offside_flag_until = s.cycle + 25
        if s.kickable_opp:
            self.offside_flag_until = -1
        offside_now = s.cycle <= self.offside_flag_until

        # 1) tackle
        if self.try_tackle(agent, s, t_me, t_opp):
            return True

        our_ball = s.kickable_mate or (not s.kickable_opp and t_us < t_opp)

        # 2) chase a ball we can win (or contest)
        if not s.kickable_mate and not offside_now:
            # the last defender does not dive into a 50/50 in front of him
            last_man = s.mx <= float(s.fP[:, 0].min()) + 1e-6 and s.bx > s.mx + 2.0
            slack = 0 if last_man else 1
            if chaser == s.unum and (our_ball or t_me + (1 - slack) <= t_opp + slack):
                note(agent, "chase", t=t_me, opp=t_opp)
                Action.intercept(agent)
                return True

        if our_ball:
            return self.support(agent, s, f_reach, ball_path, t_us)
        return self.defend(agent, s, f_reach, o_reach, ball_path, t_opp)

    def try_tackle(self, agent, s, t_me, t_opp):
        wm = agent.wm
        prob = wm.self.tackle_probability
        if prob <= 0.0:
            return False
        if not (s.kickable_opp or (t_opp <= 1 and t_opp < t_me)):
            return False
        m = self.model
        dgoal = math.hypot(s.bx + m.hl, s.by)
        thr = 0.8
        if dgoal < 30.0:
            thr = 0.65
        if dgoal < 18.0:
            thr = 0.5
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

    # --- attack support --------------------------------------------------

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
        ax = min(ax, offx - 1.0)
        return ax, ay

    def support(self, agent, s, f_reach, ball_path, t_us):
        m = self.model
        # where will we hold the ball
        if s.kickable_mate:
            hx, hy = s.bx, s.by
        else:
            ti = int(_clip(t_us, 1, T_MAX)) - 1
            hx, hy = float(ball_path[ti, 0]), float(ball_path[ti, 1])
        holder = s.fnum[int(np.argmin(f_reach))]
        offx = max(s.offside_x, hx)
        idx_of = {u: i for i, u in enumerate(s.fnum)}

        # roles: by uniform number; if the back (#2) holds the ball the
        # deepest supporter takes over his job
        role = {u: u for u in s.fnum if u != holder}
        if holder == 2 and role:
            sub = min(role, key=lambda u: (s.fP[idx_of[u]][0], u))
            role[sub] = 2
        # the most advanced opponent outfielder (their next counter-attack)
        fwd = None
        for j in range(len(s.onum)):
            if not s.oG[j] and (fwd is None or s.oP[j, 0] < fwd[0]):
                fwd = (float(s.oP[j, 0]), float(s.oP[j, 1]))

        # deterministic greedy assignment in unum order
        chosen = []
        my_target = None
        for u in sorted(role, key=lambda u: (role[u] != 2, u)):
            i = idx_of[u]
            px, py = s.fP[i]
            ax, ay = self.anchor(role[u], hx, hy, offx)
            if role[u] == 2 and fwd is not None:
                # stay goal-side of their most advanced player
                ax = max(min(ax, fwd[0] - 4.0), -m.hl + 8.0)
                ay = 0.5 * ay + 0.5 * fwd[1]
            q = self.best_spot(s, ax, ay, px, py, hx, hy, offx, chosen,
                               tight=role[u] == 2)
            chosen.append(q)
            if u == s.unum:
                my_target = q
                break
        if my_target is None:
            return False
        tx, ty = my_target
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, urgent=d > (4.0 if role[s.unum] == 2 else 8.0))
        note(agent, "support", q=(round(tx, 1), round(ty, 1)))
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=1.0)
        Action.turn_to_ball(agent)
        return True

    def best_spot(self, s, ax, ay, px, py, hx, hy, offx, chosen, tight=False):
        m = self.model
        # candidate grid around the anchor
        g = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        QX = (ax + g[:, None] + 0 * g[None, :]).ravel()
        QY = (ay + 0 * g[:, None] + g[None, :]).ravel()
        QX = np.clip(QX, -m.hl + 3.0, min(m.hl - 3.0, offx - 0.8))
        QY = np.clip(QY, -m.hw + 2.0, m.hw - 2.0)
        val = 0.5 * self.phi(QX, QY)
        if len(s.oP):
            ox = s.oP[:, 0][None, :]; oy = s.oP[:, 1][None, :]
            # lane from holder to Q: distance of each opp to the segment
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
        val -= 3.0 * np.clip(7.0 - dh, 0.0, 7.0)          # not on top of the holder
        val -= 1.0 * np.clip(dh - 30.0, 0.0, 20.0)        # not too far for a pass
        for cx, cy in chosen:
            val -= 3.0 * np.clip(9.0 - np.hypot(QX - cx, QY - cy), 0.0, 9.0)
        val -= (2.0 if tight else 0.35) * np.hypot(QX - ax, QY - ay)
        val -= 0.15 * np.hypot(QX - px, QY - py)
        k = int(np.argmax(val))
        return float(QX[k]), float(QY[k])

    # --- defence -----------------------------------------------------------

    DRIBBLE_SPEED = 0.8

    def block_plan(self, s, hx, hy, t0):
        """Where the ball owner will dribble (straight at our goal), and the
        earliest point on that path each of our field players can get to
        first. -> (blocker unum, (x, y), path array, path index)"""
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

    def defend(self, agent, s, f_reach, o_reach, ball_path, t_opp):
        m = self.model
        gx, gy = -m.hl, 0.0
        # where the opponent will have the ball
        ti = int(_clip(t_opp, 0, T_MAX))
        if ti == 0:
            hx, hy = s.bx, s.by
        else:
            hx, hy = float(ball_path[ti - 1, 0]), float(ball_path[ti - 1, 1])
        holder_i = int(np.argmin(o_reach)) if len(o_reach) else -1

        # blocker: whoever gets in front of the owner's dribble path first
        blocker, (Dx, Dy), k = self.block_plan(s, hx, hy, ti)
        if s.unum == blocker:
            return self.press(agent, s, hx, hy, Dx, Dy, k)
        # ---- CSP: the remaining field players share out the jobs
        defenders = [u for u in sorted(s.fnum) if u != blocker]
        idx_of = {u: i for i, u in enumerate(s.fnum)}
        jobs = []                      # (tag, opp unum, x, y, value)
        for j, onum in enumerate(s.onum):
            if j == holder_i or s.oG[j]:
                continue
            ox, oy = s.oP[j]
            ox += s.oV[j, 0] * 2.0; oy += s.oV[j, 1] * 2.0
            dgo = math.hypot(ox - gx, oy - gy)
            r = 1.0 + 0.05 * min(dgo, 40.0)
            ux, uy = gx - ox, gy - oy
            n = math.hypot(ux, uy) + 1e-6
            bx_, by_ = hx - ox, hy - oy
            nb = math.hypot(bx_, by_) + 1e-6
            dx_ = 0.7 * ux / n + 0.3 * bx_ / nb
            dy_ = 0.7 * uy / n + 0.3 * by_ / nb
            nd = math.hypot(dx_, dy_) + 1e-6
            sx, sy = ox + r * dx_ / nd, oy + r * dy_ / nd
            danger = 55.0 * math.exp(-dgo / 22.0)
            if ox < hx:
                danger += 8.0
            jobs.append(("mark", onum, sx, sy, danger))
        # covers: deeper points on the owner's path to goal; worth more the
        # closer the owner is to our goal
        dhg = math.hypot(hx - gx, hy - gy)
        hd = 30.0 + 60.0 * math.exp(-dhg / 30.0)
        for off, w in ((8, 1.0), (18, 0.6)):
            kk = int(min(k + off, T_MAX - 1))
            cx, cy = max(float(Dx[kk]), -m.hl + 4.0), float(Dy[kk])
            jobs.append(("cover", None, cx, cy, hd * w))

        assign = self.solve_jobs(s, defenders, idx_of, jobs)
        job = assign.get(s.unum)
        if job is None:
            job = jobs[-1]
        tag, onum, tx, ty, _ = job
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, urgent=d > 3.0)
        note(agent, tag, q=(round(tx, 1), round(ty, 1)), o=onum)
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=0.7)
        Action.turn_to_ball(agent)
        return True

    def solve_jobs(self, s, defenders, idx_of, jobs):
        """CSP / assignment: every defender takes exactly one job, every job
        at most one defender (all-different). Minimise travel cost minus the
        value of the jobs covered. Backtracking with MRV variable ordering,
        least-cost value ordering, forward checking and branch & bound."""
        nj = len(jobs)
        cost = {}
        for u in defenders:
            px, py = s.fP[idx_of[u]]
            for v in range(nj):
                _, _, tx, ty, val = jobs[v]
                cost[(u, v)] = math.hypot(tx - px, ty - py) - val
        best = [float("inf"), {}]
        domains = {u: list(range(nj)) for u in defenders}

        def bound(assigned, cur):
            lb = cur
            for u in defenders:
                if u not in assigned:
                    lb += min(cost[(u, v)] for v in domains[u])
            return lb

        def rec(assigned, cur):
            if len(assigned) == len(defenders):
                if cur < best[0]:
                    best[0], best[1] = cur, dict(assigned)
                return
            if bound(assigned, cur) >= best[0]:
                return
            u = min((x for x in defenders if x not in assigned),
                    key=lambda x: (len(domains[x]), x))          # MRV
            for v in sorted(domains[u], key=lambda v: cost[(u, v)]):
                pruned = []                                       # forward checking
                for w in defenders:
                    if w not in assigned and w != u and v in domains[w]:
                        domains[w].remove(v)
                        pruned.append(w)
                if all(domains[w] for w in defenders if w not in assigned and w != u):
                    assigned[u] = v
                    rec(assigned, cur + cost[(u, v)])
                    del assigned[u]
                for w in pruned:
                    domains[w].append(v)

        rec({}, 0.0)
        return {u: jobs[v] for u, v in best[1].items()}

    def press(self, agent, s, hx, hy, Dx, Dy, k):
        """Get in front of the dribble path, then go for the ball."""
        m = self.model
        gx, gy = -m.hl, 0.0
        d = math.hypot(hx - gx, hy - gy) + 1e-6
        ux, uy = (gx - hx) / d, (gy - hy) / d
        dm = math.hypot(s.bx - s.mx, s.by - s.my)
        # am I goal-side of the ball?
        goal_side = (s.mx - s.bx) * ux + (s.my - s.by) * uy > 0.5
        if dm < 2.0 or (goal_side and dm < 5.0):
            note(agent, "press_ball")
            Action.intercept(agent)
            return True
        kk = int(min(k + 1, len(Dx) - 1))
        tx, ty = float(Dx[kk]), float(Dy[kk])
        note(agent, "block", q=(round(tx, 1), round(ty, 1)), k=k)
        Action.go_to_point(agent, Vector2D(tx, ty), max_dash_power=100.0,
                           distance_threshold=0.5)
        Action.intercept(agent)
        return True

    def dash_power(self, agent, s, urgent):
        stamina_max = agent.server_params.stamina_max or 8000.0
        r = s.stamina / stamina_max
        if r < 0.32:
            return 30.0
        if urgent:
            return 100.0 if r > 0.45 else 70.0
        return 60.0 if r > 0.6 else 40.0
