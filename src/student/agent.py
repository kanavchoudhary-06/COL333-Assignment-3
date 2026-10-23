# Your agent goes here. The Agent1 class must hold your final submission.
#
# Agent1: a WrightEagle-2013-style planner rebuilt from syllabus methods only.
#
#   on_ball  : greedy best-first search (AIMA 3.5) over chains of macro actions
#              (pass / lead pass / dribble / shoot), each chain scored by its
#              expected discounted utility with chance nodes (AIMA 5.5, 17.1):
#                E = sum_j Pc_<j * g^C_<j * (1-P_j) * V_fail_j
#                    + Pc * g^C * H(s_k)
#              Root action value = max over every prefix of every chain that
#              starts with it (the WrightEagle "replace H by 0 once expanded").
#              Success probability P = logistic(time margin) where the margin
#              comes from a relaxed-problem lower bound on reach time (AIMA 3.6).
#              Anytime: stops on a per-decision CPU/wall budget.
#   off_ball : deterministic role rule computed identically by every player
#              (no communication): fastest player intercepts (choosing a
#              ranked trap point on the ball path, after HELIOS2023), next one
#              covers. With our ball the others "unmark" (CYRUS2022): predict
#              the holder's pass with the same planner, search a depth-2
#              predicted-pass tree to find who will pass to me, then move to
#              the reachable point with the best pass value. Without it they
#              mark by an OMAM-style assignment search (CYRUS) or take the
#              best point of a 7x7 candidate grid around the formation slot,
#              ranked by a linear scorer per mode (YuShan2022).
#
# Plain Python only (math, time, heapq). No numpy, no scipy.

import heapq
import math
import time

from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# ---------------------------------------------------------------- parameters
# Every number here is a starting value to be tuned offline (plan, Section 8).
class P:
    GAMMA = 0.98            # per-cycle discount
    D_MAX = 2               # search depth (macro actions per chain)
    N_MAX = 60              # max node expansions per decision
    CAP = 12                # children kept per expanded node (depth >= 2)
    CHECK_EVERY = 4         # clock check every n expansions

    # time manager (ms)
    RESERVE_MS = 7500.0     # 10% of 75 s kept back
    K_ONBALL = 4.0          # on-ball decisions may spend k x the fair share
    K_OFFBALL = 0.3
    B_MIN_MS = 2.0
    B_MAX_MS = 250.0        # far below the 1000 ms wall cap
    WALL_SAFETY_MS = 300.0  # stop if less than this is left of the 1 s cap
    MATCH_CYCLES = 1500

    # success models: P = sigmoid(K * (margin - C)), margin in cycles
    PASS_K, PASS_C = 1.0, 0.5
    DRIB_K, DRIB_C = 1.0, 1.0
    SHOT_K, SHOT_C = 1.2, 1.0
    HOLD_K, HOLD_C = 1.0, 1.5
    HOLD_DISCOUNT = 0.65    # holding wins nothing: opportunity cost
    OPP_REACT = 1.0         # cycles added to the relaxed opponent bound (turn)

    # heuristic H(s) = sigmoid(H_B + H_D * dist_goal + H_R * min(free, 6))
    H_B, H_D, H_R = -0.5, -0.10, 0.20
    # failure value V_fail = -ALPHA * sigmoid(F_B + F_D * dist_own_goal)
    ALPHA, F_B, F_D = 1.0, 0.1, -0.10
    V_FAIL_SHOT = -0.015
    CLEAR_VALUE = -0.022

    # macro action geometry
    PASS_MIN, PASS_MAX = 3.0, 40.0
    PASS_END_SPEEDS = (1.5, 0.9)      # ball speed on arrival (fast, soft)
    LEAD_DIST = (4.0, 7.0)
    DRIB_DIRS = 8
    DRIB_DISTS = (4.0, 8.0)
    DRIB_SPEED = 0.65                 # m / cycle with the ball
    SHOT_POINTS = 5
    SHOT_RANGE = 32.0
    COMMIT_DELTA = 0.05               # keep last kick unless beaten by this

    # YuShan2022-style positioning ranker, one weight row per mode:
    # (dist from slot, x, free space | cover distance, pass lane)
    GRID_W = {"aggressive":   (-0.25, 0.08, 1.0, 1.2),
              "conservative": (-0.35, 0.02, 0.8, 1.5),
              "defend":       (-0.30, -0.02, -0.6, 0.0)}


def _sig(z):
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# ------------------------------------------------------------- M1/M2 models
class Phys:
    """Constants read once from server params (with rcssserver defaults)."""

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
        # S[t] = distance factor after t cycles: (1 - d^t) / (1 - d)
        self.S = [0.0]
        d, acc, p = self.decay, 0.0, 1.0
        for _ in range(80):
            acc += p
            p *= d
            self.S.append(acc)

    def steps_for(self, dist, v0):
        """Cycles for a ball kicked at v0 to cover dist, or None."""
        S = self.S
        for t in range(1, len(S)):
            if v0 * S[t] >= dist - 1e-6:
                return t
        return None

    def speed_for(self, dist, n):
        """First speed so the ball covers dist in exactly n cycles."""
        return dist / self.S[max(1, n)]


class State:
    """Planner state: ball at (bx, by) held by `holder`; positions as tuples."""
    __slots__ = ("bx", "by", "holder", "mates", "opps", "prev")

    def __init__(self, bx, by, holder, mates, opps, prev=None):
        self.bx, self.by, self.holder = bx, by, holder
        self.mates = mates      # dict unum -> (x, y, is_goalie)
        self.opps = opps        # list of (x, y, is_goalie)
        self.prev = prev        # previous holder (cycle check)


def snapshot(agent):
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
            opps.append((o.position.x() + o.velocity.x(),
                         o.position.y() + o.velocity.y(), bool(o.is_goalie)))
    b = wm.ball.position
    return State(b.x(), b.y(), me.uniform_number, mates, opps)


# ------------------------------------------------------------- the planner
class Planner:
    def __init__(self, phys, offside_x):
        self.ph = phys
        self.offside_x = offside_x

    # ---- relaxed reach time (lower bound; plan M2)
    def _opp_t(self, ox, oy, px, py, goalie):
        ph = self.ph
        r = ph.ka
        if goalie and px > ph.pen_x and abs(py) < ph.pen_hw:
            r = ph.catch
        d = math.hypot(px - ox, py - oy) - r
        return d / ph.vmax + P.OPP_REACT if d > 0.0 else 0.0

    def _margin_path(self, st, pts, t_end):
        """min over path cycles t of (earliest opponent time - t)."""
        m = 1e9
        opp_t = self._opp_t
        for t, (px, py) in enumerate(pts, 1):
            if t > t_end:
                break
            for ox, oy, g in st.opps:
                v = opp_t(ox, oy, px, py, g) - t
                if v < m:
                    m = v
        return m

    def _ball_pts(self, x0, y0, ux, uy, v0, n):
        S = self.ph.S
        return [(x0 + ux * v0 * S[t], y0 + uy * v0 * S[t])
                for t in range(1, n + 1)]

    # ---- heuristic H and failure value (plan M6)
    def H(self, st):
        ph = self.ph
        d = math.hypot(ph.hl - st.bx, st.by)
        free = min((math.hypot(ox - st.bx, oy - st.by)
                    for ox, oy, _ in st.opps), default=10.0)
        return _sig(P.H_B + P.H_D * d + P.H_R * min(free, 6.0))

    def v_fail(self, x, y):
        d = math.hypot(x + self.ph.hl, y)
        return -P.ALPHA * _sig(P.F_B + P.F_D * d)

    # ---- successor model
    def _move_opps(self, st, nx, ny, cycles):
        step = self.ph.vmax * cycles * 0.8
        out = []
        for ox, oy, g in st.opps:
            dx, dy = nx - ox, ny - oy
            d = math.hypot(dx, dy)
            mv = min(step, max(0.0, d - 1.0))
            if d > 1e-6:
                ox, oy = ox + dx / d * mv, oy + dy / d * mv
            out.append((ox, oy, g))
        return out

    def _child(self, st, nx, ny, holder, cycles):
        mates = dict(st.mates)
        hx = mates[holder]
        mates[holder] = (nx, ny, hx[2])
        return State(nx, ny, holder, mates,
                     self._move_opps(st, nx, ny, cycles), prev=st.holder)

    def pass_options(self, st, unum, rx, ry, tx, ty, lead):
        """Pass from st's ball to (tx, ty) for teammate unum standing at
        (rx, ry). -> list of (P, cycles, child, meta), one per pace."""
        ph = self.ph
        bx, by = st.bx, st.by
        out = []
        if abs(tx) > ph.hl - 1.0 or abs(ty) > ph.hw - 1.0:
            return out
        if lead and tx > self.offside_x and tx > 0.0:
            return out
        dist = math.hypot(tx - bx, ty - by)
        if dist < P.PASS_MIN or dist > P.PASS_MAX:
            return out
        ux, uy = (tx - bx) / dist, (ty - by) / dist
        t_recv = 0.0
        if lead:
            t_recv = max(0.0, (math.hypot(tx - rx, ty - ry) - ph.ka)
                         / ph.vmax) + 1.0
        speeds = P.PASS_END_SPEEDS if not lead else P.PASS_END_SPEEDS[:1]
        for e in speeds:
            # cycles so the ball arrives at end speed ~e
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
            m = self._margin_path(st, pts, n)
            if lead and t_ctrl > n:
                # ball waits at the target until the receiver arrives
                tail = [(tx, ty)] * int(math.ceil(t_ctrl - n))
                m = min(m, self._margin_path(
                    st, pts + tail, int(math.ceil(t_ctrl))))
            p = _sig(P.PASS_K * (m - P.PASS_C))
            p *= max(0.5, 1.0 - 0.01 * max(0.0, dist - 20.0))
            c = int(math.ceil(t_ctrl))
            out.append((p, c, self._child(st, tx, ty, unum, c),
                        ("pass", tx, ty, n, unum)))
        return out

    def pass_value(self, st, unum, tx, ty):
        """Best one-step expected value of a direct pass to (tx, ty)."""
        best = -1e9
        for p, c, child, meta in self.pass_options(st, unum, tx, ty, tx, ty,
                                                   False):
            e, _ = self._step_value(1.0, 0, 0.0, p, c, child, meta, st)
            best = max(best, e)
        return best

    # ---- action generator (plan M4); yields (P, cycles, child|None, meta)
    def actions(self, st, root):
        ph = self.ph
        bx, by = st.bx, st.by
        out = []

        # shoot (terminal)
        gx = ph.hl
        dgoal = math.hypot(gx - bx, by)
        if dgoal < P.SHOT_RANGE:
            half = ph.goal_w / 2.0 - 0.7
            for i in range(P.SHOT_POINTS):
                ty = -half + 2.0 * half * i / (P.SHOT_POINTS - 1)
                dist = math.hypot(gx - bx, ty - by)
                n = ph.steps_for(dist, ph.vball)
                if n is None:
                    continue
                ux, uy = (gx - bx) / dist, (ty - by) / dist
                pts = self._ball_pts(bx, by, ux, uy, ph.vball, n)
                m = self._margin_path(st, pts, n)
                p = _sig(P.SHOT_K * (m - P.SHOT_C))
                p *= max(0.3, 1.0 - 0.012 * max(0.0, dist - 12.0))
                out.append((p, n, None, ("shoot", gx, ty, n)))

        # passes (direct + lead)
        for unum, (rx, ry, rg) in st.mates.items():
            if unum == st.holder or rg:
                continue
            if not root and unum == st.prev:
                continue                        # no immediate give-and-go
            if rx > self.offside_x and rx > 0.0:
                continue                        # receiver offside now
            targets = [(rx, ry, False)]
            ang = math.atan2(-ry, gx - rx)
            for ld in P.LEAD_DIST:
                targets.append((rx + ld * math.cos(ang),
                                ry + ld * math.sin(ang), True))
                targets.append((rx + ld, ry, True))
            for tx, ty, lead in targets:
                out.extend(self.pass_options(st, unum, rx, ry, tx, ty, lead))

        # dribble
        base = math.atan2(-by, gx - bx)
        for k in range(P.DRIB_DIRS):
            a = base + 2.0 * math.pi * k / P.DRIB_DIRS
            ca, sa = math.cos(a), math.sin(a)
            for L in P.DRIB_DISTS:
                tx, ty = bx + L * ca, by + L * sa
                if abs(tx) > ph.hl - 1.0 or abs(ty) > ph.hw - 1.0:
                    continue
                n = int(math.ceil(L / P.DRIB_SPEED))
                pts = [(bx + ca * min(P.DRIB_SPEED * t, L),
                        by + sa * min(P.DRIB_SPEED * t, L))
                       for t in range(1, n + 1)]
                m = self._margin_path(st, pts, n)
                p = _sig(P.DRIB_K * (m - P.DRIB_C))
                out.append((p, n, self._child(st, tx, ty, st.holder, n),
                            ("dribble", tx, ty, n)))

        if root:
            near = min((self._opp_t(ox, oy, bx, by, g) for ox, oy, g in st.opps),
                       default=20.0)
            p = _sig(P.HOLD_K * (near - P.HOLD_C))
            held = State(bx, by, st.holder, st.mates,
                         self._move_opps(st, bx, by, 3), prev=st.prev)
            out.append((p, 3, held, ("hold",)))
            out.append((1.0, 0, None, ("clear",)))
        return out

    # ---- value of one step from a node (plan M7)
    def _step_value(self, pc, csum, acc, p, c, child, meta, st):
        g = P.GAMMA ** csum
        kind = meta[0]
        if kind == "clear":
            return acc + pc * g * P.CLEAR_VALUE, None
        if kind == "shoot":
            gs = g * P.GAMMA ** c
            return acc + pc * gs * (p + (1.0 - p) * P.V_FAIL_SHOT), None
        if kind == "hold":
            fail = acc + pc * g * (1.0 - p) * self.v_fail(st.bx, st.by)
            return fail + (pc * g * p * P.GAMMA ** c * P.HOLD_DISCOUNT
                           * self.H(child)), None
        fx, fy = (st.bx + child.bx) * 0.5, (st.by + child.by) * 0.5
        acc2 = acc + pc * g * (1.0 - p) * self.v_fail(fx, fy)
        pc2, c2 = pc * p, csum + c
        e = acc2 + pc2 * (P.GAMMA ** c2) * self.H(child)
        return e, (pc2, c2, acc2)

    def search(self, root, deadline):
        """Greedy best-first over chains. Returns (best_meta, value, stats)."""
        best = {}            # root index -> best value
        metas = []
        heap = []
        tie = 0
        for p, c, child, meta in self.actions(root, True):
            i = len(metas)
            metas.append(meta)
            e, nxt = self._step_value(1.0, 0, 0.0, p, c, child, meta, root)
            best[i] = e
            if nxt is not None and P.D_MAX > 1:
                tie += 1
                heapq.heappush(heap, (-e, tie, 1, child, nxt, i))
        expanded = 0
        clock = time.perf_counter
        while heap and expanded < P.N_MAX:
            if expanded % P.CHECK_EVERY == 0 and clock() > deadline:
                break
            _, _, depth, st, (pc, cs, acc), i = heapq.heappop(heap)
            expanded += 1
            kids = []
            for p, c, child, meta in self.actions(st, False):
                e, nxt = self._step_value(pc, cs, acc, p, c, child, meta, st)
                if e > best[i]:
                    best[i] = e
                if nxt is not None and depth + 1 < P.D_MAX:
                    kids.append((e, child, nxt))
            if kids:
                kids.sort(key=lambda k: -k[0])
                for e, child, nxt in kids[:P.CAP]:
                    tie += 1
                    heapq.heappush(heap, (-e, tie, depth + 1, child, nxt, i))
        bi = max(best, key=lambda k: (best[k], -k))
        return metas[bi], best[bi], best, metas, expanded


# ---------------------------------------------------------------- the agent
class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        self._phys = None
        self._commit = None       # (cycle, meta)
        self._unmark_cache = None # (cycle, Vector2D)
        self._marks_cache = None  # (cycle, {unum: Vector2D})

    # ---- time manager (plan Section 7)
    def _deadline(self, agent, k):
        now = time.perf_counter()
        b = agent.budget
        rem = b.remaining_ms
        if rem == float("inf"):
            rem = 75000.0
        left = max(50, P.MATCH_CYCLES - int(agent.wm.cycle))
        share = (rem - P.RESERVE_MS) / left
        ms = min(P.B_MAX_MS, max(P.B_MIN_MS, k * share))
        ms = min(ms, b.decision_remaining_ms - P.WALL_SAFETY_MS)
        if rem < P.RESERVE_MS:
            ms = min(ms, P.B_MIN_MS)
        return now + max(0.0, ms) / 1000.0, ms

    def _ph(self, agent):
        if self._phys is None:
            self._phys = Phys(agent.server_params)
        return self._phys

    # ---- on ball
    def on_ball(self, agent) -> None:
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
        deadline, ms = self._deadline(agent, P.K_ONBALL)
        if ms <= 0.5:
            return False
        ph = self._ph(agent)
        root = snapshot(agent)
        pl = Planner(ph, wm.offside_line_x)
        meta, val, best, metas, nexp = pl.search(root, deadline)

        # commitment (plan M8): keep last cycle's kick if still close in value
        cyc = int(wm.cycle)
        if self._commit is not None and self._commit[0] == cyc - 1:
            old = self._commit[1]
            if old[0] in ("pass", "shoot"):
                for i, m in enumerate(metas):
                    if (m[0] == old[0] and abs(m[1] - old[1]) < 0.6
                            and abs(m[2] - old[2]) < 0.6):
                        if best[i] + P.COMMIT_DELTA >= val:
                            meta, val = m, best[i]
                        break
        self._commit = (cyc, meta)
        self._execute(agent, meta)
        note(agent, meta[0], v=round(val, 3), n=nexp, ms=round(ms, 1))
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
            Action.kick(agent, Vector2D(meta[1], meta[2]), meta[3],
                        accept_late=3)
        elif kind == "dribble":
            Action.dribble(agent, Vector2D(meta[1], meta[2]))
            Action.hold(agent)
        elif kind == "hold":
            Action.hold(agent)
        Action.clear(agent)              # last line: cannot fail

    # ---- off ball (plan M9 + simple slots)
    def off_ball(self, agent) -> None:
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
        rows = [(int(wm.intercept_table.self_reach_steps),
                 wm.self.uniform_number)]
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

        # opponent has the ball: tackle, press, OMAM marking, cover
        if theirs and not ours:
            if (me.tackle_probability >= 0.75
                    and ball.dist(me.position) < 2.0):
                goal = Vector2D(sp.pitch_half_length, 0.0)
                rel = (goal - me.position).th().degree() - me.body_direction
                while rel > 180.0:
                    rel -= 360.0
                while rel < -180.0:
                    rel += 360.0
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
            our_best = min(int(it.self_reach_steps),
                           int(it.first_teammate_reach_steps))
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
        ph = self._ph(agent)
        my = wm.self.uniform_number
        mates = [t for t in wm.teammates if t.uniform_number > 0]
        if not mates:
            return home
        holder = min(mates, key=lambda t: t.dist_from_ball)
        base = snapshot(agent)
        root = State(base.bx, base.by, holder.uniform_number, base.mates,
                     base.opps)
        pl = Planner(ph, wm.offside_line_x)

        def predicted(st):
            """Receivers the holder's own planner would pick, best first."""
            best = {}
            for p, c, child, meta in pl.actions(st, False):
                if meta[0] != "pass":
                    continue
                e, _ = pl._step_value(1.0, 0, 0.0, p, c, child, meta, st)
                r = meta[4]
                if r not in best or e > best[r][0]:
                    best[r] = (e, child)
            return sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))

        # best-first over predicted passes, depth 2: who will pass to me?
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
            if abs(cx) > ph.hl - 2.0 or abs(cy) > ph.hw - 2.0:
                continue
            if cx > offx - 0.5 and cx > 0.0:
                continue
            t_me = math.hypot(cx - mx, cy - myy) / ph.vmax + 1.0
            t_op = min((pl._opp_t(ox, oy, cx, cy, g)
                        for ox, oy, g in passer_state.opps), default=99.0)
            if t_op <= t_me:
                continue                  # an opponent gets there first
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
        pos = {wm.self.uniform_number: (wm.self.position.x(),
                                        wm.self.position.y())}
        for t in wm.teammates:
            if t.uniform_number > 0 and not t.is_goalie:
                pos[t.uniform_number] = (t.position.x(), t.position.y())
        presser = rank[0] if rank else None
        defenders = sorted(u for u in pos if u != presser)
        opps = [o for o in wm.opponents if o.uniform_number > 0
                and not o.is_goalie]
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
                continue                      # too far upfield to mark
            dx, dy = -hl - ox, -oy
            d = math.hypot(dx, dy) or 1.0
            mark = (ox + 1.5 * dx / d, oy + 1.5 * dy / d)   # goal side
            tasks.append((o.uniform_number, mark, dgoal < 35.0))
        if not tasks:
            self._marks_cache = (cyc, {})
            return {}
        options = {}
        for u in defenders:
            x, y = pos[u]
            ranked = sorted(tasks, key=lambda t: (math.hypot(t[1][0] - x,
                                                              t[1][1] - y), t[0]))
            options[u] = ranked[:3]
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
            search(i + 1, used, n_danger, cost + 8.0, plan)   # leave free

        search(0, set(), 0, 0.0, {})
        marks = {u: Vector2D(m[0], m[1]) for u, m in best[1].items()}
        self._marks_cache = (cyc, marks)
        return marks

    # ---- HELIOS2023-style ranked ball trapping (hand-set linear ranker)
    def _trap_point(self, agent):
        wm = agent.wm
        ph = self._ph(agent)
        v = wm.ball.velocity
        if v.r() < 0.4:
            return None
        x, y = wm.ball.position.x(), wm.ball.position.y()
        vx, vy = v.x(), v.y()
        mx, myy = wm.self.position.x(), wm.self.position.y()
        opps = [(o.position.x(), o.position.y()) for o in wm.opponents
                if o.uniform_number > 0]
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
            for ox, oy in opps:
                d = math.hypot(x - ox, y - oy) - ph.ka
                t_op = min(t_op, d / ph.vmax + P.OPP_REACT if d > 0.0 else 0.0)
            safety = t_op - t
            if safety < 2.0:
                continue
            score = 0.1 * x + 0.3 * min(safety, 6.0) - 0.15 * t
            if best is None or score > best[0]:
                best = (score, x, y, t)
        if best is None:
            return None
        if best[3] <= int(wm.intercept_table.self_reach_steps) + 1:
            return None                       # earliest point: plain intercept
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
