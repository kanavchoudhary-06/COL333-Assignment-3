# Your agent goes here. The Agent1 class must hold your final submission.
#
# Agent1 is a Python port of the public HELIOS base team (agent2d /
# helios-base, H. Akiyama, H. Shimora et al., GPLv3,
# https://github.com/helios-base/helios-base), restricted to the actions
# the interface allows, with published improvements layered on top:
#
#   ON THE BALL (helios-base, ported faithfully)
#     Bhv_StrictCheckShoot / ShootGenerator   25 goal targets, goalie and
#                                             field-player reach checks,
#                                             HELIOS course scoring
#     ActionChainGraph best-first search      first action from
#       StrictCheckPassGenerator              direct / leading / through
#       ShortDribbleGenerator                 kick-turn-dash dribbles
#       SelfPassGenerator                     long kick-and-run dribbles
#     then a shot from the resulting state, scored with
#     sample_field_evaluator (ball.x + max(0, 40 - goal dist), shoot bonus)
#     Bhv_PlannedAction::doTurnToForward
#
#   OFF THE BALL (helios-base Bhv_BasicMove + Bhv_BasicTackle)
#     tackle (V12 direction search, foul when it is likelier and safe),
#     intercept rule, strategic position with HELIOS dash-power rules,
#     plus Cyrus2D Bhv_BasicBlock (dribble-path block).
#
#   OPTIONAL LAYERS (switches on Agent1, measured before enabling)
#     Cyrus2D OMAM marking and Bhv_Unmark.
#
#   THE ALGORITHMS
#     action chain            Greedy Best-First Search, frontier ordered by
#                             a heuristic game-state evaluation, anytime;
#                             transposition table (TRANSPOSITION); UCT /
#                             MCTS as an alternative (MCTS_CHAIN)
#     pass / dribble / shot   depth-1 minimax: our MAX over kicks, the
#       safety                opponents' MIN over interceptors, with an
#                             alpha-beta bound and move ordering (ORDERED_MIN)
#     marking                 CSP: backtracking, MRV, forward checking,
#                             branch & bound (CSP_MARKING)
#     carrier's dribble       MAX node over 36 directions
#     block point             breadth-first search over time layers
#
# All players share the default player type in this league, so every
# reach estimate uses the exact type-0 dash table built from the server
# parameters, the same model HELIOS uses.

import heapq
import math
import time

from pyrusgeom.geom_2d import Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _f(v, default=0.0):
    try:
        r = float(v)
        return r if r == r else default
    except (TypeError, ValueError):
        try:
            return float(v.degree())
        except Exception:
            return default


def _xy(v):
    return v.x(), v.y()


def _norm(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _line_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l = math.hypot(dx, dy)
    if l < 1e-9:
        return math.hypot(px - ax, py - ay)
    return abs((px - ax) * dy - (py - ay) * dx) / l


class _Timeout(Exception):
    pass


# Formation roles (agent2d numbering: 2-5 defenders, 6-8 midfielders,
# 9-11 forwards). In the 5-a-side formation the given strategy maps the
# uniform numbers 2, 3, 4, 5 onto roles 2, 7, 8, 11.
_ROLE_5 = {1: 1, 2: 2, 3: 7, 4: 8, 5: 11}


def _role(unum, team_size):
    return _ROLE_5.get(unum, unum) if team_size <= 5 else unum


class _Rand:
    """Small deterministic LCG, so searches are repeatable."""

    def __init__(self, seed):
        self.s = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def uniform(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 2147483648.0

    def gauss(self):
        return (self.uniform() + self.uniform() + self.uniform()
                + self.uniform() - 2.0) * 1.732


class _P:
    """Lightweight player snapshot."""
    __slots__ = ("unum", "x", "y", "vx", "vy", "body", "goalie", "tackling",
                 "speed")

    def __init__(self, unum, x, y, vx, vy, body, goalie, tackling):
        self.unum = unum
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.body = body
        self.goalie = goalie
        self.tackling = tackling
        self.speed = math.hypot(vx, vy)


def _snap(p):
    x, y = _xy(p.position)
    try:
        vx, vy = _xy(p.velocity)
    except Exception:
        vx = vy = 0.0
    return _P(int(p.uniform_number), x, y, vx, vy,
              _f(getattr(p, "body_direction", 0.0)),
              bool(getattr(p, "is_goalie", False)),
              bool(getattr(p, "is_tackling", False)))


# --------------------------------------------------------------------------
# Physics: server parameters + the default player type
# --------------------------------------------------------------------------

class _Phys:
    def __init__(self, agent):
        sp = agent.server_params

        def g(name, d):          # parameter that must be non-zero
            v = _f(getattr(sp, name, d), d)
            return v if v else d

        def g0(name, d):         # parameter that may legitimately be 0
            v = getattr(sp, name, None)
            return d if v is None else _f(v, d)

        self.hl = g("pitch_half_length", 52.5)
        self.hw = g("pitch_half_width", 34.0)
        self.goal_half = g("goal_width", 14.02) / 2.0
        self.pen_x = g("their_penalty_area_line_x", self.hl - 16.5)
        self.our_pen_x = g0("our_penalty_area_line_x", -self.hl + 16.5)
        self.pen_hw = g("penalty_area_half_width", 20.16)
        self.decay = g("ball_decay", 0.94)
        self.bmax = g("ball_speed_max", 3.0)
        self.ball_accel_max = g("ball_accel_max", 2.7)
        self.max_power = g("max_power", 100.0)
        self.kick_power_rate = g("kick_power_rate", 0.027)
        self.max_moment = g("max_moment", 180.0)
        self.max_dash = g("max_dash_power", 100.0)
        self.min_dash = g0("min_dash_power", -100.0)
        self.pdecay = g("player_decay", 0.4)
        self.inertia_moment = g("inertia_moment", 5.0)
        self.dpr = g("dash_power_rate", 0.006)
        self.effort_max = g("effort_max", 1.0)
        self.psm = g("player_speed_max", 1.05)
        self.player_size = g("player_size", 0.3)
        self.ball_size = g("ball_size", 0.085)
        self.kickable_margin = g("kickable_margin", 0.7)
        self.kickable = g("kickable_area", self.player_size
                          + self.kickable_margin + self.ball_size)
        cl, cw = g("catch_area_l", 1.2), g("catch_area_w", 1.0)
        self.catchable = g("catchable_area",
                           math.sqrt(cl * cl + (cw * 0.5) ** 2))
        self.stamina_max = g("stamina_max", 8000.0)
        self.recover_dec_thr_value = g("recover_dec_thr", 0.3) * \
            self.stamina_max
        self.stamina_inc_max = g("stamina_inc_max", 45.0)
        self.tackle_power_rate = g("tackle_power_rate", 0.027)
        self.max_tackle_power = g("max_tackle_power", 100.0)
        self.max_back_tackle_power = g0("max_back_tackle_power", 0.0)

        self.accel_max = self.max_dash * self.dpr * self.effort_max
        self.rsm = min(self.psm, self.accel_max / (1.0 - self.pdecay))
        self.dpow = [self.decay ** i for i in range(160)]
        self.ppow = [self.pdecay ** i for i in range(160)]
        # dash distance table (rcsc PlayerType::cyclesToReachDistance)
        tab = []
        speed, dist, accel = 0.0, 0.0, self.accel_max
        for _ in range(50):
            if speed + accel > self.psm:
                accel = self.psm - speed
            speed += accel
            dist += speed
            tab.append(dist)
            speed *= self.pdecay
        self.dash_tab = tab

    # ---- player model -------------------------------------------------
    def cycles_to_reach(self, dist):
        if dist <= 0.001:
            return 0
        tab = self.dash_tab
        lo, hi = 0, len(tab)
        while lo < hi:
            mid = (lo + hi) // 2
            if tab[mid] >= dist - 0.001:
                hi = mid
            else:
                lo = mid + 1
        if lo < len(tab):
            return lo + 1
        return len(tab) + int(math.ceil((dist - tab[-1]) / self.rsm))

    def inertia(self, p, n):
        f = (1.0 - self.ppow[min(n, 159)]) / (1.0 - self.pdecay)
        return p.x + p.vx * f, p.y + p.vy * f

    def inertia_final(self, p):
        f = 1.0 / (1.0 - self.pdecay)
        return p.x + p.vx * f, p.y + p.vy * f

    def eff_turn(self, speed):
        return self.max_moment / (1.0 + self.inertia_moment * speed)

    def turn_cycle(self, body, speed, target_dist, target_angle, dist_thr,
                   back_dash):
        diff = abs(_norm(target_angle - body))
        if (back_dash and target_dist < 5.0 and diff > 90.0
                and self.min_dash < -self.max_dash + 1.0):
            diff = abs(diff - 180.0)
        margin = 180.0
        if dist_thr < target_dist:
            margin = max(15.0, math.degrees(math.asin(
                min(1.0, dist_thr / target_dist))))
        n = 0
        while diff > margin and n < 8:
            diff -= self.eff_turn(speed)
            speed *= self.pdecay
            n += 1
        return n

    # ---- ball model ---------------------------------------------------
    def ball_at(self, fx, fy, vx, vy, n):
        f = (1.0 - self.dpow[min(n, 159)]) / (1.0 - self.decay)
        return fx + vx * f, fy + vy * f

    def first_speed(self, dist, n):
        return dist * (1.0 - self.decay) / (1.0 - self.dpow[min(n, 159)])

    def move_step(self, speed, dist):
        """rcsc ServerParam::ballMoveStep (ceil of the geometric length)."""
        if speed <= 0.0:
            return -1
        rem = 1.0 - dist * (1.0 - self.decay) / speed
        if rem <= 0.0:
            return -1
        return int(math.ceil(math.log(rem) / math.log(self.decay) - 1e-10))

    def min_reach_cycle(self, px, py, fx, fy, angle):
        a = math.radians(angle)
        dx, dy = px - fx, py - fy
        rx = dx * math.cos(a) + dy * math.sin(a)
        ry = -dx * math.sin(a) + dy * math.cos(a)
        if rx < -1.0:
            return -1
        return max(1, int(abs(ry) / self.rsm))

    def max_vel(self, angle, krate, bvx, bvy):
        """rcsc KickTable::calc_max_velocity, returned as a speed."""
        acc = min(self.max_power * krate, self.ball_accel_max)
        a = math.radians(angle)
        ux, uy = math.cos(a), math.sin(a)
        uc = ux * bvx + uy * bvy
        c2 = bvx * bvx + bvy * bvy
        disc = uc * uc - c2 + acc * acc
        if disc < 0.0:
            return 0.0
        sq = math.sqrt(disc)
        t1, t2 = uc + sq, uc - sq
        if c2 < acc * acc:               # current ball vel inside circle
            return min(t1, self.bmax) if t1 > 0.0 else 0.0
        if t1 <= 0.0:
            return 0.0
        if t1 > self.bmax:
            if t2 > self.bmax:
                return 0.0
            return self.bmax
        return t1

    def in_their_pen(self, x, y):
        return x > self.pen_x and abs(y) < self.pen_hw


# --------------------------------------------------------------------------
# World snapshot
# --------------------------------------------------------------------------

class _World:
    def __init__(self, agent):
        wm = agent.wm
        self.bx, self.by = _xy(wm.ball.position)
        try:
            self.bvx, self.bvy = _xy(wm.ball.velocity)
        except Exception:
            self.bvx = self.bvy = 0.0
        s = wm.self
        self.me = _snap(s)
        self.mates = [_snap(t) for t in wm.teammates if t.uniform_number > 0]
        self.opps = [_snap(o) for o in wm.opponents if o.uniform_number > 0]
        self.offside = _f(getattr(wm, "offside_line_x", 0.0))
        self.their_def = _f(getattr(wm, "their_defense_line_x", 0.0))
        self.our_def = _f(getattr(wm, "our_defense_line_x", 0.0))
        self.kick_rate = _f(getattr(s, "kick_rate", 0.0))
        self.stamina = _f(getattr(s, "stamina", 8000.0), 8000.0)
        self.effort = _f(getattr(s, "effort", 1.0), 1.0)
        self.recovery = _f(getattr(s, "recovery", 1.0), 1.0)
        self.cycle = int(getattr(wm, "cycle", 0))


class _Act:
    __slots__ = ("kind", "tx", "ty", "unum", "steps", "flight", "n_kick",
                 "n_turn", "n_dash", "speed", "desc", "final")

    def __init__(self, kind, tx, ty, unum, steps, flight, n_kick, speed,
                 desc, final=False, n_turn=0, n_dash=0):
        self.kind = kind
        self.tx, self.ty = tx, ty
        self.unum = unum
        self.steps = steps
        self.flight = flight
        self.n_kick = n_kick
        self.n_turn = n_turn
        self.n_dash = n_dash
        self.speed = speed
        self.desc = desc
        self.final = final


# --------------------------------------------------------------------------
# HELIOS planner port
# --------------------------------------------------------------------------

class _Helios:
    def __init__(self, ph, w, deadline_cpu, deadline_wall, deep=True,
                 cross=True, gran=1, risky=False):
        self.risky = risky
        self.gran = gran
        self.deep = deep
        self.cross = cross
        self.ph = ph
        self.w = w
        self.deadline = deadline_cpu
        self.wall = deadline_wall
        self.count = 0

    def tick(self):
        self.count += 1
        if (self.count & 15) == 0 and (time.process_time() > self.deadline
                                       or time.perf_counter() > self.wall):
            raise _Timeout()

    # ================= StrictCheckPassGenerator ======================
    def passes(self):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        recvs = []
        for p in w.mates:
            if p.tackling:
                continue
            if p.x > w.offside:
                continue
            if p.goalie and p.x < ph.our_pen_x + 15.0:
                continue
            if math.hypot(p.x - fx, p.y - fy) > 40.0:
                continue
            recvs.append(p)
        recvs.sort(key=lambda p: math.hypot(p.x - ph.hl, p.y))
        self._recvs = recvs
        out = []
        try:
            for r in recvs:
                self._direct(r, out)
            for r in recvs:
                self._leading(r, out)
            for r in recvs:
                self._through(r, out)
        except _Timeout:
            pass
        out.sort(key=lambda a: math.hypot(a.tx - ph.hl, a.ty))
        return out

    def _nearest_recv(self, x, y):
        best, bd = None, 1e18
        for p in self._recvs:
            d = (p.x - x) ** 2 + (p.y - y) ** 2
            if d < bd:
                bd, best = d, p.unum
        return best

    def _recv_step(self, r, x, y, ptype_l):
        ph = self.ph
        ix, iy = ph.inertia_final(r)
        td = math.hypot(x - ix, y - iy)
        ang = math.degrees(math.atan2(y - iy, x - ix))
        n_turn = ph.turn_cycle(r.body, r.speed, td, ang, ph.kickable, False)
        dash = td
        if ptype_l:
            dash *= 1.05
            da = math.degrees(math.atan2(y - r.y, x - r.x))
            if abs(da) > 90.0 or abs(_norm(da - r.body)) > 30.0:
                n_turn += 1
        n_dash = ph.cycles_to_reach(dash)
        return n_dash if n_turn == 0 else n_turn + n_dash + 1

    def _direct(self, r, out):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        if (r.x > ph.hl - 1.5 or r.x < -ph.hl + 5.0
                or abs(r.y) > ph.hw - 1.5):
            return
        if r.x < fx + 1.0 and math.hypot(r.x + ph.hl, r.y) < 18.0:
            return
        tx, ty = ph.inertia_final(r)
        dist = math.hypot(tx - fx, ty - fy)
        if dist < ph.kickable * 2.2 or dist > 0.8 * ph.bmax / (1 - ph.decay):
            return
        max_recv = min(ph.bmax * ph.dpow[3],
                       ph.kickable + ph.accel_max * 1.8)
        ang = math.degrees(math.atan2(ty - fy, tx - fx))
        start = max(3, ph.move_step(ph.bmax, dist), 0)
        self._common(r, tx, ty, start, start + 2, ph.rsm, ph.bmax,
                     ph.rsm, max_recv, dist, ang, "D", out)

    def _leading(self, r, out):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        max_recv = min(ph.bmax * ph.dpow[4],
                       ph.kickable + ph.accel_max * 1.5)
        ix, iy = ph.inertia_final(r)
        afb = math.degrees(math.atan2(r.y - fy, r.x - fx))
        g = self.gran
        ndiv = 24 * g
        for d in range(1, 4 * g + 1):
            pmd = 1.1 * d / g
            a_step = 2 if pmd * 2.0 * math.pi / ndiv < 0.6 / g else 1
            for a in range(0, ndiv, a_step):
                self.tick()
                ang = math.radians(afb + 360.0 / ndiv * a)
                tx, ty = ix + pmd * math.cos(ang), iy + pmd * math.sin(ang)
                if (tx > ph.hl - 3.0 or tx < -ph.hl + 5.0
                        or abs(ty) > ph.hw - 3.0):
                    continue
                if tx < fx and math.hypot(tx + ph.hl, ty) < 16.0:
                    continue
                dist = math.hypot(tx - fx, ty - fy)
                if dist < 3.0 or dist > 0.8 * ph.bmax / (1 - ph.decay):
                    continue
                if self._nearest_recv(tx, ty) != r.unum:
                    break
                pen = int(math.floor(_line_dist(r.x, r.y, fx, fy, tx, ty)
                                     * 0.3))
                rstep = self._recv_step(r, tx, ty, True) + pen
                bang = math.degrees(math.atan2(ty - fy, tx - fx))
                start = max(4, ph.move_step(ph.bmax, dist), rstep)
                self._common(r, tx, ty, start, start + 3, ph.rsm, ph.bmax,
                             0.001, max_recv, dist, bang, "L", out)

    def _through(self, r, out):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        min_rx = min(min(max(10.0, fx + 10.0), w.offside - 10.0),
                     ph.pen_x - 5.0)
        if r.x < min_rx - 30.0 or r.x < 1.0:
            return
        max_recv = min(ph.bmax * ph.dpow[6],
                       ph.kickable + ph.accel_max * 1.5)
        ix, iy = ph.inertia_final(r)
        vel_ang = math.degrees(math.atan2(r.vy, r.vx))
        g = self.gran
        for a in range(0, 14 * g + 1):
            angle = -40.0 + (80.0 / (14 * g)) * a
            ca, sa = math.cos(math.radians(angle)), math.sin(
                math.radians(angle))
            md = 6.0
            while md < 30.001:
                self.tick()
                tx, ty = ix + ca * md, iy + sa * md
                md += 2.0 / g
                if tx < min_rx:
                    continue
                if tx > ph.hl - 1.5 or abs(ty) > ph.hw - 1.5:
                    break
                dist = math.hypot(tx - fx, ty - fy)
                if dist < 5.0 or dist > 0.9 * ph.bmax / (1 - ph.decay):
                    continue
                if self._nearest_recv(tx, ty) != r.unum:
                    break
                rstep = self._recv_step(r, tx, ty, False)
                bang = math.degrees(math.atan2(ty - fy, tx - fx))
                start = rstep
                if not (r.speed > 0.2 and abs(_norm(vel_ang - angle)) < 15.0):
                    start += 1
                    if ((tx > ph.hl - 5.0 or abs(ty) > ph.hw - 5.0)
                            and abs(bang) > 30.0 and start >= 10):
                        start += 1
                start = max(6, ph.move_step(ph.bmax, dist), start)
                self._common(r, tx, ty, start, start + 3, 1.4, ph.bmax,
                             0.001, max_recv, dist, bang, "T", out)

    def _kick_count(self, speed, angle):
        ph, w = self.ph, self.w
        if ph.max_vel(angle, w.kick_rate, w.bvx, w.bvy) >= speed:
            return 1
        if speed > 2.5:
            return 3
        if speed > 1.5:
            return 2
        return 1

    def _common(self, r, tx, ty, min_step, max_step, min_first, max_first,
                min_recv, max_recv, dist, angle, ptype, out):
        ph, w = self.ph, self.w
        for step in range(min_step, max_step + 1):
            first = ph.first_speed(dist, step)
            if first < min_first:
                break
            if first > max_first:
                continue
            rs = first * ph.dpow[min(step, 159)]
            if rs < min_recv:
                break
            if rs > max_recv:
                continue
            kc = self._kick_count(first, angle)
            o_step, opp = self._opps_reach(first, angle, tx, ty,
                                           step + (kc - 1) + 5, ptype)
            risk = 0
            if (self.risky and w.bx < w.offside and tx > w.offside + 3.0
                    and w.offside - r.x < 5.0):
                risk = 5        # Gliders2d v1.6 value against helios-base
            o_step += risk
            if ptype == "T":
                failed = o_step <= step
                if (tx > 30.0 and step >= 15 and (opp is None
                                                  or not opp.goalie)
                        and o_step >= step):
                    rma = math.degrees(math.atan2(ty - r.y, tx - r.x))
                    if abs(_norm(r.body - rma)) < 15.0:
                        failed = False
            else:
                failed = o_step <= step + (kc - 1)
            if failed:
                break
            final = (w.bx <= 30.0 and w.bx > w.their_def)
            out.append(_Act("pass", tx, ty, r.unum, step + kc, step, kc,
                            first, ptype, final))
            break

    def _opps_reach(self, first, angle, rx, ry, max_cycle, ptype):
        w = self.w
        a = math.radians(angle)
        vx, vy = first * math.cos(a), first * math.sin(a)
        no_bonus = (ptype == "T" and vx > 2.0
                    and (rx > w.offside or rx > 30.0))
        # MIN node of a depth-1 minimax: the opponents' best interception.
        # min(max_cycle, best) is the alpha-beta bound (an opponent only
        # matters if he beats the best so far).
        best, who = 1000, None
        if self.ORDERED_MIN:
            # move ordering: likeliest interceptors first (lower bound on
            # their reach), so the bound tightens early; ties still go to
            # the opponent listed first, as without ordering
            ph = self.ph

            def lower_bound(i):
                mc = ph.min_reach_cycle(w.opps[i].x, w.opps[i].y,
                                        w.bx, w.by, angle)
                return (mc if mc >= 0 else 1000, i)

            order = sorted(range(len(w.opps)), key=lower_bound)
            who_i = None
            for i in order:
                o = w.opps[i]
                s = self._opp_reach(o, w.bx, w.by, vx, vy, angle, rx,
                                    min(max_cycle, best), no_bonus)
                if s < best or (s == best and s < 1000 and i < who_i):
                    best, who, who_i = s, o, i
            return best, who
        for o in w.opps:
            s = self._opp_reach(o, w.bx, w.by, vx, vy, angle, rx,
                                min(max_cycle, best), no_bonus)
            if s < best:
                best, who = s, o
        return best, who

    def _opp_reach(self, o, fx, fy, vx, vy, angle, rx, max_cycle,
                   no_bonus):
        ph = self.ph
        mc = ph.min_reach_cycle(o.x, o.y, fx, fy, angle)
        if mc < 0:
            return 1000
        for cyc in range(max(1, mc), max_cycle + 1):
            bx, by = ph.ball_at(fx, fy, vx, vy, cyc)
            ctrl = ph.catchable if (o.goalie and ph.in_their_pen(bx, by)) \
                else ph.kickable
            ix, iy = ph.inertia(o, cyc)
            td = math.hypot(bx - ix, by - iy)
            dash = td
            if dash - ctrl - 0.15 < 0.001:
                return cyc
            if no_bonus:
                dash -= ctrl
            elif rx < 25.0:
                dash -= ctrl + 0.5
            else:
                dash -= ctrl + 0.2
            if dash > ph.rsm * cyc:
                continue
            n_dash = ph.cycles_to_reach(dash)
            if n_dash > cyc:
                continue
            n_turn = ph.turn_cycle(o.body, o.speed, td,
                                   math.degrees(math.atan2(by - iy,
                                                           bx - ix)),
                                   ctrl, True)
            n_step = n_dash if n_turn == 0 else n_turn + n_dash + 1
            bonus = -5 if o.tackling else 0
            if n_step - bonus <= cyc:
                return cyc
        return 1000

    # ================= CrossGenerator ==================================
    def crosses(self):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        if math.hypot(ph.hl - fx, fy) > 35.0:
            return []
        min_d = ph.kickable * 2.2
        max_d = ph.bmax * (1.0 - ph.dpow[9]) / (1.0 - ph.decay)
        gang = math.degrees(math.atan2(-fy, ph.hl - fx))
        opps = []
        for o in w.opps:
            oa = math.degrees(math.atan2(o.y - fy, o.x - fx))
            if abs(_norm(oa - gang)) > 90.0:
                continue
            if math.hypot(o.x - fx, o.y - fy) > 20.0:
                continue
            opps.append(o)
        out = []
        try:
            for r in w.mates:
                if r.x > w.offside:
                    continue
                if math.hypot(r.x - ph.hl, r.y) > 16.0:
                    continue
                d = math.hypot(r.x - fx, r.y - fy)
                if d < min_d or d > max_d:
                    continue
                self._cross(r, opps, out)
        except _Timeout:
            pass
        return out

    def _cross(self, r, opps, out):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        rx, ry = ph.inertia_final(r)
        rdist = math.hypot(rx - fx, ry - fy)
        rang = math.degrees(math.atan2(ry - fy, rx - fx))
        min_first = ph.bmax * 0.67
        best, best_w = None, -1.0
        for a in range(-2, 3):
            cang = rang + 3.0 * a
            ca, sa = math.cos(math.radians(cang)), math.sin(math.radians(cang))
            for d in range(5):
                self.tick()
                sub = 0.9 * d
                bmd = rdist - sub
                tx, ty = fx + ca * bmd, fy + sa * bmd
                if tx > ph.hl - 0.5 or abs(ty) > ph.hw - 3.0:
                    continue
                rstep = ph.cycles_to_reach(sub) + 1
                for step in range(max(2, rstep), 13):
                    first = ph.first_speed(bmd, step)
                    if first < min_first:
                        break
                    if first > ph.bmax:
                        continue
                    if first * ph.dpow[step] < ph.psm:
                        break
                    kc = self._kick_count(first, cang)
                    if not self._cross_opp_ok(opps, rdist, tx, ty, first,
                                              cang, step + kc - 1):
                        break
                    width = 180.0
                    for o in opps:
                        if math.hypot(o.x - fx, o.y - fy) > bmd + 1.0:
                            continue
                        ad = abs(_norm(math.degrees(math.atan2(o.y - fy,
                                                               o.x - fx))
                                       - cang))
                        width = min(width, ad)
                    act = _Act("pass", tx, ty, r.unum, step + kc, step, kc,
                               first, "cross")
                    out.append(act)
                    if width > best_w:
                        best_w, best = width, act
                    break
        if best is not None:
            out.append(best)

    def _cross_opp_ok(self, opps, rdist, tx, ty, first, angle, max_cycle):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        a = math.radians(angle)
        vx, vy = first * math.cos(a), first * math.sin(a)
        for o in opps:
            ctrl = ph.catchable if o.goalie else ph.kickable
            ox, oy = ph.inertia_final(o)
            if math.hypot(ox - fx, oy - fy) > rdist + 1.0:
                continue
            mc = ph.min_reach_cycle(ox, oy, fx, fy, angle)
            for cyc in range(max(1, mc), max_cycle + 1):
                bx, by = ph.ball_at(fx, fy, vx, vy, cyc)
                td = math.hypot(ox - bx, oy - by)
                if td - ctrl - 0.15 < 0.001:
                    return False
                dash = td - ctrl * 0.6 if cyc > 1 else td
                if dash > ph.rsm * cyc:
                    continue
                n_dash = ph.cycles_to_reach(dash * 1.05)
                if n_dash > cyc:
                    continue
                n_turn = ph.turn_cycle(o.body, o.speed, td,
                                       math.degrees(math.atan2(by - oy,
                                                               bx - ox)),
                                       ctrl, True)
                n_step = n_turn + n_dash + 1
                if o.tackling:
                    n_step += 5
                if n_step <= cyc:
                    return False
        return True

    # ================= ShortDribbleGenerator ===========================
    def short_dribbles(self):
        ph, w = self.ph, self.w
        me = w.me
        out = []
        try:
            nd = 16 * self.gran
            for a in range(nd):
                self.tick()
                rel = _norm(360.0 / nd * a)
                dash_angle = _norm(me.body + rel)
                if me.x < 16.0 and abs(dash_angle) > 100.0:
                    continue
                if me.x < -36.0 and abs(me.y) < 20.0 and \
                        abs(dash_angle) > 45.0:
                    continue
                n_turn = 0
                spd = me.speed * ph.pdecay
                dd = abs(rel)
                while dd > 10.0:
                    dd -= ph.eff_turn(spd)
                    if dd < 0.0:
                        dd = 0.0
                    spd *= ph.pdecay
                    n_turn += 1
                if n_turn >= 3:
                    continue
                if 360.0 / nd * a < 180.0:
                    dash_angle = _norm(dash_angle - dd)
                else:
                    dash_angle = _norm(dash_angle + dd)
                self._kick_turn_dash(dash_angle, n_turn, out)
        except _Timeout:
            pass
        out.sort(key=lambda a: math.hypot(a.tx - ph.hl, a.ty))
        return out

    def _self_cache(self, dash_angle, n_turn, n_dash, thr_extra,
                    include_kick, stop_at_edges):
        ph, w = self.ph, self.w
        me = w.me
        cache = []
        x, y = me.x + me.vx, me.y + me.vy
        vx, vy = me.vx * ph.pdecay, me.vy * ph.pdecay
        if include_kick:
            cache.append((x, y))
        for _ in range(n_turn):
            x += vx
            y += vy
            vx *= ph.pdecay
            vy *= ph.pdecay
            cache.append((x, y))
        stamina = w.stamina
        thr = ph.recover_dec_thr_value + thr_extra
        a = math.radians(dash_angle)
        ca, sa = math.cos(a), math.sin(a)
        for _ in range(n_dash):
            if stop_at_edges and stamina < thr:
                break
            power = min(ph.max_dash, max(0.0, stamina - thr))
            acc = power * ph.dpr * w.effort
            vx += ca * acc
            vy += sa * acc
            x += vx
            y += vy
            if stop_at_edges:
                if x > ph.hl - 2.5:
                    break
                if abs(y) > ph.hw - 3.0 and ((y > 0 and dash_angle > 0)
                                             or (y < 0 and dash_angle < 0)):
                    break
            vx *= ph.pdecay
            vy *= ph.pdecay
            stamina += -power + ph.stamina_inc_max * w.recovery
            cache.append((x, y))
        return cache

    def _kick_turn_dash(self, dash_angle, n_turn, out):
        ph, w = self.ph, self.w
        max_dash, min_dash = 4, 2
        cache = self._self_cache(dash_angle, n_turn, max_dash, 300.0,
                                 True, False)
        a = math.radians(dash_angle)
        trl = ph.player_size + ph.kickable_margin * 0.2 + ph.ball_size
        trx, try_ = trl * math.cos(a), trl * math.sin(a)
        for n_dash in range(max_dash, min_dash - 1, -1):
            if n_turn + n_dash >= len(cache):
                continue
            px, py = cache[n_turn + n_dash]
            tx, ty = px + trx, py + try_
            if abs(tx) > ph.hl - 1.0 or abs(ty) > ph.hw - 1.0:
                continue
            n = 1 + n_turn + n_dash
            term = (1.0 - ph.dpow[n]) / (1.0 - ph.decay)
            fvx, fvy = (tx - w.bx) / term, (ty - w.by) / term
            kax, kay = fvx - w.bvx, fvy - w.bvy
            ka = math.hypot(kax, kay)
            kp = ka / w.kick_rate if w.kick_rate > 0 else 1e9
            if (kp > ph.max_power or ka > ph.ball_accel_max
                    or math.hypot(fvx, fvy) > ph.bmax):
                continue
            if math.hypot(w.bx + fvx - cache[0][0], w.by + fvy - cache[0][1]) \
                    < ph.player_size + ph.ball_size + 0.1:
                continue
            if self._dribble_opp_ok(tx, ty, n):
                out.append(_Act("dribble", tx, ty, w.me.unum, n, n, 1,
                                math.hypot(fvx, fvy), "shortDribble",
                                n_turn=n_turn, n_dash=n_dash))

    def _dribble_opp_ok(self, tx, ty, step):
        ph, w = self.ph, self.w
        me = w.me
        bang = math.atan2(ty - w.by, tx - w.bx)
        cb, sb = math.cos(bang), math.sin(bang)
        for o in sorted(w.opps, key=lambda o: (o.x - me.x) ** 2
                        + (o.y - me.y) ** 2):
            if math.hypot(o.x - me.x, o.y - me.y) > 20.0:
                break
            ctrl = ph.catchable if (o.goalie and ph.in_their_pen(tx, ty)) \
                else ph.kickable
            ox, oy = ph.inertia(o, step)
            relx = (o.x - w.bx) * cb + (o.y - w.by) * sb
            if relx < -4.0:
                continue
            td = math.hypot(ox - tx, oy - ty)
            if td - ctrl < 0.001:
                return False
            dash = td - ctrl * 0.5 - 0.2
            n_dash = ph.cycles_to_reach(dash)
            n_turn = ph.turn_cycle(o.body, o.speed, td,
                                   math.degrees(math.atan2(ty - oy, tx - ox)),
                                   ctrl, True)
            n_step = n_dash if n_turn == 0 else n_turn + n_dash + 1
            bonus = 0
            if tx < 30.0:
                bonus += 1
            if tx < 0.0:
                bonus += 1
            if o.tackling:
                bonus = -5
            if n_step - bonus <= step:
                return False
        return True

    # ================= SelfPassGenerator ===============================
    def self_passes(self):
        ph, w = self.ph, self.w
        me = w.me
        out = []
        bx = w.bx
        min_dash = 5
        max_dash = (6 if bx < -20.0 else 7 if bx < 0.0 else 13 if bx < 10.0
                    else 15 if bx < 20.0 else 20)
        met = ph.eff_turn(me.speed * ph.pdecay)
        try:
            for a in range(60):
                self.tick()
                add = _norm(6.0 * a)
                n_turn = 0
                if a != 0:
                    if abs(add) > met:
                        continue
                    n_turn = 1
                dash_angle = _norm(me.body + add)
                if bx < ph.pen_x + 5.0 and abs(dash_angle) > 85.0:
                    continue
                cache = self._self_cache(dash_angle, n_turn, max_dash, 350.0,
                                         False, True)
                n_dash = len(cache) - n_turn
                if n_dash < min_dash:
                    continue
                count, dec = 0, 2
                while n_dash >= min_dash:
                    if n_dash <= 10:
                        dec = 1
                    rx, ry = cache[n_turn + n_dash - 1]
                    nd = n_dash
                    n_dash -= dec
                    if math.hypot(rx + ph.hl, ry) < 18.0:
                        continue
                    n = 1 + n_turn + nd
                    if not self._sp_can_kick(n, rx, ry):
                        continue
                    if not self._sp_opp_ok(n, rx, ry):
                        continue
                    sp = ph.first_speed(math.hypot(rx - w.bx, ry - w.by), n)
                    out.append(_Act("dribble", rx, ry, me.unum, n, n, 1, sp,
                                    "selfPass", n_turn=n_turn, n_dash=nd))
                    count += 1
                    if count >= 10:
                        break
        except _Timeout:
            pass
        out.sort(key=lambda a: math.hypot(a.tx - ph.hl, a.ty))
        return out

    def _sp_can_kick(self, n, rx, ry):
        ph, w = self.ph, self.w
        me = w.me
        dist = math.hypot(rx - w.bx, ry - w.by)
        first = ph.first_speed(dist, n)
        ang = math.degrees(math.atan2(ry - w.by, rx - w.bx))
        if ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy) < first:
            return False
        a = math.radians(ang)
        bnx, bny = w.bx + math.cos(a) * first, w.by + math.sin(a) * first
        if math.hypot(me.x + me.vx - bnx, me.y + me.vy - bny) < \
                ph.player_size + ph.ball_size + 0.1:
            return False
        for o in w.opps:
            ox, oy = o.x + o.vx, o.y + o.vy
            ctrl = ph.catchable if (o.goalie and ph.in_their_pen(bnx, bny)) \
                else ph.kickable
            if math.hypot(bnx - ox, bny - oy) < ctrl + 0.1:
                return False
            ob = math.radians(o.body)
            ox += math.cos(ob) * ph.accel_max
            oy += math.sin(ob) * ph.accel_max
            if math.hypot(bnx - ox, bny - oy) < ctrl:
                return False
        return True

    def _sp_opp_ok(self, n, rx, ry):
        ph, w = self.ph, self.w
        me = w.me
        ang = math.atan2(ry - w.by, rx - w.bx)
        ca, sa = math.cos(ang), math.sin(ang)
        in_pen = ph.in_their_pen(rx, ry)
        for o in w.opps:
            relx = (o.x - w.bx) * ca + (o.y - w.by) * sa
            if relx < -4.0:
                continue
            ctrl = ph.catchable if (o.goalie and in_pen) else ph.kickable
            ox, oy = ph.inertia(o, n)
            td = math.hypot(ox - rx, oy - ry)
            if td > ph.rsm * n + ctrl:
                continue
            if td - ctrl < 0.001:
                return False
            dash = td - ctrl - 0.2 - math.hypot(o.x - me.x, o.y - me.y) * 0.01
            n_dash = ph.cycles_to_reach(dash)
            n_turn = ph.turn_cycle(o.body, o.speed, td,
                                   math.degrees(math.atan2(ry - oy, rx - ox)),
                                   ctrl, False)
            n_step = n_dash if n_turn == 0 else n_turn + n_dash + 1
            bonus = 1 if rx < 27.0 else 0
            if o.tackling:
                bonus = -5
            if relx > 0.8:
                bonus += 1
            else:
                bonus = 0
            if n_step - bonus <= n:
                return False
        return True

    # ================= ShootGenerator ==================================
    def shoots(self):
        ph, w = self.ph, self.w
        me = w.me
        if math.hypot(me.x - ph.hl, me.y) > 30.0:
            return []
        fx, fy = w.bx, w.by
        glx, gly = ph.hl, -ph.goal_half
        grx, gry = ph.hl, ph.goal_half
        gly += min(1.5, 0.6 + math.hypot(glx - fx, gly - fy) * 0.042)
        gry -= min(1.5, 0.6 + math.hypot(grx - fx, gry - fy) * 0.042)
        if me.x > ph.hl - 1.0 and abs(me.y) < ph.goal_half:
            glx = grx = me.x + 1.5
        courses = []
        nt = 24 * self.gran
        step = abs(gly - gry) / nt
        for i in range(nt + 1):
            self.tick()
            self._create_shoot(glx, gly + step * i, courses)
        # HELIOS evaluateCourses
        goalie = next((o for o in w.opps if o.goalie), None)
        gang = (math.degrees(math.atan2(goalie.y - fy, goalie.x - fx))
                if goalie else 180.0)
        best, best_s = None, -1.0
        for c in courses:
            tx, ty, speed, ang, kick1, gnr, onr = c
            score = 1.0
            if kick1:
                score += 50.0
            if gnr:
                score += 100.0
            if onr:
                score += 100.0
            grate = 1.0
            if goalie is not None:
                var2 = 1.0 if gnr else 100.0
                ad = abs(_norm(ang - gang))
                grate = 1.0 - math.exp(-(ad * ad) / (2.0 * var2))
            yrate = 1.0
            if math.hypot(tx - fx, ty - fy) > 8.0:
                yd = max(0.0, abs(ty) - 4.0)
                yrate = math.exp(-(yd * yd)
                                 / (2.0 * (ph.goal_half - 1.5) ** 2))
            score *= grate * yrate
            if score > best_s:
                best_s, best = score, c
        return [best] if best is not None else []

    def _create_shoot(self, tx, ty, courses):
        ph, w = self.ph, self.w
        fx, fy = w.bx, w.by
        ang = math.degrees(math.atan2(ty - fy, tx - fx))
        dist = math.hypot(tx - fx, ty - fy)
        one = ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy)
        speed = max((dist + 5.0) * (1.0 - ph.decay), max(one, 1.5))
        over = False
        while not over:
            if speed > ph.bmax - 0.001:
                over, speed = True, ph.bmax
            res = self._shoot_ok(fx, fy, tx, ty, speed, ang, dist)
            if res is not None:
                gnr, onr = res
                courses.append((tx, ty, speed, ang, speed <= one + 0.001,
                                gnr, onr))
                return
            speed += 0.3

    def _shoot_ok(self, fx, fy, tx, ty, speed, ang, dist):
        ph, w = self.ph, self.w
        me = w.me
        n = ph.move_step(speed, dist)
        if n <= 1:
            return True, True
        a = math.radians(ang)
        vx, vy = speed * math.cos(a), speed * math.sin(a)
        gnr = onr = True
        for o in w.opps:
            if o.tackling:
                continue
            if o.x < ph.pen_x - 30.0 or abs(o.y) > ph.pen_hw:
                continue
            afs = math.degrees(math.atan2(o.y - me.y, o.x - me.x))
            if abs(_norm(ang - afs)) > 90.0:
                continue
            mc = ph.min_reach_cycle(o.x, o.y, fx, fy, ang)
            if mc < 0:
                continue
            if o.goalie:
                noise = math.hypot(o.x - me.x, o.y - me.y) * 0.02
                for cyc in range(mc, n):
                    bx, by = ph.ball_at(fx, fy, vx, vy, cyc)
                    if bx > ph.hl:
                        break
                    inpen = ph.in_their_pen(bx, by)
                    ctrl = ph.catchable if inpen else ph.kickable
                    ix, iy = ph.inertia(o, cyc)
                    td = math.hypot(bx - ix, by - iy)
                    if inpen:
                        td -= noise
                    if td - ctrl - 0.15 < 0.001:
                        return None
                    dash = td
                    if cyc > 1:
                        dash = (dash - ctrl * 0.9) * 0.999
                    n_dash = ph.cycles_to_reach(dash)
                    if n_dash > cyc:
                        continue
                    n_turn = ph.turn_cycle(o.body, o.speed, td,
                                           math.degrees(math.atan2(
                                               by - iy, bx - ix)),
                                           ctrl + 0.1, True)
                    n_step = n_dash if n_turn == 0 else n_turn + n_dash + 1
                    bonus = 0 if inpen else -1
                    if n_step <= cyc + bonus:
                        return None
                    if inpen and n_step <= cyc + 1:
                        gnr = False
                continue
            for cyc in range(mc, n):
                bx, by = ph.ball_at(fx, fy, vx, vy, cyc)
                ix, iy = ph.inertia(o, cyc)
                td = math.hypot(bx - ix, by - iy)
                if td - ph.kickable < 0.001:
                    return None
                dash = td - ph.kickable * 0.8 if cyc > 1 else td
                n_dash = ph.cycles_to_reach(dash)
                if n_dash > cyc:
                    continue
                n_turn = ph.turn_cycle(o.body, o.speed, td,
                                       math.degrees(math.atan2(by - iy,
                                                               bx - ix)),
                                       ph.kickable, True)
                n_step = n_dash if n_turn == 0 else n_turn + n_dash + 1
                if n_step <= cyc - 1:
                    return None
                if n_step <= cyc + 1:
                    onr = False
        return gnr, onr

    # ================= FieldAnalyzer::can_shoot_from ===================
    def can_shoot_from(self, is_self, x, y):
        ph, w = self.ph, self.w
        if math.hypot(ph.hl - x, y) > 17.0:
            return False
        thr = 20.0 if is_self else 15.0
        am = math.degrees(math.atan2(-ph.goal_half + 0.5 - y, ph.hl - x))
        ap = math.degrees(math.atan2(ph.goal_half - 0.5 - y, ph.hl - x))
        cands = []
        for o in w.opps:
            d = math.hypot(o.x - x, o.y - y)
            if d > 20.0:
                continue
            ctrl = ph.catchable if o.goalie else ph.kickable
            hide = math.degrees(math.asin(min(1.0, ctrl / max(d, 1e-6))))
            cands.append((math.degrees(math.atan2(o.y - y, o.x - x)), hide))
        width = abs(_norm(ap - am))
        st = max(2.0, width / 10.0)
        best = 0.0
        a = 0.0
        while a < width + 0.001:
            sa = am + a
            mn = 180.0
            for oa, hide in cands:
                diff = abs(_norm(oa - sa)) - (hide if is_self else hide * 0.5)
                if diff < mn:
                    mn = diff
                    if mn < thr:
                        break
            if mn > best:
                best = mn
            a += st
        return best >= thr

    # ================= sample_field_evaluator ==========================
    def evaluate(self, holder_self, bx, by):
        ph = self.ph
        if bx > ph.hl - 0.1 and abs(by) < ph.goal_half + 2.0:
            return 1.0e7
        if abs(bx) > ph.hl or abs(by) > ph.hw:
            return -1.0e9
        v = bx + max(0.0, 40.0 - math.hypot(ph.hl - bx, by))
        if self.can_shoot_from(holder_self, bx, by):
            v += 1.0e6
            if holder_self:
                v += 5.0e5
        elif self.ys_key:
            v += 5.0 * _ys_key_density(ph, bx, by, bx, by)
        return v

    # ================= deeper-layer generators =======================
    # ActGen_DirectPass + SimplePassChecker and ActGen_SimpleDribble:
    # helios-base ships them, and the HELIOS planner used by FULL enables
    # them ("simple_pass", "simple_dribble") for chain steps 2..4.
    @staticmethod
    def _pass_speed(dist):
        if dist >= 20.0:
            return 2.5
        if dist >= 8.0:
            return 2.0
        if dist >= 5.0:
            return 1.8
        return 1.5

    def _simple_pass_ok(self, hx, hy, h_goalie, tx, ty, t_goalie, speed,
                        offside):
        ph, w = self.ph, self.w
        if speed < 1.5:
            return False
        pdist = math.hypot(tx - hx, ty - hy)
        if pdist <= 4.0 or pdist >= 35.0:
            return False
        if tx >= offside:
            return False
        if tx <= ph.our_pen_x + 3.0 and abs(ty) <= ph.pen_hw + 3.0:
            return False
        if abs(tx) >= ph.hl or abs(ty) >= ph.hw:
            return False
        recv_move = 0.0
        pang = math.degrees(math.atan2(ty - hy, tx - hx))
        ignore2 = (pdist + (2.0 if tx >= 25.0 else 6.0)) ** 2
        cone = 360.0
        for o in w.opps:
            if (o.x - tx) ** 2 + (o.y - ty) ** 2 < 25.0:
                return False
            ox, oy = ph.inertia_final(o)
            d2 = (ox - hx) ** 2 + (oy - hy) ** 2
            if d2 > ignore2:
                continue
            if math.hypot(ox - tx, oy - ty) < recv_move * 0.85:
                return False
            ad = abs(_norm(math.degrees(math.atan2(oy - hy, ox - hx))
                           - pang))
            if ad < cone:
                cone = ad
        thr = 14.5
        if tx - 2.0 <= hx:
            thr = 17.5
        if hx >= 25.0 and tx >= 25.0:
            thr = 14.5
        if h_goalie:
            thr = 15.0
        thr *= 2.5 / speed
        return cone > thr

    def _deep_children(self, st):
        """Children of a predicted state (chain step >= 2)."""
        ph, w = self.ph, self.w
        holder, hx, hy, pos, olds, spend = st
        out = []
        offside = max(w.offside, hx)
        # shoot (ActGen_Shoot)
        if self.can_shoot_from(holder == w.me.unum, hx, hy):
            out.append((_Act("shoot", ph.hl, 0.0, holder, 1, 1, 1, ph.bmax,
                             "shoot", final=True),
                        (holder, ph.hl, 0.0, pos, olds, spend + 1)))
        if not self.deep:
            return out
        # direct passes
        for unum, (px, py, gk) in pos.items():
            if unum == holder or unum in olds:
                continue
            d = math.hypot(px - hx, py - hy)
            sp = self._pass_speed(d)
            if not self._simple_pass_ok(hx, hy, pos[holder][2], px, py, gk,
                                        sp, offside):
                continue
            n = ph.move_step(sp, d)
            if n < 0:
                continue
            npos = dict(pos)
            npos[unum] = (px, py, gk)
            out.append((_Act("pass", px, py, unum, n + 2, n, 2, sp,
                             "directPass"),
                        (unum, px, py, npos, olds | {unum}, spend + n + 2)))
        # simple dribbles
        for a in range(8):
            ang = 45.0 * a
            if hx < 16.0 and abs(_norm(ang)) > 100.0:
                continue
            if hx < -36.0 and abs(hy) < 20.0 and abs(_norm(ang)) > 45.0:
                continue
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            for dd in (1.75, 3.5, 5.25):
                tx, ty = hx + ca * dd, hy + sa * dd
                if abs(tx) > ph.hl - 1.0 or abs(ty) > ph.hw - 1.0:
                    continue
                hstep = 2 + ph.cycles_to_reach(dd - ph.kickable * 0.5)
                blocked = False
                for o in w.opps:
                    ostep = 1 + ph.cycles_to_reach(
                        math.hypot(o.x - tx, o.y - ty) - ph.kickable)
                    if ostep - 2 <= hstep:
                        blocked = True
                        break
                if blocked:
                    continue
                npos = dict(pos)
                npos[holder] = (tx, ty, pos[holder][2])
                out.append((_Act("dribble", tx, ty, holder, hstep, hstep, 1,
                                 0.0, "simpleDribble"),
                            (holder, tx, ty, npos, olds, spend + hstep)))
        return out

    # ================= ActionChainGraph (best-first) ===================
    MAX_CHAIN = 4
    ys_key = False
    ORDERED_MIN = False      # move ordering in the interception MIN node
    TRANSPOSITION = False    # transposition table in the chain search

    @staticmethod
    def _state_key(st):
        holder, hx, hy, pos, olds, _ = st
        return (holder, round(hx, 3), round(hy, 3), olds,
                tuple(sorted((u, round(p[0], 3), round(p[1], 3))
                             for u, p in pos.items())))

    def chain(self):
        """Greedy best-first search over action chains (helios
        ActionChainGraph): the frontier is ordered by the heuristic state
        evaluation, run until the time allowance is spent instead of a
        fixed 500 evaluations. With TRANSPOSITION, a state reached again
        (same holder, ball, teammates and ball history, e.g. two short
        dribbles = one long one) is not expanded twice: its subtree is the
        same, so the result is unchanged and the time goes to new states.
        Returns (first action, value, evaluated count)."""
        w = self.w
        me = w.me
        pos0 = {m.unum: (m.x, m.y, m.goalie) for m in w.mates}
        pos0[me.unum] = (w.bx, w.by, me.goalie)
        cur = self.evaluate(True, w.bx, w.by)
        best_path, best_ev = None, cur
        n_eval = 0
        roots = []
        try:
            roots += self.passes()
            if self.cross:
                roots += self.crosses()
            roots += self.short_dribbles()
            roots += self.self_passes()
        except _Timeout:
            pass
        heap = []
        tie = 0
        for c in roots:
            npos = dict(pos0)
            if c.kind == "pass":
                npos[c.unum] = (c.tx, c.ty, pos0[c.unum][2])
            else:
                npos[me.unum] = (c.tx, c.ty, me.goalie)
            st = (c.unum, c.tx, c.ty, npos, frozenset((me.unum, c.unum)),
                  c.steps)
            ev = self.evaluate(c.unum == me.unum, c.tx, c.ty)
            n_eval += 1
            path = (c,)
            if ev > best_ev:
                best_ev, best_path = ev, path
            tie += 1
            heapq.heappush(heap, (-ev, tie, path, st))
        closed = {}
        try:
            while heap:
                _, _, path, st = heapq.heappop(heap)
                if len(path) >= self.MAX_CHAIN or path[-1].final:
                    continue
                if self.TRANSPOSITION:
                    key = self._state_key(st)
                    if closed.get(key, 99) <= len(path):
                        continue
                    closed[key] = len(path)
                self.tick()
                for act, nst in self._deep_children(st):
                    ev = self.evaluate(act.unum == me.unum, nst[1], nst[2])
                    n_eval += 1
                    npath = path + (act,)
                    if ev > best_ev:
                        best_ev, best_path = ev, npath
                    tie += 1
                    heapq.heappush(heap, (-ev, tie, npath, nst))
                    if n_eval > 20000:
                        raise _Timeout()
        except _Timeout:
            pass
        if best_path is None:
            return None, best_ev, n_eval
        return best_path[0], best_ev, n_eval


# --------------------------------------------------------------------------
# YuShan2023 key receiving areas
# --------------------------------------------------------------------------

def _ys_key_density(ph, bx, by, x, y):
    """Kernel-density map of good receiving points given the ball position.
    YuShan2023 fit Gaussian KDEs to passes that ended in shots: from a wide
    attacking position the receiver goes to the same-side edge of the
    penalty area, the free forward to the middle of the box. Here the
    kernels sit at those areas instead of being fitted from logs."""
    if bx < 10.0:
        return 0.0
    if abs(by) > 12.0 and bx > 25.0:
        s = 1.0 if by > 0.0 else -1.0
        kern = ((ph.hl - 12.0, s * 10.0, 4.0), (ph.hl - 8.0, 0.0, 4.0))
    else:
        kern = ((ph.hl - 14.0, 9.0, 4.5), (ph.hl - 14.0, -9.0, 4.5),
                (ph.hl - 9.0, 0.0, 4.0))
    best = 0.0
    for cx, cy, h in kern:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        best = max(best, math.exp(-d2 / (2.0 * h * h)))
    return best


# --------------------------------------------------------------------------
# Marking assignment
#   costs[i] = (distance row of marker i to every threat's marking spot,
#               the threats marker i may take)
#   objective (Cyrus OMAM): cover the most dangerous threats first
#   (lexicographic over threats in danger order), then the least total
#   running distance. Returns one threat index (or -1: none) per marker.
# --------------------------------------------------------------------------

def _omam_marking(costs, n_threats):
    """Original Cyrus2D OMAM: enumerate every assignment."""
    best = [None, None]

    def rec(i, used, cost, asg):
        if i == len(costs):
            key = (tuple(1 if j in used else 0 for j in range(n_threats)), -cost)
            if best[0] is None or key > best[0]:
                best[0], best[1] = key, list(asg)
            return
        row, ok = costs[i]
        asg.append(-1)
        rec(i + 1, used, cost, asg)
        asg.pop()
        for j in ok:
            if j in used:
                continue
            used.add(j)
            asg.append(j)
            rec(i + 1, used, cost + row[j], asg)
            asg.pop()
            used.discard(j)

    rec(0, set(), 0.0, [])
    return best[1]


def _csp_marking(costs, n_threats):
    """The same assignment as a constraint satisfaction problem.

    variables  : the markers
    domains    : the (up to 3) threats each may take, plus 'none' (-1)
    constraint : all-different over threats (one marker per attacker)
    objective  : (coverage of threats in danger order, -total distance),
                 maximised lexicographically

    Backtracking search with MRV variable ordering, forward checking
    (a threat taken by one marker leaves the other domains) and branch &
    bound: a partial assignment is cut when even covering every threat
    still left in the remaining domains cannot beat the best coverage, or
    ties it while having run further already. Values are tried in the
    original order ('none' first, then nearest threat first), so ties go
    the same way as the exhaustive enumeration."""
    n = len(costs)
    best = [None, None]
    asg = [-1] * n

    def cov(used):
        return tuple(1 if j in used else 0 for j in range(n_threats))

    def rec(unassigned, used, cost, doms):
        if not unassigned:
            key = (cov(used), -cost)
            if best[0] is None or key > best[0]:
                best[0], best[1] = key, list(asg)
            return
        if best[0] is not None:
            reach = set(used)
            for v in unassigned:
                reach.update(doms[v])
            opt = cov(reach)
            if opt < best[0][0] or (opt == best[0][0] and -cost <= best[0][1]):
                return
        # MRV: the marker with the fewest threats left; ties by index so
        # the search order matches the enumeration
        v = min(unassigned, key=lambda i: (len(doms[i]), i))
        rest = [i for i in unassigned if i != v]
        row = costs[v][0]
        asg[v] = -1
        rec(rest, used, cost, doms)
        for j in doms[v]:
            nd = dict(doms)
            for i in rest:                      # forward checking
                if j in nd[i]:
                    nd[i] = [k for k in nd[i] if k != j]
            asg[v] = j
            rec(rest, used | {j}, cost + row[j], nd)
        asg[v] = -1

    rec(list(range(n)), frozenset(),
        0.0, {i: list(costs[i][1]) for i in range(n)})
    return best[1]


# --------------------------------------------------------------------------
# UCT over action chains (HfutEngine2019 / Alice2022 style MCTS)
# --------------------------------------------------------------------------

class _UNode:
    __slots__ = ("act", "st", "depth", "kids", "n", "total", "expanded",
                 "value")

    def __init__(self, act, st, depth, value):
        self.act, self.st, self.depth = act, st, depth
        self.kids = []
        self.n = 0
        self.total = 0.0
        self.expanded = False
        self.value = value


def _uct_scale(v):
    """Map the helios evaluator (with its 1e6 / 1e7 shoot bonuses) onto a
    bounded reward so averages are meaningful."""
    if v >= 1.0e7:
        return 3.0
    if v >= 1.0e6:
        return 2.0
    return max(-1.0, min(1.5, v / 100.0))


def _uct_chain(hp, rnd, max_iter=100000):
    w = hp.w
    me = w.me
    pos0 = {m.unum: (m.x, m.y, m.goalie) for m in w.mates}
    pos0[me.unum] = (w.bx, w.by, me.goalie)
    cur = _uct_scale(hp.evaluate(True, w.bx, w.by))
    roots = []
    try:
        roots += hp.passes()
        if hp.cross:
            roots += hp.crosses()
        roots += hp.short_dribbles()
        roots += hp.self_passes()
    except _Timeout:
        pass
    if not roots:
        return None, cur, 0
    root = _UNode(None, None, 0, cur)
    for c in roots:
        npos = dict(pos0)
        if c.kind == "pass":
            npos[c.unum] = (c.tx, c.ty, pos0[c.unum][2])
        else:
            npos[me.unum] = (c.tx, c.ty, me.goalie)
        st = (c.unum, c.tx, c.ty, npos, frozenset((me.unum, c.unum)),
              c.steps)
        v = _uct_scale(hp.evaluate(c.unum == me.unum, c.tx, c.ty))
        root.kids.append(_UNode(c, st, 1, v))
    root.expanded = True
    it = 0
    try:
        while it < max_iter:
            hp.tick()
            it += 1
            path = [root]
            node = root
            # selection
            while node.expanded and node.kids:
                ln = math.log(node.n + 1.0)
                best, bu = None, -1e18
                for k in node.kids:
                    if k.n == 0:
                        u = 10.0 + k.value
                    else:
                        u = k.total / k.n + 0.7 * math.sqrt(ln / k.n)
                    if u > bu:
                        bu, best = u, k
                node = best
                path.append(node)
            # expansion
            if (not node.expanded and node.depth < hp.MAX_CHAIN
                    and not node.act.final):
                for act, nst in hp._deep_children(node.st):
                    v = _uct_scale(hp.evaluate(act.unum == me.unum,
                                               nst[1], nst[2]))
                    node.kids.append(_UNode(act, nst, node.depth + 1, v))
                node.expanded = True
            # rollout: random continuation, reward = best state reached
            reward = node.value
            st, depth = node.st, node.depth
            final = node.act.final
            while depth < hp.MAX_CHAIN and not final:
                ch = hp._deep_children(st)
                if not ch:
                    break
                act, st = ch[int(rnd.uniform() * len(ch))]
                depth += 1
                final = act.final
                reward = max(reward, _uct_scale(
                    hp.evaluate(act.unum == me.unum, st[1], st[2])))
            for nd in path:
                nd.n += 1
                nd.total += reward
    except _Timeout:
        pass
    vis = [k for k in root.kids if k.n > 0]
    if not vis:
        best = max(root.kids, key=lambda k: k.value)
        return (best.act if best.value > cur else None), best.value, it
    best = max(vis, key=lambda k: (k.total / k.n, k.n))
    if best.total / best.n <= cur:
        return None, cur, it
    return best.act, best.total / best.n, it


# --------------------------------------------------------------------------
# The team
# --------------------------------------------------------------------------

class Agent1(StarterAgent):
    SPEND_FRACTION = 0.68      # of the total match budget
    DEFAULT_BUDGET_MS = 75000.0
    ON_BALL_SHARE = 12.0       # an on-ball decision gets ~12 cycles' worth
    OFF_BALL_SHARE = 0.8
    MAX_ON_MS = 420.0
    MAX_OFF_MS = 80.0

    DEEP_CHAIN = True          # DirectPass / SimpleDribble at steps 2-4
    CHAIN_DEPTH = 4            # max actions per chain (helios uses 4)
    USE_CROSS = False          # helios CrossGenerator (1W-10L when on)
    GRANULARITY = 1            # 1 = helios-base sampling, 2 = 2x finer
    USE_BLOCK = True           # Cyrus2D Bhv_BasicBlock
    USE_MARK = True            # Cyrus OMAM marking (5W-3L vs 0W-8L off)
    USE_UNMARK = False         # Cyrus Bhv_Unmark (optional layer)
    USE_MINIMAX = False        # minimax refinement of defensive spots
    # Gliders2d (Prokopenko & Wang 2018, arXiv:1812.10202) steps
    # Single-step gains below came from 11-12 match arms (+-0.6/g noise).
    # Pooled vs FULL: these three on -0.70/g (54 g), all off -0.50/g (46 g).
    G2D_STAMINA = False        # v1.1 max dash in four situations
    G2D_PRESSING = False       # v1.2 evolved pressing levels (-0.07/g)
    G2D_VORONOI = False        # v1.4 forwards to open spots (+0.1/g)
    G2D_DEEP_FORM = False      # v1.5 defence nearer own goal (-0.30/g, 23 g)
    G2D_RISKY = False          # v1.6 risky through passes (+0.4/g)
    MC_BLOCK = False           # Alice2022-style MC block (1W-10L)
    TACKLE_NEAR_GOAL = False   # LightAgent tackle thresholds near our goal
    # the same decisions, written as textbook algorithms
    CSP_MARKING = True         # marking as a CSP (MRV, forward checking, B&B)
                               # (same assignment as OMAM: 3000/3000 cases)
    ORDERED_MIN = True         # move ordering in the interception MIN node
                               # (same candidates: 300/300 states)
    TRANSPOSITION = False      # transposition table in the chain search
                               # (vs FULL 58 g each: -0.69/g on, -0.34/g off)
    # football heuristics. Vs FULL, 28-30 games each: none -0.03/g,
    # on-ball five -0.79/g, off-ball three -0.62/g, all eight -0.73/g.
    H_WING_RUN = False        # carry it along the touchline when clear
    H_CUTBACK = False          # at the byline: pull back to the spot
    H_KEEPER_OUT = False       # goalie off his line: shoot at the empty side
    H_CLEAR_DANGER = False     # pressed in our box: clear it
    H_PROTECT_LEAD = False     # leading late: take it to the corner flag
    H_GIVE_AND_GO = False      # passer runs on for the return
    H_COUNTER = False          # their defence caught up field: forward goes
    H_FAR_POST = False         # ball wide in the final third: far post run
    MCTS_CHAIN = False         # UCT over the action chain
    # YuShan (TDPs 2019-2023)
    YS_RUN = False             # 2022 9x9 run-target grid + ranker
    YS_KEY = False             # 2023 key receiving areas (KDE map)

    def __init__(self):
        super().__init__()
        self._b0 = None
        self._total_cycles = None
        self._recover_mode = False
        self._block = None
        self._unmark = None
        self._ph = None
        self._last_kick = None     # (cycle, kind, tx, ty, launch speed)

    # ------------------------------------------------------------------
    # Budget pacing
    # ------------------------------------------------------------------
    def _remaining(self, agent):
        try:
            return float(agent.budget.remaining_ms)
        except Exception:
            return float("inf")

    def _decision_left(self, agent):
        try:
            return float(agent.budget.decision_remaining_ms)
        except Exception:
            return 1000.0

    def _match_cycles(self, agent):
        if self._total_cycles is None:
            sp = agent.server_params
            ht = _f(getattr(sp, "half_time", 0.0))
            step = _f(getattr(sp, "simulator_step", 100.0), 100.0) or 100.0
            if ht <= 0.0:
                total = 6000
            elif ht < 600.0:          # seconds
                total = int(2 * ht * 1000.0 / step)
            else:                      # already cycles
                total = int(2 * ht)
            self._total_cycles = max(600, total)
        return self._total_cycles

    def _allowance_ms(self, agent, on_ball):
        rem = self._remaining(agent)
        cyc = max(0, int(getattr(agent.wm, "cycle", 0)))
        total = self._match_cycles(agent)
        if rem == float("inf"):
            rem = self.DEFAULT_BUDGET_MS * max(0.05, 1.0 - cyc / total)
        if self._b0 is None or rem > self._b0:
            self._b0 = rem
        if rem < 2500.0:
            return 1.5 if on_ball else 0.5
        left = max(120, int((total - cyc) * 0.75) + 40)
        reserve = self._b0 * (1.0 - self.SPEND_FRACTION)
        rate = max(0.0, rem - reserve) / left
        if on_ball:
            allow = max(15.0, rate * self.ON_BALL_SHARE)
            cap = self.MAX_ON_MS
        else:
            allow = rate * self.OFF_BALL_SHARE
            cap = self.MAX_OFF_MS
        allow = min(allow, cap, self._decision_left(agent) - 300.0,
                    0.03 * rem)
        return max(1.0, allow)

    def _deadlines(self, agent, allow_ms, wall_margin_ms):
        left = self._decision_left(agent)
        cpu = time.process_time() + allow_ms / 1000.0
        wall = time.perf_counter() + max(
            1.0, min(allow_ms * 1.6, left - wall_margin_ms)) / 1000.0
        return cpu, wall

    def _phys(self, agent):
        if self._ph is None:
            self._ph = _Phys(agent)
        return self._ph

    # ------------------------------------------------------------------
    # ON BALL
    # ------------------------------------------------------------------
    def on_ball(self, agent) -> None:
        try:
            if self._on_ball(agent):
                return
        except _Timeout:
            pass
        except Exception as e:
            note(agent, "ERR_on", err=type(e).__name__ + ":" + str(e)[:60])
        Action.hold(agent)
        Action.clear(agent)

    def _on_ball(self, agent):
        t0 = time.process_time()
        ph = self._phys(agent)
        w = _World(agent)
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, True),
                                    250.0)
        hp = _Helios(ph, w, cpu, wall, self.DEEP_CHAIN, self.USE_CROSS,
                     self.GRANULARITY, self.G2D_RISKY)
        hp.ys_key = self.YS_KEY
        hp.MAX_CHAIN = self.CHAIN_DEPTH
        hp.ORDERED_MIN = self.ORDERED_MIN
        hp.TRANSPOSITION = self.TRANSPOSITION

        # 1. Bhv_StrictCheckShoot
        try:
            shots = hp.shoots()
        except _Timeout:
            shots = []
        if shots:
            tx, ty, speed, ang, kick1, gnr, onr = shots[0]
            one = ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy)
            note(agent, "shoot", speed=round(speed, 2), one=round(one, 2))
            if one > speed * 0.99:
                Action.kick_one_step(agent, Vector2D(tx, ty))
            else:
                n = Tools.steps_required(speed, math.hypot(tx - w.bx,
                                                           ty - w.by),
                                         ph.decay, ph.bmax) or 1
                Action.kick(agent, Vector2D(tx, ty), n, accept_late=1)
            return True
        if self._extra_shot(agent):
            return True
        # 1b. football heuristics that do not need the search
        if self._heur_pre(agent, ph, w):
            return True

        # 2. Bhv_PlannedAction::doTurnToForward
        if self._turn_forward(agent, ph, w):
            return True

        # 3. action chain
        if self.MCTS_CHAIN:
            best, ev, n = _uct_chain(hp, _Rand(w.cycle * 31 + w.me.unum))
        else:
            best, ev, n = hp.chain()
        ms = round((time.process_time() - t0) * 1000.0, 1)
        # 3b. football heuristics that override the search's choice
        if self._heur_post(agent, ph, w, hp, best):
            return True
        if best is None:
            note(agent, "hold", n=n, ms=ms)
            if w.bx < -ph.hl + 8.0 and abs(w.by) < ph.goal_half + 1.0:
                Action.clear(agent)
            else:
                Action.hold(agent)
            return True
        note(agent, best.desc, x=round(best.tx, 1), y=round(best.ty, 1),
             n=n, ms=ms)
        # pass: smart kick at the planned pace. dribble / self pass: one
        # kick to the trap point, then the off-ball intercept runs onto it
        # (Bhv_NormalDribble kicks, then turns and dashes onto the ball).
        Action.kick(agent, Vector2D(best.tx, best.ty), max(1, best.flight),
                    accept_late=1)
        dist = math.hypot(best.tx - w.bx, best.ty - w.by)
        self._last_kick = (w.cycle, best.kind, best.tx, best.ty,
                           min(ph.bmax, ph.first_speed(dist, max(1, best.flight))))
        return True

    def _extra_shot(self, agent):
        """Hook: other shot models, tried when the strict shoot finds none."""
        return False

    # ------------------------------------------------------------------
    # Football heuristics (each behind its own switch)
    # ------------------------------------------------------------------
    def _heur_pre(self, agent, ph, w):
        if self.H_KEEPER_OUT and self._keeper_out_shot(agent, ph, w):
            return True
        if self.H_CLEAR_DANGER and self._clear_danger(agent, ph, w):
            return True
        if self.H_PROTECT_LEAD and self._protect_lead(agent, ph, w):
            return True
        return False

    def _heur_post(self, agent, ph, w, hp, best):
        if self.H_CUTBACK and self._cutback(agent, ph, w, hp):
            return True
        if self.H_WING_RUN and self._wing_run(agent, ph, w, best):
            return True
        return False

    def _path_clear(self, ph, w, tx, ty, speed, delay=0):
        """True if no opponent reaches a ball kicked at `speed` from the
        ball towards (tx, ty) before it gets there."""
        d = math.hypot(tx - w.bx, ty - w.by)
        if d < 0.5:
            return False
        n = ph.move_step(speed, d)
        if n < 0:
            n = 40
        ux, uy = (tx - w.bx) / d, (ty - w.by) / d
        for o in w.opps:
            ctrl = ph.catchable if o.goalie else ph.kickable
            for c in range(1, n + 1):
                bx, by = ph.ball_at(w.bx, w.by, ux * speed, uy * speed, c)
                if ph.cycles_to_reach(math.hypot(bx - o.x, by - o.y)
                                      - ctrl) + delay <= c:
                    return False
        return True

    def _keeper_out_shot(self, agent, ph, w):
        """Their goalie is off his line: shoot into the far side of the
        empty goal from distance."""
        gk = next((o for o in w.opps if o.goalie), None)
        if gk is None or gk.x > ph.hl - 4.0:
            return False
        if math.hypot(ph.hl - w.bx, w.by) > 32.0:
            return False
        side = -1.0 if gk.y > 0.0 else 1.0
        best = None
        for ty in (side * (ph.goal_half - 1.0), side * (ph.goal_half - 2.5), 0.0):
            ang = math.degrees(math.atan2(ty - w.by, ph.hl - w.bx))
            v = ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy)
            if v < 2.0:
                continue
            if self._path_clear(ph, w, ph.hl + 0.5, ty, v):
                best = ty
                break
        if best is None:
            return False
        note(agent, "h_keeper_out", gx=round(gk.x, 1))
        Action.kick_one_step(agent, Vector2D(ph.hl + 0.5, best))
        return True

    def _clear_danger(self, agent, ph, w):
        """In our own box with an attacker on us: get rid of it."""
        if w.bx > -ph.hl + 18.0 or abs(w.by) > ph.pen_hw + 2.0:
            return False
        near = min((math.hypot(o.x - w.bx, o.y - w.by) for o in w.opps
                    if not o.goalie), default=99.0)
        if near > 3.5:
            return False
        note(agent, "h_clear", near=round(near, 1))
        Action.clear(agent)
        return True

    def _protect_lead(self, agent, ph, w):
        """Leading late in their half: take the ball to the corner flag
        and shield it there."""
        wm = agent.wm
        try:
            lead = int(wm.our_team_score) - int(wm.their_team_score)
        except Exception:
            return False
        if lead <= 0 or w.cycle < 0.85 * self._match_cycles(agent):
            return False
        if w.bx < 10.0:
            return False
        s = 1.0 if w.by >= 0.0 else -1.0
        tx, ty = ph.hl - 1.5, s * (ph.hw - 1.5)
        near = min((math.hypot(o.x - w.bx, o.y - w.by) for o in w.opps),
                   default=99.0)
        note(agent, "h_protect", near=round(near, 1))
        if near < 2.0 or math.hypot(tx - w.bx, ty - w.by) < 2.0:
            Action.hold(agent)
        else:
            Action.dribble(agent, Vector2D(tx, ty))
        Action.hold(agent)
        return True

    def _cutback(self, agent, ph, w, hp):
        """Wide at the byline: pull it back to the penalty-spot area."""
        if w.bx < ph.hl - 16.0 or abs(w.by) < 8.0:
            return False
        best, best_v = None, None
        try:
            for a in hp.passes():
                if not (ph.hl - 17.0 <= a.tx <= ph.hl - 5.0
                        and abs(a.ty) < 9.0):
                    continue
                v = hp.evaluate(False, a.tx, a.ty)
                if best_v is None or v > best_v:
                    best, best_v = a, v
        except _Timeout:
            pass
        if best is None:
            return False
        note(agent, "h_cutback", x=round(best.tx, 1), y=round(best.ty, 1))
        Action.kick(agent, Vector2D(best.tx, best.ty), max(1, best.flight),
                    accept_late=1)
        self._last_kick = (w.cycle, "pass", best.tx, best.ty, best.speed)
        return True

    def _wing_run(self, agent, ph, w, best):
        """Wide in their half with the touchline corridor free: carry it
        along the line (a tackle there tends to put it out off the
        defender: our throw-in). Not when the search found a pass that
        gains a lot of ground."""
        if w.bx < -10.0 or w.bx > ph.hl - 12.0 or abs(w.by) < ph.hw - 14.0:
            return False
        if best is not None and best.kind == "pass" and best.tx > w.bx + 8.0:
            return False
        s = 1.0 if w.by > 0.0 else -1.0
        ty = s * (ph.hw - 2.5)
        tx = min(w.bx + 6.0, ph.hl - 6.0)
        for o in w.opps:
            if o.goalie:
                continue
            ahead = o.x - w.bx
            if -0.5 < ahead < 7.0 and abs(o.y - ty) < 3.5:
                return False
            if math.hypot(o.x - w.bx, o.y - w.by) < 1.5:
                return False
        note(agent, "h_wing_run", x=round(tx, 1))
        Action.dribble(agent, Vector2D(tx, ty))
        Action.hold(agent)
        return True

    # off-ball heuristics ------------------------------------------------
    def _heur_off(self, agent, ph, w, role, our_ball):
        """-> (x, y, power) target or None."""
        if not our_ball:
            return None
        me = w.me
        if self.H_GIVE_AND_GO and role >= 6:
            lk = self._last_kick
            if (lk is not None and lk[1] == "pass" and 0 < w.cycle - lk[0] <= 12
                    and w.stamina > 0.35 * ph.stamina_max):
                tx = min(me.x + 10.0, w.offside - 1.0, ph.hl - 6.0)
                ty = me.y * 0.7
                if tx > me.x + 2.0:
                    note(agent, "h_give_go")
                    return tx, ty, 100.0
        if self.H_COUNTER and role >= 9:
            back = sum(1 for o in w.opps if not o.goalie and o.x > w.bx)
            if back <= 2 and w.bx < ph.hl - 25.0:
                tx = min(w.offside - 1.0, ph.hl - 8.0)
                ty = max(-12.0, min(12.0, me.y))
                note(agent, "h_counter", back=back)
                return tx, ty, 100.0
        if self.H_FAR_POST and role >= 9:
            if w.bx > 20.0 and abs(w.by) > 10.0:
                s = 1.0 if w.by > 0.0 else -1.0
                tx = min(ph.hl - 5.0, w.offside - 0.5)
                note(agent, "h_far_post")
                return tx, -s * 4.0, 100.0
        return None

    def _turn_forward(self, agent, ph, w):
        me = w.me
        fx, fy = 42.0, 0.0
        if abs(_norm(math.degrees(math.atan2(fy - me.y, fx - me.x))
                     - me.body)) < 110.0:
            return False
        for o in w.opps:
            if math.hypot(o.x - me.x, o.y - me.y) < 4.0:
                return False
        snx, sny = me.x + me.vx, me.y + me.vy
        bnx, bny = w.bx + w.bvx, w.by + w.bvy
        if math.hypot(snx - bnx, sny - bny) < ph.kickable - 0.2:
            note(agent, "turnForward")
            Action.turn_to_point(agent, Vector2D(fx, me.y * 0.5))
            return True
        return False

    # ------------------------------------------------------------------
    # OFF BALL: helios Bhv_BasicMove (+ Cyrus block)
    # ------------------------------------------------------------------
    def off_ball(self, agent) -> None:
        try:
            if self._off_ball(agent):
                return
        except _Timeout:
            pass
        except Exception as e:
            note(agent, "ERR_off", err=type(e).__name__ + ":" + str(e)[:60])
        Action.go_to_point(agent, Tools.home_position(agent),
                           max_dash_power=self.dash_power(agent))
        Action.turn_to_ball(agent)

    def _off_ball(self, agent):
        ph = self._phys(agent)
        w = _World(agent)
        wm = agent.wm
        me = w.me
        if me.goalie:
            return False
        it = wm.intercept_table
        self_min = int(it.self_reach_steps)
        mate_min = int(it.first_teammate_reach_steps)
        opp_min = int(it.first_opponent_reach_steps)

        # Bhv_BasicTackle( 0.8, 80 )
        if self._basic_tackle(agent, ph, w, self_min, mate_min, opp_min):
            return True

        role = _role(me.unum, len(w.mates) + 1)
        pressing = 3
        if self.G2D_PRESSING:
            pressing = 13
            if 6 <= role <= 8 and w.bx > -30.0 and me.x < 10.0:
                pressing = 7
            if (abs(w.by) > 22.0 and -36.5 < w.bx < 0.0
                    and role in (4, 5)):
                pressing = 23
        # intercept
        if not wm.kickable_teammate_existance and (
                self_min <= 3
                or (self_min <= mate_min and self_min < opp_min + pressing)):
            note(agent, "intercept", steps=self_min)
            Action.intercept(agent)
            Action.go_to_point(agent, Vector2D(w.bx, w.by), 100.0, 0.5)
            return True

        hx, hy = _xy(Tools.home_position(agent))
        team = [m for m in w.mates if not m.goalie] + [me]
        our_ball = wm.kickable_teammate_existance or mate_min < opp_min
        if self.G2D_DEEP_FORM and not our_ball and role <= 8:
            shift = 6.0 if role <= 5 else 4.0
            hx = max(-ph.hl + 4.0, hx - shift)
        if self.G2D_VORONOI and our_ball and role >= 9:
            hx, hy = self._open_spot(agent, ph, w, hx, hy)

        if opp_min < min(self_min, mate_min):
            ipx, ipy = ph.ball_at(w.bx, w.by, w.bvx, w.bvy, opp_min)
            if self.USE_BLOCK and self._block_move(agent, ph, me, team, ipx,
                                                   ipy, opp_min):
                return True
            if self.USE_MARK and self._mark_move(agent, ph, me, team, w.opps,
                                                 ipx, ipy):
                return True
            if self.USE_MINIMAX:
                holder = min(w.opps, key=lambda o: math.hypot(o.x - ipx,
                                                              o.y - ipy),
                             default=None)
                if holder is not None:
                    hx, hy = self._minimax_spot(agent, ph, me, team, w.opps,
                                                holder, ipx, ipy, hx, hy, 6.0)
        elif self.USE_UNMARK and mate_min < opp_min:
            if self._unmark_move(agent, ph, w, hx, hy, mate_min):
                return True
        elif self.YS_RUN and our_ball and role >= 6:
            hx, hy = self._ys_run_target(agent, ph, w, hx, hy, mate_min)
        hz = self._heur_off(agent, ph, w, role, our_ball)
        if hz is not None:
            Action.go_to_point(agent, Vector2D(hz[0], hz[1]), hz[2], 1.0)
            Action.turn_to_ball(agent)
            return True

        power = self._normal_dash_power(agent, ph, w, self_min, mate_min,
                                        opp_min)
        thr = max(1.0, math.hypot(w.bx - me.x, w.by - me.y) * 0.1)
        note(agent, "position")
        Action.go_to_point(agent, Vector2D(hx, hy), power, thr)
        Action.turn_to_ball(agent)
        return True

    def _normal_dash_power(self, agent, ph, w, self_min, mate_min, opp_min):
        """helios-base BasicMove dash-power rule (get_normal_dash_power)."""
        wm = agent.wm
        me = w.me
        cap = _f(getattr(wm.self, "stamina_capacity", 1.0), 1.0)
        if cap < 0.01:
            return min(ph.max_dash, w.stamina + 50.0)
        if w.stamina < ph.stamina_max * 0.5:
            self._recover_mode = True
        elif w.stamina > ph.stamina_max * 0.7:
            self._recover_mode = False
        my_inc = ph.stamina_inc_max * w.recovery
        if w.our_def > me.x and w.bx < w.our_def + 20.0:
            return ph.max_dash
        if self._recover_mode:
            return max(0.0, my_inc - 25.0)
        if wm.kickable_teammate_existance and \
                math.hypot(w.bx - me.x, w.by - me.y) < 20.0:
            return min(my_inc * 1.1, ph.max_dash)
        if me.x > w.offside:
            return ph.max_dash
        if self.G2D_STAMINA:
            role = _role(me.unum, len(w.mates) + 1)
            if (w.bx > 0.0 and me.x < w.offside
                    and abs(w.bx - me.x) < 25.0):
                return ph.max_dash
            if w.bx < 10.0 and 2 <= role <= 5:
                return ph.max_dash
            if w.bx < -10.0 and 6 <= role <= 8:
                return ph.max_dash
            if w.bx > 36.0 and me.x > 36.0 and mate_min < opp_min - 4:
                return ph.max_dash
        if (w.bx > 25.0 and w.bx > me.x + 10.0 and self_min < opp_min - 6
                and mate_min < opp_min - 6):
            return min(max(ph.max_dash * 0.1, my_inc * 0.5), ph.max_dash)
        return min(my_inc * 1.7, ph.max_dash)

    # ---- helios Bhv_BasicTackle (V12 direction search) -------------------
    def _basic_tackle(self, agent, ph, w, self_min, mate_min, opp_min):
        wm = agent.wm
        me = w.me
        tp = _f(getattr(wm.self, "tackle_probability", 0.0))
        fp = _f(getattr(wm.self, "foul_probability", 0.0))
        use_foul = False
        card = int(_f(getattr(wm.self, "card", 0)))
        if (card == 0 and (w.bx > ph.our_pen_x + 0.5
                           or abs(w.by) > ph.pen_hw + 0.5)
                and tp < fp):
            tp, use_foul = fp, True
        thr = 0.8
        if self.TACKLE_NEAR_GOAL:
            # LightAgent: accept a riskier tackle the nearer our goal it is
            dgoal = math.hypot(w.bx + ph.hl, w.by)
            if dgoal < 18.0:
                thr = 0.55
            elif dgoal < 30.0:
                thr = 0.7
        if tp < thr:
            return False
        into_goal = False
        sx, sy = ph.ball_at(w.bx, w.by, w.bvx, w.bvy, self_min)
        if sx < -ph.hl and w.bvx < -1e-6:
            t = (-ph.hl - w.bx) / w.bvx
            iy = w.by + w.bvy * t
            if abs(iy) < ph.goal_half + 1.0:
                into_goal = True
        facing_goal = abs(_norm(math.degrees(math.atan2(-me.y, ph.hl - me.x))
                                - me.body)) < 45.0
        if not (wm.kickable_opponent_existance or into_goal
                or (opp_min < self_min - 3 and opp_min < mate_min - 3)
                or (self_min >= 5 and math.hypot(w.bx - ph.hl, w.by) < 10.0
                    and facing_goal)):
            return False
        vax = vay = 0.0
        if wm.kickable_opponent_existance:
            dx, dy = -ph.hl - w.bx, -w.by
            d = math.hypot(dx, dy) or 1.0
            vax, vay = dx / d * 0.5, dy / d * 0.5
        shoot_chance = math.hypot(w.bx - ph.hl, w.by) < 20.0
        brel = _norm(math.degrees(math.atan2(w.by - me.y, w.bx - me.x))
                     - me.body)
        rate = ph.tackle_power_rate * (1.0 - 0.5 * abs(brel) / 180.0)
        best_a, best_s = None, -1.0
        for i in range(36):
            a = -180.0 + 10.0 * i
            rel = _norm(a - me.body)
            eff = ph.max_back_tackle_power + (
                ph.max_tackle_power - ph.max_back_tackle_power) * (
                1.0 - abs(rel) / 180.0)
            eff *= rate
            ar = math.radians(a)
            vx = w.bvx + math.cos(ar) * eff + vax
            vy = w.bvy + math.sin(ar) * eff + vay
            sp = math.hypot(vx, vy)
            if sp > ph.bmax:
                vx, vy = vx / sp * ph.bmax, vy / sp * ph.bmax
                sp = ph.bmax
            if abs(math.degrees(math.atan2(vy, vx))) > 90.0:
                continue
            bnx, bny = w.bx + vx, w.by + vy
            bad = False
            for o in w.opps:
                if o.tackling:
                    continue
                if math.hypot(o.x - w.bx, o.y - w.by) > 6.0:
                    continue
                if math.hypot(o.x + o.vx - bnx, o.y + o.vy - bny) < \
                        ph.kickable + 0.1:
                    bad = True
                    break
            if bad:
                continue
            if shoot_chance and vx > 1e-6:
                t = (ph.hl - w.bx) / vx
                if abs(w.by + vy * t) < ph.goal_half - 3.0:
                    sp += 10.0
            if sp > best_s:
                best_s, best_a = sp, a
        if best_a is None:
            return False
        note(agent, "tackle", prob=round(tp, 2), foul=use_foul)
        Action.tackle(agent, _norm(best_a - me.body), use_foul)
        return True

    # ---- Cyrus2D Bhv_BasicBlock -------------------------------------------
    def _dribble_dir(self, ph, x, y):
        """The ball carrier's MAX node: of 36 dribble directions he takes
        the one best for HIM under the field evaluator mirrored to their
        goal (heuristic game-state evaluation, depth 1)."""
        best, best_s = 180.0, -1e9
        for d in range(-180, 180, 10):
            a = math.radians(d)
            tx, ty = x + 10.0 * math.cos(a), y + 10.0 * math.sin(a)
            if abs(tx) > ph.hl or abs(ty) > ph.hw:
                continue
            s = -tx + max(0.0, 40.0 - math.hypot(tx + ph.hl - 2.5, ty))
            if s > best_s:
                best_s, best = s, float(d)
        return best

    def _block_move(self, agent, ph, me, team, ipx, ipy, opp_min):
        """Cyrus2D BasicBlock as a breadth-first search over time: layer
        t is the carrier's predicted position after t cycles of his best
        dribble (_dribble_dir); the goal test is "one of our players can be
        there by then". The first layer that passes gives the block point
        and the blocker (ties by uniform number, so all agree)."""
        blockers = [m for m in team
                    if math.hypot(m.x - ipx, m.y - ipy) <= 40.0]
        if not blockers:
            return False
        blockers.sort(key=lambda m: m.unum)
        x, y = ipx, ipy
        target, who = None, None
        for cyc in range(opp_min + 1, opp_min + 41):
            dd = math.radians(self._dribble_dir(ph, x, y))
            x += 0.7 * math.cos(dd)
            y += 0.7 * math.sin(dd)
            if abs(x) > ph.hl or abs(y) > ph.hw:
                break
            for m in blockers:
                d = math.hypot(x - m.x - m.vx, y - m.y - m.vy)
                if ph.cycles_to_reach(d) <= cyc:
                    target, who = (x, y), m.unum
                    break
            if who is not None:
                break
        if who != me.unum or target is None:
            return False
        if self.MC_BLOCK:
            target = self._mc_block(agent, ph, me, team, ipx, ipy, opp_min,
                                    target)
        cyc_now = int(agent.wm.cycle)
        if self._block is not None:
            (lx, ly), lc = self._block
            safe = 5.0 if math.hypot(me.x - target[0],
                                     me.y - target[1]) > 15.0 else 2.0
            if lc >= cyc_now - 5 and math.hypot(lx - target[0],
                                                ly - target[1]) < safe:
                target = (lx, ly)
            else:
                self._block = (target, cyc_now)
        else:
            self._block = (target, cyc_now)
        note(agent, "block")
        Action.go_to_point(agent, Vector2D(*target), 100.0, 0.5)
        Action.turn_to_point(agent, Vector2D(ipx, ipy))
        return True

    # ---- Gliders2d v1.4: forwards to open spots ------------------------------
    def _open_spot(self, agent, ph, w, hx, hy):
        """Approximates the Voronoi-vertex choice: among points near the
        formation slot and up to the offside line, take the one farthest
        from every opponent (a largest-empty-circle centre), preferring
        depth."""
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, False),
                                    400.0)
        best, best_s = (hx, hy), None
        xmax = w.offside - 0.7
        for dx in (-4.0, 0.0, 4.0, 8.0, 12.0, 16.0):
            if time.process_time() > cpu or time.perf_counter() > wall:
                break
            x = min(hx + dx, xmax)
            for dy in (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0):
                y = max(-ph.hw + 3.0, min(ph.hw - 3.0, hy + dy))
                dmin = min((math.hypot(o.x - x, o.y - y) for o in w.opps
                            if not o.goalie), default=20.0)
                dm = min((math.hypot(m.x - x, m.y - y) for m in w.mates),
                         default=20.0)
                sc = min(dmin, 12.0) + 0.3 * x - 0.25 * math.hypot(
                    x - hx, y - hy)
                if dm < 6.0:
                    sc -= (6.0 - dm)
                if best_s is None or sc > best_s:
                    best_s, best = sc, (x, y)
        return best

    # ---- YuShan2022: run-target grid with a state ranker ---------------------
    def _ys_run_target(self, agent, ph, w, hx, hy, mate_min):
        """YuShan2022 keeps the formation point but corrects it: a 9x9 grid
        (1 m) around it gives 81 candidate states that differ only in this
        runner's position, and a ranker picks the best. Their ranker is a
        LambdaMART model trained on logs; this one scores the features it
        found most important: opponent crowding and a safe pass lane in the
        conservative state, ball x / goal threat in the aggressive one."""
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, False),
                                    400.0)
        me = w.me
        sx, sy = ph.ball_at(w.bx, w.by, w.bvx, w.bvy, mate_min)
        press = min((math.hypot(o.x - sx, o.y - sy) for o in w.opps),
                    default=20.0)
        aggressive = sx > 0.0 and press > 4.0
        xmax = w.offside - 0.5
        best, best_s = (hx, hy), None
        for dx in range(-4, 5):
            if time.process_time() > cpu or time.perf_counter() > wall:
                break
            x = hx + dx
            if x > xmax or abs(x) > ph.hl - 2.0:
                continue
            for dy in range(-4, 5):
                y = hy + dy
                if abs(y) > ph.hw - 2.0:
                    continue
                d = math.hypot(x - sx, y - sy)
                lane = 0.0
                if d > 2.0:
                    v0 = _Helios._pass_speed(d)
                    n = ph.move_step(v0, d)
                    lane = 1.0 if n >= 0 else 0.0
                    if lane > 0.0:
                        vx, vy = (x - sx) / d * v0, (y - sy) / d * v0
                        for o in w.opps:
                            ctrl = ph.catchable if o.goalie else ph.kickable
                            for c in range(1, n + 1):
                                bx, by = ph.ball_at(sx, sy, vx, vy, c)
                                if ph.cycles_to_reach(math.hypot(
                                        bx - o.x, by - o.y) - ctrl) <= c:
                                    lane = 0.0
                                    break
                            if lane == 0.0:
                                break
                dopp, crowd = 10.0, 0
                for o in w.opps:
                    if o.goalie:
                        continue
                    do = math.hypot(o.x - x, o.y - y)
                    dopp = min(dopp, do)
                    if do < 5.0:
                        crowd += 1
                dm = min((math.hypot(m.x - x, m.y - y) for m in w.mates
                          if not m.goalie), default=20.0)
                prog = x + max(0.0, 40.0 - math.hypot(ph.hl - x, y))
                dev = math.hypot(x - hx, y - hy)
                key = (_ys_key_density(ph, sx, sy, x, y) if self.YS_KEY
                       else 0.0)
                if aggressive:
                    s = (6.0 * lane + 0.25 * prog + 0.3 * dopp - 0.5 * crowd
                         + 6.0 * key - 0.3 * dev)
                else:
                    s = (8.0 * lane + 0.6 * dopp - 1.5 * crowd + 0.08 * prog
                         + 2.0 * key - 0.3 * dev)
                s -= 0.8 * max(0.0, 6.0 - dm)
                s -= 0.1 * ph.cycles_to_reach(math.hypot(x - me.x, y - me.y))
                if best_s is None or s > best_s:
                    best_s, best = s, (x, y)
        note(agent, "ys_run", x=round(best[0], 1), y=round(best[1], 1),
             aggr=aggressive)
        return best

    # ---- Alice2022-style Monte-Carlo block --------------------------------
    def _mc_block(self, agent, ph, me, team, ipx, ipy, opp_min, target):
        """Sample likely dribble paths of the ball holder (direction around
        the attacker's preferred one, varying pace) and choose the block
        point whose expected outcome for the attacker is worst: the value
        where the ball is won, or the end of the run if never won."""
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, False),
                                    400.0)
        gx = -ph.hl
        base = self._dribble_dir(ph, ipx, ipy)
        rnd = _Rand(int(agent.wm.cycle) * 7919 + me.unum)
        samples = []
        for _ in range(24):
            d = base + rnd.gauss() * 25.0
            sp = 0.55 + rnd.uniform() * 0.3
            samples.append((math.cos(math.radians(d)) * sp,
                            math.sin(math.radians(d)) * sp))
        others = [m for m in team if m.unum != me.unum]
        bdx, bdy = target[0] - ipx, target[1] - ipy
        bd = math.hypot(bdx, bdy)
        if bd < 1e-6:
            a = math.radians(base)
            ux, uy = math.cos(a), math.sin(a)
        else:
            ux, uy = bdx / bd, bdy / bd
        cands = [target]
        for along in (2.0, 4.0, 6.0, 9.0, 12.0):
            for lat in (0.0, -2.0, 2.0):
                cx, cy = ipx + ux * along - uy * lat, ipy + uy * along + ux * lat
                if abs(cx) < ph.hl - 0.5 and abs(cy) < ph.hw - 0.5:
                    cands.append((cx, cy))
        best, best_v = target, None
        for cx, cy in cands:
            if time.process_time() > cpu or time.perf_counter() > wall:
                break
            t_arr = ph.cycles_to_reach(math.hypot(cx - me.x, cy - me.y))
            tot = 0.0
            for vx, vy in samples:
                x, y = ipx, ipy
                val = None
                for k in range(1, 31):
                    x += vx
                    y += vy
                    if abs(x) > ph.hl or abs(y) > ph.hw:
                        val = -10.0
                        break
                    c = opp_min + k
                    won = min(ph.cycles_to_reach(math.hypot(x - me.x,
                                                            y - me.y) - 1.0),
                              t_arr + ph.cycles_to_reach(
                                  math.hypot(x - cx, y - cy) - 1.0)) <= c
                    if not won:
                        for m in others:
                            if ph.cycles_to_reach(math.hypot(x - m.x, y - m.y)
                                                  - 1.0) <= c:
                                won = True
                                break
                    if won:
                        val = -x + max(0.0, 40.0 - math.hypot(x - gx, y)) \
                            - 30.0
                        break
                if val is None:
                    val = -x + max(0.0, 40.0 - math.hypot(x - gx, y)) + 20.0
                tot += val
            v = tot / len(samples) + 0.1 * math.hypot(cx - me.x, cy - me.y)
            if best_v is None or v < best_v:
                best_v, best = v, (cx, cy)
        return best

    # ---- optional: minimax defensive spot ---------------------------------
    def _opp_best(self, ph, team, opps, holder, ipx, ipy):
        """Value of the opponent holder's best pass or dribble, scored
        with the HELIOS evaluator from their side, our players
        intercepting (ball-step vs reach-step check)."""
        gx = -ph.hl
        best = -1e9
        acts = []
        for o in opps:
            if o.goalie or o.unum == holder.unum:
                continue
            for dd, aa in ((0.0, 0), (4.0, 180), (4.0, 135), (4.0, -135)):
                a = math.radians(aa)
                acts.append((o.x + dd * math.cos(a), o.y + dd * math.sin(a),
                             True))
        for k in range(8):
            a = math.radians(45.0 * k)
            acts.append((ipx + 6.0 * math.cos(a), ipy + 6.0 * math.sin(a),
                         False))
        for tx, ty, is_pass in acts:
            if abs(tx) > ph.hl or abs(ty) > ph.hw:
                continue
            d = math.hypot(tx - ipx, ty - ipy)
            if d < 1.0:
                continue
            v = -tx + max(0.0, 40.0 - math.hypot(tx - gx, ty))
            if is_pass:
                n = max(ph.move_step(ph.bmax, d), 3)
                sp = min(ph.bmax, ph.first_speed(d, n))
                ux, uy = (tx - ipx) / d, (ty - ipy) / d
                ok = True
                for m in team:
                    for c in range(1, n + 1):
                        bx, by = ph.ball_at(ipx, ipy, ux * sp, uy * sp, c)
                        if ph.cycles_to_reach(math.hypot(bx - m.x, by - m.y)
                                              - ph.kickable) <= c:
                            ok = False
                            break
                    if not ok:
                        break
            else:
                hstep = 2 + ph.cycles_to_reach(d - ph.kickable * 0.5)
                ok = all(1 + ph.cycles_to_reach(math.hypot(m.x - tx,
                                                           m.y - ty)
                                                - ph.kickable) - 1 > hstep
                         for m in team)
            if ok and v > best:
                best = v
        return best

    def _minimax_spot(self, agent, ph, me, team, opps, holder, ipx, ipy,
                      bx0, by0, radius):
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, False),
                                    400.0)
        cands = [(bx0, by0)]
        for r in (radius * 0.5, radius):
            for k in range(8):
                a = math.radians(45.0 * k)
                cands.append((bx0 + r * math.cos(a), by0 + r * math.sin(a)))
        others = [m for m in team if m.unum != me.unum]
        best, best_s = (bx0, by0), None
        for cx, cy in cands:
            if time.process_time() > cpu or time.perf_counter() > wall:
                break
            if abs(cx) > ph.hl - 1.0 or abs(cy) > ph.hw - 1.0:
                continue
            ghost = _P(me.unum, cx, cy, 0.0, 0.0, me.body, False, False)
            v = self._opp_best(ph, others + [ghost], opps, holder, ipx, ipy)
            v += 0.25 * math.hypot(cx - me.x, cy - me.y) + \
                0.3 * math.hypot(cx - bx0, cy - by0)
            if best_s is None or v < best_s:
                best_s, best = v, (cx, cy)
        return best

    # ---- optional: Cyrus OMAM marking --------------------------------------
    def _mark_move(self, agent, ph, me, team, opps, ipx, ipy):
        gx = -ph.hl
        holder = min(opps, key=lambda o: math.hypot(o.x - ipx, o.y - ipy),
                     default=None)
        if holder is None:
            return False
        free = sorted(team, key=lambda m: (math.hypot(m.x - ipx, m.y - ipy),
                                           m.unum))
        fwd = max(m.unum for m in team)
        markers = [m for m in free[1:] if m.unum != fwd or ipx < -30.0]
        if me.unum not in [m.unum for m in markers]:
            return False
        threats = []
        for o in opps:
            if o.goalie or o.unum == holder.unum:
                continue
            gd = math.hypot(o.x - gx, o.y)
            if gd > 45.0:
                continue
            threats.append((-o.x + max(0.0, 40.0 - gd), o))
        if not threats:
            return False
        threats.sort(key=lambda t: (-t[0], t[1].unum))
        threats = threats[:4]
        pts = []
        for _, o in threats:
            gdx, gdy = gx - o.x, -o.y
            gd = math.hypot(gdx, gdy) or 1.0
            bdx, bdy = ipx - o.x, ipy - o.y
            bd = math.hypot(bdx, bdy) or 1.0
            pts.append((o.x + gdx / gd * 1.5 + bdx / bd * 0.7,
                        o.y + gdy / gd * 1.5 + bdy / bd * 0.7))
        costs = []
        for m in markers:
            row = [math.hypot(m.x - px, m.y - py) for px, py in pts]
            ok = sorted(range(len(pts)), key=lambda j: (row[j], j))[:3]
            costs.append((row, [j for j in ok if row[j] < 35.0]))
        asg = (_csp_marking(costs, len(threats)) if self.CSP_MARKING
               else _omam_marking(costs, len(threats)))
        if asg is None:
            return False
        j = asg[[m.unum for m in markers].index(me.unum)]
        if j < 0:
            return False
        px, py = pts[j]
        if self.USE_MINIMAX:
            px, py = self._minimax_spot(agent, ph, me, team, opps, holder,
                                        ipx, ipy, px, py, 2.5)
        note(agent, "mark", opp=threats[j][1].unum)
        Action.go_to_point(agent, Vector2D(px, py), 100.0, 0.6)
        Action.turn_to_ball(agent)
        return True

    # ---- optional: Cyrus Bhv_Unmark ------------------------------------
    def _unmark_move(self, agent, ph, w, hx, hy, mate_min):
        me = w.me
        cyc = w.cycle
        if w.stamina < 0.45 * ph.stamina_max:
            return False
        sx, sy = ph.ball_at(w.bx, w.by, w.bvx, w.bvy, mate_min)
        if math.hypot(sx - me.x, sy - me.y) > 35.0:
            return False
        if math.hypot(hx - me.x, hy - me.y) > 10.0:
            return False
        if self._unmark is not None:
            (tx, ty), until = self._unmark
            if until > cyc and tx < w.offside:
                note(agent, "unmark_keep")
                Action.go_to_point(agent, Vector2D(tx, ty), 100.0, 1.0)
                Action.turn_to_ball(agent)
                return True
        cands = []
        for dist in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
            for ang in range(-180, 180, 20):
                a = math.radians(ang)
                cands.append((me.x + dist * math.cos(a),
                              me.y + dist * math.sin(a)))
        best, best_s = None, -1e9
        for tx, ty in cands:
            if tx > w.offside or abs(tx) > 52.0 or abs(ty) > 31.5:
                continue
            if math.hypot(tx - hx, ty - hy) > 7.0:
                continue
            if min((math.hypot(m.x - tx, m.y - ty) for m in w.mates),
                   default=99.0) < (5.0 if ph.in_their_pen(tx, ty) else 8.0):
                continue
            evals = []
            for pd, pa in ((0.0, 0), (3.0, 0), (3.0, 90), (3.0, 180),
                           (3.0, -90)):
                a = math.radians(pa)
                px, py = tx + pd * math.cos(a), ty + pd * math.sin(a)
                d = math.hypot(px - sx, py - sy)
                if d < 1.0:
                    continue
                v0 = 1.5 if d < 5 else 2.0 if d < 8 else 2.6 if d < 20 \
                    else 3.0
                if pd < 0.1:
                    v0 = min(3.0, v0 + 0.5)
                n = ph.move_step(v0, d)
                if n < 0:
                    continue
                vx, vy = (px - sx) / d * v0, (py - sy) / d * v0
                cut = 1000
                for o in w.opps:
                    for c in range(1, n + 1):
                        bx, by = ph.ball_at(sx, sy, vx, vy, c)
                        if ph.cycles_to_reach(math.hypot(bx - o.x, by - o.y)
                                              - 0.5) <= c:
                            cut = min(cut, c)
                            break
                if n < cut:
                    evals.append(px + max(0.0, 40.0 - math.hypot(50.0 - px,
                                                                 py)))
            if not evals:
                continue
            avg = sum(evals) / len(evals)
            s = avg + avg * len(evals) / 10.0 + max(evals)
            s += min(10.0, min((math.hypot(o.x - tx, o.y - ty)
                                for o in w.opps), default=10.0))
            if abs(_norm(math.degrees(math.atan2(ty - me.y, tx - me.x))
                         - me.body)) < 15.0:
                s += 10.0
            if s > best_s:
                best_s, best = s, (tx, ty)
        if best is None:
            return False
        self._unmark = (best, cyc + 5)
        note(agent, "unmark")
        Action.go_to_point(agent, Vector2D(*best), 100.0, 1.0)
        Action.turn_to_ball(agent)
        return True


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None

# STARTER CODE ENDS HERE #
