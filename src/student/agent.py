# Your agent goes here. The Agent1 class must hold your final submission.
#
# Agent1 plays with the HELIOS engine and plans against an OPPONENT MODEL.
#
#   Our bot      HeliosAgent: HELIOS-base action-chain search (greedy
#                best-first over passes / dribbles / self-passes / shots,
#                minimax interception checks), CSP marking, block.
#
#   Opponent     For every opponent player, all three engines below
#   model        (HeliosAgent, ColAgent, LightAgent) are run AS THAT PLAYER
#                on a mirrored copy of the world (a stand-in agent that
#                records the first action instead of sending it).
#                Off the ball: each engine gives a target point; his
#                position t cycles ahead is a convex combination of
#                    inertia,  HELIOS target,  Col target,  Light target
#                (Bayesian linear regression on the simplex: prior fitted
#                to many teams' recordings, updated with every movement
#                observed in the match).
#                On the ball: each engine gives the carrier's action (pass /
#                shot target, dribble direction, hold); the mixture weights
#                are a Dirichlet posterior over "which engine is he",
#                updated by Bayes' rule from what he then does.
#
#   Planning     The chain search becomes a deeper minimax tree: at every
#                node our MAX choice is a pass / dribble / shot, and the
#                opponents' reply is the model's predicted position at that
#                node's time; the interception test from those positions is
#                the worst case (MIN). Depth CHAIN_DEPTH (6 vs HELIOS's 4).
#
#   Defending    The block follows the carrier's predicted dribble; marking
#                ranks his predicted receivers higher and cuts their lanes.

import heapq
import itertools
import math
import random
import time

import numpy as np
from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# ==========================================================================
# Stand-in agent: lets any engine decide for an OPPONENT player
# ==========================================================================

class _MP:
    """A player in the mirrored world (the opponent's own frame)."""

    def __init__(self, unum, x, y, vx, vy, body, goalie, tackling):
        self.uniform_number = unum
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(vx, vy)
        self.body_direction = body
        self.neck_direction = body
        self.is_goalie = goalie
        self.is_tackling = tackling
        self.dist_from_self = 0.0
        self.angle_from_self = 0.0
        self.dist_from_ball = 0.0
        self.ball_reach_steps = 0
        self.inertia_final_point = Vector2D(x + vx / 0.6, y + vy / 0.6)
        self.stamina = 8000.0
        self.effort = 1.0
        self.recovery = 1.0
        self.stamina_capacity = 130600.0
        self.is_kickable = False
        self.kick_rate = 0.0
        self.tackle_probability = 0.0
        self.foul_probability = 0.0
        self.card = 0


class _MNS:
    """A plain attribute holder (ball, intercept table, world, budget)."""

    def __init__(holder, **kw):     # (fields may be called "self")
        holder._kv = kw

    def __getattr__(self, name):
        try:
            return self._kv[name]
        except KeyError:
            raise AttributeError(name)


class _Mirror:
    """What an engine receives as `agent` when it decides for an opponent.
    Actions are recorded (the first one counts, as in the real game), not
    sent."""

    def __init__(self, wm, server_params, home, think_ms=20.0):
        self.wm = wm
        self.server_params = server_params
        # the engines size their searches from the budget: think_ms is what
        # one of them may spend deciding for the opponent
        self.budget = _MNS(remaining_ms=3500.0,
                           decision_remaining_ms=300.0 + think_ms)
        self.home = home
        self.memory = {}
        self.act = None

    def add_action(self, msg):
        if self.act is None:
            self.act = ("other", (), {})


def _rec(stand_in, name, args, kw):
    if stand_in.act is None:
        stand_in.act = (name, args, kw)


class _A:
    """The Action API, or a recorder when the agent is a stand-in."""

    @staticmethod
    def kick(agent, target, arrive_in, accept_late=3):
        if isinstance(agent, _Mirror):
            return _rec(agent, "kick", (target,), {})
        Action.kick(agent, target, arrive_in, accept_late)

    @staticmethod
    def kick_one_step(agent, target):
        if isinstance(agent, _Mirror):
            return _rec(agent, "kick_one_step", (target,), {})
        Action.kick_one_step(agent, target)

    @staticmethod
    def dribble(agent, target):
        if isinstance(agent, _Mirror):
            return _rec(agent, "dribble", (target,), {})
        Action.dribble(agent, target)

    @staticmethod
    def hold(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "hold", (), {})
        Action.hold(agent)

    @staticmethod
    def clear(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "clear", (), {})
        Action.clear(agent)

    @staticmethod
    def advance_ball(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "advance_ball", (), {})
        Action.advance_ball(agent)

    @staticmethod
    def stop_ball(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "stop_ball", (), {})
        Action.stop_ball(agent)

    @staticmethod
    def go_to_point(agent, target, max_dash_power=100.0,
                    distance_threshold=None, dodge=False):
        if isinstance(agent, _Mirror):
            return _rec(agent, "go_to_point", (target,), {})
        Action.go_to_point(agent, target, max_dash_power, distance_threshold,
                           dodge)

    @staticmethod
    def intercept(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "intercept", (), {})
        Action.intercept(agent)

    @staticmethod
    def tackle(agent, direction=0.0, foul=False):
        if isinstance(agent, _Mirror):
            return _rec(agent, "tackle", (), {})
        Action.tackle(agent, direction, foul)

    @staticmethod
    def turn_to_point(agent, target):
        if isinstance(agent, _Mirror):
            return _rec(agent, "turn", (), {})
        Action.turn_to_point(agent, target)

    @staticmethod
    def turn_to_angle(agent, angle):
        if isinstance(agent, _Mirror):
            return _rec(agent, "turn", (), {})
        Action.turn_to_angle(agent, angle)

    @staticmethod
    def turn_to_ball(agent):
        if isinstance(agent, _Mirror):
            return _rec(agent, "turn", (), {})
        Action.turn_to_ball(agent)


def _home(player):
    if isinstance(player, _Mirror):
        return player.home
    return Tools.home_position(player)


def _note(agent, what, **fields):
    if not isinstance(agent, _Mirror):
        note(agent, what, **fields)


# ==========================================================================
# ENGINE: HeliosAgent (our bot, and an opponent model)
# ==========================================================================
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

def _keeper_margin(ph, bx, by, tx, ty, speed, gk, delay=0):
    """How far (m) the keeper stays from the ball's path towards (tx, ty):
    his distance to the nearest point of the path, minus his catch reach
    (1.3 m), minus what he runs (1 m/cycle) before the ball passes that
    point. Positive: he cannot get there. Fitted on recorded shots (see
    _gap_shot)."""
    if gk is None:
        return 99.0
    dist = math.hypot(tx - bx, ty - by)
    if dist < 1e-6:
        return 99.0
    ux, uy = (tx - bx) / dist, (ty - by) / dist
    t = max(0.0, min(dist, (gk.x - bx) * ux + (gk.y - by) * uy))
    lat = math.hypot(gk.x - (bx + ux * t), gk.y - (by + uy * t))
    n = ph.move_step(speed, t) if t > 0.0 else 0
    if n < 0:
        n = 60
    return lat - 1.3 - max(0, n + delay - 1) * 1.0


class _Helios:
    SAFE = 0        # extra cycles an opponent must be late by (meta mode)

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
            o_step += risk - self.SAFE
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
            if n_step - bonus - self.SAFE <= step:
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
            if n_step - bonus - self.SAFE <= n:
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
    def can_shoot_from(self, is_self, x, y, opps=None):
        ph, w = self.ph, self.w
        if math.hypot(ph.hl - x, y) > 17.0:
            return False
        thr = 20.0 if is_self else 15.0
        am = math.degrees(math.atan2(-ph.goal_half + 0.5 - y, ph.hl - x))
        ap = math.degrees(math.atan2(ph.goal_half - 0.5 - y, ph.hl - x))
        cands = []
        for o in (w.opps if opps is None else opps):
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
    def evaluate(self, holder_self, bx, by, t=None, opps=None):
        ph = self.ph
        if bx > ph.hl - 0.1 and abs(by) < ph.goal_half + 2.0:
            return 1.0e7
        if abs(bx) > ph.hl or abs(by) > ph.hw:
            return -1.0e9
        v = bx + max(0.0, 40.0 - math.hypot(ph.hl - bx, by))
        ops = self._opps(t) if opps is None else opps
        if self.can_shoot_from(holder_self, bx, by, ops):
            v += 1.0e6
            if holder_self:
                v += 5.0e5
        elif self.GAP_EVAL and self._gap_open(bx, by, ops):
            v += 5.0e5          # a _gap_shot will be there (lower bonus)
        return v

    GAP_EVAL = False         # value spots with a keeper-gap shot (_gap_shot)

    def _gap_open(self, x, y, opps):
        """Within 18 m, a direction into the goal the keeper cannot get to
        (_keeper_margin >= 0 at 2.7 m/cycle) with no field player within 8
        degrees of it nearer than the goal."""
        ph = self.ph
        d = math.hypot(ph.hl - x, y)
        if d > 18.0:
            return False
        gk = next((o for o in opps if o.goalie), None)
        lim = ph.goal_half - 0.8
        blk = []
        for o in opps:
            if o.goalie:
                continue
            if math.hypot(o.x - x, o.y - y) < d:
                blk.append(math.degrees(math.atan2(o.y - y, o.x - x)))
        for ty in (-lim, -lim / 2.0, 0.0, lim / 2.0, lim):
            if _keeper_margin(ph, x, y, ph.hl + 0.5, ty, 2.7, gk, 0) < 0.0:
                continue
            a = math.degrees(math.atan2(ty - y, ph.hl - x))
            if all(abs(_norm(a - b)) > 8.0 for b in blk):
                return True
        return False

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
                        offside, opps=None):
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
        for o in (w.opps if opps is None else opps):
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
        thr *= (2.5 / speed) * (1.0 + 0.2 * self.SAFE)
        return cone > thr

    def _deep_children(self, st):
        """Children of a predicted state (chain step >= 2)."""
        ph, w = self.ph, self.w
        holder, hx, hy, pos, olds, spend = st[:6]
        ctx = st[6:]
        out = []
        offside = max(w.offside, hx)
        opps = self._opps_st(st)
        # shoot (ActGen_Shoot)
        if self.can_shoot_from(holder == w.me.unum, hx, hy, opps):
            out.append((_Act("shoot", ph.hl, 0.0, holder, 1, 1, 1, ph.bmax,
                             "shoot", final=True),
                        (holder, ph.hl, 0.0, pos, olds, spend + 1) + ctx))
        if not self.deep:
            return out
        # direct passes
        for unum, (px, py, gk) in pos.items():
            if unum == holder or unum in olds:
                continue
            d = math.hypot(px - hx, py - hy)
            sp = self._pass_speed(d)
            if not self._simple_pass_ok(hx, hy, pos[holder][2], px, py, gk,
                                        sp, offside, opps):
                continue
            n = ph.move_step(sp, d)
            if n < 0:
                continue
            npos = dict(pos)
            npos[unum] = (px, py, gk)
            out.append((_Act("pass", px, py, unum, n + 2, n, 2, sp,
                             "directPass"),
                        (unum, px, py, npos, olds | {unum}, spend + n + 2) + ctx))
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
                for o in opps:
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
                            (holder, tx, ty, npos, olds, spend + hstep) + ctx))
        return out

    # ================= ActionChainGraph (best-first) ===================
    MAX_CHAIN = 4
    opp_at = None            # t -> predicted opponents (opponent model)
    root_ctx = None          # root action -> branch prediction, or None
    ROOT_CTX_K = 0           # ... for this many of the best roots
    EVAL_CAP = 20000         # chain evaluations per decision
    ORDERED_MIN = False      # move ordering in the interception MIN node
    TRANSPOSITION = False    # transposition table in the chain search

    def _opps_st(self, st):
        """Opponents at a chain state's time: from the prediction made
        for its branch (opponents re-deciding after our first action)
        when there is one, else the root prediction."""
        if len(st) > 6 and st[6] is not None:
            return st[6](int(st[5]))
        return self._opps(st[5])

    def _opps(self, t):
        """Opponents as they will be t cycles from now: predicted by the
        opponent model when one is attached, else where they stand."""
        if self.opp_at is None or t is None:
            return self.w.opps
        return self.opp_at(int(t))

    @staticmethod
    def _state_key(st):
        holder, hx, hy, pos, olds = st[:5]
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
        ctxs = {}
        if self.root_ctx is not None and self.ROOT_CTX_K > 0 and roots:
            rank = sorted(roots, key=lambda c: -self.evaluate(
                c.unum == me.unum, c.tx, c.ty, c.steps))
            for c in rank[:self.ROOT_CTX_K]:
                try:
                    self.tick()
                    ctxs[id(c)] = self.root_ctx(c)
                except _Timeout:
                    break
        for c in roots:
            npos = dict(pos0)
            if c.kind == "pass":
                npos[c.unum] = (c.tx, c.ty, pos0[c.unum][2])
            else:
                npos[me.unum] = (c.tx, c.ty, me.goalie)
            st = (c.unum, c.tx, c.ty, npos, frozenset((me.unum, c.unum)),
                  c.steps)
            if ctxs.get(id(c)) is not None:
                st = st + (ctxs[id(c)],)
            ev = self.evaluate(c.unum == me.unum, c.tx, c.ty, c.steps)
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
                    ev = self.evaluate(act.unum == me.unum, nst[1], nst[2],
                                       nst[5], self._opps_st(nst))
                    n_eval += 1
                    npath = path + (act,)
                    if ev > best_ev:
                        best_ev, best_path = ev, npath
                    tie += 1
                    heapq.heappush(heap, (-ev, tie, npath, nst))
                    if n_eval > self.EVAL_CAP:
                        raise _Timeout()
        except _Timeout:
            pass
        if best_path is None:
            return None, best_ev, n_eval
        return best_path[0], best_ev, n_eval


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
# Meta-reasoning on the score
#   The match result is what counts (win / draw / loss), not the goal
#   difference. A finite-horizon MDP: state (goal difference, time slices
#   left), action = playing mode, each mode scaling both teams' scoring
#   rates. Each slice either side may score (Poisson rates); the terminal
#   reward is the utility of the result. Solved by backward induction
#   (value iteration over the horizon). The rates are the posterior mean
#   of a Gamma prior updated by the goals of this match (Bayesian
#   updating of a Poisson rate).
# --------------------------------------------------------------------------

def _meta_policy(lead, left, elapsed, gf, ga, total, slice_, util, g0, kappa,
                 effect):
    """-> the mode maximising expected match utility now."""
    k = int(math.ceil(left / float(slice_)))
    if k <= 0:
        return "normal"
    prior = kappa * total               # prior exposure, cycles
    lu = (g0 * kappa + gf) / (prior + elapsed)    # goals per cycle, us
    lt = (g0 * kappa + ga) / (prior + elapsed)    # ... them
    D = 6
    n = 2 * D + 1
    V = [util[0] if d > 0 else util[1] if d == 0 else util[2]
         for d in range(-D, D + 1)]
    probs = []
    for a, (mu, mt) in effect:
        pu = 1.0 - math.exp(-lu * mu * slice_)
        pt = 1.0 - math.exp(-lt * mt * slice_)
        probs.append((a, pu * (1.0 - pt), pt * (1.0 - pu),
                      pu * pt + (1.0 - pu) * (1.0 - pt)))
    pol = None
    for _ in range(k):
        nv, pol = [], []
        for i in range(n):
            up, dn = V[min(i + 1, n - 1)], V[max(i - 1, 0)]
            best_q, best_a = None, None
            for a, p_up, p_dn, p_eq in probs:      # "normal" listed first:
                q = p_up * up + p_dn * dn + p_eq * V[i]   # kept on ties
                if best_q is None or q > best_q + 1e-9:
                    best_q, best_a = q, a
            nv.append(best_q)
            pol.append(best_a)
        V = nv
    return pol[max(0, min(n - 1, lead + D))]


# --------------------------------------------------------------------------
# The team
# --------------------------------------------------------------------------

class HeliosAgent(StarterAgent):
    SPEND_FRACTION = 0.68      # of the total match budget
    DEFAULT_BUDGET_MS = 75000.0
    ON_BALL_SHARE = 12.0       # an on-ball decision gets ~12 cycles' worth
    OFF_BALL_SHARE = 0.8
    MAX_ON_MS = 420.0
    MAX_OFF_MS = 80.0
    ON_WALL_MARGIN = 250.0     # wall ms kept free at the end of an on-ball decision

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
    G2D_OPEN_SPOT = False      # v1.4 forwards to open spots (+0.1/g)
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
    EVAL_CAP = 20000           # chain evaluations per decision
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
    GAP_SHOT = False           # shoot past the keeper's side (_gap_shot)
    GAP_SHOT_DIST = 25.0
    GAP_MARGIN = 0.0           # keeper reach margin needed (m)
    GAP_EVAL = False           # the planner values spots with such a shot
    PRESS = False              # nearest of us goes at a carrier in our half
    PRESS_X = 0.0
    # meta-reasoning on the score (_meta_policy): protect a lead, chase a
    # deficit. Modes: "protect" = safer passes / dribbles (one cycle more
    # margin), defence deeper, clear in danger, shield it at the corner
    # flag late; "attack" = risky through passes, defenders push up, forwards
    # to open spots, press harder, shoot past an advanced keeper.
    META = False
    META_FORCE = None          # a fixed mode (calibration runs), or "score":
                               # attack when behind, protect when ahead
    META_SLICE = 50            # MDP time step, cycles
    META_UTIL = (3.0, 1.0, 0.0)            # win, draw, loss
    META_GOALS = 2.0           # Gamma prior mean, goals per match each side
    META_PRIOR_W = 0.5         # ... worth this many matches
    # (scoring-rate factor us, them) per mode, measured against FULL with
    # the mode forced for whole matches (24 / 23 / 24 matches): normal
    # 2.04 for, 1.92 against; protect 1.29 / 1.79; attack 1.57 / 2.04.
    # Both modes cost goal difference, so the policy keeps "normal" except
    # to protect a lead in the last 50-150 cycles.
    META_EFFECT = (("normal", (1.0, 1.0)), ("protect", (0.63, 0.93)),
                   ("attack", (0.77, 1.07)))

    def __init__(self):
        super().__init__()
        self._mode = "normal"
        self._meta = (-1, "normal")
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

    def _meta_mode(self, agent):
        """Playing mode for this cycle (see _meta_policy)."""
        if self.META_FORCE and self.META_FORCE != "score":
            return self.META_FORCE
        if not self.META and self.META_FORCE != "score":
            return "normal"
        wm = agent.wm
        try:
            cyc = int(wm.cycle)
            gf, ga = int(wm.our_team_score), int(wm.their_team_score)
        except Exception:
            return "normal"
        if self.META_FORCE == "score":
            # plain rule: chase when behind, protect when ahead
            return "attack" if gf < ga else "protect" if gf > ga else "normal"
        if self._meta[0] != cyc:
            total = self._match_cycles(agent)
            mode = _meta_policy(gf - ga, max(0, total - cyc), cyc, gf, ga,
                                total, self.META_SLICE, self.META_UTIL,
                                self.META_GOALS, self.META_PRIOR_W,
                                self.META_EFFECT)
            if mode != self._meta[1]:
                _note(agent, "meta", mode=mode, gf=gf, ga=ga, cyc=cyc)
            self._meta = (cyc, mode)
        return self._meta[1]

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
            _note(agent, "ERR_on", err=type(e).__name__ + ":" + str(e)[:60])
        _A.hold(agent)
        _A.clear(agent)

    def _on_ball(self, agent):
        t0 = time.process_time()
        ph = self._phys(agent)
        w = _World(agent)
        cpu, wall = self._deadlines(agent, self._allowance_ms(agent, True),
                                    self.ON_WALL_MARGIN)
        self._mode = mode = self._meta_mode(agent)
        hp = _Helios(ph, w, cpu, wall, self.DEEP_CHAIN, self.USE_CROSS,
                     self.GRANULARITY, self.G2D_RISKY or mode == "attack")
        hp.SAFE = 1 if mode == "protect" else 0
        hp.GAP_EVAL = self.GAP_EVAL
        hp.MAX_CHAIN = self.CHAIN_DEPTH
        hp.ORDERED_MIN = self.ORDERED_MIN
        hp.TRANSPOSITION = self.TRANSPOSITION
        hp.EVAL_CAP = self.EVAL_CAP
        self._prep_planner(agent, ph, w, hp)

        # 1. Bhv_StrictCheckShoot
        try:
            shots = hp.shoots()
        except _Timeout:
            shots = []
        if shots:
            tx, ty, speed, ang, kick1, gnr, onr = shots[0]
            one = ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy)
            _note(agent, "shoot", speed=round(speed, 2), one=round(one, 2))
            if one > speed * 0.99:
                _A.kick_one_step(agent, Vector2D(tx, ty))
            else:
                n = Tools.steps_required(speed, math.hypot(tx - w.bx,
                                                           ty - w.by),
                                         ph.decay, ph.bmax) or 1
                _A.kick(agent, Vector2D(tx, ty), n, accept_late=1)
            return True
        if self._extra_shot(agent):
            return True
        if self.GAP_SHOT and self._gap_shot(agent, ph, w):
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
            _note(agent, "hold", n=n, ms=ms)
            if w.bx < -ph.hl + 8.0 and abs(w.by) < ph.goal_half + 1.0:
                _A.clear(agent)
            else:
                _A.hold(agent)
            return True
        _note(agent, best.desc, x=round(best.tx, 1), y=round(best.ty, 1),
             n=n, ms=ms)
        # pass: smart kick at the planned pace. dribble / self pass: one
        # kick to the trap point, then the off-ball intercept runs onto it
        # (Bhv_NormalDribble kicks, then turns and dashes onto the ball).
        _A.kick(agent, Vector2D(best.tx, best.ty), max(1, best.flight),
                    accept_late=1)
        dist = math.hypot(best.tx - w.bx, best.ty - w.by)
        self._last_kick = (w.cycle, best.kind, best.tx, best.ty,
                           min(ph.bmax, ph.first_speed(dist, max(1, best.flight))))
        return True

    def _carrier_hint(self, agent):
        """Hook: a model of the opponent ball carrier's next action, as
        {'p_keep': P(dribble / hold), 'keep_dir': degrees, 'recv': {unum:
        P(pass to him)}}, or None (no model: HELIOS behaviour)."""
        return None

    def _prep_planner(self, agent, ph, w, hp):
        """Hook: attach an opponent model to the planner (hp.opp_at)."""

    def _extra_shot(self, agent):
        """Hook: other shot models, tried when the strict shoot finds none."""
        return False

    # ------------------------------------------------------------------
    # Football heuristics (each behind its own switch)
    # ------------------------------------------------------------------
    def _heur_pre(self, agent, ph, w):
        mode = self._mode
        if ((self.H_KEEPER_OUT or mode == "attack")
                and self._keeper_out_shot(agent, ph, w)):
            return True
        if ((self.H_CLEAR_DANGER or mode == "protect")
                and self._clear_danger(agent, ph, w)):
            return True
        if ((self.H_PROTECT_LEAD or mode == "protect")
                and self._protect_lead(agent, ph, w)):
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

    def _gap_shot(self, agent, ph, w):
        """Shoot where the keeper cannot get to. Fitted on 83 recordings
        (443 shots against the framework's own keeper): with the keeper's
        reach margin (_keeper_margin) >= 0 a shot scored 91% (235/258), at
        any distance; -2..0 58%, below -2 34%. HELIOS's strict check
        assumes the keeper reaches every ball he could in theory, so it
        rarely fires. Here: inside GAP_SHOT_DIST, the target in the goal
        mouth with the largest margin (at least GAP_MARGIN), with no field
        player able to reach the ball first."""
        if math.hypot(ph.hl - w.bx, w.by) > self.GAP_SHOT_DIST:
            return False
        gk = next((o for o in w.opps if o.goalie), None)
        field = [o for o in w.opps if not o.goalie]
        near = min((math.hypot(o.x - w.bx, o.y - w.by) for o in field),
                   default=99.0)
        lim = ph.goal_half - 0.8
        tx = ph.hl + 0.5
        best = None
        for k in range(41):
            ty = -lim + 2.0 * lim * k / 40.0
            ang = math.degrees(math.atan2(ty - w.by, tx - w.bx))
            v1 = ph.max_vel(ang, w.kick_rate, w.bvx, w.bvy)
            if v1 >= 2.2:
                speed, steps = min(v1, ph.bmax), 1
            elif near > 4.0:
                speed, steps = 2.6, 2           # wind up: nobody close
            else:
                continue
            m = _keeper_margin(ph, w.bx, w.by, tx, ty, speed, gk, steps - 1)
            if m < self.GAP_MARGIN:
                continue
            if not self._lane_clear(ph, w, field, tx, ty, speed, steps - 1):
                continue
            if best is None or m > best[0]:
                best = (m, ty, steps, speed)
        if best is None:
            return False
        m, ty, steps, speed = best
        _note(agent, "gap_shot", d=round(math.hypot(ph.hl - w.bx, w.by), 1),
             margin=round(m, 2), v=round(speed, 2), steps=steps)
        if steps == 1:
            _A.kick_one_step(agent, Vector2D(tx, ty))
        else:
            n = Tools.steps_required(speed, math.hypot(tx - w.bx, ty - w.by),
                                     ph.decay, ph.bmax) or 1
            _A.kick(agent, Vector2D(tx, ty), n, accept_late=1)
        return True

    def _lane_clear(self, ph, w, field, tx, ty, speed, delay):
        """No field opponent reaches a ball kicked at `speed` towards
        (tx, ty) before it crosses the goal line."""
        d = math.hypot(tx - w.bx, ty - w.by)
        ux, uy = (tx - w.bx) / d, (ty - w.by) / d
        for c in range(1, 30):
            bx, by = ph.ball_at(w.bx, w.by, ux * speed, uy * speed, c)
            for o in field:
                if ph.cycles_to_reach(math.hypot(bx - o.x, by - o.y)
                                      - ph.kickable) <= c + delay:
                    return False
            if bx >= ph.hl:
                return True
        return False

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
        _note(agent, "h_keeper_out", gx=round(gk.x, 1))
        _A.kick_one_step(agent, Vector2D(ph.hl + 0.5, best))
        return True

    def _clear_danger(self, agent, ph, w):
        """In our own box with an attacker on us: get rid of it."""
        if w.bx > -ph.hl + 18.0 or abs(w.by) > ph.pen_hw + 2.0:
            return False
        near = min((math.hypot(o.x - w.bx, o.y - w.by) for o in w.opps
                    if not o.goalie), default=99.0)
        if near > 3.5:
            return False
        _note(agent, "h_clear", near=round(near, 1))
        _A.clear(agent)
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
        _note(agent, "h_protect", near=round(near, 1))
        if near < 2.0 or math.hypot(tx - w.bx, ty - w.by) < 2.0:
            _A.hold(agent)
        else:
            _A.dribble(agent, Vector2D(tx, ty))
        _A.hold(agent)
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
        _note(agent, "h_cutback", x=round(best.tx, 1), y=round(best.ty, 1))
        _A.kick(agent, Vector2D(best.tx, best.ty), max(1, best.flight),
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
        _note(agent, "h_wing_run", x=round(tx, 1))
        _A.dribble(agent, Vector2D(tx, ty))
        _A.hold(agent)
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
                    _note(agent, "h_give_go")
                    return tx, ty, 100.0
        if self.H_COUNTER and role >= 9:
            back = sum(1 for o in w.opps if not o.goalie and o.x > w.bx)
            if back <= 2 and w.bx < ph.hl - 25.0:
                tx = min(w.offside - 1.0, ph.hl - 8.0)
                ty = max(-12.0, min(12.0, me.y))
                _note(agent, "h_counter", back=back)
                return tx, ty, 100.0
        if self.H_FAR_POST and role >= 9:
            if w.bx > 20.0 and abs(w.by) > 10.0:
                s = 1.0 if w.by > 0.0 else -1.0
                tx = min(ph.hl - 5.0, w.offside - 0.5)
                _note(agent, "h_far_post")
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
            _note(agent, "turnForward")
            _A.turn_to_point(agent, Vector2D(fx, me.y * 0.5))
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
            _note(agent, "ERR_off", err=type(e).__name__ + ":" + str(e)[:60])
        _A.go_to_point(agent, _home(agent),
                           max_dash_power=self.dash_power(agent))
        _A.turn_to_ball(agent)

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
        self._mode = mode = self._meta_mode(agent)

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
        if mode == "attack":
            pressing = max(pressing, 6)
        # press: the nearest of us goes at a carrier in our half instead of
        # shadowing him (opponents kept the ball 84% of a match, losing it
        # in open play once or twice, against the block + marking alone)
        if (self.PRESS and wm.kickable_opponent_existance
                and w.bx < self.PRESS_X and self_min <= mate_min):
            _note(agent, "press_carrier", steps=self_min)
            _A.intercept(agent)
            _A.go_to_point(agent, Vector2D(w.bx, w.by), 100.0, 0.3)
            return True
        # intercept
        if not wm.kickable_teammate_existance and (
                self_min <= 3
                or (self_min <= mate_min and self_min < opp_min + pressing)):
            _note(agent, "intercept", steps=self_min)
            _A.intercept(agent)
            _A.go_to_point(agent, Vector2D(w.bx, w.by), 100.0, 0.5)
            return True

        hx, hy = _xy(_home(agent))
        team = [m for m in w.mates if not m.goalie] + [me]
        our_ball = wm.kickable_teammate_existance or mate_min < opp_min
        if ((self.G2D_DEEP_FORM or mode == "protect") and not our_ball
                and role <= 8):
            shift = 6.0 if role <= 5 else 4.0
            hx = max(-ph.hl + 4.0, hx - shift)
        if mode == "attack" and our_ball and role <= 8:
            hx = max(hx, min(w.offside - 1.0, hx + (6.0 if role <= 5 else 4.0)))
        if ((self.G2D_OPEN_SPOT or mode == "attack") and our_ball
                and role >= 9):
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
        hz = self._heur_off(agent, ph, w, role, our_ball)
        if hz is not None:
            _A.go_to_point(agent, Vector2D(hz[0], hz[1]), hz[2], 1.0)
            _A.turn_to_ball(agent)
            return True

        power = self._normal_dash_power(agent, ph, w, self_min, mate_min,
                                        opp_min)
        thr = max(1.0, math.hypot(w.bx - me.x, w.by - me.y) * 0.1)
        _note(agent, "position")
        _A.go_to_point(agent, Vector2D(hx, hy), power, thr)
        _A.turn_to_ball(agent)
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
        if self.TACKLE_NEAR_GOAL or self.PRESS:
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
        _note(agent, "tackle", prob=round(tp, 2), foul=use_foul)
        _A.tackle(agent, _norm(best_a - me.body), use_foul)
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
        hint = self._carrier_hint(agent)
        keep_dir = None
        if hint is not None and hint.get("p_keep", 0.0) >= 0.5:
            keep_dir = hint.get("keep_dir")
        x, y = ipx, ipy
        target, who = None, None
        for cyc in range(opp_min + 1, opp_min + 41):
            if keep_dir is not None and cyc - opp_min <= 12:
                dd = math.radians(keep_dir)
            else:
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
        _note(agent, "block")
        _A.go_to_point(agent, Vector2D(*target), 100.0, 0.5)
        _A.turn_to_point(agent, Vector2D(ipx, ipy))
        return True

    # ---- Gliders2d v1.4: forwards to open spots ------------------------------
    def _open_spot(self, agent, ph, w, hx, hy):
        """Gliders2d v1.4 sends the forward to the most open space ahead.
        Exhaustive search (generate-and-test) over a 6 x 7 grid of points
        near the formation slot and up to the offside line, scored by the
        distance to the nearest opponent, depth and distance from the slot."""
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
        hint = self._carrier_hint(agent)
        recv = hint.get("recv") if hint is not None else None
        threats = []
        for o in opps:
            if o.goalie or o.unum == holder.unum:
                continue
            gd = math.hypot(o.x - gx, o.y)
            if gd > 45.0:
                continue
            danger = -o.x + max(0.0, 40.0 - gd)
            if recv:
                danger += 25.0 * recv.get(o.unum, 0.0)
            threats.append((danger, o))
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
            px = o.x + gdx / gd * 1.5 + bdx / bd * 0.7
            py = o.y + gdy / gd * 1.5 + bdy / bd * 0.7
            pr = min(1.0, recv.get(o.unum, 0.0)) if recv else 0.0
            if pr >= 0.4 and bd > 4.0:
                # likely receiver: stand in the passing lane, 2.5 m in
                # front of him towards the ball
                lx = o.x + bdx / bd * 2.5
                ly = o.y + bdy / bd * 2.5
                px, py = px + (lx - px) * pr, py + (ly - py) * pr
            pts.append((px, py))
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
        _note(agent, "mark", opp=threats[j][1].unum)
        _A.go_to_point(agent, Vector2D(px, py), 100.0, 0.6)
        _A.turn_to_ball(agent)
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
                _note(agent, "unmark_keep")
                _A.go_to_point(agent, Vector2D(tx, ty), 100.0, 1.0)
                _A.turn_to_ball(agent)
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
        _note(agent, "unmark")
        _A.go_to_point(agent, Vector2D(*best), 100.0, 1.0)
        _A.turn_to_ball(agent)
        return True


# STARTER CODE ENDS HERE #


# ==========================================================================
# ENGINE: ColAgent (opponent model)
# ==========================================================================
#
# COL333 A3 - Agent1 (v31c: compute only)
#
# Big picture (see COL333_A3_ideas.txt for the reasoning behind every choice):
#
#   * A small "physics model" of our own: how many cycles a player needs to
#     reach a point (reach), and where a kicked ball is after t cycles.
#     All search runs on this model, because no forward simulator is exposed.
#
#   * kick_eval(): follow a kicked ball cycle by cycle and compare, for every
#     defender, "when can he get there" with "when is the ball there".
#     The smallest gap (margin, in cycles) becomes a success probability.
#     This is the MIN layer of our minimax: the opponents' best interception.
#
#   * on_ball: depth-1 scoring of shots / passes / dribbles / hold with
#     alpha-beta style cutoffs, a greedy shortcut, then depth-2 (receiver's
#     best next action) on the best few passes while time allows.
#
#   * off_ball: every player computes the SAME team plan from the same exact
#     world state (no communication allowed), so they agree on who presses,
#     who marks whom, who supports, who runs.

# =============================================================================
# Tunable parameters (all in one place so tuning by matches is easy)
# =============================================================================
#
# QUICK TUNING GUIDE  (what to change if you see ...)
#   too safe / sideways / backwards ......... W_PROG up  (or BACK_PASS_PEN up)
#   play drifts to the wings ................ W_CENTER up, W_EDGE up
#   too few shots ........................... SHOOT_NOW_P down, GOAL_VALUE up
#                                             (keep GOAL_VALUE >> H_MAX ~ 17)
#   shots from hopeless positions ........... SHOOT_NOW_P up, SHOT_RANGE down
#   too many / too risky passes ............. PASS_COST up
#   too few dribbles ........................ DRIB_M0 down (dribbles look safer)
#   players stand still with the ball ....... HOLD_COST up, MAX_HOLDS down
#   too few tackles ......................... TACKLE_MIN / MARKER_TACKLE_MIN down,
#                                             FOUL_MAX_* up
#   too many fouls / cards .................. FOUL_MAX_* down
#   markers too far / too close to their man  MARK_GAP, MARK_GAP_MIN
#   attackers stray from their positions .... ROLE_W up (OPEN_W down)
#   attackers not finding space ............. OPEN_W up, ANGLE_SPREAD up
#   decisions too slow (time budget) ........ MC_TIME_MS down, MC_MAX down
#   (v22) holder dribbles back towards our goal  RETREAT_PEN up
#   (v22) self-passes (kick ahead + run) lost .. SELFPASS_MIN_PRESS up, or SELFPASS = False
#   (v22) lead passes intercepted ............ LEAD_PASS_R smaller (e.g. (4.0, 7.0))
#   (v22) forwards too far from the holder ... FWD_MIN_AHEAD / WING_MIN_AHEAD down
#   (v22) anchor too deep when we attack ..... ANCHOR_MAX_BEHIND down
#   (v22) markers too deep / too goal-side ... MARK_LEAD down
#   (v22) passes still tackled in the wind-up  ONE_STEP_PRESS up (4)
#   (v24) still too risky with the ball ...... KEEP_BALL up (1.5), RISKY_P up (0.7)
#   (v24) now too timid / never goes forward . KEEP_BALL down (0.5), SAFE_P up (0.9)
#   (v25) online learning misbehaves ......... LEARN = False (or LEARN_ADJ_PRIOR down)
#   (v26) dense options too slow / too wild .. DENSE_ACTIONS = False, or fewer
#                                              GRID_DIRS / GRID_SPEEDS
#   (v26) look-ahead options all too similar . DUP_D up (3.0); too odd: DUP_D down
# After changing a number, delete __pycache__ if a quick re-run behaves the
# same (python may reuse an old compiled copy of an edited file).

MATCH_CYCLES = 1500

# --- movement model
MATE_SPEED = 1.0          # m/cycle, our players (approx. player_speed_max)
OPP_SPEED = 1.05          # slightly pessimistic for opponents
SPEED_PER_POWER = 0.0105  # steady speed per unit dash power (1.05 at 100)
CTRL_R = 1.0              # distance at which a receiver controls the ball
PLAYER_R = 1.1            # interception reach of a field player
TACKLE_R = 1.4            # reach when the ball is at someone's feet (tackle)
GOALIE_R = 1.8            # goalie can catch with hands inside his box (passes)
SHOT_GOALIE_R = 1.3       # ... but vs a fast shot his real catch reach is ~1.2 m
SHOT_REACT = 1            # defenders/goalie need a cycle to react to a shot
PEN_LEN = 16.5
PEN_HW = 20.16

# --- margin (cycles) -> probability curves: p = sigmoid(K * (margin - M0))
P_CAP = 0.97
PASS_M0, PASS_K = 1.0, 1.2
DRIB_M0, DRIB_K = 0.0, 1.5
HOLD_M0, HOLD_K = 2.0, 1.0
SHOT_M0, SHOT_K = 0.0, 1.3   # goalie arriving with the ball = 50/50

# --- evaluation H(state) and Loss
W_PROG = 15.0             # progress up the pitch (x): 5 m forward ~ +0.48
W_CENTER = 1.2            # ball in the middle strip, away from the touchlines
W_CLOSE = 1.4             # closeness to their goal centre (captures angle)
W_SHOTQ = 6.0             # geometric shooting chance from the point
W_CLEAR = 0.4             # clearance: how many cycles until an opponent arrives
CLEAR_CAP = 8.0           # more space than this adds nothing
SHOTQ_RANGE = 25.0
H_MAX = W_PROG + W_CLOSE + W_SHOTQ + W_CLEAR + W_CENTER
L_BASE = 0.35             # losing the ball always costs this much...
L_K = 1.6                 # ...plus more the closer it is to OUR goal

# --- shooting
GOAL_VALUE = 40.0         # must stay well above H_MAX (best position value)
SHOT_MISS = -0.5          # missed shot ~ goal kick: we lose the ball, far from our goal
SHOOT_NOW_P = 0.20        # shoot immediately above this goal probability
                          # (v28: with the fitted shot model below; was 0.20)
# --- v28 shot model FITTED on 162 real shots (peer tournament + our logs,
# same automatic goalie for every team): logistic regression (MLE)
#   P(goal) = sigmoid(SHOT_FIT_B0 + SHOT_FIT_DIST * distance
#                     + SHOT_FIT_GK * min(goalie's offset from the shot line, 8))
# raw rates: < 16 m and goalie > 5 m off the line 89 %, 16-20 m 61 %,
# > 20 m 33 %. Our old margin model said 18 m -> 0.18, 22 m -> 0.01.
SHOT_FIT = False
# --- v29 driven passes: a firm ball (about FIRM_ARRIVE m/cycle when it gets
# there) spends less time in the air; the search decides when it is worth it.
FIRM_PASS = False
BEHIND_JOB_W = 0.0        # v29 defence assignment: extra cost per metre a defender
                          # is behind (further from our goal than) his job
FIRM_ARRIVE = 2.0
FAST_RECV = 2.2           # arriving faster than this is hard to control ...
FAST_RECV_P = 1.0         # ... success x this
SHOT_FIT_B0, SHOT_FIT_DIST, SHOT_FIT_GK = 0.877, -0.154, 0.386
SHOT_RANGE = 35.0
ONE_STEP_SPEED = 2.4      # realistic speed of a single-touch kick
SHOT_POST_MARGIN = 0.8    # aim this far inside the posts (kick noise)

# --- search
GAMMA = 0.9               # discount on the depth-2 part (less reliable)
HOLD_COST = 0.3           # holding gains nothing and lets defenders close in
HOLD_FRAC = 0.3           # ... and costs this SHARE of the position value too
                          # (v9 used 40 %: it held only ~6 % of the time and scored)
                          # (fixed cost, so it does not grow with W_PROG)
GREEDY_P = 0.80           # greedy shortcut: forward pass this safe... (v24: was 0.70)
GREEDY_GAP = 0.4          # ...and this much better than the runner-up
CLEAR_VALUE = -0.9        # below this, clear (if under pressure / own third)
THROUGH_R = 7.0
ALLOW_THROUGH = True      # set False if through passes keep being offside
PASS_MIN, PASS_MAX = 3.0, 35.0
DRIB_SPEED = 0.8          # m/cycle while dribbling (log: ~1.0); 0.7 was far too slow
DRIB_TACKLE_R = 1.5       # opponent must get this close to the ball on a dribble
                          # (v22: tackle_dist is 2.0 m, 1.2 was too optimistic)
DRIB_FWD_W = 0.5          # EXTRA value per 10 m a dribble gains towards their goal
DRIB_GAMMA = 1.0          # (no extra penalty: dribbles now get depth 2 too)
OPEN_PRESSURE = 6         # "nobody close" = nearest opponent >= this many cycles
PASS_COST = 0.2           # every pass has execution risk (first touch, turn)
BACK_PASS_PEN = 1.0       # backward pass/dribble when NOT under pressure (no ping-pong)
# --- anti ping-pong (v21): logs showed ~30 A->B->A passes per match and a
#     median pass gaining 0 m; passes must EARN their cost when not pressed
PASS_MIN_GAIN = 5.0       # a pass should move the ball at least this far forward ...
NOGAIN_PEN = 1.2          # ... else up to this penalty (scaled by how little it gains)
RETURN_PEN = 2.5          # passing straight back to whoever just passed to me
RETURN_WINDOW = 30        # ... within this many cycles
RETURN_OK_GAIN = 8.0      # ... is fine if it gains this much (a real one-two)
PRESSED = 2               # "under pressure" = opponent within this many cycles:
                          # then none of these penalties apply
NO_OPTION_P = 0.6         # holder has no pass above this -> attackers push up
AHEAD_SPRINT = 5.0        # sprint when the target is this far ahead of me

# --- ball near the touchline / goal line is bad (goes out easily)
EDGE_ZONE = 4.0
W_EDGE = 0.6
# --- stalling
HOLD_STALL = 0.25         # each consecutive hold makes holding this much worse
MAX_HOLDS = 2             # after this many holds in a row: must do something
ECO_MIN_MARGIN = 6        # economical (slow) chase only if this many cycles ahead
                          # AND the ball is nearly still

# --- small candidate set + Monte Carlo (v4)
PASS_RING_R = 2.5         # pass target = freest point within this of the receiver
DRIB_SHORT = 5.0
DRIB_ANGLES = (0.0, 25.0, -25.0, 50.0, -50.0, 80.0, -80.0, 135.0, -135.0, 180.0)
                          # dribble directions, degrees from the ball->goal line
ESCAPE_D = 3.0            # under pressure: also short escape dribbles of this length
ESCAPE_PRESSURE = 5       # "under pressure" = an opponent within this many cycles
DRIB_LONG1 = 10.0
MC_DEPTH = 4              # actions per chain (root + 3 follow-ups)   (v31: was 3)
MC_TIME_MS = 350.0        # Monte Carlo time per on-ball decision (v31: was 80; also capped by budget)
MC_MAX = 3000             # max rollouts per decision (v31: was 400)
MC_C = 1.0                # UCB1 exploration constant
MC_PRIOR_W = 3.0          # deterministic chain value counts as this many samples
MC_TIE = 0.15             # keep the teammates' predicted choice unless MC beats it by this
MC_SIGMA = 1.0            # execution noise on margins (cycles)
MC_POS_NOISE = 0.4        # opponent position noise (m)
MC_SPEED_NOISE = 0.15     # opponent speed noise (+-15 %)
MC_KICK_DEG = 2.5         # kick direction noise (degrees)
MC_DRIB_OFF = 0.5
COMMIT_P = 0.5            # keep a committed action while its success stays above this
COMMIT_PASS = 3           # cycles to stick with a pass (multi-touch kick)
COMMIT_DRIB = 8           # cycles to stick with a dribble

# --- defending
TACKLE_MIN = 0.30         # presser tackles at this win probability (a tackle
                          # freezes the player 10 cycles, so not on a whim)
MARKER_TACKLE_MIN = 0.3   # any other defender close enough tackles at this
FOUL_MAX = 1.0            # max foul chance, middle / their third (no limit)
FOUL_MAX_THIRD = 1.0      # ... in our own third (no limit: tackling the ball
                          #     carrier is almost always "foul likely" here)
FOUL_MAX_BOX = 0.5        # ... in our own penalty box (fouls = free kicks there)
CLOSE_DOWN = 4.0          # presser this close: go straight at the ball
MARK_MARGIN = 3.0         # mark only opponents goal-side of the ball (+ this margin)
BEHIND_PEN = 4            # presser cost (cycles) for coming from BEHIND the carrier
SANDWICH_STEPS = 3        # a 2nd player behind the carrier this close also presses
FRONT_MAX_STEPS = 10      # goal-side player presses if he arrives within this
SCREEN_D = 8.0            # screens stand this far from the ball, on ball->post lines
COVER_DIST = 14.0         # cover player stands this far out from our goal
COUNTER_MARGIN = 2        # we win the ball by >= this -> start counter run
HYST = 6.0                # metres: keep old assignment unless clearly worse
PRESS_HYST = 2            # cycles: keep the old presser unless clearly worse
MARK_STICKY = 0.3         # danger bonus for opponents we were already marking
MARK_GAP = 3.0            # marker: at most this far in front of his man (ball side)
MARK_GAP_MIN = 2.0        # ... and at least this far (room to react)
DANGER_ZONE = 25.0        # anyone this close to our goal is always marked

# --- attacking positioning
SUP_RADII = (7.0, 9.0)
SUP_ANGLES = (90.0, 120.0, 150.0, -90.0, -120.0, -150.0)
RUN_AHEAD = 20.0
SPACING = 8.0
COMPACT = 25.0            # attackers stay within this many metres behind ball
PASS_MOVE_CYCLES = 12
# --- attacking shape (v11): deep supporter / last defender + two wide forwards
ANCHOR_BEHIND = 14.0      # anchor stays this far behind the ball (goal-side)
ANCHOR_MAX_X = 12.0       # ... and never further up than this
WING_Y = 15.0             # near wing: this far to the side of the middle (was 18)
WING_AHEAD = 25.0         # wide forwards at most this far ahead of the ball
MAX_SUPPORT_DIST = 30.0   # NO attacker further than this from the holder ...
REACH_P = 0.30            # ... AND a pass from the holder must reach him there with
                          # at least this probability (else he comes closer)
REACH_MIN_D = 12.0         # ... but he never comes closer than this for it
BRK_CLEAR = 10.0          # breakaway: no opponent within this distance in front ...
BRK_CONE = 40.0           # ... inside this cone (deg) around the line to their goal
BRK_LEAD = 8.0            # teammates position as if the holder were this far ahead
BRK_AHEAD = 12.0          # ... and go at most this far ahead of that point
BRK_PASS_AHEAD = 8.0      # ... unless a teammate this much further forward ...
BRK_PASS_GAP = 0.3        # ... can get a very safe pass worth this much more
WING_PASS_MIN = 0.6       # ... and only as far as a pass still reaches them
SHOW_R = 9.0              # holder has no option: nearest teammate "shows" for the
                          # ball at about this distance from him
DEEP_D = 18.0             # defending: second line this far in front of the ball
DEEP_W = 8.0              # ... two players this far either side of the ball->goal line
PASS_W = 1.5              # get-open: weight of "a pass to me here would work"
WING_SPACING = 12.0       # get-open: keep this far from teammates and the holder
THROUGH_AHEAD = 8.0       # through ball: this far into the space behind their line
# --- defending a dribbler (v11)
BLOCK_D = 2.0             # presser stands this far in front of the carrier, on
                          # his line to our goal, facing him
PATH_D = 8.0              # 2nd defender: on the carrier's line to goal this far ahead
CARRIER_SPEED = 0.9       # m/cycle a dribbler covers (for recovery runs)
# --- stamina (v11)
SPRINT_MIN = 0.45         # sprint (100) only above this stamina fraction
OPEN_W = 1.0              # get-open: weight of free space around the point
GOALWARD_W = 1.5          # get-open: bonus for being ahead of the holder (per 20 m)
OPEN_REACH_P = 0.6        # get-open: a spot only counts if a pass reaches it with this P
OPEN_UNREACH_PEN = 3.0    # ... (unreachable spots get this penalty = only a fallback)
ANGLE_SPREAD = 40.0       # get-open: teammates within this angle (seen from the
                          # holder) are "in the same direction" -> penalty
HOLD_MEMORY = 5           # cycles a dribbling teammate keeps "his" ball
ROLE_W = 0.6              # get-open: cost per 10 m away from the role spot

# --- v22 (match vs claude_light: lost 1-3 with 60 % of the ball, 1 shot)
RUN_GAIN = 2.5            # a running player's reported velocity is AFTER the
                          # 0.4 decay: he really moves ~2.5x that per cycle
MARK_LEAD = 3.0           # markers aim where their man will be this many cycles on
ONE_STEP_PRESS = 3        # opponent within this many cycles: pass in ONE kick
                          # (a wound-up multi-kick pass got tackled 4x in the log)
RETREAT_PEN = 0.5         # backward dribble in our own half, even under pressure (v24: 0.8)
OWN_THIRD_BACK_PEN = 3.0  # ... and in our own third (= never, except to escape)
SELFPASS = True           # kick the ball ahead into space and run onto it
SELFPASS_L = (8.0, 13.0)  # ... this far
SELFPASS_ANGLES = (0.0, 25.0, -25.0, 50.0, -50.0)
SELFPASS_MIN_PRESS = 5    # ... only if no opponent is within this many cycles (v24: 3)
WING_MIN_AHEAD = 2.0      # wing never drops further back than ball x + this (v24: 4)
FWD_MIN_AHEAD = 5.0       # forward never drops further back than ball x + this (v24: 8)
ANCHOR_GUARD = 3.0        # anchor stays this far goal-side of their deepest forward
ANCHOR_MAX_BEHIND = 28.0  # ... but never more than this far behind the ball
LEAD_PASS_R = (5.0, 9.0)  # lead passes: into the space this far ahead of a mate
LEAD_PASS_ANG = (0.0, 40.0, -40.0)
CARRIER_HEAD_W = 0.6      # presser: carrier's real running direction vs. "to goal"
                          # (v27: 0.6 -> 0.85, FULL's wingers run straight down
                          # the touchline and the blend aimed 3-5 m inside them)
PRESS_CUT_D = 12.0        # v27: presser further than this cuts the path; closer
                          # he chases a point BLOCK_D in front of the carrier
# --- v23 (match vs FULL: every kick-off lost within 10 cycles)
# --- v24 (user: "too uncautious with the ball, keep it like the older versions")
KEEP_BALL = 1.0           # EXTRA cost of losing the ball (added to Loss): every
                          # risky option looks worse, keeping it looks better
SAFE_P = 0.85             # if some pass / dribble / hold is at least this safe ...
RISKY_P = 0.6             # ... passes / dribbles below this are not even considered
SELFPASS_MIN_P = 0.9      # self-pass (kick ahead + run) only when this safe
MC_RISK_TIE = 0.5         # Monte Carlo may pick a riskier action than the safest
                          # good one only if it is worth this much more
RUN_PASSES = ("lead", "through", "behind", "cutback")
# --- v25 ONLINE LEARNING during the match (ISIS-style: correct the model's
# own parameters from its prediction errors against THIS opponent)
LEARN = True              # set False to switch all online learning off
LEARN_ADJ_STEP = 0.25     # grid of candidate margin offsets (MAP by enumeration)
LEARN_ADJ_PRIOR = 1.0     # Gaussian prior on the offset (std, cycles)
LEARN_ADJ_MIN, LEARN_ADJ_MAX = -0.5, 3.0   # learned margin offsets stay in here
                          # (asymmetric on purpose: it may get a lot more careful,
                          # only a little bolder - v24 asked for caution)
LEARN_PASS_SPEED = 1.3    # ball leaving our feet faster than this = a pass
LEARN_SPEED_Q = 0.95      # opponent speed = this quantile of observed running
LEARN_SPEED_BIN = 0.05    # ... from a Laplace-smoothed histogram (MLE) of it
LEARN_SPEED_MIN, LEARN_SPEED_MAX = 0.95, 1.15
LEARN_CARRIER_K0 = 30.0   # prior strength (observations) of their dribbling speed
LEARN_NOTE_EVERY = 100    # write the learned values into the log this often
# --- v26 DENSE ACTIONS: the holder scores (almost) every kick and dribble
DENSE_ACTIONS = True      # set False to go back to the hand-picked options (v25)
GRID_DIRS = 36            # kicks in this many directions (v31: every 10 deg, was 15) ...
GRID_SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0)   # ... at these ball speeds (v31: 5, was 4)
GRID_TMAX = 40            # follow each kicked ball for at most this many cycles (v31: was 30)
DENSE_DRIB_DIRS = 24      # dribbles in this many directions (v31: every 15 deg, was 22.5)
TOP_K = 14                # options handed to the look-ahead + Monte Carlo (v31: was 10)
# v31: the FOLLOW-UP actions of a chain (depth 2..4) also try a coarse kick grid
NEXT_GRID_DIRS = 12       # every 30 deg ...
NEXT_GRID_SPEEDS = (1.6, 2.3, 3.0)
NEXT_GRID_TMAX = 30
# --- v27 tricks
BOX_ATTACK_X = 999.0       # holder this far up the pitch -> box runs (cutback /
                          # far post) replace the wing / forward spots
CUTBACK_SPOT = (11.0, 1.5)   # cutback zone: this far in front of their goal line,
                          # this far to the far side of the centre (penalty spot)
FARPOST_SPOT = (5.0, 4.5)    # far-post run: this far out, this far across
PROTECT_LEAD_CYCLE = 99999 # leading after this cycle: keep the ball, run the clock
PROTECT_KEEP = 1.5        # ... extra cost of losing the ball while protecting
# --- v31 tricks
BAIT = False               # goalkeeper bait: in / near their box with the goalie
BAIT_DIST = 20.0          # coming out (holder this close to goal, goalie within
BAIT_GK_D = 9.0           # this of the ball) -> square balls across the goal
FEINT = False              # sudden stop (CYRUS feint): an opponent sprinting at us
FEINT_D = 5.0             # (within this, real speed above FEINT_V) overruns if we
FEINT_V = 0.6             # stop the ball; value = holding now + the best dribble
FEINT_CARRY = 1.5         # in the world where he carried on this many cycles
DECOY = False              # decoy run: a free attacker runs at the defender who
DECOY_R = 9.0             # guards the predicted receiver's spot (within this)
CORNER_SHIELD = False      # protecting a lead in their half: carry it to the corner
DUP_D = 2.0               # look-ahead options ending this close (same kind and
                          # receiver) count as duplicates
                          # passes the receiver must RUN onto:
                          # judged on the ball reaching the TARGET (not on the
                          # receiver touching it on the way) and on him getting there

# --- time management
ONBALL_MULT, ONBALL_CAP = 12.0, 450.0   # v31: was 8 / 300
OFFBALL_MULT, OFFBALL_CAP = 1.0, 30.0   # v31: was 40 (wall-clock safety)
SAFETY_MS = 300.0         # v31: was 150 (a busy machine stretches wall time)
DEPTH2_MIN_MS = 25.0
QUICK_MIN_MS = 4.0
QUICK_REMAIN_MS = 1500.0  # below this match budget: cheap starter behaviour
QUICK_DEC_MS = 80.0       # below this wall time left in a decision: same


# =============================================================================
# Small helpers
# =============================================================================

def _deg(a):
    """Angle (AngleDeg or number) -> float degrees."""
    if a is None:
        return 0.0
    try:
        return float(a.degree())
    except AttributeError:
        pass
    try:
        return float(a)
    except (TypeError, ValueError):
        return 0.0


def _cxy(v):
    if v is None:
        return 0.0, 0.0
    return float(v.x()), float(v.y())


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def _unit(dx, dy):
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return 1.0, 0.0
    return dx / d, dy / d


def _sig(m, m0, k):
    z = k * (m - m0)
    if z > 30.0:
        return P_CAP
    if z < -30.0:
        return 0.0
    return min(P_CAP, 1.0 / (1.0 + math.exp(-z)))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _err_label(where, e):
    """Error type + function + line (in agent.py) packed into the label, so
    `python3 a3.py why` shows exactly what broke."""
    fn, line, last = "?", "?", "?"
    try:
        tb = e.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_filename.endswith("agent.py"):
                fn, line = code.co_name, tb.tb_lineno
            last = code.co_name
            tb = tb.tb_next
    except Exception:
        pass
    msg = "".join(ch if ch.isalnum() else "_" for ch in str(e)[:40])
    return "ERR_%s_%s_in_%s_L%s_last_%s_%s" % (where, type(e).__name__, fn, line, last, msg)


_GCACHE = {}


def _geo(decay, n=80):
    """G[t] = 1 + d + ... + d^(t-1): distance factor of a ball after t moves."""
    key = round(decay, 6)
    g = _GCACHE.get(key)
    if g is None:
        g = [0.0]
        p = 1.0
        for _ in range(n):
            g.append(g[-1] + p)
            p *= decay
        _GCACHE[key] = g
    return g


def arrive_steps(dist, v, decay):
    """Cycles for a ball kicked at speed v to cover dist (None if never)."""
    r = dist * (1.0 - decay) / v
    if r >= 1.0:
        return None
    return max(1, int(math.ceil(math.log(1.0 - r) / math.log(decay))))


def first_speed(dist, n, decay, vmax):
    return min(vmax, dist * (1.0 - decay) / (1.0 - decay ** n))


def kick_delay(v0):
    """kick() may wind up over several touches for a fast ball."""
    if v0 <= 1.5:
        return 0
    if v0 <= 2.4:
        return 1
    return 2


# =============================================================================
# World snapshot (plain floats: much faster than Vector2D in tight loops)
# =============================================================================

class Body:
    __slots__ = ("unum", "x", "y", "vx", "vy", "body", "goalie", "speed",
                 "side", "key")

    def __init__(self, unum, x, y, vx, vy, body, goalie, speed, side, key):
        self.unum = unum
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.body = body
        self.goalie = goalie
        self.speed = speed
        self.side = side          # -1: our player (goal at -x), +1: opponent
        self.key = key


def _body_from(p, speed, side, key):
    x, y = _cxy(p.position)
    vx, vy = _cxy(getattr(p, "velocity", None))
    return Body(int(getattr(p, "uniform_number", 0) or 0), x, y, vx, vy,
                _deg(getattr(p, "body_direction", 0.0)),
                bool(getattr(p, "is_goalie", False)), speed, side, key)


class Snap:
    """Everything one decision needs, read once from agent.wm."""

    def __init__(self, agent):
        wm = agent.wm
        sp = agent.server_params
        self.cycle = int(getattr(wm, "cycle", 0) or 0)
        self.hl = float(getattr(sp, "pitch_half_length", 52.5))
        self.hw = float(getattr(sp, "pitch_half_width", 34.0))
        self.goal_hw = float(getattr(sp, "goal_width", 14.02)) / 2.0
        self.decay = float(getattr(sp, "ball_decay", 0.94))
        self.vmax = float(getattr(sp, "ball_speed_max", 3.0))
        self.stamina_max = float(getattr(sp, "stamina_max", 8000.0)) or 8000.0
        self.G = _geo(self.decay)

        self.me = _body_from(wm.self, MATE_SPEED, -1, ("m", wm.self.uniform_number))
        self.mates = []
        for tm in wm.teammates:
            u = int(getattr(tm, "uniform_number", 0) or 0)
            if u <= 0 or tm.position is None:
                continue
            self.mates.append(_body_from(tm, MATE_SPEED, -1, ("m", u)))
        self.field_mates = [b for b in self.mates if not b.goalie]
        # our field players including me, sorted by uniform number
        self.team = sorted(self.field_mates + ([self.me] if not self.me.goalie else []),
                           key=lambda b: b.unum)
        self.ours_all = self.mates + [self.me]
        self.opps = []
        for i, o in enumerate(wm.opponents):
            if o.position is None:
                continue
            u = int(getattr(o, "uniform_number", 0) or 0)
            self.opps.append(_body_from(o, OPP_SPEED, +1, ("o", u if u > 0 else 100 + i)))

        self.bx, self.by = _cxy(wm.ball.position)
        self.bvx, self.bvy = _cxy(getattr(wm.ball, "velocity", None))
        self.offside = float(getattr(wm, "offside_line_x", self.hl) or 0.0)
        try:
            self.lead = int(getattr(wm, "our_team_score", 0) or 0) - int(
                getattr(wm, "their_team_score", 0) or 0)
        except Exception:
            self.lead = 0
        self.our_line = float(getattr(wm, "our_defense_line_x", -self.hl) or 0.0)
        self._ball_path = None

    def ball_at(self, t):
        g = self.G[min(t, len(self.G) - 1)]
        return self.bx + self.bvx * g, self.by + self.bvy * g

    def in_field(self, x, y, margin=1.0):
        return abs(x) < self.hl - margin and abs(y) < self.hw - margin


# =============================================================================
# The model: reach times, kick simulation, evaluation
# =============================================================================

def reach(b, tx, ty, radius):
    """Cycles player b needs to get within `radius` of (tx, ty).

    turn cycles (if facing away) + acceleration loss + distance / speed.
    Crude but cheap, and the SAME for every player of ours, which matters for
    coordination."""
    px = b.x + b.vx
    py = b.y + b.vy
    dx = tx - px
    dy = ty - py
    dist = math.hypot(dx, dy)
    d = dist - radius
    if d <= 0.0:
        return 0
    diff = abs(_norm180(math.degrees(math.atan2(dy, dx)) - b.body))
    if diff < 25.0:
        turn = 0
    elif diff < 100.0 or (b.vx * b.vx + b.vy * b.vy) < 0.09:
        turn = 1
    else:
        turn = 2
    along = (b.vx * dx + b.vy * dy) / dist
    acc = 0.7 * max(0.0, 1.0 - along / b.speed)
    return turn + int(math.ceil((d + acc) / b.speed))


def radius_of(b, hl):
    if b.goalie and b.side * b.x > hl - PEN_LEN and abs(b.y) < PEN_HW:
        return GOALIE_R
    return PLAYER_R


def ball_reach_steps(b, snap, radius, tmax=40):
    """First cycle t at which b can be at the (free-rolling) ball."""
    for t in range(tmax + 1):
        cx, cy = snap.ball_at(t)
        if reach(b, cx, cy, radius) <= t:
            return t
    return tmax + 10


def kick_eval(snap, sx, sy, tx, ty, v0, delay, receiver, defenders,
              tmax=30, mode="pass", cut=None):
    """Simulate a kick from (sx, sy) towards (tx, ty).

    mode 'pass'    : receiver is a Body; ends when he can reach the ball.
    mode 'virtual' : receiver stands at the target; ends when ball gets there.
    mode 'shot'    : ends when the ball crosses the goal line x = hl.

    Returns (margin, t_end, ix, iy, end_margin, ex, ey) or None if the ball
    never reaches the receiver / goal (or leaves the pitch first).
      margin     : min over defenders and cycles of (defender reach - t).
                   >0 means everybody is late, <0 means intercepted.
      (ix, iy)   : where the most dangerous interception happens.
      end_margin : how late the defenders are at the receive point
                   (= clearance of the receiver).
    cut(margin) -> True stops checking further defenders (alpha-beta cutoff).
    """
    dx = tx - sx
    dy = ty - sy
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L
    G = snap.G
    hl, hw = snap.hl, snap.hw
    path = []
    t_end = None
    for t in range(1, tmax + 1):
        k = t - delay
        if k <= 0:
            x, y = sx, sy
        else:
            s = v0 * G[min(k, len(G) - 1)]
            x, y = sx + ux * s, sy + uy * s
        path.append((x, y))
        if mode == "shot":
            if x >= hl:
                t_end = t
                break
        else:
            if abs(x) > hl or abs(y) > hw:
                return None
            if mode == "virtual":
                if k > 0 and math.hypot(tx - x, ty - y) <= CTRL_R:
                    t_end = t
                    break
            elif k > 0 and reach(receiver, x, y, CTRL_R) <= t:
                t_end = t
                break
    if t_end is None:
        return None
    ex, ey = path[t_end - 1]

    ordered = sorted(defenders, key=lambda d: _seg_dist(d.x, d.y, sx, sy, ex, ey))
    margin = 99.0
    end_m = 99.0
    ix, iy = ex, ey
    react = SHOT_REACT if mode == "shot" else 0
    for d in ordered:
        r = radius_of(d, hl)
        if mode == "shot" and r > SHOT_GOALIE_R:
            r = SHOT_GOALIE_R
        # circle pruning: too far from the whole ball path to matter
        if _seg_dist(d.x, d.y, sx, sy, ex, ey) - r > d.speed * t_end + 3.0:
            break      # list is sorted, everyone after is even further
        for t in range(1, t_end + 1):
            x, y = path[t - 1]
            m = reach(d, x, y, r) + react - t
            if m < margin:
                margin = m
                ix, iy = x, y
            if t == t_end and m < end_m:
                end_m = m
            if m <= -4:
                break
        if cut is not None and cut(margin):
            break
    return margin, t_end, ix, iy, end_m, ex, ey


def shot_quality(snap, x, y):
    """Geometric 0..1 shooting chance: open goal angle and distance."""
    dgx = snap.hl - x
    if dgx <= 0.0:
        return 0.0
    dg = math.hypot(dgx, y)
    if dg > SHOTQ_RANGE:
        return 0.0
    ang = abs(math.atan2(snap.goal_hw - y, dgx) - math.atan2(-snap.goal_hw - y, dgx))
    return min(1.0, ang / 0.9) * (1.0 - dg / SHOTQ_RANGE)


def H(snap, x, y, clear):
    """Value of having the ball at (x, y) with `clear` cycles of space."""
    hl = snap.hl
    prog = (x + hl) / (2.0 * hl)
    close = max(0.0, 1.0 - math.hypot(hl - x, y) / 60.0)
    c = min(max(clear, 0.0), CLEAR_CAP) / CLEAR_CAP
    # distance to the nearest line the ball could go out over (the goal mouth
    # itself does not count)
    edge = snap.hw - abs(y)
    if not (x > 0.0 and abs(y) < snap.goal_hw):
        edge = min(edge, hl - abs(x))
    edge_pen = W_EDGE * max(0.0, (EDGE_ZONE - edge) / EDGE_ZONE)
    center = 1.0 - (min(abs(y), snap.hw) / snap.hw) ** 2
    return (W_PROG * prog + W_CLOSE * close + W_CENTER * center
            + W_SHOTQ * shot_quality(snap, x, y) + W_CLEAR * c - edge_pen)


_KEEP_EXTRA = [0.0]       # v27: set per decision (protecting a lead)


def Loss(snap, x):
    """Value of losing the ball at x: worse the closer to OUR goal."""
    prog = (x + snap.hl) / (2.0 * snap.hl)
    return -(L_BASE + KEEP_BALL + _KEEP_EXTRA[0] + L_K * (1.0 - prog))


def offside_line_of(opps, ball_x):
    """Offside line from a (predicted) opponent set, in our frame."""
    xs = sorted((o.x for o in opps), reverse=True)
    line = xs[1] if len(xs) >= 2 else (xs[0] if xs else 0.0)
    return max(line, ball_x, 0.0)


def predict_opps(snap, rx, ry, T, frac=0.6):
    """Where opponents will roughly be after T cycles: moved towards the
    receive point by part of what they could run (kept simple on purpose:
    HELIOS found bad predictions make deeper search worse)."""
    out = []
    for o in snap.opps:
        dx, dy = rx - o.x, ry - o.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            out.append(o)
            continue
        move = min(max(0.0, d - 2.0), o.speed * T * frac)
        out.append(Body(o.unum, o.x + dx / d * move, o.y + dy / d * move, 0.0, 0.0,
                        math.degrees(math.atan2(dy, dx)), o.goalie, o.speed,
                        o.side, o.key))
    return out


def run_vel(b):
    """Real per-cycle displacement of a running player (reported velocity is
    after the 0.4 player decay; a player who keeps dashing moves ~2.5x it)."""
    sp = math.hypot(b.vx, b.vy)
    if sp < 0.08:
        return 0.0, 0.0
    k = min(1.05, sp * RUN_GAIN) / sp
    return b.vx * k, b.vy * k


def _moving_to(b, x, y):
    """True if b is already running towards (x, y) (velocity points there)."""
    sp = math.hypot(b.vx, b.vy)
    if sp < 0.3:
        return False
    ux, uy = _unit(x - b.x, y - b.y)
    return (b.vx * ux + b.vy * uy) / sp > 0.8


def _pressure_of(opps, x, y):
    return min((reach(o, x, y, TACKLE_R) for o in opps), default=99)


def _pressure(snap, x, y):
    return min((reach(o, x, y, TACKLE_R) for o in snap.opps), default=99)


# =============================================================================
# ColAgent
# =============================================================================

class ColAgent(StarterAgent):
    def __init__(self):
        super().__init__()
        # Per-player memory across cycles. The framework's memory object is
        # NOT a dict (no [] or .get), but `self` persists for each player
        # (one process per player), so state is kept here instead.
        self.mem = {}
        self._deadline = 0.0
        # v25 learned during the match (start = the hand-set model)
        self.L = dict(pass_adj=0.0, drib_adj=0.0, opp_speed=OPP_SPEED,
                      carrier=CARRIER_SPEED, n_pass=0, n_lost=0, n_drib=0, n_drib_lost=0)
        # Bayesian / MLE state behind those numbers
        k = int(round((LEARN_ADJ_MAX - LEARN_ADJ_MIN) / LEARN_ADJ_STEP))
        self._adj_grid = [LEARN_ADJ_MIN + LEARN_ADJ_STEP * i for i in range(k + 1)]
        prior = [-0.5 * (g / LEARN_ADJ_PRIOR) ** 2 for g in self._adj_grid]
        self._lp_pass = list(prior)          # log posterior over the offset grid
        self._lp_drib = list(prior)
        self._spd_bins = [0] * int(round((1.6 - 0.3) / LEARN_SPEED_BIN))
        self._car_sum, self._car_n = 0.0, 0

    def _pm0(self):
        """Pass success curve offset: hand-set PASS_M0 + what was learned."""
        return PASS_M0 + self.L["pass_adj"]

    def _dm0(self):
        return DRIB_M0 + self.L["drib_adj"]

    # ------------------------------------------------------------ v25 learning
    def _learn(self, agent, snap, i_have):
        """Online learning, every cycle, from what this player observes.
          1. PASSES (ours, by anyone): when the ball leaves one of our players'
             feet fast, our model predicts its interception margin from the
             real ball path; when somebody controls it we know the outcome.
             The offset of the success curve is a MAP estimate over a grid
             of candidate offsets (Bayesian parameter estimation by
             enumeration: Gaussian prior + the log-likelihood of every
             observed outcome): passes lost more often than predicted ->
             more careful, and the other way round. Every player learns from ALL our passes, so the
             four players end up with (almost) the same value and keep
             predicting each other's choices (HFO: shared experience helps).
          2. DRIBBLES (only the dribbler knows his prediction): same update.
          3. OPPONENT SPEED: the 95% quantile of a Laplace-smoothed
             histogram of their observed running speed (MLE).
          4. THEIR DRIBBLING SPEED: posterior mean of their ball speed while
             they carry it (conjugate prior worth LEARN_CARRIER_K0 samples).
        Then the opponents in this snapshot get the learned speed."""
        if not LEARN:
            return
        L, mem = self.L, self.mem
        wm = agent.wm
        cyc = snap.cycle
        mate_kick = bool(getattr(wm, "kickable_teammate_existance", False))
        opp_kick = bool(getattr(wm, "kickable_opponent_existance", False))
        ours = i_have or mate_kick
        try:
            # --- 3. opponent running speed (smoothed histogram quantile)
            prev = mem.get("opp_prev", {})
            cur = {}
            for o in snap.opps:
                cur[o.key] = (o.x, o.y, cyc)
                q = prev.get(o.key)
                if o.goalie or q is None or q[2] != cyc - 1:
                    continue
                d = math.hypot(o.x - q[0], o.y - q[1])
                if 0.3 < d < 1.6:
                    b = min(len(self._spd_bins) - 1, int((d - 0.3) / LEARN_SPEED_BIN))
                    self._spd_bins[b] += 1
                    L["opp_speed"] = min(LEARN_SPEED_MAX, max(LEARN_SPEED_MIN,
                                         self._speed_quantile()))
            mem["opp_prev"] = cur
            # --- 4. their dribbling speed (ball speed while they carry it)
            pb = mem.get("ball_prev")
            if pb is not None and pb[2] == cyc - 1 and opp_kick and pb[3]:
                d = math.hypot(snap.bx - pb[0], snap.by - pb[1])
                if d < 1.6:
                    self._car_sum += min(1.05, 1.15 * d)
                    self._car_n += 1
                    mean = ((LEARN_CARRIER_K0 * CARRIER_SPEED + self._car_sum)
                            / (LEARN_CARRIER_K0 + self._car_n))
                    L["carrier"] = min(1.05, max(0.5, mean))
            mem["ball_prev"] = (snap.bx, snap.by, cyc, opp_kick)
            # --- 1. passes: resolve a pending one
            ev = mem.get("pass_ev")
            if ev is not None and cyc > ev[0]:
                y = None
                if opp_kick and not ours:
                    y = 0.0
                elif ours and cyc - ev[0] >= 2:
                    y = 1.0
                elif cyc - ev[0] > 40:
                    mem["pass_ev"] = None
                if y is not None:
                    L["pass_adj"] = self._map_update(self._lp_pass, ev[1], y,
                                                     PASS_M0, PASS_K)
                    L["n_pass"] += 1
                    L["n_lost"] += int(y == 0.0)
                    mem["pass_ev"] = None
            # --- 1. passes: detect a new one (ball just left our feet, fast)
            sp = math.hypot(snap.bvx, snap.bvy)
            if ours or opp_kick:
                mem["feet"] = cyc if ours and not opp_kick else -99
            elif (mem.get("pass_ev") is None and cyc - mem.get("feet", -99) <= 2
                  and sp >= LEARN_PASS_SPEED and not self._is_shot(snap)):
                # the prediction made BEFORE the kick (by the passer himself,
                # or by me predicting him) if the ball went that way: that is
                # the number our decisions are made with, so that is the one
                # to calibrate
                dec = mem.get("dec")
                if dec is not None and cyc - dec[0] <= 4:
                    da = abs(_norm180(math.degrees(math.atan2(snap.bvy, snap.bvx)
                                                   - math.atan2(dec[2] - dec[5], dec[1] - dec[4]))))
                    if da < 25.0:
                        mem["pass_ev"] = (cyc, dec[3])
                mem["feet"] = -99
            # --- 2. my own dribble: resolve
            dv = mem.get("drib_ev")
            if dv is not None and cyc > dv[0]:
                y = None
                if opp_kick and not i_have:
                    y = 0.0
                elif cyc >= dv[0] + dv[2] + 1 and ours:
                    y = 1.0
                elif cyc > dv[0] + dv[2] + 8:
                    mem["drib_ev"] = None
                if y is not None:
                    L["drib_adj"] = self._map_update(self._lp_drib, dv[1], y,
                                                     DRIB_M0, DRIB_K)
                    L["n_drib"] += 1
                    L["n_drib_lost"] += int(y == 0.0)
                    mem["drib_ev"] = None
        except Exception as e:          # learning must never break a decision
            _note(agent, _err_label("learn", e))
        for o in snap.opps:
            if not o.goalie:
                o.speed = L["opp_speed"]

    def _map_update(self, lp, margin, y, m0, k):
        """Bayesian update of the log posterior over the offset grid with
        one observed outcome (y = 1 success, 0 lost); -> the MAP offset."""
        for i, g in enumerate(self._adj_grid):
            p = min(0.98, max(0.02, _sig(margin, m0 + g, k)))
            lp[i] += math.log(p if y >= 0.5 else 1.0 - p)
        return self._adj_grid[max(range(len(lp)), key=lambda i: lp[i])]

    def _speed_quantile(self):
        """LEARN_SPEED_Q quantile of the running-speed histogram with
        add-one (Laplace) smoothing."""
        counts = [c + 1.0 for c in self._spd_bins]
        total, acc = sum(counts), 0.0
        for i, c in enumerate(counts):
            acc += c
            if acc >= LEARN_SPEED_Q * total:
                return 0.3 + LEARN_SPEED_BIN * (i + 1)
        return 1.6

    def _learn_note(self, agent):
        """Every LEARN_NOTE_EVERY cycles the learned values become this
        player's label (visible in `a3.py why`): pass/drib offsets in tenths
        of a cycle (+ = more careful), speeds in cm per cycle."""
        try:
            cyc = int(getattr(agent.wm, "cycle", 0) or 0)
        except Exception:
            return
        if not LEARN or cyc % LEARN_NOTE_EVERY != 0:
            return
        L = self.L
        _note(agent, "learn_pass%+d_drib%+d_spd%d_car%d" % (
            round(10 * L["pass_adj"]), round(10 * L["drib_adj"]),
            round(100 * L["opp_speed"]), round(100 * L["carrier"])),
             passes=L["n_pass"], lost=L["n_lost"], dribbles=L["n_drib"],
             drib_lost=L["n_drib_lost"])

    def _is_shot(self, snap):
        if snap.bvx <= 0.5 or snap.bx < 15.0:
            return False
        t = (snap.hl - snap.bx) / snap.bvx
        return abs(snap.by + snap.bvy * t) < snap.goal_hw + 2.0

    # ------------------------------------------------------------------ time
    def _budget(self, agent):
        try:
            return float(agent.budget.remaining_ms), float(agent.budget.decision_remaining_ms)
        except Exception:
            return 1e9, 1e9

    def _allowance(self, agent, snap, on_ball):
        """ms this decision may use: a share of what is left for the match."""
        try:
            rem = float(agent.budget.remaining_ms)
            dec = float(agent.budget.decision_remaining_ms)
        except Exception:
            return 20.0
        cycles_left = max(60, MATCH_CYCLES - snap.cycle)
        avg = rem / cycles_left
        if on_ball:
            a = min(ONBALL_MULT * avg, ONBALL_CAP)
        else:
            a = min(OFFBALL_MULT * avg, OFFBALL_CAP)
        return min(a, dec - SAFETY_MS)

    # =================================================================
    # ON BALL
    # =================================================================
    def on_ball(self, agent) -> None:
        try:
            self._on_ball(agent)
        except Exception as e:      # never crash a player: fall back
            _note(agent, _err_label("on", e))
            StarterAgent.on_ball(self, agent)
        self._learn_note(agent)

    def _on_ball(self, agent):
        t0 = time.perf_counter()
        snap = Snap(agent)
        self._learn(agent, snap, True)
        # v27 game management: leading late -> losing the ball costs more,
        # holding / shielding costs nothing (runs the clock)
        self.protect = snap.lead > 0 and snap.cycle >= PROTECT_LEAD_CYCLE
        _KEEP_EXTRA[0] = PROTECT_KEEP if self.protect else 0.0
        self.mem["last_poss"] = "us"
        prev = self.mem.get("our_holder")
        if (prev is not None and prev[0] != snap.me.unum
                and snap.cycle - prev[1] <= RETURN_WINDOW):
            self.mem["received_from"] = (prev[0], snap.cycle)      # he passed to me
        self.mem["our_holder"] = (snap.me.unum, snap.cycle)
        rf = self.mem.get("received_from")
        passer_unum = rf[0] if rf is not None and snap.cycle - rf[1] <= RETURN_WINDOW else None
        rem, dec = self._budget(agent)
        if rem < QUICK_REMAIN_MS or dec < QUICK_DEC_MS:
            _note(agent, "quick_on_ball")
            StarterAgent.on_ball(self, agent)
            return
        allow = max(self._allowance(agent, snap, True), QUICK_MIN_MS)
        deadline = t0 + allow / 1000.0
        self._deadline = deadline
        bx, by = snap.bx, snap.by
        me = snap.me
        opps = snap.opps
        pressure = _pressure(snap, bx, by)

        # ---- 0. commitment: keep doing what we decided while it stays safe
        #      (re-deciding every cycle restarts multi-touch kicks = "slow")
        if self._continue_plan(agent, snap, pressure):
            return

        # ---- 1. shots first ----------------------------------------------
        best_shot = None
        if math.hypot(snap.hl - bx, by) < SHOT_RANGE:
            for c in self._shots(snap, bx, by, opps, pressure):
                if best_shot is None or c["v"] > best_shot["v"]:
                    best_shot = c
            if best_shot is not None and best_shot["p"] >= SHOOT_NOW_P:
                _note(agent, "shoot_now", p=round(best_shot["p"], 2))
                self._execute(agent, snap, best_shot, pressure)
                return

        # ---- 1b. breakaway: clear run ahead -> carry the ball forward ------
        if self._breakaway(snap, bx, by):
            fwd = [self._eval_dribble(snap, bx, by, d, opps)
                   for d in self._dribble_dirs(snap, bx, by, opps, pressure,
                                               names=("g+0", "g+25", "g-25"))]
            fwd = [d for d in fwd if d["p"] >= 0.8]
            better_pass = self._better_forward_pass(snap, bx, by, me, opps, pressure,
                                                    max((d["v"] for d in fwd), default=-1e9))
            if fwd and better_pass is None:
                b = max(fwd, key=lambda d: d["v"])
                _note(agent, "breakaway_dribble", p=round(b["p"], 2))
                self._execute(agent, snap, b, pressure)
                return

        # ---- 2. small candidate set with deterministic depth-1 values -----
        cands = self._candidates(snap, bx, by, me, snap.field_mates, snap.offside, opps, pressure,
                                 passer_unum)
        if best_shot is not None:
            cands.append(best_shot)
        holds = self.mem.get("holds", 0) if self.mem.get("hold_cycle") == snap.cycle - 1 else 0
        if holds:
            cands = [c for c in cands if c.get("sub") != "feint"]   # v31: one stop, then go
        if holds < MAX_HOLDS or not cands:
            hc = self._eval_hold(snap, bx, by, pressure)
            hc["v"] -= HOLD_STALL * holds          # stalling gets worse every cycle
            hc["stall"] = HOLD_STALL * holds
            cands.append(hc)
        cands = self._risk_filter(cands)
        cands.sort(key=lambda c: c["v"], reverse=True)

        # ---- 3. greedy shortcut (user's idea) ------------------------------
        b0 = cands[0]
        gap = b0["v"] - (cands[1]["v"] if len(cands) > 1 else -1e9)
        if (b0["kind"] == "pass" and b0["tx"] > bx + 1.0
                and b0["p"] >= GREEDY_P and gap >= GREEDY_GAP):
            _note(agent, "pass_greedy", to=b0["recv"].unum, p=round(b0["p"], 2))
            self._execute(agent, snap, b0, pressure)
            return

        # ---- 4. depth-3 chains (deterministic) as priors -------------------
        #      v26: only the TOP_K most valuable AND most different options
        #      (the depth-1 best is always the first one)
        if DENSE_ACTIONS:
            cands = self._diverse_top(cands, TOP_K)
        roots = []
        for c in cands:
            if time.perf_counter() > deadline:
                break
            if c["kind"] in ("pass", "dribble") and c["p"] >= 0.15:
                chain = self._chain(snap, c, deadline)
            else:
                chain = [c]
            roots.append(dict(c=c, chain=chain,
                              prior=self._chain_value(snap, chain) - c.get("stall", 0.0)))
        if not roots:
            roots = [dict(c=c, chain=[c], prior=c["v"]) for c in cands]

        # ---- 5. Monte Carlo evaluation with UCB1 over the roots -----------
        mc_deadline = min(deadline, time.perf_counter() + MC_TIME_MS / 1000.0)
        rng = random.Random(snap.cycle * 131 + me.unum)
        means, nroll = self._mc_select(snap, roots, mc_deadline, rng, pressure)
        bi = max(range(len(roots)), key=lambda i: means[i])
        # coordination: teammates predict the depth-1 best; prefer it unless
        # Monte Carlo says something is clearly better
        di = max(range(len(roots)), key=lambda i: roots[i]["c"]["v"])
        tie = MC_TIE
        if roots[bi]["c"].get("p", 1.0) < roots[di]["c"].get("p", 1.0) - 0.1:
            tie += MC_RISK_TIE        # v24: a RISKIER plan must win clearly
        if means[di] >= means[bi] - tie:
            bi = di
        best = roots[bi]["c"]
        bv = means[bi]

        # ---- 6. fallback / execute ----------------------------------------
        if bv < CLEAR_VALUE - KEEP_BALL and (pressure <= 2 or bx < -snap.hl * 0.4):
            _note(agent, "clear", v=round(bv, 2))
            self.mem["plan"] = None
            _A.clear(agent)
            return
        _note(agent, best["kind"] + "_" + best.get("sub", best["kind"]) + "_mc",
             v=round(bv, 2), n=nroll, depth=len(roots[bi]["chain"]))
        self._execute(agent, snap, best, pressure)

    def _continue_plan(self, agent, snap, pressure):
        plan = self.mem.get("plan")
        if not plan:
            return False
        age = snap.cycle - plan["cycle"]
        if age <= 0 or age > plan["commit"]:
            return False
        bx, by = snap.bx, snap.by
        if plan["kind"] == "pass":
            recv = next((m for m in snap.field_mates if m.unum == plan["recv"]), None)
            if recv is None:
                return False
            c = dict(kind="pass", sub=plan.get("sub", "commit"), tx=plan["tx"], ty=plan["ty"], n=plan["n"],
                     v0=plan["v0"], delay=0, recv=recv, one=plan.get("one", False))
            if pressure <= ONE_STEP_PRESS and not c["one"]:
                # v22: he is on us now - finish it with one kick, no wind-up
                c["v0"] = min(c["v0"], ONE_STEP_SPEED) if c["v0"] < ONE_STEP_SPEED else ONE_STEP_SPEED
                c["one"] = True
            r = self._eval_pass(snap, bx, by, c, snap.opps, -1e9)
            if r is None or r["p"] < COMMIT_P:
                return False
            _note(agent, "commit_pass", p=round(r["p"], 2))
            self._execute(agent, snap, r, pressure, keep_plan=True)
            return True
        if plan["kind"] == "dribble":
            D = math.hypot(plan["tx"] - bx, plan["ty"] - by)
            if D < 1.5:
                return False
            c = dict(kind="dribble", sub="commit", tx=plan["tx"], ty=plan["ty"], D=D,
                     ang=math.degrees(math.atan2(plan["ty"] - by, plan["tx"] - bx)))
            r = self._eval_dribble(snap, bx, by, c, snap.opps)
            if r["p"] < COMMIT_P:
                return False
            _note(agent, "commit_dribble", p=round(r["p"], 2))
            self._execute(agent, snap, r, pressure, keep_plan=True)
            return True
        return False

    # ------------------------------------------------------------ execute
    def _execute(self, agent, snap, c, pressure=99, keep_plan=False):
        k = c["kind"]
        tgt = Vector2D(c.get("tx", snap.bx), c.get("ty", snap.by))
        if not keep_plan:
            self.mem["plan"] = None
        if k == "shot":
            if c["mode"] == "one":
                _A.kick_one_step(agent, tgt)
            else:
                _A.kick(agent, tgt, 1, accept_late=2)   # arrive_in=1 = max pace
            _A.kick_one_step(agent, tgt)
        elif k == "pass":
            self.mem["pm_until"] = snap.cycle + PASS_MOVE_CYCLES
            if "m" in c:
                self.mem["dec"] = (snap.cycle, c["tx"], c["ty"], c["m"], snap.bx, snap.by)
            if not keep_plan:
                self.mem["plan"] = dict(kind="pass", tx=c["tx"], ty=c["ty"], n=c["n"], v0=c["v0"],
                                        recv=c["recv"].unum, cycle=snap.cycle,
                                        commit=COMMIT_PASS, one=c.get("one", False),
                                        sub=c.get("sub", "commit"))
            if c.get("one") and c["v0"] >= ONE_STEP_SPEED - 0.05:
                _A.kick_one_step(agent, tgt)      # v22: no wind-up when pressed
            elif c.get("one"):
                _A.kick(agent, tgt, c["n"], accept_late=1)   # slow = one touch
            elif pressure <= 1:
                _A.kick_one_step(agent, tgt)      # no time for a wind-up
            else:
                _A.kick(agent, tgt, c["n"], accept_late=5)
        elif k == "dribble" and c.get("kick"):
            # v22 self-pass: kick it ahead, then run onto it (see _off_ball)
            self.mem["plan"] = None
            self.mem["selfpass"] = (snap.cycle, c["tx"], c["ty"], c["tend"] + 4)
            _A.kick(agent, tgt, max(1, c["n"]), accept_late=3)
            _A.dribble(agent, tgt)
        elif k == "dribble":
            if not keep_plan:
                if "m" in c and self.mem.get("drib_ev") is None:
                    self.mem["drib_ev"] = (snap.cycle, c["m"], int(c.get("tend", 5)))
                self.mem["plan"] = dict(kind="dribble", tx=c["tx"], ty=c["ty"],
                                        cycle=snap.cycle, commit=COMMIT_DRIB)
            _A.dribble(agent, tgt)
            _A.hold(agent)
            _A.clear(agent)
        elif k == "hold":
            self.mem["holds"] = (self.mem.get("holds", 0)
                                 if self.mem.get("hold_cycle") == snap.cycle - 1 else 0) + 1
            self.mem["hold_cycle"] = snap.cycle
            _A.hold(agent)
            _A.clear(agent)
        else:
            _A.clear(agent)
        if k != "hold":
            self.mem["holds"] = 0

    # ------------------------------------------------------------ shots
    def _shots(self, snap, sx, sy, defenders, pressure, modes=("one", "multi")):
        """Shots at 7 points across the goal mouth; the scoring prefers the
        point the goalie needs the longest to reach (usually the far corner)."""
        out = []
        ghw = snap.goal_hw - SHOT_POST_MARGIN
        dist = math.hypot(snap.hl - sx, sy)
        acc = 1.0 if dist <= 14.0 else max(0.25, 1.0 - (dist - 14.0) / 22.0)
        for i in range(7):
            gy = -ghw + 2.0 * ghw * i / 6.0
            for mode in modes:
                if mode == "one":
                    v0, delay = ONE_STEP_SPEED, 0
                else:
                    if pressure <= 1:
                        continue
                    v0, delay = snap.vmax, 1
                if SHOT_FIT:
                    p = self._shot_fit(snap, sx, sy, gy, v0, delay, defenders)
                    if p is None:
                        continue
                else:
                    res = kick_eval(snap, sx, sy, snap.hl, gy, v0, delay, None, defenders,
                                    tmax=40, mode="shot")
                    if res is None:
                        continue
                    margin = res[0]
                    p = _sig(margin, SHOT_M0, SHOT_K) * acc
                v = p * GOAL_VALUE + (1 - p) * SHOT_MISS
                out.append(dict(kind="shot", mode=mode, tx=snap.hl, ty=gy, p=p, v=v))
        return out

    def _shot_fit(self, snap, sx, sy, gy, v0, delay, defenders):
        """v28: P(goal) from the fitted logistic model (distance, goalie's
        offset from the shot line) x the chance no FIELD player blocks it
        (our interception model, goalie excluded)."""
        gk = next((d for d in defenders if d.goalie), None)
        field = [d for d in defenders if not d.goalie]
        dist = math.hypot(snap.hl - sx, gy - sy)
        if gk is not None and gk.x > sx:
            t = (gk.x - sx) / max(snap.hl - sx, 1e-6)
            off = abs(gk.y - (sy + (gy - sy) * t))
        else:
            off = 8.0
        z = SHOT_FIT_B0 + SHOT_FIT_DIST * dist + SHOT_FIT_GK * min(off, 8.0)
        p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        if field:
            res = kick_eval(snap, sx, sy, snap.hl, gy, v0, delay, None, field,
                            tmax=40, mode="shot")
            if res is None:
                return None
            p *= _sig(res[0], SHOT_M0, SHOT_K)
        return min(p, P_CAP)

    # ------------------------------------------------------------ passes
    def _eval_runpass(self, snap, bx, by, c, defenders, v0=None, ang_noise=0.0):
        """v23: a pass into space that the receiver runs onto. The old check
        ended as soon as the receiver could touch the ball anywhere on its
        path, so a lead pass rolling past him into an opponent standing on the
        target scored p 0.97 (every kick-off in the FULL match was lost that
        way). Now: interception margin along the WHOLE path to the target, and
        at the target the defenders' margin when the receiver gets there.
        Returns (margin, t_end, ix, iy, end_margin, ex, ey) like kick_eval."""
        tx, ty = c["tx"], c["ty"]
        if ang_noise:
            L = math.hypot(tx - bx, ty - by)
            a = math.atan2(ty - by, tx - bx) + ang_noise
            tx, ty = bx + L * math.cos(a), by + L * math.sin(a)
        res = kick_eval(snap, bx, by, tx, ty, c["v0"] if v0 is None else v0, c["delay"],
                        None, defenders, tmax=40, mode="virtual")
        if res is None:
            return None
        margin, tend, ix, iy, endm, ex, ey = res
        t_recv = reach(c["recv"], ex, ey, CTRL_R)
        late = max(0, t_recv - tend)              # ball waits (rolls) for him
        return min(margin, endm - late), max(tend, t_recv), ix, iy, endm - late, ex, ey

    def _eval_pass(self, snap, bx, by, c, defenders, alpha):
        if c.get("sub") == "grid":
            a = dict(c)
            a["ang"] = math.degrees(math.atan2(c["ty"] - by, c["tx"] - bx))
            res = self._grid_eval(snap, bx, by, a, snap.me, defenders)
            if res is None:
                return None
            margin, tend, ex, ey, endm, ix, iy = res
            p = _sig(margin, self._pm0(), PASS_K)
            vs = H(snap, ex, ey, endm)
            r = dict(c)
            r.update(p=p, vs=vs, v=p * (vs - PASS_COST) + (1 - p) * Loss(snap, ix), ix=ix,
                     iy=iy, ex=ex, ey=ey, tend=tend, endm=endm, m=margin, cut=False)
            return r
        if c.get("sub") in RUN_PASSES:
            res = self._eval_runpass(snap, bx, by, c, defenders)
            if res is None:
                return None
            margin, tend, ix, iy, endm, ex, ey = res
            p = _sig(margin, self._pm0(), PASS_K)
            vs = H(snap, ex, ey, endm)
            r = dict(c)
            r.update(p=p, vs=vs, v=p * (vs - PASS_COST) + (1 - p) * Loss(snap, ix), ix=ix,
                     iy=iy, ex=ex, ey=ey, tend=tend, endm=endm, m=margin, cut=False)
            return r
        vs_ub = H(snap, c["tx"], c["ty"], CLEAR_CAP)
        state = {"cut": False}

        def cut(m):
            p = _sig(m, self._pm0(), PASS_K)
            if p * vs_ub + (1 - p) * (-L_BASE) <= alpha:
                state["cut"] = True
                return True
            return False

        res = kick_eval(snap, bx, by, c["tx"], c["ty"], c["v0"], c["delay"], c["recv"],
                        defenders, tmax=min(40, c["n"] + 12), mode="pass",
                        cut=cut if alpha > -1e8 else None)
        if res is None:
            return None
        margin, tend, ix, iy, endm, ex, ey = res
        p = _sig(margin, self._pm0(), PASS_K)
        # v29: a ball still going fast when it reaches the receiver is
        # sometimes not controlled
        k = max(0, tend - c.get("delay", 0) - 1)
        if c["v0"] * snap.decay ** k > FAST_RECV:
            p *= FAST_RECV_P
        vs = H(snap, ex, ey, endm)
        r = dict(c)
        r.update(p=p, vs=vs, v=p * (vs - PASS_COST) + (1 - p) * Loss(snap, ix), ix=ix, iy=iy,
                 ex=ex, ey=ey, tend=tend, endm=endm, m=margin, cut=state["cut"])
        return r

    # ------------------------------------------------------------ dribbles
    def _eval_dribble(self, snap, bx, by, c, defenders, holder_body=None):
        me = holder_body or snap.me
        tx, ty = c["tx"], c["ty"]
        diff = abs(_norm180(math.degrees(math.atan2(ty - by, tx - bx)) - me.body))
        T = (1 if diff > 30.0 else 0) + int(math.ceil(c["D"] / DRIB_SPEED))
        margin = 99.0
        end_m = 99.0
        mx, my = (bx + tx) / 2.0, (by + ty) / 2.0
        for d in defenders:
            if math.hypot(d.x - mx, d.y - my) - c["D"] > d.speed * T + 4.0:
                continue
            for t in range(1, T + 1):
                f = min(1.0, t / float(T))
                x, y = bx + (tx - bx) * f, by + (ty - by) * f
                m = reach(d, x, y, DRIB_TACKLE_R) - t
                if m < margin:
                    margin = m
                if t == T and m < end_m:
                    end_m = m
        p = _sig(margin, self._dm0(), DRIB_K)
        # forward carries get an extra reward proportional to the ground won
        vs = DRIB_GAMMA * H(snap, tx, ty, end_m) + DRIB_FWD_W * max(0.0, tx - bx) / 10.0
        r = dict(c)
        r.update(p=p, vs=vs, v=p * vs + (1 - p) * Loss(snap, mx), ix=mx, iy=my,
                 ex=tx, ey=ty, tend=T, recv=me, m=margin, endm=end_m)
        return r

    # ------------------------------------------------------------ hold
    def _eval_hold(self, snap, bx, by, pressure):
        p = _sig(pressure, HOLD_M0, HOLD_K)
        # no shooting-position credit: the real shots were already scored
        # separately, holding does not create one
        h = H(snap, bx, by, pressure) - W_SHOTQ * shot_quality(snap, bx, by)
        if getattr(self, "protect", False) and bx > -10.0:
            v = p * h + (1 - p) * Loss(snap, bx)          # v27: run the clock
        else:
            v = p * (h * (1.0 - HOLD_FRAC) - HOLD_COST) + (1 - p) * Loss(snap, bx)
        return dict(kind="hold", sub="hold", tx=bx, ty=by, p=p, v=v, pressure=pressure)

    # ------------------------------------------------------------ candidates
    def _onside(self, m, offside):
        return m.x + m.vx + 2.0 * max(m.vx, 0.0) < offside - 0.5 or m.x + m.vx < 0.0

    def _space_point(self, snap, m, opps, hx, hy):
        """The ONE pass target for teammate m: the point next to him that is
        furthest from opponents and has the clearest lane from the holder."""
        cx0, cy0 = m.x + m.vx, m.y + m.vy
        pts = [(cx0, cy0)]
        for i in range(8):
            a = math.radians(i * 45.0)
            pts.append((cx0 + PASS_RING_R * math.cos(a), cy0 + PASS_RING_R * math.sin(a)))
        best, bs = None, -1e9
        for px, py in pts:
            if not snap.in_field(px, py, 1.5):
                continue
            L = math.hypot(px - hx, py - hy)
            if L < PASS_MIN or L > PASS_MAX:
                continue
            sp = min(min((math.hypot(o.x - px, o.y - py) for o in opps), default=12.0), 12.0)
            lane = min(min((_seg_dist(o.x, o.y, hx, hy, px, py) for o in opps), default=6.0), 6.0)
            s = sp + 0.7 * lane + 0.15 * (px - cx0)
            if s > bs:
                bs, best = s, (px, py)
        return best

    def _time_runpass(self, snap, hx, hy, c):
        """v23: a lead pass should arrive about when the receiver does (a ball
        that gets there long before him just rolls on to a defender)."""
        L = math.hypot(c["tx"] - hx, c["ty"] - hy)
        t_recv = reach(c["recv"], c["tx"], c["ty"], CTRL_R)
        if t_recv > c["n"]:
            v0 = max(first_speed(L, t_recv, snap.decay, snap.vmax), 1.2)
            n = arrive_steps(L, v0, snap.decay)
            if n is not None:
                c["v0"], c["n"], c["delay"] = v0, n, kick_delay(v0)

    def _firm(self, snap, hx, hy, c):
        """v29: the same pass driven: arrives at ~FIRM_ARRIVE m/cycle."""
        L = math.hypot(c["tx"] - hx, c["ty"] - hy)
        v0 = min(snap.vmax, FIRM_ARRIVE + L * (1.0 - snap.decay))
        if v0 <= c["v0"] + 0.2:
            return None
        n = arrive_steps(L, v0, snap.decay)
        if n is None:
            return None
        f = dict(c)
        f.update(sub="firm", v0=v0, n=n, delay=kick_delay(v0))
        return f

    def _mk_pass(self, snap, hx, hy, tx, ty, m, sub):
        L = math.hypot(tx - hx, ty - hy)
        if L < PASS_MIN or L > PASS_MAX + 5.0:
            return None
        v = 2.0 if L < 10.0 else (2.4 if L < 20.0 else 2.8)
        n = arrive_steps(L, v, snap.decay) or arrive_steps(L, snap.vmax, snap.decay)
        if n is None:
            return None
        v0 = first_speed(L, n, snap.decay, snap.vmax)
        return dict(kind="pass", sub=sub, tx=tx, ty=ty, n=n, v0=v0, delay=kick_delay(v0), recv=m)

    def _dribble_dirs(self, snap, hx, hy, opps, pressure, names=None):
        """Fixed angles from the ball->goal line (0, +-25, +-50, +-80, back)
        plus 'away' from the nearest opponent."""
        base = math.degrees(math.atan2(-hy, snap.hl - hx))
        angles = DRIB_ANGLES
        if DENSE_ACTIONS and names is None:
            angles = tuple(-180.0 + 360.0 * i / DENSE_DRIB_DIRS for i in range(1, DENSE_DRIB_DIRS + 1))
        dirs = [("g%+d" % int(off) if abs(off) != 180.0 else "back", base + off) for off in angles]
        near = min(opps, key=lambda o: math.hypot(o.x - hx, o.y - hy), default=None)
        if near is not None and math.hypot(near.x - hx, near.y - hy) < 15.0:
            dirs.append(("away", math.degrees(math.atan2(hy - near.y, hx - near.x))))
        open_ = pressure >= OPEN_PRESSURE
        tight = pressure <= ESCAPE_PRESSURE
        out = []
        for name, a in dirs:
            if names is not None and name not in names:
                continue
            ds = (DRIB_SHORT, DRIB_LONG1) if (open_ and name in ("g+0", "g+25", "g-25", "away")) \
                else (DRIB_SHORT,)
            if tight:                    # trapped: short escapes buy a cycle or two
                ds = (ESCAPE_D,) + ds
            for D in ds:
                r = math.radians(a)
                tx, ty = hx + D * math.cos(r), hy + D * math.sin(r)
                if snap.in_field(tx, ty, 1.5):
                    out.append(dict(kind="dribble", sub=name, tx=tx, ty=ty, D=D, ang=a))
        return out

    def _pass_pen(self, r, hx, pressure, passer_unum):
        """Penalty for a pass that does not move play forward, when nobody is
        pressing: backward (BACK_PASS_PEN), too little gain (NOGAIN_PEN scaled)
        and straight back to the player who just passed to us (RETURN_PEN)."""
        if pressure <= PRESSED:
            return 0.0
        gain = r["ex"] - hx
        pen = 0.0
        if r["tx"] < hx - 1.0:
            pen += BACK_PASS_PEN
        if gain < PASS_MIN_GAIN:
            pen += NOGAIN_PEN * min(1.0, (PASS_MIN_GAIN - gain) / PASS_MIN_GAIN)
        if passer_unum is not None and r["recv"].unum == passer_unum and gain < RETURN_OK_GAIN:
            pen += RETURN_PEN
        return pen

    def _candidates(self, snap, hx, hy, holder, mates, offside, opps, pressure,
                    passer_unum=None):
        """~10 options: 1 pass per onside teammate (to his free space), 1
        through ball for the most advanced one, a handful of dribbles.
        Returned already scored (deterministic depth 1)."""
        raw = []
        onside = [m for m in mates if not m.goalie and self._onside(m, offside)]

        def beyond_ok_lead(tx, off):
            # a lead pass may go past the line only for a teammate who is onside
            # now and runs onto it (like the 'behind' ball): allow 3 m beyond
            return tx < off + 3.0
        for m in onside:
            pt = self._space_point(snap, m, opps, hx, hy)
            if pt is not None:
                c = self._mk_pass(snap, hx, hy, pt[0], pt[1], m, "space")
                if c:
                    raw.append(c)
            fx, fy = m.x + m.vx, m.y + m.vy           # straight to his feet
            if pt is None or math.hypot(pt[0] - fx, pt[1] - fy) > 1.5:
                c = self._mk_pass(snap, hx, hy, fx, fy, m, "feet")
                if c:
                    raw.append(c)
                    if FIRM_PASS:
                        f = self._firm(snap, hx, hy, c)
                        if f:
                            raw.append(f)
        if onside and ALLOW_THROUGH:
            r = max(onside, key=lambda b: (b.x, -b.unum))
            a = 0.5 * math.atan2(-r.y, snap.hl - r.x)
            tx, ty = r.x + THROUGH_R * math.cos(a), r.y + THROUGH_R * math.sin(a)
            if snap.in_field(tx, ty, 2.0) and tx > hx:
                c = self._mk_pass(snap, hx, hy, tx, ty, r, "through")
                if c:
                    raw.append(c)
        # into the space BEHIND their line, in front of each onside teammate
        # who is level with or ahead of the ball (the receiver is onside now
        # and runs onto it; their high line leaves this space empty)
        for m in onside:
            if m.x < hx - 5.0:
                continue
            tx = max(m.x, offside) + THROUGH_AHEAD
            ty = m.y * 0.9
            if tx > snap.hl - 3.0 or not snap.in_field(tx, ty, 2.0):
                continue
            c = self._mk_pass(snap, hx, hy, tx, ty, m, "behind")
            if c:
                c["v0"] = max(c["v0"], min(snap.vmax, 2.4))      # firm, so it gets there
                raw.append(c)
        # v22 lead passes: into the space AHEAD of a teammate (towards their
        # goal and either side of that), he runs onto it (claude_light's
        # passes to points around each receiver found the gaps behind a press)
        for m in onside:
            ga = math.atan2(-m.y, snap.hl - m.x)
            for R_ in LEAD_PASS_R:
                for da in LEAD_PASS_ANG:
                    a = ga + math.radians(da)
                    tx, ty = m.x + R_ * math.cos(a), m.y + R_ * math.sin(a)
                    if tx < hx + 2.0 or not snap.in_field(tx, ty, 2.0):
                        continue
                    if not beyond_ok_lead(tx, offside):
                        continue
                    c = self._mk_pass(snap, hx, hy, tx, ty, m, "lead")
                    if c:
                        self._time_runpass(snap, hx, hy, c)
                        raw.append(c)
        # v27 cutback: from near their byline, a square ball back to the
        # penalty-spot area or across to the far post, for whoever gets there
        if hx > BOX_ATTACK_X and abs(hy) > 6.0 and onside:
            sd = 1.0 if hy >= 0 else -1.0
            for (dx, dy) in (CUTBACK_SPOT, FARPOST_SPOT, (8.0, -2.0)):
                tx, ty = snap.hl - dx, -sd * dy
                m = min(onside, key=lambda b: (reach(b, tx, ty, CTRL_R), b.unum))
                c = self._mk_pass(snap, hx, hy, tx, ty, m, "cutback")
                if c:
                    self._time_runpass(snap, hx, hy, c)
                    raw.append(c)
        # v31 goalkeeper bait: the goalie has come out to the ball -> square
        # balls across the goal mouth to teammates for a tap-in
        if BAIT and onside and math.hypot(snap.hl - hx, hy) < BAIT_DIST:
            gk = next((o for o in opps if o.goalie), None)
            if gk is not None and math.hypot(gk.x - hx, gk.y - hy) < BAIT_GK_D:
                for m in onside:
                    gd = math.hypot(snap.hl - m.x, m.y)
                    if gd > 16.0 or abs(m.y - hy) < 3.0:
                        continue
                    ux, uy = _unit(snap.hl - m.x, -m.y)
                    c = self._mk_pass(snap, hx, hy, m.x + ux, m.y + uy, m, "bait")
                    if c:
                        raw.append(c)
                        f = self._firm(snap, hx, hy, c) if FIRM_PASS else None
                        if f:
                            f["sub"] = "bait"
                            raw.append(f)
        raw += self._dribble_dirs(snap, hx, hy, opps, pressure)
        # v31 corner shielding: leading late in their half -> carry the ball
        # towards their corner flag and keep it there
        if CORNER_SHIELD and getattr(self, "protect", False) and hx > 15.0:
            cy = snap.hw - 2.0 if hy >= 0 else -(snap.hw - 2.0)
            a = math.degrees(math.atan2(cy - hy, snap.hl - 2.0 - hx))
            for D in (3.0, DRIB_SHORT):
                r_ = math.radians(a)
                tx, ty = hx + D * math.cos(r_), hy + D * math.sin(r_)
                if snap.in_field(tx, ty, 1.5):
                    raw.append(dict(kind="dribble", sub="corner", tx=tx, ty=ty, D=D, ang=a))
        # v22: an opponent within ONE_STEP_PRESS cycles -> every pass is a
        # single kick (no wind-up he can tackle); evaluated at that speed
        if pressure <= ONE_STEP_PRESS:
            for c in raw:
                if c["kind"] == "pass" and (c["delay"] > 0 or c["v0"] > ONE_STEP_SPEED):
                    c["v0"] = min(c["v0"], ONE_STEP_SPEED) if c["delay"] == 0 else ONE_STEP_SPEED
                    c["delay"] = 0
                    c["one"] = True
                    L = math.hypot(c["tx"] - hx, c["ty"] - hy)
                    c["n"] = arrive_steps(L, c["v0"], snap.decay) or 40
                elif c["kind"] == "pass":
                    c["one"] = True
        out = []
        for c in raw:
            if time.perf_counter() > self._deadline and out:
                break                     # v31: wall-clock guard (1 s ceiling)
            if c["kind"] == "pass":
                r = self._eval_pass(snap, hx, hy, c, opps, -1e9)
                if r is None:
                    continue
                pen = self._pass_pen(r, hx, pressure, passer_unum)
                if pen > 0.0:
                    r["pen"] = pen
                    r["v"] -= pen * r["p"]
            else:
                r = self._eval_dribble(snap, hx, hy, c, opps, holder_body=holder)
                pen = self._retreat_pen(snap, r, hx, pressure)
                if pen > 0.0:
                    r["pen"] = pen
                    r["v"] -= pen * r["p"]
            out.append(r)
        if SELFPASS and pressure >= SELFPASS_MIN_PRESS:
            out += self._selfpasses(snap, hx, hy, holder, opps)
        if FEINT and hx > -snap.hl / 3.0:
            f = self._feint(snap, hx, hy, holder, opps)
            if f is not None:
                out.append(f)
        if DENSE_ACTIONS:
            for c in self._grid_kicks(snap, hx, hy, holder, mates, offside, opps, pressure,
                                      passer_unum):
                if c["kind"] == "dribble" and (pressure < SELFPASS_MIN_PRESS
                                               or c["p"] < SELFPASS_MIN_P):
                    continue          # self-passes keep the v24 safety rule
                out.append(c)
        return out

    def _feint(self, snap, hx, hy, holder, opps):
        """v31 sudden stop (CYRUS feint). An opponent sprinting at the ball
        cannot stop at once (player decay): if we stop the ball now he carries
        on ~FEINT_CARRY cycles past it. Value = keep it this cycle (pressure
        in that world) x the best short dribble in that world."""
        near = None
        for o in opps:
            if o.goalie:
                continue
            d = math.hypot(o.x - hx, o.y - hy)
            if d > FEINT_D:
                continue
            vx, vy = run_vel(o)
            sp = math.hypot(vx, vy)
            if sp < FEINT_V:
                continue
            # running at (or just in front of) the ball, not away from it
            if ((hx - o.x) * vx + (hy - o.y) * vy) / max(d * sp, 1e-6) < 0.5:
                continue
            if near is None or d < near[0]:
                near = (d, o, vx, vy)
        if near is None:
            return None
        _, o, vx, vy = near
        moved = []
        for b in opps:
            if b.key == o.key:
                bx_ = b.x + vx * FEINT_CARRY
                by_ = b.y + vy * FEINT_CARRY
                moved.append(Body(b.unum, bx_, by_, 0.0, 0.0,
                                  math.degrees(math.atan2(vy, vx)), b.goalie, b.speed,
                                  b.side, b.key))
            else:
                rv = run_vel(b)
                moved.append(Body(b.unum, b.x + rv[0], b.y + rv[1], 0.0, 0.0, b.body,
                                  b.goalie, b.speed, b.side, b.key))
        press = _pressure_of(moved, hx, hy)
        p_keep = _sig(press, HOLD_M0, HOLD_K)
        hb = Body(holder.unum, hx, hy, 0.0, 0.0, holder.body, False, holder.speed, -1,
                  holder.key)
        best = None
        for d in self._dribble_dirs(snap, hx, hy, moved, press):
            r = self._eval_dribble(snap, hx, hy, d, moved, holder_body=hb)
            r["v"] -= self._retreat_pen(snap, r, hx, press) * r["p"]
            if best is None or r["v"] > best["v"]:
                best = r
        if best is None:
            return None
        v = p_keep * GAMMA * best["v"] + (1 - p_keep) * Loss(snap, hx)
        return dict(kind="hold", sub="feint", tx=hx, ty=hy, p=p_keep * best["p"], v=v,
                    pressure=press, then=best.get("sub"))

    def _grid_kicks(self, snap, hx, hy, holder, mates, offside, opps, pressure,
                    passer_unum=None, dirs=None, speeds_=None, tmax=None):
        """v26: kick the ball in GRID_DIRS directions at GRID_SPEEDS speeds and
        let the model say what happens: follow the ball cycle by cycle, the
        FIRST player who can reach it (ours or theirs) gets it. Our first
        player is the receiver (a teammate = a pass, me = a self-pass), the
        gap to their first player is the margin. This covers passes to feet,
        lead / through balls, self-passes and clearances with ONE rule, and
        the whole ball path is always checked (the v23 bug cannot happen)."""
        out = []
        G = snap.G
        hl, hw = snap.hl, snap.hw
        decay = snap.decay
        recvs = [holder] + [m for m in mates if not m.goalie and self._onside(m, offside)]
        pm0 = self._pm0()
        one = pressure <= ONE_STEP_PRESS
        ndir = dirs or GRID_DIRS
        tmax = tmax or GRID_TMAX
        speeds = sorted(set(min(v, ONE_STEP_SPEED) if one else v for v in (speeds_ or GRID_SPEEDS)))
        for k in range(ndir):
            if time.perf_counter() > self._deadline:
                break                     # v31: wall-clock guard (1 s ceiling)
            a = 2.0 * math.pi * k / ndir
            ux, uy = math.cos(a), math.sin(a)
            for v0 in speeds:
                path = []
                t_out = None
                for t in range(1, tmax + 1):
                    sd = v0 * G[min(t, len(G) - 1)]
                    x, y = hx + ux * sd, hy + uy * sd
                    if abs(x) > hl or abs(y) > hw:
                        t_out = t
                        break
                    path.append((x, y))
                if not path:
                    continue
                T = len(path)
                # our first player at the ball
                tm, who = None, None
                for b in recvs:
                    lim = T if tm is None else min(T, tm - 1)
                    react = 1 if b is holder else 0
                    for t in range(1, lim + 1):
                        x, y = path[t - 1]
                        # a teammate who has to MOVE sees the kick a cycle late
                        rr = react or (1 if (x - b.x) ** 2 + (y - b.y) ** 2 > 4.0 else 0)
                        if reach(b, x, y, CTRL_R) + rr <= t:
                            tm, who = t, b
                            break
                if tm is None:
                    continue                      # nobody of ours gets it in time
                # their first player, only as far as it can matter (tm + 4)
                to, ix, iy = 99, None, None
                lim = min(T, tm + 4)
                for o in opps:
                    r = radius_of(o, hl)
                    for t in range(1, min(lim, to - 1) + 1):
                        x, y = path[t - 1]
                        if reach(o, x, y, r) <= t:
                            to, ix, iy = t, x, y
                            break
                margin = min(to, tm + 5) - tm
                rx, ry = path[tm - 1]
                endm = min((reach(o, rx, ry, PLAYER_R) for o in opps), default=99) - tm
                if ix is None or to > tm:
                    ix, iy = rx, ry
                p = _sig(margin, pm0, PASS_K)
                if who is holder:
                    vs = H(snap, rx, ry, endm) + DRIB_FWD_W * max(0.0, rx - hx) / 10.0
                    c = dict(kind="dribble", sub="gkick", kick=True, tx=rx, ty=ry, v0=v0,
                             n=tm, D=math.hypot(rx - hx, ry - hy),
                             ang=math.degrees(a), p=p, vs=vs,
                             v=p * vs + (1 - p) * Loss(snap, ix), ix=ix, iy=iy, ex=rx, ey=ry,
                             tend=tm, recv=holder, m=margin, endm=endm)
                    if c["D"] < 3.0:
                        continue                  # a tap: the dribbles cover that
                    pen = self._retreat_pen(snap, c, hx, pressure)
                else:
                    vs = H(snap, rx, ry, endm)
                    c = dict(kind="pass", sub="grid", tx=rx, ty=ry, v0=v0, n=tm,
                             delay=0 if (one or v0 <= 1.5) else kick_delay(v0), one=one,
                             ang=math.degrees(a), recv=who, p=p, vs=vs,
                             v=p * (vs - PASS_COST) + (1 - p) * Loss(snap, ix), ix=ix, iy=iy,
                             ex=rx, ey=ry, tend=tm, endm=endm, m=margin, cut=False)
                    pen = self._pass_pen(c, hx, pressure, passer_unum)
                if pen > 0.0:
                    c["pen"] = pen
                    c["v"] -= pen * c["p"]
                out.append(c)
        return out

    def _grid_eval(self, snap, sx, sy, a, holder, opps, rng=None):
        """One noisy replay of a grid kick (Monte Carlo): same first-to-the-ball
        rule as _grid_kicks, for the planned receiver only.
        Returns (margin, t, x, y, clear, ix, iy) or None."""
        ang = math.radians(a["ang"]) + (math.radians(rng.gauss(0.0, MC_KICK_DEG)) if rng else 0.0)
        v0 = a["v0"] * (1.0 + (rng.gauss(0.0, 0.04) if rng else 0.0))
        ux, uy = math.cos(ang), math.sin(ang)
        G = snap.G
        recv = a["recv"]
        react = 1 if a["kind"] == "dribble" else 0
        if a["kind"] == "dribble":
            recv = holder
        path = []
        for t in range(1, GRID_TMAX + 1):
            sd = v0 * G[min(t, len(G) - 1)]
            x, y = sx + ux * sd, sy + uy * sd
            if not snap.in_field(x, y, 0.0):
                break
            path.append((x, y))
        tm = None
        for t in range(1, len(path) + 1):
            x, y = path[t - 1]
            rr = react or (1 if (x - recv.x) ** 2 + (y - recv.y) ** 2 > 4.0 else 0)
            if reach(recv, x, y, CTRL_R) + rr <= t:
                tm = t
                break
        if tm is None:
            return None
        to, ix, iy = 99, None, None
        for o in opps:
            r = radius_of(o, snap.hl)
            for t in range(1, min(tm + 4, len(path), to - 1) + 1):
                x, y = path[t - 1]
                if reach(o, x, y, r) <= t:
                    to, ix, iy = t, x, y
                    break
        rx, ry = path[tm - 1]
        endm = min((reach(o, rx, ry, PLAYER_R) for o in opps), default=99) - tm
        if ix is None or to > tm:
            ix, iy = rx, ry
        return min(to, tm + 5) - tm, tm, rx, ry, endm, ix, iy

    def _diverse_top(self, cands, k):
        """Beam selection: keep the k best options by value, skipping an
        option that is a near-duplicate of one already kept (same kind and
        receiver, ball ending within DUP_D metres) - merged the way a
        transposition table merges equal states - so the look-ahead spends
        its time on k DIFFERENT ideas, not 10 copies of one pass."""
        if len(cands) <= k:
            return list(cands)
        ranked = sorted(cands, key=lambda c: -c["v"])

        def end(c):
            return c.get("ex", c.get("tx", 0.0)), c.get("ey", c.get("ty", 0.0))

        def key(c):
            return (c["kind"], getattr(c.get("recv"), "unum", None))

        chosen, skipped = [], []
        for c in ranked:
            cx, cy = end(c)
            dup = any(key(b) == key(c) and math.hypot(cx - end(b)[0], cy - end(b)[1]) < DUP_D
                      for b in chosen)
            (skipped if dup else chosen).append(c)
            if len(chosen) >= k:
                break
        for c in skipped:                 # fewer than k distinct: fill up
            if len(chosen) >= k:
                break
            chosen.append(c)
        return chosen

    def _retreat_pen(self, snap, r, hx, pressure):
        """Backward dribbles. Unpressured: BACK_PASS_PEN (no ping-pong).
        v22: in our own half also under pressure (replay: 4 kick-offs in a
        row the holder dribbled 10-30 m back towards our goal, one step at a
        time, each step 'safe'; the last one lost the ball at our box)."""
        back = hx - r["tx"]
        if back <= 1.0:
            return 0.0
        pen = BACK_PASS_PEN if pressure > 2 else 0.0
        if hx < 0.0:
            pen += RETREAT_PEN * min(1.0, back / 5.0) * (1.0 + min(1.0, -hx / snap.hl))
        if r["tx"] < -snap.hl / 3.0 and pressure > 1:
            pen += OWN_THIRD_BACK_PEN
        return pen

    def _selfpasses(self, snap, hx, hy, holder, opps):
        """v22: kick the ball 8-13 m ahead into space and run onto it (the way
        claude_light carried the ball 25-45 m in 30 cycles; our close dribble
        made 0.5 m/cycle, mostly sideways or back). Scored like a pass to
        myself: interception margin along the path, and the defenders'
        margin at the point when I get there."""
        out = []
        ga = math.degrees(math.atan2(-hy, snap.hl - hx))
        for da in SELFPASS_ANGLES:
            a = math.radians(ga + da)
            for L in SELFPASS_L:
                tx, ty = hx + L * math.cos(a), hy + L * math.sin(a)
                if not snap.in_field(tx, ty, 2.0) or tx > snap.offside + 30.0:
                    continue
                r = self._eval_selfpass(snap, hx, hy, tx, ty, holder, opps)
                if r is not None and r["p"] >= SELFPASS_MIN_P:
                    r.update(sub="kick%d" % int(da), ang=ga + da, D=L)
                    out.append(r)
        return out

    def _eval_selfpass(self, snap, hx, hy, tx, ty, holder, opps, v0=None, rng=None):
        L = math.hypot(tx - hx, ty - hy)
        t_me = reach(holder, tx, ty, CTRL_R) + 1           # +1: the kick cycle
        if v0 is None:
            v0 = first_speed(L, max(t_me, 2), snap.decay, snap.vmax)
            v0 = min(v0, ONE_STEP_SPEED)
            if v0 < 0.6:
                return None
        res = kick_eval(snap, hx, hy, tx, ty, v0, 0, None, opps,
                        tmax=max(t_me, 4) + 6, mode="virtual")
        if res is None:
            return None
        margin, tend, ix, iy, endm, ex, ey = res
        late = max(0, t_me - tend)                  # ball waits for me
        m = min(margin, endm - late)
        if rng is not None:
            return m, ex, ey, max(tend, t_me), endm - late, ix, iy
        p = _sig(m, self._pm0(), PASS_K)
        vs = H(snap, ex, ey, endm - late) + DRIB_FWD_W * max(0.0, ex - hx) / 10.0
        return dict(kind="dribble", kick=True, tx=tx, ty=ty, v0=v0, n=max(tend, 2),
                    p=p, vs=vs, v=p * vs + (1 - p) * Loss(snap, ix), ix=ix, iy=iy,
                    ex=ex, ey=ey, tend=max(tend, t_me), recv=holder, m=m, endm=endm - late)

    # ------------------------------------------------------------ depth 2-3
    def _best_next(self, snap, holder, hx, hy, T, passer):
        """The next holder's best action in the world predicted for that
        moment (opponents moved towards him). None = better to just keep it."""
        popps = predict_opps(snap, hx, hy, T)
        k = min(T, 6)
        mates = []
        for b in snap.team:
            if b.unum == holder.unum:
                continue
            if passer is not None and b.unum == passer.unum:
                mates.append(Body(b.unum, b.x + 3.0, b.y, 0.0, 0.0, 0.0, False, b.speed, -1, b.key))
            else:
                mates.append(Body(b.unum, b.x + b.vx * k, b.y + b.vy * k, 0.0, 0.0, b.body,
                                  False, b.speed, -1, b.key))
        hb = Body(holder.unum, hx, hy, 0.0, 0.0, 0.0, False, holder.speed, -1, holder.key)
        off2 = offside_line_of(popps, hx)
        best = None
        if math.hypot(snap.hl - hx, hy) < SHOT_RANGE:
            for s in self._shots(snap, hx, hy, popps, 99, modes=("multi",)):
                if best is None or s["v"] > best["v"]:
                    best = s
        for m in mates:
            if not self._onside(m, off2):
                continue
            pt = self._space_point(snap, m, popps, hx, hy)
            if pt is None:
                continue
            c = self._mk_pass(snap, hx, hy, pt[0], pt[1], m, "space")
            if c is None:
                continue
            r = self._eval_pass(snap, hx, hy, c, popps, -1e9)
            if r is None:
                continue
            pen = self._pass_pen(r, hx, _pressure_of(popps, hx, hy),
                                 passer.unum if passer is not None else None)
            if pen > 0.0:
                r["pen"] = pen
                r["v"] -= pen * r["p"]
            if best is None or r["v"] > best["v"]:
                best = r
        for d in self._dribble_dirs(snap, hx, hy, popps, 0, names=("g+0", "g+25", "g-25", "away")):
            d["D"] = 6.0
            r = math.radians(d["ang"])
            d["tx"], d["ty"] = hx + 6.0 * math.cos(r), hy + 6.0 * math.sin(r)
            if not snap.in_field(d["tx"], d["ty"], 1.5):
                continue
            r = self._eval_dribble(snap, hx, hy, d, popps, holder_body=hb)
            if best is None or r["v"] > best["v"]:
                best = r
        # v31: plus a coarse kick grid for the next holder (passes into space,
        # lead balls, self-passes the hand-made options above do not cover)
        if NEXT_GRID_DIRS and time.perf_counter() < self._deadline:
            npress = _pressure_of(popps, hx, hy)
            for r in self._grid_kicks(snap, hx, hy, hb, mates, off2, popps, npress,
                                      passer.unum if passer is not None else None,
                                      dirs=NEXT_GRID_DIRS, speeds_=NEXT_GRID_SPEEDS,
                                      tmax=NEXT_GRID_TMAX):
                if r["p"] < RISKY_P:
                    continue
                if best is None or r["v"] > best["v"]:
                    best = r
        here = ((H(snap, hx, hy, 4.0) - W_SHOTQ * shot_quality(snap, hx, hy)) * (1.0 - HOLD_FRAC)
                - HOLD_COST)
        if best is None or best["v"] <= here:
            return None
        return best

    def _chain(self, snap, c, deadline):
        """Root action + the best next action + the best one after that (depth 3)."""
        chain = [c]
        cur = c
        if c["kind"] == "pass":
            holder, passer = c["recv"], snap.me
        else:
            holder, passer = snap.me, None
        T = c["tend"]
        for _ in range(MC_DEPTH - 1):
            if cur["kind"] not in ("pass", "dribble") or cur["p"] < 0.15:
                break
            if time.perf_counter() > deadline:
                break
            nxt = self._best_next(snap, holder, cur["ex"], cur["ey"], T, passer)
            if nxt is None:
                break
            chain.append(nxt)
            if nxt["kind"] == "pass":
                passer, holder = holder, nxt["recv"]
            elif nxt["kind"] == "dribble":
                passer = None
            T += nxt.get("tend", 0)
            cur = nxt
        return chain

    def _chain_value(self, snap, chain, i=0):
        """Deterministic value of a chain (used as the Monte Carlo prior)."""
        a = chain[i]
        if a["kind"] in ("shot", "hold"):
            return a["v"]
        cost = (PASS_COST if a["kind"] == "pass" else 0.0) + a.get("pen", 0.0)
        succ = a["vs"]
        if i + 1 < len(chain):
            succ = max(succ, GAMMA * self._chain_value(snap, chain, i + 1))
        return a["p"] * (succ - cost) + (1 - a["p"]) * Loss(snap, a["ix"])

    # ------------------------------------------------------------ Monte Carlo
    def _noisy(self, opps, rng):
        """Opponents a little faster/slower and a little elsewhere than we
        think: our model is crude, the samples cover that uncertainty."""
        g = rng.gauss
        return [Body(o.unum, o.x + g(0.0, MC_POS_NOISE), o.y + g(0.0, MC_POS_NOISE),
                     o.vx, o.vy, o.body, o.goalie,
                     o.speed * rng.uniform(1.0 - MC_SPEED_NOISE, 1.0 + MC_SPEED_NOISE),
                     o.side, o.key) for o in opps]

    def _sample(self, snap, a, sx, sy, holder, opps, rng, pressure):
        """One noisy execution of action a from (sx, sy).
        Returns (ok, x, y, t, clearance): where the ball ends up (or is lost)."""
        k = a["kind"]
        nopps = self._noisy(opps, rng)
        if k == "hold":
            return _sig(pressure, HOLD_M0, HOLD_K), sx, sy, 1, pressure
        if a.get("sub") in ("grid", "gkick"):
            res = self._grid_eval(snap, sx, sy, a, holder, nopps, rng)
            if res is None:
                return False, sx, sy, a.get("tend", 5), 0
            m, t, ex, ey, endm, ix, iy = res
            if m + rng.gauss(0.0, MC_SIGMA) > self.L["pass_adj"]:
                return True, ex, ey, t, endm
            return False, ix, iy, t, 0
        if k == "dribble" and a.get("kick"):
            ang = math.radians(a["ang"] + rng.gauss(0.0, MC_KICK_DEG))
            tx, ty = sx + a["D"] * math.cos(ang), sy + a["D"] * math.sin(ang)
            if not snap.in_field(tx, ty, 0.5):
                return False, sx, sy, 1, 0
            v0 = a["v0"] * (1.0 + rng.gauss(0.0, 0.05))
            res = self._eval_selfpass(snap, sx, sy, tx, ty, holder, nopps, v0=v0, rng=rng)
            if res is None:
                return False, tx, ty, a["tend"], 0
            m, ex, ey, t, endm, ix, iy = res
            if m + rng.gauss(0.0, MC_SIGMA) > self.L["pass_adj"]:
                return True, ex, ey, t, endm
            return False, ix, iy, t, 0
        if k == "dribble":
            r = math.radians(a["ang"])
            c = dict(a, tx=sx + a["D"] * math.cos(r), ty=sy + a["D"] * math.sin(r))
            if not snap.in_field(c["tx"], c["ty"], 0.5):
                return False, sx, sy, 1, 0
            res = self._eval_dribble(snap, sx, sy, c, nopps, holder_body=holder)
            ok = res["m"] + rng.gauss(0.0, MC_SIGMA) > MC_DRIB_OFF + self.L["drib_adj"]
            if ok:
                return True, res["ex"], res["ey"], res["tend"], res["endm"]
            return False, res["ix"], res["iy"], res["tend"], 0
        # pass or shot: angle and speed noise on the kick
        tx, ty = a["tx"], a["ty"]
        ang = math.atan2(ty - sy, tx - sx) + math.radians(rng.gauss(0.0, MC_KICK_DEG))
        L = math.hypot(tx - sx, ty - sy)
        ntx, nty = sx + L * math.cos(ang), sy + L * math.sin(ang)
        v0 = min(snap.vmax, a.get("v0", snap.vmax) * (1.0 + rng.gauss(0.0, 0.04)))
        if k == "shot":
            v0 = snap.vmax if a.get("mode") == "multi" else ONE_STEP_SPEED
            res = kick_eval(snap, sx, sy, ntx, nty, v0, 1 if a.get("mode") == "multi" else 0,
                            None, nopps, tmax=40, mode="shot")
            if res is None:
                return 0.0, sx, sy, 1, 0
            if abs(res[6]) >= snap.goal_hw - 0.1:       # noisy kick misses the goal
                return 0.0, res[5], res[6], res[1], 0
            if SHOT_FIT:
                pf = self._shot_fit(snap, sx, sy, res[6], v0,
                                    1 if a.get("mode") == "multi" else 0, nopps)
                return (pf or 0.0), res[5], res[6], res[1], 0
            return _sig(res[0], SHOT_M0, SHOT_K), res[5], res[6], res[1], 0
        if a.get("sub") in RUN_PASSES:
            res = self._eval_runpass(snap, sx, sy, a, nopps, v0=v0,
                                     ang_noise=math.radians(rng.gauss(0.0, MC_KICK_DEG)))
        else:
            res = kick_eval(snap, sx, sy, ntx, nty, v0, a["delay"], a["recv"], nopps,
                            tmax=min(40, a["n"] + 12), mode="pass")
        if res is None:
            return False, ntx, nty, a["n"], 0
        margin, tend, ix, iy, endm, ex, ey = res
        if margin + rng.gauss(0.0, MC_SIGMA) > self.L["pass_adj"]:
            return True, ex, ey, tend, endm
        return False, ix, iy, tend, 0

    def _rollout(self, snap, chain, rng, pressure, subs):
        sx, sy = snap.bx, snap.by
        holder = snap.me
        T = 0
        total, disc = 0.0, 1.0
        for i, a in enumerate(chain):
            opps = snap.opps if i == 0 else predict_opps(snap, sx, sy, T,
                                                          frac=rng.uniform(0.4, 0.8))
            ok, x, y, t, clear = self._sample(snap, a, sx, sy, holder, opps, rng, pressure)
            k = a["kind"]
            if k == "shot":            # ok = goal probability here
                return total + disc * (ok * GOAL_VALUE + (1 - ok) * SHOT_MISS)
            if a.get("sub") == "feint":   # v31: valued in its own (overrun) world
                return total - a.get("stall", 0.0) + disc * (a["v"] + rng.gauss(0.0, 0.3))
            if k == "hold":            # ok = keep probability here
                h = H(snap, sx, sy, pressure) - W_SHOTQ * shot_quality(snap, sx, sy)
                return (total - a.get("stall", 0.0)
                        + disc * (ok * (h * (1.0 - HOLD_FRAC) - HOLD_COST)
                                  + (1 - ok) * Loss(snap, sx)))
            total -= disc * ((PASS_COST if k == "pass" else 0.0) + a.get("pen", 0.0))
            if not ok:
                return total + disc * Loss(snap, x)
            here = H(snap, x, y, clear)
            if k == "dribble" and i == len(chain) - 1:
                total += disc * DRIB_FWD_W * max(0.0, x - sx) / 10.0
            # stop here if continuing is EXPECTED (deterministic value of the
            # rest of the chain) to be worth less: a real player would just
            # keep the ball. Not decided on the sampled future -> no peeking.
            if i == len(chain) - 1 or GAMMA * subs[i + 1] <= here:
                return total + disc * here
            if k == "dribble":
                total += disc * DRIB_FWD_W * max(0.0, x - sx) / 10.0
            if k == "pass":
                r = a["recv"]
                holder = Body(r.unum, x, y, 0.0, 0.0, 0.0, False, r.speed, -1, r.key)
            else:
                holder = Body(holder.unum, x, y, 0.0, 0.0, holder.body, False,
                              holder.speed, -1, holder.key)
            sx, sy = x, y
            T += t
            disc *= GAMMA
        return total

    def _mc_select(self, snap, roots, deadline, rng, pressure):
        """UCB1 over the root actions: spend more samples on the close calls.
        The deterministic chain value acts as a prior worth MC_PRIOR_W samples."""
        R = len(roots)
        subs = [[self._chain_value(snap, r["chain"], k) for k in range(len(r["chain"]))]
                for r in roots]
        n = [0] * R
        s = [0.0] * R
        total = 0
        while total < MC_MAX and time.perf_counter() < deadline:
            if total < R:
                i = total
            else:
                lt = math.log(total + 1)
                i = max(range(R), key=lambda j: (s[j] + MC_PRIOR_W * roots[j]["prior"])
                        / (n[j] + MC_PRIOR_W) + MC_C * math.sqrt(lt / (n[j] + 1)))
            v = self._rollout(snap, roots[i]["chain"], rng, pressure, subs[i])
            n[i] += 1
            s[i] += v
            total += 1
        means = [(s[j] + MC_PRIOR_W * roots[j]["prior"]) / (n[j] + MC_PRIOR_W) for j in range(R)]
        return means, total

    # ------------------------------------------------------------ prediction
    def _predict_holder(self, snap, holder):
        """What will our ball holder do? Every off-ball player runs the same
        deterministic depth-1 choice the holder uses as its default."""
        hx, hy = snap.bx, snap.by
        mates = [b for b in snap.team if b.unum != holder.unum]
        pressure = _pressure(snap, hx, hy)
        cands = self._candidates(snap, hx, hy, holder, mates, snap.offside, snap.opps, pressure)
        if math.hypot(snap.hl - hx, hy) < SHOT_RANGE:
            shots = self._shots(snap, hx, hy, snap.opps, pressure)
            if shots:
                cands.append(max(shots, key=lambda c: c["v"]))
        cands.append(self._eval_hold(snap, hx, hy, pressure))
        return max(self._risk_filter(cands), key=lambda c: c["v"])

    def _risk_filter(self, cands):
        """v24: no gambling when a safe option exists. If any pass / dribble /
        hold keeps the ball with p >= SAFE_P, passes and dribbles below
        RISKY_P are dropped (shots are never dropped)."""
        safe = any(c["kind"] in ("pass", "dribble", "hold") and c["p"] >= SAFE_P for c in cands)
        if not safe:
            return cands
        return [c for c in cands if c["kind"] not in ("pass", "dribble") or c["p"] >= RISKY_P]

    # =================================================================
    # OFF BALL
    # =================================================================
    def off_ball(self, agent) -> None:
        try:
            self._off_ball(agent)
        except Exception as e:
            _note(agent, _err_label("off", e))
            StarterAgent.off_ball(self, agent)
        self._learn_note(agent)

    def _off_ball(self, agent):
        t0 = time.perf_counter()
        snap = Snap(agent)
        wm = agent.wm
        mem = self.mem
        rem, dec = self._budget(agent)
        if rem < QUICK_REMAIN_MS or dec < QUICK_DEC_MS or snap.me.goalie:
            _note(agent, "quick_off_ball")
            StarterAgent.off_ball(self, agent)
            return
        self._learn(agent, snap, False)
        allow = max(self._allowance(agent, snap, False), 1.0)
        self._deadline = t0 + allow / 1000.0
        me = snap.me

        # --- who reaches the ball first: computed with OUR model for every
        #     player, so all teammates get identical numbers and agree.
        our_t = {b.unum: ball_reach_steps(b, snap, CTRL_R) for b in snap.team}
        chaser = min(snap.team, key=lambda b: (our_t[b.unum], b.unum))
        our_min = our_t[chaser.unum]
        opp_min, carrier = 99, None
        for o in snap.opps:
            t = ball_reach_steps(o, snap, radius_of(o, snap.hl))
            if t < opp_min or (t == opp_min and carrier is not None and o.key < carrier.key):
                opp_min, carrier = t, o

        mate_kick = bool(getattr(wm, "kickable_teammate_existance", False))
        opp_kick = bool(getattr(wm, "kickable_opponent_existance", False))

        # --- dribbling: the ball leaves the holder's feet for a moment. If a
        #     teammate had it within HOLD_MEMORY cycles and is still the one
        #     closest to it, it is STILL his ball: nobody else runs at it.
        lh0 = mem.get("our_holder")
        if (not mate_kick and not opp_kick and lh0 is not None and lh0[0] == me.unum
                and snap.cycle - lh0[1] <= HOLD_MEMORY
                and math.hypot(me.x - snap.bx, me.y - snap.by) < 4.0
                and our_t.get(me.unum, 99) <= opp_min):
            _note(agent, "chase_own")
            _A.intercept(agent)
            return
        if not mate_kick and not opp_kick:
            lh = mem.get("our_holder")
            if lh is not None and snap.cycle - lh[1] <= HOLD_MEMORY:
                hb = next((b for b in snap.mates if b.unum == lh[0] and not b.goalie), None)
                if (hb is not None and math.hypot(hb.x - snap.bx, hb.y - snap.by) < 4.0
                        and our_t.get(hb.unum, 99) < opp_min):
                    mate_kick = True
        if mate_kick:
            near_m = min(snap.mates, key=lambda b: math.hypot(b.x - snap.bx, b.y - snap.by))
            if math.hypot(near_m.x - snap.bx, near_m.y - snap.by) < 2.0:
                mem["our_holder"] = (near_m.unum, snap.cycle)

        # --- v22 self-pass in flight: the kicker runs onto it (nobody else)
        sp = mem.get("selfpass")
        if sp is not None and not mate_kick and not opp_kick:
            if snap.cycle - sp[0] <= sp[3] and our_t.get(me.unum, 99) <= opp_min + 1:
                _note(agent, "selfpass_run")
                _A.intercept(agent)
                return
        if sp is not None and snap.cycle - sp[0] > 2:
            if mate_kick or opp_kick or snap.cycle - sp[0] > sp[3]:
                mem["selfpass"] = None

        # --- contested (one of ours AND one of theirs at the ball): defend.
        if mate_kick and opp_kick:
            mate_kick = False

        # --- A: a teammate has the ball -> attacking positions
        if mate_kick:
            mem["last_poss"] = "us"
            holder = min(snap.mates, key=lambda b: math.hypot(b.x - snap.bx, b.y - snap.by))
            if mem.get("our_holder") and snap.cycle - mem["our_holder"][1] <= HOLD_MEMORY:
                holder = next((b for b in snap.mates if b.unum == mem["our_holder"][0]), holder)
            pred = None
            if not holder.goalie and time.perf_counter() < self._deadline:
                pred = self._predict_holder(snap, holder)
            if pred is not None and pred["kind"] == "pass" and "m" in pred:
                # v25: what we expect the holder to play, with its predicted
                # margin -> checked against the outcome if he really plays it
                mem["dec"] = (snap.cycle, pred["tx"], pred["ty"], pred["m"], snap.bx, snap.by)
            self._attack(agent, snap, snap.bx, snap.by, {holder.unum}, False, pred)
            return

        # --- B: ball loose and we get there first
        if not opp_kick and our_min <= opp_min:
            if chaser.unum == me.unum:
                self._chase(agent, snap, our_min, opp_min)
                return
            if mem.get("last_poss", "us") == "us":
                # our own pass in flight: keep attacking shape, runner may go
                ix, iy = snap.ball_at(our_min)
                self._attack(agent, snap, ix, iy, {chaser.unum}, False)
                return
            # we are winning THEIR pass
            if our_min + COUNTER_MARGIN <= opp_min:
                ix, iy = snap.ball_at(our_min)
                rest = [b for b in snap.team if b.unum != chaser.unum]
                if rest:
                    runner = max(rest, key=lambda b: (b.x, -b.unum))
                    if runner.unum == me.unum:
                        tx, ty = self._runner_target(snap, ix, iy, False)
                        _note(agent, "counter_run")
                        self._go(agent, snap, tx, ty, urgent=True)
                        return
                    self._defend(agent, snap, None, carrier, opp_kick,
                                 {chaser.unum, runner.unum})
                    return
            self._defend(agent, snap, None, carrier, opp_kick, {chaser.unum})
            return

        # --- C: opponent has it / gets there first -> press + mark
        mem["last_poss"] = "them"
        self._defend(agent, snap, "auto", carrier, opp_kick, set(),
                     our_min=our_min, opp_min=opp_min, our_t=our_t)

    # ------------------------------------------------------------ movement
    def _dash(self, agent, snap, urgent, dist):
        """Sprint only when it matters (presser, recovery run, receiver);
        everything else at a pace that lets stamina recover. Below the
        server's effort threshold (30 %) players get slower for a long time."""
        ratio = float(agent.wm.self.stamina) / snap.stamina_max
        if urgent:
            if ratio > SPRINT_MIN:
                return 100.0
            return 80.0 if ratio > 0.32 else 55.0
        if ratio < 0.5:
            return 40.0
        if dist > 10.0:
            return 70.0
        if dist > 4.0:
            return 55.0
        return 40.0

    def _go(self, agent, snap, tx, ty, urgent=False, thr=None):
        me = snap.me
        tx = max(-snap.hl + 0.5, min(snap.hl - 0.5, tx))
        ty = max(-snap.hw + 0.5, min(snap.hw - 0.5, ty))
        d = math.hypot(tx - me.x, ty - me.y)
        _A.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self._dash(agent, snap, urgent, d),
                           distance_threshold=thr)
        _A.turn_to_ball(agent)

    def _chase(self, agent, snap, my_t, opp_t):
        """Economical interception: run only as fast as needed when we are
        clearly first; sprint (built-in intercept) when the race is close."""
        me = snap.me
        moving = math.hypot(snap.bvx, snap.bvy) > 0.5
        if (my_t <= 2 or opp_t - my_t < ECO_MIN_MARGIN or moving
                or self.mem.get("last_poss") == "us"):
            _note(agent, "chase")
            _A.intercept(agent)
            _A.go_to_point(agent, Vector2D(snap.bx, snap.by))
            return
        t_meet = min(opp_t - 3, my_t + 4)
        cx, cy = snap.ball_at(t_meet)
        if not snap.in_field(cx, cy, 0.0):
            _note(agent, "chase")
            _A.intercept(agent)
            return
        d = max(0.0, math.hypot(cx - me.x, cy - me.y) - 0.5 * CTRL_R)
        need = d / max(1, t_meet - 1)
        power = max(45.0, min(100.0, need / SPEED_PER_POWER * 1.25))
        _note(agent, "chase_eco", power=int(power))
        _A.go_to_point(agent, Vector2D(cx, cy), max_dash_power=power,
                           distance_threshold=0.6)
        _A.intercept(agent)

    # ------------------------------------------------------------ defending
    def _defend(self, agent, snap, presser_unum, carrier, opp_kick, excluded,
                our_min=99, opp_min=99, our_t=None):
        """Every player gets ONE explicit role, computed identically by all.
        Priority:
          1. FRONT PRESS  - fastest to the carrier, but coming from BEHIND him
                            costs BEHIND_PEN cycles, so a player between him
                            and our goal is preferred (blocks his way to goal,
                            tackles from behind give fouls)
          2. BACK PRESS   - "sandwich": a 2nd player behind the carrier who is
                            within SANDWICH_STEPS also harasses him
          3. MARK         - one-to-one, ONLY opponents goal-side of the ball
                            (+MARK_MARGIN), most dangerous first
          4. SCREEN near / far - on the lines ball -> near post / far post
          5. COVER        - in front of our goal
          6. ZONE         - shifted formation slot (only if still spare)
        Assignment of players to tasks 3-5 minimises total running distance
        (all permutations), with hysteresis so roles don't flicker."""
        me = snap.me
        hl = snap.hl
        # carrier position: at the ball, or where he will receive it
        if carrier is None:
            cx, cy = snap.bx, snap.by
        elif opp_kick:
            carrier = min(snap.opps, key=lambda o: math.hypot(o.x - snap.bx, o.y - snap.by))
            cx, cy = snap.bx, snap.by
        else:
            cx, cy = snap.ball_at(min(opp_min, 40))

        weights, top_recv = self._opp_threats(snap, carrier, cx, cy)

        pool = [b for b in snap.team if b.unum not in excluded]
        back_unum = None
        if presser_unum == "auto":
            # a player between the carrier and our goal presses if he can get
            # there in reasonable time; only otherwise the fastest overall
            # (coming from behind costs BEHIND_PEN cycles)
            # "goal-side" = closer to OUR GOAL than the carrier (works in the
            # corners too, where x alone says nobody is goal-side)
            dc = math.hypot(cx + hl, cy)
            def gside(b):
                return math.hypot(b.x + hl, b.y) <= dc + 1.0
            goal_side = [b for b in pool if gside(b)
                         and our_t.get(b.unum, 99) <= FRONT_MAX_STEPS]
            def pcost(b):
                # stickiness everybody can SEE: a player already running at
                # the ball keeps the job (no private memory -> all agree)
                c = our_t.get(b.unum, 99) + (0 if gside(b) else BEHIND_PEN)
                return c - (PRESS_HYST if _moving_to(b, cx, cy) else 0)
            front = min(goal_side or pool, key=lambda b: (pcost(b), b.unum))
            presser_unum = front.unum
            behind = [b for b in pool if b.unum != front.unum and not gside(b)
                      and our_t.get(b.unum, 99) <= SANDWICH_STEPS]
            if len(pool) < 4:
                behind = []          # short-handed: keep everyone else in shape
            if behind and (opp_kick or opp_min <= 3):
                back_unum = min(behind, key=lambda b: (our_t.get(b.unum, 99), b.unum)).unum

        if presser_unum is not None and me.unum == presser_unum:
            self._press(agent, snap, cx, cy, top_recv, opp_kick, our_min, opp_min)
            return
        if back_unum is not None and me.unum == back_unum:
            self._press_back(agent, snap)
            return

        # anyone else close enough takes a decent tackle chance; if he is close
        # but not facing the ball (tackle chance 0), he turns to face it first
        if opp_kick and self._try_tackle(agent, snap, MARKER_TACKLE_MIN):
            return
        if opp_kick:
            bd = math.hypot(snap.bx - me.x, snap.by - me.y)
            ang = math.degrees(math.atan2(snap.by - me.y, snap.bx - me.x))
            if bd < 2.5 and abs(_norm180(ang - me.body)) > 35.0:
                _note(agent, "def_face")
                _A.turn_to_ball(agent)
                return

        markers = [b for b in pool if b.unum != presser_unum and b.unum != back_unum]
        if not markers:
            self._go(agent, snap, *_cxy(_home(agent)))
            return

        # --- task list in priority order
        tasks = []
        marks = []
        # "already marked" = one of our markers is right next to him: visible
        # to everybody, so all teammates agree (no private memory)
        prev_marked = {o.key for o in snap.opps
                       if any(math.hypot(b.x - o.x, b.y - o.y) < 2.5 for b in markers)}
        dball = math.hypot(cx + hl, cy)
        for o in snap.opps:
            if o.goalie or (carrier is not None and o.key == carrier.key):
                continue
            if math.hypot(o.x - cx, o.y - cy) < 4.0:
                continue          # right next to the carrier: the presser is there
            # mark him if he is closer to OUR GOAL than the ball (+margin) or
            # anywhere near our goal; already-marked men get extra slack
            do = math.hypot(o.x + hl, o.y)
            slack = 4.0 if o.key in prev_marked else 0.0
            if do > dball + MARK_MARGIN + slack and do > DANGER_ZONE + slack:
                continue
            dg = math.hypot(o.x + hl, o.y)
            danger = (1.0 - min(dg, 80.0) / 80.0
                      + 0.2 * (1.0 - min(abs(o.y), snap.hw) / snap.hw)
                      + 1.5 * weights.get(o.key, 0.0))
            if o.key in prev_marked:
                danger += MARK_STICKY
            marks.append((danger, o.key, o))
        marks.sort(key=lambda t: (-t[0], t[1]))
        for _, k, o in marks:
            tasks.append(("mark", k, self._mark_point(
                snap, o, cx, cy, top_recv is not None and o.key == top_recv.key)))
        near = 1.0 if cy >= 0 else -1.0
        # 2nd line on the carrier's own path to goal (behind the presser),
        # then the far-post screen, then cover in front of goal
        pux, puy = _unit(-hl - cx, -cy)
        dball = math.hypot(cx + hl, cy)
        dpath = min(PATH_D, 0.5 * dball)
        tasks.append(("path", "pa", (cx + pux * dpath, cy + puy * dpath)))
        # second line: two players DEEP_D in front of the ball, either side of
        # its line to goal (not another two around the ball: when the carrier
        # escapes the press, somebody must already be organised behind)
        dd = min(DEEP_D, 0.6 * dball)
        bxl, byl = cx + pux * dd, cy + puy * dd
        for name, key, sgn in (("deep_l", "dl", -1.0), ("deep_r", "dr", 1.0)):
            px, py = bxl - puy * DEEP_W * sgn, byl + pux * DEEP_W * sgn
            px = max(px, -hl + 4.0)
            py = max(-snap.hw + 3.0, min(snap.hw - 3.0, py))
            tasks.append((name, key, (px, py)))
        tasks.append(("cover", "cv", self._cover_point(snap, cx, cy)))
        tasks = tasks[:len(markers)]
        # always keep somebody DEEP: if no job is 12+ m goal-side of the ball,
        # the last job becomes a central deep position on the ball->goal line
        if len(tasks) >= 2 and not any(
                math.hypot(t[2][0] + hl, t[2][1]) < dball - 12.0 for t in tasks):
            dd = min(DEEP_D, 0.6 * dball)
            tasks[-1] = ("deep", "dc", (cx + pux * dd, cy + puy * dd))
        tpos = {t[1]: t[2] for t in tasks}
        tname = {t[1]: t[0] for t in tasks}

        # --- assignment (identical on every player)
        tids = [t[1] for t in tasks]
        # distance from where each player WILL be (position + 3 cycles of his
        # current velocity): someone already running to a task is cheaper for
        # it -> stable roles that every teammate computes identically
        best_cost, best_asg = 1e9, ()
        # v29: a defender who is FURTHER from our goal than the job (i.e.
        # behind it) can't get goal-side of a runner in time - replay vs FULL:
        # the deep runner was "marked" by a player 13 m behind him. Each metre
        # behind the job counts BEHIND_JOB_W extra metres.
        gdist = {b.unum: math.hypot(b.x + hl, b.y) for b in markers}
        tgd = {k: math.hypot(tpos[k][0] + hl, tpos[k][1]) for k in tids}
        pair = {}
        for p in markers:
            for k in tids:
                pair[(p.unum, k)] = (math.hypot(p.x + 3 * p.vx - tpos[k][0], p.y + 3 * p.vy - tpos[k][1])
                                     + BEHIND_JOB_W * max(0.0, gdist[p.unum] - tgd[k] - 1.0))
        for perm in itertools.permutations(markers, len(tids)):
            cost = sum(pair[(p.unum, k)] for p, k in zip(perm, tids))
            if cost < best_cost - 1e-9:
                best_cost, best_asg = cost, tuple((p.unum, k) for p, k in zip(perm, tids))

        for u, k in best_asg:
            if u == me.unum:
                tx, ty = tpos[k]
                name = tname[k]
                # zone jobs (path / screen / cover) for a player still BEHIND
                # the ball: don't chase from behind, cut the path instead
                if name != "mark" and not self._goal_side(snap, me, cx, cy):
                    # v22: if he cannot win the race to the carrier's path, run
                    # to his OWN job (replay: 3 recovering players all ran to the
                    # same spot near our goal and left the scorer free)
                    if name == "path":
                        tx, ty = self._recovery_point(snap, me, cx, cy, fallback=(tx, ty))
                    name = "recover"
                d = math.hypot(tx - me.x, ty - me.y)
                _note(agent, "def_" + name, team=len(snap.team), marks=len(marks))
                self._go(agent, snap, tx, ty, urgent=(name == "recover" or d > 6.0),
                         thr=0.4 if name == "mark" else 1.0)
                return

        # zone: formation slot pulled towards the ball (keeps shape, compact)
        hx, hy = _cxy(_home(agent))
        zx = hx + 0.3 * (cx - hx)
        zy = hy + 0.3 * (cy - hy)
        zx = max(min(zx, cx + 5.0), -hl + 12.0)
        _note(agent, "def_zone", team=len(snap.team))
        self._go(agent, snap, zx, zy)

    def _screen_point(self, snap, cx, cy, side):
        """On the line from the ball to one post, SCREEN_D from the ball:
        blocks the dribbling lane and the central pass on that side."""
        px, py = -snap.hl, side * snap.goal_hw
        d = math.hypot(px - cx, py - cy)
        k = min(SCREEN_D, 0.45 * d) / max(d, 1e-6)
        return cx + (px - cx) * k, cy + (py - cy) * k

    def _cover_point(self, snap, cx, cy):
        """Between the ball and the centre of our goal, in front of the box."""
        gx = -snap.hl
        d = math.hypot(cx - gx, cy)
        k = min(COVER_DIST, 0.5 * d) / max(d, 1e-6)
        return gx + (cx - gx) * k, cy * k

    def _mark_point(self, snap, o, cx, cy, is_top):
        """On the line from the opponent to the ball, MARK_GAP in front of
        him (cuts the pass to him), shaded slightly goal-side."""
        # v22: where he WILL be: a runner moves ~2.5x his reported velocity
        # per cycle (replay: our marker trailed a runner by 5 m all the way
        # into our box and he scored from the cross)
        rvx, rvy = run_vel(o)
        lx, ly = o.x + rvx * MARK_LEAD, o.y + rvy * MARK_LEAD
        bxu, byu = _unit(cx - lx, cy - ly)
        gx, gy = _unit(-snap.hl - lx, -ly)
        # ON the line man -> ball, a few metres in front of him (in the
        # passing lane, with room to react to his next move instead of
        # trailing behind him), shaded a little towards our goal - much more
        # when he is running at our goal or already close to it
        dgoal = math.hypot(lx + snap.hl, ly)
        at_goal = max(0.0, -(rvx * (lx + snap.hl) + rvy * ly) / max(dgoal, 1e-6))
        gw = 0.2 + 0.4 * min(1.0, at_goal / 0.8) + 0.3 * max(0.0, 1.0 - dgoal / 30.0)
        # v29: a man well behind the ball (closer to our goal) is a runner in
        # behind - stand between him and the goal, not between him and the ball
        gw = min(gw, 0.8)
        dx, dy = _unit((1.0 - gw) * bxu + gw * gx, (1.0 - gw) * byu + gw * gy)
        dist_b = math.hypot(cx - lx, cy - ly)
        gap = min(max(MARK_GAP_MIN, 0.2 * dist_b), MARK_GAP, 0.5 * dist_b)
        return lx + dx * gap, ly + dy * gap

    def _try_tackle(self, agent, snap, min_p):
        """Tackle if the win chance is high enough and the foul chance is low
        enough for this part of the pitch. tackle_probability is only > 0 when
        the ball is in front of us within reach, so this is cheap to ask."""
        wm = agent.wm
        me = snap.me
        hl = snap.hl
        try:
            tp = float(wm.self.tackle_probability)
            fp = float(wm.self.foul_probability)
        except Exception:
            return False
        in_box = snap.bx < -hl + PEN_LEN and abs(snap.by) < PEN_HW
        fmax = (FOUL_MAX_BOX if in_box
                else FOUL_MAX_THIRD if snap.bx < -hl / 3.0 else FOUL_MAX)
        if tp < min_p or fp > fmax:
            # diagnostic: close to the ball but no tackle -> why? (bucketed
            # tackle / foul probability shows up in `a3.py why`)
            if math.hypot(snap.bx - me.x, snap.by - me.y) < 2.5:
                _note(agent, "tk_no_p%d_f%d" % (min(9, int(tp * 10)), min(9, int(fp * 10))))
            return False
        if snap.bx < -hl / 3.0:   # own third: knock it wide and forward
            tx, ty = snap.bx + 20.0, (snap.hw if snap.by >= 0 else -snap.hw)
        else:
            tx, ty = hl, 0.0
        ang = math.degrees(math.atan2(ty - snap.by, tx - snap.bx))
        _note(agent, "tackle", p=round(tp, 2), f=round(fp, 2))
        _A.tackle(agent, _norm180(ang - me.body), foul=False)
        return True

    def _press_back(self, agent, snap):
        """Second presser coming from behind the carrier: harass him so he
        has to protect the ball while the front presser blocks. Tackles from
        behind foul easily, so it uses the strictest foul limit."""
        me = snap.me
        try:
            tp = float(agent.wm.self.tackle_probability)
            fp = float(agent.wm.self.foul_probability)
        except Exception:
            tp, fp = 0.0, 1.0
        if self._try_tackle(agent, snap, TACKLE_MIN):
            return
        bd = math.hypot(snap.bx - me.x, snap.by - me.y)
        _note(agent, "press_back")
        _A.go_to_point(agent, Vector2D(snap.bx + snap.bvx, snap.by + snap.bvy),
                           max_dash_power=self._dash(agent, snap, True, bd),
                           distance_threshold=0.3)
        _A.turn_to_ball(agent)

    def _press(self, agent, snap, cx, cy, top_recv, opp_kick, our_min, opp_min):
        """Front presser (v11). Replays showed that chasing a dribbler from
        behind never works (he runs as fast as we do) and that approaching
        from the side never gives a tackle (the ball must be IN FRONT of us).
        So: get onto his line to our goal, BLOCK_D in front of him, face him,
        and tackle when the ball comes into our tackle cone."""
        me = snap.me
        hl = snap.hl
        if self._try_tackle(agent, snap, TACKLE_MIN):
            return
        if not opp_kick and our_min <= opp_min + 1:
            _note(agent, "press_intercept")
            _A.intercept(agent)
            return
        # where the dribble is going (ball + its velocity over 2 cycles)
        lx, ly = snap.bx + 2.0 * snap.bvx, snap.by + 2.0 * snap.bvy
        # v22: along the way he is REALLY going (replays: carriers ran 30 m
        # down the touchline while we blocked the straight line to goal)
        ux, uy, _ = self._carrier_dir(snap, carrier_xy=(lx, ly))
        if not self._goal_side(snap, me, lx, ly, slack=-2.0):
            # clearly behind him: race to a point on his path we can reach first
            rx, ry = self._recovery_point(snap, me, lx, ly, heading=True)
            _note(agent, "press_recover")
            self._go(agent, snap, rx, ry, urgent=True, thr=0.5)
            return
        px, py = lx + ux * BLOCK_D, ly + uy * BLOCK_D
        bd = math.hypot(snap.bx - me.x, snap.by - me.y)
        if bd > CLOSE_DOWN + 2.0 and math.hypot(px - me.x, py - me.y) > 4.0:
            # goal-side but still far: cut him off where we get first instead
            # of running after a block point that moves away with him
            rx, ry = self._recovery_point(snap, me, lx, ly, heading=True)
            _note(agent, "press_cut")
            self._go(agent, snap, rx, ry, urgent=True, thr=0.5)
            return
        if math.hypot(px - me.x, py - me.y) < 1.0 or bd < BLOCK_D + 0.5:
            # in position: face the ball; if it is close, step into it
            ang = math.degrees(math.atan2(snap.by - me.y, snap.bx - me.x))
            if abs(_norm180(ang - me.body)) > 25.0:
                _note(agent, "press_face")
                _A.turn_to_ball(agent)
                return
            _note(agent, "press_block")
            _A.go_to_point(agent, Vector2D(snap.bx + snap.bvx, snap.by + snap.bvy),
                               max_dash_power=self._dash(agent, snap, True, bd),
                               distance_threshold=0.3)
            _A.turn_to_ball(agent)
            return
        _note(agent, "press")
        _A.go_to_point(agent, Vector2D(px, py),
                           max_dash_power=self._dash(agent, snap, True, bd),
                           distance_threshold=0.5)
        _A.turn_to_ball(agent)

    def _goal_side(self, snap, b, cx, cy, slack=0.5):
        """Is b between (cx, cy) and our goal (closer to our goal)?
        slack > 0 is strict; slack < 0 also accepts players slightly behind."""
        return math.hypot(b.x + snap.hl, b.y) < math.hypot(cx + snap.hl, cy) - slack

    def _carrier_dir(self, snap, carrier_xy=None):
        """Unit direction the ball carrier is heading + his speed: a blend of
        where he is really going (ball velocity while he dribbles, else his own
        running velocity) and the straight line to our goal."""
        cx, cy = carrier_xy if carrier_xy is not None else (snap.bx, snap.by)
        gx, gy = _unit(-snap.hl - cx, -cy)
        car = min(snap.opps, key=lambda o: math.hypot(o.x - snap.bx, o.y - snap.by),
                  default=None)
        hvx, hvy = snap.bvx, snap.bvy
        if math.hypot(hvx, hvy) < 0.3 and car is not None:
            hvx, hvy = run_vel(car)
        hs = math.hypot(hvx, hvy)
        if hs < 0.2:
            return gx, gy, 0.0
        hx, hy = hvx / hs, hvy / hs
        if hx * gx + hy * gy < -0.2:          # running away from our goal
            return gx, gy, hs
        ux, uy = _unit(CARRIER_HEAD_W * hx + (1 - CARRIER_HEAD_W) * gx,
                       CARRIER_HEAD_W * hy + (1 - CARRIER_HEAD_W) * gy)
        return ux, uy, hs

    def _recovery_point(self, snap, b, cx, cy, heading=False, fallback=None):
        """First point on the carrier's line to our goal (or, heading=True, on
        the way he is really going) that b can reach BEFORE the carrier (who
        dribbles at ~CARRIER_SPEED). If none, a point in front of our goal on
        that line."""
        gx = -snap.hl
        L = math.hypot(gx - cx, cy)
        ux, uy = _unit(gx - cx, -cy)
        if heading:
            ux, uy, _ = self._carrier_dir(snap, carrier_xy=(cx, cy))
            L = min(L, 30.0)
        # how fast is he actually going? (a shielding / standing carrier is
        # not running away; dribbling pushes the ball at about his speed)
        v = min(self.L["carrier"], max(0.35, 1.2 * math.hypot(snap.bvx, snap.bvy)))
        s = 2.0
        while s < L - 4.0:
            px, py = cx + ux * s, cy + uy * s
            if not snap.in_field(px, py, 0.5):
                break
            if reach(b, px, py, 0.5) <= s / v - 1.0:
                return px, py
            s += 2.0
        if fallback is not None:
            return fallback
        if heading:
            ux, uy = _unit(gx - cx, -cy)
            L = math.hypot(gx - cx, cy)
        # cannot win the race anywhere: get goal-side of him a bit further on
        # his path (not all the way back to the goal mouth)
        k = min(max(6.0, 2.0 * math.hypot(b.x - cx, b.y - cy)), max(L - 10.0, 0.0))
        return cx + ux * k, cy + uy * k

    def _opp_threats(self, snap, carrier, cx, cy):
        """Depth-1 adversarial step: be their ball carrier, score his passes
        with OUR model (our players as interceptors), keep the best 2-3."""
        evs = []
        defenders = snap.ours_all
        for o in snap.opps:
            if o.goalie or (carrier is not None and o.key == carrier.key):
                continue
            # their receiver offside w.r.t. our line -> not a legal target
            if o.x < snap.our_line - 0.5 and o.x < 0.0:
                continue
            rx, ry = o.x + o.vx, o.y + o.vy
            L = math.hypot(rx - cx, ry - cy)
            if L < PASS_MIN or L > PASS_MAX:
                continue
            n = arrive_steps(L, 2.2, snap.decay)
            if n is None:
                continue
            v0 = first_speed(L, n, snap.decay, snap.vmax)
            res = kick_eval(snap, cx, cy, rx, ry, v0, 1, o, defenders,
                            tmax=min(40, n + 10), mode="pass")
            if res is None:
                continue
            margin, tend, ix, iy, endm, ex, ey = res
            p = _sig(margin, PASS_M0, PASS_K)
            ev = p * H(snap, -ex, ey, endm) + (1 - p) * Loss(snap, -ix)   # mirrored
            evs.append((ev, o.key, o))
        evs.sort(key=lambda t: (-t[0], t[1]))
        top = [t for t in evs[:3] if t[0] > 0.0]
        tot = sum(t[0] for t in top)
        weights = {k: ev / tot for ev, k, _ in top} if tot > 0 else {}
        return weights, (top[0][2] if top else None)

    # ------------------------------------------------------------ attacking
    def _pass_score(self, snap, hx, hy, px, py, v=2.3):
        """Chance a pass from the holder to a player standing at (px, py) works."""
        L = math.hypot(px - hx, py - hy)
        if L < 2.0:
            return 0.5
        if L > PASS_MAX:
            return 0.0
        n = arrive_steps(L, v, snap.decay) or arrive_steps(L, snap.vmax, snap.decay)
        if n is None:
            return 0.0
        v0 = first_speed(L, n, snap.decay, snap.vmax)
        res = kick_eval(snap, hx, hy, px, py, v0, 1, None, snap.opps,
                        tmax=min(40, n + 6), mode="virtual")
        if res is None:
            return 0.0
        return _sig(res[0], self._pm0(), PASS_K)

    def _space(self, snap, x, y):
        return min((math.hypot(o.x - x, o.y - y) for o in snap.opps), default=30.0)

    def _support_target(self, snap, hx, hy):
        """Safe back/side option ~7-9 m from the holder, lane unblocked."""
        best, bp = -1e9, (hx - 6.0, hy)
        for r in SUP_RADII:
            for a in SUP_ANGLES:
                ra = math.radians(a)
                px, py = hx + r * math.cos(ra), hy + r * math.sin(ra)
                if not snap.in_field(px, py, 2.0):
                    continue
                s = self._pass_score(snap, hx, hy, px, py, v=1.9)
                s += 0.05 * min(self._space(snap, px, py), 10.0)
                if px < -snap.hl + PEN_LEN and abs(py) < PEN_HW:
                    s -= 1.0      # don't offer back passes inside our own box
                if s > best:
                    best, bp = s, (px, py)
        return bp

    def _runner_target(self, snap, hx, hy, beyond_ok):
        """Depth: level with their last defender (onside) or in behind once the
        ball has been played."""
        if beyond_ok:
            xr = min(hx + RUN_AHEAD, snap.hl - 8.0)
        else:
            xr = min(snap.offside - 1.0, hx + RUN_AHEAD, snap.hl - 8.0)
        best, bp = -1e9, (xr, hy)
        for y in (-14.0, -7.0, 0.0, 7.0, 14.0):
            if not snap.in_field(xr, y, 2.0):
                continue
            s = (0.5 * self._pass_score(snap, hx, hy, xr, y, v=2.6)
                 + 0.5 * min(self._space(snap, xr, y), 10.0) / 10.0
                 + 0.2 * (1.0 - abs(y) / snap.hw))
            if s > best:
                best, bp = s, (xr, y)
        return bp

    def _spot_score(self, snap, hx, hy, px, py, avoid):
        p = self._pass_score(snap, hx, hy, px, py)
        clear = min((reach(o, px, py, PLAYER_R) for o in snap.opps), default=99)
        s = p * H(snap, px, py, clear)
        for ax, ay in avoid:
            d = math.hypot(px - ax, py - ay)
            if d < SPACING:
                s -= 0.6 * (1.0 - d / SPACING)
        return s

    def _attack(self, agent, snap, hx, hy, exclude, beyond_ok, pred=None):
        """Attacking shape while a teammate has the ball (v11):
             ANCHOR    deep supporter + last defender: ~15 m behind the ball,
                       central, goal-side (safe back pass AND someone already
                       between the ball and our goal when we lose it)
             WING far  wide forward on the far flank, level with their line
             WING near wide forward on the ball's flank, level with their line
        Priority anchor > far wing > near wing (fewer players -> fewer roles).
        Assignment minimises total distance from where each player WILL be."""
        me = snap.me
        mem = self.mem
        exclude = set(exclude)
        if pred is not None and pred["kind"] == "pass":
            if pred["recv"].unum == me.unum:
                _note(agent, "att_receive", p=round(pred["p"], 2))
                self._go(agent, snap, pred["tx"], pred["ty"], urgent=True, thr=0.5)
                return
            exclude.add(pred["recv"].unum)
        players = [b for b in snap.team if b.unum not in exclude]
        if me.unum in exclude or not players:
            _note(agent, "att_home")
            self._go(agent, snap, *_cxy(_home(agent)))
            return

        # breakaway: the holder has a clear run -> everybody positions as if
        # he were already BRK_LEAD further on (so passes stay possible as he
        # dribbles) and runs there
        brk = self._breakaway(snap, hx, hy)
        if (brk and pred is not None and pred["kind"] == "pass"
                and pred["p"] >= GREEDY_P and pred["tx"] >= hx + BRK_PASS_AHEAD):
            brk = False          # he will pass forward instead (same test as his)
        brk_tag = "_brk" if brk else ""
        if brk:
            ga = math.atan2(-hy, snap.hl - hx)
            hx, hy = hx + BRK_LEAD * math.cos(ga), hy + BRK_LEAD * math.sin(ga)
        anchor = self._anchor_point(snap, hx, hy)
        side = 1.0 if hy >= 0 else -1.0
        ahead = BRK_AHEAD if brk else WING_AHEAD
        near = self._wing_spot(snap, hx, hy, side, ahead)
        # does the holder have anybody to pass to right now?
        best_opt = max((self._pass_score(snap, hx, hy, bb.x, bb.y) for bb in snap.team
                        if math.hypot(bb.x - hx, bb.y - hy) > 2.0), default=0.0)
        push = best_opt < NO_OPTION_P
        # no far wing: anchor + a close SUPPORT next to the holder + the near
        # wing. If the holder has no option at all, the support "shows" first
        # (gets the job before the wing when fewer players are available).
        if hx > BOX_ATTACK_X and abs(hy) > 6.0 and not brk:
            # v27 box runs: the ball is near their byline - one player to the
            # cutback zone (penalty spot), one to the far post
            sd = 1.0 if hy >= 0 else -1.0
            cut = (min(snap.hl - CUTBACK_SPOT[0], snap.offside - 0.8), -sd * CUTBACK_SPOT[1])
            far = (min(snap.hl - FARPOST_SPOT[0], snap.offside - 0.8), -sd * FARPOST_SPOT[1])
            tasks = [("anchor", anchor), ("box_cut", cut), ("box_far", far)][:len(players)]
        elif push:
            # nobody available: one player comes SHORT first
            tasks = [("anchor", anchor), ("show", self._show_spot(snap, hx, hy)),
                     ("wing_near", near)][:len(players)]
        else:
            # options exist: create play AHEAD - near wing + a forward in the
            # central channel / other side of the holder, both as far forward
            # as a pass still reaches them
            tasks = [("anchor", anchor), ("wing_near", near),
                     ("forward", self._forward_spot(snap, hx, hy, side, ahead))][:len(players)]
            # v31 decoy run: the third attacker runs AT the defender guarding
            # the predicted receiver's spot (from the other side), so that
            # defender has to choose and the receiver gets more room
            dec = self._decoy_spot(snap, hx, hy, pred) if DECOY else None
            if dec is not None and len(tasks) == 3:
                tasks[2] = ("decoy", dec)

        def d(bb, t):
            return math.hypot(bb.x + 3 * bb.vx - t[0], bb.y + 3 * bb.vy - t[1])
        best_cost, best = 1e9, None
        for perm in itertools.permutations(players, len(tasks)):
            cost = sum(d(p, t[1]) for p, t in zip(perm, tasks))
            if cost < best_cost - 1e-9:
                best_cost, best = cost, {p.unum: t for p, t in zip(perm, tasks)}
        role, (tx, ty) = best.get(me.unum, ("open", _cxy(_home(agent))))

        if role.startswith("box_"):
            # small search for space around the run target (keeps the run)
            tx, ty = self._get_open(snap, hx, hy, tx, ty, [(hx, hy)], False, radius=(2.5,))
            _note(agent, "att_" + role)
            self._go(agent, snap, tx, ty, urgent=True, thr=0.7)
            return

        if role == "decoy":
            if not beyond_ok:
                tx = min(tx, snap.offside - 0.7)
            _note(agent, "att_decoy")
            self._go(agent, snap, tx, ty, urgent=True, thr=0.7)
            return

        if role == "show":
            tx, ty = self._clamp_near(snap, hx, hy, tx, ty)
            _note(agent, "att_" + role + brk_tag)
            self._go(agent, snap, tx, ty, urgent=(role == "show" or brk), thr=0.7)
            return

        if role == "anchor":
            # small refinement: keep a clear lane for the back pass
            tx, ty = self._get_open(snap, hx, hy, tx, ty, [(hx, hy)], False, radius=(3.0,),
                                    allow_ahead=False)
            _note(agent, "att_anchor" + brk_tag)
            self._go(agent, snap, tx, ty, urgent=me.x > hx or brk)
            return

        # wings / forward: pass-and-move first, then get open around the spot
        if mem.get("pm_until", -1) > snap.cycle and role.startswith(("wing", "forward")):
            fx = me.x + 8.0
            if not beyond_ok:
                fx = min(fx, snap.offside - 0.7)
            fx = min(fx, snap.hl - 6.0)
            # only if there is real room ahead (vs a high line the run would
            # stop at the offside line right next to the ball = crowding)
            # and a pass can still reach him there
            if (fx - me.x >= 4.0
                    and self._pass_score(snap, hx, hy, fx, me.y) >= OPEN_REACH_P):
                tx, ty = fx, me.y
                role = "pass_move"
        avoid = [(hx, hy)] + [(bb.x, bb.y) for bb in snap.field_mates]
        min_x = None
        if role.startswith("wing") or role.startswith("forward"):
            min_x = min(hx + WING_MIN_AHEAD, snap.offside - 1.5, snap.hl - 6.0)
        tx, ty = self._get_open(snap, hx, hy, tx, ty, avoid, beyond_ok, min_x=min_x)
        if role.startswith(("wing", "forward")):
            tx, ty, checked = self._checking(snap, hx, hy, tx, ty, avoid, beyond_ok)
            if checked:
                role += "_" + checked
        if not beyond_ok:
            tx = min(tx, snap.offside - 0.7)
        tx, ty = self._clamp_near(snap, hx, hy, tx, ty)
        dist = math.hypot(tx - me.x, ty - me.y)
        urgent = (role == "pass_move" or "burst" in role or (push and dist > 6.0)
                  or beyond_ok or brk)
        _note(agent, "att_" + role + ("_push" if push else "") + brk_tag)
        self._go(agent, snap, tx, ty, urgent=urgent)

    def _decoy_spot(self, snap, hx, hy, pred):
        """v31: where a decoy runner goes. Only when a pass to a teammate in
        their half is predicted and one defender stands near its target: a
        point 4 m past that defender on the far side from the target, 2 m
        towards their goal (onside, inside the pitch). None otherwise."""
        if pred is None or pred["kind"] != "pass" or hx < -5.0:
            return None
        tx, ty = pred.get("ex", pred["tx"]), pred.get("ey", pred["ty"])
        if tx < 0.0:
            return None
        d = min((o for o in snap.opps if not o.goalie),
                key=lambda o: math.hypot(o.x - tx, o.y - ty), default=None)
        if d is None or math.hypot(d.x - tx, d.y - ty) > DECOY_R:
            return None
        ux, uy = _unit(d.x - tx, d.y - ty)
        px, py = d.x + 4.0 * ux + 2.0, d.y + 4.0 * uy
        px = min(px, snap.offside - 0.8, snap.hl - 4.0)
        py = max(-snap.hw + 3.0, min(snap.hw - 3.0, py))
        if math.hypot(px - tx, py - ty) < 5.0:
            return None             # would crowd the receiver
        return px, py

    def _better_forward_pass(self, snap, hx, hy, holder, opps, pressure, drib_v):
        """A teammate clearly better placed than carrying the ball: a very safe
        (p >= GREEDY_P) pass at least BRK_PASS_AHEAD further forward whose
        value beats the best breakaway dribble by BRK_PASS_GAP. None if not."""
        mates = [b for b in snap.team if b.unum != holder.unum]
        best = None
        for c in self._candidates(snap, hx, hy, holder, mates, snap.offside, opps, pressure):
            if (c["kind"] == "pass" and c["p"] >= GREEDY_P and c["tx"] >= hx + BRK_PASS_AHEAD
                    and c["v"] >= drib_v + BRK_PASS_GAP):
                if best is None or c["v"] > best["v"]:
                    best = c
        return best

    def _breakaway(self, snap, hx, hy):
        """Clear run: no opponent field player within BRK_CLEAR in front of the
        ball (inside a BRK_CONE cone around its line to their goal), nobody at
        all within 4 m, and the nearest opponent at least OPEN_PRESSURE cycles
        away. Same answer for the holder and for every teammate."""
        ga = math.degrees(math.atan2(-hy, snap.hl - hx))
        for o in snap.opps:
            dx, dy = o.x - hx, o.y - hy
            d = math.hypot(dx, dy)
            if d < 4.0:
                return False
            if o.goalie:
                continue
            if d < BRK_CLEAR and abs(_norm180(math.degrees(math.atan2(dy, dx)) - ga)) < BRK_CONE:
                return False
        return _pressure(snap, hx, hy) >= OPEN_PRESSURE

    def _clamp_near(self, snap, hx, hy, tx, ty):
        """Distance limit (MAX_SUPPORT_DIST) plus a REACH limit: a pass from
        the holder must get to the spot with P >= REACH_P. If not, the spot
        slides towards the holder (2 m steps) until it does, never closer than
        REACH_MIN_D; if no point on that line qualifies, the most reachable one."""
        dt = math.hypot(tx - hx, ty - hy)
        if dt > MAX_SUPPORT_DIST:
            k = MAX_SUPPORT_DIST / dt
            tx, ty = hx + (tx - hx) * k, hy + (ty - hy) * k
            dt = MAX_SUPPORT_DIST
        if dt <= REACH_MIN_D:
            return tx, ty
        ux, uy = (tx - hx) / dt, (ty - hy) / dt
        best_p, best = -1.0, (tx, ty)
        d = dt
        while d >= REACH_MIN_D:
            px, py = hx + ux * d, hy + uy * d
            p = self._pass_score(snap, hx, hy, px, py)
            if p >= REACH_P:
                return px, py
            if p > best_p + 0.05:
                best_p, best = p, (px, py)
            d -= 2.0
            if time.perf_counter() > self._deadline:
                break
        return best

    def _forward_spot(self, snap, hx, hy, side, ahead=WING_AHEAD):
        """Second attacker ahead of the ball: central channel or the other side
        of the holder from the wing; the MOST FORWARD point a pass still
        reaches (P >= WING_PASS_MIN), never past the offside line."""
        top = min(hx + ahead, snap.offside - 1.0, snap.hl - 6.0)
        best, bp = -1.0, None
        # v22: never level with / behind the ball (vs a high press the whole
        # team used to retreat WITH the holder, leaving him no forward option)
        low = min(hx + FWD_MIN_AHEAD, top)
        for fy in (hy - side * 10.0, 0.0, hy - side * 4.0):
            fy = max(-snap.hw + 3.0, min(snap.hw - 3.0, fy))
            x = top
            while x >= low:
                p = self._pass_score(snap, hx, hy, x, fy)
                if p >= WING_PASS_MIN and math.hypot(x - hx, fy - hy) >= 8.0:
                    return x, fy
                if p - 0.02 * (x - low) > best:        # v22: ties -> closer to the ball
                    best, bp = p - 0.02 * (x - low), (x, fy)
                x -= 4.0
                if time.perf_counter() > self._deadline:
                    break
        return bp if bp is not None else (low, hy - side * 10.0)

    def _wing_spot(self, snap, hx, hy, side, ahead=WING_AHEAD):
        """Wide forward: as far forward as a pass from the holder can still
        reach him (P >= WING_PASS_MIN), never past the offside line. Tries
        spots from far ahead back towards the ball on his flank; if nothing on
        the flank is reachable, comes inside (|y| ~ 10)."""
        top = min(hx + ahead, snap.offside - 1.0, snap.hl - 6.0)
        best, bp = -1.0, None
        low = min(hx + WING_MIN_AHEAD, top)          # v22 (was hx - 6)
        for wy in (side * WING_Y, side * (WING_Y - 8.0)):
            if abs(wy) > snap.hw - 3.0:
                wy = side * (snap.hw - 3.0)
            x = top
            while x >= low:
                p = self._pass_score(snap, hx, hy, x, wy)
                if p >= WING_PASS_MIN and math.hypot(x - hx, wy - hy) >= 8.0:
                    return x, wy                    # most forward reachable spot
                if p - 0.02 * (x - low) > best:        # v22: ties -> closer to the ball
                    best, bp = p - 0.02 * (x - low), (x, wy)
                x -= 4.0
                if time.perf_counter() > self._deadline:
                    break
        return bp if bp is not None else (low, side * WING_Y)

    def _show_spot(self, snap, hx, hy):
        """Free spot ~SHOW_R from the holder that a short pass reaches,
        preferring forward and away from the touchline."""
        best, bp = -1e9, (hx - SHOW_R, hy)
        for i in range(12):
            a = math.radians(i * 30.0)
            px, py = hx + SHOW_R * math.cos(a), hy + SHOW_R * math.sin(a)
            if not snap.in_field(px, py, 2.5) or px > snap.offside - 0.7:
                continue
            sc = (1.5 * self._pass_score(snap, hx, hy, px, py, v=1.8)
                  + 0.3 * min(self._space(snap, px, py), 8.0) / 8.0
                  + 0.25 * (px - hx) / SHOW_R)
            if sc > best:
                best, bp = sc, (px, py)
        return bp

    def _anchor_point(self, snap, hx, hy):
        """Deep supporter / last defender: behind the ball, central."""
        ax = min(hx - ANCHOR_BEHIND, ANCHOR_MAX_X)
        ay = 0.35 * hy
        # v22 rest defence: stay goal-side of their most advanced field player
        # (goal 1 of the replay: their striker was already behind our anchor
        # when we lost the ball, and ran 45 m to score unchallenged)
        fwd = min((o for o in snap.opps if not o.goalie), key=lambda o: o.x, default=None)
        if fwd is not None and fwd.x - ANCHOR_GUARD < ax:
            ax = max(fwd.x - ANCHOR_GUARD, hx - ANCHOR_MAX_BEHIND)
            ay = 0.5 * ay + 0.5 * fwd.y
        ax = max(ax, -snap.hl + 8.0)
        return ax, ay

    def _get_open(self, snap, hx, hy, rx, ry, avoid, beyond_ok, radius=(3.0, 6.0),
                  allow_ahead=True, min_x=None):
        """Every cycle: points around HIMSELF and around his role spot, scored by
          PASS_W * P(pass from the holder reaches me here)
          + OPEN_W * free space + GOALWARD_W * metres ahead of the ball
          - ROLE_W * distance from the role spot (keeps the shape)
          - spacing penalty (WING_SPACING from teammates / holder)
          - same-direction penalty (ANGLE_SPREAD seen from the holder).
        Position value H is NOT used here any more: with a big W_PROG it made
        the furthest-forward spot best for everybody -> all at the offside
        line next to the ball (the centre-circle crowd seen in the replays)."""
        me = snap.me
        pts = [(rx, ry), (me.x, me.y)]
        for r in radius:
            for i in range(8):
                a = math.radians(i * 45.0)
                pts.append((me.x + r * math.cos(a), me.y + r * math.sin(a)))
                pts.append((rx + r * math.cos(a + 0.39), ry + r * math.sin(a + 0.39)))
        best, bp = -1e9, (rx, ry)
        for px, py in pts:
            if time.perf_counter() > self._deadline:
                break
            if not snap.in_field(px, py, 2.0):
                continue
            if not beyond_ok and px > snap.offside - 0.7:
                continue
            if not allow_ahead and px > hx - 5.0:
                continue
            if min_x is not None and px < min_x and (px, py) != (rx, ry):
                continue          # v22: forwards stay ahead of the ball
            if math.hypot(px - hx, py - hy) < 4.0:
                continue
            sc = self._open_score(snap, hx, hy, px, py, rx, ry, avoid)
            if sc > best:
                best, bp = sc, (px, py)
        key = "open_pt_a" if not allow_ahead else "open_pt"
        prev = self.mem.get(key)
        if prev is not None and math.hypot(prev[0] - bp[0], prev[1] - bp[1]) > 1.0:
            if (snap.in_field(prev[0], prev[1], 2.0)
                    and (beyond_ok or prev[0] <= snap.offside - 0.7)
                    and (min_x is None or prev[0] >= min_x)
                    and math.hypot(prev[0] - rx, prev[1] - ry) < 8.0):
                if self._open_score(snap, hx, hy, prev[0], prev[1], rx, ry, avoid) + 0.1 >= best:
                    bp = prev
        self.mem[key] = bp
        return bp

    def _open_score(self, snap, hx, hy, px, py, rx, ry, avoid):
        p = self._pass_score(snap, hx, hy, px, py)
        # a spot no pass can reach creates nothing: only reachable spots
        # compete (unreachable ones only if nothing reachable exists)
        unreach = OPEN_UNREACH_PEN if p < OPEN_REACH_P else 0.0
        space = min(self._space(snap, px, py), 10.0) / 10.0
        s = (PASS_W * p + OPEN_W * space
             + GOALWARD_W * max(0.0, px - hx) / 20.0
             - ROLE_W * math.hypot(px - rx, py - ry) / 10.0 - unreach)
        pa = math.atan2(py - hy, px - hx)
        for b in snap.field_mates:
            if math.hypot(b.x - hx, b.y - hy) < 2.0:
                continue
            da = abs(_norm180(math.degrees(pa - math.atan2(b.y - hy, b.x - hx))))
            if da < ANGLE_SPREAD:
                s -= 0.4 * (1.0 - da / ANGLE_SPREAD)
        for ax, ay in avoid:
            dd = math.hypot(px - ax, py - ay)
            if dd < WING_SPACING:
                s -= 1.0 * (1.0 - dd / WING_SPACING)
        return s

    def _open_spot(self, agent, snap, hx, hy, avoid, beyond_ok, push=False):
        """Small grid around the (shifted) formation slot, scored by how good
        a pass to me there would be (YuShan-style local search)."""
        home = _home(agent)
        h0x, h0y = _cxy(home)
        h0x = max(h0x, hx - COMPACT)
        if push:   # nobody to pass to: come forward and closer to the ball
            h0x = max(h0x, hx + 6.0)
            h0y = 0.7 * h0y + 0.3 * hy
        if not beyond_ok:
            h0x = min(h0x, snap.offside - 1.0)
        pts = [(h0x, h0y)]
        for r in (4.0, 8.0):
            for i in range(8):
                a = math.radians(i * 45.0)
                pts.append((h0x + r * math.cos(a), h0y + r * math.sin(a)))
        best, bp = -1e9, (h0x, h0y)
        for px, py in pts:
            if time.perf_counter() > self._deadline:
                break
            if not snap.in_field(px, py, 2.0):
                continue
            if not beyond_ok and px > snap.offside - 0.7:
                continue
            s = self._spot_score(snap, hx, hy, px, py, avoid)
            if s > best:
                best, bp = s, (px, py)
        # hysteresis on the chosen spot
        prev = self.mem.get("open_t")
        if prev is not None and snap.in_field(prev[0], prev[1], 2.0):
            ps = self._spot_score(snap, hx, hy, prev[0], prev[1], avoid)
            if ps + 0.15 >= best:
                bp = prev
        self.mem["open_t"] = bp
        return bp

    def _checking(self, snap, hx, hy, tx, ty, avoid, beyond_ok):
        """If tightly marked and not open: check towards the ball, then burst
        (behind the marker's back if possible). Phase offset by uniform number
        so teammates don't all do it at once."""
        me = snap.me
        if not snap.opps:
            return tx, ty, None
        marker = min(snap.opps, key=lambda o: reach(o, me.x, me.y, PLAYER_R))
        if reach(marker, me.x, me.y, PLAYER_R) > 2:
            return tx, ty, None
        if self._pass_score(snap, hx, hy, me.x, me.y) >= 0.6:
            return tx, ty, None
        phase = (snap.cycle + 7 * me.unum) % 16
        opts = [((tx, ty), None, 0.0)]
        if phase < 5:
            ux, uy = _unit(hx - me.x, hy - me.y)
            opts.append(((me.x + 3.0 * ux, me.y + 3.0 * uy), "check", 0.0))
        elif phase < 9:
            bx_ = me.x + 4.5
            if not beyond_ok:
                bx_ = min(bx_, snap.offside - 0.7)
            side = 1.0 if me.y >= marker.y else -1.0
            by_ = me.y + 1.5 * side
            ang = math.degrees(math.atan2(by_ - marker.y, bx_ - marker.x))
            behind = abs(_norm180(ang - marker.body)) > 90.0
            opts.append(((bx_, by_), "burst", 0.15 if behind else 0.0))
        best, bt, lab = -1e9, (tx, ty), None
        for (px, py), name, bonus in opts:
            if not snap.in_field(px, py, 1.5):
                continue
            s = self._spot_score(snap, hx, hy, px, py, avoid) + bonus
            if s > best:
                best, bt, lab = s, (px, py), name
        return bt[0], bt[1], lab


# STARTER CODE ENDS HERE #


# ==========================================================================
# ENGINE: LightAgent (opponent model)
# ==========================================================================
#
# Agent1 - a hybrid planner built only from classic search / probabilistic
# reasoning techniques.
#
#   * A kinematic world model (ball flight + player reach time), evaluated in
#     batch with numpy. It plays the role of the transition model in a
#     model-based planner.
#   * ON THE BALL: an expectimax-style action search. Shots, passes (direct,
#     lead and through balls) and self-passes/dribbles are generated, every
#     candidate is rolled out through the model and turned into a success
#     probability (chance node), and scored as
#         p * V(success state) + (1 - p) * V(failure state)
#     with a hand-built heuristic state evaluation V. The best few are then
#     expanded one more ply (beam search) to look at what the receiver can
#     do next, while the CPU budget allows (anytime).
#   * OFF THE BALL: every player computes the same team-wide allocation from
#     the same full state, so no communication is needed.
#       - who chases: minimum model reach time (ties by uniform number)
#       - defence: marking is a small CSP (variables = our defenders,
#         values = opponents to mark or 'cover'), solved with backtracking,
#         MRV variable ordering, forward checking and branch & bound on the
#         total cost.
#       - attack: support spots chosen by greedy local search over a grid
#         around a ball-relative anchor, scored for pass-lane openness,
#         threat and spacing.
#

T_MAX = 40              # rollout horizon in cycles
NEVER = 10000
L_GOAL_VALUE = 200.0      # value of a goal on the evaluation scale
LOSS_WEIGHT = 1.0       # weight of the opponent's value when we lose the ball
# The goalie as seen by a shot: fitted to the observed conversion of shots
# in our own match logs (it reacts late and covers less than a field
# player's model would suggest).
GOALIE_REACT = 2.0
GOALIE_CTRL = 1.2


def _lxy(v):
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
        p = 1.0 / (1.0 + np.exp(np.clip(sl / 0.6, -50.0, 50.0)))
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
        self.bx, self.by = _lxy(wm.ball.position)
        self.bvx, self.bvy = _lxy(wm.ball.velocity)
        me = wm.self
        self.unum = me.uniform_number
        self.mx, self.my = _lxy(me.position)
        self.mvx, self.mvy = _lxy(me.velocity)
        self.mbody = me.body_direction
        self.stamina = me.stamina
        self.offside_x = wm.offside_line_x

        def pack(players):
            n = len(players)
            P = np.zeros((n, 2)); V = np.zeros((n, 2)); U = np.zeros((n, 2))
            G = np.zeros(n, dtype=bool); num = []
            for i, p in enumerate(players):
                P[i] = _lxy(p.position)
                V[i] = _lxy(p.velocity)
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


class LightAgent(StarterAgent):

    # design switches (kept as class attributes so variants can be A/B tested)
    PHI = "hybrid"        # state value: "v2" | "new" | "hybrid"
    FOLLOW = True         # do not re-kick a soft kick that is rolling as planned
    REFINE = True         # second search ply (receiver's follow-up shot)
    OFFBALL = "v2"        # "v2": single chaser + blocker + marking CSP; "jobs"
    STAMINA = "tiered"    # "v2" | "tiered" (VTired: tiered)
    TACKLE = "v2"         # "v2": only at a held/arriving ball; "loose"

    def __init__(self):
        super().__init__()
        self.model = None
        self.snap = None
        self.last_plan = None        # (cycle, kind, qx, qy, n, extra, speed)
        self.offside_flag_until = -1
        self.recovering = False

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
        try:
            if self.plan_with_ball(agent, s):
                return
        except Exception as e:          # never leave the ball without an order
            _note(agent, "onball_error", err=str(e)[:80])
        _A.clear(agent)

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
            _note(agent, "follow", q=(round(px, 1), round(py, 1)))
            _A.go_to_point(agent, Vector2D(px, py),
                               max_dash_power=self.dash_power(agent, s, "max"),
                               distance_threshold=0.3)
            _A.turn_to_ball(agent)
        else:
            _note(agent, "release")
            _A.turn_to_ball(agent)
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
        for p in s.mates:
            if p.is_goalie:
                continue
            px, py = _lxy(p.position)
            if px > offx + 0.3:           # offside now: cannot receive
                continue
            vx, vy = _lxy(p.velocity)
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
        # (exponent bounded: margin is about -10000 when nobody of ours can
        # reach the ball, which overflowed exp; p is 0 there either way)
        p = 1.0 / (1.0 + np.exp(np.clip(-(margin - 1.0) * 1.3, -50.0, 50.0)))
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
            v_shot = pg * L_GOAL_VALUE + (1.0 - pg) * self.loss(Ix, Iy)
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
            _note(agent, "hold", v=round(v_hold, 1))
            _A.hold(agent)
            self.last_plan = None
            return True

        kind, qx, qy, n, vx, vy, Kb, extra = cands[best]
        self.last_plan = (s.cycle, kind, qx, qy, n, extra, math.hypot(vx, vy))
        tgt = Vector2D(qx, qy)
        _note(agent, kind, to=extra, q=(round(qx, 1), round(qy, 1)), n=n, K=Kb,
             p=round(float(p[best]), 2), v=round(float(score[best]), 1),
             ps=round(best_shot_p, 2), cont=int(best == cont))
        if kind == "shoot1":
            _A.kick_one_step(agent, tgt)
        elif kind == "shoot":
            _A.kick(agent, tgt, 1)
        elif kind == "pass":
            _A.kick(agent, tgt, n, accept_late=2)
        else:
            _A.kick(agent, tgt, n, accept_late=3)
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

    def refine(self, agent, s, cands, score, p, BP, t_m, who, best):
        """Beam search, second ply: expand the top candidates with the
        receiver's best follow-up (a shot from the reception point against
        the opponents' predicted positions), anytime under the CPU budget."""
        if not self.REFINE:
            return best
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
            follow = pg * L_GOAL_VALUE * 0.9 + (1.0 - pg) * float(self.loss(m.hl - 2.0, ry))
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
            _note(agent, "offball_error", err=str(e)[:80])
        _A.go_to_point(agent, _home(agent))

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
                _note(agent, "chase", t=self_min, opp=opp_min)
                _A.intercept(agent)
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
        _note(agent, "tackle", prob=round(prob, 2))
        _A.tackle(agent, rel, False)
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
        _note(agent, "support", q=(round(tx, 1), round(ty, 1)))
        _A.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=1.0)
        _A.turn_to_ball(agent)
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
        _note(agent, tag, o=jid, q=(round(tx, 1), round(ty, 1)))
        _A.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self.dash_power(agent, s, level),
                           distance_threshold=0.7)
        _A.turn_to_ball(agent)
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
            _note(agent, "press_ball")
            _A.intercept(agent)
            return True
        kk = int(min(k + 1, len(Dx) - 1))
        tx, ty = float(Dx[kk]), float(Dy[kk])
        _note(agent, "block", q=(round(tx, 1), round(ty, 1)), k=k)
        _A.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self.dash_power(agent, s, "max"),
                           distance_threshold=0.5)
        _A.intercept(agent)
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
                _note(agent, "chase", t=t_me, opp=t_opp)
                _A.intercept(agent)
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
            chosen.append(q)
            if u == s.unum:
                my_target = q
                break
        if my_target is None:
            return False
        tx, ty = my_target
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, "high" if d > 8.0 else "normal")
        _note(agent, "support", q=(round(tx, 1), round(ty, 1)))
        _A.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=1.0)
        _A.turn_to_ball(agent)
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

        defenders = [u for u in sorted(s.fnum) if u != blocker]
        pos = {u: s.fP[i] for i, u in enumerate(s.fnum)}

        def cost(u, job):
            px, py = pos[u]
            d = math.hypot(job[2] - px, job[3] - py)
            return (0.8 * d if job[0] == "cover" else d) - job[4]

        sol = self.csp_assign(defenders, jobs, cost)
        job = jobs[sol[s.unum]] if s.unum in sol else jobs[-1]
        tag, jid, tx, ty, _ = job
        d = math.hypot(tx - s.mx, ty - s.my)
        power = self.dash_power(agent, s, "high" if d > 3.0 else "normal")
        _note(agent, tag, q=(round(tx, 1), round(ty, 1)), o=jid)
        _A.go_to_point(agent, Vector2D(tx, ty), max_dash_power=power,
                           distance_threshold=0.7)
        _A.turn_to_ball(agent)
        return True



# ==========================================================================
# Opponent model: linear combinations of the three engines
#
#   off the ball  each opponent's position t cycles ahead is
#                     w . (inertia, HELIOS, Col, Light predictions)
#                 w: Bayesian linear regression on the simplex (MAP, found
#                 by steepest-ascent hill climbing). The prior
#                 is a fit to many teams' recordings (not FULL only), worth
#                 OPP_KAPPA samples; every observed movement in the match is
#                 added to the sufficient statistics and w is re-solved, so
#                 the weights drift towards THIS opponent as evidence grows.
#
#   on the ball   the carrier's next action is predicted by each engine
#                 deciding as him; the mixture weights are a Dirichlet over
#                 "which engine is he", updated by Bayes' rule from what he
#                 then really does (the online form of EM for a mixture).
#                 Our defence uses it: block his predicted dribble, mark /
#                 cut the lane to his predicted receivers.
#
#   "none"        both mixtures carry a component for "he is none of the
#                 three": an uninformative likelihood (carrier: kick or
#                 keep 50/50, any direction; movement: anywhere he can run
#                 to in OPP_H cycles). It only gains weight when every
#                 engine explains what he did worse than chance would, so an
#                 opponent unlike the models is detected (EM / Bayesian
#                 updating, as above). Its weight lowers the trust in the
#                 model: predictions are shrunk towards plain HELIOS
#                 (opponents where they stand, no carrier hint).
# ==========================================================================

OPP_PARTS = ("inertia", "helios", "col", "light")
OPP_W0 = (0.5314, 0.1604, 0.3082, 0.0)     # prior weights (fit to many teams)
OPP_Q0 = ((0.16791, 0.81267, 1.01435, 0.33968), (0.81267, 19.3998, 11.47361, 7.85224), (1.01435, 11.47361, 21.14621, 6.49803), (0.33968, 7.85224, 6.49803, 18.42854))     # prior sufficient statistics, per sample
OPP_B0 = (1.1077, 7.65533, 9.47261, 3.54336)
OPP_KAPPA = 150.0       # prior strength, in samples
OPP_H = 6               # horizon the weights are learned at (cycles)
OPP_REFIT = 8           # re-solve the weights after this many new samples
OPP_SIGMA = 1.5862   # per-axis noise of the mixture's prediction (m, EM)
OPP_NONE0 = 0.0205    # prior P(none) for movement (EM on recordings)
OPP_NONE_KAPPA = 60.0   # Beta prior strength (samples)

CAR_PARTS = ("helios", "col", "light", "none")
CAR_W0 = (0.4067, 0.4613, 0.082, 0.05)      # prior mixture weights (EM on recordings)
CAR_KAPPA = 12.0        # Dirichlet prior strength (observed decisions)
CAR_EVERY = 1           # re-predict the carrier every this many cycles
CAR_SIGMA = 25.0        # direction noise of a prediction (degrees)
CAR_CLASS = 0.75        # P(right class: kick vs keep the ball)
# mean over the circle of the direction kernel below, so the kernel divided
# by it is a density relative to "any direction" (= 1)
CAR_DIR_Z = CAR_SIGMA * math.sqrt(2.0 * math.pi) / 360.0 + 0.05


def _simplex_climb(Q, b, w0, step=0.25, tol=1e-4):
    """Steepest-ascent hill climbing on the simplex (here descending the
    squared error E(w) = w'Qw - 2 b'w): a move shifts `step` of weight from
    one part to another; the best improving move is taken, and the step is
    halved when no move improves. E is convex, so the point where no shift
    improves is the optimum over w >= 0, sum w = 1."""
    k = len(b)

    def err(w):
        return (sum(w[i] * Q[i][j] * w[j] for i in range(k) for j in range(k))
                - 2.0 * sum(b[i] * w[i] for i in range(k)))

    w = list(w0)
    e = err(w)
    while step > tol:
        best = None
        for i in range(k):
            for j in range(k):
                if i == j or w[j] <= 0.0:
                    continue
                d = min(step, w[j])
                w2 = list(w)
                w2[i] += d
                w2[j] -= d
                e2 = err(w2)
                if e2 < e - 1e-12 and (best is None or e2 < best[0]):
                    best = (e2, w2)
        if best is None:
            step *= 0.5
        else:
            e, w = best
    return w


def _ball_reach(ph, p, bx, by, bvx, bvy, ctrl, maxc=30):
    """First cycle player p can control the free-rolling ball."""
    for c in range(maxc + 1):
        x, y = ph.ball_at(bx, by, bvx, bvy, c)
        ix, iy = ph.inertia(p, c)
        need = math.hypot(x - ix, y - iy) - ctrl
        if need <= 0.0 or ph.cycles_to_reach(need) <= c:
            return c
    return maxc + 10


def _kick_rate(ph, p, bx, by):
    """rcssserver kick power rate for a ball at (bx, by)."""
    d = math.hypot(bx - p.x, by - p.y)
    ang = abs(_norm(math.degrees(math.atan2(by - p.y, bx - p.x)) - p.body))
    gap = max(0.0, d - ph.player_size - ph.ball_size)
    return ph.kick_power_rate * max(0.0, 1.0 - 0.25 * ang / 180.0
                                    - 0.25 * gap / ph.kickable_margin)


def _car_outcome(c0, holder, bx0, by0, now_cycle, bx, by, kick_us, kick_them):
    """What the carrier did, judged when the ball is next under someone's
    control. kick_them: unums of their players who can kick it now.
    -> None (not known yet), "lost" (we won it / out of play), or
    (kind, direction from where the ball was, or None)."""
    age = now_cycle - c0
    if kick_us:
        return "lost"
    moved = math.hypot(bx - bx0, by - by0)
    d = math.degrees(math.atan2(by - by0, bx - bx0)) if moved > 1.5 else None
    others = [u for u in kick_them if u != holder]
    if others:
        return ("kick", d)
    if holder in kick_them:
        return ("keep", d) if age >= 4 else None
    return "lost" if age > 12 else None


def _car_lik(pred, obs, bx0, by0):
    """Likelihood of what he did under one engine's prediction (class
    probability x direction density relative to 'any direction'), or under
    "none" (pred None: either class, any direction)."""
    okind, odir = obs
    if pred is None:
        return 0.5 * (0.8 if odir is None and okind == "keep" else 1.0)
    kind, tx, ty = pred
    pk = "kick" if kind == "kick" else "keep"
    lik = CAR_CLASS if pk == okind else 1.0 - CAR_CLASS
    if odir is not None and kind != "hold" and pk == okind:
        pdir = math.degrees(math.atan2(ty - by0, tx - bx0))
        e = abs(_norm(pdir - odir)) / CAR_SIGMA
        lik *= (math.exp(-0.5 * e * e) + 0.05) / CAR_DIR_Z
    elif odir is None and okind == "keep" and pk == okind:
        lik *= 1.0 if kind == "hold" else 0.6
    return lik


class _OppModel:
    ENGINES = {"helios": HeliosAgent, "col": ColAgent, "light": LightAgent}

    def __init__(self, sp):
        self.sp = sp
        self.w = list(OPP_W0)
        self.Q = [[OPP_KAPPA * q for q in row] for row in OPP_Q0]
        self.b = [OPP_KAPPA * x for x in OPP_B0]
        self.new = 0
        self.n = 0
        self.alpha = [CAR_KAPPA * x for x in CAR_W0]
        self.n_car = 0
        # Beta posterior on P(movement is none of the models)
        self.a_none = OPP_NONE_KAPPA * OPP_NONE0
        self.a_fam = OPP_NONE_KAPPA * (1.0 - OPP_NONE0)
        # "none": anywhere he can run to in OPP_H cycles (uniform on a disk)
        vmax = float(getattr(sp, "player_speed_max", 1.05) or 1.05)
        self.none_r = vmax * OPP_H + 1.0
        self.eng = {}
        self.form = LightAgent()          # LightAgent.formation: a least-
        self.form.model = Model(sp)       # squares fit of the shared formation
        self.samples = []
        self.car_pending = []
        self.think_ms = 20.0      # thinking time per engine decision

    def _engine(self, name, unum, pool=""):
        # pool: engines answering hypothetical (future) positions are
        # kept apart, so they never disturb the memory of the ones
        # following the real game
        key = (pool, name, unum)
        if key not in self.eng:
            self.eng[key] = self.ENGINES[name]()
        return self.eng[key]

    # ---- the opponent's world, as he sees it ----------------------------
    def _context(self, ph, v):
        ours, theirs = v.mates, v.opps
        bx, by, bvx, bvy = v.bx, v.by, v.bvx, v.bvy
        reach = {}
        for p in ours + theirs:
            ctrl = ph.catchable if p.goalie else ph.kickable
            reach[id(p)] = _ball_reach(ph, p, bx, by, bvx, bvy, ctrl)
        xs = sorted((-p.x for p in ours), reverse=True)
        second = xs[1] if len(xs) > 1 else (xs[0] if xs else 0.0)
        return dict(
            reach=reach,
            kick_us=any(math.hypot(p.x - bx, p.y - by) <= ph.kickable for p in ours),
            kick_them=any(math.hypot(p.x - bx, p.y - by) <= ph.kickable for p in theirs),
            second=second, offside=max(second, -bx, 0.0),
            own_line=min((-p.x for p in theirs if not p.goalie), default=0.0),
            r_us=min((reach[id(p)] for p in ours), default=99),
            gk_us=next((p.unum for p in ours if p.goalie), 0),
            gk_them=next((p.unum for p in theirs if p.goalie), 0))

    def _mirror(self, ph, v, cx, j, slots, on_ball):
        """The stand-in agent for opponent j (world rotated 180 degrees)."""
        bx, by = v.bx, v.by
        reach = cx["reach"]
        sx, sy = -j.x, -j.y

        def mp(p):
            q = _MP(p.unum, -p.x, -p.y, -p.vx, -p.vy, _norm(p.body + 180.0),
                    p.goalie, p.tackling)
            q.dist_from_self = math.hypot(-p.x - sx, -p.y - sy)
            q.angle_from_self = math.degrees(math.atan2(-p.y - sy, -p.x - sx))
            q.dist_from_ball = math.hypot(p.x - bx, p.y - by)
            q.ball_reach_steps = reach[id(p)]
            return q

        me = mp(j)
        if on_ball:
            me.is_kickable = True
            me.kick_rate = _kick_rate(ph, j, bx, by)
        ball = _MNS(position=Vector2D(-bx, -by), velocity=Vector2D(-v.bvx, -v.bvy),
                    dist_from_self=math.hypot(bx - j.x, by - j.y))
        it = _MNS(self_reach_steps=reach[id(j)],
                  first_teammate_reach_steps=min(
                      (reach[id(p)] for p in v.opps if p is not j), default=99),
                  first_opponent_reach_steps=cx["r_us"])
        wm = _MNS(self=me, ball=ball,
                  teammates=[mp(p) for p in v.opps if p is not j],
                  opponents=[mp(p) for p in v.mates],
                  intercept_table=it, offside_line_x=cx["offside"],
                  our_defense_line_x=cx["own_line"],
                  their_defense_line_x=cx["second"],
                  kickable_teammate_existance=cx["kick_them"] and not on_ball,
                  kickable_opponent_existance=cx["kick_us"],
                  our_goalie_uniform_number=cx["gk_them"],
                  their_goalie_uniform_number=cx["gk_us"],
                  last_kicker_uniform_number=0, cycle=v.cycle,
                  game_mode_type=v.gmode,
                  our_team_score=v.score_them, their_team_score=v.score_us)
        hx, hy = slots.get(j.unum, (sx, sy))
        return wm, Vector2D(hx, hy)

    # ---- off the ball ----------------------------------------------------
    def targets(self, ph, v, pool=""):
        """-> {unum: [None, helios target, col target, light target]} for
        the opponents' field players off the ball (targets in our frame)."""
        cx = self._context(ph, v)
        slots = self.form.formation(-v.bx, -v.by, cx["offside"])
        out = {}
        for j in v.opps:
            if j.goalie or math.hypot(j.x - v.bx, j.y - v.by) <= ph.kickable:
                continue
            wm, home = self._mirror(ph, v, cx, j, slots, False)
            row = [None]
            for name in OPP_PARTS[1:]:
                m = _Mirror(wm, self.sp, home, self.think_ms)
                try:
                    self._engine(name, j.unum, pool).off_ball(m)
                except Exception:
                    pass
                row.append(self._target(ph, m, j, cx["reach"][id(j)], v))
            out[j.unum] = row
        return out

    @staticmethod
    def _target(ph, m, j, r, v):
        act = m.act
        if act is not None and act[0] == "go_to_point":
            t = act[1][0]
            return (-t.x(), -t.y())
        if act is not None and act[0] == "intercept":
            return ph.ball_at(v.bx, v.by, v.bvx, v.bvy, min(r, 30))
        return (j.x, j.y)

    @staticmethod
    def _move(ph, o, T, t):
        """Where o is after t cycles heading for T (one turn if needed)."""
        dx, dy = T[0] - o.x, T[1] - o.y
        d = math.hypot(dx, dy)
        if d < 1e-6 or t <= 0:
            return o.x, o.y
        n = t
        if d > 1.0 and abs(_norm(math.degrees(math.atan2(dy, dx)) - o.body)) > 30.0:
            n = t - 1
        s = min(d, ph.dash_tab[min(n, len(ph.dash_tab)) - 1] if n > 0 else 0.0)
        return o.x + dx / d * s, o.y + dy / d * s

    def parts(self, ph, o, row, t):
        """Each part's prediction of o's position t cycles ahead."""
        out = [ph.inertia(o, t)]
        for T in row[1:]:
            out.append(self._move(ph, o, T, t) if T is not None else out[0])
        return out

    def p_none(self):
        """P(his movement is none of the models)."""
        return self.a_none / (self.a_none + self.a_fam)

    def trust(self):
        return 1.0 - self.p_none()

    def predict(self, ph, opps, tg, t):
        """Mixture prediction, shrunk towards plain HELIOS (he stays where
        he is) by P(none)."""
        tr = self.trust()
        res = []
        for o in opps:
            row = tg.get(o.unum)
            if row is None:
                x, y = ph.inertia(o, t)
            else:
                x = y = 0.0
                for wk, (px, py) in zip(self.w, self.parts(ph, o, row, t)):
                    x += wk * px
                    y += wk * py
            x = tr * x + (1.0 - tr) * o.x
            y = tr * y + (1.0 - tr) * o.y
            res.append(_P(o.unum, x, y, 0.0, 0.0, o.body, o.goalie, o.tackling))
        return res

    def none_update(self, d, ax, ay):
        """EM step for the outlier component: the responsibility of "none"
        for one observed movement (a Gaussian around the mixture's
        prediction vs uniform over his reach) is added to its Beta
        posterior."""
        px = sum(wk * dx for wk, (dx, _) in zip(self.w, d))
        py = sum(wk * dy for wk, (_, dy) in zip(self.w, d))
        r2 = (ax - px) ** 2 + (ay - py) ** 2
        s2 = OPP_SIGMA * OPP_SIGMA
        g = math.exp(-0.5 * r2 / s2) / (2.0 * math.pi * s2)
        u = 1.0 / (math.pi * self.none_r * self.none_r)
        pn = self.p_none()
        z = pn * u + (1.0 - pn) * g
        if z > 0.0:
            gam = pn * u / z
            self.a_none += gam
            self.a_fam += 1.0 - gam

    def add_sample(self, ph, cycle, opps, tg):
        rows = []
        for o in opps:
            row = tg.get(o.unum)
            if row is not None:
                rows.append((o.unum, o.x, o.y, self.parts(ph, o, row, OPP_H)))
        if rows:
            self.samples.append((cycle, rows))

    def learn(self, cycle, opps):
        """Bayesian linear regression: add every matured sample to the
        sufficient statistics (prior + match) and re-fit the MAP weights on
        the simplex by hill climbing every OPP_REFIT samples."""
        if not self.samples:
            return
        now = {o.unum: (o.x, o.y) for o in opps}
        keep = []
        k = len(self.w)
        for c, rows in self.samples:
            age = cycle - c
            if age < OPP_H:
                keep.append((c, rows))
                continue
            if age > OPP_H + 2:
                continue
            for unum, x0, y0, pr in rows:
                if unum not in now:
                    continue
                ax, ay = now[unum][0] - x0, now[unum][1] - y0
                if math.hypot(ax, ay) > 15.0:          # restart / set play
                    continue
                d = [(px - x0, py - y0) for px, py in pr]
                self.none_update(d, ax, ay)
                for i in range(k):
                    for j in range(k):
                        self.Q[i][j] += d[i][0] * d[j][0] + d[i][1] * d[j][1]
                    self.b[i] += d[i][0] * ax + d[i][1] * ay
                self.new += 1
                self.n += 1
        self.samples = keep
        if self.new >= OPP_REFIT:
            self.w = _simplex_climb(self.Q, self.b, self.w)
            self.new = 0

    # ---- the carrier -----------------------------------------------------
    def carrier(self, ph, v):
        """-> (carrier unum, [(kind, tx, ty) per engine]) or None. kind:
        'kick' (pass / shot / clear: ball goes to (tx, ty)), 'keep'
        (dribble towards (tx, ty)) or 'hold'."""
        holders = [j for j in v.opps
                   if math.hypot(j.x - v.bx, j.y - v.by) <= ph.kickable]
        if not holders:
            return None
        j = min(holders, key=lambda p: math.hypot(p.x - v.bx, p.y - v.by))
        cx = self._context(ph, v)
        slots = self.form.formation(-v.bx, -v.by, cx["offside"])
        wm, home = self._mirror(ph, v, cx, j, slots, True)
        preds = []
        for name in CAR_PARTS:
            if name == "none":
                preds.append(None)          # no prediction: any action
                continue
            m = _Mirror(wm, self.sp, home, self.think_ms)
            try:
                self._engine(name, j.unum).on_ball(m)
            except Exception:
                pass
            preds.append(self._car_act(ph, m.act, j, v))
        return j.unum, preds

    @staticmethod
    def _car_act(ph, act, j, v):
        bx, by = v.bx, v.by
        if act is None:
            return ("hold", bx, by)
        name = act[0]
        if name in ("kick", "kick_one_step", "dribble"):
            t = act[1][0]
            tx, ty = -t.x(), -t.y()
            if name == "dribble":
                return ("keep", tx, ty)
            # a short kick he runs onto himself is a dribble (HELIOS style)
            if math.hypot(tx - bx, ty - by) < 7.0:
                mates = [p for p in v.opps if p is not j and not p.goalie]
                dj = math.hypot(j.x - tx, j.y - ty)
                if all(math.hypot(p.x - tx, p.y - ty) > dj for p in mates):
                    return ("keep", tx, ty)
            return ("kick", tx, ty)
        if name == "clear":
            return ("kick", bx - 20.0, math.copysign(30.0, by if by else 1.0))
        if name == "advance_ball":
            return ("kick", bx - 30.0, by)
        return ("hold", bx, by)

    def car_weights(self):
        s = sum(self.alpha) or 1.0
        return [a / s for a in self.alpha]

    def hint(self, ph, v, cycle):
        """Carrier prediction for our defence, and a pending observation
        for learning."""
        r = self.carrier(ph, v)
        if r is None:
            return None
        unum, preds = r
        holder = next(p for p in v.opps if p.unum == unum)
        self.car_pending.append((cycle, unum, holder.x, holder.y, v.bx, v.by, preds))
        w = self.car_weights()
        p_keep, sx, sy = 0.0, 0.0, 0.0
        recv = {}
        for wk, pred in zip(w, preds):
            if pred is None:            # "none": its mass backs no action
                continue
            kind, tx, ty = pred
            if kind in ("keep", "hold"):
                p_keep += wk
                if kind == "keep":
                    a = math.atan2(ty - v.by, tx - v.bx)
                    sx += wk * math.cos(a)
                    sy += wk * math.sin(a)
            else:
                cand = [p for p in v.opps if p.unum != unum and not p.goalie]
                if cand:
                    r_ = min(cand, key=lambda p: math.hypot(p.x - tx, p.y - ty))
                    if math.hypot(r_.x - tx, r_.y - ty) < 6.0:
                        recv[r_.unum] = recv.get(r_.unum, 0.0) + wk
        keep_dir = math.degrees(math.atan2(sy, sx)) if (sx or sy) else None
        return dict(p_keep=p_keep, keep_dir=keep_dir, recv=recv)

    def car_observe(self, ph, cycle, v, kick_us):
        """Bayes' rule on the mixture: once the carrier's action is visible,
        alpha_k += P(engine k | what he did)."""
        if not self.car_pending:
            return
        kick_them = [p.unum for p in v.opps
                     if math.hypot(p.x - v.bx, p.y - v.by) <= ph.kickable]
        keep = []
        for item in self.car_pending:
            c0, unum, hx0, hy0, bx0, by0, preds = item
            obs = _car_outcome(c0, unum, bx0, by0, cycle, v.bx, v.by,
                               kick_us, kick_them)
            if obs is None:
                keep.append(item)
                continue
            if obs == "lost":
                continue
            self.car_update(preds, obs, bx0, by0)
        self.car_pending = keep[-12:]

    def car_update(self, preds, obs, bx0, by0):
        w = self.car_weights()
        lik = [_car_lik(p, obs, bx0, by0) for p in preds]
        z = sum(wk * l for wk, l in zip(w, lik))
        if z > 0.0:
            for k in range(len(self.alpha)):
                self.alpha[k] += w[k] * lik[k] / z
            self.n_car += 1


# ==========================================================================
# Agent1: the HELIOS engine planning and defending with the opponent model
# ==========================================================================

class Agent1(HeliosAgent):
    OPP_MODEL = True          # plan against the opponent model
    OPP_LEARN = True          # update the model during the match
    CARRIER_MODEL = True      # defend against the predicted carrier action
    META = True               # protect a lead / chase a deficit (_meta_policy)
    CHAIN_DEPTH = 6           # deeper tree than HELIOS's 4
    SAMPLE_EVERY = 2          # learning samples while we have the ball
    # more of the budget (agent A: 0.68 / 12 / 420 / 0.8 / 80 / 20000)
    SPEND_FRACTION = 0.85     # share of the match budget to spend
    ON_BALL_SHARE = 24.0      # an on-ball decision gets ~24 cycles' worth
    MAX_ON_MS = 450.0         # (was 650: decisions overran 1 s on busy machines)
    OFF_BALL_SHARE = 2.5
    MAX_OFF_MS = 120.0        # (was 160)
    ON_WALL_MARGIN = 450.0    # wall ms kept free after the on-ball search (A: 250)
    EVAL_CAP = 100000         # chain evaluations per decision
    ROOT_REPLAN_MS = 12.0     # time per root whose opponents re-decide
    ROOT_REPLAN_MAX = 16      # ... at most this many roots
    SAMPLE_MIN_FRAC = 0.35    # ... only while this share of the budget is left

    def __init__(self):
        super().__init__()
        self._om = None
        self._tg = (-1, None)
        self._hint = (-1, None)
        self._last_sample = -99
        self._b_start = None

    def _model(self, agent):
        if self._om is None:
            self._om = _OppModel(agent.server_params)
        return self._om

    def _view(self, agent, w):
        wm = agent.wm
        try:
            su, st = int(wm.our_team_score), int(wm.their_team_score)
        except Exception:
            su = st = 0
        return _MNS(mates=[m for m in w.mates] + [w.me], opps=list(w.opps),
                    bx=w.bx, by=w.by, bvx=w.bvx, bvy=w.bvy, cycle=w.cycle,
                    score_us=su, score_them=st,
                    gmode=getattr(wm, "game_mode_type", 0))

    def _targets(self, agent, ph, w):
        if self._tg[0] != w.cycle:
            self._tg = (w.cycle, self._model(agent).targets(ph, self._view(agent, w)))
        return self._tg[1]

    def _budget_ok(self, agent):
        rem = self._remaining(agent)
        if self._b_start is None or rem > self._b_start:
            self._b_start = rem
        return (rem >= self.SAMPLE_MIN_FRAC * self._b_start
                and self._decision_left(agent) >= 500.0)

    # ---- planning (on the ball) -----------------------------------------
    def _prep_planner(self, agent, ph, w, hp):
        if not self.OPP_MODEL:
            return
        try:
            tg = self._targets(agent, ph, w)
        except Exception as e:
            _note(agent, "ERR_oppmodel", err=type(e).__name__ + ":" + str(e)[:60])
            return
        om = self._om
        cache = {}

        def opp_at(t):
            t = max(0, t)
            if t not in cache:
                cache[t] = om.predict(ph, w.opps, tg, t)
            return cache[t]

        hp.opp_at = opp_at

        def root_ctx(c):
            # the opponents re-decide once our first action has played
            # out: ball at its target, receiver (or me) there, the others
            # carried on; their engines run again on that position
            t0 = max(1, int(c.steps))
            opp0 = om.predict(ph, w.opps, tg, t0)
            mover = c.unum if c.kind == "pass" else w.me.unum
            mates = []
            for m in list(w.mates) + [w.me]:
                if m.unum == mover:
                    mates.append(_P(m.unum, c.tx, c.ty, 0.0, 0.0, m.body, m.goalie, False))
                else:
                    x, y = ph.inertia(m, t0)
                    mates.append(_P(m.unum, x, y, 0.0, 0.0, m.body, m.goalie, False))
            v2 = _MNS(mates=mates, opps=opp0, bx=c.tx, by=c.ty, bvx=0.0, bvy=0.0,
                      cycle=w.cycle + t0, score_us=0, score_them=0, gmode=0)
            tg2 = om.targets(ph, v2, "future")
            cache2 = {}

            def at(t):
                if t < t0:
                    return opp_at(t)
                if t not in cache2:
                    cache2[t] = om.predict(ph, opp0, tg2, t - t0)
                return cache2[t]
            return at

        allow = self._allowance_ms(agent, True)
        hp.root_ctx = root_ctx
        # re-deciding opponents is only worth it while the model is trusted
        hp.ROOT_CTX_K = int(max(0.0, min(self.ROOT_REPLAN_MAX,
                                         0.5 * allow / self.ROOT_REPLAN_MS)
                                * om.trust()))
        if self.OPP_LEARN:
            om.add_sample(ph, w.cycle, w.opps, tg)
            self._last_sample = w.cycle

    # ---- defending (off the ball) ---------------------------------------
    def _carrier_hint(self, agent):
        if not self.CARRIER_MODEL:
            return None
        try:
            wm = agent.wm
            cyc = int(wm.cycle)
            key = cyc // CAR_EVERY
            if self._hint[0] == key:
                return self._hint[1]
            hint = None
            if bool(wm.kickable_opponent_existance) and self._budget_ok(agent):
                ph = self._phys(agent)
                w = _World(agent)
                om = self._model(agent)
                # the carrier's three engines share this decision's time
                om.think_ms = max(8.0, min(40.0, self._allowance_ms(agent, False) / 3.0))
                hint = om.hint(ph, self._view(agent, w), cyc)
                om.think_ms = 20.0
            self._hint = (key, hint)
            return hint
        except Exception as e:
            _note(agent, "ERR_carrier", err=type(e).__name__ + ":" + str(e)[:60])
            return None

    # ---- learning --------------------------------------------------------
    def on_ball(self, agent) -> None:
        self._observe(agent, False)
        HeliosAgent.on_ball(self, agent)

    def off_ball(self, agent) -> None:
        HeliosAgent.off_ball(self, agent)
        self._observe(agent, True)

    def _observe(self, agent, may_sample):
        if not self.OPP_LEARN or not (self.OPP_MODEL or self.CARRIER_MODEL):
            return
        try:
            ph = self._phys(agent)
            w = _World(agent)
            om = self._model(agent)
            wm = agent.wm
            om.learn(w.cycle, w.opps)
            if self.CARRIER_MODEL:
                om.car_observe(ph, w.cycle, self._view(agent, w),
                               bool(wm.kickable_teammate_existance)
                               or math.hypot(w.me.x - w.bx, w.me.y - w.by) <= ph.kickable)
            if w.cycle % 150 == 0:
                _note(agent, "oppmodel", w=[round(x, 2) for x in om.w], n=om.n,
                      none=round(om.p_none(), 3),
                      car=[round(x, 2) for x in om.car_weights()], n_car=om.n_car)
            if (not self.OPP_MODEL or not may_sample
                    or w.cycle - self._last_sample < self.SAMPLE_EVERY):
                return
            it = wm.intercept_table
            ours = bool(wm.kickable_teammate_existance) or (
                not bool(wm.kickable_opponent_existance)
                and min(int(it.self_reach_steps), int(it.first_teammate_reach_steps))
                < int(it.first_opponent_reach_steps))
            if not ours or not self._budget_ok(agent):
                return
            om.add_sample(ph, w.cycle, w.opps, self._targets(agent, ph, w))
            self._last_sample = w.cycle
        except Exception as e:
            _note(agent, "ERR_oppmodel", err=type(e).__name__ + ":" + str(e)[:60])
