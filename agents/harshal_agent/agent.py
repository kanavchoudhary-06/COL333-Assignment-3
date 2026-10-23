# Your agent goes here. The Agent1 class must hold your final submission.
#
# COL333 A3 - Agent1 (v1)
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

import math
import time
import itertools
import random

from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


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
DRIB_M0, DRIB_K = 0.2, 1.5
DRIB_M0, DRIB_K = 0.0, 1.5
HOLD_M0, HOLD_K = 2.0, 1.0
SHOT_M0, SHOT_K = 0.0, 1.3   # goalie arriving with the ball = 50/50

# --- evaluation H(state) and Loss
W_PROG = 15.0             # progress up the pitch (x): 5 m forward ~ +0.48
W_CENTER = 1.6            # ball in the middle strip, away from the touchlines
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
SHOT_RANGE = 35.0
ONE_STEP_SPEED = 2.4      # realistic speed of a single-touch kick
SHOT_POST_MARGIN = 0.8    # aim this far inside the posts (kick noise)

# --- search
GAMMA = 0.9               # discount on the depth-2 part (less reliable)
HOLD_COST = 0.6           # holding gains nothing and lets defenders close in
HOLD_COST = 0.3           # holding gains nothing and lets defenders close in
HOLD_FRAC = 0.3           # ... and costs this SHARE of the position value too
                          # (v9 used 40 %: it held only ~6 % of the time and scored)
                          # (fixed cost, so it does not grow with W_PROG)
GREEDY_P = 0.85           # greedy shortcut: forward pass this safe...
GREEDY_P = 0.70           # greedy shortcut: forward pass this safe...
GREEDY_GAP = 0.4          # ...and this much better than the runner-up
CLEAR_VALUE = -0.9        # below this, clear (if under pressure / own third)
THROUGH_R = 7.0
ALLOW_THROUGH = True      # set False if through passes keep being offside
PASS_MIN, PASS_MAX = 3.0, 35.0
DRIB_SPEED = 0.7
DRIB_SPEED = 0.8          # m/cycle while dribbling (log: ~1.0); 0.7 was far too slow
DRIB_TACKLE_R = 1.2       # opponent must get this close to the ball on a dribble
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
MC_DEPTH = 3              # actions per chain (root + 2 follow-ups)
MC_TIME_MS = 80.0         # Monte Carlo time per on-ball decision (also capped by budget)
MC_MAX = 400              # max rollouts per decision
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
TACKLE_MIN = 0.45         # presser tackles at this win probability (a tackle
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
MAX_SUPPORT_DIST = 20.0   # NO attacker further than this from the holder
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
OPEN_W = 0.5              # get-open: weight of free space around the point
GOALWARD_W = 0.6          # get-open: bonus for being ahead of the holder (per 20 m)
ANGLE_SPREAD = 30.0       # get-open: teammates within this angle (seen from the
OPEN_W = 1.0              # get-open: weight of free space around the point
GOALWARD_W = 1.5          # get-open: bonus for being ahead of the holder (per 20 m)
OPEN_REACH_P = 0.6        # get-open: a spot only counts if a pass reaches it with this P
OPEN_UNREACH_PEN = 3.0    # ... (unreachable spots get this penalty = only a fallback)
ANGLE_SPREAD = 40.0       # get-open: teammates within this angle (seen from the
                          # holder) are "in the same direction" -> penalty
HOLD_MEMORY = 5           # cycles a dribbling teammate keeps "his" ball
ROLE_W = 0.6              # get-open: cost per 10 m away from the role spot

# --- time management
ONBALL_MULT, ONBALL_CAP = 8.0, 300.0
OFFBALL_MULT, OFFBALL_CAP = 1.0, 40.0
SAFETY_MS = 150.0
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


def _xy(v):
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
    x, y = _xy(p.position)
    vx, vy = _xy(getattr(p, "velocity", None))
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

        self.bx, self.by = _xy(wm.ball.position)
        self.bvx, self.bvy = _xy(getattr(wm.ball, "velocity", None))
        self.offside = float(getattr(wm, "offside_line_x", self.hl) or 0.0)
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


def Loss(snap, x):
    """Value of losing the ball at x: worse the closer to OUR goal."""
    prog = (x + snap.hl) / (2.0 * snap.hl)
    return -(L_BASE + L_K * (1.0 - prog))


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
# Agent1
# =============================================================================

class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        # Per-player memory across cycles. The framework's memory object is
        # NOT a dict (no [] or .get), but `self` persists for each player
        # (one process per player), so state is kept here instead.
        self.mem = {}

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
            note(agent, _err_label("on", e))
            StarterAgent.on_ball(self, agent)

    def _on_ball(self, agent):
        t0 = time.perf_counter()
        snap = Snap(agent)
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
            note(agent, "quick_on_ball")
            StarterAgent.on_ball(self, agent)
            return
        allow = max(self._allowance(agent, snap, True), QUICK_MIN_MS)
        deadline = t0 + allow / 1000.0
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
                note(agent, "shoot_now", p=round(best_shot["p"], 2))
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
                note(agent, "breakaway_dribble", p=round(b["p"], 2))
                self._execute(agent, snap, b, pressure)
                return

        # ---- 2. small candidate set with deterministic depth-1 values -----
        cands = self._candidates(snap, bx, by, me, snap.field_mates, snap.offside, opps, pressure)
        cands = self._candidates(snap, bx, by, me, snap.field_mates, snap.offside, opps, pressure,
                                 passer_unum)
        if best_shot is not None:
            cands.append(best_shot)
        holds = self.mem.get("holds", 0) if self.mem.get("hold_cycle") == snap.cycle - 1 else 0
        if holds < MAX_HOLDS or not cands:
            hc = self._eval_hold(snap, bx, by, pressure)
            hc["v"] -= HOLD_STALL * holds          # stalling gets worse every cycle
            hc["stall"] = HOLD_STALL * holds
            cands.append(hc)
        cands.sort(key=lambda c: c["v"], reverse=True)

        # ---- 3. greedy shortcut (user's idea) ------------------------------
        b0 = cands[0]
        gap = b0["v"] - (cands[1]["v"] if len(cands) > 1 else -1e9)
        if (b0["kind"] == "pass" and b0["tx"] > bx + 1.0
                and b0["p"] >= GREEDY_P and gap >= GREEDY_GAP):
            note(agent, "pass_greedy", to=b0["recv"].unum, p=round(b0["p"], 2))
            self._execute(agent, snap, b0, pressure)
            return

        # ---- 4. depth-3 chains (deterministic) as priors -------------------
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
        if means[di] >= means[bi] - MC_TIE:
            bi = di
        best = roots[bi]["c"]
        bv = means[bi]

        # ---- 6. fallback / execute ----------------------------------------
        if bv < CLEAR_VALUE and (pressure <= 2 or bx < -snap.hl * 0.4):
            note(agent, "clear", v=round(bv, 2))
            self.mem["plan"] = None
            Action.clear(agent)
            return
        note(agent, best["kind"] + "_" + best.get("sub", best["kind"]) + "_mc",
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
            c = dict(kind="pass", sub="commit", tx=plan["tx"], ty=plan["ty"], n=plan["n"],
                     v0=plan["v0"], delay=0, recv=recv)
            r = self._eval_pass(snap, bx, by, c, snap.opps, -1e9)
            if r is None or r["p"] < COMMIT_P:
                return False
            note(agent, "commit_pass", p=round(r["p"], 2))
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
            note(agent, "commit_dribble", p=round(r["p"], 2))
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
                Action.kick_one_step(agent, tgt)
            else:
                Action.kick(agent, tgt, 1, accept_late=2)   # arrive_in=1 = max pace
            Action.kick_one_step(agent, tgt)
        elif k == "pass":
            self.mem["pm_until"] = snap.cycle + PASS_MOVE_CYCLES
            if not keep_plan:
                self.mem["plan"] = dict(kind="pass", tx=c["tx"], ty=c["ty"], n=c["n"], v0=c["v0"],
                                        recv=c["recv"].unum, cycle=snap.cycle,
                                        commit=COMMIT_PASS)
            if pressure <= 1:
                Action.kick_one_step(agent, tgt)      # no time for a wind-up
            else:
                Action.kick(agent, tgt, c["n"], accept_late=5)
        elif k == "dribble":
            if not keep_plan:
                self.mem["plan"] = dict(kind="dribble", tx=c["tx"], ty=c["ty"],
                                        cycle=snap.cycle, commit=COMMIT_DRIB)
            Action.dribble(agent, tgt)
            Action.hold(agent)
            Action.clear(agent)
        elif k == "hold":
            self.mem["holds"] = (self.mem.get("holds", 0)
                                 if self.mem.get("hold_cycle") == snap.cycle - 1 else 0) + 1
            self.mem["hold_cycle"] = snap.cycle
            Action.hold(agent)
            Action.clear(agent)
        else:
            Action.clear(agent)
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
                res = kick_eval(snap, sx, sy, snap.hl, gy, v0, delay, None, defenders,
                                tmax=40, mode="shot")
                if res is None:
                    continue
                margin = res[0]
                p = _sig(margin, SHOT_M0, SHOT_K) * acc
                v = p * GOAL_VALUE + (1 - p) * SHOT_MISS
                out.append(dict(kind="shot", mode=mode, tx=snap.hl, ty=gy, p=p, v=v))
        return out

    # ------------------------------------------------------------ passes
    def _eval_pass(self, snap, bx, by, c, defenders, alpha):
        vs_ub = H(snap, c["tx"], c["ty"], CLEAR_CAP)
        state = {"cut": False}

        def cut(m):
            p = _sig(m, PASS_M0, PASS_K)
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
        p = _sig(margin, PASS_M0, PASS_K)
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
                m = reach(d, x, y, TACKLE_R) - t
                m = reach(d, x, y, DRIB_TACKLE_R) - t
                if m < margin:
                    margin = m
                if t == T and m < end_m:
                    end_m = m
        p = _sig(margin, DRIB_M0, DRIB_K)
        vs = DRIB_GAMMA * H(snap, tx, ty, end_m)
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
        v = p * (h - HOLD_COST) + (1 - p) * Loss(snap, bx)
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
        dirs = [("g%+d" % int(off) if off != 180.0 else "back", base + off) for off in DRIB_ANGLES]
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

    def _candidates(self, snap, hx, hy, holder, mates, offside, opps, pressure):
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
        raw += self._dribble_dirs(snap, hx, hy, opps, pressure)
        out = []
        for c in raw:
            if c["kind"] == "pass":
                r = self._eval_pass(snap, hx, hy, c, opps, -1e9)
                if r is None:
                    continue
                if pressure > 2 and r["tx"] < hx - 1.0:   # no ping-pong
                    r["pen"] = BACK_PASS_PEN
                    r["v"] -= BACK_PASS_PEN * r["p"]
                pen = self._pass_pen(r, hx, pressure, passer_unum)
                if pen > 0.0:
                    r["pen"] = pen
                    r["v"] -= pen * r["p"]
            else:
                r = self._eval_dribble(snap, hx, hy, c, opps, holder_body=holder)
                if pressure > 2 and r["tx"] < hx - 1.0:   # no retreating either
                    r["pen"] = BACK_PASS_PEN
                    r["v"] -= BACK_PASS_PEN * r["p"]
            out.append(r)
        return out

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
            if r is not None and (best is None or r["v"] > best["v"]):
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
        here = H(snap, hx, hy, 4.0) - W_SHOTQ * shot_quality(snap, hx, hy) - HOLD_COST
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
        if k == "dribble":
            r = math.radians(a["ang"])
            c = dict(a, tx=sx + a["D"] * math.cos(r), ty=sy + a["D"] * math.sin(r))
            if not snap.in_field(c["tx"], c["ty"], 0.5):
                return False, sx, sy, 1, 0
            res = self._eval_dribble(snap, sx, sy, c, nopps, holder_body=holder)
            ok = res["m"] + rng.gauss(0.0, MC_SIGMA) > MC_DRIB_OFF
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
            return _sig(res[0], SHOT_M0, SHOT_K), res[5], res[6], res[1], 0
        res = kick_eval(snap, sx, sy, ntx, nty, v0, a["delay"], a["recv"], nopps,
                        tmax=min(40, a["n"] + 12), mode="pass")
        if res is None:
            return False, ntx, nty, a["n"], 0
        margin, tend, ix, iy, endm, ex, ey = res
        if margin + rng.gauss(0.0, MC_SIGMA) > 0.0:
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
            if k == "hold":            # ok = keep probability here
                h = H(snap, sx, sy, pressure) - W_SHOTQ * shot_quality(snap, sx, sy)
                return (total - a.get("stall", 0.0)
                        + disc * (ok * (h - HOLD_COST) + (1 - ok) * Loss(snap, sx)))
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
        return max(cands, key=lambda c: c["v"])

    # =================================================================
    # OFF BALL
    # =================================================================
    def off_ball(self, agent) -> None:
        try:
            self._off_ball(agent)
        except Exception as e:
            note(agent, _err_label("off", e))
            StarterAgent.off_ball(self, agent)

    def _off_ball(self, agent):
        t0 = time.perf_counter()
        snap = Snap(agent)
        wm = agent.wm
        mem = self.mem
        rem, dec = self._budget(agent)
        if rem < QUICK_REMAIN_MS or dec < QUICK_DEC_MS or snap.me.goalie:
            note(agent, "quick_off_ball")
            StarterAgent.off_ball(self, agent)
            return
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
            note(agent, "chase_own")
            Action.intercept(agent)
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
                        note(agent, "counter_run")
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
        Action.go_to_point(agent, Vector2D(tx, ty),
                           max_dash_power=self._dash(agent, snap, urgent, d),
                           distance_threshold=thr)
        Action.turn_to_ball(agent)

    def _chase(self, agent, snap, my_t, opp_t):
        """Economical interception: run only as fast as needed when we are
        clearly first; sprint (built-in intercept) when the race is close."""
        me = snap.me
        moving = math.hypot(snap.bvx, snap.bvy) > 0.5
        if (my_t <= 2 or opp_t - my_t < ECO_MIN_MARGIN or moving
                or self.mem.get("last_poss") == "us"):
            note(agent, "chase")
            Action.intercept(agent)
            Action.go_to_point(agent, Vector2D(snap.bx, snap.by))
            return
        t_meet = min(opp_t - 3, my_t + 4)
        cx, cy = snap.ball_at(t_meet)
        if not snap.in_field(cx, cy, 0.0):
            note(agent, "chase")
            Action.intercept(agent)
            return
        d = max(0.0, math.hypot(cx - me.x, cy - me.y) - 0.5 * CTRL_R)
        need = d / max(1, t_meet - 1)
        power = max(45.0, min(100.0, need / SPEED_PER_POWER * 1.25))
        note(agent, "chase_eco", power=int(power))
        Action.go_to_point(agent, Vector2D(cx, cy), max_dash_power=power,
                           distance_threshold=0.6)
        Action.intercept(agent)

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
                note(agent, "def_face")
                Action.turn_to_ball(agent)
                return

        markers = [b for b in pool if b.unum != presser_unum and b.unum != back_unum]
        if not markers:
            self._go(agent, snap, *_xy(Tools.home_position(agent)))
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
        for perm in itertools.permutations(markers, len(tids)):
            cost = sum(math.hypot(p.x + 3 * p.vx - tpos[k][0], p.y + 3 * p.vy - tpos[k][1])
                       for p, k in zip(perm, tids))
            if cost < best_cost - 1e-9:
                best_cost, best_asg = cost, tuple((p.unum, k) for p, k in zip(perm, tids))

        for u, k in best_asg:
            if u == me.unum:
                tx, ty = tpos[k]
                name = tname[k]
                # zone jobs (path / screen / cover) for a player still BEHIND
                # the ball: don't chase from behind, cut the path instead
                if name != "mark" and not self._goal_side(snap, me, cx, cy):
                    tx, ty = self._recovery_point(snap, me, cx, cy)
                    name = "recover"
                d = math.hypot(tx - me.x, ty - me.y)
                note(agent, "def_" + name, team=len(snap.team), marks=len(marks))
                self._go(agent, snap, tx, ty, urgent=(name == "recover" or d > 6.0),
                         thr=0.4 if name == "mark" else 1.0)
                return

        # zone: formation slot pulled towards the ball (keeps shape, compact)
        hx, hy = _xy(Tools.home_position(agent))
        zx = hx + 0.3 * (cx - hx)
        zy = hy + 0.3 * (cy - hy)
        zx = max(min(zx, cx + 5.0), -hl + 12.0)
        note(agent, "def_zone", team=len(snap.team))
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
        lx, ly = o.x + o.vx, o.y + o.vy                    # next cycle
        bxu, byu = _unit(cx - lx, cy - ly)
        gx, gy = _unit(-snap.hl - lx, -ly)
        # ON the line man -> ball, a few metres in front of him (in the
        # passing lane, with room to react to his next move instead of
        # trailing behind him), shaded a little towards our goal
        dx, dy = _unit(0.8 * bxu + 0.2 * gx, 0.8 * byu + 0.2 * gy)
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
                note(agent, "tk_no_p%d_f%d" % (min(9, int(tp * 10)), min(9, int(fp * 10))))
            return False
        if snap.bx < -hl / 3.0:   # own third: knock it wide and forward
            tx, ty = snap.bx + 20.0, (snap.hw if snap.by >= 0 else -snap.hw)
        else:
            tx, ty = hl, 0.0
        ang = math.degrees(math.atan2(ty - snap.by, tx - snap.bx))
        note(agent, "tackle", p=round(tp, 2), f=round(fp, 2))
        Action.tackle(agent, _norm180(ang - me.body), foul=False)
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
        note(agent, "press_back")
        Action.go_to_point(agent, Vector2D(snap.bx + snap.bvx, snap.by + snap.bvy),
                           max_dash_power=self._dash(agent, snap, True, bd),
                           distance_threshold=0.3)
        Action.turn_to_ball(agent)

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
            note(agent, "press_intercept")
            Action.intercept(agent)
            return
        # where the dribble is going (ball + its velocity over 2 cycles)
        lx, ly = snap.bx + 2.0 * snap.bvx, snap.by + 2.0 * snap.bvy
        ux, uy = _unit(-hl - lx, -ly)
        if not self._goal_side(snap, me, lx, ly, slack=-2.0):
            # clearly behind him: race to a point on his path we can reach first
            rx, ry = self._recovery_point(snap, me, lx, ly)
            note(agent, "press_recover")
            self._go(agent, snap, rx, ry, urgent=True, thr=0.5)
            return
        px, py = lx + ux * BLOCK_D, ly + uy * BLOCK_D
        bd = math.hypot(snap.bx - me.x, snap.by - me.y)
        if math.hypot(px - me.x, py - me.y) < 1.0 or bd < BLOCK_D + 0.5:
            # in position: face the ball; if it is close, step into it
            ang = math.degrees(math.atan2(snap.by - me.y, snap.bx - me.x))
            if abs(_norm180(ang - me.body)) > 25.0:
                note(agent, "press_face")
                Action.turn_to_ball(agent)
                return
            note(agent, "press_block")
            Action.go_to_point(agent, Vector2D(snap.bx + snap.bvx, snap.by + snap.bvy),
                               max_dash_power=self._dash(agent, snap, True, bd),
                               distance_threshold=0.3)
            Action.turn_to_ball(agent)
            return
        note(agent, "press")
        Action.go_to_point(agent, Vector2D(px, py),
                           max_dash_power=self._dash(agent, snap, True, bd),
                           distance_threshold=0.5)
        Action.turn_to_ball(agent)

    def _goal_side(self, snap, b, cx, cy, slack=0.5):
        """Is b between (cx, cy) and our goal (closer to our goal)?
        slack > 0 is strict; slack < 0 also accepts players slightly behind."""
        return math.hypot(b.x + snap.hl, b.y) < math.hypot(cx + snap.hl, cy) - slack

    def _recovery_point(self, snap, b, cx, cy):
        """First point on the carrier's straight line to our goal that b can
        reach BEFORE the carrier (who dribbles at ~CARRIER_SPEED). If none,
        a point in front of our goal on that line."""
        gx = -snap.hl
        L = math.hypot(gx - cx, cy)
        ux, uy = _unit(gx - cx, -cy)
        # how fast is he actually going? (a shielding / standing carrier is
        # not running away; dribbling pushes the ball at about his speed)
        v = min(CARRIER_SPEED, max(0.35, 1.2 * math.hypot(snap.bvx, snap.bvy)))
        s = 2.0
        while s < L - 4.0:
            px, py = cx + ux * s, cy + uy * s
            if reach(b, px, py, 0.5) <= s / v - 1.0:
                return px, py
            s += 2.0
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
        return _sig(res[0], PASS_M0, PASS_K)

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
                note(agent, "att_receive", p=round(pred["p"], 2))
                self._go(agent, snap, pred["tx"], pred["ty"], urgent=True, thr=0.5)
                return
            exclude.add(pred["recv"].unum)
        players = [b for b in snap.team if b.unum not in exclude]
        if me.unum in exclude or not players:
            note(agent, "att_home")
            self._go(agent, snap, *_xy(Tools.home_position(agent)))
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
        near = self._wing_spot(snap, hx, hy, side)
        ahead = BRK_AHEAD if brk else WING_AHEAD
        near = self._wing_spot(snap, hx, hy, side, ahead)
        # does the holder have anybody to pass to right now?
        best_opt = max((self._pass_score(snap, hx, hy, bb.x, bb.y) for bb in snap.team
                        if math.hypot(bb.x - hx, bb.y - hy) > 2.0), default=0.0)
        push = best_opt < NO_OPTION_P
        # no far wing: anchor + a close SUPPORT next to the holder + the near
        # wing. If the holder has no option at all, the support "shows" first
        # (gets the job before the wing when fewer players are available).
        support = ("show" if push else "support", self._show_spot(snap, hx, hy))
        if push:
            tasks = [("anchor", anchor), support, ("wing_near", near)][:len(players)]
            # nobody available: one player comes SHORT first
            tasks = [("anchor", anchor), ("show", self._show_spot(snap, hx, hy)),
                     ("wing_near", near)][:len(players)]
        else:
            tasks = [("anchor", anchor), ("wing_near", near), support][:len(players)]
            # options exist: create play AHEAD - near wing + a forward in the
            # central channel / other side of the holder, both as far forward
            # as a pass still reaches them
            tasks = [("anchor", anchor), ("wing_near", near),
                     ("forward", self._forward_spot(snap, hx, hy, side, ahead))][:len(players)]

        def d(bb, t):
            return math.hypot(bb.x + 3 * bb.vx - t[0], bb.y + 3 * bb.vy - t[1])
        best_cost, best = 1e9, None
        for perm in itertools.permutations(players, len(tasks)):
            cost = sum(d(p, t[1]) for p, t in zip(perm, tasks))
            if cost < best_cost - 1e-9:
                best_cost, best = cost, {p.unum: t for p, t in zip(perm, tasks)}
        role, (tx, ty) = best.get(me.unum, ("open", _xy(Tools.home_position(agent))))

        if role in ("show", "support"):
            tx, ty = self._clamp_near(hx, hy, tx, ty)
            note(agent, "att_" + role)
            self._go(agent, snap, tx, ty, urgent=(role == "show"), thr=0.7)
        if role == "show":
            tx, ty = self._clamp_near(snap, hx, hy, tx, ty)
            note(agent, "att_" + role + brk_tag)
            self._go(agent, snap, tx, ty, urgent=(role == "show" or brk), thr=0.7)
            return

        if role == "anchor":
            # small refinement: keep a clear lane for the back pass
            tx, ty = self._get_open(snap, hx, hy, tx, ty, [(hx, hy)], False, radius=(3.0,),
                                    allow_ahead=False)
            note(agent, "att_anchor")
            self._go(agent, snap, tx, ty, urgent=me.x > hx)   # behind the ball fast
            note(agent, "att_anchor" + brk_tag)
            self._go(agent, snap, tx, ty, urgent=me.x > hx or brk)
            return

        # wings: pass-and-move first, then get open around the wing spot
        if mem.get("pm_until", -1) > snap.cycle and role.startswith("wing"):
        # wings / forward: pass-and-move first, then get open around the spot
        if mem.get("pm_until", -1) > snap.cycle and role.startswith(("wing", "forward")):
            fx = me.x + 8.0
            if not beyond_ok:
                fx = min(fx, snap.offside - 0.7)
            fx = min(fx, snap.hl - 6.0)
            # only if there is real room ahead (vs a high line the run would
            # stop at the offside line right next to the ball = crowding)
            if fx - me.x >= 4.0:
            # and a pass can still reach him there
            if (fx - me.x >= 4.0
                    and self._pass_score(snap, hx, hy, fx, me.y) >= OPEN_REACH_P):
                tx, ty = fx, me.y
                role = "pass_move"
        avoid = [(hx, hy)] + [(bb.x, bb.y) for bb in snap.field_mates]
        tx, ty = self._get_open(snap, hx, hy, tx, ty, avoid, beyond_ok)
        if role.startswith("wing"):
        if role.startswith(("wing", "forward")):
            tx, ty, checked = self._checking(snap, hx, hy, tx, ty, avoid, beyond_ok)
            if checked:
                role += "_" + checked
        if not beyond_ok:
            tx = min(tx, snap.offside - 0.7)
        tx, ty = self._clamp_near(hx, hy, tx, ty)
        tx, ty = self._clamp_near(snap, hx, hy, tx, ty)
        dist = math.hypot(tx - me.x, ty - me.y)
        urgent = (role == "pass_move" or "burst" in role or (push and dist > 6.0)
                  or beyond_ok)
        note(agent, "att_" + role + ("_push" if push else ""))
                  or beyond_ok or brk)
        note(agent, "att_" + role + ("_push" if push else "") + brk_tag)
        self._go(agent, snap, tx, ty, urgent=urgent)

    def _clamp_near(self, hx, hy, tx, ty):
        """Nobody further than MAX_SUPPORT_DIST from the holder."""
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
            return hx + (tx - hx) * k, hy + (ty - hy) * k
        return tx, ty
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
        for fy in (hy - side * 10.0, 0.0, hy - side * 4.0):
            fy = max(-snap.hw + 3.0, min(snap.hw - 3.0, fy))
            x = top
            while x >= hx + 2.0:
                p = self._pass_score(snap, hx, hy, x, fy)
                if p >= WING_PASS_MIN and math.hypot(x - hx, fy - hy) >= 8.0:
                    return x, fy
                if p > best:
                    best, bp = p, (x, fy)
                x -= 4.0
                if time.perf_counter() > self._deadline:
                    break
        return bp if bp is not None else (min(top, hx + 8.0), hy - side * 10.0)

    def _wing_spot(self, snap, hx, hy, side):
    def _wing_spot(self, snap, hx, hy, side, ahead=WING_AHEAD):
        """Wide forward: as far forward as a pass from the holder can still
        reach him (P >= WING_PASS_MIN), never past the offside line. Tries
        spots from far ahead back towards the ball on his flank; if nothing on
        the flank is reachable, comes inside (|y| ~ 10)."""
        top = min(hx + WING_AHEAD, snap.offside - 1.0, snap.hl - 6.0,
                  hx + MAX_SUPPORT_DIST - 2.0)
        top = min(hx + ahead, snap.offside - 1.0, snap.hl - 6.0)
        best, bp = -1.0, None
        for wy in (side * WING_Y, side * (WING_Y - 8.0)):
            if abs(wy) > snap.hw - 3.0:
                wy = side * (snap.hw - 3.0)
            x = top
            while x >= hx - 6.0:
                p = self._pass_score(snap, hx, hy, x, wy)
                if p >= WING_PASS_MIN and math.hypot(x - hx, wy - hy) >= 8.0:
                    return x, wy                    # most forward reachable spot
                if p > best:
                    best, bp = p, (x, wy)
                x -= 4.0
                if time.perf_counter() > self._deadline:
                    break
        return bp if bp is not None else (min(top, hx), side * WING_Y)

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
        ax = max(ax, -snap.hl + 12.0)
        return ax, 0.35 * hy

    def _get_open(self, snap, hx, hy, rx, ry, avoid, beyond_ok, radius=(3.0, 6.0),
                  allow_ahead=True):
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
             - ROLE_W * math.hypot(px - rx, py - ry) / 10.0)
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
        home = Tools.home_position(agent)
        h0x, h0y = _xy(home)
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


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None