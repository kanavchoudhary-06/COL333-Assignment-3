# Your agent goes here. The Agent1 class must hold your final submission.
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
#   (v22) holder dribbles back towards our goal  RETREAT_PEN up
#   (v22) self-passes (kick ahead + run) lost .. SELFPASS_MIN_PRESS up, or SELFPASS = False
#   (v22) lead passes intercepted ............ LEAD_PASS_R smaller (e.g. (4.0, 7.0))
#   (v22) forwards too far from the holder ... FWD_MIN_AHEAD / WING_MIN_AHEAD down
#   (v22) anchor too deep when we attack ..... ANCHOR_MAX_BEHIND down
#   (v22) markers too deep / too goal-side ... MARK_LEAD down
#   (v22) passes still tackled in the wind-up  ONE_STEP_PRESS up (4)
#   (v24) still too risky with the ball ...... KEEP_BALL up (1.5), RISKY_P up (0.7)
#   (v24) now too timid / never goes forward . KEEP_BALL down (0.5), SAFE_P up (0.9)
#   (v25) online learning misbehaves ......... LEARN = False (or LEARN_LR down)
#   (v26) dense options too slow / too wild .. DENSE_ACTIONS = False, or fewer
#                                              GRID_DIRS / GRID_SPEEDS
#   (v26) look-ahead options all too similar . DIV_W up (3.5); too odd: DIV_W down
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
LEARN_LR = 0.25           # step size (cycles of margin per unit of surprise)
LEARN_ADJ_MIN, LEARN_ADJ_MAX = -0.5, 3.0   # learned margin offsets stay in here
                          # (asymmetric on purpose: it may get a lot more careful,
                          # only a little bolder - v24 asked for caution)
LEARN_PASS_SPEED = 1.3    # ball leaving our feet faster than this = a pass
LEARN_SPEED_Q = 0.95      # opponent speed = this quantile of observed running
LEARN_SPEED_ETA = 0.01    # ... step size of that quantile estimate
LEARN_SPEED_MIN, LEARN_SPEED_MAX = 0.95, 1.15
LEARN_CARRIER_A = 0.03    # moving-average rate for their dribbling speed
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
DIV_W = 2.5               # diversity: value lost per unit of similarity to an
                          # option already chosen (MMR selection)
DIV_SIGMA = 4.0           # end points this many metres apart are ~60 % similar   # passes the receiver must RUN onto:
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
# Agent1
# =============================================================================

class Agent1(StarterAgent):
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
             The offset of the success curve moves by LEARN_LR * (p - y):
             passes lost more often than predicted -> more careful, and the
             other way round. Every player learns from ALL our passes, so the
             four players end up with (almost) the same value and keep
             predicting each other's choices (HFO: shared experience helps).
          2. DRIBBLES (only the dribbler knows his prediction): same update.
          3. OPPONENT SPEED: a high quantile of their observed running speed.
          4. THEIR DRIBBLING SPEED: average ball speed while they carry it.
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
            # --- 3. opponent running speed (quantile estimate, SGD)
            prev = mem.get("opp_prev", {})
            cur = {}
            for o in snap.opps:
                cur[o.key] = (o.x, o.y, cyc)
                q = prev.get(o.key)
                if o.goalie or q is None or q[2] != cyc - 1:
                    continue
                d = math.hypot(o.x - q[0], o.y - q[1])
                if 0.3 < d < 1.6:
                    L["opp_speed"] += LEARN_SPEED_ETA * (LEARN_SPEED_Q - (1.0 if d < L["opp_speed"] else 0.0))
                    L["opp_speed"] = min(LEARN_SPEED_MAX, max(LEARN_SPEED_MIN, L["opp_speed"]))
            mem["opp_prev"] = cur
            # --- 4. their dribbling speed (ball speed while they carry it)
            pb = mem.get("ball_prev")
            if pb is not None and pb[2] == cyc - 1 and opp_kick and pb[3]:
                d = math.hypot(snap.bx - pb[0], snap.by - pb[1])
                if d < 1.6:
                    L["carrier"] += LEARN_CARRIER_A * (min(1.05, 1.15 * d) - L["carrier"])
                    L["carrier"] = min(1.05, max(0.5, L["carrier"]))
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
                    p = _sig(ev[1], self._pm0(), PASS_K)
                    L["pass_adj"] = min(LEARN_ADJ_MAX, max(LEARN_ADJ_MIN,
                                        L["pass_adj"] + LEARN_LR * (p - y)))
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
                    p = _sig(dv[1], self._dm0(), DRIB_K)
                    L["drib_adj"] = min(LEARN_ADJ_MAX, max(LEARN_ADJ_MIN,
                                        L["drib_adj"] + LEARN_LR * (p - y)))
                    L["n_drib"] += 1
                    L["n_drib_lost"] += int(y == 0.0)
                    mem["drib_ev"] = None
        except Exception as e:          # learning must never break a decision
            note(agent, _err_label("learn", e))
        for o in snap.opps:
            if not o.goalie:
                o.speed = L["opp_speed"]

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
        note(agent, "learn_pass%+d_drib%+d_spd%d_car%d" % (
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
            note(agent, _err_label("on", e))
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
            note(agent, "quick_on_ball")
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
            note(agent, "pass_greedy", to=b0["recv"].unum, p=round(b0["p"], 2))
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
            c = dict(kind="pass", sub=plan.get("sub", "commit"), tx=plan["tx"], ty=plan["ty"], n=plan["n"],
                     v0=plan["v0"], delay=0, recv=recv, one=plan.get("one", False))
            if pressure <= ONE_STEP_PRESS and not c["one"]:
                # v22: he is on us now - finish it with one kick, no wind-up
                c["v0"] = min(c["v0"], ONE_STEP_SPEED) if c["v0"] < ONE_STEP_SPEED else ONE_STEP_SPEED
                c["one"] = True
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
            if "m" in c:
                self.mem["dec"] = (snap.cycle, c["tx"], c["ty"], c["m"], snap.bx, snap.by)
            if not keep_plan:
                self.mem["plan"] = dict(kind="pass", tx=c["tx"], ty=c["ty"], n=c["n"], v0=c["v0"],
                                        recv=c["recv"].unum, cycle=snap.cycle,
                                        commit=COMMIT_PASS, one=c.get("one", False),
                                        sub=c.get("sub", "commit"))
            if c.get("one") and c["v0"] >= ONE_STEP_SPEED - 0.05:
                Action.kick_one_step(agent, tgt)      # v22: no wind-up when pressed
            elif c.get("one"):
                Action.kick(agent, tgt, c["n"], accept_late=1)   # slow = one touch
            elif pressure <= 1:
                Action.kick_one_step(agent, tgt)      # no time for a wind-up
            else:
                Action.kick(agent, tgt, c["n"], accept_late=5)
        elif k == "dribble" and c.get("kick"):
            # v22 self-pass: kick it ahead, then run onto it (see _off_ball)
            self.mem["plan"] = None
            self.mem["selfpass"] = (snap.cycle, c["tx"], c["ty"], c["tend"] + 4)
            Action.kick(agent, tgt, max(1, c["n"]), accept_late=3)
            Action.dribble(agent, tgt)
        elif k == "dribble":
            if not keep_plan:
                if "m" in c and self.mem.get("drib_ev") is None:
                    self.mem["drib_ev"] = (snap.cycle, c["m"], int(c.get("tend", 5)))
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
        p = 1.0 / (1.0 + math.exp(-z))
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
        """v26 MMR selection (maximal marginal relevance): pick the best
        option, then repeatedly the one with the highest
            value - DIV_W * (similarity to the most similar option already picked)
        so the look-ahead gets K DIFFERENT ideas, not 10 copies of one pass.
        Similarity of two options: exp(-d^2 / 2 sigma^2) of the distance
        between where the ball ends up, x1 if same receiver / same kind, x0.5
        otherwise; shots vs anything else 0, holds vs anything else 0."""
        if len(cands) <= k:
            return list(cands)
        rest = sorted(cands, key=lambda c: -c["v"])
        chosen = [rest.pop(0)]
        two_s2 = 2.0 * DIV_SIGMA * DIV_SIGMA

        def sim(a, b):
            if (a["kind"] in ("shot", "hold")) != (b["kind"] in ("shot", "hold")):
                return 0.0
            if a["kind"] != b["kind"] and "shot" in (a["kind"], b["kind"]):
                return 0.0
            ax, ay = a.get("ex", a.get("tx", 0.0)), a.get("ey", a.get("ty", 0.0))
            bx, by = b.get("ex", b.get("tx", 0.0)), b.get("ey", b.get("ty", 0.0))
            d2 = (ax - bx) ** 2 + (ay - by) ** 2
            ra = getattr(a.get("recv"), "unum", a["kind"])
            rb = getattr(b.get("recv"), "unum", b["kind"])
            same = 1.0 if (ra == rb and a["kind"] == b["kind"]) else 0.5
            return same * math.exp(-d2 / two_s2)

        msim = [sim(c, chosen[0]) for c in rest]
        while len(chosen) < k and rest:
            j = max(range(len(rest)), key=lambda i: rest[i]["v"] - DIV_W * msim[i])
            c = rest.pop(j)
            msim.pop(j)
            chosen.append(c)
            msim = [max(m, sim(r, c)) for m, r in zip(msim, rest)]
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
            note(agent, _err_label("off", e))
            StarterAgent.off_ball(self, agent)
        self._learn_note(agent)

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

        # --- v22 self-pass in flight: the kicker runs onto it (nobody else)
        sp = mem.get("selfpass")
        if sp is not None and not mate_kick and not opp_kick:
            if snap.cycle - sp[0] <= sp[3] and our_t.get(me.unum, 99) <= opp_min + 1:
                note(agent, "selfpass_run")
                Action.intercept(agent)
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
        # v22: along the way he is REALLY going (replays: carriers ran 30 m
        # down the touchline while we blocked the straight line to goal)
        ux, uy, _ = self._carrier_dir(snap, carrier_xy=(lx, ly))
        if not self._goal_side(snap, me, lx, ly, slack=-2.0):
            # clearly behind him: race to a point on his path we can reach first
            rx, ry = self._recovery_point(snap, me, lx, ly, heading=True)
            note(agent, "press_recover")
            self._go(agent, snap, rx, ry, urgent=True, thr=0.5)
            return
        px, py = lx + ux * BLOCK_D, ly + uy * BLOCK_D
        bd = math.hypot(snap.bx - me.x, snap.by - me.y)
        if bd > CLOSE_DOWN + 2.0 and math.hypot(px - me.x, py - me.y) > 4.0:
            # goal-side but still far: cut him off where we get first instead
            # of running after a block point that moves away with him
            rx, ry = self._recovery_point(snap, me, lx, ly, heading=True)
            note(agent, "press_cut")
            self._go(agent, snap, rx, ry, urgent=True, thr=0.5)
            return
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
        role, (tx, ty) = best.get(me.unum, ("open", _xy(Tools.home_position(agent))))

        if role.startswith("box_"):
            # small search for space around the run target (keeps the run)
            tx, ty = self._get_open(snap, hx, hy, tx, ty, [(hx, hy)], False, radius=(2.5,))
            note(agent, "att_" + role)
            self._go(agent, snap, tx, ty, urgent=True, thr=0.7)
            return

        if role == "decoy":
            if not beyond_ok:
                tx = min(tx, snap.offside - 0.7)
            note(agent, "att_decoy")
            self._go(agent, snap, tx, ty, urgent=True, thr=0.7)
            return

        if role == "show":
            tx, ty = self._clamp_near(snap, hx, hy, tx, ty)
            note(agent, "att_" + role + brk_tag)
            self._go(agent, snap, tx, ty, urgent=(role == "show" or brk), thr=0.7)
            return

        if role == "anchor":
            # small refinement: keep a clear lane for the back pass
            tx, ty = self._get_open(snap, hx, hy, tx, ty, [(hx, hy)], False, radius=(3.0,),
                                    allow_ahead=False)
            note(agent, "att_anchor" + brk_tag)
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
        note(agent, "att_" + role + ("_push" if push else "") + brk_tag)
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

# STARTER CODE ENDS HERE #