"""
Subclass `StarterAgent` and implement `on_ball` and `off_ball`. The framework
calls exactly one of them per cycle, per player:

    kickable  -> on_ball(agent)
    otherwise -> off_ball(agent)

Everything else - perception, formations, set pieces, the goalie,
penalties, neck and view - is handled for you and is identical for
every team.

Try not to get overwhelmed by the different helper functions and world model parameters.
Many of these may not even be required by you initially. A lot of them are mainly there
in CASE you wish to build an evaluator that factors them in, but you can make a competitive
agent utilizing only a few of them if you want. Try to reason about what you want to do, and 
then look at the helpers, rather than trying to use all of them forcibly.

WHERE YOUR CODE GOES

  src/student/agent.py. One class, subclassing StarterAgent, overriding
  on_ball and off_ball. That one file is what you hand in.

    from src.student.interface import StarterAgent, Action, Tools, note

  src/student/framework/ is plumbing. You never need to open it, and
  importing from it directly is not permitted.

World model: agent.wm

  Everything is exact. Full state is on, so there is no uncertainty
  (except in pass success rate).

  Positions and velocities are pyrusgeom Vector2D, so
  wm.ball.position.dist(x), .x(), .y(), .r(), .th() all work directly.
  Note .x() is a METHOD, not an attribute.

  Coordinates are always from your team's point of view: your goal is
  at x = -52.5 and you attack towards +x, whichever side you are really
  playing on. Facing +x, -y is your left and +y your right, so angles
  turn clockwise: 0 = ahead, 90 = right, 180 = behind, -90 = left.

  wm.self          position velocity body_direction neck_direction
                   stamina effort recovery stamina_capacity
                   is_kickable kick_rate uniform_number is_goalie
                   tackle_probability foul_probability
  wm.ball          position velocity dist_from_self
  wm.teammates     list[Player]  - excludes you
  wm.opponents     list[Player]
     Player:       uniform_number position velocity body_direction
                   dist_from_self angle_from_self dist_from_ball
                   ball_reach_steps inertia_final_point is_goalie

  wm.intercept_table.self_reach_steps
                   .first_teammate_reach_steps
                   .first_opponent_reach_steps
  wm.offside_line_x  wm.our_defense_line_x  wm.their_defense_line_x
  wm.kickable_teammate_existance / _opponent_existance
  wm.our_goalie_uniform_number / their_goalie_uniform_number
  wm.last_kicker_uniform_number  wm.cycle  wm.game_mode_type
  wm.our_team_score  wm.their_team_score

WHAT YOU CAN DO       (first emitted body action is the one that counts)

  This is the whole action space. There is nothing else to find.

  ON BALL   (only reachable from on_ball)
    Action.kick(agent, target, arrive_in, accept_late=3)
    Action.kick_one_step(agent, target)       # 1 touch, max pace, now
    Action.dribble(agent, target)
    Action.hold(agent)
    Action.clear(agent)                       # wide, fast, never fails
    Action.advance_ball(agent)                # up the clearest lane
    Action.stop_ball(agent)

  OFF BALL
    Action.go_to_point(agent, target, max_dash_power=100,
                    distance_threshold=None,   # None -> ball_dist*0.1
                    dodge=False)
    Action.intercept(agent)
    Action.tackle(agent, direction=0.0, foul=False)

  BODY ANGLE   (either half)
    Action.turn_to_point(agent, target)
    Action.turn_to_angle(agent, angle)
    Action.turn_to_ball(agent)

  If every action you emit fails, NO command is sent and your player
  coasts to a stop. Always end with a fallback e.g. hold() or clear()
  on the ball, go_to_point(home) off it.

  kick() is the exception: when it cannot get there in time it shields
  the ball itself, so a fallback behind it never runs.


YOUR THINKING TIME

  agent.budget.remaining_ms is the cpu time left for the WHOLE REST OF
  THE MATCH, not for this cycle. Deciding how much of it to spend here
  is part of the problem. Write searches that it can stop and return the
  best answer so far.

      while candidates and agent.budget.remaining_ms > 400:
          deepen(candidates)
      return best_so_far

  The budget is teamwide and counted as the MAX across your players,
  not the sum. Loading your file (imports, module-level code, __init__)
  is charged to it too.

  Separately, one decision may take at most 1 second of wall clock, so
  the server never stalls.

      agent.budget.remaining_ms            cpu ms left in the whole match
      agent.budget.decision_remaining_ms   wall ms left of this decision's 1s

  A decision is cut off when EITHER reaches 0, and that player does
  nothing this cycle. Stop well before both:

      while (candidates and agent.budget.remaining_ms > 400
             and agent.budget.decision_remaining_ms > 50):
          deepen(candidates)

  Being cut off is meant to never happen, so it is punished:
    * a player cut off 5 times is benched: no more decisions this match.
    * a decision still running at 1.5s forfeits: that player leaves the
      match. Only a call that can't be interrupted gets there - sleep(),
      or one huge numpy call started just before the deadline.

  agent.budget is read-only.


SEEING WHAT YOU DID

    note(agent, "chase")
    note(agent, "pass", to=best.uniform_number, score=0.82)

  Then `python3 a3.py match A B` and `python3 a3.py why`, which will show
  the labels, how often each fired, and how many of your players shared
  one at the same time (`python3 a3.py why --stack chase`). Printing does not
  work - all processes share a terminal. (Use the --no-trace flag to disable tracing)


HELPERS

  Pass `agent` to these, never anything else; they unwrap what they
  need internally.
  You can read src/utils/tools.py or this file for more information.

  Ball flight
    Tools.first_ball_speed(dist, steps, ball_decay)   dist+time -> speed
    Tools.ball_move_step(speed, dist, ball_decay)     speed -> cycles
    Tools.calculate_ball_inertia_final_point(pos, vel, decay)
    Tools.inertia_point(pos, vel, n_step, decay)      where it is at cycle n
                                                      (ball or player - pass
                                                       the matching decay)

  Who gets there first
    Tools.predict_opponent_reach_step(agent, opp, ball_pos, ball_vel, max_cycle)
    Tools.cycles_to_reach_distance(agent, player, dash_dist)
    Tools.predict_player_turn_cycle(sp, ptype, body, speed, dist, ...)
    Tools.inertia_final_point(ptype, pos, vel)
    Tools.effective_turn(moment, speed, inertia_moment)
    Tools.estimate_min_reach_cycle(pos, speed_max, from_point, move_angle)
    Tools.estimate_virtual_dash_distance(player, real_speed_max)

  Kicking
    Tools.steps_required(speed, dist, ball_decay, ball_speed_max)
                                                  the arrive_in for kick() that
                                                  comes closest to `speed`
    Tools.predict_kick_count(agent, unum, first_speed, move_angle)
    Tools.calc_max_velocity(angle, kick_rate, ball_vel, server_params)

  Lookups
    Tools.nearest_to(players, point)              closest of ANY list of
                                                  players, e.g. your own subset
    Tools.get_opponent_nearest_to(agent, point)
    Tools.get_opponent_nearest_to_self(agent)
    Tools.get_opponent_goalie(agent)
    Tools.get_teammate_nearest_to(agent, point)
    Tools.exist_opponent_in(agent, region)        # region: Sector2D etc
    Tools.get_teammates_from_ball(agent)          # sorted by ball distance
    Tools.get_opponents_from_ball(agent)
    Tools.get_teammates_from_self(agent)
    Tools.get_opponents_from_self(agent)

  Stamina
    Tools.get_dash_power_to_keep_speed(agent, speed, effort)

  Positioning
    Tools.home_position(agent) -> Vector2D        your formation slot
    Tools.points_around(center, radius, n, start_deg=0.0)
                                                  n points evenly spaced on
                                                  a circle round center;
                                                  start_deg 0 = +x, towards
                                                  the opponent goal; 90 =
                                                  your right, -90 = left

  Framework
    agent.server_params.ball_decay / ball_speed_max / max_power /
                        stamina_max / pitch_half_length / ...
    agent.memory                     dict, persists across cycles for
                                     THIS player only - it is not a
                                     channel to your teammates
    agent.add_log_message(...)       draws on the replay in soccerwindow2

  Geometry: Vector2D, Sector2D, Line2D, AngleDeg from pyrusgeom


NOT AVAILABLE

  Any helios_* or bhv_* action. Building that behaviour yourself,
  out of the actions above, is the assignment.

  say / attentionto / point_to. There is no communication. Under full
  state every agent already has the same information, so coordinating
  means computing the same allocation as your teammates, not announcing
  it to them.

  No filesystem, no network, no shared state between agents. Every
  player runs in its OWN process. If you want your players to agree on
  something, they must each compute it from what they can see.
"""
from abc import ABC, abstractmethod

from pyrusgeom.geom_2d import Vector2D
from pyrusgeom.sector_2d import Sector2D
from service_pb2 import (PlayerAction, Body_SmartKick, Body_KickOneStep,
                         Body_Dribble, Body_HoldBall, Body_ClearBall,
                         Body_AdvanceBall, Body_StopBall, Body_GoToPoint,
                         Body_GoToPointDodge, Body_Intercept,
                         Body_TurnToPoint, Body_TurnToAngle, Body_TurnToBall,
                         Tackle)

from src.student.framework.world import _convert, _unwrap
from src.utils.tools import Tools as _Tools
from src.student.framework.trace import note   

# The framework helpers, with the converted agent.
_EXPOSED = (
    # ball flight
    "first_ball_speed", "ball_move_step", "calculate_ball_inertia_final_point",
    "inertia_point",
    # who gets there first
    "predict_opponent_reach_step", "cycles_to_reach_distance",
    "predict_player_turn_cycle", "inertia_final_point", "effective_turn",
    "estimate_min_reach_cycle", "estimate_virtual_dash_distance",
    # kicking
    "predict_kick_count", "calc_max_velocity",
    # lookups
    "get_opponent_nearest_to", "get_opponent_nearest_to_self",
    "get_opponent_goalie", "get_teammate_nearest_to", "exist_opponent_in",
    "get_teammates_from_ball", "get_opponents_from_ball",
    "get_teammates_from_self", "get_opponents_from_self",
    # stamina
    "get_dash_power_to_keep_speed",
)


def _wrapper(name, fn):
    def call(*args, **kw):
        return _convert(fn(*[_unwrap(a) for a in args],
                           **{k: _unwrap(v) for k, v in kw.items()}))
    call.__name__ = name
    call.__doc__ = getattr(fn, "__doc__", None)
    return staticmethod(call)


class Tools:
    @staticmethod
    def home_position(agent) -> Vector2D:
        """Where the formation says you should be this cycle.

        The formation is given and identical for every team."""
        a = _unwrap(agent)
        return _convert(a.strategy.get_position(a.wm.self.uniform_number, a))

    @staticmethod
    def points_around(center: Vector2D, radius: float, n: int,
                      start_deg: float = 0.0) -> list:
        """n points evenly spaced on a circle of `radius` round `center`,
        the first at `start_deg` (0 = towards the opponent goal, +x;
        90 = your right, -90 = your left), going clockwise."""
        step = 360.0 / max(1, n)
        return [center + Vector2D.polar2vector(radius, start_deg + step * i)
                for i in range(n)]

    @staticmethod
    def steps_required(speed: float, dist: float, ball_decay: float,
                       ball_speed_max: float):
        """The arrive_in for kick() whose launch speed comes closest to
        `speed` over `dist`. None if a ball kicked at `speed` never gets
        that far.

        Tools.ball_move_step always rounds up.
        This takes whichever neighbouring step is nearer."""
        n = _Tools.ball_move_step(speed, dist, ball_decay)
        if n < 0:
            return None
        if n <= 1:
            return 1
        lo = min(_Tools.first_ball_speed(dist, n - 1, ball_decay),
                 ball_speed_max)
        hi = _Tools.first_ball_speed(dist, n, ball_decay)
        return n - 1 if abs(lo - speed) <= abs(hi - speed) else n

    @staticmethod
    def nearest_to(players, point: Vector2D):
        """Whichever of `players` is closest to `point`, or None if the
        list is empty. Any list works: wm.teammates, wm.opponents, or a
        subset you picked."""
        return min(players, key=lambda p: p.position.dist2(point),
                   default=None)


for _name in _EXPOSED:
    _fn = getattr(_Tools, _name, None)
    if _fn is None:
        raise AttributeError(
            f"_EXPOSED names {_name!r}, which src/utils/tools.py does not "
            "define")
    setattr(Tools, _name, _wrapper(_name, _fn))
del _name, _fn


# Action wrappers; this is your intended action space.

class Action:
    @staticmethod
    def _v(p):
        """Accepts a pyrusgeom Vector2D and returns the protobuf form the
        proxy expects."""
        return _Tools.convert_vector2d_to_rpc_vector2d(p)

    # on the ball

    @staticmethod
    def kick(agent, target: Vector2D, arrive_in: int, accept_late: int = 3):
        """Send the ball to `target`, over up to 3 touches. Pass, shot and
        clearance are all this - only target and pace differ.

        arrive_in is whole cycles until the ball gets there. The speed
        is capped, so the ball lands short of a target beyond ~45m.

        accept_late is how many cycles later than that you will still
        settle for. If no sequence of touches is that quick, NOTHING IS
        KICKED: you just shield the ball.

        accept_late:
            1    within a cycle - the strictest you can ask for
            3    the default
            10   almost never refuses

        arrive_in=1 asks the ball to cover the whole distance in one
        cycle, so beyond 2.7m it is simply 'maximum power' and always
        succeeds regardless of accept_late."""
        accept_late = max(1, accept_late)

        sp = agent.server_params
        decay = sp.ball_decay
        n = max(1, int(arrive_in))
        dist = target.dist(agent.wm.ball.position)
        ideal = dist * (1.0 - decay) / (1.0 - decay ** n)
        if ideal >= sp.ball_speed_max:
            first_speed, thr = sp.ball_speed_max, 0.1
        else:
            first_speed = ideal
            late = n + int(accept_late)
            thr = dist * (1.0 - decay) / (1.0 - decay ** late)
        agent.add_action(PlayerAction(body_smart_kick=Body_SmartKick(
            target_point=Action._v(target),
            first_speed=first_speed,
            first_speed_threshold=min(max(thr, 0.1), first_speed),
            max_steps=3)))

    @staticmethod
    def kick_one_step(agent, target: Vector2D):
        """One touch kicked immediately, as kick() may spend three cycles
        winding up. Never refuses."""
        agent.add_action(PlayerAction(body_kick_one_step=Body_KickOneStep(
            target_point=Action._v(target),
            first_speed=agent.server_params.ball_speed_max)))

    @staticmethod
    def dribble(agent, target: Vector2D):
        """Advance with the ball, keeping it at your feet."""
        agent.add_action(PlayerAction(body_dribble=Body_Dribble(
            target_point=Action._v(target),
            distance_threshold=1.0, dash_power=100.0,
            dash_count=2, dodge=True)))

    @staticmethod
    def hold(agent):
        """Shield the ball. Buys a cycle when nothing is on."""
        agent.add_action(PlayerAction(body_hold_ball=Body_HoldBall()))

    @staticmethod
    def clear(agent):
        """Get rid of it. Aims wide and forward, toward the touchline
        ahead of you, and releases in a single touch as soon as an
        opponent is close enough to take the ball off you. Never fails,
        so it is the right last line of on_ball."""
        agent.add_action(PlayerAction(body_clear_ball=Body_ClearBall()))

    @staticmethod
    def advance_ball(agent):
        """Put the ball 30m up the clearest lane ahead of you. Unlike
        clear() it takes its time choosing, so use it when you have
        nothing on but nobody is on you either."""
        agent.add_action(PlayerAction(body_advance_ball=Body_AdvanceBall()))

    @staticmethod
    def stop_ball(agent):
        """Trap the ball: kill its speed relative to you so it sits at
        your feet, ready to be kicked anywhere next cycle.

        One touch can only change the ball's velocity by so much, so a
        ball arriving fast cannot be sent back the way it came in a
        single kick. Spending a cycle here buys that freedom.

        This is the one on-ball action that reports failure without
        doing anything, so a fallback after it still runs."""
        agent.add_action(PlayerAction(body_stop_ball=Body_StopBall()))

    # off the ball

    @staticmethod
    def go_to_point(agent, target: Vector2D, max_dash_power: float = 100.0,
                    distance_threshold: float = None, dodge: bool = False):
        """Run to a point.

        distance_threshold is 'close enough'. The default is loose when
        the ball is far and tight when it is near.

        dodge steers around anyone in the way rather than running through
        them. If dodging, distance_threshold is irrelevant."""
        if dodge:
            agent.add_action(PlayerAction(
                body_go_to_point_dodge=Body_GoToPointDodge(
                    target_point=Action._v(target),
                    dash_power=max_dash_power)))
            return
        if distance_threshold is None:
            distance_threshold = max(agent.wm.ball.dist_from_self * 0.1, 1.0)
        agent.add_action(PlayerAction(body_go_to_point=Body_GoToPoint(
            target_point=Action._v(target),
            distance_threshold=distance_threshold,
            max_dash_power=max_dash_power)))

    @staticmethod
    def intercept(agent):
        """Go and get the ball. Predicts where the ball will be and how
        to arrive soonest."""
        agent.add_action(PlayerAction(body_intercept=Body_Intercept()))

    @staticmethod
    def tackle(agent, direction: float = 0.0, foul: bool = False):
        """Lunge at a ball roughly 1-2m ahead. wm.self.tackle_probability
        is chance of winning it.

        direction is DEGREES RELATIVE TO YOUR BODY: 0 sends the ball
        straight ahead, +-180 behind you. Straight ahead is twice as
        much force as straight back, and you lose up to half again if the ball
        is behind you rather than in front, so turning to face it first
        can be worth the cycle.

        foul=True is a deliberate professional foul: always charges the
        opponent, and yields a yellow if the referee spots it
        (foul_detect_probability = 0.5). Gives away a free kick, never a
        penalty."""
        agent.add_action(PlayerAction(
            tackle=Tackle(power_or_dir=direction, foul=foul)))

    # body angle

    @staticmethod
    def turn_to_point(agent, target: Vector2D):
        """Face a point. Body angle sets your kick rate and which way a
        dash takes you, so it is worth a cycle when nothing else is on."""
        agent.add_action(PlayerAction(body_turn_to_point=Body_TurnToPoint(
            target_point=Action._v(target), cycle=1)))

    @staticmethod
    def turn_to_angle(agent, angle: float):
        """Face an absolute pitch angle, in degrees."""
        agent.add_action(PlayerAction(
            body_turn_to_angle=Body_TurnToAngle(angle=angle)))

    @staticmethod
    def turn_to_ball(agent):
        """Face the ball."""
        agent.add_action(PlayerAction(
            body_turn_to_ball=Body_TurnToBall(cycle=1)))


class Agent(ABC):
    """Your team, in one class.

    The framework calls exactly one method per cycle, per player:

        on_ball(agent)   when wm.self.is_kickable
        off_ball(agent)  otherwise

    At most one player per team is on the ball at a time, so off_ball is
    roughly 95% of every decision your team makes.

    `self` persists across cycles for a single player.
    """

    @abstractmethod
    def on_ball(self, agent) -> None:
        ...

    @abstractmethod
    def off_ball(self, agent) -> None:
        ...


# The baseline you start from. 
# It has some obvious weaknesses like:
#
#   * pass_ scores candidates with max(..., key=lambda p: p.x()) -
#     furthest upfield wins, regardless of other positioning.
#
#   * block_point depends only on where the BALL is.
#     No teammate-aware decision.
#
#   * the chase test fires for ANY player within 3 steps, so several
#     teammates can set off after the same ball.
#
#   * shoot fires from anywhere inside 25m and never looks at where the
#     goalkeeper is standing.
#
#   * it only passes when an opponent is within 10m. No creation.

class StarterAgent(Agent):
    SHOOT_RANGE = 25.0        # shoot from anywhere inside this
    PRESSURE = 10.0           # "under pressure" = look for a pass
    COMFORT = 2.5             # nobody this close = just hold
    PASS_ARRIVE = 5           # cycles for a pass to arrive
    DRIBBLE_DIST = 5.0

    def on_ball(self, agent) -> None:
        wm = agent.wm
        print(wm.our_team_score, wm.their_team_score, wm.cycle)
        opps = [o for o in wm.opponents if o.uniform_number > 0]
        nearest = min((o.dist_from_self for o in opps), default=1000.0)

        if self.shoot(agent):
            return
        if nearest < self.PRESSURE and self.pass_(agent):
            return
        if self.dribble(agent):
            return
        if nearest > self.COMFORT:
            Action.hold(agent)
            return
        Action.clear(agent)      # always end with something that cannot fail

    def shoot(self, agent) -> bool:
        """Aim at the far post from inside 25m."""
        wm = agent.wm
        ball = wm.ball.position
        half = agent.server_params.pitch_half_length
        post = agent.server_params.goal_width / 2.0 - 0.5

        if ball.dist(Vector2D(half, 0.0)) > self.SHOOT_RANGE:
            return False

        near, far = Vector2D(half, -post), Vector2D(half, post)
        target = far if ball.dist(near) < ball.dist(far) else near

        Action.kick_one_step(agent, target)
        return True

    def pass_(self, agent) -> bool:
        """Sample a ring of points around each teammate, drop the ones
        that are illegal or obviously blocked, take whichever survivor
        is furthest upfield."""
        wm = agent.wm

        candidates = []
        for tm in wm.teammates:
            if tm.uniform_number <= 0 or tm.is_goalie:
                continue
            for p in Tools.points_around(tm.position, 2.0, 12):
                if self.valid_target(agent, p):
                    candidates.append(p)

        if not candidates:
            return False

        best = max(candidates, key=lambda p: p.x())

        Action.kick(agent, best, self.PASS_ARRIVE)
        return True

    def valid_target(self, agent, target: Vector2D) -> bool:
        wm = agent.wm
        ball = wm.ball.position

        if target.dist(ball) > 30.0:
            return False
        if wm.self.position.dist(target) < 2.0:
            return False
        if target.x() > wm.offside_line_x - 0.5:
            return False

        lane = Sector2D(ball, 1.0, target.dist(ball) + 3.0,
                        (target - ball).th().degree() - 15.0,
                        (target - ball).th().degree() + 15.0)
        return not Tools.exist_opponent_in(agent, lane)

    def dribble(self, agent) -> bool:
        """Three directions, take whichever ends nearest their goal."""
        wm = agent.wm
        ball = wm.ball.position
        sp = agent.server_params
        goal = Vector2D(sp.pitch_half_length, 0.0)

        base = (goal - ball).th().degree()
        targets = []
        for off in (-30.0, 0.0, 30.0):
            p = ball + Vector2D.polar2vector(self.DRIBBLE_DIST, base + off)
            if abs(p.x()) > sp.pitch_half_length - 1.0:
                continue
            if abs(p.y()) > sp.pitch_half_width - 1.0:
                continue
            lane = Sector2D(ball, 0.5, self.DRIBBLE_DIST + 1.0,
                            base + off - 20.0, base + off + 20.0)
            if Tools.exist_opponent_in(agent, lane):
                continue
            targets.append(p)

        if not targets:
            return False

        Action.dribble(agent, min(targets, key=lambda p: p.dist(goal)))
        return True

    def off_ball(self, agent) -> None:
        wm = agent.wm
        it = wm.intercept_table

        my_steps = it.self_reach_steps
        nearest_mate = it.first_teammate_reach_steps
        nearest_opp = it.first_opponent_reach_steps

        if (not wm.kickable_teammate_existance
                and (my_steps <= 3
                     or (my_steps <= nearest_mate and my_steps < nearest_opp + 3))):
            Action.intercept(agent)
            return

        if nearest_opp < min(my_steps, nearest_mate):
            block = self.block_point(agent)
            if block is not None:
                Action.go_to_point(agent, block, max_dash_power=100.0)
                return

        Action.go_to_point(agent, Tools.home_position(agent),
                           max_dash_power=self.dash_power(agent))

    def block_point(self, agent):
        """A few metres up the line from the ball toward our own goal."""
        ball = agent.wm.ball.position
        goal = Vector2D(-agent.server_params.pitch_half_length, 0.0)
        to_goal = goal - ball
        if to_goal.r() < 1.0:
            return None
        return ball + to_goal.set_length_vector(min(8.0, to_goal.r() * 0.4))

    def dash_power(self, agent) -> float:
        """Conserve below half stamina, full above 70 percent."""
        ratio = agent.wm.self.stamina / agent.server_params.stamina_max
        if ratio > 0.7:
            return 100.0
        if ratio < 0.5:
            return 50.0
        return 75.0