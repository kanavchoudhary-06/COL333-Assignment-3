COL333 Assignment 3 - Football Agent: Design Notes
Status: DRAFT (ideas being discussed; "AGREED" = finalised, "OPEN" = still to decide)
====================================================================================

0. GENERAL PRINCIPLES (from the interface / brief)
------------------------------------------------
- No forward simulator is exposed. "Search" runs on our own light model built from
  the helpers: ball flight (inertia_point, ball_decay) + who-gets-there-first
  (intercept_table, predict_opponent_reach_step, cycles_to_reach_distance).
- Time: ~75 s CPU / 1500 cycles = ~50 ms per cycle per player on average.
  off_ball is ~95% of calls -> keep it cheap; spend more in on_ball.
  Allowance per decision should adapt to agent.budget (no hard-coded timings).
- Coordination without communication: every player runs the SAME deterministic
  calculation on the same (exact) world state, with fixed tie-breaks
  (e.g. lower uniform number wins), so they all reach the same plan.


1. DEFENDING (opponent has the ball, or will reach it first)
-----------------------------------------------------------
1.1 Opponent modelling - depth-1 adversarial idea            [AGREED - core idea]
    - Put ourselves in the shoes of their ball carrier: generate their candidate
      actions (passes, shots, dribbles) and score them with OUR evaluator,
      seen from their side.
    - Keep the best 2-3, weighted by score (not a hard top-1), because our
      evaluator is only a guess of theirs.
    - Defend against those threats:
        shot     -> stand on the line ball -> goal centre
        dribble  -> presser stays between carrier and our goal
        pass     -> mark the likely receiver, on the pass line, goal-side
    - Proper minimax version: for each candidate defensive assignment, re-check
      their best option given our new positions; pick the assignment that
      minimises it (only 4! = 24 assignments).

1.2 Who is the "carrier"?                                     [PROPOSED]
    - Opponent kickable -> that player.
    - Ball loose / in flight and an opponent reaches it first (intercept table)
      -> treat that opponent at the interception point as the future carrier.
    - We reach it first -> not defending; our fastest player intercepts.

1.3 Roles                                                     [PROPOSED]
    - PRESSER: always exactly one = fewest reach-steps to the ball
      (tie -> lower uniform number). Closeness measured in CYCLES, not metres.
        near -> close down / tackle if allowed
        far  -> hold a delaying spot between ball and our goal (don't chase blindly)
    - MARKERS: remaining players assigned ONE-TO-ONE to the most dangerous
      opponents / predicted receivers. Mark the PLAYER (not a fixed point):
      stand goal-side, slightly towards the ball / on the pass line.
      Danger = close to our goal, central, ahead of the ball, easy to pass to.
    - FALLBACK (nobody worth marking): formation slot shifted towards the ball
      and our goal (keep shape) - NOT everyone parking at the goal.

1.4 Extra rules                                               [PROPOSED]
    - Stability: keep the previous assignment unless the new one is better by
      a margin (avoids zig-zagging).
    - Recompute assignment every k cycles (same rule for everyone, e.g. cycle % 3).
    - Keep one coherent defensive line (a lagging defender plays everyone onside).
    - Tackling: only if tackle_probability high AND foul_probability low;
      stricter in our own third; never foul=True near our box.
    - Stamina: full dash only when pressing / recovering, otherwise conserve.
    - Never pass back to our own goalie (indirect free kick risk).

1.5 Pressing intensity                                       [AGREED]
    - Press ALWAYS (whole pitch), exactly one presser.
    - Teammates' stamina is NOT visible (only wm.self has stamina), so the team
      cannot coordinate on "who is tired". A tired presser only lowers his own
      dash power; revisit if matches show it is a problem.

1.6 Counter-attack runner (anticipation)                      [AGREED]
    - Trigger: an opponent pass is in flight AND our first player reaches it
      clearly before any opponent (margin >= ~2 cycles; tune).
    - Exactly ONE runner, chosen deterministically (e.g. most advanced
      non-interceptor, tie -> lower uniform number). Everyone else keeps marking
      until the ball is actually ours (safety if the interception fails).
    - Target: sample points ahead; score = space from opponents + progress
      + reachable by a pass from the interception point; must stay ONSIDE
      (x < offside_line_x).

1.7 Offside                                                   [AGREED]
    - No active offside trap for now (failure = 1v1 vs our goalie).
    - Defenders hold one common, reasonably high line and don't drop deeper than
      needed (a tidy line already blocks many of their forward passes).
    - Real "step up" trap = later experiment only.

1.8 Economical interception ("just fast enough")              [AGREED]
    - Scan t = 1..N: ball position at t (inertia_point); our cycles to reach it;
      fastest opponent's cycles. Pick the point where we arrive first with a
      safety margin, and the LOWEST dash power that still makes it in time
      (go_to_point max_dash_power).
    - Close race (small margin) or opponent nearby -> sprint with Action.intercept
      (predictions are approximate; arriving late also gives less time on ball).
    - Last 2-3 cycles before reaching the ball -> switch to Action.intercept for
      precise control (go_to_point may overshoot).


2. ATTACKING (we have the ball)                               [OPTIONS - not decided]
-------------------------------------------------------------
2.1 On-ball decision framework
    Opt 1  Depth-1: score each candidate as
           P(success)*V(result) + (1-P)*V(lose ball there).
    Opt 2  Action-chain search (HELIOS): best-first tree over sequences
           (pass -> dribble -> pass -> shot). HELIOS found deeper search did NOT
           clearly help, because future-state prediction gets worse with depth.
           -> suggestion: depth 2 (pass + receiver's best next action), anytime.
    Opt 3  Multiple "desirable states" (Gliders). Lesson: an x-only evaluation
           over-rates dribbles vs passes, so V needs more than progress.

2.2 Candidate actions
    - Passes: direct (to feet), lead (space beside receiver), through (in front
      of receiver, behind defenders); 2-3 speeds (arrive_in) each.
    - Shots: points across goal mouth (inside posts); pick the safe point where
      opponents/goalie need the LONGEST to reach the ball (Cyrus).
    - Dribbles: 6-8 directions x 1-2 distances.
    - Fallbacks: hold, advance_ball, clear.
    - TO TEST: does the simulator judge offside by receiver position at kick
      time (real rule)? If yes, through passes are legal and valuable.

2.3 Pass / shot / dribble success estimate (CORE FUNCTION)
    - Simulate ball cycle by cycle; at each cycle compare ball arrival with
      each opponent's reach time; opponent there at same time or sooner ->
      interceptable (Cyrus).
    - Prune opponents clearly too far from the path.
    - Margin (cycles) -> probability via a smooth curve (sigmoid).

2.4 Evaluation V(state) - features
    - progress (x), distance + angle to goal, safe-shot-available bonus,
      pressure on receiver, space, big penalty for losing ball near own goal.
    - Weights tuned by matches.

2.5 Off-ball attacking positioning
    - YuShan 2022: local grid (9x9, 1 m) around formation target, best point
      chosen by a ranker; separate conservative / aggressive models.
    - Ours: ~16 points (2 rings x 8) around home_position, scored by
      "value of a pass to me here" + progress + space; must be onside;
      choose in uniform-number order to avoid clustering.
    - Possible roles: supporter (short safe option), runner (depth, opens
      through passes), wide player (width).

2.6 On-ball flow - user's plan + refinements                [PROPOSED]
    User's plan: greedy pass to an unmarked teammate ahead; else depth-2
    search with alpha-beta in order (1) forward passes/kicks, (2) forward
    dribbles, (3) backward passes; if all fail -> clear.
    Refined flow:
      1. SHOT CHECK first: safe, high-chance shot -> shoot.
      2. DEPTH-1 scoring of all candidates, generated in the user's order
         (forward passes direct/lead/through -> forward dribbles -> hold ->
         backward passes). The order drives search order / pruning, NOT strict
         priority; the evaluation compares across categories.
      3. GREEDY SHORTCUT (user's idea): if best is a safe forward pass that
         clearly dominates (high P + clear margin over the rest) -> play it now.
         "Unmarked" = pass-success function says safe (lane matters, not only
         distance to receiver).
      4. Else DEPTH-2 on top ~5 passes (receiver's best next action), anytime,
         until the per-decision allowance runs out.
      5. FALLBACK if best value < threshold: clear if under pressure or in own
         third, otherwise hold. Never pass back to own goalie.
    Alpha-beta mapping (for report):
      - MAX node = our action; MIN node = opponents as possible interceptors
        (value = worst case over them).
      - Cutoff: while checking a pass, once one opponent intercepts and the
        value drops below alpha (best found so far), stop checking other
        opponents. Order opponents by closeness to the ball path -> early cutoffs.
      - Depth 2: skip expanding a pass if P(success) * V_max <= alpha
        (branch-and-bound, same spirit).
    Cost control: full depth 2 ~ 100 x 100 = 10k evals (too slow) -> beam of
    top ~5 at depth 2 (~500 evals). After a pass, approximate opponents by
    shifting each towards the receiver by (max speed x flight time).
    Kicks "into space" count only if OUR player wins the race there (= lead /
    through passes), otherwise they are giveaways.

2.7 Depth-2 definition and heuristic                        [AGREED structure; refinements PROPOSED]
    User: depth 2 = two macro actions (a1 = our action, e.g. pass to A;
    a2 = A's best action), then evaluate with: primary = ball still with our
    team; secondary = function of ball x + how clear the holder is
    (min distance from opponents).
    Formalised:
      Value(a1) = P1 * [ P2 * H(after a2) + (1-P2) * Loss(where a2 fails) ]
                + (1-P1) * Loss(where a1 is intercepted)
      - P1, P2 from the MIN layer (opponent interception check).
      - A chooses a2 = argmax of the bracket (second MAX layer).
      - P1*P2 = probability we keep possession (user's primary heuristic).
    Refinements:
      a) Expected value, NOT strict lexicographic "possession first" (strict
         = timid agent that never scores). Loss is negative and gets MORE
         negative closer to OUR goal -> careful at the back, brave up front.
      b) Position score = progress(x) + closeness to their goal centre
         (captures angle) + safe-shot-available bonus. If a2 is a shot:
         value = P(goal) * big reward (terminal).
      c) Clearance in CYCLES (min opponent reach steps to holder), capped at
         ~8-10 (more space beyond that adds nothing; avoids lonely-wing bias).
      d) Discount depth-2 part by gamma ~0.9 (less reliable predictions;
         prefers "good now" over "equally good later").
      H(state) = w1*progress + w2*closeness_to_goal + w3*shot_bonus
               + w4*min(clearance_cycles, cap)
      Loss(p)  = -(base + k*closeness_of_p_to_OUR_goal)
      Weights tuned by matches.
    DECISION: combined expected-value score (not strict possession-first)
    [AGREED - may revisit after testing].

2.8 Off-ball attacking positioning                            [SUPPORTER AGREED; rest PROPOSED]
    SUPPORTER (user's idea, AGREED): one player close to the ball holder,
    a little behind / sideways, keeping an unblocked lane for a safe back pass.
      - Distance ~6-10 m (closer: one opponent covers both; further: pass
        easier to intercept).
      - Side AWAY from the opponent pressing the holder.
      - Spot = best of points on an arc (sideways..behind, both sides), scored by
        pass-success from holder + space from opponents.
      - Doubles as cover if we lose the ball.
      - Near our goal: sideways, not directly behind (risky back pass / goalie
        back-pass rule).
    FORWARD IDEAS (to choose from):
      1. Runner: on the offside line, sprint in behind when holder has time
         and an open lane (through-pass target; pushes their line back).
      2. Width: one player near the touchline (stretches defenders, opens
         central lanes).
      3. Open-spot search: small grid around home_position, score each point =
         P(pass to me here succeeds) * H(point) -> off-ball players go where the
         on-ball search would like to pass (consistent model).
      4. Leave the passing shadow: if an opponent is on the holder->me line,
         step a few metres sideways.
      5. Pass-and-move / one-two: after passing (wm.last_kicker_uniform_number
         == me), run forward into space; depth-2 search can find the return.
      6. Overlap: player behind runs past the holder on the outside (2v1).
      7. Spacing: teammates >= ~8 m apart.
      8. Box occupation when ball is wide in final third (cutback spot near
         penalty spot, slightly behind ball x).
      9. Never stand offside (x < offside_line_x unless running back).
    ROLE ASSIGNMENT: 3 off-ball players -> {supporter, runner, width/open-spot};
      all players compute the same assignment: try all 3! = 6, minimise total
      distance to role targets; switch only if clearly better (hysteresis).
      Zone-dependent: own half -> width can become a 2nd safe option;
      final third -> width fills the box.

    FINAL CHOICE (user delegated role arrangement)             [AGREED]
      Roles: supporter + runner (idea 1) + open-spot search (idea 3).
      Always on: leave passing shadow (4), spacing (7), never offside (9),
      pass-and-move (5). Later experiments: overlap (6), box filling (8).

2.9 Dynamic movement to lose markers ("checking")             [AGREED - user's idea]
    - Trigger only when tightly marked (opponent within ~2-3 cycles) AND pass
      score to current spot is low (if already open, stay - moving wastes
      stamina and may close the lane).
    - Choose, don't jitter: evaluate 3 options with the open-spot score:
      (a) check back towards ball, (b) step sideways, (c) burst forward into
      space behind the marker.
    - Use marker's body_direction: burst behind his back forces a turn
      (predict_player_turn_cycle) -> real separation.
    - Pattern: check towards ball a few cycles, then burst. Bursts 3-5 cycles
      at full power, then jog (stamina).
    - Stay within ~3-5 m of role target (keep shape).
    - Onside: forward burst stops at the offside line unless the holder can
      pass right now (has ball, not pressed) -> then run in behind.
    - Timing offset by uniform number so teammates don't all check together.

2.10 Team moves up with the ball                              [AGREED - user's idea]
    - First test: log home_position at several ball positions - the framework
      formation probably already shifts with the ball; only add what's missing.
    - All role targets + defensive line shift upfield with the ball; keep the
      team COMPACT (last defender to ball <= ~25-30 m): passing options stay
      close, less space for them on turnovers (fits always-press), higher line
      blocks their forward passes.
    - Guardrails: supporter stays slightly behind the ball; line never so high
      that one long ball = 1v1 vs our goalie (cap tuned by matches).

2.11 Shooting at the corners                                  [AGREED - user's idea]
    - Sample points across the goal mouth, 0.5-1 m inside the posts (kick
      noise). Pick the point the goalie needs the MOST cycles to reach ->
      usually the corner away from the goalie (emerges from scoring, not
      hard-coded).
    - Model goalie with a larger reach (can catch with hands).
    - Shoot immediately if P(goal) >= threshold (~0.6, tune); else the shot
      competes in the normal search.
    - Missed shot -> usually goal kick / keeper catch, not a counter -> mild
      Loss (encourages shooting from good positions).
    - Opponent close -> kick_one_step (instant); free -> kick at max pace
      (2-3 touches, faster cleaner shot).

2.12 OPEN decisions
    - Search depth: 1 only, or 2 when time allows? (suggested: 2)
    - Through passes: include (after offside test)?
    - Off-ball: roles + local grid, or plain grid?
    - Risk style switch by score/time (conservative vs aggressive)?


3. TIME MANAGEMENT                                            [PROPOSED]
----------------------------------------------------------------
    1. Per-decision allowance = remaining_ms / cycles_left.
       On-ball: up to ~8x allowance (cap ~300 ms) - rare (~5% of calls).
       Off-ball: ~1x; shared computations (role assignment) every k cycles,
       reused in between.
    2. Hard stop: always finish when decision_remaining_ms < 150.
    3. Graceful degradation when budget is low:
       depth 2 -> depth 1 -> greedy rules (never time out = do nothing).
    4. Measure, don't guess: time one candidate evaluation, compute how many
       are affordable this decision (adapts to grading machine).
    5. Keep module-level / __init__ work tiny (charged to the budget).


4. REFERENCES (RoboCup 2D team description papers)
--------------------------------------------------
- HELIOS2013 TDP (action-chain best-first search; deeper not clearly better):
  https://staff.fnwi.uva.nl/a.visser/activities/robocup/RoboCup2013/Symposium/TeamDescriptionPapers/SoccerSimulation/Soccer2D/TDP_HELIOS2013.pdf
- HELIOS2017 TDP (action-sequence search + pruning):
  https://2017.robocup.org/file/symposium/soccer_sim_2D/TDP_HELIOS2017.pdf
- Gliders2012 (action-dependent evaluation, chess analogy):
  https://arxiv.org/abs/1211.3882
- Cyrus 2013 TDP (lead/through passes, cycle-by-cycle pass safety, shot point):
  https://staff.fnwi.uva.nl/a.visser/activities/robocup/RoboCup2013/Symposium/TeamDescriptionPapers/SoccerSimulation/Soccer2D/TDP_Cyrus.pdf
- YuShan2022 TDP (local grid around formation point for off-ball positioning):
  https://tdp.robocup.org/wp-content/uploads/tdp/robocup/2022/robocupsoccer-simulation-2d/yushan2022-339/robocup-2022-robocupsoccer-simulation-2d-yushan2022e8oPEuwKCV.pdf


5. CHANGE LOG
-------------
- 2026-10-22: file created; defending ideas (user's depth-1 opponent modelling +
  press/mark/fallback) recorded with refinements.
- 2026-10-22: agreed always-press, counter-attack runner, no offside trap
  (hold the line), economical "just fast enough" interception.
- 2026-10-22: attacking options laid out from RoboCup 2D papers (not decided).
- 2026-10-22: agent.py v1 written. Smoke-tested on a fake world (no simulator
  here): no crashes in 3000 random situations; on_ball ~2 ms, off_ball
  ~0.4 ms average (well inside budget); all 4 players compute the same plan
  (1 presser/chaser, 1 supporter, 1 runner, no duplicate marks).
  NOT yet verified in real matches: reach-model accuracy, kick delays,
  offside on through passes, tackle/foul behaviour. All tunables are at the
  top of agent.py.
- 2026-10-22: v2 after first real matches (user feedback):
  * Defence: everyone behaved the same. Cause: danger threshold (0.45)
    filtered out most opponents in midfield -> all players "zone" and the
    zone clamp lined them up behind the ball. Fix: explicit roles for every
    player, identical on all teammates: PRESS (fastest to ball), MARK
    (strict one-to-one, every relevant opponent, most dangerous first,
    min-total-distance assignment), COVER (spare player between ball and our
    goal), ZONE (any further spare). Only opponents >15 m behind the ball are
    left unmarked. Removed the starter fallback for small time allowances
    (only when the match budget is almost gone).
  * Attack: nobody dribbled. Cause: only passes got depth 2 (follow-up bonus).
    Fix: depth 2 for dribbles too; PASS_COST 0.15 per pass (execution risk);
    long carries (12/16 m) when nobody within 6 cycles.
  * Hold no longer gets shooting-position credit (it never creates a shot);
    GAMMA_HOLD 0.9 -> 0.75.
  * Off-ball: if the holder has no pass above 0.6, attackers "push": sprint,
    open-spot search centred ahead of the ball. Sprint whenever the target is
    >5 m ahead.
- 2026-10-22: ROOT CAUSE of "nothing changed" in v1/v2: agent.memory in the
  real framework is a Memory object, NOT a dict (no [] / .get) -> every
  off-ball/on-ball decision crashed and fell back to the starter. v2.2 keeps
  per-player state in self.mem (self persists per player). Fake test world
  now uses a strict memory object to catch this.
- 2026-10-22: v3 after first real run (lost 0-4 to FULL; defending much better):
  * Tackling (1 tackle/match): presser stood goal-side facing our goal, ball
    behind him -> tackle_probability ~0; foul limit 0.15 too strict.
    Fix: within 4 m go straight at the ball (face it, turn if ball beside/
    behind); tackle at p >= 0.6; foul limits by zone (0.3 mid/their third,
    0.15 own third, 0.05 own box); other defenders tackle at p >= 0.8.
  * Ping-pong passing: depth-2 beam (10) was always filled by the ~100 pass
    variants -> dribbles never got the follow-up bonus; weights too
    risk-averse. Fix: separate depth-2 quotas (10 passes + 8 dribbles,
    interleaved); dribbles evaluated before passes; W_PROG 1.0->1.5,
    W_CLEAR 0.8->0.4, L_BASE 0.5->0.35, L_K 2.0->1.6, DRIB_M0 1.5->1.0,
    PASS_COST 0.2, BACK_PASS_PEN 0.3 for backward passes when not pressed;
    GAMMA_HOLD 0.6.
- 2026-10-22: v4 (user: marking too aggressive, press from the front, slow /
  uncoordinated attack, try Monte Carlo, fewer options):
  DEFENCE  front presser = goal-side player arriving within 10 cycles (else
    fastest, +4 cycles if from behind); "sandwich" back presser if a player
    behind the carrier is within 3 cycles (strict foul limit); MARK only
    opponents goal-side of the ball (+3 m); spare players SCREEN on the
    ball->near-post and ball->far-post lines (8 m from ball), then COVER,
    then ZONE. Assignment by min total distance, hysteresis.
  ATTACK   ~10 options: 1 pass per onside mate to the freest point within
    2.5 m of him (space + clear lane), 1 through ball for the most advanced,
    dribbles goal / away-from-nearest / left / right / back (5 m, +10 m
    towards goal/away when open), best shot, hold.
    Depth-3 deterministic chains (root + 2 best follow-ups) = prior.
    Monte Carlo evaluation: noisy rollouts (opponent position/speed noise,
    kick angle/speed noise, margin noise) chosen by UCB1, prior worth 3
    samples, up to 400 rollouts / 80 ms. Rollouts stop early when the
    expected (deterministic) continuation is worth less than keeping the
    ball; terminal shot/hold use expected values (less variance).
    Teammates' predicted choice (depth-1 best) kept unless MC beats it by
    0.15 -> coordination.
    COMMITMENT: keep a pass 3 cycles / a dribble 8 cycles while its success
    stays >= 0.5 (re-deciding restarted multi-touch kicks = "slow").
    Under pressure (opponent <= 1 cycle) passes use kick_one_step.
  OFF-BALL  every off-ball player predicts the holder's choice; the
    predicted receiver runs to the pass target early ("att_receive"), the
    others leave him out of the role assignment.
- 2026-10-22: v5 after v4 match (lost 0-6; holder chose HOLD in ~200 of ~250
  fresh decisions):
  DEFENCE  tackle thresholds: presser p>=0.45, others p>=0.7; foul limits
    0.5 mid/their third, 0.25 own third, 0.08 own box. Sticky roles:
    presser kept unless new one better by 2 cycles; previously marked
    opponents +0.3 danger; assignment hysteresis 6 m. Marker now stands ON
    the line opponent->ball, 2 m in front of his man (slightly goal-side).
  ATTACK   dribbles at fixed angles from the ball->goal line (0, +-25,
    +-50, +-80, back) + away from nearest opponent; 2 passes per teammate
    (freest point near him + to feet); freest point leans forward.
    Edge penalty in H: within 4 m of touchline / goal line (not the goal
    mouth) up to -0.6. Hold: -0.25 per consecutive hold, max 2 in a row.
    Chase: always sprint (intercept) for a moving ball or after our own
    pass; slow economical run only for a nearly still ball 6+ cycles ahead.
- 2026-10-22: v6 after v5 match (lost 0-5, "playing much better"):
  * Too safe / sideways: 5 m of progress was worth only ~0.07 vs ~1 for
    losing the ball. W_PROG 1.5 -> 3.5 (5 m forward ~ +0.17, 10 m back
    pass ~ -0.33).
  * Markers too far in front of their man: MARK_GAP 2 -> 1 m (and at most
    15 % of the distance to the ball).
  * Attackers not getting open: new _get_open every cycle for supporter /
    runner / open roles: points around HIMSELF (3 m, 6 m rings) and around
    his role spot (4 m ring); score = P(pass from holder) * H(point)
    + 0.5 * space - 0.3 per 10 m from role spot - spacing penalty;
    onside; hysteresis 0.1.
- 2026-10-22: v7 after v6 match (lost 0-3; ZERO shots in the whole match):
  * Shot model was far too pessimistic (16 m central = 6 %, 22 m = 0 %):
    goalie reach 1.8 m + "same cycle = saved". Now for shots: goalie reach
    1.3 m (real catch area ~1.2), +1 cycle reaction for defenders/goalie,
    SHOT_M0 0 (arriving together = 50/50). Result: 12 m ~93 %, 16 m
    45-65 %, 22 m+ ~0 % vs a goalie on his line.
    GOAL_VALUE 8 -> 12, SHOOT_NOW_P 0.6 -> 0.4, SHOT_MISS -0.1,
    W_SHOTQ 1.5 -> 2.5 (pulls the ball towards shooting positions).
  * More forward: W_PROG 3.5 -> 5.0 (5 m ~ +0.24); BACK_PASS_PEN 0.5, now
    also for backward dribbles when not pressed.
  * Marking very tight: 0.5 m in front of the man on the ball line (max
    10 % of the distance to the ball), lead 1 cycle, arrive threshold 0.4 m,
    sprint if > 3 m away; marked opponents stay marked until 4 m further
    behind the ball than the normal rule (no flip-flop).
- 2026-10-23: v8 after v7 match + screenshot (3 of ours on the ball in our
  corner while their men in our box were free):
  * "Goal-side" now = closer to OUR GOAL than the ball/carrier (distance to
    goal), not "smaller x": in the corners x said nobody was goal-side, so
    nobody got marked and spare players screened next to the ball.
    Anyone within 25 m of our goal is always markable.
  * Role memory removed (presser / marking / attack roles): players kept
    private memories that drifted apart -> two players with the same job.
    Stickiness now from what everyone can SEE: cost uses position + 3 cycles
    of velocity; a player already running at the ball gets -2 cycles for the
    press job; "already marked" = one of ours within 2.5 m of him.
  * Dribble steal: ball leaves the dribbler's feet -> others thought it was
    loose and the fastest ran at it. Possession now continues for 5 cycles
    while the dribbler is within 4 m and reaches it before any opponent; the
    dribbler himself goes back for it (chase_own). With an opponent close,
    a teammate may still help.
  * Tackling: any defender close enough tackles at p >= 0.5 (zone foul
    limits); back presser uses zone limits; front presser always rushes in
    (1 m goal-side of the carrier, no "delaying" distance).
  * Markers block instead of trailing: 1.2 m from the man, direction
    60 % towards ball / 40 % towards our goal, anticipating 3 cycles of his
    run.
  * Attack: W_PROG 5 -> 6.5; new W_CENTER 0.8 (middle strip); get-open adds
    goal-ward bonus and a penalty for being in the same direction (within
    30 deg, seen from the holder) as another teammate.
- 2026-10-23: v9 (screenshot: markers on top of / behind their men; still
  almost no tackles):
  * Marking: ON the line man -> ball, 2-3 m in front of the man (20 % of his
    distance to the ball, clamped 2..3 m, never more than half way), shaded
    20 % towards our goal; anticipate only 1 cycle (3 cycles made the target
    jump when he turned -> our marker always a step late = trailing).
    Sprint whenever > 1.5 m from the mark point.
  * Tackling: presser and anyone near the ball tackle at p >= 0.3; foul
    limits: none outside our third, 0.6 own third, 0.2 own box.
    Diagnostic label tk_no_pX_fY when within 2.5 m but not tackling
    (X = tackle prob x10, Y = foul prob x10).
- 2026-10-23: v10 (v9 diagnostic: tk_no_p0_f0 x44 = close but NOT facing the
  ball; tk_no_p7..9_f9 = good chance but foul limit in own third):
  * Within 2.5 m of their ball: anyone not facing it turns to it first (then
    tackles); presser turns if > 35 deg off. Foul limit only in our box
    (0.5); none elsewhere (tackling the carrier is ~always "foul likely").
  * User's weights: W_PROG 10, W_CENTER 1.6 -> H_MAX ~17, so GOAL_VALUE
    12 -> 30 (a goal must beat any position), SHOT_MISS -0.5,
    SHOOT_NOW_P 0.25, SHOT_RANGE 35, W_SHOTQ 4.
  * Hold cost is now a fixed HOLD_COST 0.6 instead of x0.6 of the position
    value (with big W_PROG the old rule made even a 20 m back pass beat
    holding for a cycle).
  * Quick tuning guide added at the top of the constants in agent.py.
- 2026-10-23: FULL-MATCH ANALYSIS of 2026-10-23_02-28-35 (lost 0-5, we = side r),
  from the .rcg log + re-running agent.py on every cycle to recover labels:
  * 0 shots; ball in our half/midfield 94 % of our possession; reached
    x>20 once, never their box. 7 of 9 possession spells ended with them
    simply taking it, mostly around midfield.
  * Attack crowding: their back line sits at ~midfield -> runner/open spots
    clamped to the offside line next to the ball, and W_PROG=10 makes the
    furthest-forward spot best for everyone -> all 8 outfield players inside
    the centre circle, holder has no option ("_push" everywhere).
  * Goals 1,4,5: one opponent dribbles 40-50 m; 3-4 of ours run alongside /
    behind him at the same speed (~1 m/cycle) -> a chase from behind can
    never win. Even when ours are goal-side (goal 4: 4 of 4 goal-side at the
    turnover) nobody wins it: carrier shields the ball on his far side, our
    players are never facing it (tackle needs ball in front, back dist 0).
    Only 3 tackles all match; a tackle freezes the player 10 cycles.
  * Stamina: ours repeatedly at ~2450 (effort threshold 2400); theirs
    4000-8000. We sprint for almost everything.
  * Goal 3: our goalie's (automatic) kick intercepted at our box.
- 2026-10-23: v11 = fixes A-E from that analysis:
  A+C ATTACK SHAPE: roles while a teammate has the ball = ANCHOR (deep
    supporter + last defender: 15 m behind the ball, x <= 10, central, keeps a
    back-pass lane) > WING far > WING near (level with their line, |y| ~ 18).
    Get-open no longer uses H (W_PROG made the furthest spot best for all):
    score = 1.5*P(pass) + 0.5*space + goal-ward bonus - 0.6/10 m from role
    spot - spacing (12 m) - same-direction penalty. Pass-and-move only if
    >= 4 m of room before the offside line.
  B THROUGH BALLS: per onside teammate level with/ahead of the ball, a pass
    8 m into the space behind their line in front of him.
  D DRIBBLER: presser goes onto the carrier's line to our goal 2 m in front
    of him (ball + 2 cycles of its velocity), faces him, steps in when the
    ball is close, tackles at p >= 0.45 (a tackle freezes 10 cycles). If he
    is clearly (> 2 m) behind the carrier: recovery run to the first point on
    the carrier's path he can reach first (carrier speed estimated from the
    ball speed); if none, a point further along the path, never deeper than
    the edge of the box. Near-post screen replaced by a "path" point 8 m ahead
    of the carrier on his line to goal; any zone job for a player still
    behind the ball becomes a recovery run.
  E STAMINA: sprint (100) only when urgent AND stamina > 45 %; otherwise
    40-70 by distance; < 50 % stamina -> 40.
  Check on the recorded match states (v10 -> v11): distance between
  attackers' spots 15.8 -> 30.9 m; attackers' spots within 8 m of the ball
  32 % -> 6 %; anchor 12.5 m behind the ball; defenders chasing from behind
  11 % -> 6 %; full-sprint orders 90 % -> 50 %.
- 2026-10-23: v12 (user: wings go so far forward that no pass reaches them;
  at kick-offs the two wings run into their half and the other two pass to
  each other until they lose it). Measured on the recorded match: only 14 %
  of wing target spots were reachable by a pass (median 23 m ahead of ball).
  Fix: wing spot = MOST FORWARD point on his flank (from 25 m ahead back to
  6 m behind the ball, 4 m steps, |y| 18 then 10) where P(pass) >= 0.6
  (WING_PASS_MIN), >= 8 m from the holder, never past the offside line; if
  none, the most passable one. Wings no longer go past the offside line while
  our pass is in flight (they would be offside for the next pass).
  Result on the same states: reachable 14 % -> 50 % (kick-offs 13 % -> 59 %),
  wings median 4 m ahead of the ball, attackers still ~24 m apart.
- 2026-10-23: v13 (user screenshots: holder trapped at the touchline with no
  option; 4 of ours around their carrier with nobody behind):
  * "show" role: when the holder has no pass (best P < 0.6) the near-wing
    job becomes SHOW: one player sprints to the free spot ~9 m from the
    holder (12 directions; pass reachable, space, slightly forward).
  * More MC actions: dribbles at +-135 deg (back-diagonal) and, when an
    opponent is within 5 cycles, short 3 m escape dribbles in every direction.
    Trapped test: hold one cycle (team-mate comes short) -> 3 m escape
    backwards-diagonal -> commit.
  * Defence shape: near-post screen / far screen replaced by a SECOND LINE:
    after marks and the "path" player, two players DEEP_D=18 m in front of
    the ball either side (8 m) of its line to goal, then cover. If no job is
    12+ m goal-side of the ball, the last job becomes a central deep spot.
    Opponents within 4 m of the carrier are left to the presser (no extra
    marker on top of the ball). Back presser only with all 4 field players.
    Replay check: 3+ of ours within 8 m of their ball 16 % -> 12 %; players
    12+ m goal-side 1.30 -> 1.54 on average.
- 2026-10-23: v14 = user: "attack was better around v9, defence is better
  now": ATTACKING POSITIONS back to v9 (_attack, _get_open: supporter /
  runner / open roles, get-open with P*H, checking, pass-and-move;
  ROLE_W 0.3); anchor / wing-spot code (v11-v13) removed.
  Kept: all v13 DEFENCE; on-ball search with the newer actions (escape
  dribbles, +-135 deg dribbles, passes behind their line), v10 shot/hold
  settings, stamina management, "no running past the offside line while our
  pass is in flight" (v12 fix). "Show" kept inside v9: when the holder has no
  option, the v9 supporter goes to the free spot ~9 m from him.
  New: a CONTESTED ball (one of ours and one of theirs both at it) counts as
  defending, not attacking.
- 2026-10-23: v15 (user: no far players, close enough to coordinate):
  MAX_SUPPORT_DIST 16 m: no attacker's target further than this from the
  holder (final clamp + get-open ignores points beyond it); runner at most
  12 m ahead (RUN_AHEAD 20 -> 12) and within 10 m sideways of the holder
  (RUN_Y_SPREAD); open player's formation slot pulled in towards the holder.
  Replay: attackers' targets 90th percentile 19.2 -> 15.5 m from the ball,
  beyond 18 m 11 % -> 3 % (only receivers running onto a chosen pass);
  within 6 m unchanged (16 %).
- 2026-10-23: v16 = user: back to v13 for attack AND defence, just no far
  wing. Attack roles: ANCHOR (12 m behind the ball, was 15) + NEAR WING
  (|y| ~ 12, most forward passable spot, within 16 m of the holder) + close
  SUPPORT (free spot ~9 m from the holder; called "show" and given priority
  when the holder has no option). Every attacker clamped to 16 m from the
  holder. Contested ball = defend (kept from v14). v14/v15 (v9-style attack)
  dropped. Replay vs v13: attackers' targets median 14.9 -> ~12 m from the
  ball, beyond 18 m 28 % -> ~10 % or less; defence identical to v13.
- 2026-10-23: v17 on top of the USER'S PARAMETERS (W_PROG 15, W_CLOSE 1.4,
  W_SHOTQ 6, GOAL_VALUE 40, SHOOT_NOW_P 0.20, DRIB_M0/K 0.2/1.5,
  BACK_PASS_PEN 1.0, ANCHOR_BEHIND 14, ANCHOR_MAX_X 12, WING_Y 15):
  * Distance limit + REACH limit: MAX_SUPPORT_DIST 20 -> 30, but a pass
    from the holder must reach the spot with P >= REACH_P (0.55); otherwise
    the spot slides towards the holder in 2 m steps (never closer than 6 m),
    or the most reachable point on that line. Replay: reachable spots
    72 % -> 76 %, 90th percentile distance 16 -> 19 m.
  * BREAKAWAY: no opponent field player within 14 m in a 50 deg cone towards
    their goal, nobody within 4 m, nearest opponent >= 6 cycles away ->
    holder carries the ball forward (best of 0/+-25 deg dribbles with
    p >= 0.8; after the shot check), teammates position as if he were 8 m
    further on and sprint there (labels breakaway_dribble, att_*_brk).
- 2026-10-23: v18: breakaway only if no teammate is clearly better placed:
  a very safe pass (p >= GREEDY_P 0.85) at least 8 m further forward
  (BRK_PASS_AHEAD) worth at least 0.3 more than the best breakaway dribble
  (BRK_PASS_GAP) wins instead. Teammates apply the same test through their
  prediction of the holder's choice, so they don't push up for a dribble
  that won't happen.
- 2026-10-23: v19 on top of the user's v18 file (his params: DRIB_M0 0.0,
  W_CENTER 1.2, HOLD_COST 0.3, GREEDY_P 0.70, TACKLE_MIN 0.30, REACH_P 0.30,
  REACH_MIN_D 12, BRK_CLEAR 10, BRK_CONE 40, OPEN_W 1.0, ANGLE_SPREAD 40).
  Replay of his version: HOLD was the most common on-ball choice (64/190);
  at those moments the best forward dribble had p ~0.05 in our model.
  * Dribble model too pessimistic: DRIB_SPEED 0.7 -> 0.85 (log shows ~1.0),
    dribble-only opponent reach DRIB_TACKLE_R 1.1 (was TACKLE_R 1.4) ->
    forward dribble p median 0.05 -> 0.50 at those moments; holds 64 -> 46,
    fresh dribbles 7 -> ~28.
  * DRIB_FWD_W 1.0: extra value per 10 m a dribble gains (also in rollouts).
  * Off-ball: with options, 3rd attacker = FORWARD (central channel / other
    side of the holder, most forward spot a pass reaches) instead of the
    close support; "show" still when the holder has no option.
    GOALWARD_W 0.6 -> 1.5. Get-open only counts spots a pass reaches with
    P >= OPEN_REACH_P 0.6 (others -3.0, fallback only) - before, get-open
    moved the forward from a 77 % spot to an 8 % one. Pass-and-move only
    if the run spot is reachable. Breakaway positions at most BRK_AHEAD 12 m
    beyond the projected holder.
  Replay: cycles with 0 reachable teammates 3+ m ahead 55 % -> 50 %, with 2
  15 % -> 18 %. Note: pressed holder (opponent 1.5 m) now picks a 3 m
  escape dribble instead of the safe back pass.
- 2026-10-23: ANALYSIS of the match where we scored (2026-10-23_02-16-20,
  1-5, played with v9): possession 60 % (vs 45 %), 38 % of our possession in
  their half (vs 6 %), 7 interceptions (vs 14), 21 dribble/hold kicks kept
  (vs 3), 2 shots. Same spacing as the 0-5 match (~9.8 m to the ball), so the
  difference was CARRYING the ball, not positions. The goal: 230-cycle
  possession from midfield, #2 dribbles ~15 m up the left, #3 carries into
  their half and down the right wing, cut-back #4 -> #2 -> #5, shot from
  ~19 m angled. Replaying decisions on that match: v9 held 6 % of on-ball
  decisions, the user's v18-based file 28 %.
  Cause: v9 valued HOLD at 60 % of the position value (expensive), v10
  changed it to a small fixed cost (0.6, user 0.3) -> with W_PROG 15 holding
  became almost free.
- 2026-10-23: v20: hold = p * (h * (1 - HOLD_FRAC) - HOLD_COST) + (1-p)*Loss
  with HOLD_FRAC 0.3 (also in rollouts and the stop-here check);
  DRIB_SPEED 0.8, DRIB_TACKLE_R 1.2, DRIB_FWD_W 0.5. Replay mix on the goal
  match: pass 37 / dribble 60 / hold 1 % (user file 34 / 35 / 28); on the
  0-5 match: 59 / 38 / 3 % (user file 49 / 17 / 33).
- 2026-10-23: v21 ANTI PING-PONG. Logs of both matches: ~30 A->B->A passes
  within 40 cycles per match, median pass gains 0 m, ~90 % gain < 5 m.
  Rejected: hard pass-rate limit (ignores the situation, can't be shared
  between players), novelty bonus (rewards change, not progress).
  Chosen: passes must EARN their cost when not pressed (opponent > 2
  cycles): NOGAIN_PEN 1.2 scaled by how far short of PASS_MIN_GAIN 5 m the
  pass falls; RETURN_PEN 2.5 for passing straight back to whoever passed to
  me within 30 cycles unless it gains >= 8 m (one-two); backward penalty as
  before. Same penalties applied to the receiver's follow-up in the
  look-ahead (else the search still plans pass + return pass). "Who passed
  to me" = the previous holder in the player's own possession memory.
  Replay on both matches (real states): unpressed passes gaining < 5 m
  77 -> 29, straight-back passes 60 -> 21; mix pass/dribble/hold
  62/30/4 % -> 34/48/13 % (holds still max 2 in a row).
- 2026-10-25: ANALYSIS vs claude_light (20261024-222919_t0000.rcg, we were
  left, lost 1-3). Possession 61 % for us, 95 completed passes, but ONE shot.
  Their 4 long possessions all reached our box (max x +42, +30, +41, +45 in
  their frame); ours reached their half twice (one solo breakaway = our goal).
  Findings, each from the replay:
  * Kick-off retreat: 4 of our kick-offs went BACK (x 0 -> -24, -32, -10,
    -21). Their 3-man press stood 5-7 m in front; our wing/forward spots
    fell back to "best reachable" points level with the ball, so the holder
    had only lateral/back options and dribbled back one "safe" step at a
    time (p 0.97 each) until trapped at our box -> goal 2.
  * Tackled in the wind-up: 9 tackles by them vs 2 by us. Before 4 of them
    our holder had chosen a pass and spent 2-4 cycles winding up a
    multi-kick (commit_pass p 0.77 -> 0.5) while the opponent closed in.
  * Dribbles: ours 0.51 m/cycle with NEGATIVE forward progress; theirs
    0.74 m/cycle, 0.55 forward (they kick the ball 8-14 m ahead and run).
  * Goal 1: their striker was already behind our anchor when we lost the
    ball; ran 45 m, our anchor chased from 10-12 m behind all the way.
  * Goal 3: carrier ran 30 m down the touchline at ~0.85 m/cycle; our
    presser blocked the straight line to goal and ran parallel 10 m inside.
    Meanwhile our marker trailed their runner by ~5 m (mark point used his
    REPORTED velocity, which is after the 0.4 decay: a sprinter reports
    ~0.4 but moves 1.0 m/cycle). Three recovering players all ran to the
    same spot near our goal; the runner got the cross free and scored.
  From claude_light's code: single-kick passes / steal check for wind-ups,
  self-pass dribbles (kick to a point and run), passes to points around
  each receiver, back player goal-side of their most advanced forward.
- 2026-10-25: v22.
  Defence: markers aim at their man's position MARK_LEAD 3 cycles ahead
  using his real running speed (reported velocity x RUN_GAIN 2.5, capped
  1.05) and shade goal-side more when he runs at our goal / is near it.
  Presser uses the carrier's real direction (ball velocity while dribbling,
  else his running velocity; CARRIER_HEAD_W 0.6 blend with "to goal");
  goal-side but > 6 m away -> "press_cut" = earliest point on that path we
  reach first (instead of chasing a block point that moves with him).
  Recovering zone players run to their OWN job spot (only the "path" job
  still races to the carrier's path) -> no more 3 players on one spot.
  Anchor stays ANCHOR_GUARD 3 m goal-side of their most advanced field
  player (at most ANCHOR_MAX_BEHIND 28 m behind the ball).
  Attack: wing never below ball x + WING_MIN_AHEAD 4, forward never below
  ball x + FWD_MIN_AHEAD 8 (also in get-open); when no spot is reachable
  the one closest to the ball is used. Lead passes: 5 and 9 m ahead of each
  onside teammate at 0 / +-40 deg from his line to goal. Self-passes
  (SELFPASS): kick 8 or 13 m ahead at 0 / +-25 / +-50 deg, only if no
  opponent within 3 cycles; scored like a pass to myself (interception
  margin + defenders' margin at the point when I arrive); the kicker runs
  onto it ("selfpass_run"). Passes when an opponent is within
  ONE_STEP_PRESS 3 cycles are single kicks (evaluated at that speed; a
  committed pass switches to one kick when pressed). Backward dribbles cost
  RETREAT_PEN 0.8 (scaled by distance and depth) in our half even under
  pressure, +3.0 into our own third. DRIB_TACKLE_R 1.2 -> 1.5 (tackle_dist
  is 2 m).
  Replay check (static states of that match): kick-off spots of the two
  forwards move from (-6,-10)/(-3,14) to (13,-6)/(13,17); the four tackled
  situations become single-kick passes; consistency test still 0 problems;
  on-ball mean 9 ms, max ~110 ms.
- 2026-10-25: ANALYSIS v22 vs FULL (2026-10-24_23-46-05, lost 0-3, 17 % of
  the ball). Defence judged good by the user (keep). All 3 of our kick-offs
  in play were lost within 10 cycles, identically: kick-off to #4, #4 plays
  a LEAD pass "to #5" into the space at (4..7, -2..-5), where their #5 was
  standing; the ball rolled past our #5 and their #5 took it.
  Cause: kick_eval in "pass" mode ends as soon as the receiver can touch
  the ball ANYWHERE on its path. The lead pass passed 1.9 m from our #5, so
  it counted as received early, opponents beyond that point were never
  checked -> p 0.97. In reality the receiver ran to the target and the
  ball went on to the opponent.
- 2026-10-25: v23. Run-onto passes (lead / through / behind) are judged by
  _eval_runpass: interception margin along the whole path to the TARGET
  ("virtual" mode), then the defenders' margin at the target at the moment
  the receiver gets there (ball waits/rolls if he is late). Same test in
  the Monte Carlo samples and when continuing a committed pass. Lead
  passes are timed to arrive when the receiver does (min 1.2 m/cycle).
  Replay on the FULL match: lead passes chosen 17 -> 2 times; at the
  kick-off #4 now plays to #5's feet (p 0.97) instead. Nothing else changed
  (defence identical to v22).
- 2026-10-25: v24 SAFER ON THE BALL (user: v23 "very uncautious, unnecessary
  forward kicks, high risks - keep the ball like the older versions").
  No new log; measured on the two previous logs by re-deciding every
  on-ball state (model success probability p of the chosen action).
  * KEEP_BALL 1.0 added to every loss value: losing the ball now costs
    1.35..2.95 instead of 0.35..1.95 -> risk is priced in everywhere
    (depth 1, look-ahead, Monte Carlo, hold). Clear threshold moved with it.
  * Risk filter: if any pass/dribble/hold has p >= SAFE_P 0.85, passes and
    dribbles with p < RISKY_P 0.6 are dropped (shots never). Same filter in
    the teammates' prediction of the holder.
  * Monte Carlo may only overrule the safest-good (depth-1) choice with a
    riskier one (p lower by > 0.1) if it is worth MC_RISK_TIE 0.5 more.
  * GREEDY_P 0.70 -> 0.80; self-pass only with p >= 0.9 and nobody within
    5 cycles (was 3); RETREAT_PEN 0.8 -> 0.5; supports closer:
    WING_MIN_AHEAD 4 -> 2, FWD_MIN_AHEAD 8 -> 5.
  Replay (chosen actions): claude_light match mean p 0.81 -> 0.87, actions
  with p < 0.6 18 % -> 8 %; FULL match mean p 0.89 -> 0.91, p < 0.6 3 -> 0 %.
  Defence untouched.