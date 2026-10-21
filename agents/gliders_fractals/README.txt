AGENT: gliders_fractals  (agents/gliders_fractals/agent.py, class Agent1)
Core idea: Gliders2d's incremental tactics (Voronoi evaluator, risk level, pressing level, stamina rules, Voronoi positioning) plus Fractals2019 Dynamic Constraint Annealing to assign players to positions.
On-ball decision: one-step choice of the highest-valued feasible action. Value = territory + max(0, 40 - distance to goal), plus a Voronoi term when >= 2 opponents are within 10 m.
Voronoi term (Gliders2d v1.3): every 3 m cell is owned by the nearest player of either team (a discrete Voronoi diagram by nearest-site assignment). Targets near a teammate and away from opponents score higher.
Shooting: 5 goal-mouth targets. Shoot at once if any target's opponent/goalie margin is >= 0.
Passing: direct passes, plus passes to the centroid of the receiver's own Voronoi region (within 15 m of him).
Risk level (Gliders2d v1.6): a pass may have a negative time margin of up to 0 / 0.5 / 1.5 cycles in the defensive / middle / attacking third.
Dribble/hold/clear: 8 dribble headings of 6 m with a margin >= 1 cycle. Hold if nothing is feasible and no opponent is within 3 m, else clear.
Loose ball: shared (reach steps, unum) ranking. Rank 0 intercepts.
Pressing (Gliders2d v1.2): rank 1 also presses the holder when its reach time is within PRESS[zone] = 5/3/4 cycles of the opponent's.
Tackling: tackle probability >= 0.75 and ball within 2 m, never a deliberate foul.
Stamina (Gliders2d v1.1): full dash for the chaser, for defenders when the ball is in our third, and for forwards near the offside line. Otherwise the starter's stamina rule.
Attack positioning (Gliders2d v1.4): candidates are local maxima of the free-space field on a 5 m grid (largest empty circles = Voronoi vertices of the opponents), points on the offside line, and each player's current spot.
Defence positioning: candidates are goal-side marking points for each attacker (weighted by danger), the ball-to-goal cover point, and current spots.
Assignment (Fractals2019 DCA): a greedy start, hill climbing on moves and swaps (phase 1), then simulated annealing (T 1.0 to 0.01, 250 steps, phase 2). Spacing >= 8 m, onside, and <= 18 m of movement are penalty constraints.
Coordination: the plan is recomputed on shared 4-cycle blocks from identical full-state inputs with an RNG seeded by the block. Every player derives the same assignment without messages.
Time budget: on-ball is a single pass over about 40 actions (about 5 ms with the Voronoi grid). Positioning is about 300 grid cells + 250 SA steps every 4 cycles. Off-ball falls back to the formation slot when the budget runs low.
Gaps filled by assumption: Gliders tuned these parameters with human-based evolution and PSO. Here they are exposed constants for a GA / hill-climbing tuner from the course. Teammates' formation slots are not visible, so current positions anchor the movement constraint.
Papers used:
- M. Prokopenko, P. Wang, "Gliders2d: Source Code Base for RoboCup 2D Soccer Simulation League", RoboCup 2019 Symposium, arXiv:1812.10202.
- M. Prokopenko, P. Wang, "Disruptive innovations in RoboCup 2D Soccer Simulation League: from Cyberoos'98 to Gliders2016", arXiv:1612.00947.
- M. Prokopenko, P. Wang, "Fractals2019: Combinatorial Optimisation with Dynamic Constraint Annealing", arXiv:1909.01788.
