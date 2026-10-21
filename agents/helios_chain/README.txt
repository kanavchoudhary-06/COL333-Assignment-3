AGENT: helios_chain  (agents/helios_chain/agent.py, class Agent1)
Core idea: HELIOS "action chain" planning. Search sequences of kicks between teammates and play the first kick of the best sequence.
On-ball search: best-first search (AIMA 3.5) over chains up to 3 actions, at most 400 nodes, stopped by the per-decision time budget (anytime).
Chain value: only the final state is scored, by a HELIOS-like field evaluator (ball x + max(0, 40 - distance to goal)). There are no success probabilities. This is the main contrast with the WrightEagle main agent.
Feasibility: strict check. Every opponent's relaxed reach time (AIMA 3.6 relaxed problem) must exceed the ball's arrival time by a buffer: 1.0 cycle for passes, 1.5 for dribbles, 0.5 for shots.
Shooting: if any of 5 goal-mouth targets passes the strict check at a node, that chain ends with a shot worth 1000 (minus depth). At the root the agent shoots at once.
Passing: direct passes plus through passes 5 m and 9 m ahead of each onside, non-goalie teammate. The receiver must arrive within 2 cycles of the ball. No immediate return pass.
Dribble/hold/clear: 8 dribble headings of 5 m. If no feasible chain exists: hold when no opponent is within 3 m, else clear.
Tactical pruning: HELIOS2017 trained an SVM to remove chains that break the intended tactics. Here a linear classifier (perceptron form) with hand-set weights prunes chains with backward passes in the attacking half, endings in our box, or lateral passes and losing dribbles near our goal.
Loose ball: shared (reach steps, unum) ranking. Rank 0 intercepts.
Pressing/tackling: rank 0 presses, rank 1 blocks the ball-to-goal line. Tackle at probability >= 0.75.
Defence marking: in our half, the remaining players greedily mark the nearest free attacker goal-side, in rank order. Every player computes the same assignment.
Off-ball attack: go to the formation slot (the framework's formation is HELIOS base's Delaunay triangulation model), with dash power set by stamina.
Coordination: deterministic shared ranking. No messages are sent.
Time budget: about 4x the fair CPU share for on-ball, capped at 250 ms and 300 ms below the 1 s wall cap. Clock checked every 8 expansions. Off-ball costs almost nothing.
Gaps filled by assumption: the evaluator, buffers and pruning weights are hand-set. The SVM pruning is replaced by a hand-set linear classifier that can be refit by perceptron learning on labelled chains.
Papers used:
- H. Akiyama, T. Nakashima, "HELIOS Base: An Open Source Package for the RoboCup Soccer 2D Simulation", RoboCup 2013 Symposium (LNCS 8371).
- H. Akiyama et al., "HELIOS2013 Team Description Paper", RoboCup 2013 (action-chain / cooperative tree-search planning).
- H. Akiyama, T. Nakashima, S. Tanaka, T. Fukushima, "HELIOS2017: Team Description Paper", RoboCup 2017 (SVM pruning of action sequences).
