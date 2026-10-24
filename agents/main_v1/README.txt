AGENT: main submission  (src/student/agent.py, class Agent1)
Core idea: WrightEagle-2013 chain planning with MAXQ cut, plus CYRUS / YuShan / HELIOS 2021-2023 ideas for off-ball play. Course methods only.
On-ball search: greedy best-first search (AIMA 3.5) over chains of macro actions (shoot, direct/lead pass at two paces, dribble 8x2, hold, clear), depth 2.
Chain value: expected discounted utility with chance nodes: sum of P-weighted losses + P_chain * gamma^C * H(final state). A root action is worth the max over all chain prefixes that start with it.
Success probability: P = logistic(time margin). The margin comes from a relaxed-problem lower bound on opponent reach time along the ball path (AIMA 3.6).
State value: H = logistic of distance to goal and free space. Losing the ball costs -logistic(distance to our goal).
Shooting: 5 targets across the goal mouth, scored by the goalie/defender margin and a distance penalty for kick noise.
Passing: direct and lead passes to non-goalie, onside teammates only. A kick in progress is kept unless a new option beats it by 0.05.
Dribble/hold/clear: hold has an opportunity cost. Clear is a neutral fallback when every chain is negative.
Loose ball: every player computes the same (reach steps, unum) ranking, and rank 0 chases. HELIOS2023-style trap point: the chaser ranks safe ball-path points with a linear scorer (x + safety - time) instead of LambdaMART.
Pressing/tackling: rank 0 presses. Tackle at probability >= 0.75, never a deliberate foul.
Marking (CYRUS OMAM): an assignment problem solved by backtracking search (CSP-style, MRV ordering). Each defender keeps its top-3 goal-side mark points. Maximise dangerous opponents covered, then minimise distance.
Cover: rank 1 blocks the ball-to-goal line when it is not marking.
Attack support (CYRUS2022 unmarking): predict the holder's pass with our own planner (it replaces their pass-prediction DNN). A depth-2 best-first predicted-pass tree finds who will pass to me. Then move to the reachable onside point (16 candidates) with the highest pass value.
Positioning without the ball (YuShan2022): a 7x7 grid (1.5 m) around the formation slot, ranked by hand-set linear weights for aggressive, conservative and defend modes (replaces their LambdaMART ranker).
Coordination: every role is a deterministic function of the shared full-state world model. No messages are sent.
Time budget: per-decision CPU share x4 for on-ball, capped at 250 ms and 300 ms below the wall cap. Anytime search. Unmarking is cached for 3 cycles, marking for 1 cycle. Cheap mode below 7.5 s of CPU left.
Gaps filled by assumption: every weight is a hand-set start value to be fitted offline (logistic regression / hill climbing, per the plan).
Papers used:
- H. Zhang, X. Chen, "The Decision-Making Framework of WrightEagle, the RoboCup 2013 Soccer Simulation 2D League Champion Team", RoboCup 2013: Robot World Cup XVII, LNCS 8371, Springer 2014.
- N. Zare et al., "CYRUS Soccer Simulation 2D Team Description Paper 2022", arXiv:2205.10953.
- N. Zare et al., "Improving Dribbling, Passing, and Marking Actions in Soccer Simulation 2D Games Using Machine Learning", arXiv:2401.03406 (OMAM marking).
- Z. Cheng, J. Guo, Y. Ren, Y. Tang, L. Rong, "YuShan2022 Team Description Paper for RoboCup2022".
- H. Akiyama, T. Nakashima, K. Hatakeyama, T. Fujikawa, A. Hishiki, "HELIOS2023: Team Description Paper", RoboCup 2023.
