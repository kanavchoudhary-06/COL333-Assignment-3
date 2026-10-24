AGENT: main submission v2  (src/student/agent.py, class Agent1; previous version kept in agents/main_v1)
Core idea: one planner, one value function. A* finds a dribble route, best-first search scores macro-action chains, MCTS re-checks the best root actions under sampled uncertainty.
Evaluation V(s): possession value from a 54-state abstract MDP (6 x-zones x 3 lanes x 3 pressure levels, 6 actions). It is solved by value iteration at start-up (AIMA 17.2) and interpolated over the pitch.
Expected goals: V adds xG = logistic(distance to goal, goal angle left open by opponents and goalie). Losing the ball is worth -(opponent's possession value at that spot): a zero-sum loss value.
Success probabilities: logistic of a time margin, i.e. a relaxed lower bound on each opponent's reach time minus the ball's arrival (AIMA 3.6). Dribbles are only charged for the cycles actually committed to.
Opponent model: a Beta-Bernoulli posterior per opponent on "chases when it should" (AIMA 20.2), with the same update on every player. It is OFF by default (P.OPP_MIN_FACTOR = 1.0) because the movement signal couldn't tell full from idle.
On-ball step 1, A* (AIMA 3.5): 2 m grid to the shooting zone, cost = dribble time + risk penalty (risk judged within a 10-cycle horizon), admissible distance heuristic. The first waypoint 7 m along the route becomes a candidate.
On-ball step 2, chains: greedy best-first search, depth 3, over shots, direct/lead/through passes, dribbles (4/8/12 m), advance (only when pressed), hold and clear. Chance-node expected utility; a root action is worth the best chain that starts with it.
On-ball step 3, MCTS: UCT (AIMA 5.4) over the 6 best root actions. Each simulation samples opponent reaction times and execution noise, then scores the best follow-up. The visit mean is blended with the chain value (prior weight 6 simulations).
Holding: only allowed when an opponent is within 2.5 m or nothing else scores above 0. Each repeated hold multiplies its value by 0.6.
Passing details: offside is judged on the receiver's position at the kick, so through balls behind the line are allowed. A kick in progress is kept unless beaten by 0.05; kick(accept_late=5).
Shooting: 5 goal-mouth targets within 32 m; kick_one_step under pressure, otherwise full-power kick.
Loose ball: shared (reach steps, unum) ranking. Rank 0 runs to a ranked trap point on the ball path (HELIOS2023), else intercepts; rank 1 covers.
Defence: rank 0 presses, tackles at probability >= 0.75 and never fouls on purpose. Others use OMAM-style marking (backtracking assignment, CYRUS) or block the ball-to-goal line.
Attack support: CYRUS2022 unmarking. The same planner predicts the holder's pass (depth-2 predicted-pass tree), then the player moves to the reachable onside spot with the best pass value.
Positioning otherwise: YuShan2022-style 7x7 grid around the formation slot, ranked with linear weights per mode (aggressive / conservative / defend).
Coordination: every role and prediction is a deterministic function of the shared full-state world model. No messages are sent.
Time: budget = min(300 ms, 5 x the fair CPU share, wall cap - 300 ms), with a 7.5 s reserve. A* <= 15%, chain search to 70%, MCTS the rest. Every stage is anytime. Any exception falls back to the starter.
Tested locally: idle 8-0, 8-0 and 7-0 (v1 won 5-0); still loses to full 0-4 / 0-5 (v1: 0-2 / 0-4). The weakness against full is losing the ball in our own third.
Full walkthrough: the "Main Agent v2 Walkthrough" doc. Tuning: eval/sweep.py with "P.NAME" overrides.
Papers used:
- H. Zhang, X. Chen, "The Decision-Making Framework of WrightEagle, the RoboCup 2013 Soccer Simulation 2D League Champion Team", RoboCup 2013, LNCS 8371.
- N. Zare et al., "CYRUS Soccer Simulation 2D Team Description Paper 2022", arXiv:2205.10953 (unmarking); "Improving Dribbling, Passing, and Marking Actions...", arXiv:2401.03406 (OMAM marking).
- Z. Cheng et al., "YuShan2022 Team Description Paper"; H. Akiyama et al., "HELIOS2023: Team Description Paper" (positioning grid; ranked ball trapping).
- F. Wang, X. Lin, Y. Xiao, "Alice2022: Team Description Paper" and C. B. Browne et al., "A Survey of Monte Carlo Tree Search Methods", IEEE TCIAIG 2012 (MCTS).
- M. Hausknecht et al., "Half Field Offense", AAMAS ALA 2016 (high-level actions behind the abstract MDP).
