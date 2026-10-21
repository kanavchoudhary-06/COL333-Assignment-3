AGENT: mcts_alice  (agents/mcts_alice/agent.py, class Agent1)
Core idea: Monte Carlo Tree Search for attacking decisions, plus a one-level MCTS (UCB1 bandit) for the defender facing the ball holder, after Alice2022.
On-ball search: UCT (AIMA 5.4). UCB1 selection (c = 0.3), untried arms first in prior order, expansion of the top 10 actions at the root and top 6 below, depth <= 3. Runs until the per-decision time budget ends. The most-visited root arm is played.
Simulation: each macro action's success is sampled from its probability (a chance node). A failure ends the sample with -logistic(distance to our goal), and a shot with 1 or -0.02. Otherwise an evaluation function H (logistic of goal distance and free space) replaces a long random playout. Discount 0.98 per cycle, averaged backup.
Shooting: Alice2022 used an SVM over sampled ball speeds. Here success at the root is a Monte Carlo estimate M/N: 12 sampled kick angles (+-0.05 rad) and speeds (2.5-3.0). M counts the shots that beat every defender and the goalie on the relaxed reach bound.
Passing: direct and 5 m lead passes. P = logistic(margin - 0.5), where the margin is a relaxed lower bound on opponent reach time along the ball path (AIMA 3.6).
Dribble/hold/clear: 8 dribble headings of 5 m. Hold (0.85 x H) and clear (-0.03) are always available at the root. A started pass or shot is re-issued for up to 2 more cycles while the kick completes.
Game phase: offence (we hold the ball), defence (they hold it) or transition (loose ball). Alice2022 learned this with Q-learning; here it is read directly from the full-state world model.
Loose ball: shared (reach steps, unum) ranking. Rank 0 intercepts.
Defence (Alice2022 MCTS for defensive outcomes): the presser chooses among intercept, block the goal line, mark the most dangerous receiver, and tackle (if probability >= 0.5) with 40 UCB1 samples.
Defence sampling: each sample draws the holder's response from a softmax over its options (passes to each receiver, a dribble at our goal), then its success from P. Threat = the opponent's scoring heuristic at the target. The arm with the lowest mean threat is played.
Other defenders: rank 1 blocks the ball-to-goal line, the rest hold formation. Tackle at probability >= 0.75 otherwise.
Attack support: the best of 9 points around the formation slot by free space, passing lane, forward bonus and onside rule.
Coordination: deterministic shared ranking and phase. The search RNG is seeded per player and cycle, and no messages are sent.
Time budget: about 4x the fair CPU share on-ball (capped at 250 ms, 300 ms below the wall cap). Each iteration checks the clock. Defence MCTS is 40 cheap samples over precomputed option tables.
Gaps filled by assumption: Alice's SVM and Q-learning parts are replaced by Monte Carlo estimation and direct phase rules. The opponent response model is a hand-set softmax.
Papers used:
- F. Wang, X. Lin, Y. Xiao, "Alice2022: Team Description Paper", RoboCup 2022 (MCTS for defensive actions, SVM shot decision with sampled ball speeds, Q-learning phase detection).
- C. B. Browne et al., "A Survey of Monte Carlo Tree Search Methods", IEEE Trans. Computational Intelligence and AI in Games 4(1), 2012 (UCT, UCB1 selection, evaluation-truncated simulation).
