AGENT: bayes_minimax  (agents/bayes_minimax/agent.py, class Agent1)
Core idea: learn who the opponent is during the match (Bayesian formation identification + action frequencies), and plan against their worst response with expectiminimax.
Formation identification (Fukushima et al. 2017): each cycle the opponent field players fill a 6x4 grid occupancy vector that ignores uniform numbers, smoothed with EMA 0.95. It is classified into wall / line / normal.
Formation filtering: HMM forward filtering (AIMA 14.2). The transition keeps the class with probability 0.98. The observation model is Gaussian naive Bayes against hand-built class prototypes (variance 0.02, tempered by 0.1).
Opponent behaviour model: every opponent kick is classified as a pass, dribble or shot from the ball's velocity change. P(action | zone) comes from MLE with Laplace smoothing (AIMA 20.2) per third of the pitch.
On-ball search: expectiminimax with alpha-beta (AIMA 5.2-5.5). Levels: our macro action (MAX), its success with P (CHANCE), the opponents' worst response (MIN: all drift, or one of the 3 nearest sprints to the ball), then our next action (MAX), then the evaluation function.
Search control: iterative deepening (depth 1, then 3), so it is anytime. Alpha-beta bounds are carried through chance nodes. Move ordering by P x H. 8 root actions and 6 inner ones are considered.
Evaluation/failure: H = logistic(goal distance, free space). Losing the ball costs -logistic(distance to our goal). If every line is negative in our third, clear.
Shooting: 3 goal-mouth targets. P = logistic(margin - 1) with a distance factor. The range grows to 32 m against a "wall".
Passing: direct and 5 m lead passes. Against a "line" defence, through balls 6 m behind the offside line are added, and wide passes get a bonus against a "wall".
Dribble/hold/clear: 8 headings of 5 m, with a small extra bonus against a "line". Hold or clear when nothing is generated. A started kick is re-issued for up to 2 cycles.
Loose ball: shared (reach steps, unum) ranking. Rank 0 intercepts.
Defence: rank 0 presses. If P(dribble | zone) > P(pass | zone), rank 1 doubles on the holder by blocking the goal line. The rest mark receivers greedily in order of danger (distance to our goal). Tackle at probability >= 0.75, no deliberate fouls.
Attack support: best of 9-10 points around the formation slot by free space, lane, progress and onside rule. Wide points are added against "wall", and points on the offside line (ready to run in behind) against "line".
Coordination: all players see the same full state, so their beliefs, counts and rankings stay identical without messages.
Time budget: about 4x the fair CPU share on-ball, capped at 250 ms and 300 ms below the wall cap. The clock is checked every 16 nodes and a timeout keeps the deepest completed result. The opponent model costs about 50 operations per cycle.
Gaps filled by assumption: the class prototypes and kick classifier are hand-built (the paper used EM-fitted GMM clusters plus NN/SVM/RF classifiers). Opponent responses are reduced to "who sprints".
Papers used:
- T. Fukushima, T. Nakashima, H. Akiyama, "Online Opponent Formation Identification Based on Position Information", RoboCup 2017 Symposium.
- N. Zare et al., "CYRUS Soccer Simulation 2D Team Description Paper 2021", arXiv:2206.02310 (predicting the ball holder's behaviour; here by naive-Bayes counts).
