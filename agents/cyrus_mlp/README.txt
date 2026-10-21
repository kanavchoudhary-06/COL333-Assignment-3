AGENT: cyrus_mlp  (agents/cyrus_mlp/agent.py, class Agent1)
Core idea: CYRUS-style learned evaluation. A small feedforward network scores every kick, and the same network predicts teammates' passes for unmarking.
Network: 8 engineered inputs, 6 tanh hidden units, 1 sigmoid output (AIMA ch.21 / Goodfellow ch.6), written in pure Python.
Inputs: time margin, pass length, progress, target's distance to goal, free space at the target, opponents in a 15-degree lane cone, target x, lead/dribble flag.
Weights: hand-initialised so each hidden unit encodes one feature group (safety, goal closeness, space, crowding, progress, length). They are meant to be refit with backpropagation and mini-batch SGD on logged pass outcomes.
On-ball decision: one-step argmax over network-scored passes and dribbles, shots, hold and clear. A pass started last cycle is kept unless beaten by 0.05.
Shooting: 5 goal-mouth targets. P = logistic(1.2 x (margin - 1)) x distance factor, scored 0.3 + P so a good shot outranks passes.
Passing: direct pass and two 5 m lead passes per onside, non-goalie teammate. The margin comes from a relaxed lower bound on opponent reach time (AIMA 3.6).
Dribble/hold/clear: 8 dribble headings (5 m) through the same network with the dribble flag. Hold is worth 0.35 when no opponent is within 4 m, and clear is the low-value fallback.
Loose ball: shared (reach steps, unum) ranking. Rank 0 intercepts.
Pressing/tackling: rank 0 presses, rank 1 blocks if it isn't marking. Tackle at probability >= 0.75, no deliberate fouls.
Marking (CYRUS OMAM): an assignment problem. Each defender keeps its top-3 goal-side mark points. Backtracking search with MRV ordering maximises dangerous attackers covered, then minimises distance.
Attack support (CYRUS2022 unmarking): the network predicts the holder's pass, and the predicted receiver's next pass, in a best-first predicted-pass tree of depth 2. This finds my likely passer. I move to the reachable onside point (16 candidates) the network rates highest from that passer.
Coordination: rankings, predictions and assignments are deterministic functions of the shared full-state world model. No messages are sent.
Time budget: on-ball is about 60 network evaluations (well under 10 ms). Unmarking is cached for 3 cycles and marking for 1 cycle. Off-ball falls back to the formation slot when CPU runs low.
Gaps filled by assumption: CYRUS trained deep networks on large logged datasets. Here the network is small, its weights are hand-set until refit offline, and the 738-feature set is reduced to 8 features.
Papers used:
- N. Zare, A. Sayareh, M. Sarvmaili, O. Amini, A. Soares, S. Matwin, "CYRUS Soccer Simulation 2D Team Description Paper 2021", arXiv:2206.02310.
- N. Zare, A. Firouzkouhi, O. Amini, M. Sarvmaili, A. Sayareh, S. Ramezani Rad, S. Matwin, A. Soares, "CYRUS Soccer Simulation 2D Team Description Paper 2022", arXiv:2205.10953.
- N. Zare et al., "Engineering Features to Improve Pass Prediction in Soccer Simulation 2D Games", arXiv:2401.03410.
- N. Zare et al., "Improving Dribbling, Passing, and Marking Actions in Soccer Simulation 2D Games Using Machine Learning", arXiv:2401.03406.
