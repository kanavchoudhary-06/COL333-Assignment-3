AGENT: mdp_qlearning  (agents/mdp_qlearning/agent.py, class Agent1)
Core idea: the on-ball problem is an abstract MDP with 54 states (ball x-zone 6 x lane 3 x pressure 3) and 6 high-level actions.
On-ball decision: a hand-written transition/reward model is solved by value iteration at start-up (AIMA 17.2). Q* seeds a Q table, and the agent acts epsilon-greedily on Q.
Learning: tabular Q-learning (AIMA 22.3). A macro action closes when a teammate, an opponent or the scoreboard shows its outcome. Reward +1 for a goal, -1 for conceding, -danger(zone) for losing the ball. Otherwise the target is gamma^k * max Q(s').
Exploration: epsilon falls from 0.05 to 0.01 over the 1500 cycles. Each player keeps its own table (separate processes).
Shooting: aim at the goal corner away from the goalie. kick_one_step under pressure, otherwise kick(arrive_in=1) for full pace.
Passing: "forward pass" goes to the furthest-upfield onside teammate whose time margin is >= 0. "Safe pass" goes to the teammate with the largest margin.
Margin: a relaxed problem (straight dash at top speed, no turning) gives a lower bound on when each opponent can reach each point on the ball path.
Dribble/hold/clear: dribble 6 m along the best of 5 headings toward goal. Hold shields the ball. Clear is the always-safe fallback.
Multi-touch kicks: a kick is re-issued for up to 3 cycles, so a half-finished kick is never replaced by a new decision.
Loose ball: every player ranks field players by (reach steps, uniform number). Only rank 0 intercepts, so no messages are needed.
Pressing/tackling: rank 0 presses the opponent holder. Any player tackles when tackle probability >= 0.75 and the ball is within 2 m, never with a deliberate foul.
Off-ball attack: a 5x5 grid MDP (4 m cells) around the formation slot. Reward = free space + clear passing lane + small forward bonus - offside - distance from home. Value iteration, then follow the policy.
Defence positioning: same grid MDP, with a reward for standing on the line between the ball and our goal.
Coordination: all choices are deterministic functions of the shared full-state world model.
Time budget: VI runs once at start-up (about 54 x 6 x 80 backups). The grid MDP is 25 states x 15 sweeps, cached for 2 cycles. Below 5 s of CPU left the agent just goes to its formation slot.
Gaps filled by assumption: the transition probabilities and danger values are hand-set priors that Q-learning then corrects.
Papers used:
- M. Hausknecht, P. Mupparaju, S. Subramanian, S. Kalyanakrishnan, P. Stone, "Half Field Offense: An Environment for Multiagent Learning and Ad Hoc Teamwork", AAMAS ALA Workshop, 2016.
- M. Hausknecht, P. Stone, "Deep Reinforcement Learning in Parameterized Action Space", ICLR 2016 (deep RL replaced by tabular Q-learning over parameterised macro actions).
- F. Wang, X. Lin, Y. Xiao, "Alice2022: Team Description Paper", RoboCup 2022 (Q-learning with intercept/pass/dribble style actions and area-based rewards).
