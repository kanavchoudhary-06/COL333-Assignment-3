AGENT: claude_planner  (agents/claude_planner/agent.py, class Agent1)
The same file is installed as src/student/agent.py on this branch. The full
write-up is src/student/README.md.

Core idea: a batched kinematic world model (ball flight + player reach time),
planned over with classic search. A budget controller paces each player to
about 70-80% of its 75 s CPU budget, and every search is anytime: it stops at
a CPU deadline and at a wall-clock deadline 300 ms inside the 1 s cap.
- On the ball: expectimax over shots, passes and dribbles. Monte Carlo
  sampling (up to 64 rollouts) estimates each option's success, plus a
  beam-search second ply for the receiver's follow-up shot.
- Defence: a blocker on the ball owner's dribble path. The rest solve a
  marking CSP (backtracking, MRV, forward checking, branch & bound), valued
  by each opponent's 2-ply simulated threat as a pass receiver.
- Support: hill climbing over spots scored by simulated passes, with a second
  ply for the best spots.

Run any version:
  python3 a3.py match A FULL --left agents/claude_planner/versions/v2.py:Agent1 --right full
  python3 a3.py match A B --left agents/claude_planner/versions/tourney_t3.py:VTired --right agents/claude_planner/agent.py:Agent1

Scores are always us-them vs FULL (the HELIOS reference) unless stated.
OLD = OLD_light_VTired.py, the light (~2% CPU) tournament winner. "CPU" is
the worst player's share of the match budget. Most later checks are 1-4
matches, so treat single results as noisy.

VERSIONS (versions/)
v1 was not saved (first draft: 2-1 in its first match, then 0-3, 2-2, 0-2).

Light planner, ~1-2% CPU:
v2.py   single chaser, blocker on the dribble path, marking CSP, role-anchor
        support. 6 matches W1 D2 L3, 7-10. A 9-match re-run went W0 D3 L6, 7-16.
v3.py   + multi-touch kick continuation, body-crossing kick check, back
        player stays goal-side of their forward. W0 D2 L4, 4-12.
v4.py   + cover jobs in the CSP, "last man" caution, smoother shot model.
        W0 D0 L6, 3-16.
v5.py   + formation fitted to the reference 5-a-side formation, role CSP,
        HELIOS-style chase rule, slack-based shot model. W0 D0 L9, 4-25.
v6.py   + follow-the-kick (had a bug), tiered stamina, loose-ball tackles.
        Stopped after 1-5, 0-4, 0-2.
v7.py   + goalie fitted to observed shots, new value function, job-CSP
        defence. W0 D0 L7, 0-26 (still had the follow bug).
v8.py   follow bug fixed. W0 D1 L7, 0-22.

Tournament files (the v8 code with design switches; each file ends with the
variant classes that were played):
tourney_t1.py  VBase VJobs VTired VPhiV2 VNoFollow VNoRefine VJobsT. Stage 1:
               2 vs STARTER and 2 vs FULL each. Every variant beat STARTER.
tourney_t2.py  adds VTiredPhi VJobsTPhi. Stage 2 round robin.
tourney_t3.py  VTired vs VJobs head-to-head, and vs FULL. Winner: VTired.
OLD_light_VTired.py  = tourney_t3.py, class VTired (OLD). vs peers W8 D1 L5
               11-7; vs FULL W0 D1 L4 3-16; vs STARTER 11-0.
tourney_t4.py  HOn / HSup / HThr: each heavy component alone vs OLD.
               HOn L,L 1-3; HSup L,W 1-1; HThr W,L 2-2.
tourney_t5.py, tourney_t6.py  L06 / L075 (loss weight 0.6 / 0.75).
               Each W1 D1 L2, 3-4 vs OLD; vs FULL 0-4 / 1-4.

Budgeted planner, ~40-80% CPU:
new1.py  budget controller, generic options() evaluator with a full 2-ply on
         the ball, simulated support spots, minimax defensive spots.
         vs OLD 2-0, 1-2; vs FULL 0-5. 72-75% CPU.
new2.py  denser root candidates, bigger on-ball share. vs FULL 0-6,
         vs OLD 0-1. 73-77%.
new3.py  SIM_SUPPORT / MINIMAX_DEF switches, test classes NoMinimax and
         OnBallOnly. NoMinimax 1-0 OLD (58%); OnBallOnly 1-2 OLD (7%).
new4.py  threat-valued marking CSP, minimax off. vs OLD 2-1, vs FULL 2-4.
         39-47%.
new5.py  threat search widened to the top 16 passes. vs FULL 0-4,
         vs OLD 1-1. 48-62%.
new6.py  top 30 passes. vs FULL 1-2, vs OLD 0-1. 57-58%.
new7.py  blocking along the owner's planner-predicted dribble (reverted).
         vs FULL 0-6, vs OLD 0-5. 66-69%.
new8.py  new6 + larger support second-ply cap. 4 vs OLD W0 D2 L2, 3-6. 67-76%.
new9.py  Monte Carlo sampled on-ball evaluation, shot-only second ply.
         4 vs OLD W0 D2 L2, 2-4. 65-77%.
final2.py  loss weight 0.75, TARGET_USE 0.88. vs FULL 0-2 (73%),
           vs OLD 0-1 (84%).
final3.py  TARGET_USE 0.84. vs FULL 1-3 (75%), vs OLD 1-0 (83%).
agent.py   (shipped) TARGET_USE 0.80. vs STARTER 7-0 at 75%. No cut-offs or
           forfeits in any budgeted match; max wall time per decision ~700 ms.

TOOLS (tools/) - the evaluation scripts used. They contain the absolute
paths of the session that made them (REPO = /home/user/repo, snapshot
folders next to the script), so adjust those before reuse.
runmatches.py  N matches with P in parallel, start retries, frozen snapshot
               of the agent, optional --variant subclass
tourney.py     fixtures between variant classes (variants.json), FULL, or
               any path:Class
table.py       league table over tourney results
rcgstats.py    score, possession, territory, passes, shots, tackles from .rcg
goals.py       kick sequence before every goal
frames.py      ASCII pitch frames from an .rcg
shots.py       shot quality and conversion by distance
calib.py       predicted vs actual pass/dribble success (calibration)
onball.py      on-ball decision summary from traces
prof.py        CPU/wall time per decision type from traces
killall.sh     stop leftover simulator processes

RESULTS (results/) - raw per-match JSON from those runs (v1..v8, v2ctl,
fullvsfull, t1..t6, d8, d9).
