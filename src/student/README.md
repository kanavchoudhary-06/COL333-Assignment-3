# Agent1 — what I built

`src/student/agent.py`, class `Agent1`. Setup and running are in the top-level
`README.md`; to play it:

```
python3 a3.py match ME FULL --right full        # vs the HELIOS reference
python3 a3.py match ME STARTER --right :StarterAgent
python3 a3.py why -t ME                          # what each player decided
```

## Techniques (all from the allowed list)

| Part | Technique |
|---|---|
| World model | Kinematic rollout of ball flight and player reach time, batched with numpy. It is the transition model every decision below plans with. |
| CPU budget | A controller paces each decision so a player finishes the match at about 70–80% of its 75 s budget. Every search is **anytime**: it stops at a CPU deadline and at a wall-clock deadline 300 ms inside the 1 s cap. |
| On the ball | **Expectimax** over shots, passes (direct, lead, through) and self-pass dribbles. Each option is a chance node, scored `p·V(success) + (1−p)·V(loss)` with a **heuristic state evaluation** V. |
| Success estimates | p and V come from **Monte Carlo approximate sampling**: up to 64 rollouts with sampled kick noise, opponent reaction and reach, and goalie reaction. |
| Look-ahead | **Beam search**, second ply: the receiver's follow-up shot. |
| Shot model | Goalie reaction and catch radius fitted to observed shot outcomes in my own logs (**MLE-style parameter fit**). |
| Who chases | Minimum model reach time across teammates. Every player computes the same answer from the same full state, so no communication is needed. |
| Defence | A blocker gets goal-side of the owner's dribble path. The rest solve an **assignment CSP**: **backtracking, MRV, forward checking, branch & bound**. Each opponent's marking value is his threat as a receiver, computed by a fixed-size **2-ply expectimax** in the opponent's attacking frame. It is fixed-size so every defender gets identical numbers. |
| Support play | **Hill climbing** over coarse-to-fine grids. Each spot is scored by simulating the holder's pass to it, then a **second ply** (what I could do with the ball there) for the best spots while time lasts. |
| Stamina | Hysteresis recovery policy. |

The design choices are class attributes on `Agent1` (`PHI`, `FOLLOW`, `REFINE`,
`OFFBALL`, `STAMINA`, `TACKLE`, `SIM_SUPPORT`, `MINIMAX_DEF`, `THREAT_CSP`,
`LOSS_W`, `TARGET_USE`), so variants can be compared as subclasses.

## Budget use (shipped file)

Each player's spend over a full match, all with zero cut-offs and zero
forfeits:

| Match | CPU used (worst player) | Max wall time per decision |
|---|---|---|
| vs FULL | 75% | ~700 ms |
| vs previous light version | 83% (before lowering the target to 0.80) | ~700 ms |

Those wall times were measured with two matches sharing 4 cores; the
99th-percentile decision took 330–440 ms. On-ball decisions average
~140 ms and off-ball ~40 ms, against ~1 ms before the controller.

## How it was chosen, and honest results

1. **Round robin.** Nine variants of the light (~2% CPU) planner played
   STARTER, FULL and each other. The winner, `STAMINA = "tiered"`, went
   W8 D1 L5 against the other variants and won 11–0 over STARTER.
2. **More budget.** I added the budgeted searches and tested each one
   against that light winner:
   - A full second ply on every candidate hurt: maximizing over noisy
     look-ahead values favoured options the model over-rated. Monte Carlo
     sampling replaced it.
   - Minimax re-positioning of each defender broke the defensive shape and
     was switched off (`MINIMAX_DEF = False`).
   - Better success estimates made the team too conservative (about 1 shot
     per match). Lowering the loss weight to 0.75 brought shots back to 2+.
3. **Where it stands.** With the budget used, the agent is roughly level
   with the light version head-to-head, a little less than one would hope.
   It shoots more and keeps more of the ball. **It still loses to FULL**
   (1–3 and 0–2 in the last checks). More search on the same model gives
   diminishing returns: the limits are the model's fidelity (how FULL
   dribbles, how HELIOS's kicks execute) and the value function, not
   search depth.

## Known issue

Under heavy CPU load the harness sometimes gives up before kickoff
("NO_SHOW_DATA"). Re-running fixes it, and it is not caused by the agent.
