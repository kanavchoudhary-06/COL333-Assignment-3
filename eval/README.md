# Evaluation on AWS with cdeploy

Plays `Agent1` (src/student/agent.py) against the grading reference team (`full`)
and the unmodified starter (`baseline`) on AWS VMs. It alternates sides and
reports W/D/L, goals, CPU budget use, cut-offs and forfeits.

| File | What it does |
| --- | --- |
| `run_cloud_eval.ps1` / `run_cloud_eval.sh` | One command: write the spec, `cdeploy plan`, `cdeploy up`, merge the results |
| `gen_spec.py` | Writes `cdeploy_eval.json`, one job (VM) per shard |
| `setup_vm.sh` | Sets up a fresh Ubuntu VM: Python 3.11 venv via uv, requirements, rcssserver + proxy (extracted), gRPC stubs, `a3.py doctor` |
| `run_matches.py` | Runs a shard of matches in parallel (separate ports) and writes `eval/out/results_shard<i>.json` / `.csv` |
| `aggregate.py` | Merges the shards cdeploy pulled back into `.cdeploy/runs/<id>/artifacts/` and writes `eval/out/summary.json` |

## Run it (Windows PowerShell, from the repo root)

```powershell
# 1. free check of the whole pipeline (cdeploy simulator, nothing on AWS)
.\eval\run_cloud_eval.ps1 -CdeployHome C:\path\to\Cloud_deployer

# 2. AWS keys for this shell only (or keep them in ~/.aws/credentials)
$env:AWS_ACCESS_KEY_ID = "..."; $env:AWS_SECRET_ACCESS_KEY = "..."

# 3. real run: 6 games per opponent on 3 spot c6i.xlarge VMs, $2 cap
.\eval\run_cloud_eval.ps1 -CdeployHome C:\path\to\Cloud_deployer -Real -Matches 6 -Shards 3
```

With the default SkyPilot backend, run `sky check` once so SkyPilot can see the AWS keys.
For the libcloud backend, add `-Backend libcloud -ImageId ami-...` (an Ubuntu 22.04 AMI in your region).
cdeploy shows the plan and asks before it spends money, and always tears the VMs down.
If a run is interrupted, use `python -m cdeploy status` and `python -m cdeploy down <run-id>`.

## Run on any Linux box without the cloud

```bash
bash eval/setup_vm.sh
.venv/bin/python eval/run_matches.py --matches 4 --opponents full,baseline
```

Notes
* One match takes about 80–100 s of wall time. Each VM runs (vCPUs − 1) matches at once.
* rcssserver 19 has no seed option, so repeats differ only through kick noise. The `seed` column is just a label.
* A match is marked invalid if any player process printed a traceback (the same rule a3.py uses).

## Tournaments and repeated matches between chosen agents (`tournament.py`)

```powershell
# round-robin: every pair plays --games games, sides alternate
python eval\tournament.py tournament main helios cyrus mcts full --games 2

# repeated matches between two agents
python eval\tournament.py matches main full --games 10

# free check of the whole pipeline first (fake scores, cdeploy simulator)
python eval\tournament.py tournament main helios full --aws --dry
```

* To run on AWS, fill in `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `CDEPLOY_HOME` at the top of `eval/tournament.py`, or set them as environment variables. With keys present it deploys automatically. Without keys it runs locally, which needs Linux/macOS or WSL. Use `--local` / `--aws` to force one.
* Agent names: `main mdp helios gliders cyrus mcts bayes full baseline idle`, or any `path/to/agent.py[:Class]`.
* Output: a standings table (3/1/0 points, then goal difference, then goals scored), a head-to-head grid, and `eval/out/<mode>_<time>.json` + `.csv`. Columns also show cut-offs, forfeits and the worst CPU use per agent.
* Once keys are in the file, don't commit it.
* Matches run through `eval/a3_patient.py`: `a3.py` unchanged, except a slow kick-off isn't mistaken for a finished match. Under parallel load the server's 600 pre-kick-off cycles can take over 30 s, and `a3.py` then ended matches early with `NO_SHOW_DATA (match never kicked off)`. Defaults: CPU cores / 2 matches at once, starts 15 s apart, one automatic retry of a match that never kicked off. `--half 15` gives short matches for quick checks.
