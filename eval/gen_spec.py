#!/usr/bin/env python3
"""
Write a cdeploy spec that runs the evaluation on AWS, one VM per shard.

    python eval/gen_spec.py --matches 6 --shards 3              # dry-run spec
    python eval/gen_spec.py --matches 6 --shards 3 --mode real  # real AWS

Each shard job:  setup = eval/setup_vm.sh   (python 3.11, binaries, stubs)
                 run   = eval/run_matches.py --shard i --shards k
                 output= eval/out/results_shard<i>.json (+ .csv)
Matches are spread round-robin over shards, so every shard sees every
opponent. AWS credentials are never written into the spec: SkyPilot reads
them from your AWS config / environment; the libcloud backend references
them as "$AWS_ACCESS_KEY_ID" / "$AWS_SECRET_ACCESS_KEY".
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--matches", type=int, default=6, help="games per opponent")
    ap.add_argument("--opponents", default="full,baseline")
    ap.add_argument("--shards", type=int, default=2, help="VMs to use")
    ap.add_argument("--parallel", type=int, default=0,
                    help="matches at once per VM (0 = vCPUs - 1)")
    ap.add_argument("--mode", choices=["dry-run", "real"], default="dry-run")
    ap.add_argument("--backend", choices=["skypilot", "libcloud"],
                    default="skypilot")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--instance", default="c6i.xlarge",
                    help="first choice (4 vCPU)")
    ap.add_argument("--fallback", default="c5.xlarge")
    ap.add_argument("--image-id", default="",
                    help="Ubuntu 22.04 AMI, required by --backend libcloud")
    ap.add_argument("--no-spot", action="store_true")
    ap.add_argument("--budget", type=float, default=2.0, help="total USD cap")
    ap.add_argument("--max-hourly", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(HERE, "cdeploy_eval.json"))
    a = ap.parse_args()

    if a.backend == "libcloud" and a.mode == "real" and not a.image_id:
        ap.error("--backend libcloud needs --image-id (an Ubuntu AMI)")

    # ~80 s wall per match on 2 vCPUs, about 3 at a time on 4 vCPUs
    total = a.matches * len([o for o in a.opponents.split(",") if o.strip()])
    per_shard = -(-total // a.shards)
    est_h = round(max(0.15, 0.1 + per_shard * 90.0 / 3.0 / 3600.0), 2)

    provider = {"backend": a.backend, "regions": [a.region],
                "max_concurrent": 2, "rate_limit_per_sec": 4, "burst": 4,
                "breaker_threshold": 3, "breaker_cooldown_s": 10}
    if a.backend == "libcloud":
        provider["credentials"] = {"access_key": "$AWS_ACCESS_KEY_ID",
                                   "secret_key": "$AWS_SECRET_ACCESS_KEY"}

    def cand(itype, spot):
        c = {"cloud": "aws", "instance_type": itype, "spot": spot,
             "est_hours": est_h, "disk_gb": 30}
        if a.image_id:
            c["image_id"] = a.image_id
        return c

    dry = a.mode == "dry-run"
    jobs = []
    for i in range(a.shards):
        out = f"eval/out/results_shard{i}.json"
        run = (f"python3 eval/run_matches.py --dry --shard {i} "
               f"--shards {a.shards} --matches {a.matches} "
               f"--opponents {a.opponents} --out {out}") if dry else (
               f".venv/bin/python eval/run_matches.py --shard {i} "
               f"--shards {a.shards} --matches {a.matches} "
               f"--opponents {a.opponents} --parallel {a.parallel} "
               f"--out {out}")
        jobs.append({
            "name": f"eval{i}",
            "est_hours": est_h,
            "candidates": [cand(a.instance, not a.no_spot),
                           cand(a.fallback, False)],
            "setup": ("python3 -c \"print('dry-run: setup skipped')\""
                      if dry else "bash eval/setup_vm.sh"),
            "run": run,
            "env": {"PYTHONUNBUFFERED": "1"},
            "outputs": [out, out.replace(".json", ".csv")],
        })

    spec = {
        "project": "col333-a3-eval",
        "mode": a.mode,
        "workdir": "..",               # the repo root, relative to this file
        "budget": {"total_usd": a.budget, "max_hourly_usd": a.max_hourly,
                   "alert_fraction": 0.8},
        "providers": {"aws": provider},
        "jobs": jobs,
        "agent": {"max_workers": 8, "offload": "thread", "memory_mb": None,
                  "cpu_percent": None, "health_timeout_s": 600,
                  "ws_port": 8765},
        "simulation": {"seed": 8, "capacity_failure_rate": 0.0,
                       "throttle_rate": 0.0, "provision_seconds": [0.2, 0.6],
                       "time_scale": 30},
    }
    with open(a.out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {a.out}: {a.mode}, {a.shards} job(s), {total} matches, "
          f"~{est_h} h per VM, budget ${a.budget}")


if __name__ == "__main__":
    main()
