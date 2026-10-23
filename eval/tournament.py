#!/usr/bin/env python3
"""
Tournaments and repeated matches between chosen agents, on AWS or locally.

  # round-robin between any agents (every pair plays --games, sides alternate)
  python eval/tournament.py tournament main helios cyrus full --games 2

  # repeated matches between two agents
  python eval/tournament.py matches main full --games 10

Where it runs:
  * AWS keys filled in below (or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in
    the environment)  -> the fixtures are split over VMs through cdeploy.
  * no keys            -> runs here (needs Linux/macOS with `a3.py doctor`
    passing; on Windows use AWS or WSL).
  --local / --aws force one or the other. --dry fakes every result (and
  uses cdeploy's free simulator), to check the whole pipeline first.

Agent names (see AGENTS below): main mdp helios gliders cyrus mcts bayes
full baseline idle, or any path/to/agent.py[:Class].

Standings use the assignment rules: 3/1/0 points, then goal difference,
then goals scored. Output: eval/out/tournament_<time>.json and .csv.
"""

# ============================ fill in to run on AWS ==========================
# Keep this file out of git once keys are in it (or use environment variables).
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "us-east-1"
CDEPLOY_HOME = ""            # folder that contains the cdeploy/ package
AWS_INSTANCE = "c6i.xlarge"  # 4 vCPU; about 3 matches at a time
AWS_SHARDS = 3               # number of VMs
AWS_BUDGET_USD = 3.0
# =============================================================================

import argparse
import csv
import glob
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "eval")
OUT = os.path.join(HERE, "out")
AGENTS = {
    "main": "src/student/agent.py",
    "mdp": "agents/mdp_qlearning/agent.py",
    "helios": "agents/helios_chain/agent.py",
    "gliders": "agents/gliders_fractals/agent.py",
    "cyrus": "agents/cyrus_mlp/agent.py",
    "mcts": "agents/mcts_alice/agent.py",
    "bayes": "agents/bayes_minimax/agent.py",
    "full": "full",                          # grading reference team
    "baseline": "src/student/agent.py:Baseline",
    "idle": "src/student/agent.py:Agent2",
    "harshal": "agents/harshal_agent/agent.py",
    "claude_heavy": "agents/claude_planner/agent.py",
    "claude_light": "agents/claude_planner/versions/OLD_light_VTired.py",
}
RESULT = re.compile(r"result\s*:\s*(\S+)\s+(\d+)-(\d+)\s+(\S+)\s+\[(.*?)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ------------------------------------------------------------------ fixtures
def spec_of(name):
    s = AGENTS.get(name, name)
    if s == "full":
        return s
    path, _, cls = s.partition(":")
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return full + (":" + cls if cls else "")


def team_name(agent, other):
    """Short, legal rcssserver team name: the agent's key, or for a path the
    folder holding its agent.py (plus the class, if one is given)."""
    if agent in AGENTS:
        raw = agent
    else:
        path, _, cls = agent.partition(":")
        raw = os.path.basename(os.path.dirname(os.path.abspath(path))) + cls
    return (re.sub(r"[^A-Za-z0-9]", "", raw).upper() or "TEAM")[:10]


def fixtures(mode, agents, games):
    pairs = []
    if mode == "matches":
        pairs = [(agents[0], agents[1])]
    else:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pairs.append((agents[i], agents[j]))
    out = []
    for a, b in pairs:
        for g in range(games):
            left, right = (a, b) if g % 2 == 0 else (b, a)
            out.append({"id": len(out), "left": left, "right": right, "game": g})
    return out


# ---------------------------------------------------------------- one match
def timing(stamp, team):
    d = os.path.join(ROOT, "runlogs", "timing", stamp)
    spent, cuts = 0.0, 0
    for p in glob.glob(os.path.join(d, f"{team}_*.json")):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        spent = max(spent, r.get("spent_ms", 0.0) or 0.0)
        cuts += r.get("killed", 0) or 0
    forf = len(glob.glob(os.path.join(d, f"FORFEIT_{team}_*.json")))
    return round(spent / 1000.0, 1), cuts, forf


def play(fx, port, half=0, timeout=1200):
    ln, rn = team_name(fx["left"], fx["right"]), team_name(fx["right"], fx["left"])
    if ln == rn:
        ln, rn = ln[:9] + "1", rn[:9] + "2"
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"_t{fx['id']:04d}"
    # a3_patient.py = a3.py unchanged, but a slow kick-off under parallel
    # load is not mistaken for a finished match (see that file)
    cmd = [sys.executable, os.path.join(HERE, "a3_patient.py"), "match", ln, rn,
           "--left", spec_of(fx["left"]), "--right", spec_of(fx["right"]),
           "-p", str(port), "--no-trace", "--stamp", stamp]
    if half:
        cmd += ["-t", str(half)]
    row = dict(fx)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        text = ANSI.sub("", p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired:
        text = "TIMEOUT"
    row["wall_s"] = round(time.time() - t0, 1)
    m = RESULT.search(text)
    if m:
        n1, s1, s2, n2, tag = m.groups()
        sc = {n1: int(s1), n2: int(s2)}
        row.update(lg=sc.get(ln, 0), rg=sc.get(rn, 0), valid=tag.strip() == "valid",
                   problem="" if tag.strip() == "valid" else tag.strip())
    else:
        row.update(lg=None, rg=None, valid=False, problem=text.strip()[-200:])
    row["l_cpu_s"], row["l_cuts"], row["l_forfeits"] = timing(stamp, ln)
    row["r_cpu_s"], row["r_cuts"], row["r_forfeits"] = timing(stamp, rn)
    return row


def fake(fx):
    rng = random.Random(fx["id"])
    return dict(fx, lg=rng.randint(0, 3), rg=rng.randint(0, 3), valid=True,
                problem="", wall_s=0.0, l_cpu_s=0.0, l_cuts=0, l_forfeits=0,
                r_cpu_s=0.0, r_cuts=0, r_forfeits=0, dry=True)


STAGGER_S = 15.0   # gap between match starts: start-up is the CPU-heavy part


def default_parallel():
    """A match is ~24 processes and needs about one core while playing, more
    while starting; more than cores/2 at once slows kick-off and can push
    decisions toward the 1 s wall cap, which would distort results."""
    return max(1, (os.cpu_count()-1))


def run_local(fxs, parallel, dry, out_path=None, half=0):
    if dry:
        return [fake(f) for f in fxs]
    if os.name == "nt":
        sys.exit("a3.py's server needs Linux or macOS: fill in the AWS keys at "
                 "the top of this file, or run it inside WSL.")
    par = parallel or default_parallel()
    ports = queue.Queue()
    for k in range(par):
        ports.put(6000 + 10 * k)
    gate = threading.Lock()
    last = [0.0]

    def job(fx):
        port = ports.get()
        try:
            for attempt in range(2):            # one retry if it never started
                with gate:                      # stagger the start-ups
                    wait = last[0] + STAGGER_S - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    last[0] = time.time()
                row = play(fx, port, half)
                if row["valid"] or "NO_SHOW_DATA" not in row["problem"]:
                    break
            row["attempts"] = attempt + 1
            return row
        finally:
            ports.put(port)

    rows = []
    print(f"{len(fxs)} matches, {par} at a time")
    with ThreadPoolExecutor(max_workers=par) as ex:
        for r in ex.map(job, fxs):
            rows.append(r)
            print(f"  #{r['id']:03d} {r['left']} {r['lg']}-{r['rg']} {r['right']}"
                  f"  valid={r['valid']} {r['wall_s']}s", flush=True)
            if out_path:
                save_json(out_path, {"rows": rows})
    return rows


# --------------------------------------------------------------- standings
def standings(rows):
    t = {}
    for r in rows:
        for side, opp in (("l", "r"), ("r", "l")):
            name = r["left"] if side == "l" else r["right"]
            s = t.setdefault(name, dict(P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0,
                                        cuts=0, forfeits=0, max_cpu_s=0.0,
                                        invalid=0))
            s["cuts"] += r.get(side + "_cuts", 0) or 0
            s["forfeits"] += r.get(side + "_forfeits", 0) or 0
            s["max_cpu_s"] = max(s["max_cpu_s"], r.get(side + "_cpu_s", 0.0) or 0.0)
            if not r.get("valid") or r.get("lg") is None:
                s["invalid"] += 1
                continue
            gf, ga = r[side + "g"], r[opp + "g"]
            s["P"] += 1
            s["GF"] += gf
            s["GA"] += ga
            if gf > ga:
                s["W"] += 1
                s["Pts"] += 3
            elif gf == ga:
                s["D"] += 1
                s["Pts"] += 1
            else:
                s["L"] += 1
    for s in t.values():
        s["GD"] = s["GF"] - s["GA"]
    return sorted(t.items(), key=lambda kv: (-kv[1]["Pts"], -kv[1]["GD"],
                                             -kv[1]["GF"], kv[0]))


def head_to_head(rows):
    h = {}
    for r in rows:
        if not r.get("valid") or r.get("lg") is None:
            continue
        for a, b, ga, gb in ((r["left"], r["right"], r["lg"], r["rg"]),
                             (r["right"], r["left"], r["rg"], r["lg"])):
            x = h.setdefault((a, b), [0, 0, 0])
            x[0 if ga > gb else 1 if ga == gb else 2] += 1
    return h


def report(rows, mode):
    table = standings(rows)
    print()
    print(f"{'#':>2} {'agent':<14}{'P':>3}{'W':>4}{'D':>4}{'L':>4}{'GF':>5}{'GA':>5}"
          f"{'GD':>5}{'Pts':>5}{'cuts':>6}{'forf':>6}{'maxCPU':>8}{'inv':>5}")
    for i, (n, s) in enumerate(table, 1):
        print(f"{i:>2} {n:<14}{s['P']:>3}{s['W']:>4}{s['D']:>4}{s['L']:>4}"
              f"{s['GF']:>5}{s['GA']:>5}{s['GD']:>5}{s['Pts']:>5}{s['cuts']:>6}"
              f"{s['forfeits']:>6}{s['max_cpu_s']:>7.1f}s{s['invalid']:>5}")
    if mode == "tournament":
        print("\nhead to head (W-D-L, row vs column)")
        names = [n for n, _ in table]
        h = head_to_head(rows)
        print(" " * 14 + "".join(f"{n[:9]:>10}" for n in names))
        for a in names:
            cells = ["-".join(map(str, h[(a, b)])) if (a, b) in h else "."
                     for b in names]
            print(f"{a[:13]:<14}" + "".join(f"{c:>10}" for c in cells))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(OUT, f"{mode}_{stamp}.json")
    save_json(path, {"mode": mode, "rows": rows,
                     "standings": [dict(agent=n, **s) for n, s in table]})
    with open(path.replace(".json", ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "P", "W", "D", "L", "GF", "GA", "GD", "Pts",
                    "cuts", "forfeits", "max_cpu_s", "invalid"])
        for n, s in table:
            w.writerow([n] + [s[k] for k in ("P", "W", "D", "L", "GF", "GA", "GD",
                                             "Pts", "cuts", "forfeits",
                                             "max_cpu_s", "invalid")])
    print(f"\nwrote {path}")


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(path + ".tmp", path)


# --------------------------------------------------------------------- AWS
def aws_keys():
    kid = AWS_ACCESS_KEY_ID or os.environ.get("AWS_ACCESS_KEY_ID", "")
    sec = AWS_SECRET_ACCESS_KEY or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    return (kid, sec) if kid and sec else None


def run_aws(fxs, a):
    keys = aws_keys()
    if not keys and not a.dry:
        sys.exit("no AWS keys: fill them in at the top of eval/tournament.py")
    home = a.cdeploy_home or CDEPLOY_HOME or os.environ.get("CDEPLOY_HOME", "")
    env = dict(os.environ)
    if home:
        env["PYTHONPATH"] = home + os.pathsep + env.get("PYTHONPATH", "")
    if keys:                                   # handed to cdeploy/SkyPilot only
        env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"] = keys
        env["AWS_DEFAULT_REGION"] = a.region
    if subprocess.run([sys.executable, "-c", "import cdeploy"], env=env,
                      capture_output=True).returncode != 0:
        sys.exit("cannot import cdeploy: set CDEPLOY_HOME at the top of this file")

    shards = max(1, min(a.shards, len(fxs)))
    save_json(os.path.join(OUT, "fixtures.json"), {"fixtures": fxs})
    per_vm = -(-len(fxs) // shards)
    est_h = round(max(0.2, 0.15 + per_vm * 100.0 / 2.0 / 3600.0), 2)  # 2 at a time on 4 vCPU
    jobs = []
    for i in range(shards):
        out = f"eval/out/tour_shard{i}.json"
        worker = (f"eval/tournament.py worker --fixtures eval/out/fixtures.json "
                  f"--shard {i} --shards {shards} --out {out}"
                  + (f" --half {a.half}" if a.half else ""))
        jobs.append({
            "name": f"tour{i}",
            "est_hours": est_h,
            "candidates": [
                {"cloud": "aws", "instance_type": a.instance, "spot": True,
                 "est_hours": est_h, "disk_gb": 30},
                {"cloud": "aws", "instance_type": "c5.xlarge", "spot": False,
                 "est_hours": est_h, "disk_gb": 30}],
            "setup": ("python3 -c \"print('dry-run')\"" if a.dry
                      else "bash eval/setup_vm.sh"),
            "run": ("python3 " + worker + " --dry") if a.dry
                   else (".venv/bin/python " + worker),
            "env": {"PYTHONUNBUFFERED": "1"},
            "outputs": [out],
        })
    spec = {
        "project": "col333-a3-tournament",
        "mode": "dry-run" if a.dry else "real",
        "workdir": "..",
        "budget": {"total_usd": a.budget, "max_hourly_usd": 2.0,
                   "alert_fraction": 0.8},
        "providers": {"aws": {"backend": "skypilot", "regions": [a.region],
                              "max_concurrent": 2, "rate_limit_per_sec": 4,
                              "burst": 4, "breaker_threshold": 3,
                              "breaker_cooldown_s": 10}},
        "jobs": jobs,
        "agent": {"max_workers": 8, "offload": "thread", "memory_mb": None,
                  "cpu_percent": None, "health_timeout_s": 600, "ws_port": 8765},
        "simulation": {"seed": 8, "capacity_failure_rate": 0.0,
                       "throttle_rate": 0.0, "provision_seconds": [0.2, 0.6],
                       "time_scale": 30},
    }
    spec_path = os.path.join(HERE, "cdeploy_tournament.json")
    save_json(spec_path, spec)
    print(f"{len(fxs)} matches on {shards} VM(s), ~{est_h} h each, "
          f"budget ${a.budget} ({'simulator' if a.dry else 'real AWS'})")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "cdeploy", "plan", spec_path],
                   cwd=ROOT, env=env)
    up = [sys.executable, "-m", "cdeploy", "up", spec_path]
    if not a.dry:
        up.append("--real")                    # cdeploy asks before spending
    code = subprocess.run(up, cwd=ROOT, env=env).returncode
    if code != 0:
        print(f"cdeploy exited {code}; check `python -m cdeploy status` and "
              "`python -m cdeploy down <run-id>` for leftovers")
    roots = [ROOT, HERE] + ([home] if home else [])
    files = []
    for r in roots:
        for d in glob.glob(os.path.join(r, ".cdeploy", "runs", "*")):
            if os.path.getmtime(d) >= t0 - 5:
                files += glob.glob(os.path.join(d, "**", "tour_shard*.json"),
                                   recursive=True)
    rows = []
    for f in sorted(set(files)):
        rows += json.load(open(f)).get("rows", [])
    if not rows:
        sys.exit("no results came back")
    return sorted(rows, key=lambda r: r["id"])


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["tournament", "matches", "worker"])
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--games", type=int, default=2,
                    help="games per pairing (sides alternate)")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--aws", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--parallel", type=int, default=0,
                    help="matches at once (default: CPU cores / 2)")
    ap.add_argument("--half", type=int, default=0,
                    help="half length in seconds for quick tests (default: base.conf)")
    ap.add_argument("--shards", type=int, default=AWS_SHARDS)
    ap.add_argument("--instance", default=AWS_INSTANCE)
    ap.add_argument("--region", default=AWS_REGION)
    ap.add_argument("--budget", type=float, default=AWS_BUDGET_USD)
    ap.add_argument("--cdeploy-home", default="")
    # worker (runs on each VM)
    ap.add_argument("--fixtures")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.mode == "worker":
        fxs = json.load(open(os.path.join(ROOT, a.fixtures)))["fixtures"]
        mine = [f for f in fxs if f["id"] % a.shards == a.shard]
        out = os.path.join(ROOT, a.out)
        rows = run_local(mine, a.parallel, a.dry, out, a.half)
        save_json(out, {"rows": rows})
        return

    if a.mode == "matches" and len(a.agents) != 2:
        ap.error("matches needs exactly two agents")
    if a.mode == "tournament" and len(a.agents) < 2:
        ap.error("a tournament needs at least two agents")
    for n in a.agents:
        s = AGENTS.get(n, n)
        if s != "full" and not os.path.isfile(spec_of(n).split(":")[0]):
            ap.error(f"unknown agent {n!r}: use {', '.join(AGENTS)} or a path")
    fxs = fixtures(a.mode, a.agents, a.games)
    use_aws = a.aws or (not a.local and aws_keys() is not None)
    rows = (run_aws(fxs, a) if use_aws
            else run_local(fxs, a.parallel, a.dry, None, a.half))
    report(rows, a.mode)


if __name__ == "__main__":
    main()
