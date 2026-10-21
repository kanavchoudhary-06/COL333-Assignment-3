#!/usr/bin/env python3
"""
Merge the per-shard results that cdeploy pulled back and print one table.

    python eval/aggregate.py                 # newest run under .cdeploy/runs
    python eval/aggregate.py --run <run-id>  # a specific run
    python eval/aggregate.py PATH [PATH...]  # any results_*.json files / dirs

Writes eval/out/summary.json and eval/out/all_matches.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_matches import summarize, print_summary  # noqa: E402


def run_dirs(extra_roots):
    roots = [os.getcwd(), ROOT, HERE] + extra_roots
    seen, out = set(), []
    for r in roots:
        for d in glob.glob(os.path.join(r, ".cdeploy", "runs", "*")):
            d = os.path.abspath(d)
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                out.append(d)
    return sorted(out, key=os.path.getmtime)


def find_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "results_*.json"),
                               recursive=True)
            files += glob.glob(os.path.join(p, "**", "results.json"),
                               recursive=True)
        elif os.path.isfile(p):
            files.append(p)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--run", help="cdeploy run id")
    ap.add_argument("--cdeploy-home", default=os.environ.get("CDEPLOY_HOME", ""))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    a = ap.parse_args()

    paths = a.paths
    if not paths:
        dirs = run_dirs([a.cdeploy_home] if a.cdeploy_home else [])
        if a.run:
            dirs = [d for d in dirs if os.path.basename(d) == a.run]
        if not dirs:
            sys.exit("no cdeploy runs found; pass results files or dirs")
        paths = [dirs[-1]]
        print("run:", dirs[-1])
    files = find_files(paths)
    if not files:
        sys.exit("no results_*.json under " + ", ".join(paths))

    rows = []
    for f in files:
        data = json.load(open(f))
        for r in data.get("rows", []):
            r["source"] = os.path.relpath(f, os.getcwd())
            rows.append(r)
    rows.sort(key=lambda r: r.get("index", 0))
    per = summarize(rows)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as fh:
        json.dump({"files": files, "summary": per, "matches": len(rows)},
                  fh, indent=1)
    cols = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out_dir, "all_matches.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} matches from {len(files)} file(s)")
    print_summary(per)
    bad = [r for r in rows if not r.get("valid")]
    for r in bad[:10]:
        print(f"  invalid #{r.get('index')}: {r.get('problem', '')[:120]}")
    print("wrote", os.path.join(a.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
