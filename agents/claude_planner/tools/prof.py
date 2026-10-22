import json, glob, statistics, sys
d = sorted(glob.glob(f"/home/user/repo/runlogs/trace/{sys.argv[1]}*"))[-1]
team = sys.argv[2] if len(sys.argv) > 2 else "ME"
print(d)
errs = 0
for f in sorted(glob.glob(f"{d}/{team}/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    errs += sum(1 for r in rows for n in (r.get("notes") or []) if "error" in n.get("note", ""))
    on = [r["ms"] for r in rows if r.get("kickable")]; off = [r["ms"] for r in rows if not r.get("kickable")]
    walls = [r.get("wall_ms", 0) for r in rows]
    by = {}
    for r in rows:
        k = (r.get("notes") or [{}])[0].get("note"); by.setdefault(k, []).append(r["ms"])
    print(f.split('/')[-1], "n", len(rows), "on", len(on), f"on {statistics.mean(on) if on else 0:.0f}ms off {statistics.mean(off):.1f}ms",
          f"wall max {max(walls):.0f} p99 {sorted(walls)[int(.99*len(walls))]:.0f}", f"left {rows[-1].get('left_ms', 0)/1000:.1f}s",
          {k: round(statistics.mean(v), 1) for k, v in by.items() if len(v) > 20})
print("errors", errs)
