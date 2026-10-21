#!/usr/bin/env python3
"""
Run a3.py exactly as shipped, with one change used by the evaluation scripts:
a match that is slow to KICK OFF is not mistaken for a finished one.

Why: before kick-off the server runs `kick_off_wait` (600) idle cycles in
synch mode, each waiting for all 24 clients. When several matches share the
CPU this can take more than 30 s, and the replay does not grow until
kick-off. a3.py ends a match after 30 s without replay growth, so it killed
these matches just before kick-off -> "NO_SHOW_DATA (match never kicked off)".

Here the 30 s "replay stopped growing" rule only applies once the match has
started; before kick-off we wait up to KICKOFF_GRACE seconds. The server
still exits by itself at full time, which ends the wait as before. Nothing
about the match itself (config, budget, agents) changes.

    python3 eval/a3_patient.py match A B --left ... --right ...   # same args as a3.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KICKOFF_GRACE = 300.0      # seconds allowed between server start and kick-off
STARTED_BYTES = 64_000     # replay larger than this = the match has kicked off

sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv[0] = os.path.join(ROOT, "a3.py")
import a3  # noqa: E402


def _wait_for_end(srv, rcg, end_at_iters=6, step=5):
    sys.stderr.write("-- playing ")
    last, stable, t0 = -1, 0, time.time()
    while True:
        time.sleep(step)
        try:
            sz = os.path.getsize(rcg)
        except OSError:
            sz = 0
        if sz == last:
            stable += 1
        else:
            stable = 0
            sys.stderr.write(".")
            sys.stderr.flush()
        last = sz
        if srv.poll() is not None:
            break                                   # server finished the game
        if sz > STARTED_BYTES:
            if stable >= end_at_iters:
                break                               # a3.py's own rule
        elif time.time() - t0 > KICKOFF_GRACE:
            break                                   # never kicked off
    sys.stderr.write(" done\n")
    return t0, time.time()


a3._wait_for_end = _wait_for_end

if __name__ == "__main__":
    sys.exit(a3.main())
