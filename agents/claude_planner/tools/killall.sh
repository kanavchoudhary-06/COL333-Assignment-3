#!/bin/sh
# stop every simulator process (server, proxies, python brains)
for pat in squashfs-root/AppRun "start.py -t" "server.py --rpc-port" sample_player sample_coach; do
  pgrep -f "$pat" | xargs -r kill -9 2>/dev/null
done
exit 0
