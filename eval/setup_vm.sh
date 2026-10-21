#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu x86_64 VM so `python3 a3.py match` works.
# Run from anywhere; it works on the repo that contains this file.
# Safe to re-run: every step is skipped when already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "[setup] repo: $ROOT"

# Files shipped from Windows can carry CRLF line endings, which break sh.
for f in generate.sh start.sh scripts/*.sh scripts/start scripts/startAll eval/*.sh; do
  [ -f "$f" ] && sed -i 's/\r$//' "$f" || true
done

# System tools the download scripts need.
need=""
for t in wget curl tar; do command -v "$t" >/dev/null 2>&1 || need="$need $t"; done
if [ -n "$need" ]; then
  echo "[setup] installing:$need"
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need
fi

# Python 3.11 venv with the exact starter requirements (uv brings its own 3.11).
if [ ! -x .venv/bin/python ]; then
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  uv venv -q -p 3.11 .venv
  uv pip install -q --python .venv/bin/python -r requirements.txt
fi
PY="$ROOT/.venv/bin/python"

# Server + proxy binaries, extracted so FUSE is not needed (README, Linux).
if [ ! -x scripts/rcssserver/squashfs-root/AppRun ]; then
  (cd scripts && sh download-rcssserver.sh >/dev/null && \
   cd rcssserver && ./rcssserver --appimage-extract >/dev/null)
fi
if [ ! -e scripts/proxy/sample_player ] || [ ! -d scripts/proxy/extract_sample_player ]; then
  (cd scripts && sh download-proxy.sh >/dev/null && cd proxy && \
   for b in sample_player sample_coach sample_trainer; do
     mkdir -p "extract_$b" && (cd "extract_$b" && "../$b" --appimage-extract >/dev/null)
     mv "$b" "$b.AppImage"
     ln -s "extract_$b/squashfs-root/AppRun" "$b"
   done)
fi

# gRPC stubs.
if [ ! -f service_pb2_grpc.py ]; then
  PATH="$ROOT/.venv/bin:$PATH" sh ./generate.sh >/dev/null
fi

"$PY" -c "import service_pb2_grpc" && echo "[setup] grpc stubs ok"
"$PY" a3.py doctor | tail -3
echo "[setup] done"
