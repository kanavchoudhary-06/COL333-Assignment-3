#!/usr/bin/env bash
# Run the agent evaluation on AWS through cdeploy, then merge the results.
#
#   CDEPLOY_HOME=/path/to/Cloud_deployer eval/run_cloud_eval.sh            # dry run
#   CDEPLOY_HOME=/path/to/Cloud_deployer eval/run_cloud_eval.sh --real \
#       --matches 6 --shards 3
#
# Extra flags go straight to eval/gen_spec.py (--opponents, --instance,
# --region, --budget, --backend, --image-id ...).
# AWS keys come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or
# ~/.aws/credentials. They are never written or printed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

MODE=dry-run
ARGS=()
for x in "$@"; do
  if [ "$x" = "--real" ]; then MODE=real; else ARGS+=("$x"); fi
done

if [ -n "${CDEPLOY_HOME:-}" ]; then
  [ -d "$CDEPLOY_HOME/cdeploy" ] || { echo "no cdeploy package in $CDEPLOY_HOME"; exit 1; }
  export PYTHONPATH="$CDEPLOY_HOME${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PY" -c "import cdeploy" 2>/dev/null || {
  echo "cannot import cdeploy: set CDEPLOY_HOME or pip install -e it"; exit 1; }

if [ "$MODE" = real ]; then
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ ! -f "$HOME/.aws/credentials" ]; then
    echo "no AWS credentials: export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or run 'aws configure'"
    exit 1
  fi
fi

SPEC=eval/cdeploy_eval.json
"$PY" eval/gen_spec.py --mode "$MODE" --out "$SPEC" "${ARGS[@]+"${ARGS[@]}"}"
echo "== plan";  "$PY" -m cdeploy plan "$SPEC"
echo "== up ($MODE)"
set +e
if [ "$MODE" = real ]; then "$PY" -m cdeploy up "$SPEC" --real; else "$PY" -m cdeploy up "$SPEC"; fi
code=$?
set -e
echo "== results"; "$PY" eval/aggregate.py || true
if [ $code -ne 0 ]; then
  echo "cdeploy exited with $code. Leftovers: $PY -m cdeploy status / $PY -m cdeploy down <run-id>"
fi
exit $code
