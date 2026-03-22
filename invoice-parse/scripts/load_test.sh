#!/usr/bin/env bash
# Load test: burst-enqueue invoices N times via Docker.
# Usage: ./scripts/load_test.sh [rounds]   (default: 100)
#
# Prerequisites: docker compose --profile app already running
# Monitor: http://localhost:8501 (Streamlit dashboard)

set -euo pipefail

ROUNDS="${1:-100}"
COMPOSE="docker compose -f infra/docker-compose.yaml"

echo "=== Load test: ${ROUNDS} rounds × 17 invoices = $((ROUNDS * 17)) jobs ==="
echo "Dashboard: http://localhost:8501"
echo ""

start=$(date +%s)

for i in $(seq 1 "$ROUNDS"); do
    $COMPOSE --profile ingest run --rm --no-log-prefix ingest 2>&1 | tail -1
done

end=$(date +%s)
elapsed=$((end - start))

echo ""
echo "=== Done: $((ROUNDS * 17)) jobs enqueued in ${elapsed}s ==="
echo ""
echo "Queue depth:"
$COMPOSE exec redis redis-cli XLEN queue:a
$COMPOSE exec redis redis-cli XLEN queue:b
