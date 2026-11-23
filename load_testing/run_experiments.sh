#!/usr/bin/env bash
# Run the same Locust workload against 1, 2 and 3 API containers and several user counts.
# Requires Docker. Results: load_testing/results/c<containers>_u<users>_stats.csv
set -euo pipefail
cd "$(dirname "$0")/.."
DURATION=${DURATION:-60s}
USERS=${USERS:-"5 20 50"}
for containers in 1 2 3; do
  docker compose up -d --build --scale api=$containers
  echo "Waiting for $containers healthy API container(s)..."
  until [ "$(docker compose ps api --format json | grep -c '"Health":"healthy"')" -ge "$containers" ]; do sleep 3; done
  docker compose ps api > "load_testing/results/c${containers}_containers.txt"
  for users in $USERS; do
    docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' > /dev/null
    ( sleep 30; docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' \
        > "load_testing/results/c${containers}_u${users}_docker_stats.csv" ) &
    locust -f load_testing/locustfile.py --host http://localhost:8000 --headless \
      -u "$users" -r 5 -t "$DURATION" --csv "load_testing/results/c${containers}_u${users}" --only-summary
    wait
  done
done
python load_testing/analyze_results.py
