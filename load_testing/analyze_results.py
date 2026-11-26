"""Summarise Locust CSV output into a table and four plots.

    python load_testing/analyze_results.py
Reads load_testing/results/c<containers>_u<users>_stats.csv (written by Locust --csv).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
rows = []
for path in sorted(RESULTS.glob("c*_u*_stats.csv")):
    match = re.match(r"c(\d+)_u(\d+)_stats\.csv", path.name)
    if not match:   # e.g. c1_u5_docker_stats.csv (container CPU/memory snapshot), not a Locust file
        continue
    stats = pd.read_csv(path)
    total = stats[stats["Name"] == "Aggregated"].iloc[0]
    rows.append({"containers": int(match.group(1)), "users": int(match.group(2)),
                 "requests": int(total["Request Count"]), "failures": int(total["Failure Count"]),
                 "failure_rate": total["Failure Count"] / max(total["Request Count"], 1),
                 "rps": total["Requests/s"], "mean_ms": total["Average Response Time"],
                 "median_ms": total["Median Response Time"], "p95_ms": total["95%"], "max_ms": total["Max Response Time"]})
if not rows:
    raise SystemExit(f"No Locust results in {RESULTS}")
table = pd.DataFrame(rows).sort_values(["containers", "users"])
table.to_csv(RESULTS / "load_test_summary.csv", index=False)
print(table.to_string(index=False))

def lines(y, ylabel, filename, x="users"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    group = "containers" if x == "users" else "users"
    for key, part in table.groupby(group):
        ax.plot(part[x], part[y], marker="o", label=f"{key} {group}")
    ax.set_xlabel("concurrent users" if x == "users" else "API containers")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=120)
    plt.close(fig)

lines("median_ms", "median response time (ms)", "users_vs_response_time.png")
lines("rps", "throughput (requests/s)", "containers_vs_throughput.png", x="containers")
lines("p95_ms", "95th percentile response time (ms)", "containers_vs_p95.png", x="containers")
lines("failure_rate", "failure rate", "failure_rate.png")
