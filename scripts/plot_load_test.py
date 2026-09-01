# Generates a visual summary chart from Locust CSV results for README and judges

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = "reports/load_test_stats.csv"
OUT_PATH = "reports/load_test_summary.png"


# Reads the Locust stats CSV and returns rows as list of dicts
def read_stats(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] != "Aggregated":
                rows.append(row)
    return rows


# Generates and saves a 3-panel load test summary chart
def plot_summary(rows: list[dict], out_path: str) -> None:
    names = [r["Name"] for r in rows]
    p50 = [float(r["50%"]) for r in rows]
    p95 = [float(r["95%"]) for r in rows]
    p99 = [float(r["99%"]) for r in rows]
    rps  = [float(r["Requests/s"]) for r in rows]
    failures = [float(r["Failure Count"]) for r in rows]
    requests = [float(r["Request Count"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1e2433")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines["bottom"].set_color("#2d3748")
        ax.spines["left"].set_color("#2d3748")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = np.arange(len(names))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, p50, w, label="p50", color="#60a5fa")
    ax.bar(x, p95, w, label="p95", color="#fbbf24")
    ax.bar(x + w, p99, w, label="p99", color="#f87171")
    ax.set_title("Latency (ms)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("/score ", "\n") for n in names], color="#94a3b8")
    ax.set_ylabel("ms", color="#94a3b8")
    ax.legend(facecolor="#1e2433", labelcolor="#e2e8f0", fontsize=8)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=7, color="#e2e8f0")

    ax2 = axes[1]
    colors = ["#60a5fa", "#f87171", "#a78bfa"]
    bars = ax2.bar(x, rps, color=colors[:len(names)], width=0.5)
    ax2.set_title("Throughput (req/s)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("/score ", "\n") for n in names], color="#94a3b8")
    ax2.set_ylabel("req/s", color="#94a3b8")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#e2e8f0")

    ax3 = axes[2]
    success = [r - f for r, f in zip(requests, failures)]
    ax3.bar(x, success, 0.5, label="Success", color="#34d399")
    ax3.bar(x, failures, 0.5, bottom=success, label="Failed", color="#f87171")
    ax3.set_title("Request Outcomes", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace("/score ", "\n") for n in names], color="#94a3b8")
    ax3.set_ylabel("count", color="#94a3b8")
    ax3.legend(facecolor="#1e2433", labelcolor="#e2e8f0", fontsize=8)

    total_rps = sum(rps)
    total_req = sum(requests)
    total_fail = sum(failures)
    fail_pct = total_fail / total_req * 100 if total_req > 0 else 0
    fig.suptitle(
        f"RazorSentry Load Test — 100 concurrent users · 60s · MacBook Air M1\n"
        f"Total: {total_req:.0f} requests · {total_rps:.1f} req/s · {fail_pct:.1f}% failures",
        color="#f8fafc", fontsize=12, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117", edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    rows = read_stats(CSV_PATH)
    plot_summary(rows, OUT_PATH)
