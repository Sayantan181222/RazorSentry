# Generates a clean architecture diagram for video recording and README

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Draws a rounded box with label and optional sublabel
def draw_box(ax, x, y, w, h, label, sublabel="", color="#1e2433", text_color="#f8fafc", border="#60a5fa"):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.65, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)
        ax.text(x + w/2, y + h*0.3, sublabel, ha="center", va="center",
                fontsize=7, color="#94a3b8")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)


# Draws an arrow between two points with a label
def draw_arrow(ax, x1, y1, x2, y2, label="", color="#60a5fa"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=6.5, color="#94a3b8",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f1117", edgecolor="none"))


# Generates and saves the full architecture diagram
def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.6, "RazorSentry — Production Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#f8fafc")
    ax.text(7, 8.25, "4 uvicorn workers · PostgreSQL · Redis Queue · HMAC-SHA256 PII blinding",
            ha="center", va="center", fontsize=9, color="#64748b")

    # Incoming traffic
    draw_box(ax, 5.5, 7.5, 3, 0.55, "Incoming Traffic",
             "Merchant API / Razorpay Webhooks", color="#0f1117", border="#475569")

    # Load balancer
    draw_box(ax, 5.5, 6.7, 3, 0.55, "Docker Port 8000",
             "Load Balancer (future: nginx)", color="#1a2035", border="#475569")

    # Workers
    for i, x in enumerate([1.5, 4.0, 6.5, 9.0]):
        draw_box(ax, x, 5.7, 2.2, 0.7,
                 f"Worker {i+1}", "FastAPI + uvicorn",
                 color="#1e2433", border="#60a5fa")

    ax.text(7, 5.45, "← Input Validation · Rate Limiting · SHAP Reason Codes →",
            ha="center", va="center", fontsize=7, color="#475569")

    # Sync path
    draw_box(ax, 1.2, 4.3, 3.2, 0.75,
             "POST /score (Sync)", "Decision in <60ms",
             color="#064e3b", border="#34d399")

    # Async path
    draw_box(ax, 5.8, 4.3, 3.2, 0.75,
             "POST /score/async", "Returns job_id instantly",
             color="#1e3a5f", border="#60a5fa")

    # Redis
    draw_box(ax, 5.8, 3.1, 3.2, 0.75,
             "Redis 7", "RQ Job Queue",
             color="#3b1f2b", border="#f87171")

    # RQ Worker
    draw_box(ax, 5.8, 1.9, 3.2, 0.75,
             "RQ Worker Container", "Dequeues + Scores",
             color="#2d1b4e", border="#a78bfa")

    # PII Blinding
    draw_box(ax, 10.0, 3.7, 3.5, 0.75,
             "PII Blinding Layer", "HMAC-SHA256[:16]",
             color="#1a2035", border="#fbbf24")

    # PostgreSQL
    draw_box(ax, 10.0, 2.6, 3.5, 0.75,
             "PostgreSQL 16", "Append-only audit log · pool=10",
             color="#1e2433", border="#34d399")

    # Monitors
    draw_box(ax, 1.2, 2.6, 3.2, 0.75,
             "EWMA Spike Monitor", "GET /monitor/spike",
             color="#1a2035", border="#fbbf24")

    draw_box(ax, 1.2, 1.5, 3.2, 0.75,
             "PSI Drift Monitor", "GET /monitor/drift",
             color="#1a2035", border="#fbbf24")

    # Dashboard
    draw_box(ax, 10.0, 1.5, 3.5, 0.75,
             "Live Dashboard", "GET /dashboard · 10s refresh",
             color="#1e2433", border="#a78bfa")

    # Arrows — traffic flow
    draw_arrow(ax, 7, 7.5, 7, 7.25)
    draw_arrow(ax, 7, 6.7, 7, 6.4)

    # Workers to sync
    draw_arrow(ax, 2.6, 5.7, 2.6, 5.05, "sync")
    # Workers to async
    draw_arrow(ax, 7.4, 5.7, 7.4, 5.05, "async")

    # Async to Redis
    draw_arrow(ax, 7.4, 4.3, 7.4, 3.85)
    # Redis to RQ Worker
    draw_arrow(ax, 7.4, 3.1, 7.4, 2.65)

    # Sync to PII
    draw_arrow(ax, 4.4, 4.65, 10.0, 4.05, "decision")
    # RQ Worker to PII
    draw_arrow(ax, 9.0, 2.25, 10.0, 4.0, "decision")

    # PII to PostgreSQL
    draw_arrow(ax, 11.75, 3.7, 11.75, 3.35)

    # PostgreSQL to monitors
    draw_arrow(ax, 10.0, 2.95, 4.4, 2.95, "reads")

    # PostgreSQL to dashboard
    draw_arrow(ax, 11.75, 2.6, 11.75, 2.25)

    # Monitors read from PG
    draw_arrow(ax, 2.8, 2.6, 2.8, 2.25)

    plt.tight_layout(pad=0.5)
    import os
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/architecture_diagram.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print("Saved: docs/architecture_diagram.png")


if __name__ == "__main__":
    draw_architecture()
