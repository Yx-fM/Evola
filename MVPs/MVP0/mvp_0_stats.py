"""MVP 0 — statistics tracker.

Tracks agent state over time and produces summary plots.
"""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt


class StatsTracker:
    def __init__(self, agent_id: str, total_cells: int):
        self.agent_id = agent_id
        self.total_cells = total_cells
        self.records: list[dict] = []

    def record(self, step: int, agent):
        h = agent.homeo
        drive = h.get_drive()
        nv = len(agent.visited)
        record = {
            "step": step,
            "energy": h.energy.value,
            "energy_drive": drive.energy,
            "novelty": h.novelty.value,
            "novelty_drive": drive.novelty,
            "safety": h.safety.value,
            "safety_drive": drive.safety,
            "visited": nv,
            "exploration_pct": round(nv / self.total_cells * 100, 1),
        }
        self.records.append(record)

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.records[0].keys())
            w.writeheader()
            w.writerows(self.records)

    def save_plot(self, path: str):
        steps = [r["step"] for r in self.records]
        energy = [r["energy"] for r in self.records]
        edrive = [r["energy_drive"] for r in self.records]
        novelty = [r["novelty"] for r in self.records]
        ndrive = [r["novelty_drive"] for r in self.records]
        safety = [r["safety"] for r in self.records]
        sdrive = [r["safety_drive"] for r in self.records]
        explored = [r["exploration_pct"] for r in self.records]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Evola MVP 0 — {self.agent_id} over {len(steps)} steps", fontsize=14)

        ax = axes[0, 0]
        ax.plot(steps, energy, "b-", alpha=0.5, label="energy level")
        ax.plot(steps, edrive, "r-", label="energy drive")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="comfort")
        ax.set_ylabel("Energy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(steps, explored, "g-", label="explored %")
        ax.set_ylabel("Exploration %")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(steps, novelty, "b-", alpha=0.5, label="novelty level")
        ax.plot(steps, ndrive, "purple", label="novelty drive")
        ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.3)
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Novelty")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(steps, safety, "orange", alpha=0.5, label="safety level")
        ax.plot(steps, sdrive, "red", label="safety drive")
        ax.set_xlabel("Step")
        ax.set_ylabel("Safety")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=120)
        plt.close(fig)
