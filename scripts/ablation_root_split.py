"""Ablation: what does splitting merged incidents by root candidate buy downstream?

    python scripts/ablation_root_split.py

Runs the pipeline twice per seed with the same detector, once with time-plus-topology
correlation and once with the root split added, and prints the downstream accuracy.
"""

import numpy as np

from aiops.config import Config
from aiops.detect import SeasonalResidual
from aiops.pipeline import build

SEEDS = (7, 11, 13, 17, 19)
VARIANTS = ("time_and_topology", "time_topology_root_split")


def main() -> None:
    rows = {variant: {"rca": [], "agent": []} for variant in VARIANTS}
    for seed in SEEDS:
        for variant in VARIANTS:
            metrics = build(Config(seed=seed), detectors=[SeasonalResidual()], correlation=variant).metrics
            rows[variant]["rca"].append(metrics["root_cause"]["methods"]["random_walk"]["top1"])
            rows[variant]["agent"].append(metrics["agent"]["fully_correct"])
    print(f"{'correlation':28s} {'random-walk top-1':>20s} {'agent fully correct':>22s}")
    for variant in VARIANTS:
        rca, agent = np.array(rows[variant]["rca"]), np.array(rows[variant]["agent"])
        print(f"{variant:28s} {rca.mean():>12.1%} ± {rca.std():.1%} {agent.mean():>14.1%} ± {agent.std():.1%}")


if __name__ == "__main__":
    main()
