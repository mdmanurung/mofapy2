"""Example: run Consensus MOFA2 on the bundled three-view test data.

Usage
-----
    python scripts/run_consensus_mofa.py

Outputs
-------
    consensus.hdf5          — consensus Z/W, per-run results, clustering.
    consensus_density.png   — histogram of component local densities.
    consensus_clustergram.png — pairwise distance matrix heatmap.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mofapy2.run.consensus import ConsensusMOFA

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mofapy2",
    "run",
    "test_data",
)


def main():
    # Each file is a (samples, features) matrix; stack as a single group.
    views = [
        np.loadtxt(os.path.join(DATA_DIR, "view_0.txt")),
        np.loadtxt(os.path.join(DATA_DIR, "view_1.txt")),
        np.loadtxt(os.path.join(DATA_DIR, "view_2.txt")),
    ]
    data = [[v] for v in views]  # M views × 1 group

    cm = ConsensusMOFA(
        data=data,
        K=5,
        views_names=["view0", "view1", "view2"],
        train_options={"iter": 200, "convergence_mode": "fast"},
    )
    cm.factorize(n_iter=10, seeds=range(10))
    cm.combine(source="W")
    cm.cluster(density_threshold=np.inf)
    result = cm.consensus()

    print(f"Consensus Z shape: {result['Z'].shape}")
    for m, W in enumerate(result["W"]):
        print(f"  view{m} W shape: {W.shape}")
    print(f"Per-factor stability: {np.round(result['stability'], 3)}")

    cm.save("consensus.hdf5")

    fig, ax = plt.subplots()
    cm.plot_local_density(ax=ax)
    fig.savefig("consensus_density.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    cm.plot_clustergram(ax=ax)
    fig.savefig("consensus_clustergram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
