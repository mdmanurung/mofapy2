"""Tests for mofapy2.run.consensus.ConsensusMOFA."""

import numpy as np
import pytest

from mofapy2.run.consensus import ConsensusMOFA


def _make_toy_data(N=40, D=(30, 25), K_true=3, seed=0):
    """Generate a simple low-rank two-view dataset with additive noise."""
    rng = np.random.default_rng(seed)
    Z_true = rng.standard_normal((N, K_true))
    views = []
    for d in D:
        W_true = rng.standard_normal((d, K_true))
        Y = Z_true @ W_true.T + 0.1 * rng.standard_normal((N, d))
        views.append([Y.astype(np.float64)])  # 1 group
    return views  # nested list data[m][g]


@pytest.fixture(scope="module")
def toy_data():
    return _make_toy_data()


def _fit(data, K=3, n_iter=3, seeds=(0, 1, 2)):
    cm = ConsensusMOFA(
        data=data,
        K=K,
        train_options={"iter": 100, "convergence_mode": "fast"},
    )
    cm.factorize(n_iter=n_iter, seeds=seeds, verbose=False)
    return cm


def test_factorize_returns_fixed_K(toy_data):
    cm = _fit(toy_data, K=3, n_iter=3, seeds=[0, 1, 2])
    assert len(cm.runs) == 3
    for run in cm.runs:
        assert run["Z"].shape == (40, 3)
        assert len(run["W"]) == 2
        assert run["W"][0].shape == (30, 3)
        assert run["W"][1].shape == (25, 3)


def test_combine_shapes_and_alignment(toy_data):
    cm = _fit(toy_data, K=3, n_iter=3, seeds=[0, 1, 2])
    cm.combine(source="W")
    # 3 runs × K=3 factors = 9 components, feature dim = 30 + 25 = 55.
    assert cm.components.shape == (9, 55)
    # Each component is L2-normalized.
    np.testing.assert_allclose(
        np.linalg.norm(cm.components, axis=1), 1.0, rtol=1e-6
    )
    # Sign-alignment: max-|value| entry of each component is positive.
    for row in cm.components:
        assert row[int(np.argmax(np.abs(row)))] > 0


def test_consensus_end_to_end(toy_data):
    cm = _fit(toy_data, K=3, n_iter=3, seeds=[0, 1, 2])
    cm.combine(source="W")
    cm.cluster(density_threshold=np.inf)
    out = cm.consensus()
    assert out["Z"].shape == (40, 3)
    assert all(W.shape[1] == 3 for W in out["W"])
    assert out["W"][0].shape == (30, 3)
    assert out["W"][1].shape == (25, 3)
    assert out["stability"].shape == (3,)
    # On clean synthetic data the consensus should be highly stable.
    assert float(np.mean(out["stability"])) > 0.8
    # All clusters should absorb all 3 runs (no filtering).
    assert np.all(out["n_used"] == 3)


def test_managed_option_override_warns(toy_data):
    with pytest.warns(UserWarning, match="ard_factors"):
        ConsensusMOFA(
            data=toy_data,
            K=3,
            model_options={"ard_factors": True},
        )
    with pytest.warns(UserWarning, match="dropR2"):
        ConsensusMOFA(
            data=toy_data,
            K=3,
            train_options={"dropR2": 0.01},
        )


def test_density_threshold_too_strict(toy_data):
    cm = _fit(toy_data, K=3, n_iter=3, seeds=[0, 1, 2])
    cm.combine(source="W")
    with pytest.raises(ValueError, match="density_threshold"):
        cm.cluster(density_threshold=-1.0)


def test_save_roundtrip(toy_data, tmp_path):
    import h5py

    cm = _fit(toy_data, K=3, n_iter=3, seeds=[0, 1, 2])
    cm.combine(source="W")
    cm.cluster(density_threshold=np.inf)
    out = cm.consensus()

    outfile = tmp_path / "consensus.hdf5"
    cm.save(str(outfile))

    with h5py.File(outfile, "r") as f:
        np.testing.assert_array_equal(f["consensus/Z"][()], out["Z"])
        np.testing.assert_array_equal(
            f["consensus/W/view0"][()], out["W"][0]
        )
        np.testing.assert_array_equal(
            f["consensus/W/view1"][()], out["W"][1]
        )
        assert f["params"].attrs["K"] == 3
        assert f["params"].attrs["n_iter"] == 3
        assert len(f["runs"]) == 3


def test_k_selection_sweep(toy_data):
    cm = ConsensusMOFA(
        data=toy_data,
        K=3,  # placeholder; k_selection overrides
        train_options={"iter": 100, "convergence_mode": "fast"},
    )
    df = cm.k_selection(
        K_range=[2, 3], n_iter=2, seeds=[0, 1], verbose=False
    )
    assert list(df["K"]) == [2, 3]
    assert "mean_stability" in df.columns
    assert "min_stability" in df.columns
    # K should be restored.
    assert cm.K == 3
