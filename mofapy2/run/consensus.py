"""
Consensus MOFA2
===============

Multi-run consensus matrix factorization for MOFA2, inspired by cNMF
(Kotliar et al., 2019; https://github.com/dylkot/cNMF).

The :class:`ConsensusMOFA` wrapper trains MOFA2 several times from different
random seeds, aligns and pools the resulting factors, filters outlier
components by local density, clusters the survivors, and takes the
element-wise median of each cluster as a consensus factor / loading.

Key differences from cNMF:

* MOFA factors are sign-invariant, so each component is sign-aligned
  (max-abs-entry positive) before clustering, and the same flip is
  applied to ``Z`` and every view's ``W``.
* MOFA is multi-view: loadings are stacked feature-wise across views to
  form the clustering target (analogous to cNMF gene spectra).
* The wrapper force-disables factor pruning (``ard_factors=False`` and
  ``dropR2=None``) so every run returns exactly ``K`` factors; varying K
  is done by calling :meth:`k_selection` over a range of K values.
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from mofapy2.run.entry_point import entry_point


class ConsensusMOFA:
    """Consensus wrapper around :class:`mofapy2.run.entry_point.entry_point`.

    Parameters
    ----------
    data:
        Nested list ``data[m][g]`` of (samples, features) arrays, exactly as
        accepted by :meth:`entry_point.set_data_matrix`. A single 2D array
        or a 1-level list is also accepted and wrapped as ``M = G = 1``.
    K:
        Number of factors. Fixed across all runs (no ARD pruning).
    data_options, model_options, train_options:
        Keyword dictionaries forwarded to the corresponding
        ``entry_point.set_*_options`` methods. ``factors`` in
        ``model_options`` and ``seed`` / ``ard_factors`` / ``dropR2``
        are managed by the wrapper and will be overridden with a
        warning if supplied.
    likelihoods, views_names, groups_names, samples_names, features_names:
        Forwarded to :meth:`entry_point.set_data_matrix`.
    """

    _MANAGED_MODEL_OPTS = ("factors", "ard_factors")
    _MANAGED_TRAIN_OPTS = ("seed", "dropR2")

    def __init__(
        self,
        data,
        K: int = 10,
        *,
        data_options: Optional[dict] = None,
        model_options: Optional[dict] = None,
        train_options: Optional[dict] = None,
        likelihoods: Optional[Sequence[str]] = None,
        views_names: Optional[Sequence[str]] = None,
        groups_names: Optional[Sequence[str]] = None,
        samples_names=None,
        features_names=None,
    ):
        self.data = data
        self.K = int(K)
        self.data_options = dict(data_options or {})
        self.model_options = dict(model_options or {})
        self.train_options = dict(train_options or {})
        self.likelihoods = likelihoods
        self.views_names = views_names
        self.groups_names = groups_names
        self.samples_names = samples_names
        self.features_names = features_names

        # Strip and warn about managed options supplied by the user.
        for opt in self._MANAGED_MODEL_OPTS:
            if opt in self.model_options:
                warnings.warn(
                    f"ConsensusMOFA manages model_options['{opt}']; "
                    f"the supplied value is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                del self.model_options[opt]
        for opt in self._MANAGED_TRAIN_OPTS:
            if opt in self.train_options:
                warnings.warn(
                    f"ConsensusMOFA manages train_options['{opt}']; "
                    f"the supplied value is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                del self.train_options[opt]

        # Populated by factorize() / combine() / cluster() / consensus().
        self.runs: List[Dict] = []
        self.runs_aligned: List[Dict] = []
        self.components: Optional[np.ndarray] = None
        self.component_index: Optional[np.ndarray] = None  # (n_runs*K, 2)
        self.distance_matrix: Optional[np.ndarray] = None
        self.local_density: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.kept_index: Optional[np.ndarray] = None
        self.consensus_result: Optional[Dict] = None
        self._source: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Step 1 — multiple independent MOFA fits                            #
    # ------------------------------------------------------------------ #
    def factorize(
        self,
        n_iter: int = 20,
        seeds: Optional[Iterable[int]] = None,
        verbose: bool = True,
    ) -> "ConsensusMOFA":
        """Train ``n_iter`` independent MOFA models with different seeds.

        Runs are executed serially. Each run starts from a fresh
        :class:`entry_point`, so there is no state leak between runs.
        Results are stored in :attr:`runs`.
        """
        if seeds is None:
            rng = np.random.default_rng()
            seeds = rng.integers(0, 2**31 - 1, size=n_iter).tolist()
        else:
            seeds = [int(s) for s in seeds]
            if len(seeds) != n_iter:
                raise ValueError(
                    f"seeds has length {len(seeds)} but n_iter={n_iter}."
                )

        self.runs = []
        for i, seed in enumerate(seeds):
            if verbose:
                print(
                    f"[ConsensusMOFA] run {i + 1}/{n_iter} (seed={seed}, K={self.K})"
                )
            self.runs.append(self._single_run(seed))

        # Invalidate downstream caches.
        self.runs_aligned = []
        self.components = None
        self.component_index = None
        self.distance_matrix = None
        self.local_density = None
        self.labels = None
        self.kept_index = None
        self.consensus_result = None
        return self

    def _single_run(self, seed: int) -> Dict:
        ep = entry_point()
        ep.set_data_options(**self.data_options)
        # Deep-copy the data: entry_point.set_data_matrix casts in place and
        # training may center/mutate the arrays, so each run needs its own.
        data_copy = copy.deepcopy(self.data)
        ep.set_data_matrix(
            data_copy,
            likelihoods=self.likelihoods,
            views_names=self.views_names,
            groups_names=self.groups_names,
            samples_names=self.samples_names,
            features_names=self.features_names,
        )

        # Force pruning off so every run returns exactly K factors.
        model_opts = dict(self.model_options)
        model_opts["factors"] = self.K
        model_opts["ard_factors"] = False
        ep.set_model_options(**model_opts)

        train_opts = dict(self.train_options)
        train_opts["seed"] = int(seed)
        train_opts["dropR2"] = None
        if "quiet" not in train_opts:
            train_opts["quiet"] = True
        ep.set_train_options(**train_opts)

        ep.build()
        ep.run()

        m = ep.model
        Z = np.asarray(m.nodes["Z"].getExpectation())  # (N, K)
        W = [
            np.asarray(w["E"]) for w in m.nodes["W"].getExpectations()
        ]  # M × (D_m, K)
        r2 = m.calculate_variance_explained()
        elbo = None
        if hasattr(m, "train_stats") and isinstance(m.train_stats, dict):
            e = m.train_stats.get("elbo")
            if e is not None and len(e) > 0:
                elbo = float(np.asarray(e)[-1])

        if Z.shape[1] != self.K or not all(w.shape[1] == self.K for w in W):
            raise RuntimeError(
                "ConsensusMOFA: run returned K="
                f"{Z.shape[1]} (expected {self.K}). Factor pruning was "
                "not fully disabled."
            )

        return {
            "seed": int(seed),
            "Z": Z,
            "W": W,
            "r2": r2,
            "elbo": elbo,
        }

    # ------------------------------------------------------------------ #
    # Step 2 — pool components across runs with sign alignment           #
    # ------------------------------------------------------------------ #
    def combine(self, source: str = "W") -> "ConsensusMOFA":
        """Pool per-run factors into a single component matrix.

        Parameters
        ----------
        source:
            ``"W"`` (default) stacks loadings across views feature-wise —
            analogous to cNMF gene spectra. ``"Z"`` uses factor scores.
            ``"both"`` L2-normalizes each separately and concatenates.
        """
        if not self.runs:
            raise RuntimeError("Call factorize() before combine().")
        if source not in {"W", "Z", "both"}:
            raise ValueError("source must be one of 'W', 'Z', 'both'.")
        self._source = source

        aligned = []
        comps = []
        index = []
        for r, run in enumerate(self.runs):
            Z = run["Z"].copy()
            W_list = [w.copy() for w in run["W"]]
            for k in range(self.K):
                comp = _build_component(Z, W_list, k, source)
                # Sign-align: flip so max-|value| entry is positive.
                imax = int(np.argmax(np.abs(comp)))
                if comp[imax] < 0:
                    comp = -comp
                    Z[:, k] = -Z[:, k]
                    for m in range(len(W_list)):
                        W_list[m][:, k] = -W_list[m][:, k]
                # L2 normalize.
                norm = np.linalg.norm(comp)
                if norm > 0:
                    comp = comp / norm
                comps.append(comp)
                index.append((r, k))
            aligned.append({**run, "Z": Z, "W": W_list})

        self.runs_aligned = aligned
        self.components = np.stack(comps, axis=0)
        self.component_index = np.asarray(index, dtype=int)
        # Invalidate later steps.
        self.distance_matrix = None
        self.local_density = None
        self.labels = None
        self.kept_index = None
        self.consensus_result = None
        return self

    # ------------------------------------------------------------------ #
    # Step 3 — cluster components with local-density filtering           #
    # ------------------------------------------------------------------ #
    def cluster(
        self,
        density_threshold: float = 0.5,
        local_neighborhood_size: float = 0.30,
        linkage: str = "average",
        metric: str = "cosine",
    ) -> "ConsensusMOFA":
        """Cluster pooled components into K consensus groups.

        Uses hierarchical clustering with the given ``linkage`` on a
        precomputed ``metric`` distance matrix. Components whose mean
        distance to their local neighborhood exceeds
        ``density_threshold`` are filtered out as outliers.
        """
        if self.components is None:
            raise RuntimeError("Call combine() before cluster().")
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(self.components, metric=metric)
        n_runs = len(self.runs)
        k_nn = max(1, int(round(local_neighborhood_size * n_runs)))
        # Exclude self-distance at index 0.
        nearest = np.sort(D, axis=1)[:, 1 : k_nn + 1]
        local_density = nearest.mean(axis=1)

        keep = local_density < density_threshold
        if keep.sum() < self.K:
            raise ValueError(
                f"density_threshold={density_threshold} is too strict: only "
                f"{int(keep.sum())} components survive, but K={self.K} are "
                "needed. Try raising the threshold or inspect "
                "plot_local_density()."
            )

        D_keep = D[np.ix_(keep, keep)]
        hc = AgglomerativeClustering(
            n_clusters=self.K,
            metric="precomputed",
            linkage=linkage,
        )
        labels = hc.fit_predict(D_keep)

        self.distance_matrix = D
        self.local_density = local_density
        self.labels = labels
        self.kept_index = np.where(keep)[0]
        self.consensus_result = None
        return self

    # ------------------------------------------------------------------ #
    # Step 4 — element-wise median consensus per cluster                 #
    # ------------------------------------------------------------------ #
    def consensus(self) -> Dict:
        """Compute median consensus ``Z`` and per-view ``W``."""
        if self.labels is None:
            raise RuntimeError("Call cluster() before consensus().")
        if not self.runs_aligned:
            raise RuntimeError("runs_aligned is empty; call combine() first.")

        N = self.runs_aligned[0]["Z"].shape[0]
        M = len(self.runs_aligned[0]["W"])
        D_m = [self.runs_aligned[0]["W"][m].shape[0] for m in range(M)]

        Z_out = np.zeros((N, self.K))
        W_out = [np.zeros((D_m[m], self.K)) for m in range(M)]
        stability = np.zeros(self.K)
        n_used = np.zeros(self.K, dtype=int)

        for c in range(self.K):
            member_idx = self.kept_index[self.labels == c]
            members = self.component_index[member_idx]  # (n_c, 2) of (run, k)
            n_used[c] = len(members)
            if n_used[c] == 0:
                continue

            Z_stack = np.stack(
                [self.runs_aligned[r]["Z"][:, k] for r, k in members], axis=0
            )
            Z_out[:, c] = np.median(Z_stack, axis=0)
            for m in range(M):
                W_stack = np.stack(
                    [self.runs_aligned[r]["W"][m][:, k] for r, k in members],
                    axis=0,
                )
                W_out[m][:, c] = np.median(W_stack, axis=0)

            # Per-cluster stability = 1 - mean pairwise cosine distance.
            if self.distance_matrix is not None and n_used[c] > 1:
                sub = self.distance_matrix[np.ix_(member_idx, member_idx)]
                iu = np.triu_indices(n_used[c], k=1)
                stability[c] = 1.0 - float(sub[iu].mean())
            else:
                stability[c] = 1.0

        # Reorder factors by descending total |W| mass (mirrors
        # mofapy2/build_model/save_model.py:114-122).
        mass = np.zeros(self.K)
        for m in range(M):
            mass += np.abs(W_out[m]).sum(axis=0)
        order = np.argsort(mass)[::-1]
        Z_out = Z_out[:, order]
        W_out = [W[:, order] for W in W_out]
        stability = stability[order]
        n_used = n_used[order]

        self.consensus_result = {
            "Z": Z_out,
            "W": W_out,
            "stability": stability,
            "n_used": n_used,
            "order": order,
        }
        return self.consensus_result

    # ------------------------------------------------------------------ #
    # K selection diagnostic                                             #
    # ------------------------------------------------------------------ #
    def k_selection(
        self,
        K_range: Iterable[int],
        n_iter: int = 10,
        seeds: Optional[Iterable[int]] = None,
        source: str = "W",
        local_neighborhood_size: float = 0.30,
        verbose: bool = True,
    ):
        """Sweep K and report consensus stability for each.

        Returns a :class:`pandas.DataFrame` with one row per K containing
        ``mean_stability``, ``min_stability``, and ``mean_local_density``.
        Leaves :attr:`K` unchanged.
        """
        import pandas as pd

        original_K = self.K
        rows = []
        base_seeds = None if seeds is None else list(seeds)
        try:
            for K_val in K_range:
                self.K = int(K_val)
                if verbose:
                    print(f"[ConsensusMOFA.k_selection] K={K_val}")
                self.factorize(
                    n_iter=n_iter,
                    seeds=base_seeds,
                    verbose=False,
                )
                self.combine(source=source)
                self.cluster(
                    density_threshold=np.inf,
                    local_neighborhood_size=local_neighborhood_size,
                )
                out = self.consensus()
                rows.append(
                    {
                        "K": int(K_val),
                        "mean_stability": float(np.mean(out["stability"])),
                        "min_stability": float(np.min(out["stability"])),
                        "mean_local_density": float(
                            np.mean(self.local_density)
                        ),
                    }
                )
        finally:
            self.K = original_K
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # IO                                                                 #
    # ------------------------------------------------------------------ #
    def save(self, outfile: str) -> None:
        """Persist runs, clustering, and consensus to an HDF5 file."""
        import h5py

        if self.consensus_result is None:
            raise RuntimeError("Nothing to save: run consensus() first.")

        with h5py.File(outfile, "w") as f:
            g_cons = f.create_group("consensus")
            g_cons.create_dataset("Z", data=self.consensus_result["Z"])
            g_cons.create_dataset(
                "stability", data=self.consensus_result["stability"]
            )
            g_cons.create_dataset(
                "n_used", data=self.consensus_result["n_used"]
            )
            g_cons_W = g_cons.create_group("W")
            for m, W in enumerate(self.consensus_result["W"]):
                name = (
                    self.views_names[m]
                    if self.views_names is not None
                    else f"view{m}"
                )
                g_cons_W.create_dataset(str(name), data=W)

            g_runs = f.create_group("runs")
            for i, run in enumerate(self.runs):
                g = g_runs.create_group(str(i))
                g.attrs["seed"] = run["seed"]
                if run["elbo"] is not None:
                    g.attrs["elbo"] = run["elbo"]
                g.create_dataset("Z", data=run["Z"])
                g_W = g.create_group("W")
                for m, W in enumerate(run["W"]):
                    name = (
                        self.views_names[m]
                        if self.views_names is not None
                        else f"view{m}"
                    )
                    g_W.create_dataset(str(name), data=W)

            g_cl = f.create_group("clustering")
            if self.labels is not None:
                g_cl.create_dataset("labels", data=self.labels)
            if self.local_density is not None:
                g_cl.create_dataset("local_density", data=self.local_density)
            if self.kept_index is not None:
                g_cl.create_dataset("kept_index", data=self.kept_index)
            if self.distance_matrix is not None:
                g_cl.create_dataset(
                    "distance_matrix", data=self.distance_matrix
                )

            g_p = f.create_group("params")
            g_p.attrs["K"] = self.K
            g_p.attrs["n_iter"] = len(self.runs)
            g_p.attrs["source"] = self._source or ""

    # ------------------------------------------------------------------ #
    # Diagnostic plots                                                   #
    # ------------------------------------------------------------------ #
    def plot_local_density(self, ax=None):
        """Histogram of per-component local density (for picking threshold)."""
        import matplotlib.pyplot as plt

        if self.local_density is None:
            raise RuntimeError("Call cluster() before plot_local_density().")
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(self.local_density, bins=30)
        ax.set_xlabel("Mean distance to local neighborhood")
        ax.set_ylabel("# components")
        ax.set_title("Local density of pooled components")
        return ax

    def plot_k_selection(self, df, ax=None):
        """Plot stability and local density vs K from k_selection()."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(df["K"], df["mean_stability"], "o-", label="mean stability")
        ax.plot(df["K"], df["min_stability"], "s--", label="min stability")
        ax.set_xlabel("K")
        ax.set_ylabel("stability (1 - cosine distance)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.set_title("Consensus MOFA — K selection")
        return ax

    def plot_clustergram(self, ax=None):
        """Heatmap of the pairwise component distance matrix."""
        import matplotlib.pyplot as plt

        if self.distance_matrix is None:
            raise RuntimeError("Call cluster() before plot_clustergram().")
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(self.distance_matrix, aspect="auto", cmap="viridis")
        ax.set_title("Pooled component distance matrix")
        ax.figure.colorbar(im, ax=ax)
        return ax


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _build_component(Z: np.ndarray, W_list: Sequence[np.ndarray], k: int, source: str) -> np.ndarray:
    """Return the length-F vector representing factor k for clustering."""
    if source == "W":
        return np.concatenate([W[:, k] for W in W_list])
    if source == "Z":
        return Z[:, k].copy()
    # "both": L2-normalize each separately then concatenate.
    w_vec = np.concatenate([W[:, k] for W in W_list]).astype(float)
    z_vec = Z[:, k].astype(float).copy()
    for v in (w_vec, z_vec):
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
    return np.concatenate([w_vec, z_vec])
