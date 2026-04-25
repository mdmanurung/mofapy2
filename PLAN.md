# Evaluation: ConsensusMOFA wrapper

## Context

The fork at `mdmanurung/mofapy2` adds one feature on top of upstream `bioFAM/mofapy2@f4e4978`: a cNMF-style consensus wrapper at `mofapy2/run/consensus.py` (~700 lines), exported as `mofapy2.run.ConsensusMOFA`. The wrapper trains MOFA2 multiple times from different seeds, sign-aligns and L2-normalizes the resulting factors, filters outlier components by local density, clusters survivors with KMeans into K groups, takes the element-wise median per cluster as a consensus factor, and optionally refits Z by closed-form least squares against the consensus W.

The user asked for an evaluation against the original cNMF algorithm (Kotliar et al., 2019) and against the broader MOFA2 feature set, identifying anything missing. Per the user's direction, this is an evaluation report only — **no code changes are proposed**.

## What is implemented

Files added on top of upstream:

- `mofapy2/run/consensus.py` — `ConsensusMOFA` class (factorize → combine → cluster → consensus → refit → k_selection → save → diagnostic plots)
- `mofapy2/run/__init__.py` — re-exports `ConsensusMOFA`
- `scripts/run_consensus_mofa.py` — single-group, three-view example
- `tests/test_consensus.py` — 10 pytest tests (Gaussian, single-group, K=3 only)

Pipeline (numbered to match the file):

1. `factorize(n_iter, seeds)` — runs serial fits with `ard_factors=False`, `dropR2=None`, force-pruning off (`mofapy2/run/consensus.py:161-217`).
2. `combine(source="W"|"Z"|"both")` — concatenates per-view loadings, sign-flips so the max-|·| entry is positive, L2-normalizes (`consensus.py:222-270`).
3. `cluster(density_threshold=0.5, local_neighborhood_size=0.30, clustering="kmeans")` — pairwise Euclidean distances, k-NN local-density filter, KMeans `n_init=10, random_state=1` (`consensus.py:275-362`).
4. `consensus()` — per-cluster element-wise median; reorders factors by descending |W| mass (`consensus.py:367-433`).
5. `refit(data)` — closed-form OLS `Z = Y_c · W · pinv(WᵀW)` against feature-centered data, NaN-aware row-wise fallback (`consensus.py:438-519`).
6. `k_selection(K_range)` — sweeps K, returns DataFrame with `mean_stability`, `min_stability`, `mean_local_density` (`consensus.py:524-572`).
7. `save(outfile)` — HDF5 dump (`consensus.py:577-633`).

## cNMF reference behaviour (Kotliar et al., 2019; `dylkot/cNMF`)

Verified against `src/cnmf/cnmf.py` in the upstream repo:

- **L2-normalize spectra** before clustering. Matches.
- **Local density**: `n_neighbors = int(local_neighborhood_size * n_iter * k / k) = int(local_neighborhood_size * n_iter)`; filter is strict `<`. Default `density_threshold=0.5`. Matches the wrapper's formula and operator (`consensus.py:320, 325`).
- **Clustering**: `KMeans(n_clusters=k, n_init=10, random_state=1)`. Matches (`consensus.py:337-341`).
- **Aggregation**: per-cluster **median**. Matches (`consensus.py:393, 399`).
- **Stability metric**: `sklearn.metrics.silhouette_score(l2_spectra, labels, metric="euclidean")`. **Differs.** The wrapper invented a custom `1 − ½·d²` mapping from mean intra-cluster pairwise distance (`consensus.py:407-413`).
- **Refit**: cNMF refits **both** usage AND spectra (NNLS, holds the other fixed; `cnmf.py:798, 919, 952, 974`). The wrapper refits **only Z** (`consensus.py:438-519`).
- **K-selection plot**: silhouette score (left axis) AND Frobenius reconstruction error (right axis). The wrapper omits reconstruction error.
- **Replicate filtering by reconstruction error**: cNMF tracks per-replicate error; the wrapper does not.

## Findings

### Critical — semantic/output gaps

1. **Stability metric is non-standard.** `consensus.py:407-413` uses `1 − ½·mean(pairwise_d)²` and assigns 1.0 for singleton clusters. cNMF reports the silhouette score. Consequence: K-selection numbers are not comparable to cNMF or to other consensus-NMF tools, and lone components register as maximally stable.

2. **Multi-group support is silently broken.** MOFA's defining feature is per-group factor structure, but `_single_run()` (`consensus.py:193`) extracts `Z` as a flat `(N, K)` matrix and `refit()` (`consensus.py:475`) explicitly takes only `v[0]`. Multi-group inputs are pooled implicitly with no warning. Recommendation matching user direction: detect G>1 in `factorize()` and raise `NotImplementedError`.

3. **Non-Gaussian likelihoods are unhandled in refit.** `refit()` is a Gaussian closed-form OLS. MOFA supports Bernoulli and Poisson likelihoods (`mofapy2/run/entry_point.py`). For a Bernoulli view, regressing centered binary data on W is meaningless. Recommendation matching user direction: detect `likelihoods` containing anything other than `"gaussian"` in `refit()` and raise.

4. **Variance explained is computed and discarded.** `_single_run()` calls `m.calculate_variance_explained()` and stores `r2` in the run dict (`consensus.py:197, 215`), but it is never aggregated into the consensus result and never written by `save()`. R² is the standard MOFA diagnostic for downstream interpretation.

5. **MOFA's intercepts and precision parameters are dropped.** `save_model.py` normally persists per-view/per-group intercepts and `Tau`/`Sigma`. None of these reach the consensus output. For Gaussian refit this is internally consistent (the wrapper feature-centers in `refit()` to compensate, `consensus.py:491-496`), but the consensus model cannot be used for prediction/imputation outside this codebase without them.

### Important — robustness

6. **Zero-norm component is silently kept.** `consensus.py:254-256` skips L2 normalization when `norm == 0`, leaving an unnormalized zero vector that participates in clustering and inflates the local density of its neighbours.

7. **NaN propagation in stability.** `consensus.py:411` uses `mean()` not `nanmean()`; if the upstream distance matrix contains a NaN, stability silently becomes NaN.

8. **No replicate filtering by reconstruction quality.** cNMF discards bad replicates before pooling. The wrapper feeds every run into the cluster step, so a degenerate fit (one bad seed) contaminates the consensus.

9. **No spectra refit.** cNMF refits both usage and spectra; the wrapper refits only Z. For MOFA where W is signed-Gaussian, a closed-form spectra refit `W = Yᵀ · Z · (ZᵀZ)⁻¹` (per view) is one-liner-cheap and would round-trip the cNMF symmetry.

10. **Sign-alignment ties.** `np.argmax(np.abs(comp))` returns the first index on ties; this is deterministic but platform-sensitive and undocumented.

### Minor — packaging, ergonomics, tests

11. **`ConsensusMOFA` is not exposed at top-level.** `mofapy2/__init__.py` does not import it; users must know the submodule path.

12. **No AnnData / MuData constructor.** Upstream `entry_point` has AnnData hooks; `ConsensusMOFA.__init__` only accepts nested-list `data[m][g]`.

13. **No documentation.** No README section, no notebook, no entry in upstream docs. Discoverability is purely via docstring.

14. **Test coverage is narrow.** All tests use Gaussian, single-group, K=3, n_iter=2-3 synthetic data (`tests/test_consensus.py:9-23`). Untested: multi-group, non-Gaussian likelihoods, missing data (NaN inputs), empty clusters, density-threshold filtering that removes non-zero counts, plot smoke tests, K=1 / K>n_features edge cases.

15. **`source="both"` rebinding pattern is brittle.** `consensus.py:692-696` mutates loop-variable arrays in-place — works because of NumPy aliasing, but a naive reader would assume the assignment is local and refactor it incorrectly.

16. **Compounding metric/normalization.** Components are L2-normalized in `combine()`, then `cluster(metric="cosine")` is offered — applying cosine distance to already-unit-norm vectors equals `½·euclidean²`, so the user gets a redundant transformation without warning.

## Recommendations (prioritized, NOT implemented)

Per the user's direction, no code changes are made. If/when this is revisited, suggested ordering:

- **P1** Replace the custom stability metric with `sklearn.metrics.silhouette_score(components[kept], labels, metric="euclidean")`, and additionally report Frobenius reconstruction error on a held-out fraction or on the full data — to match cNMF and make K-selection comparable.
- **P1** Detect multi-group (G>1) in `factorize()` and raise `NotImplementedError` until proper per-group pooling is implemented.
- **P1** Detect non-Gaussian likelihoods in `refit()` and raise.
- **P2** Persist `r2` (per-run and per-consensus-factor) into the consensus result and HDF5 output.
- **P2** Add zero-norm guard + `nanmean` in stability + replicate filtering by per-run reconstruction error.
- **P2** Add a symmetric `refit_W` analogue to mirror cNMF's spectra refit.
- **P3** Expose `ConsensusMOFA` at the top level; add an AnnData/MuData constructor.
- **P3** Expand tests: multi-group rejection, Bernoulli rejection, NaN inputs, empty-cluster path.
- **P3** Document the wrapper in README and add a usage notebook.

## Verification

This is an evaluation, not an implementation. The findings can be independently verified by:

- Reading the cited line ranges in `mofapy2/run/consensus.py` and the cNMF source at `dylkot/cNMF/src/cnmf/cnmf.py` (lines 798, 879, 882, 903, 908, 913, 919, 923, 952, 974).
- Running the existing test suite with `pytest tests/test_consensus.py` to confirm coverage is single-group / Gaussian only.
- Constructing a synthetic two-group, two-likelihood dataset and observing that `factorize()` runs without warning and `refit()` silently drops groups (reproduces findings 2, 3).
