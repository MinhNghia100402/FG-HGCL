"""Hypergraph construction (numpy) and the HGNN propagator.

Adapted from IHC-LDA @ 9c88513: scripts/phase22_diag.py (knn_hyperedges, propagator) and
src/models/hgcmlda_parts.py (gmm_incidence).

Protocol A (strict inductive, `inductive_propagator`): pool rows of G only see hyperedges built from pool nodes, so a
non-pool node can never change a pool node's embedding (IHC-LDA's phase22 propagator let the hyperedge e_v of a cold
node v also reach its train neighbours, which is Protocol B). A non-pool node reads only from the pool members of the
hyperedges it belongs to, plus itself.
Protocol B (feature-transductive, `propagator`): plain HGNN over every hyperedge, unlabeled test features included.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def cosine(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def knn_hyperedges(sim: np.ndarray, k: int, pool: np.ndarray, has: np.ndarray) -> np.ndarray:
    """Incidence H (N x N): column v = e_v = {v} U k most similar pool nodes (only for nodes with content)."""
    N = sim.shape[0]
    s = np.array(sim, np.float64, copy=True)
    s[:, ~(pool & has)] = -np.inf
    np.fill_diagonal(s, -np.inf)
    H = np.zeros((N, N), np.float32)
    H[np.arange(N), np.arange(N)] = 1.0
    k = min(k, int((pool & has).sum()) - 1)
    if k > 0:
        nb = np.argpartition(-s, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(N), k)
        ok = has[rows]
        H[nb.ravel()[ok], rows[ok]] = 1.0
    return H


def gmm_hyperedges(X: np.ndarray, g: int, pool: np.ndarray, has: np.ndarray, seed: int, pca: int = 32) -> np.ndarray:
    """Incidence H (N x g): one hyperedge per GMM component. PCA and GMM are fit on pool nodes with content only;
    every node with content is then assigned to its most likely component; nodes without content join no hyperedge."""
    fit = pool & has
    Z = X
    if pca and X.shape[1] > pca:
        Z = PCA(min(pca, int(fit.sum()) - 1), random_state=seed).fit(X[fit]).transform(X)
    try:
        gm = GaussianMixture(g, covariance_type="diag", reg_covar=1e-4, random_state=seed).fit(Z[fit])
    except ValueError:      # collapsed rows (e.g. identical GIPK rows in the leaky comparison): float64, stronger reg
        Z = np.asarray(Z, np.float64)
        gm = GaussianMixture(g, covariance_type="diag", reg_covar=1e-2, random_state=seed).fit(Z[fit])
    lab = gm.predict(Z)
    H = np.zeros((X.shape[0], g), np.float32)
    H[np.nonzero(has)[0], lab[has]] = 1.0
    return H[:, H.sum(0) > 0]


def propagator(H: np.ndarray) -> np.ndarray:
    """G = Dv^-1/2 H De^-1 H^T Dv^-1/2 (unit hyperedge weights); rows of nodes in no hyperedge are zero."""
    dv, de = H.sum(1), H.sum(0)
    dvi = np.where(dv > 0, np.maximum(dv, 1e-12) ** -0.5, 0)
    dei = np.where(de > 0, 1.0 / np.maximum(de, 1e-12), 0)
    Hn = dvi[:, None] * H
    return ((Hn * dei[None, :]) @ Hn.T).astype(np.float32)


def inductive_propagator(H: np.ndarray, pool: np.ndarray, col_ok: np.ndarray) -> np.ndarray:
    """Protocol A propagator.
    Pool rows: HGNN over the hyperedges in `col_ok` (hyperedges defined by pool nodes: KNN columns centred at pool
    nodes, GMM components fit on pool nodes) restricted to their pool members.
    Non-pool row v: mean over the hyperedges containing v (any column) of the mean over {v} U (pool members)."""
    N = H.shape[0]
    Hp = H[:, col_ok] * pool[:, None]
    G = propagator(Hp[:, Hp.sum(0) > 0])
    G[~pool] = 0.0
    for v in np.nonzero(~pool)[0]:
        E = np.nonzero(H[v] > 0)[0]
        if len(E) == 0:
            G[v, v] = 1.0
            continue
        for e in E:
            mem = np.nonzero((H[:, e] > 0) & pool)[0]
            mem = np.append(mem, v)
            G[v, mem] += 1.0 / (len(E) * len(mem))
    return G


def build_propagator(H: np.ndarray, pool: np.ndarray, col_ok: np.ndarray, protocol: str) -> np.ndarray:
    return inductive_propagator(H, pool, col_ok) if protocol == "A" else propagator(H)
