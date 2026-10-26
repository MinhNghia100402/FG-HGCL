"""DELIBERATELY LEAKY similarities, reproducing HGCMLDA's original protocol for a side-by-side comparison only.

HGCMLDA (prepareData.py) computes GIPK on the FULL lncRNA-disease matrix (test edges included) and uses Chen LFS (built
from each lncRNA's disease set), integrated as IL = LFS if non-zero else GIPK, ID = DSS if non-zero else GIPK. Under a
both-cold split the rows of test nodes then encode their own test associations. Results produced with this module must
be reported as "leaky (HGCMLDA protocol)", never as both-cold results.

gipk / chen_lfs / integrate: copied from IHC-LDA @ 9c88513 src/baselines/hgcmlda_similarity.py (verified there against
the original prepareData.py and the released LFS.txt).
"""
from __future__ import annotations

import numpy as np


def gipk(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    n = a.shape[0]
    denom = (a * a).sum()
    gamma = n / denom if denom > 0 else 0.0
    sq = (a * a).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * a @ a.T, 0.0)
    return np.exp(-gamma * d2)


def chen_lfs(ld: np.ndarray, dss: np.ndarray) -> np.ndarray:
    a = (np.asarray(ld) > 0).astype(np.float64)
    s = np.asarray(dss, dtype=np.float64)
    deg = a.sum(1)
    mx = np.zeros((a.shape[0], s.shape[0]))
    for i in np.nonzero(deg)[0]:
        mx[i] = s[a[i] > 0].max(0)
    s1 = mx @ a.T
    num = s1 + s1.T
    den = deg[:, None] + deg[None, :]
    out = np.zeros_like(num)
    both = (deg[:, None] > 0) & (deg[None, :] > 0)
    out[both] = num[both] / den[both]
    return out


def integrate(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    return np.where(primary == 0, fallback, primary)


def leaky_similarities(M_full: np.ndarray, dss: np.ndarray):
    """(IL, ID) from the FULL association matrix, exactly as HGCMLDA's prepareData.py."""
    A = (np.asarray(M_full) > 0).astype(np.float64)
    if dss is None:                                  # no disease similarity available (unit tests): GIPK only
        return gipk(A).astype(np.float32), gipk(A.T).astype(np.float32)
    IL = integrate(chen_lfs(A, dss), gipk(A))
    ID = integrate(np.asarray(dss, dtype=np.float64), gipk(A.T))
    return IL.astype(np.float32), ID.astype(np.float32)
