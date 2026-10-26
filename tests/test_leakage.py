"""Leakage checks for the both-cold path (run: /sda1/nghiavm/research/.venv/bin/python -m pytest -q)."""
import numpy as np
import pytest

from fghgcl.harness import FGHGCLHarness, inner_split
from fghgcl.hypergraph import gmm_hyperedges, inductive_propagator, knn_hyperedges, propagator, cosine
from fghgcl.model import FGHGCL, Config

rng = np.random.default_rng(0)
L, D = 60, 25
Xl = rng.normal(size=(L, 16)).astype(np.float32)
Xl[:5] = 0                                   # lncRNAs without content
Xd = rng.normal(size=(D, 12)).astype(np.float32)
DSS = cosine(Xd)
M = rng.random((L, D)) < 0.15
TL, TD = np.arange(L) >= 15, np.arange(D) >= 6     # cold = first 15 lnc x first 6 dis


def test_knn_members_are_pool_nodes():
    has = np.any(Xl != 0, axis=1)
    H = knn_hyperedges(cosine(Xl), 4, TL, has)
    off = H.copy()
    off[np.arange(L), np.arange(L)] = 0
    assert not off[~TL].any(), "a non-pool node is a member of some hyperedge other than its own"


def test_inductive_pool_rows_ignore_non_pool():
    has = np.any(Xl != 0, axis=1)
    H = np.concatenate([knn_hyperedges(cosine(Xl), 4, TL, has), gmm_hyperedges(Xl, 3, TL, has, 0, pca=0)], 1)
    ok = np.r_[TL, np.ones(H.shape[1] - L, bool)]
    G = inductive_propagator(H, TL, ok)
    assert not G[np.ix_(TL, ~TL)].any()
    # changing non-pool features must not change pool rows
    X2 = Xl.copy()
    X2[~TL & has] = rng.normal(size=((~TL & has).sum(), 16))
    H2 = np.concatenate([knn_hyperedges(cosine(X2), 4, TL, has), gmm_hyperedges(X2, 3, TL, has, 0, pca=0)], 1)
    G2 = inductive_propagator(H2, TL, ok)
    assert np.allclose(G[TL], G2[TL])
    # protocol B is allowed to differ (sanity: plain propagator mixes pool and non-pool)
    assert propagator(H)[np.ix_(TL, ~TL)].any()


@pytest.mark.parametrize("views,lam,lam_a,extra", [((), 0.0, 0.0, {}), (("knn", "gmm"), 0.5, 0.0, {}), ((), 0.0, 0.5, {}),
                                                   (("knn",), 0.5, 0.5, {}), (("knn", "gmm"), 0.5, 0.0, {"aug": "sim"}),
                                                   (("gmm",), 0.5, 0.0, {"aug": "uniform", "decoder": "bilinear"}),
                                                   (("knn",), 0.0, 0.5, {"hg_place": "inner", "hg_zscore": True, "fusion": "g"}),
                                                   (("knn", "gmm"), 0.5, 0.0, {"hg_place": "inner", "hg_zscore": True, "fusion": "gated", "aug": "sim"}),
                                                   (("knn", "gmm"), 0.5, 0.0, {"link_loss": "infonce", "link_k": 8}),
                                                   ((), 0.0, 0.0, {"link_loss": "infonce+bce", "link_k": 8})])
def test_scores_ignore_labels_outside_train_block(views, lam, lam_a, extra):
    cfg = Config(views=views, lam_cl=lam, lam_assoc=lam_a, **extra, epochs=30, eval_every=10, k_lnc=4, k_dis=3, g_lnc=3, g_dis=2, m=16, hidden=32)
    dev = "cpu"
    h1 = FGHGCLHarness(cfg, device=dev)
    h1.fit(M, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    M2 = M.copy()
    blk = ~(TL[:, None] & TD[None, :])
    M2[blk] = rng.random(blk.sum()) < 0.5          # scramble every label outside the train block
    h2 = FGHGCLHarness(cfg, device=dev)
    h2.fit(M2, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    assert np.allclose(h1.predict(), h2.predict(), atol=1e-5)


def test_inner_split_inside_train():
    ml, md, vr, vc = inner_split(TL, TD, "both", 1)
    assert not (vr & ~TL).any() and not (vc & ~TD).any()
    assert not (ml & vr).any() and not (md & vc).any()


def test_aug_hypergraph_matches_train_rows_of_inductive_propagator():
    """B9 augments the pool-restricted hypergraph; with nothing dropped its propagator must equal the pool rows of the
    Protocol A propagator used at inference."""
    import torch
    from fghgcl.model import propagator_t, side_structure
    has = np.any(Xl != 0, axis=1)
    cfg = Config(views=("knn", "gmm"), aug="sim", lam_cl=0.5, g_lnc=3)
    G, (Hp, P) = side_structure(Xl, [cosine(Xl)], has, TL, cfg, 4, 3, 0)
    Gp = propagator_t(torch.tensor(Hp)).numpy()
    assert np.allclose(Gp[TL], G[TL], atol=1e-5)
    idx = np.nonzero(TL)[0]                                # first KNN block: one column per pool node, in order
    assert (P[idx, np.arange(len(idx))] == 0).all() and (Hp[idx, np.arange(len(idx))] == 1).all()   # centres kept


def test_leaky_mode_really_leaks():
    """Sanity check of the HGCMLDA-protocol comparison: with leak=True, labels outside the train block DO change scores."""
    cfg = Config(views=("knn", "gmm"), lam_cl=0.5, leak=True, epochs=30, k_lnc=4, k_dis=3, g_lnc=3, g_dis=2, m=16, hidden=32)
    h1 = FGHGCLHarness(cfg, device="cpu", fixed=True).fit(M, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    M2 = M.copy()
    blk = ~(TL[:, None] & TD[None, :])
    M2[blk] = rng.random(blk.sum()) < 0.5
    h2 = FGHGCLHarness(cfg, device="cpu", fixed=True).fit(M2, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    assert not np.allclose(h1.predict(), h2.predict(), atol=1e-5)


@pytest.mark.parametrize("over", [{}, {"input": "content", "protocol": "A", "branch2": "tower", "loss": "bce", "cl": "feat_hg",
                                       "lam_cl": 0.5, "fusion": "gated"}])
def test_chain_clean_steps_ignore_labels_outside_train_block(over):
    from fghgcl.chain import ChainConfig
    cfg = ChainConfig(**over, k=4, g=2, hid=16, proj=8, tower_hidden=16, epochs=20, eval_every=10, lr=1e-3)
    h1 = FGHGCLHarness(cfg, device="cuda" if __import__("torch").cuda.is_available() else "cpu", fixed=True)
    h1.fit(M, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    M2 = M.copy()
    blk = ~(TL[:, None] & TD[None, :])
    M2[blk] = rng.random(blk.sum()) < 0.5
    h2 = FGHGCLHarness(cfg, device=h1.device, fixed=True).fit(M2, Xl, Xd, np.nonzero(TL)[0], np.nonzero(TD)[0])
    assert np.allclose(h1.predict(), h2.predict(), atol=1e-4)
