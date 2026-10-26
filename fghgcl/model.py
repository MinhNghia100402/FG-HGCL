"""FG-HGCL model: CDELDA-style feature towers (h) + hypergraph encoder on h (g) + fusion (z) + inner-product decoder.

Loss = class-balanced BCE over the train lncRNA x train disease block (CDELDA recipe)
     + lam_cl * feature<->hypergraph InfoNCE (same node = positive, same-type negatives, train nodes only).

Ablation IDs of the research plan map to config flags:
  B0-like feature only : views=()                       (no hypergraph, fusion ignored)
  B2  BCE only         : views=("gmm",) / ("knn",) / ("knn","gmm"), lam_cl=0
  B3  + InfoNCE        : lam_cl > 0
  B7/B8 fusion         : fusion in {"residual", "gated", "concat"}
  B9  augmentation     : aug in {"uniform", "sim"}, aug_p > 0 (with lam_cl > 0). Each epoch, hyperedge memberships among
                         train nodes are dropped (never a KNN hyperedge's own centre); "sim" drops a member with probability
                         proportional to its dissimilarity to the hyperedge centroid (similarity-guided), "uniform" with
                         aug_p. The InfoNCE contrasts h with the hypergraph embedding on the augmented hypergraph.
  Encoder placement    : hg_place = "post" (HGCN on the tower output h, default) | "inner" (IHC-LDA v22 HGTower: the
                         hypergraph layer acts on the 256-d hidden layer of its own tower over the raw content);
                         hg_zscore = z-score the hypergraph branch input with train-node statistics (v22).
                         fusion = "g" scores with g only (v22: no separate feature MLP in the score).
  Decoder              : decoder = "dot" (CDELDA, default) | "bilinear" (z_l^T W z_d, W initialised to I; separate ablation).
  C2  association view : lam_assoc > 0. Association hypergraph from TRAIN edges only (each train disease = a hyperedge
                         over its train lncRNAs and the dual); a 2-step HGNN on free node embeddings (= 2-layer LightGCN,
                         IHC-LDA test_lightgcn_two_hop_equals_hgnn_on_association_hypergraph) gives a_v for train nodes with
                         >= 1 train edge. InfoNCE(z_v, a_v) over those nodes transfers association topology into the
                         feature/content encoder; a_v is never used to score, so cold nodes are scored from z only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import build_propagator, cosine, gmm_hyperedges, knn_hyperedges


@dataclass
class Config:
    m: int = 128                    # embedding size k (CDELDA harness: TT_M = 128)
    hidden: int = 256
    drop: float = 0.2
    views: tuple = ("gmm",)         # subset of {"knn", "gmm"}; () = feature only
    k_lnc: int = 10
    k_dis: int = 10
    g_lnc: int = 16                 # GMM components
    g_dis: int = 9
    hg_layers: int = 1
    hg_place: str = "post"          # post | inner
    hg_zscore: bool = False
    protocol: str = "A"             # A = strict inductive, B = feature-transductive
    fusion: str = "residual"        # residual | gated | concat
    lam_cl: float = 0.0
    lam_assoc: float = 0.0          # C2: weight of feature <-> association-hypergraph InfoNCE
    assoc_bce: bool = True          # C2: the association view is also trained with the link loss on train pairs
    aug: str = "none"               # B9: none | uniform | sim (similarity-guided hyperedge-membership dropout)
    aug_p: float = 0.2              # B9: mean drop probability of a membership
    decoder: str = "dot"            # dot | bilinear
    tau: float = 0.2
    cl_batch: int = 1024
    lr: float = 1e-3
    wd: float = 1e-3
    epochs: int = 500
    eval_every: int = 25
    seed: int = 2026
    leak: bool = False              # LEAKY (HGCMLDA protocol, comparison only): IL/ID from the FULL matrix as extra node
                                    # features and as the KNN/GMM hypergraph basis over all nodes (fghgcl/leaky.py)
    dis_views: tuple = ()           # direction 1: extra KNN disease hyperedges from frozen-S-BioBERT text views
                                    # (subset of name / defn / syn; data/harness/data_rd_<v>/disease_views.npz)
    dis_view_feats: bool = False    # direction 1: the text views are also appended to the disease tower input
    extra_src: tuple = ()           # extra D3 train labels from D1 / D2 (data/cross; added inside the train block only)
    link_loss: str = "bce"          # bce (CDELDA) | infonce | infonce+bce: association objective
    link_k: int = 64                # infonce: sampled negatives per positive and direction (known positives masked)
    link_tau: float = 1.0           # infonce: logits = <z_l, z_d> / link_tau
    ens: int = 1                    # seed ensemble: mean sigmoid score of `ens` copies of the same model (seeds seed..seed+ens-1)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def mlp(d_in, hidden, m, drop):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, m))


class HGEncoder(nn.Module):
    """g = W_o (u_L), u_0 = h, u_{l+1} = u_l + relu(G W_l u_l)."""

    def __init__(self, m, layers):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(m, m) for _ in range(layers)])
        self.out = nn.Linear(m, m)

    def forward(self, h, G):
        u = h
        for lin in self.lins:
            u = u + torch.relu(G @ lin(u))
        return self.out(u)


class HGTower(nn.Module):
    """IHC-LDA scripts/phase22_diag.py @ 9c88513: Linear -> ReLU -> residual HGNN layer(s) on the hidden layer -> Linear."""

    def __init__(self, d_in, hidden, m, layers, drop):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.hg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, m)
        self.drop = nn.Dropout(drop)

    def forward(self, x, G):
        h = torch.relu(self.inp(x))
        for lin in self.hg:
            h = h + torch.relu(G @ lin(h))
        return self.out(self.drop(h))


class Fusion(nn.Module):
    def __init__(self, kind, m):
        super().__init__()
        self.kind = kind
        if kind == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * m, m), nn.ReLU(), nn.Linear(m, m))
            self.wg = nn.Linear(m, m, bias=False)
        elif kind == "concat":
            self.mlp = nn.Sequential(nn.Linear(2 * m, m), nn.ReLU(), nn.Linear(m, m))
        elif kind not in ("residual", "g"):
            raise ValueError(kind)
        self.norm = nn.LayerNorm(m)

    def forward(self, h, g):
        if self.kind == "residual":
            return self.norm(h + g)
        if self.kind == "g":
            return g
        if self.kind == "gated":
            r = torch.sigmoid(self.gate(torch.cat([h, g], -1)))
            return self.norm(h + r * self.wg(g))
        return self.mlp(torch.cat([h, g], -1))


def info_nce(a, b, tau):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    lg = a @ b.T / tau
    t = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t))


class Side(nn.Module):
    """One entity type: feature tower, optional hypergraph encoder, fusion, projection heads."""

    def __init__(self, d_in, cfg: Config, use_hg: bool):
        super().__init__()
        self.tower = mlp(d_in, cfg.hidden, cfg.m, cfg.drop)
        self.use_hg, self.inner = use_hg, cfg.hg_place == "inner"
        if use_hg:
            if self.inner:
                self.hgt = HGTower(d_in, cfg.hidden, cfg.m, cfg.hg_layers, cfg.drop)
            else:
                self.hg = HGEncoder(cfg.m, cfg.hg_layers)
            self.fuse = Fusion(cfg.fusion, cfg.m)
            if cfg.lam_cl > 0:
                self.qh = nn.Sequential(nn.Linear(cfg.m, cfg.m), nn.ReLU(), nn.Linear(cfg.m, cfg.m))
                self.qg = nn.Sequential(nn.Linear(cfg.m, cfg.m), nn.ReLU(), nn.Linear(cfg.m, cfg.m))
        if cfg.lam_assoc > 0:
            self.qz = nn.Sequential(nn.Linear(cfg.m, cfg.m), nn.ReLU(), nn.Linear(cfg.m, cfg.m))
            self.qa = nn.Sequential(nn.Linear(cfg.m, cfg.m), nn.ReLU(), nn.Linear(cfg.m, cfg.m))

    def encode_g(self, h, xz, G):
        return self.hgt(xz, G) if self.inner else self.hg(h, G)

    def forward(self, x, G, xz=None):
        h = self.tower(x)
        if not self.use_hg:
            return h, h, None
        g = self.encode_g(h, x if xz is None else xz, G)
        return self.fuse(h, g), h, g


def side_structure(X, sim_list, has, pool, cfg: Config, k: int, g: int, seed: int):
    """Incidence blocks of the requested views + the propagator for the chosen protocol (numpy)."""
    Hs, ok, keep = [], [], []
    N = X.shape[0]
    if "knn" in cfg.views:
        for sim in sim_list:
            Hs.append(knn_hyperedges(sim, k, pool, has))
            ok.append(pool.copy())                       # column v is centred at node v
            keep.append(np.eye(N, dtype=bool))           # augmentation never drops a hyperedge's centre
    if "gmm" in cfg.views:
        Hg = gmm_hyperedges(X, g, pool, has, seed)
        Hs.append(Hg)
        ok.append(np.ones(Hg.shape[1], bool))
        keep.append(np.zeros(Hg.shape, bool))
    if not Hs:
        return None, None
    H, col_ok, K = np.concatenate(Hs, 1), np.concatenate(ok), np.concatenate(keep, 1)
    G = build_propagator(H, pool, col_ok, cfg.protocol)
    aug = None
    if cfg.aug != "none":
        # train-time hypergraph: hyperedges defined by pool nodes, restricted to pool members (= the pool rows of G)
        Hp, Kp = H[:, col_ok] * pool[:, None], K[:, col_ok]
        nz = Hp.sum(0) > 0
        Hp, Kp = Hp[:, nz], Kp[:, nz]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        cen = (Hp.T @ Xn) / np.maximum(Hp.sum(0)[:, None], 1)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
        dis = 1.0 - Xn @ cen.T                           # member-to-centroid dissimilarity (N x E)
        if cfg.aug == "sim":
            mean = dis[Hp > 0].mean() if (Hp > 0).any() else 1.0
            P = np.clip(cfg.aug_p * dis / max(mean, 1e-12), 0.0, 0.9)
        else:
            P = np.full(Hp.shape, cfg.aug_p)
        P = np.where((Hp > 0) & ~Kp, P, 0.0)
        aug = (Hp.astype(np.float32), P.astype(np.float32))
    return G, aug


def propagator_t(H):
    dv, de = H.sum(1), H.sum(0)
    dvi = torch.where(dv > 0, dv.clamp_min(1e-12).rsqrt(), torch.zeros_like(dv))
    dei = torch.where(de > 0, 1.0 / de.clamp_min(1e-12), torch.zeros_like(de))
    Hn = dvi[:, None] * H
    return (Hn * dei[None, :]) @ Hn.T


class AssocHypergraph(nn.Module):
    """Association hypergraph view on train edges only: a = mean(e, P e, P^2 e), P = sym-normalised bipartite
    adjacency of the train block (2-step HGNN on the association hypergraph and its dual)."""

    def __init__(self, Mtr: np.ndarray, m: int, device):
        super().__init__()
        L, D = Mtr.shape
        self.L = L
        self.emb = nn.Embedding(L + D, m)
        nn.init.normal_(self.emb.weight, std=0.1)
        e = np.argwhere(Mtr)
        A = np.zeros((L + D, L + D), np.float32)
        A[e[:, 0], L + e[:, 1]] = A[L + e[:, 1], e[:, 0]] = 1.0
        dg = A.sum(1)
        di = np.where(dg > 0, np.maximum(dg, 1e-12) ** -0.5, 0)
        self.register_buffer("P", torch.tensor(di[:, None] * A * di[None, :], device=device), persistent=False)
        self.has_l = torch.as_tensor(np.nonzero(dg[:L] > 0)[0], device=device)
        self.has_d = torch.as_tensor(np.nonzero(dg[L:] > 0)[0], device=device)

    def forward(self):
        h0 = self.emb.weight
        h1 = self.P @ h0
        h = (h0 + h1 + self.P @ h1) / 3
        return h[:self.L], h[self.L:]


def load_dis_views(names, n_dis):
    """Disease text views of the current harness variant (directory of CCDIFF_DIS_SEMSIM, set by bench.sweep_run /
    fghgcl.harness.load_variant)."""
    if not names:
        return []
    import os
    z = np.load(os.path.join(os.path.dirname(os.environ["CCDIFF_DIS_SEMSIM"]), "disease_views.npz"))
    out = [z[k].astype(np.float32) for k in names]
    assert all(len(x) == n_dis for x in out)
    return out


def extra_labels(srcs, shape):
    """D1 / D2 positives mapped onto D3 pairs (scripts/cross_dataset_map.py); only defined for l2d5."""
    from pathlib import Path
    E = np.zeros(shape, bool)
    if shape != (5102, 245):
        return E
    for s in srcs:
        p = np.load(Path(__file__).resolve().parents[1] / "data" / "cross" / f"{s}_to_l2d5.npz")["pairs"]
        E[p[:, 0], p[:, 1]] = True
    return E


class FGHGCL:
    """fit on the (ml x md) block of M only; scores() returns logits for every (lncRNA, disease) pair."""

    def __init__(self, cfg: Config, device="cuda"):
        self.cfg, self.dev = cfg, device

    def _structures(self, Xl, Xd, dss, has_l, ml, md):
        c = self.cfg
        Xd = Xd[:, :self._d_base]                        # hypergraph basis: the original disease embedding
        if self._leak is not None:                       # HGCMLDA protocol: hypergraphs on IL / ID over every node
            IL, ID = self._leak
            allL, allD = np.ones(len(IL), bool), np.ones(len(ID), bool)
            Gl, self._aug_l = side_structure(IL, [IL], allL, allL, c, c.k_lnc, c.g_lnc, c.seed)
            Gd, self._aug_d = side_structure(ID, [ID], allD, allD, c, c.k_dis, c.g_dis, c.seed)
            return Gl, Gd
        Gl, self._aug_l = side_structure(Xl, [cosine(Xl)], has_l, ml, c, c.k_lnc, c.g_lnc, c.seed)
        sims_d = [s for s in (dss, cosine(Xd)) if s is not None] + [cosine(v) for v in self._views]
        Gd, self._aug_d = side_structure(Xd, sims_d, np.ones(len(Xd), bool), md, c, c.k_dis, c.g_dis, c.seed)
        return Gl, Gd

    def fit(self, M, Xl, Xd, dss, ml, md, epochs=None, val=None, leak_sims=None, train_mask=None):
        """M: bool (L, D); only M[ml][:, md] is read. val: optional callable(S) -> float, evaluated every eval_every
        epochs; returns the list of (epoch, val) and keeps the final weights."""
        c = self.cfg
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        dev = self.dev
        self._leak = leak_sims
        self._d_base = Xd.shape[1]
        self._views = load_dis_views(c.dis_views, len(Xd))
        if c.dis_view_feats and self._views:
            Xd = np.concatenate([Xd] + self._views, 1).astype(np.float32)
        if c.extra_src:                                  # extra positives, restricted to the train block (ml x md)
            M = M | (extra_labels(c.extra_src, M.shape) & ml[:, None] & md[None, :])
        if leak_sims is not None:                        # LEAKY: IL / ID rows appended to the node features
            Xl = np.concatenate([Xl, leak_sims[0]], 1).astype(np.float32)
            Xd = np.concatenate([Xd, leak_sims[1]], 1).astype(np.float32)
        has_l = np.any(Xl != 0, axis=1)
        Gl, Gd = self._structures(Xl, Xd, dss, has_l, ml, md)
        use_hg = bool(c.views)
        self.xl, self.xd = torch.tensor(Xl, device=dev), torch.tensor(Xd, device=dev)
        self.Gl = torch.tensor(Gl, device=dev) if Gl is not None else None
        self.Gd = torch.tensor(Gd, device=dev) if Gd is not None else None
        self.xlz, self.xdz = self.xl, self.xd
        if c.hg_zscore:                                  # train-node statistics; rows without content stay 0
            def zs(X, mask, has):
                mu, sd = X[mask & has].mean(0), X[mask & has].std(0) + 1e-6
                return torch.tensor(((X - mu) / sd * has[:, None]).astype(np.float32), device=dev)
            self.xlz = zs(Xl, ml, has_l)
            self.xdz = zs(Xd, md, np.ones(len(Xd), bool))
        self.L = Side(Xl.shape[1], c, use_hg).to(dev)
        self.D = Side(Xd.shape[1], c, use_hg).to(dev)
        it = torch.as_tensor(np.nonzero(ml)[0], device=dev)
        jt = torch.as_tensor(np.nonzero(md)[0], device=dev)
        Y = torch.tensor((M[np.ix_(ml, md)] > 0).astype(np.float32), device=dev)
        if train_mask is None:
            bce = nn.BCEWithLogitsLoss(pos_weight=((Y == 0).sum() / (Y.sum() + 1)).reshape(1))
        else:                                            # loss only on the given pairs (HGCMLDA protocol: train ones/zeros)
            Wm = torch.tensor(train_mask[np.ix_(ml, md)].astype(np.float32), device=dev)
            pw = ((Y == 0) * Wm).sum() / ((Y * Wm).sum() + 1)
            _b = nn.BCEWithLogitsLoss(pos_weight=pw.reshape(1), reduction="none")
            bce = lambda lg, y: (_b(lg, y) * Wm).sum() / Wm.sum()               # noqa: E731
        params = list(self.L.parameters()) + list(self.D.parameters())
        # InfoNCE link loss: positives of the train block; candidate negatives = train pairs that are not positives
        # (and, with train_mask, only pairs the protocol allows in training)
        if c.link_loss not in ("bce", "infonce", "infonce+bce"):
            raise ValueError(c.link_loss)
        if c.link_loss != "bce":
            Mb = torch.tensor(M[np.ix_(ml, md)] > 0, device=dev)
            Ab = Mb if train_mask is None else Mb & torch.tensor(train_mask[np.ix_(ml, md)], device=dev)
            neg_ok = ~Mb if train_mask is None else ~Mb & torch.tensor(train_mask[np.ix_(ml, md)], device=dev)
            pos = torch.nonzero(Ab)                      # (P, 2) indices into the train block
        self.W = None
        if c.decoder == "bilinear":
            self.W = nn.Parameter(torch.eye(c.m, device=dev))
            params.append(self.W)
        elif c.decoder != "dot":
            raise ValueError(c.decoder)
        augs = {}
        if use_hg and c.aug != "none" and c.lam_cl > 0:
            for k_, a_ in (("l", self._aug_l), ("d", self._aug_d)):
                augs[k_] = tuple(torch.tensor(x, device=dev) for x in a_)
        self.A = None
        if c.lam_assoc > 0:
            Mtr = np.zeros(M.shape, bool)
            Mtr[np.ix_(ml, md)] = M[np.ix_(ml, md)] > 0
            self.A = AssocHypergraph(Mtr, c.m, dev).to(dev)
            params += list(self.A.parameters())
        opt = torch.optim.Adam(params, lr=c.lr, weight_decay=c.wd)
        hist, self.loss_log = [], []
        for ep in range(1, (epochs or c.epochs) + 1):
            self.L.train()
            self.D.train()
            opt.zero_grad()
            zl, hl, gl = self.L(self.xl, self.Gl, self.xlz)
            zd, hd, gd = self.D(self.xd, self.Gd, self.xdz)
            if c.link_loss == "bce":
                la = bce(self._dec(zl[it], zd[jt]), Y)
            else:
                la = self._infonce(zl[it], zd[jt], pos, neg_ok)
                if c.link_loss == "infonce+bce":
                    la = la + bce(self._dec(zl[it], zd[jt]), Y)
            loss, lc = la, torch.zeros((), device=dev)
            if use_hg and c.lam_cl > 0:
                for k_, side, h, g, idx, xz in (("l", self.L, hl, gl, it, self.xlz), ("d", self.D, hd, gd, jt, self.xdz)):
                    if k_ in augs:                           # B9: hypergraph embedding on an augmented hypergraph
                        Hp, P = augs[k_]
                        g = side.encode_g(h, xz, propagator_t(Hp * (torch.rand_like(P) >= P).float()))
                    sub = idx[torch.randperm(len(idx), device=dev)[:c.cl_batch]]
                    lc = lc + 0.5 * info_nce(side.qh(h[sub]), side.qg(g[sub]), c.tau)
                loss = la + c.lam_cl * lc
            if self.A is not None:
                al, ad = self.A()
                if c.assoc_bce:
                    loss = loss + bce(al[it] @ ad[jt].T, Y)
                lx = torch.zeros((), device=dev)
                for side, z, a, idx in ((self.L, zl, al, self.A.has_l), (self.D, zd, ad, self.A.has_d)):
                    sub = idx[torch.randperm(len(idx), device=dev)[:c.cl_batch]]
                    lx = lx + 0.5 * info_nce(side.qz(z[sub]), side.qa(a[sub]), c.tau)
                loss = loss + c.lam_assoc * lx
                lc = lc + lx.detach()
            loss.backward()
            opt.step()
            if ep % c.eval_every == 0:
                self.loss_log.append((ep, la.item(), lc.item()))
                if val is not None:
                    hist.append((ep, float(val(self.scores()))))
        return hist

    @torch.no_grad()
    def embeddings(self):
        self.L.eval()
        self.D.eval()
        zl, hl, gl = self.L(self.xl, self.Gl, self.xlz)
        zd, hd, gd = self.D(self.xd, self.Gd, self.xdz)
        return {"zl": zl, "zd": zd, "hl": hl, "hd": hd, "gl": gl, "gd": gd}

    def scores(self) -> np.ndarray:
        e = self.embeddings()
        with torch.no_grad():
            return self._dec(e["zl"], e["zd"]).cpu().numpy().astype(np.float64)

    def _infonce(self, zl, zd, pos, neg_ok):
        """Two-directional sampled InfoNCE over train positives: for (l, d), the positive d competes with link_k random
        train diseases (known / disallowed pairs masked out), and symmetrically for lncRNAs."""
        c, dev = self.cfg, self.dev
        nl, nd = zl.shape[0], zd.shape[0]
        li, di = pos[:, 0], pos[:, 1]
        s_pos = (self._dec_pairs(zl[li], zd[di])) / c.link_tau
        loss = 0.0
        for side in ("d", "l"):
            n = nd if side == "d" else nl
            neg = torch.randint(0, n, (len(pos), c.link_k), device=dev)
            if side == "d":
                ok = neg_ok[li[:, None], neg]
                s_neg = self._dec_rows(zl[li], zd, neg) / c.link_tau
            else:
                ok = neg_ok[neg, di[:, None]]
                s_neg = self._dec_cols(zl, zd[di], neg) / c.link_tau
            s_neg = s_neg.masked_fill(~ok, float("-inf"))
            logits = torch.cat([s_pos[:, None], s_neg], 1)
            loss = loss + 0.5 * F.cross_entropy(logits, torch.zeros(len(pos), dtype=torch.long, device=dev))
        return loss

    def _dec_pairs(self, a, b):
        return (a * b).sum(-1) if self.W is None else ((a @ self.W) * b).sum(-1)

    def _dec_rows(self, a, zd, idx):                  # a: (P, m), zd[idx]: (P, K, m)
        a = a if self.W is None else a @ self.W
        return torch.einsum("pm,pkm->pk", a, zd[idx])

    def _dec_cols(self, zl, b, idx):                  # zl[idx]: (P, K, m), b: (P, m)
        b = b if self.W is None else b @ self.W.T
        return torch.einsum("pkm,pm->pk", zl[idx], b)

    def _dec(self, zl, zd):
        return zl @ zd.T if self.W is None else zl @ self.W @ zd.T
