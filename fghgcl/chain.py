"""HGCMLDA -> FG-HGCL module-replacement chain (thesis ablation: which HGCMLDA module fails under both-cold, what replaces it).

HGCMLDA components re-implemented from aI-area/HGCMLDA@6beede0 original/Dataset1 (Module.py, layer.py, utils.py):
  CL_HGCN (two 1-layer HGCNs, KNN / GMM view, GRACE contrast tau 0.8 through fc1-elu-fc2), MS_CAM view attention, the VAE
  over association rows, the HGCMLDA head (linear 128 / 128 -> concat 256 -> linear 64 -> relu -> dot) and the weighted
  MSE (beta 0.05) + 0.1 CL + 0.15 (KL + VAE reconstruction) objective. Differences from the release (documented):
  GMM is diagonal on a 32-d PCA (as everywhere in this repo), the VAE noise is torch.randn_like instead of numpy.

Module switches (ChainConfig), each a step of the chain:
  M1 input     : "label" (IL / ID = HGCMLDA integrated similarities) | "content" (node features) | "content+label"
                 label_src "train" (recomputed from train edges = HGCMLDA-clean) | "full" (LEAKY, HGCMLDA protocol)
  M2 protocol  : "B" (hypergraphs over every node, as HGCMLDA) | "A" (strict inductive, fghgcl/hypergraph.py)
  M6 branch2   : "vae" (VAE on association rows; cold nodes have empty rows) | "tower" (MLP on content) | "none"
  M7 objective : loss "wmse" (HGCMLDA) | "bce" (class-balanced, CDELDA); lr / epochs set per step
  M4 cl        : "grace" (KNN <-> GMM views, HGCMLDA) | "feat_hg" (tower h <-> fused hypergraph emb, FG-HGCL) |
                 "both" | "none"
  M5 fusion    : "mscam" (HGCMLDA) | "mean" | "gated" (FG-HGCL)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import build_propagator, gmm_hyperedges, knn_hyperedges
from .leaky import leaky_similarities
from .model import info_nce, mlp


@dataclass
class ChainConfig:
    input: str = "label"
    label_src: str = "train"
    protocol: str = "B"
    k: int = 11                     # HGCMLDA paper, Dataset1
    g: int = 9
    branch2: str = "vae"
    loss: str = "wmse"
    beta: float = 0.05
    cl: str = "grace"
    lam_cl: float = 0.1
    tau_grace: float = 0.8
    tau: float = 0.2
    fusion: str = "mscam"
    hid: int = 128
    proj: int = 64
    tower_hidden: int = 256
    drop: float = 0.2
    lr: float = 1e-4
    wd: float = 0.0
    epochs: int = 1550
    eval_every: int = 50
    seed: int = 2026
    ens: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def leak(self):
        return self.label_src == "full"

    def to_dict(self):
        return {"model": "chain", **asdict(self)}


# ---- HGCMLDA modules (original/Dataset1/Module.py, layer.py) -------------------------------------------------------
class MS_CAM(nn.Module):
    def __init__(self, channels=2, r=2):
        super().__init__()
        ic = int(channels // r)
        self.local_att = nn.Sequential(nn.Conv2d(channels, ic, 1), nn.BatchNorm2d(ic), nn.ReLU(inplace=True),
                                       nn.Conv2d(ic, channels, 1), nn.BatchNorm2d(channels))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, ic, 1), nn.BatchNorm2d(ic),
                                        nn.ReLU(inplace=True), nn.Conv2d(ic, channels, 1), nn.BatchNorm2d(channels))

    def forward(self, x):
        return x * torch.sigmoid(self.local_att(x) + self.global_att(x))


class HGNNConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(i, o))
        self.bias = nn.Parameter(torch.empty(o))
        s = 1.0 / np.sqrt(o)
        nn.init.uniform_(self.weight, -s, s)
        nn.init.uniform_(self.bias, -s, s)

    def forward(self, x, G):
        return G @ (x @ self.weight + self.bias)


class CLHGCN(nn.Module):
    """Two 1-layer HGCNs (KNN view, GMM view) + GRACE loss between them (HGCMLDA CL_HGCN)."""

    def __init__(self, d_in, hid, proj, tau):
        super().__init__()
        self.h1, self.h2 = HGNNConv(d_in, hid), HGNNConv(d_in, hid)
        self.fc1, self.fc2 = nn.Linear(hid, proj), nn.Linear(proj, hid)
        self.tau = tau

    def project(self, z):
        return self.fc2(F.elu(self.fc1(z)))

    def grace(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)                                   # noqa: E731
        n1, n2 = F.normalize(z1), F.normalize(z2)
        refl, betw = f(n1 @ n1.T), f(n1 @ n2.T)
        return -torch.log(betw.diag() / (refl.sum(1) + betw.sum(1) - refl.diag())).mean()

    def forward(self, x, Gk, Gg, idx):
        z1 = F.leaky_relu(self.h1(x, Gk), 0.25)
        z2 = F.leaky_relu(self.h2(x, Gg), 0.25)
        p1, p2 = self.project(z1[idx]), self.project(z2[idx])
        return z1, z2, 0.5 * self.grace(p1, p2) + 0.5 * self.grace(p2, p1)


class VAEBranch(nn.Module):
    """HGCMLDA VAE over association rows: tanh(xW + b) 192 -> (mu, log sigma) 128 -> z."""

    def __init__(self, d_in, b_scale=1.0):
        super().__init__()
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, 192), gain=1.414))
        self.b1 = nn.Parameter(torch.zeros(192))
        self.mu, self.ls = nn.Linear(192, 128), nn.Linear(192, 128)
        self.b_scale = b_scale

    def forward(self, rows):
        h = torch.tanh(rows @ self.W1 + self.b_scale * self.b1)
        mu, ls = self.mu(h), self.ls(h)
        z = mu + torch.exp(ls) * torch.randn_like(mu) if self.training else mu
        return z, mu, ls


def kl(mu, ls, n):
    return -(0.5 / n) * torch.mean(torch.sum(1 + 2 * ls - mu ** 2 - torch.exp(ls) ** 2, 1))


class ChainSide(nn.Module):
    def __init__(self, d_in, d_rows, d_tower, c: ChainConfig):
        super().__init__()
        self.c = c
        self.cl = CLHGCN(d_in, c.hid, c.proj, c.tau_grace)
        self.att = MS_CAM() if c.fusion == "mscam" else None
        if c.fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * c.hid, c.hid), nn.ReLU(), nn.Linear(c.hid, c.hid))
        self.lin2 = nn.Linear(c.hid, c.hid)                    # HGCMLDA linear_*_2 (hypergraph part)
        if c.branch2 == "vae":
            self.vae = VAEBranch(d_rows)
        elif c.branch2 == "tower":
            self.tower = mlp(d_tower, c.tower_hidden, c.hid, c.drop)        # always on the content features
        if c.branch2 != "none":                                # HGCMLDA linear_*_1 (second branch; the VAE latent is 128-d)
            self.lin1 = nn.Linear(128 if c.branch2 == "vae" else c.hid, c.hid)
        self.lin3 = nn.Linear(c.hid * (2 if c.branch2 != "none" else 1), 64)
        if c.cl in ("feat_hg", "both"):
            self.qh = nn.Sequential(nn.Linear(c.hid, c.hid), nn.ReLU(), nn.Linear(c.hid, c.hid))
            self.qg = nn.Sequential(nn.Linear(c.hid, c.hid), nn.ReLU(), nn.Linear(c.hid, c.hid))

    def forward(self, x, Gk, Gg, rows, xt, idx):
        z1, z2, lg = self.cl(x, Gk, Gg, idx)
        if self.c.fusion == "mscam":                           # HGCMLDA: (2 views) -> MS-CAM -> mean
            a = torch.stack([z1, z2], 0).unsqueeze(0).permute(2, 1, 0, 3)
            a = self.att(a).permute(2, 1, 0, 3).squeeze(0)
            g = (a[0] + a[1]) / 2
        else:
            g = (z1 + z2) / 2
        aux = {"grace": lg, "kl": None, "h": None, "g": g}
        b = None
        if self.c.branch2 == "vae":
            b, mu, ls = self.vae(rows)
            aux.update(kl=(mu, ls), vae_z=b)
        elif self.c.branch2 == "tower":
            b = self.tower(xt)
            aux["h"] = b
        if self.c.fusion == "gated" and b is not None:          # FG-HGCL gate: how much topology to take in
            r = torch.sigmoid(self.gate(torch.cat([b, g], -1)))
            g = r * g
        parts = [torch.relu(self.lin2(g))]
        if b is not None:
            parts.append(torch.relu(self.lin1(b)))
        out = self.lin3(torch.cat(parts, 1))
        return (torch.relu(out) if self.c.loss == "wmse" else out), aux


class ChainModel:
    """Same fit / scores interface as fghgcl.model.FGHGCL (harness, inner_val)."""

    def __init__(self, cfg: ChainConfig, device="cuda"):
        self.cfg, self.dev = cfg, device

    def _inputs(self, M, Xl, Xd, dss, ml, md, leak_sims):
        c = self.cfg
        if c.input == "content":
            return Xl, Xd
        if leak_sims is not None:
            IL, ID = leak_sims                                 # LEAKY: from the FULL matrix (harness, cfg.leak)
        else:
            Mtr = np.zeros(M.shape, bool)
            Mtr[np.ix_(ml, md)] = M[np.ix_(ml, md)] > 0
            IL, ID = leaky_similarities(Mtr, dss)              # same formulas on the TRAIN block only = HGCMLDA-clean
        if c.input == "label":
            return IL, ID
        return np.concatenate([Xl, IL], 1).astype(np.float32), np.concatenate([Xd, ID], 1).astype(np.float32)

    def _graphs(self, X, pool, has):
        c = self.cfg
        d2 = np.square(X).sum(1)[:, None] + np.square(X).sum(1)[None, :] - 2 * X @ X.T
        Hk = knn_hyperedges(-np.sqrt(np.maximum(d2, 0)), c.k, pool, has)        # Euclidean, as HGCMLDA
        Hg = gmm_hyperedges(X, c.g, pool, has, c.seed)
        if c.protocol == "B":
            pool = np.ones(len(X), bool)
        Gk = build_propagator(Hk, pool, pool.copy(), c.protocol)
        Gg = build_propagator(Hg, pool, np.ones(Hg.shape[1], bool), c.protocol)
        return torch.tensor(Gk, device=self.dev), torch.tensor(Gg, device=self.dev)

    def fit(self, M, Xl, Xd, dss, ml, md, epochs=None, val=None, leak_sims=None):
        c, dev = self.cfg, self.dev
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        L, D = M.shape
        Xin_l, Xin_d = self._inputs(M, Xl, Xd, dss, ml, md, leak_sims)
        has_l = np.any(Xl != 0, 1) if c.input == "content" else np.ones(L, bool)
        poolL = ml if c.protocol == "A" else np.ones(L, bool)
        poolD = md if c.protocol == "A" else np.ones(D, bool)
        self.Gl = self._graphs(Xin_l, poolL, has_l | (c.protocol == "B"))
        self.Gd = self._graphs(Xin_d, poolD, np.ones(D, bool))
        A = np.zeros((L, D), np.float32)
        A[np.ix_(ml, md)] = M[np.ix_(ml, md)] > 0
        self.At = torch.tensor(A, device=dev)
        self.xl, self.xd = torch.tensor(Xin_l, device=dev), torch.tensor(Xin_d, device=dev)
        self.tl, self.td = torch.tensor(Xl, device=dev), torch.tensor(Xd, device=dev)
        self.L = ChainSide(Xin_l.shape[1], D, Xl.shape[1], c).to(dev)
        self.D = ChainSide(Xin_d.shape[1], L, Xd.shape[1], c).to(dev)
        it = torch.as_tensor(np.nonzero(ml)[0], device=dev)
        jt = torch.as_tensor(np.nonzero(md)[0], device=dev)
        # GRACE nodes: every node under Protocol B (as HGCMLDA), train nodes only under Protocol A
        self.cl_l = torch.arange(L, device=dev) if c.protocol == "B" else it
        self.cl_d = torch.arange(D, device=dev) if c.protocol == "B" else jt
        part = torch.tensor(np.outer(ml, md), device=dev)
        one = (self.At > 0) & part
        zero = (self.At == 0) & part
        Y = self.At[it][:, jt]
        bce = nn.BCEWithLogitsLoss(pos_weight=((Y == 0).sum() / (Y.sum() + 1)).reshape(1))
        opt = torch.optim.Adam(list(self.L.parameters()) + list(self.D.parameters()), lr=c.lr, weight_decay=c.wd)
        hist, self.loss_log = [], []
        for ep in range(1, (epochs or c.epochs) + 1):
            self.L.train()
            self.D.train()
            S, al, ad = self._forward(self.cl_l, self.cl_d)
            if c.loss == "wmse":
                se = (self.At - S) ** 2
                la = (1 - c.beta) * se[one].sum() + c.beta * se[zero].sum()
            else:
                la = bce(S[it][:, jt], Y)
            loss = la
            if c.cl in ("grace", "both"):
                loss = loss + c.lam_cl * (al["grace"] + ad["grace"])
            if c.cl in ("feat_hg", "both") and al["h"] is not None:
                lf = 0.0
                for side, aux, idx in ((self.L, al, it), (self.D, ad, jt)):
                    sub = idx[torch.randperm(len(idx), device=dev)[:1024]]
                    lf = lf + 0.5 * info_nce(side.qh(aux["h"][sub]), side.qg(aux["g"][sub]), c.tau)
                loss = loss + c.lam_cl * lf
            if c.branch2 == "vae":
                rec = torch.relu(torch.sigmoid(al["vae_z"] @ ad["vae_z"].T))
                se = (self.At - rec) ** 2
                lv = kl(*ad["kl"], D) + kl(*al["kl"], L) + (1 - c.beta) * se[one].sum() + c.beta * se[zero].sum()
                loss = loss + 0.15 * lv
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % c.eval_every == 0:
                self.loss_log.append((ep, la.item(), 0.0))
                if val is not None:
                    hist.append((ep, float(val(self.scores()))))
        return hist

    def _forward(self, it, jt):
        zl, al = self.L(self.xl, *self.Gl, self.At, self.tl, it)
        zd, ad = self.D(self.xd, *self.Gd, self.At.T, self.td, jt)
        return zl @ zd.T, al, ad

    @torch.no_grad()
    def scores(self):
        self.L.eval()
        self.D.eval()
        it = torch.arange(self.xl.shape[0], device=self.dev)[:2]
        jt = torch.arange(self.xd.shape[0], device=self.dev)[:2]
        S, _, _ = self._forward(it, jt)
        return S.cpu().numpy().astype(np.float64)
