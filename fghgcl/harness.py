"""FG-HGCL behind CDELDA's harness contract (bench/interface.py):
    fit(M, Clnc, Cdis, train_lnc, train_dis) may read ONLY M[train_lnc, train_dis];  predict() -> finite (n_l, n_d).

Inner validation (never touches the harness test block), as in IHC-LDA scripts/cdelda_ihclda_adapter.py @ 9c88513:
  both-cold : 20 % of train lncRNAs x 20 % of train diseases;  lncRNA-/disease-cold : 12.5 % of the cold side.
The model is trained on the inner-train block, the epoch count is chosen on inner validation (AUPR 1:1, CDELDA's
metric), then it is refit on the whole harness train block for that many epochs.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from sklearn.metrics import average_precision_score

from .model import FGHGCL, Config


def build(cfg_dict: dict):
    """Config object + model class from a JSON-style dict; {"model": "chain", ...} selects the HGCMLDA chain."""
    d = dict(cfg_dict)
    if d.pop("model", "fghgcl") == "chain":
        from .chain import ChainConfig, ChainModel
        return ChainConfig(**d), ChainModel
    for k in ("views", "dis_views", "extra_src", "kg_lnc", "kg_dis"):
        if k in d:
            d[k] = tuple(d[k])
    return Config(**d), FGHGCL

SEED = 2026
DATA = os.environ.get("CDELDA_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "harness"))


def scenario(tl, td):
    if tl.all() and td.all():
        return "warm"
    if tl.all():
        return "disease"
    if td.all():
        return "lncRNA"
    return "both"


def harness_folds(n, k=5, seed=SEED):
    """= CDELDA snapshot_src/twoside_common.folds."""
    rng = np.random.default_rng(seed)
    return [np.sort(f) for f in np.array_split(rng.permutation(n), k)]


def inner_split(tl, td, scn, seed):
    rng = np.random.default_rng(seed)
    vl, vd = np.zeros_like(tl), np.zeros_like(td)
    frac = 0.2 if scn == "both" else 0.125
    if scn in ("lncRNA", "both"):
        idx = np.nonzero(tl)[0]
        vl[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    if scn in ("disease", "both"):
        idx = np.nonzero(td)[0]
        vd[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    rows = vl if scn in ("lncRNA", "both") else tl & ~vl
    cols = vd if scn in ("disease", "both") else td & ~vd
    return tl & ~vl, td & ~vd, rows, cols


def aupr_1to1(S, Y, draws=5, seed=SEED):
    pos, neg = S[Y], S[~Y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        n = rng.choice(len(neg), size=min(len(pos), len(neg)), replace=False)
        out.append(average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(n))], np.r_[pos, neg[n]]))
    return float(np.mean(out))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_semsim():
    p = os.environ.get("CCDIFF_DIS_SEMSIM")
    return np.load(p).astype(np.float64) if p else None


class FGHGCLHarness:
    def __init__(self, cfg: Config, refit: bool = True, inner_seed: int = 0, device: str = "cuda", fixed: bool = False,
                 scn_over: dict | None = None):
        """fixed=True: train once on the whole harness train block for cfg.epochs (the epoch count chosen beforehand
        on the averaged inner-validation curve, scripts/paired.py); no per-fold selection."""
        self.cfg, self.refit, self.inner_seed, self.device, self.fixed = cfg, refit, inner_seed, device, fixed
        self.scn_over = scn_over or {}                  # direction 2: per-scenario overrides {"disease": {...}, ...}
        self.cls = FGHGCL
        if type(cfg).__name__ == "ChainConfig":
            from .chain import ChainModel
            self.cls = ChainModel
        self.log = {}

    def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
        M = np.asarray(M) > 0
        L, D = M.shape
        tl, td = np.zeros(L, bool), np.zeros(D, bool)
        tl[np.asarray(train_lnc)], td[np.asarray(train_dis)] = True, True
        leak = None
        if self.cfg.leak:                                     # DELIBERATE contract violation (HGCMLDA protocol, comparison only)
            assert self.fixed, "leak mode is only defined for fixed-epoch harness runs"
            from .leaky import leaky_similarities
            leak = leaky_similarities(M, load_semsim())
        M = M & tl[:, None] & td[None, :]                     # contract: nothing outside the train block is visible
        scn = scenario(tl, td)
        if scn in self.scn_over:
            self.cfg = replace(self.cfg, **{k: tuple(v) if isinstance(v, list) else v for k, v in self.scn_over[scn].items()})
        Xl, Xd = np.asarray(Clnc, np.float32), np.asarray(Cdis, np.float32)
        dss = load_semsim()
        if scn == "warm" or self.fixed:                       # warm is not a target of this study
            S = 0.0
            for j in range(self.cfg.ens):                     # seed ensemble (ens = 1: a single model)
                m = self.cls(replace(self.cfg, seed=self.cfg.seed + j), self.device)
                m.fit(M, Xl, Xd, dss, tl, td, leak_sims=leak)
                S = S + sigmoid(m.scores()) / self.cfg.ens
            self.S = np.nan_to_num(S).astype(np.float32)
            return self
        ml, md, vr, vc = inner_split(tl, td, scn, SEED + self.inner_seed)
        Yv = M[np.ix_(vr, vc)]
        val = lambda S: aupr_1to1(S[np.ix_(vr, vc)], Yv)      # noqa: E731
        m = self.cls(self.cfg, self.device)
        hist = m.fit(M & ml[:, None] & md[None, :], Xl, Xd, dss, ml, md, val=val)
        best_ep, best_v = max(hist, key=lambda r: (r[1], -r[0]))
        self.log = {"scenario": scn, "hist": hist, "best_epoch": best_ep, "best_val": best_v, "loss": m.loss_log}
        if self.refit:
            m = self.cls(replace(self.cfg), self.device)
            m.fit(M, Xl, Xd, dss, tl, td, epochs=best_ep)
        else:
            m = self.cls(self.cfg, self.device)
            m.fit(M & ml[:, None] & md[None, :], Xl, Xd, dss, ml, md, epochs=best_ep)
        self.model = m
        self.S = np.nan_to_num(m.scores()).astype(np.float32)
        return self

    def predict(self):
        return self.S


def load_variant(v: str = "l2d5"):
    """M (bool), Clnc, Cdis of data/harness/data_rd_<v> (built by scripts/build_harness_data.py); sets CCDIFF_DIS_SEMSIM
    exactly as bench.sweep_run does."""
    d = os.path.join(DATA, f"data_rd_{v}")
    M = np.load(os.path.join(d, "M.npy")) > 0
    Xl = np.load(os.path.join(d, os.environ.get("CCDIFF_LNC_FILE", "lnc_ortho.npy"))).astype(np.float32)   # as bench.sweep_run
    Xd = np.load(os.path.join(d, "disease_emb.npy")).astype(np.float32)
    os.environ["CCDIFF_DIS_SEMSIM"] = os.path.join(d, "disease_semsim.npy")
    return M, Xl, Xd
