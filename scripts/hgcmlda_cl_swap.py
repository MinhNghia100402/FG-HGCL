"""Released HGCMLDA with ONLY its contrastive module changed (docs/hgcmlda_cl_swap.md).

Everything else is the released code run by HGCMLDA's own runner (scripts/run_hgcmlda.py: data preparation, IL / ID,
KNN / GMM hypergraphs, HGCN, MS-CAM, VAE, weighted MSE, 0.1 * CL weight, paper HP, fixed epochs, evaluation). Before
the runner imports it, Module.CL_HGCN.forward is replaced so that the KNN-view <-> GMM-view loss is one of:

  grace     : the released GRACE loss (tau as given; the release uses 0.8)
  none      : no contrastive term (ablation)
  hyper_fn  : GRACE, but co-members of a node's KNN hyperedge are removed from its negatives (intra- and inter-view);
              hypergraph-aware false-negative filtering
  hyper_pos : GRACE with multi-positives: node i in one view is pulled to itself AND to its KNN-hyperedge co-members in
              the other view (positives = nonzero off-diagonal entries of the KNN propagator G); they also leave the
              negatives
  grace_aug : GRACE computed on stochastically augmented views (feature masking + dropping propagator entries, p 0.2,
              drawn independently per view and per epoch), as in standard hypergraph contrastive learning; the clean
              (un-augmented) HGCN outputs still feed the rest of the model

--cross L adds a cross-type InfoNCE between the lncRNA and disease embeddings of the hypergraph branch (outputs of
linear_L_2 / linear_D_2, i.e. KNN/GMM HGCN -> MS-CAM -> linear, before ReLU). Positives are the train associations
(train_adj = the forward's LD argument: fold-test and independent positives are already zeroed); for each node the
loss is the mean multi-positive InfoNCE over its train partners (other positives leave the denominator), summed over
nodes like GRACE. lambda * cross_lnc / cross_dis is added to lnc_cl_loss / dis_cl_loss, so it gets the same 0.1 weight.

Optionally the CV split is redrawn exactly as scripts/hgcmlda_newsplit.py does (--split-seed; the independent set never
changes). Results land in HGCMLDA/results/<run_id>/ (metrics.json, split.npz) like every other HGCMLDA run.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/hgcmlda_cl_swap.py --dataset 1 --cl hyper_pos --tau 0.8 --split-seed 1
       CUDA_VISIBLE_DEVICES=1 python scripts/hgcmlda_cl_swap.py --dataset 1 --cl grace --cross 0.5 --split-seed 1
"""
import argparse
import runpy
import sys

import numpy as np
import torch
import torch.nn.functional as F

HG = "/sda1/nghiavm/research/HGCMLDA"
HG_RUNNER = f"{HG}/scripts/run_hgcmlda.py"


P_AUG = 0.2


def augment(x):
    """Feature masking: zero a random P_AUG of the feature columns."""
    return x * (torch.rand(x.shape[1], device=x.device) >= P_AUG).to(x.dtype)


def augment_g(g):
    """Drop a random P_AUG of the propagator's entries (self loops kept)."""
    keep = torch.rand_like(g) >= P_AUG
    keep.fill_diagonal_(True)
    return g * keep.to(g.dtype)


def cross_infonce(a, b, pos, tau):
    """Per-row multi-positive InfoNCE of a against b; pos = bool (len a, len b). Rows without positives are skipped."""
    s = (F.normalize(a) @ F.normalize(b).t()) / tau
    has = pos.any(1)
    s, pos = s[has], pos[has]
    neg_lse = torch.logsumexp(s.masked_fill(pos, float("-inf")), 1, keepdim=True)          # other positives leave den
    per_pos = s - torch.logaddexp(s, neg_lse)                                               # log p(pos | pos + negs)
    return -((per_pos * pos).sum(1) / pos.sum(1)).sum()


def patch_cross(Module, lam, tau):
    """Add lam * cross-type InfoNCE (hypergraph-branch embeddings, train associations as positives) to the CL terms."""
    orig = Module.HGCMLDA.forward
    cache = {}

    def forward(self, *args):
        if not hasattr(self, "_cross_hooks"):
            self._cross_hooks = [self.linear_L_2.register_forward_hook(lambda m, i, o: cache.__setitem__("l", o)),
                                 self.linear_D_2.register_forward_hook(lambda m, i, o: cache.__setitem__("d", o))]
        score, lnc_cl, dis_cl, rec = orig(self, *args)
        if self.training:
            pos = args[6] > 0                                                                   # LD = train_adj
            lnc_cl = lnc_cl + lam * cross_infonce(cache["l"], cache["d"], pos, tau)
            dis_cl = dis_cl + lam * cross_infonce(cache["d"], cache["l"], pos.t(), tau)
        return score, lnc_cl, dis_cl, rec
    Module.HGCMLDA.forward = forward


def make_forward(mode, tau):
    def norm_sim(a, b):
        return F.normalize(a) @ F.normalize(b).t()

    def loss_dir(h1, h2, nb):
        """One direction of the (modified) GRACE loss; nb = bool co-member mask with False diagonal."""
        f = lambda x: torch.exp(x / tau)                                             # noqa: E731
        refl, betw = f(norm_sim(h1, h1)), f(norm_sim(h1, h2))
        eye = torch.eye(len(h1), dtype=torch.bool, device=h1.device)
        if mode in ("grace", "grace_aug"):
            num = betw.diag()
            den = refl.sum(1) + betw.sum(1) - refl.diag()
        elif mode == "hyper_fn":
            num = betw.diag()
            den = refl.masked_fill(nb | eye, 0).sum(1) + betw.masked_fill(nb, 0).sum(1)
        elif mode == "hyper_pos":
            num = betw.diag() + betw.masked_fill(~nb, 0).sum(1)
            den = refl.masked_fill(nb | eye, 0).sum(1) + betw.sum(1)
        else:
            raise ValueError(mode)
        return (-torch.log(num / den)).sum()       # the release: loss.sum(dim=-1).mean() on a 1-D tensor = sum

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        if mode == "none":
            return z1, z2, torch.zeros((), dtype=z1.dtype, device=z1.device)
        if mode == "grace_aug" and self.training:      # contrast augmented views; clean z1, z2 go downstream
            h1 = self.projection(self.hgcn1(augment(x1), augment_g(adj1)))
            h2 = self.projection(self.hgcn2(augment(x2), augment_g(adj2)))
        if not hasattr(self, "_nb"):                    # KNN co-membership; adj1 is the fixed KNN propagator
            nb = adj1 > 0
            nb.fill_diagonal_(False)
            self._nb = nb
        return z1, z2, self.alpha * loss_dir(h1, h2, self._nb) + (1 - self.alpha) * loss_dir(h2, h1, self._nb)
    return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, choices=[1, 2], required=True)
    ap.add_argument("--cl", choices=["grace", "none", "hyper_fn", "hyper_pos", "grace_aug"], required=True)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0, help="HGCMLDA model seed (reproduction runs use 0-4)")
    ap.add_argument("--split-seed", type=int, default=0, help="CV shuffle seed of prepareData.py (0 = the release)")
    ap.add_argument("--cross", type=float, default=0.0, help="weight of the cross-type InfoNCE (0 = off)")
    ap.add_argument("--cross-tau", type=float, default=0.5)
    ap.add_argument("--epoch", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()

    sys.path.insert(0, f"{HG}/original/Dataset{a.dataset}")
    import Module                                       # noqa: E402  (the runner's later import gets this object)
    Module.CL_HGCN.forward = make_forward(a.cl, a.tau)
    if a.cross:
        patch_cross(Module, a.cross, a.cross_tau)
    print(f"[cl_swap] CL_HGCN loss = {a.cl}, tau = {a.tau}, cross = {a.cross} (tau {a.cross_tau})", flush=True)

    if a.split_seed != 0:
        orig_seed = np.random.seed

        def seed(s=None):                               # only prepareData's np.random.seed(0) is redirected
            if s == 0 and sys._getframe(1).f_code.co_filename.endswith("prepareData.py"):
                print(f"[cl_swap] prepareData shuffle seed 0 -> {a.split_seed}", flush=True)
                s = a.split_seed
            return orig_seed(s)
        np.random.seed = seed

    tag = f"cl_{a.cl}_t{a.tau}" + (f"_x{a.cross}_xt{a.cross_tau}" if a.cross else "") + f"_split{a.split_seed}"
    sys.argv = [HG_RUNNER, "--dataset", str(a.dataset), "--variant", "original", "--seed", str(a.seed), "--tag", tag] + \
        (["--epoch", str(a.epoch)] if a.epoch else [])
    runpy.run_path(HG_RUNNER, run_name="__main__")


if __name__ == "__main__":
    main()
