"""Multi-view disease text features (direction 1): name / DO definition / synonyms encoded SEPARATELY with the FROZEN
S-BioBERT used for disease_emb (feature extraction only, no fine-tuning). Missing parts fall back to the name.

Sources (label-free): l2d5 = IHC-LDA data/interim/cdelda_rebuild/.../disease_texts.json + disease_doids.txt (CDELDA's text);
D1 / D2 = IHC-LDA features/<ds>/distext_texts.json ("name. definition. Synonyms: ...", lncRNA mentions already removed).
Writes data/harness/data_rd_<v>/disease_views.npz {name, defn, syn} (n_dis x 768, mean-pooled, max_length 256).
Usage: CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/build_disease_views.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")
SBERT = IHC / "features/weights/S-BioBert-snli-multinli-stsb"


def split_text(t: str):
    t = t.strip()
    syn = ""
    if "Synonyms:" in t:
        t, syn = t.split("Synonyms:", 1)
        syn = syn.strip().rstrip(".")
    parts = t.split(". ", 1)
    name = parts[0].strip().rstrip(".")
    defn = parts[1].strip().rstrip(".") if len(parts) > 1 else ""
    return name, (defn or name), (syn or name)


@torch.no_grad()
def embed(texts, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SBERT)
    model = AutoModel.from_pretrained(SBERT).to(device).eval()
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def texts_of(v):
    if v == "l2d5":
        d = IHC / "data/interim/cdelda_rebuild/data/data_rd_l2d5"
        ids = [x.strip() for x in open(d / "disease_doids.txt") if x.strip()]
        tm = json.load(open(d / "disease_texts.json"))
        return [tm.get(i, i) for i in ids]
    return json.load(open(IHC / f"features/{v}/distext_texts.json"))


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for v in ("l2d5", "D1", "D2"):
        T = [split_text(t) for t in texts_of(v)]
        n = len(np.load(ROOT / f"data/harness/data_rd_{v}/disease_emb.npy"))
        assert len(T) == n, (v, len(T), n)
        views = {k: embed([t[i] for t in T], dev) for i, k in enumerate(("name", "defn", "syn"))}
        out = ROOT / f"data/harness/data_rd_{v}/disease_views.npz"
        if out.is_symlink() or out.exists():
            out.unlink()
        np.savez(out, **views)
        print(v, {k: x.shape for k, x in views.items()},
              "defn=name for", sum(t[1] == t[0] for t in T), "syn=name for", sum(t[2] == t[0] for t in T))
