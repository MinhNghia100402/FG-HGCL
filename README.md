# FG-HGCL: Feature-Grounded Hypergraph Contrastive Learning for lncRNA–Disease Association Prediction

## Motivation

Known lncRNA–disease associations are sparse (about 1–2 % of all pairs), and the most useful predictions are for
lncRNAs or diseases that have **no known association yet** (cold-start). Existing methods tend to fall into one of two
groups:

- **Hypergraph contrastive models** such as HGCMLDA build every node input from the association matrix (GIPK, LFS).
  These inputs leak test labels when computed on the full matrix, and they are empty for a new node.
- **Content models** such as CDELDA encode sequence and text features. They generalise to new nodes, but they encode
  each node independently and ignore structure.

FG-HGCL keeps hypergraph contrastive learning but anchors it to label-free content features and builds every
structure inductively, so the same model scores warm and cold-start pairs without label leakage.

## Problem setting

Given a binary association matrix **A** (lncRNA × disease), where unobserved cells are unlabeled rather than negative,
and content features for every node, the model learns a score for every pair. It is evaluated in four scenarios:

| Scenario | Test pairs |
|---|---|
| C-normal (warm) | Hidden associations between nodes seen during training |
| C-lnc | New lncRNAs × known diseases |
| C-dis | Known lncRNAs × new diseases |
| C-both | New lncRNAs × new diseases |

## Method

- **Feature towers:** two-layer MLPs on fixed, label-free content features (lncRNA: RNA-FM ‖ ViennaRNA ‖ GTEx;
  disease: frozen S-BioBERT text embeddings, optionally name / definition / synonym views).
- **Strict-inductive hypergraphs:** KNN (cosine) and GMM (PCA-32, diagonal covariance) hyperedges built from training
  nodes only. Under Protocol A, a new node reads from training nodes and never changes their embeddings.
- **Two-level contrastive learning:** InfoNCE between the feature embedding and the hypergraph embedding of each node,
  plus a link loss (balanced BCE, InfoNCE with masked negatives, or both).
- **Leakage control:** configurations and epochs are selected on an inner validation split carved from the training
  part of each fold. `tests/test_leakage.py` (17 cases) checks that scores do not depend on labels outside the
  training block.

## Evaluation protocols

- **CDELDA harness** (D3 primary; D1, D2): CDELDA's unmodified evaluation code. Five folds with fold seed 2026, all
  four scenarios. Primary metric is AUPR on 1:1 sampled negatives, plus AUROC 1:1, AUPR over all pairs and Recall@20.
- **HGCMLDA protocol** (D1, D2): 5-fold CV over pairs plus an independent split, with 1:1 negatives and the
  paper-style AUC / AUPR. It is reported both as published (with HGCMLDA's label-derived inputs) and in a clean
  variant without them.
- **Selection:** every choice (configuration, epoch count, link loss) is made on inner validation inside the training
  part of each fold. Test folds are scored with three model seeds once a configuration is fixed.
- **Ablations:** feature-only baseline, hypergraph without contrast, view, fusion and decoder variants, augmentation,
  an association-hypergraph view, and a step-by-step HGCMLDA → FG-HGCL module-replacement chain.

## Repository layout

| Path | Content |
|---|---|
| `fghgcl/model.py`, `fghgcl/hypergraph.py` | Model, losses, KNN/GMM hypergraphs, Protocol A/B propagators |
| `fghgcl/harness.py` | Adapter for CDELDA's evaluation harness (`fit` / `predict`), inner split |
| `fghgcl/chain.py` | Modular re-implementation of HGCMLDA and the HGCMLDA → FG-HGCL module-replacement chain |
| `fghgcl/leaky.py` | HGCMLDA-style similarities on the full matrix (comparison only) |
| `configs/` | Every configuration tried (`fg_hgcl_both_cold.yaml`), the module chain (`chain.yaml`), baselines |
| `scripts/` | Data building, inner-validation selection, harness / HGCMLDA-protocol runs, run queues |
| `data/harness/data_rd_{l2d5,D1,D2}/` | Harness inputs: association matrix, lncRNA features, disease embeddings, DSS |
| `data/splits/` | Fold and inner-validation split manifests (fold seed 2026) |
| `tests/` | Leakage tests |

## Setup

Python ≥ 3.12 with `torch`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `pytest` (`transformers` only to rebuild disease
embeddings). A CUDA GPU is recommended but not required for the tests.

```bash
python -m pytest -q tests/          # 17 leakage tests
```

### CDELDA harness

Evaluation uses CDELDA's own, unmodified harness. Clone it and point `CDELDA_REPO` at the clone:

```bash
git clone https://github.com/rlatkddnr122/CDELDA.git /path/to/CDELDA
export CDELDA_REPO=/path/to/CDELDA

# inner validation of configurations (no test block is scored)
python scripts/run_ablation.py inner  --variant l2d5 --ids B0f B2 B3
python scripts/run_ablation.py report --variant l2d5 --ref B0f --ids B2 B3

# score a locked configuration on the harness test folds, 3 model seeds
python scripts/run_ablation.py harness --variant l2d5 --ids B3 --seeds 2026 2027 2028

# a single configuration, warm (C-normal) protocol
python scripts/run_harness.py --name "FG-HGCL warm" --protocol warm --fixed \
    --cfg '{"views": ["knn", "gmm"], "lam_cl": 0.5, "link_loss": "infonce+bce", "epochs": 600}'
```

Variants: `l2d5` (= D3, primary), `D1`, `D2`. Results are written to `results/` (not tracked).

### Scripts with local paths

The scripts that build data, run baselines or run the original HGCMLDA protocol (`build_harness_data.py`,
`build_disease_views.py`, `build_cdelda_texts.py`, `cross_dataset_map.py`, `run_baseline.py`, `run_leaky.py`,
`hgcmlda_protocol.py`, `collect_reference.py`, `queue_*.sh`) read raw files and feature caches from absolute paths of
the original server (`IHC = Path(...)`, `HG = Path(...)` near the top of each file). Edit these paths before you
use them. The data shipped in `data/` is enough to run the harness and the tests.

## Data sources

All shipped data is derived from public resources. Please cite the original sources when you use it.

| Dataset | Size (lncRNA × disease, links) | Source |
|---|---|---|
| **D1** | 861 × 253, 4 517 | Dataset1 of HGCMLDA (compiled from LncRNADisease v2.0 and Lnc2Cancer 3.0), [aI-area/HGCMLDA](https://github.com/aI-area/HGCMLDA) |
| **D2** | 2 853 × 310, 7 862 | Dataset2 of HGCMLDA (compiled from MNDR v3.0), same repository |
| **D3 = l2d5** | 5 102 × 245, 20 197 | RNADisease v4.0, k-core (≥ 2 diseases per lncRNA, ≥ 5 lncRNAs per disease), built with the scripts of [CDELDA](https://github.com/rlatkddnr122/CDELDA) (legacy benchmark, before 2026-09-10) |

Features:

- **lncRNA:** RNA-FM embeddings (640-d, mean over 1 022-nt windows) of RNAcentral sequences; ViennaRNA secondary-structure
  features (8-d, l2d5 only); GTEx expression. Each block is L2-normalised, and rows without a sequence are all-zero
  (805 / 5 102 on l2d5, 174 / 861 on D1, 239 / 2 853 on D2).
- **Disease:** `pritamdeka/S-BioBert-snli-multinli-stsb` embeddings (768-d) of Disease Ontology text (name, definition,
  synonyms; `disease_views.npz` holds one view each); disease semantic similarity (DSS, Wang et al.) from the
  Disease Ontology DAG. HGCMLDA's `DSS.txt` is used for D1 and D2.

No feature is derived from the association matrix.

References:

- Zhang Z. et al. HGCMLDA: predicting lncRNA-disease associations using hypergraph contrastive learning and
  multi-scale attentional feature fusion. *Briefings in Bioinformatics* 26(3), bbaf262 (2025).
- Kim S., Yun T., Ha J. CDELDA: A Content-Based Dual-Encoder for Cold-Start lncRNA–Disease Association Prediction.
  *Biomedicines* 14(9), 2121 (2026).
- Chen J. et al. RNADisease v4.0. *Nucleic Acids Research* 51(D1), D1397–D1404 (2023).
- Bao Z. et al. LncRNADisease 2.0. *Nucleic Acids Research* 47(D1), D1034–D1037 (2019).
- Gao Y. et al. Lnc2Cancer 3.0. *Nucleic Acids Research* 49(D1), D1251–D1258 (2021).
- Ning L. et al. MNDR v3.0. *Nucleic Acids Research* 49(D1), D160–D164 (2021).
- Chen J. et al. Interpretable RNA Foundation Model from Unannotated Data (RNA-FM). arXiv:2204.00300 (2022).
- Lorenz R. et al. ViennaRNA Package 2.0. *Algorithms for Molecular Biology* 6, 26 (2011).
- The GTEx Consortium. *Science* 369(6509), 1318–1330 (2020).
- Deka P. et al. (S-BioBERT), iiWAS 2021; Lee J. et al. BioBERT, *Bioinformatics* 36(4) (2020).
- Schriml L.M. et al. The Human Disease Ontology 2022 update. *Nucleic Acids Research* 50(D1), D1255–D1261 (2022).
- Wang J.Z. et al. A new method to measure the semantic similarity of GO terms. *Bioinformatics* 23(10) (2007).
