"""Copy already-computed CDELDA-harness results from IHC-LDA (read-only) into results/reference, for citation only.

  cdelda_rerun_l2d5_cold.json    CDELDA "TwoTower-PEFT-disease (LoRA S-BioBERT)" rerun with CDELDA's code (IHC-LDA results/cdelda_repro)
  hgcmlda_clean_l2d5_cold.json   HGCMLDA-clean in the harness (IHC-LDA results/harness_hgcm/main_hgcmlda)
  cdelda_published/              CDELDA paper tables (IHC-LDA results/cdelda_published)
Prints both-cold AUPR 1:1 / AUC 1:1 of every model in those files.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA/results")
OUT = ROOT / "results" / "reference"
FILES = {"cdelda_rerun_l2d5_cold.json": IHC / "cdelda_repro/bench_rdl2d5_cold.json",
         "hgcmlda_clean_l2d5_cold.json": IHC / "harness_hgcm/main_hgcmlda/bench_rdl2d5_cold.json"}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in FILES.items():
        shutil.copy2(src, OUT / name)
    shutil.copytree(IHC / "cdelda_published", OUT / "cdelda_published", dirs_exist_ok=True)
    for name in FILES:
        d = json.loads((OUT / name).read_text())
        for model, r in d["models"].items():
            b = r["both"]
            print(f"{name:<32} {model:<45} both-cold AUPR1:1 {b['AUPR_1to1']['mean']:.4f}  AUC1:1 {b['AUC_1to1']['mean']:.4f}")
