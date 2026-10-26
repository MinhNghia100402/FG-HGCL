"""Final harness table (docs/final_lock.md): FG-HGCL ids (mean +- sd over model seeds) next to the baselines run in this
repo and the cited CDELDA-LoRA number. Writes docs/final_results.md and prints it.
Usage: /sda1/nghiavm/research/.venv/bin/python scripts/final_table.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"l2d5": "D3 (CDELDA l2d5)", "D1": "D1", "D2": "D2"}
IDS = ["B0f", "B2", "B3", "B7"]
SCN = [("both", "Both-cold"), ("lncRNA", "lncRNA-cold"), ("disease", "Disease-cold")]
METRICS = ["AUPR_1to1", "AUC_1to1", "AUPR_all"]


def load(p):
    return json.loads(p.read_text())["models"] if p.exists() else {}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{np.mean(vals):.4f}" if len(vals) == 1 else f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def main():
    out = ["# Kết quả test cuối (harness CDELDA, cold protocol, 5 fold, fold seed 2026)", "",
           "FG-HGCL: mean ± sd qua 3 seed mô hình (2026–2028). Baseline: một lần chạy. CDELDA-LoRA: chỉ trích dẫn.", ""]
    for v, vname in VARIANTS.items():
        fg = load(ROOT / f"results/harness/fg_hgcl/bench_rd{v}_cold.json")
        bl = load(ROOT / f"results/harness/baselines/bench_rd{v}_cold.json")
        ref = load(ROOT / "results/reference/cdelda_rerun_l2d5_cold.json") if v == "l2d5" else {}
        rows = defaultdict(list)
        for name, r in fg.items():
            m = re.match(r"FG-HGCL (\S+) s(\d+)", name)
            if m:
                rows[m.group(1)].append(r)
        b1, b1lab = bl.get("HGCMLDA-clean (paper HP, Dataset1)"), "B1 HGCMLDA-clean"
        if b1 is None and v == "l2d5":                      # rerun still in progress: cite IHC-LDA's harness run
            b1 = load(ROOT / "results/reference/hgcmlda_clean_l2d5_cold.json").get("HGCMLDA-clean (paper HP, Dataset1)")
            b1lab += " (trích dẫn IHC-LDA)"
        table = [("B0 CDELDA TwoTower (content)", [bl.get("TwoTower (content)")]), (b1lab, [b1])]
        if ref:
            table.append(("CDELDA-LoRA (trích dẫn)", [ref.get("TwoTower-PEFT-disease (LoRA S-BioBERT)")]))
        table += [(f"FG-HGCL {i}" + (" (chính)" if i == "B3" else ""), rows.get(i, [])) for i in IDS]
        for scn, sname in SCN:
            out += [f"## {vname} — {sname}", "", "| Mô hình | " + " | ".join(METRICS) + " | n |",
                    "|---|" + "---|" * (len(METRICS) + 1)]
            for label, rs in table:
                rs = [r for r in rs if r]
                cells = [fmt([r[scn][mt]["mean"] for r in rs]) if rs else "—" for mt in METRICS]
                out.append(f"| {label} | " + " | ".join(cells) + f" | {len(rs)} |")
            out.append("")
    text = "\n".join(out)
    (ROOT / "docs" / "final_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
