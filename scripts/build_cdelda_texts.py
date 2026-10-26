"""Disease text files that CDELDA's LoRA model (bench/hero_peft.py) reads, for D1 / D2 (l2d5 already has CDELDA's own).
data/harness/data_rd_<ds>/disease_doids.txt : one id per disease, node order (synthetic ids "<ds>:<index>"; D1 / D2 use
                                               names, not DOIDs)
data/harness/data_rd_<ds>/disease_texts.json: id -> "name. DO definition. Synonyms: ..." (IHC-LDA features/<ds>/
                                               distext_texts.json, the texts behind disease_emb; lncRNA mentions removed)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")
for ds in ("D1", "D2"):
    texts = json.load(open(IHC / f"features/{ds}/distext_texts.json"))
    d = ROOT / f"data/harness/data_rd_{ds}"
    ids = [f"{ds}:{i}" for i in range(len(texts))]
    (d / "disease_doids.txt").write_text("\n".join(ids) + "\n")
    (d / "disease_texts.json").write_text(json.dumps(dict(zip(ids, texts)), indent=0))
    print(ds, len(ids))
