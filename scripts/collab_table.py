"""3-seed table of docs/collab_hypergraph_d12.md: R0 (FG-HGCL-HG), R1 (FG-HGCL), CS0, CS0.1 (paper-style metrics, D2 neg seed 4)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "hgcmlda_protocol"
REF = {("1", "leaky"): ("D1_leaky_B0f_abl_nohg_nce", "D1_leaky_B3_final_nce"), ("1", "clean"): ("D1_clean_B0f_as2_R0", "D1_clean_B3_final_nce"),
       ("2", "leaky"): ("D2_leaky_B0fv_neg4", "D2_leaky_B3_neg4"), ("2", "clean"): ("D2_clean_B0fv_neg4", "D2_clean_B3_neg4")}
print("| Dataset | Variant | Tag | Inner-val | CV AUC | CV AUPR | Ind. AUC | Ind. AUPR |\n|---|---|---|---|---|---|---|---|")
for (ds, var), (r0, r1) in REF.items():
    for tag, f in (("R0", r0), ("R1", r1), ("CS0", f"D{ds}_{var}_B3_cs2_CS0"), ("CS0.1", f"D{ds}_{var}_B3_cs2_CS0.1")):
        p = R / f"{f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        assert len(d["seeds"]) == 3, f
        iv = d.get("inner_val_ap", [sum(d["seeds"][s]["inner_val_ap"] for s in d["seeds"]) / 3])[0]
        v = [d[k][0] for k in ("cv_auc", "cv_aupr", "independent_auc", "independent_aupr")]
        print(f"| D{ds} | {var} | {tag} | {iv:.4f} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
