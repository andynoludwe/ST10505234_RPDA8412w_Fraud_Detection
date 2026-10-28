"""
Build matched, case-level fusion data from the REAL held-out predictions of
the three branches (document CNN, metadata/OCR Random Forest, Siamese
signature network), without modifying any branch's data generator.

Why this exists
---------------
The branches were built on independent datasets, so no document, metadata
record and signature share a case_id. This script defines a case as:

    one document image   (scored by the CNN branch)
  + one metadata record  (scored by the RF branch)
  + one signature pair   (scored by the Siamese branch)

and builds cases by pairing items from each branch's prediction file
according to a scenario, so the cross-modal comparison is actually tested:

    genuine        document genuine,   signature genuine   -> label 0
    doc_forged     document forged,    signature genuine   -> label 1
    sig_forged     document genuine,   signature forged    -> label 1
    both_forged    document forged,    signature forged    -> label 1

A case is labelled 1 if any component is forged. Document-side items (CNN and
RF) always share the same genuine/forged status within a case.

Leakage rules (important for a valid RQ1 comparison)
----------------------------------------------------
1. Feed this script ONLY predictions the branches made on data they were NOT
   trained or tuned on (each branch's held-out test predictions, or
   out-of-fold predictions). Training-set predictions are over-confident and
   will inflate the fusion result.
2. Every underlying item is used in at most ONE case (sampling without
   replacement), so no item can appear in both the fusion training and test
   sets.
3. The 60/20/20 split is stratified by scenario, at case level.

Usage
-----
python build_matched_cases.py \
    --cnn_csv      document_cnn_results/resnet50_predictions.csv \
    --rf_csv       rf_results/rf_predictions.csv \
    --siamese_csv  signature_results/siamese_predictions.csv \
    --out_dir      fusion_data/

Outputs (in --out_dir)
----------------------
fusion_manifest.csv, fusion_train.csv, fusion_val.csv, fusion_test.csv
    Same schema the fusion meta-classifier already expects.
case_map.csv        case_id -> source item ids, scenario, label, split
case_scenarios.csv  case_id, scenario, split (for per-scenario reporting)
"""

import argparse
import csv
import random
from pathlib import Path

CNN_TYPES = ["splicing", "copy_move", "inpainting"]
RF_TYPES = ["tampered_text", "altered_figures", "metadata_anomaly", "layout_inconsistency"]
FEATURES = (["cnn_forgery_prob"] + [f"cnn_{t}_prob" for t in CNN_TYPES]
            + ["rf_forgery_prob"] + [f"rf_{t}_prob" for t in RF_TYPES]
            + ["siamese_forgery_prob"])
ID_CANDIDATES = ["case_id", "doc_id", "pair_id", "id", "sample_id"]
LABEL_CANDIDATES = ["label", "y_true", "true_label"]
PROB_CANDIDATES = ["forgery_prob", "prob", "fraud_prob", "probability"]
SCENARIOS = ["genuine", "doc_forged", "sig_forged", "both_forged"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick(row, candidates, what, path):
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"{path}: could not find a {what} column. Tried {candidates}. "
                   f"Columns present: {list(row.keys())}. Pass the right name explicitly in the script.")


def load_pool(path, name):
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    id_col = pick(rows[0], ID_CANDIDATES, "id", path)
    lab_col = pick(rows[0], LABEL_CANDIDATES, "label", path)
    prob_col = pick(rows[0], PROB_CANDIDATES, "forgery probability", path)
    pools = {0: [], 1: []}
    for r in rows:
        r["_id"], r["_label"], r["_prob"] = r[id_col], int(float(r[lab_col])), r[prob_col]
        pools[r["_label"]].append(r)
    print(f"{name}: {len(pools[0])} genuine, {len(pools[1])} forged held-out items  (id='{id_col}', label='{lab_col}', prob='{prob_col}')")
    return pools


def counts_for(n, frac_genuine):
    g = round(n * frac_genuine)
    f = n - g
    d = f // 3
    s = f // 3
    b = f - d - s
    return {"genuine": g, "doc_forged": d, "sig_forged": s, "both_forged": b}


def feasible(c, cnn, rf, sig):
    need_doc_gen = c["genuine"] + c["sig_forged"]
    need_doc_forg = c["doc_forged"] + c["both_forged"]
    need_sig_gen = c["genuine"] + c["doc_forged"]
    need_sig_forg = c["sig_forged"] + c["both_forged"]
    return (len(cnn[0]) >= need_doc_gen and len(cnn[1]) >= need_doc_forg
            and len(rf[0]) >= need_doc_gen and len(rf[1]) >= need_doc_forg
            and len(sig[0]) >= need_sig_gen and len(sig[1]) >= need_sig_forg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_csv", required=True)
    ap.add_argument("--rf_csv", required=True)
    ap.add_argument("--siamese_csv", required=True)
    ap.add_argument("--out_dir", default="fusion_data")
    ap.add_argument("--n_cases", type=int, default=0, help="0 = as many as the smallest pool allows")
    ap.add_argument("--frac_genuine", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cnn, rf, sig = load_pool(a.cnn_csv, "CNN"), load_pool(a.rf_csv, "RF"), load_pool(a.siamese_csv, "Siamese")
    for p in (cnn, rf, sig):
        for k in (0, 1):
            rng.shuffle(p[k])

    upper = min(len(cnn[0]) + len(cnn[1]), len(rf[0]) + len(rf[1]), len(sig[0]) + len(sig[1]))
    if a.n_cases:
        n = a.n_cases
        if not feasible(counts_for(n, a.frac_genuine), cnn, rf, sig):
            raise SystemExit(f"Not enough held-out items for {n} cases without reusing items. Reduce --n_cases or supply more held-out predictions.")
    else:
        n = next((k for k in range(upper, 3, -1) if feasible(counts_for(k, a.frac_genuine), cnn, rf, sig)), 0)
        if n < 12:
            raise SystemExit("Too few held-out items to build a usable case-level dataset. Generate more data per branch first.")
    plan = counts_for(n, a.frac_genuine)
    print(f"\nBuilding {n} matched cases: {plan}")

    def take(pool, label):
        return pool[label].pop()

    cases = []
    for scen, cnt in plan.items():
        doc_label = 1 if scen in ("doc_forged", "both_forged") else 0
        sig_label = 1 if scen in ("sig_forged", "both_forged") else 0
        for _ in range(cnt):
            c, r, s = take(cnn, doc_label), take(rf, doc_label), take(sig, sig_label)
            row = {"cnn_forgery_prob": c["_prob"], "rf_forgery_prob": r["_prob"], "siamese_forgery_prob": s["_prob"]}
            for t in CNN_TYPES:
                row[f"cnn_{t}_prob"] = c.get(f"{t}_prob", c.get(f"tamper_{t}_prob", 0.0))
            for t in RF_TYPES:
                row[f"rf_{t}_prob"] = r.get(f"{t}_prob", r.get(f"anomaly_{t}_prob", 0.0))
            row.update({"label": int(doc_label or sig_label), "_scenario": scen,
                        "_cnn_id": c["_id"], "_rf_id": r["_id"], "_sig_id": s["_id"]})
            cases.append(row)

    if not any(k in cases[0] and cases[0][k] not in (0, 0.0) for k in [f"cnn_{t}_prob" for t in CNN_TYPES]):
        print("NOTE: CNN anomaly-type columns were not found in the CNN predictions file; they are set to 0. "
              "Either add them to the CNN evaluate.py output or drop those features from FEATURES.")

    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c["case_id"] = f"case_{i:05d}"

    # stratified (by scenario) 60/20/20 split at case level
    split = {}
    for scen in SCENARIOS:
        grp = [c for c in cases if c["_scenario"] == scen]
        n_tr, n_va = int(len(grp) * 0.6), int(len(grp) * 0.2)
        for j, c in enumerate(grp):
            split[c["case_id"]] = "train" if j < n_tr else "val" if j < n_tr + n_va else "test"

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["case_id"] + FEATURES + ["label"]

    def write(name, rows):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write("fusion_manifest.csv", cases)
    for sp in ("train", "val", "test"):
        write(f"fusion_{sp}.csv", [c for c in cases if split[c["case_id"]] == sp])
    with open(out / "case_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "scenario", "label", "split", "cnn_id", "rf_id", "siamese_id"])
        for c in cases:
            w.writerow([c["case_id"], c["_scenario"], c["label"], split[c["case_id"]], c["_cnn_id"], c["_rf_id"], c["_sig_id"]])
    with open(out / "case_scenarios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "scenario", "split"])
        for c in cases:
            w.writerow([c["case_id"], c["_scenario"], split[c["case_id"]]])

    for sp in ("train", "val", "test"):
        sub = [c for c in cases if split[c["case_id"]] == sp]
        print(f"{sp}: {len(sub)} cases ({sum(c['label'] for c in sub)} fraudulent)")
    print(f"\nWrote files to {out}/. Train the fusion model with scripts/train.py using fusion_train.csv / fusion_val.csv.")


if __name__ == "__main__":
    main()
