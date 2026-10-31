#!/usr/bin/env python
"""Run the whole project.

    python run_pipeline.py                       # medium difficulty, 1800 cases  (about 5 minutes on a laptop CPU)
    python run_pipeline.py --difficulty all      # easy + medium + hard, plus a comparison table and chart
    python run_pipeline.py --n_cases 600         # quicker trial run

Outputs go to ./runs/<difficulty>/ (data, predictions, results) and ./runs/ (summary files).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adf.config import GenConfig
from adf.pipeline import run


def samples_figure(data_dir, out_png):
    recs = [json.loads(l) for l in open(os.path.join(data_dir, "cases.jsonl"))]
    pick = []
    for lab, dt in (("genuine", "statement"), ("ai_generated", "statement"), ("ai_edited", "statement"), ("ai_edited", "payslip")):
        r = next((r for r in recs if r["label"] == lab and r["doc_type"] == dt), None)
        if r: pick.append(r)
    fig, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 6))
    for a, r in zip(np.atleast_1d(ax), pick):
        a.imshow(cv2.cvtColor(cv2.imread(os.path.join(data_dir, r["image"])), cv2.COLOR_BGR2RGB)); a.axis("off")
        a.set_title(f"{r['case_id']}: {r['label']}\n{r['manipulation']}", fontsize=8)
    fig.suptitle("Synthetic documents (fictional bank and employer)", fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=1800); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "adversarial", "all"])
    ap.add_argument("--out", default="runs"); ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--reuse_data", action="store_true", help="skip regeneration if data already exists in --out")
    ap.add_argument("--n_boot", type=int, default=500)
    a = ap.parse_args()
    levels = ["easy", "medium", "hard", "adversarial"] if a.difficulty == "all" else [a.difficulty]
    rows = []
    for lv in levels:
        cfg = GenConfig(n_cases=a.n_cases, seed=a.seed).with_difficulty(lv)
        od = os.path.join(a.out, lv)
        r = run(cfg, od, n_jobs=a.n_jobs, reuse_data=a.reuse_data, n_boot=a.n_boot)
        if lv in ("medium", levels[0]) and not os.path.exists(os.path.join(a.out, "sample_documents.png")):
            samples_figure(os.path.join(od, "data"), os.path.join(a.out, "sample_documents.png"))
        t = r["test_metrics"]; bb = r["best_single_branch_by_train_oof"]
        for name in (bb, "mean_of_branches", "early_fusion_all_features", "fusion_lr", "fusion_gb"):
            rows.append(dict(difficulty=lv, model=name.replace(bb, "best_single_branch (" + bb.replace("branch_", "") + ")") if name == bb else name,
                             roc_auc=t[name]["roc_auc"], pr_auc=t[name]["pr_auc"], f1=t[name]["f1"], precision=t[name]["precision"],
                             recall=t[name]["recall"], specificity=t[name]["specificity"]))
    df = pd.DataFrame(rows).round(4); df.to_csv(os.path.join(a.out, "summary_by_difficulty.csv"), index=False)
    print("\n", df.to_string(index=False))
    if len(levels) > 1:
        piv = df.pivot(index="model", columns="difficulty", values="roc_auc")[levels]
        ax = piv.plot.bar(figsize=(9, 4.2), rot=20); ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test ROC-AUC")
        ax.set_title("Detection performance as the synthetic assumptions get harder"); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "difficulty_comparison.png"), dpi=140)


if __name__ == "__main__":
    main()
