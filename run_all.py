#!/usr/bin/env python3
"""
End-to-end, reproducible pipeline for the RPDA8412w fraud-detection framework (ALL DATA SYNTHETIC).

    python run_all.py                       # everything, defaults used for the reported results
    python run_all.py --steps docs cnn      # only some steps
    python run_all.py --pretrained          # CNN with ImageNet weights (needs internet to download them)

Steps, in order:
    docs      render fictional genuine bank documents                  -> outputs/genuine_documents/
    cnn       tamper them (split-first), train + evaluate the CNN      -> results/cnn/
    rf        generate metadata features, train + evaluate the RF      -> results/rf/
    siamese   generate signature pairs, train + evaluate the Siamese   -> results/siamese/
    fusion    build matched cases from the three branches' HELD-OUT
              predictions, train/evaluate/cross-validate the fusion    -> results/fusion/
    summary   results/summary.md

Large files (images, checkpoints) go to outputs/ and are git-ignored. Manifests go to data_manifests/ with
paths relative to the repository root, so they work on any machine. Every random choice is seeded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT, MAN, RES = ROOT / "outputs", ROOT / "data_manifests", ROOT / "results"
CNN, RF, SIG, FUS = (ROOT / n for n in ("document_cnn_branch", "document_rf_branch",
                                         "signature_siamese_branch", "fusion_meta_classifier"))
ALL_STEPS = ["docs", "cnn", "rf", "siamese", "fusion", "summary"]


def sh(args, cwd=ROOT, log: Path | None = None):
    """Run a command, stream its output, optionally keep it as a log file."""
    args = [str(a) for a in args]
    print("\n$", " ".join(args if len(args) < 12 else args[:11] + ["..."]), flush=True)
    t0 = time.time()
    p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise SystemExit(f"Command failed (exit {p.returncode}): {' '.join(args)}")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(lines), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s]", flush=True)


def py(component: Path, code: str, log: Path | None = None):
    """Run a snippet inside a component folder (each branch has its own `data` package, so one process each)."""
    sh([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, '.')\n{code}"], cwd=component, log=log)


# ----------------------------------------------------------------------------------------------- steps
def step_docs(a):
    sh([sys.executable, "-I", ROOT / "tools" / "make_genuine_documents.py", "--n", a.n_docs,
        "--out_dir", OUT / "genuine_documents", "--seed", a.seed])


def step_cnn(a):
    man = MAN / "cnn"
    py(CNN, f"""
from data.synthetic_tamper import generate_split_datasets
generate_split_datasets(genuine_dir=r'{OUT / "genuine_documents"}', output_dir=r'{OUT / "cnn_data"}',
                        manifest_dir=r'{man}', data_root=r'{ROOT}', fracs={tuple(a.cnn_fracs)}, seed={a.seed})
""", log=RES / "cnn" / "data_log.txt")
    ck = OUT / "checkpoints" / "cnn"
    sh([sys.executable, "-I", CNN / "scripts" / "train.py", "--train_manifest", man / "train.csv",
        "--val_manifest", man / "val.csv", "--data_root", ROOT, "--backbone", a.backbone,
        "--epochs", a.epochs, "--warmup_epochs", a.warmup, "--lr", a.lr, "--batch_size", a.batch_size,
        "--image_size", a.image_size, "--patience", a.patience, "--output_dir", ck,
        "--pretrained" if a.pretrained else "--no_pretrained"], log=RES / "cnn" / "train_log.txt")
    sh([sys.executable, "-I", CNN / "scripts" / "evaluate.py", "--checkpoint", ck / f"best_{a.backbone}.pt",
        "--test_manifest", man / "test.csv", "--data_root", ROOT, "--image_size", a.image_size,
        "--output_json", RES / "cnn" / "metrics.json", "--predictions_csv", RES / "cnn" / "predictions.csv"],
       log=RES / "cnn" / "evaluate_log.txt")


def step_rf(a):
    man = MAN / "rf"
    py(RF, f"""
from data.synthetic_metadata import generate_labeled_dataset
from data.dataset import stratified_split_manifest
generate_labeled_dataset(output_csv=r'{man / "manifest.csv"}', n_genuine={a.rf_genuine},
                         n_per_technique={a.rf_per_technique}, difficulty='{a.rf_difficulty}', seed={a.seed})
stratified_split_manifest(r'{man / "manifest.csv"}', r'{man / "train.csv"}', r'{man / "val.csv"}',
                          r'{man / "test.csv"}', seed={a.seed})
""", log=RES / "rf" / "data_log.txt")
    ck = OUT / "checkpoints" / "rf"
    sh([sys.executable, "-I", RF / "scripts" / "train.py", "--train_manifest", man / "train.csv",
        "--val_manifest", man / "val.csv", "--n_estimators", 300, "--output_dir", ck], log=RES / "rf" / "train_log.txt")
    sh([sys.executable, "-I", RF / "scripts" / "evaluate.py", "--checkpoint", ck / "rf_document_branch.joblib",
        "--test_manifest", man / "test.csv", "--output_json", RES / "rf" / "metrics.json",
        "--predictions_csv", RES / "rf" / "predictions.csv"], log=RES / "rf" / "evaluate_log.txt")


def step_siamese(a):
    man = MAN / "signature"
    py(SIG, f"""
from data.synthetic_signatures import generate_pairs_dataset
generate_pairs_dataset(pairs_dir=r'{OUT / "signature_pairs"}', manifest_dir=r'{man}', data_root=r'{ROOT}',
                       n_identities={a.sig_identities}, genuine_per_id={a.sig_pairs_per_id},
                       forged_per_id={a.sig_pairs_per_id}, fracs={tuple(a.sig_fracs)}, seed={a.seed})
""", log=RES / "siamese" / "data_log.txt")
    ck = OUT / "checkpoints" / "siamese"
    sh([sys.executable, "-I", SIG / "scripts" / "train.py", "--train_manifest", man / "train.csv",
        "--val_manifest", man / "val.csv", "--data_root", ROOT, "--epochs", a.sig_epochs, "--seed", a.seed,
        "--output_dir", ck], log=RES / "siamese" / "train_log.txt")
    sh([sys.executable, "-I", SIG / "scripts" / "evaluate.py", "--checkpoint", ck / "best_siamese.pt",
        "--test_manifest", man / "test.csv", "--data_root", ROOT, "--output_json", RES / "siamese" / "metrics.json",
        "--predictions_csv", RES / "siamese" / "predictions.csv"], log=RES / "siamese" / "evaluate_log.txt")


def step_fusion(a):
    man = MAN / "fusion"
    # Only HELD-OUT predictions (each branch's test split) are used, each underlying item at most once.
    sh([sys.executable, "-I", ROOT / "build_matched_cases.py", "--cnn_csv", RES / "cnn" / "predictions.csv",
        "--rf_csv", RES / "rf" / "predictions.csv", "--siamese_csv", RES / "siamese" / "predictions.csv",
        "--out_dir", man, "--seed", a.seed], log=RES / "fusion" / "build_cases_log.txt")
    ck = OUT / "checkpoints" / "fusion"
    sh([sys.executable, "-I", FUS / "scripts" / "train.py", "--train_manifest", man / "fusion_train.csv",
        "--val_manifest", man / "fusion_val.csv", "--output_dir", ck], log=RES / "fusion" / "train_log.txt")
    sh([sys.executable, "-I", FUS / "scripts" / "evaluate.py", "--checkpoint", ck / "fusion_meta_classifier.joblib",
        "--train_manifest", man / "fusion_train.csv", "--test_manifest", man / "fusion_test.csv",
        "--output_json", RES / "fusion" / "metrics.json", "--predictions_csv", RES / "fusion" / "predictions.csv",
        "--shap_csv", RES / "fusion" / "shap_values.csv", "--shap_sample_size", 60],
       log=RES / "fusion" / "evaluate_log.txt")
    sh([sys.executable, "-I", FUS / "scripts" / "cross_validate.py", "--manifest", man / "fusion_manifest.csv",
        "--output_json", RES / "fusion" / "cross_validation.json"], log=RES / "fusion" / "cross_validation_log.txt")


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def step_summary(a):
    rows = []
    for name, f, key in [("Visual document CNN", "cnn", "forgery"), ("Metadata/OCR/font Random Forest", "rf", "overall"),
                         ("Siamese signature CNN", "siamese", "forgery"), ("Fusion MLP (held-out cases)", "fusion", "overall")]:
        m = _load(RES / f / "metrics.json")
        if not m:
            continue
        m = m.get(key) or m.get("forgery") or m
        rows.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |")
    lines = ["# Results summary (generated by run_all.py; all data synthetic)", "",
             "| Component | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|", *rows, ""]
    cv = _load(RES / "fusion" / "cross_validation.json")
    if cv:
        lines += [f"Fusion cross-validation over all {cv['n_cases']} matched cases "
                  f"({cv['folds']}-fold x {cv['repeats']} repeats, mean +/- std):", "",
                  "| Model | ROC-AUC | F1 |", "|---|---|---|"]
        for k, v in cv["results"].items():
            lines.append(f"| {k} | {v['auc']['mean']:.3f} +/- {v['auc']['std']:.3f} | {v['f1']['mean']:.3f} +/- {v['f1']['std']:.3f} |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    ap.add_argument("--seed", type=int, default=42)
    # documents + CNN
    ap.add_argument("--n_docs", type=int, default=400)
    ap.add_argument("--cnn_fracs", type=float, nargs=3, default=[0.5, 0.1, 0.4],
                    help="train/val/test share of GENUINE DOCUMENTS (test is large because it also feeds the fusion layer)")
    ap.add_argument("--backbone", default="efficientnet_b0", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights (downloads them; off by default)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=None, help="frozen-backbone epochs (default 2 if pretrained, else 0)")
    ap.add_argument("--lr", type=float, default=None, help="base LR; fine-tuning uses lr/10 (default 1e-4 pretrained, 1e-3 from scratch)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=3)
    # RF
    ap.add_argument("--rf_genuine", type=int, default=800)
    ap.add_argument("--rf_per_technique", type=int, default=250)
    ap.add_argument("--rf_difficulty", default="medium", choices=["easy", "medium", "hard"])
    # Siamese
    ap.add_argument("--sig_identities", type=int, default=200)
    ap.add_argument("--sig_pairs_per_id", type=int, default=6, help="genuine pairs and forged pairs per identity")
    ap.add_argument("--sig_fracs", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    ap.add_argument("--sig_epochs", type=int, default=30)
    a = ap.parse_args()
    if a.warmup is None:   # freezing a randomly initialised backbone for warm-up would only train a head on noise
        a.warmup = 2 if a.pretrained else 0
    if a.lr is None:
        a.lr = 1e-4 if a.pretrained else 1e-3

    for s in ALL_STEPS:
        if s in a.steps:
            print(f"\n{'=' * 78}\nSTEP: {s}\n{'=' * 78}", flush=True)
            globals()[f"step_{s}"](a)


if __name__ == "__main__":
    main()
