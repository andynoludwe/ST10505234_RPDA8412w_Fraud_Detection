# Document Visual Tampering CNN — "Document Stream (Visual)"

This is the deep CNN document branch from Table 2 of your proposal
("A Hybrid Multimodal Machine Learning Framework for Detecting Fraudulent
Financial Documents and Forged Signatures in South African Digital
Banking"). It detects **visual** manipulation — splicing, copy-move,
inpainting — as distinct from your existing tabular model, which handles
metadata/OCR/font anomalies.

## How this fits your existing pipeline

You already have (per your notes):
1. Synthetic bank-statement generator (tabular-level tampering)
2. Document-branch Random Forest baseline
3. Siamese CNN architecture (signature stream)
4. Fusion meta-classifier + SHAP explainability

This module is a **new, independent visual stream** that slots in
alongside #2 and #3, feeding into #4. Its output — `forgery_prob` per
document, plus per-tamper-type probabilities — is exactly the kind of
score your fusion/stacking meta-classifier expects (see
`scripts/evaluate.py`'s `--predictions_csv` output, or call
`get_predictions()` directly).

```
Signature image ──► Siamese CNN ──────────────┐
                                                 ├──► Fusion / stacking meta-classifier ──► fraud risk score
Document image ──► DocumentTamperCNN (this) ───┤        (your existing component #4)
Document metadata/OCR ──► Random Forest ────────┘
```

## Project structure

```
document_cnn_branch/
├── model/
│   └── cnn_document_branch.py   # DocumentTamperCNN (ResNet50 / EfficientNet-B0)
├── data/
│   ├── synthetic_tamper.py      # generates labeled splicing/copy-move/inpainting examples
│   └── dataset.py                # PyTorch Dataset + stratified 60/20/20 split
├── scripts/
│   ├── train.py                  # training loop (frozen warm-up -> fine-tune, early stopping)
│   └── evaluate.py               # test-set metrics + per-document predictions for fusion
├── utils/
│   └── metrics.py                # precision/recall/F1/ROC-AUC/PR-AUC + McNemar's test
└── README.md
```

Every file above was smoke-tested end-to-end in this environment (data
generation → dataset loading → 2-epoch training → checkpointing →
evaluation → predictions export) — the mechanics all work. Metrics from
the smoke test are meaningless (they run on random noise images, not
real documents) — they're there to confirm plumbing, not model quality.

## Step-by-step: from your rasterised documents to a trained model

### 1. Generate labeled visual-tampering data

You'll already have genuine document images from Section 7.4
(post-rasterisation). Point the generator at that folder:

```python
from data.synthetic_tamper import generate_labeled_dataset

generate_labeled_dataset(
    genuine_dir="path/to/rasterised_genuine_documents",
    output_dir="data/synthetic_tampered",
    manifest_path="data/manifest.csv",
    n_per_technique=300,   # scale this up as you expand your dataset
)
```

This mirrors the four techniques named in Section 7.3 of your proposal
(copy-move, text substitution [handled by your tabular generator],
splicing, signature distortion [handled by your Siamese stream]) — this
module covers the **visual** three: splicing, copy-move, inpainting.

### 2. Stratified train/val/test split (60/20/20, per Section 7.3)

```python
from data.dataset import stratified_split_manifest

stratified_split_manifest(
    "data/manifest.csv",
    "data/train.csv", "data/val.csv", "data/test.csv",
)
```

### 3. Train

```bash
python scripts/train.py \
    --train_manifest data/train.csv \
    --val_manifest data/val.csv \
    --backbone resnet50 \
    --epochs 30 \
    --warmup_epochs 3 \
    --batch_size 16 \
    --output_dir checkpoints/
```

Run this twice — once with `--backbone resnet50`, once with
`--backbone efficientnet_b0` — and keep whichever has the better
validation F1, per Section 7.2 ("the better-performing one selected for
the fusion layer").

**Note on pretrained weights:** if you're training somewhere with
restricted internet access (as this sandbox is), add `--no_pretrained`.
On your own machine or Colab, leave `--pretrained` on (the default) to
get the ImageNet transfer-learning benefit described in Section 10.2.

### 4. Evaluate on the held-out test set

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_resnet50.pt \
    --test_manifest data/test.csv \
    --output_json results/resnet50_metrics.json \
    --predictions_csv results/resnet50_predictions.csv
```

This prints precision/recall/F1/ROC-AUC/PR-AUC (matching RQ1's metric
list exactly) plus a breakdown per tamper type, and writes a CSV of
per-document forgery probabilities — feed this CSV into your fusion
meta-classifier alongside the Random Forest's and Siamese CNN's outputs.

### 5. Statistical comparison against baselines (McNemar's test)

Once you have predictions from this CNN branch *and* from a
single-modality baseline (e.g. the tabular model alone) on the same test
set, use `utils.metrics.mcnemar_test(y_true, y_pred_cnn, y_pred_baseline)`
to test whether the difference is statistically significant — this is
what Section 7.5 commits to reporting.

## Design choices worth knowing about

- **Multi-label + binary head**: the model outputs both an overall
  forgery probability (what the fusion layer consumes) and per-technique
  probabilities (splicing/copy-move/inpainting), since a document can
  show more than one manipulation and the breakdown is useful for
  interpretability (Section 8, "Transparency and Interpretability").
- **Class-imbalance handling**: `train.py` computes a `pos_weight` for
  `BCEWithLogitsLoss` from the genuine:forged ratio automatically, so
  you don't need to manually balance the dataset.
- **Warm-up then fine-tune**: the backbone is frozen for the first
  `--warmup_epochs`, then unfrozen — this is a standard transfer-learning
  schedule that trains faster and is less likely to destroy the
  pretrained features early on, which matters given your limited compute
  budget (Section 10.2).
- **`embedding` output**: the model also returns a 256-dim embedding
  before the classification heads — useful if you want to feed richer
  features (not just the scalar probability) into a stacking
  meta-learner, or for SHAP analysis alongside your existing tabular
  SHAP work.

## Suggested next steps

1. Swap the smoke-test synthetic images for your real rasterised
   documents and re-run the generator with a larger `n_per_technique`.
2. Train both backbones properly (30+ epochs) once you're outside this
   sandbox and pretrained weights can download.
3. Wire `evaluate.py`'s prediction CSV into your fusion meta-classifier
   alongside the Random Forest and Siamese CNN outputs.
4. Add the CNN's tamper-type probabilities to your SHAP/interpretability
   writeup for Section 8.
