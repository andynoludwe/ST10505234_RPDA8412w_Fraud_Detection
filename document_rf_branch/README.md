# Document metadata / OCR / font Random Forest ("Document stream, tabular")

Detects **metadata-level** manipulation: font mismatches, edited timestamps, an unexpected PDF producer, OCR
confidence drops, layout shifts. Output per document: `forgery_prob` plus `tampered_text_prob`,
`altered_figures_prob`, `metadata_anomaly_prob`, `layout_inconsistency_prob`.

> All features are synthetic: drawn from distributions I assumed, not extracted from real PDFs.

## Layout

```
model/rf_document_branch.py    DocumentTamperRF (forgery head + multi-output anomaly head)
data/synthetic_metadata.py     feature generator with a difficulty setting
data/dataset.py                CSV loading + stratified 60/20/20 split
scripts/train.py, evaluate.py  training and test evaluation (+ predictions CSV for fusion)
utils/metrics.py               metrics + McNemar
```

## The generator and class overlap

With `difficulty="easy"` (the old behaviour) genuine and forged rows never overlap, every model scores 1.00
and the result says nothing. The default is now `"medium"`:

| difficulty | what changes | test ROC-AUC of this RF (3 seeds) |
|---|---|---|
| easy | no overlap | 1.000 |
| medium (default) | 50% of forgeries only partly visible; 30% of genuine documents look odd (re-scanned, re-saved, odd producer) | about 0.98 |
| hard | 80% subtle forgeries; 50% odd genuine documents | about 0.93 |

The overlap rates are assumptions. Real metadata would need to be extracted (e.g. `pikepdf`, `pytesseract`) and
dropped in with the same column schema (`FEATURE_NAMES`, `ANOMALY_TYPES`).

## Run it

`python run_all.py --steps rf` from the repository root, or by hand:

```bash
cd document_rf_branch
python -c "from data.synthetic_metadata import generate_labeled_dataset as g; \
  g('../data_manifests/rf/manifest.csv', n_genuine=800, n_per_technique=250, difficulty='medium')"
python -c "from data.dataset import stratified_split_manifest as s; \
  s('../data_manifests/rf/manifest.csv','../data_manifests/rf/train.csv','../data_manifests/rf/val.csv','../data_manifests/rf/test.csv')"
python scripts/train.py --train_manifest ../data_manifests/rf/train.csv --val_manifest ../data_manifests/rf/val.csv --output_dir ../outputs/checkpoints/rf
python scripts/evaluate.py --checkpoint ../outputs/checkpoints/rf/rf_document_branch.joblib \
  --test_manifest ../data_manifests/rf/test.csv --output_json ../results/rf/metrics.json \
  --predictions_csv ../results/rf/predictions.csv
```

`class_weight="balanced"` handles the genuine:forged imbalance; `model.feature_importances()` gives the per-branch
importances used alongside the fusion-level SHAP.
