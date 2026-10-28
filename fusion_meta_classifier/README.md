# Fusion meta-classifier + SHAP

Combines the three branches into one fraud probability with a small MLP (scikit-learn, standardised inputs) and
explains each score with SHAP.

Inputs per case (10 numbers, all "higher = more suspicious"):
`cnn_forgery_prob, cnn_splicing_prob, cnn_copy_move_prob, cnn_inpainting_prob, rf_forgery_prob,
rf_tampered_text_prob, rf_altered_figures_prob, rf_metadata_anomaly_prob, rf_layout_inconsistency_prob,
siamese_forgery_prob`.

## Which data it must be trained on

There are two data sources, and only one of them says anything about the framework:

| Source | What it is | Use for |
|---|---|---|
| `build_matched_cases.py` (repo root) | Joins the three branches' **real held-out predictions** into cases | **All reported results** |
| `data/synthetic_cases.py` | Draws the 10 probabilities directly from assumed ranges, no branch involved | Plumbing smoke test only |

The earlier run (F1 = 1.00, "p < 0.05 against all three branches") used `synthetic_cases.py`. Those scores were
made by the generator and say nothing about the fusion idea; they are not reported any more.

### How a matched case is built (`build_matched_cases.py`)

A case is one document image (CNN score) + one metadata record (RF score) + one signature pair (Siamese score):

| scenario | document | signature | case label |
|---|---|---|---|
| genuine | genuine | genuine | 0 |
| doc_forged | forged | genuine | 1 |
| sig_forged | genuine | forged | 1 |
| both_forged | forged | forged | 1 |

The CNN and RF items in one case always share the same genuine/forged status. Leakage rules enforced there:
only each branch's **test-split** predictions are used (never data a branch trained or tuned on); every
underlying item is used in at most one case; the 60/20/20 case split is stratified by scenario.

The three branches are still independent datasets, so "the same case" is constructed by pairing, not observed.
In real data one application would carry a document, its metadata and a signature natively, and this pairing
step would disappear.

## Scripts

```bash
python scripts/train.py --train_manifest fusion_train.csv --val_manifest fusion_val.csv --output_dir checkpoints/
python scripts/evaluate.py --checkpoint checkpoints/fusion_meta_classifier.joblib --train_manifest fusion_train.csv \
   --test_manifest fusion_test.csv --output_json m.json --predictions_csv p.csv --shap_csv shap.csv
python scripts/cross_validate.py --manifest fusion_manifest.csv --output_json cv.json
```

- `evaluate.py`: test metrics, McNemar's test of fusion vs each branch (each branch thresholded at 0.5), KernelExplainer SHAP.
- `cross_validate.py` (new): repeated stratified k-fold over **all** matched cases, comparing the MLP with logistic
  regression, mean-of-branches, max-of-branches and each branch alone on identical folds. The held-out case set
  is small, so a single split and a McNemar test are noisy; this gives a mean and spread.

`python run_all.py --steps fusion` runs all of this on the current branch outputs.
