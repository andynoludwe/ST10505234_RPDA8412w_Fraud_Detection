# Siamese signature verifier ("Signature stream")

Compares a claimed signature with a reference specimen. Output per pair: `forgery_prob` (higher = more likely the
claimed signature is **not** the reference writer's), the same convention as the other branches.

> **This component was missing from the repository** (the README listed it as "to be added") and has been
> rebuilt. It is a new implementation, not the original code, so the F1 = 0.83 quoted in the earlier report
> came from code that is no longer available and cannot be reproduced from here. Use the numbers in
> `results/siamese/` for the current code.

## Layout

```
model/siamese_cnn.py              shared-weight conv encoder -> |ea-eb| and ea*eb -> small head -> 1 logit
data/synthetic_signatures.py      stroke-based identities, genuine / skilled-forgery / random-forgery pairs
data/dataset.py                   Dataset (paths relative to --data_root)
scripts/train.py, evaluate.py     training (checkpoint on val ROC-AUC) and test evaluation
utils/metrics.py                  metrics + McNemar
```

## Data and leakage control

Label convention: `label` 1 = forged, 0 = genuine; `pair_type` is `genuine`, `skilled_forgery` (same strokes,
larger jitter; 60% of forged pairs) or `random_forgery` (a different identity; 40%).

**Identities are split first**, then pairs are built only from identities inside one split, so the test set
contains writers the model has never seen (verification, not recognition). The earlier split put all 20
identities in both train and test.

The jitter sizes that separate a forger from natural variation are assumptions; real signatures (with consent
and ethics clearance) would be needed before any claim about real forgeries.

## Run it

`python run_all.py --steps siamese` from the repository root, or by hand:

```bash
cd signature_siamese_branch
python -c "from data.synthetic_signatures import generate_pairs_dataset as g; \
  g('../outputs/signature_pairs','../data_manifests/signature',data_root='..',n_identities=200,genuine_per_id=6,forged_per_id=6)"
python scripts/train.py --train_manifest ../data_manifests/signature/train.csv --val_manifest ../data_manifests/signature/val.csv \
  --data_root .. --output_dir ../outputs/checkpoints/siamese
python scripts/evaluate.py --checkpoint ../outputs/checkpoints/siamese/best_siamese.pt \
  --test_manifest ../data_manifests/signature/test.csv --data_root .. \
  --output_json ../results/siamese/metrics.json --predictions_csv ../results/siamese/predictions.csv
```
