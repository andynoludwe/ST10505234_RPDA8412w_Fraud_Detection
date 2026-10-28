# Document visual-tampering CNN ("Document stream, visual")

Detects **visual** manipulation of a rasterised financial document: splicing, copy-move and inpainting. It
complements the Random Forest branch (metadata / OCR / font signals). Output per document: `forgery_prob` plus
one probability per tamper type, in the format the fusion layer reads.

> All data are synthetic and fictional. Scores show that the pipeline works, not how it would perform on real
> bank documents.

## Layout

```
model/cnn_document_branch.py   DocumentTamperCNN (ResNet-50 or EfficientNet-B0, binary head + 3 tamper-type heads)
data/synthetic_tamper.py       tampering functions + generate_split_datasets() (leakage-free builder)
data/dataset.py                PyTorch Dataset (paths relative to --data_root), grouped split helper
scripts/train.py               training loop (optional frozen warm-up, early stopping on val ROC-AUC)
scripts/evaluate.py            test metrics, per-tamper-type metrics, predictions CSV for fusion
utils/metrics.py               precision / recall / F1 / ROC-AUC / PR-AUC / McNemar
```

## Data: what is generated and why it is built this way

1. **Genuine pages** are rendered by `tools/make_genuine_documents.py` (fictional bank statements and payslips
   with a signature; about 30% get a print-and-scan style finish). The earlier version used random-noise images
   with a black bar, which have no document structure to tamper with.
2. **Split first, tamper second** (`generate_split_datasets`). The genuine documents are divided into
   train / val / test, and the tampered variants of a split are made only from that split's documents, including
   the donor pages used for splicing. No source document, and no donor, is shared between splits. (The previous
   code tampered first and split afterwards, so most test documents also appeared in training.)
3. **Tampering targets content.** Rectangles are centred on text-bearing areas and re-drawn until at least 150
   pixels really change. On a mostly blank page a random rectangle often changes nothing, and an unchanged image
   labelled "forged" is label noise.
4. Manifest paths are **relative** to `--data_root` (the repo root), so they work on any machine.

## Run it

From the repository root, `python run_all.py --steps docs cnn` does all of the following. To run by hand:

```bash
python tools/make_genuine_documents.py --n 400 --out_dir outputs/genuine_documents
cd document_cnn_branch
python -c "from data.synthetic_tamper import generate_split_datasets as g; \
  g('../outputs/genuine_documents','../outputs/cnn_data','../data_manifests/cnn',data_root='..',fracs=(0.5,0.1,0.4))"
python scripts/train.py --train_manifest ../data_manifests/cnn/train.csv --val_manifest ../data_manifests/cnn/val.csv \
  --data_root .. --backbone efficientnet_b0 --epochs 8 --lr 1e-3 --patience 3 \
  --output_dir ../outputs/checkpoints/cnn --no_pretrained      # drop --no_pretrained (and use --lr 1e-4) when online
python scripts/evaluate.py --checkpoint ../outputs/checkpoints/cnn/best_efficientnet_b0.pt \
  --test_manifest ../data_manifests/cnn/test.csv --data_root .. \
  --output_json ../results/cnn/metrics.json --predictions_csv ../results/cnn/predictions.csv
```

The predictions CSV has `doc_id, filepath, y_true, forgery_prob, splicing_prob, copy_move_prob, inpainting_prob`.

## Result of the reported run: the CNN did not learn

EfficientNet-B0 trained from scratch on 800 images (200 documents x 4 variants), validated on 160, tested on
640 held-out images from 160 documents it never saw. **Test ROC-AUC 0.503; it scores every page as forged**
(confusion matrix `[[0, 160], [0, 480]]`). Its F1 of 0.857 is not skill: 75% of the test images are forged, so
"always forged" already gives precision 0.75, recall 1.00, F1 0.857. Read ROC-AUC, not F1, for this branch.

What was tried (logs in `results/cnn/`):

| Attempt | Setting | Best val ROC-AUC |
|---|---|---|
| `train_log_lr1e-2_attempt.txt` | lr 1e-2, 15 epochs planned, stopped by hand after epoch 10 | 0.533 |
| `train_log.txt` (reported) | lr 1e-3, early stopping after epoch 5 | 0.502 |

Why this is a model/setup limit and not a data defect: a plain gradient-boosting classifier on a few
native-resolution noise statistics (block-wise variance of the median-filter residual) reaches ROC-AUC 0.69 on
the same test images (splicing 0.72, copy-move 0.63, inpainting 0.71). That was a one-off check and is not part
of the pipeline; it only shows the tampering leaves a trace that the 224x224 from-scratch CNN cannot find.

Most likely fixes, in order: ImageNet-pretrained weights (`--pretrained`, needs internet), a GPU so that larger
inputs (`--image_size 384` or more) are affordable, more training images. None could be run in the environment
that produced these results (no internet, 2 CPU cores). Until it is re-run, treat the CNN column of every
downstream result as "uninformative branch".

## Things to know

- **Pretrained weights.** `--pretrained` (default in `train.py`) downloads ImageNet weights. The reported
  results in `results/cnn/` were produced *without* them (`--no_pretrained`, the download was blocked in the
  environment that produced them), so they are a floor, not a best case. Re-run with `python run_all.py
  --steps docs cnn --pretrained` on a machine with internet.
- **Backbones.** The proposal compares ResNet-50 and EfficientNet-B0. Only EfficientNet-B0 was trained for the
  reported run (ResNet-50 is about 2x slower on a CPU). Use `--backbone resnet50` to run the other.
- **Image size.** Pages are 660 x 900 pixels and are resized to 224x224, which discards fine forensic traces; larger
  inputs (`--image_size 384`) are a sensible next experiment.
- The old `stratified_split_manifest()` is kept but now groups by source document. It cannot keep splice donors
  inside one split, so use `generate_split_datasets()` for anything you report.
