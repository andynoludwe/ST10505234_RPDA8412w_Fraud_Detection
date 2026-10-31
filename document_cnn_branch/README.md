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

## Result of the reported run

EfficientNet-B0 with ImageNet weights, trained on a Colab T4 GPU (`--pretrained`, lr 1e-4, 2 warm-up epochs,
15 epochs), on 800 images (200 documents x 4 variants), validated on 160, tested on 640 held-out images from
160 documents it never saw. Best validation ROC-AUC 0.765, reached at epoch 15 (the last), so training was
stopped by the epoch cap, not by early stopping.

| Test (640 images) | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| All forgeries | 0.740 | 0.963 | 0.433 | 0.598 |
| Splicing | 0.648 | no positives predicted | 0 | 0 |
| Copy-move | 0.655 | 0.526 | 0.063 | 0.11 |
| Inpainting | 0.959 | 1.000 | 0.656 | 0.793 |

Confusion matrix (rows true, columns predicted): `[[152, 8], [272, 208]]`. The model is cautious: few false
alarms, but it misses more than half of the forged pages at the 0.5 threshold. Inpainting leaves a strong trace;
splicing and copy-move are close to chance.

Earlier attempt, for the record: the same network trained from scratch on 2 CPU cores stayed at chance (test
ROC-AUC 0.503, it scored every page as forged; logs `results/cnn/train_log_lr1e-2_attempt.txt`). A plain
gradient-boosting classifier on native-resolution noise statistics reached ROC-AUC 0.69 on the same images, so
the tampering is detectable and pretrained weights plus a GPU made the difference.

Next steps: train longer (`--epochs 40 --patience 8`; validation AUC was still rising), try larger inputs
(`--image_size 384`), tune the decision threshold on the validation set (recall is the weak point), and generate
more splicing / copy-move examples. The numbers above are one run with seed 42.

## Things to know

- **Pretrained weights.** `--pretrained` (default in `train.py`) downloads ImageNet weights. The reported
  results in `results/cnn/` used them, on a Colab T4 GPU (notebook `RPDA8412w_CNN_pretrained_Colab.ipynb`).
  Without internet use `--no_pretrained`; expect chance-level results on CPU.
- **Backbones.** The proposal compares ResNet-50 and EfficientNet-B0. Only EfficientNet-B0 was trained for the
  reported run (ResNet-50 is about 2x slower on a CPU). Use `--backbone resnet50` to run the other.
- **Image size.** Pages are 660 x 900 pixels and are resized to 224x224, which discards fine forensic traces; larger
  inputs (`--image_size 384`) are a sensible next experiment.
- The old `stratified_split_manifest()` is kept but now groups by source document. It cannot keep splice donors
  inside one split, so use `generate_split_datasets()` for anything you report.
