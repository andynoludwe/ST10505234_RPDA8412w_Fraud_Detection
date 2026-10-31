# ST10505234_RPDA8412w_Fraud_Detection

**A Hybrid Multimodal Machine Learning Framework for Detecting Fraudulent Financial Documents and Forged Signatures in South African Digital Banking**

Student: Andisiwe Noludwe (ST10505234)
Supervisor: Dr Joseph Omidosu
Programme: Postgraduate Diploma in Data Analytics, The Independent Institute of Education
Module: Research Project (RPDA8412w)
Ethical clearance reference: CSREC/ST10505234/085

This repository holds the code for the four model components described in the research report. **All data are
fully synthetic and fictional.** No human participants, real documents, real signatures, real banks or personal
information were used. Scores show that the pipeline works end to end; they are **not** estimates of real-world
performance.

```
Signature image ───────► Siamese CNN ───────────────────┐
Document image ────────► Document CNN (visual) ─────────┼──► Fusion meta-classifier ──► fraud probability (+ SHAP)
Document metadata/OCR/font ─► Random Forest ────────────┘
```

Each branch outputs a forgery probability (higher = more suspicious) in the same format, so a small neural
meta-classifier can combine them. SHAP explains the fused score.

## Status

| Component | Folder | Status |
|---|---|---|
| Document visual tampering CNN | `document_cnn_branch/` | Code, data builder, results. Pretrained EfficientNet-B0 learns (test ROC-AUC 0.74) but is the weakest branch; strong on inpainting, weak on splicing and copy-move (see Results) |
| Metadata / OCR / font Random Forest | `document_rf_branch/` | Code and results |
| Siamese signature CNN | `signature_siamese_branch/` | Code and results. **Rebuilt**: the original was missing from the repository |
| Fusion meta-classifier + SHAP | `fusion_meta_classifier/` | Code and results, run on matched cases built from the branches' held-out predictions |

Each folder has its own `README.md`. `extras/ai_doc_forensics/` is a separate, earlier study of AI-generated and
AI-edited documents; it also supplies the fictional page renderer used to make the CNN's genuine documents.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+, tested on 3.13
python run_all.py                        # everything: documents, 3 branches, fusion, summary
python run_all.py --steps rf siamese     # or only some steps (docs cnn rf siamese fusion summary)
python run_all.py --pretrained           # CNN with ImageNet weights (needs internet; GPU strongly recommended)
pytest tests/                            # leakage and portability checks, about 20 seconds
```

`RPDA8412_end_to_end_run.ipynb` is a thin notebook driver around `run_all.py`. Start Jupyter in the repository
folder. Everything is seeded (`--seed 42`). Manifests in `data_manifests/` use paths relative to the repository
root, so the code runs on any machine. Images and trained weights go to `outputs/` and are git-ignored because
they are large and reproducible; logs, metrics and prediction CSVs are kept in `results/`.

On 2 CPU cores without pretrained weights the CNN step takes roughly 3 to 4 minutes per epoch; the other steps
take a few minutes each.

## Results (synthetic data, held-out test splits; `results/summary.md`)

| Component | Test set | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Visual document CNN (ImageNet-pretrained) | 640 images from 160 unseen documents | 0.963 | 0.433 | 0.598 | **0.740** | 0.908 |
| Metadata/OCR/font Random Forest | 360 records | 0.960 | 0.950 | 0.955 | 0.989 | 0.992 |
| Siamese signature CNN | 480 pairs from 40 unseen writers | 0.945 | 0.787 | 0.859 | 0.927 | 0.949 |
| Fusion MLP | 49 matched cases | 0.920 | 0.920 | 0.920 | 0.962 | 0.969 |

Fusion, repeated 5-fold cross-validation over all 241 matched cases (10 repeats, mean +/- std across resamples
of the same cases; not a confidence interval for new data):

| Model | ROC-AUC | F1 |
|---|---|---|
| Fusion MLP | 0.960 +/- 0.024 | 0.909 +/- 0.038 |
| Logistic regression on the same 10 inputs | 0.969 +/- 0.021 | 0.926 +/- 0.036 |
| Mean of the 3 branch scores (no learning) | 0.968 +/- 0.021 | 0.739 +/- 0.065 |
| Max of the 3 branch scores | 0.965 +/- 0.021 | 0.905 +/- 0.034 |
| Random Forest alone | 0.826 +/- 0.056 | 0.754 +/- 0.065 |
| Siamese alone | 0.794 +/- 0.058 | 0.700 +/- 0.078 |
| CNN alone | 0.645 +/- 0.082 | 0.430 +/- 0.106 |

### How to read these results

- **The visual CNN is the weakest branch, and its F1 is held back by recall.** With ImageNet weights on a GPU it
  reaches test ROC-AUC 0.740 (the earlier from-scratch CPU run was at chance, 0.503). At the 0.5 threshold it is
  very precise (0.963) but finds only 43% of forged pages (208 of 480). Per type: inpainting is found well
  (ROC-AUC 0.959, precision 1.0, recall 0.66); splicing (0.648, nothing flagged) and copy-move (0.655, recall
  0.06) are barely above chance. Training stopped at the 15-epoch cap with validation ROC-AUC still rising
  (0.765 at epoch 15), so a longer run (`--epochs 40`) will probably improve it. Details:
  `document_cnn_branch/README.md`.
- **Fusion clearly beats every single branch** (cross-validated ROC-AUC 0.96 to 0.97 against at most 0.83), as it
  should: a case is fraudulent if the document *or* the signature is forged, and each branch sees only one of the
  two. This is why single-branch AUCs on matched cases (RF 0.826) are lower than on their own test sets (0.989).
- **The learned MLP is not better than simple averaging or logistic regression** (0.960 against 0.968 and 0.969;
  differences are within one standard deviation). With 241 cases there is no evidence here that a learned
  meta-classifier adds value over a mean. McNemar on the 49 test cases: fusion vs CNN p = 0.0013, vs RF
  p = 0.23, vs Siamese p = 0.027; with 49 cases these tests are very noisy.
- **SHAP importance is dominated by the RF and Siamese scores** (mean |SHAP| 0.217 and 0.213). The CNN inputs
  contribute much less (`cnn_copy_move_prob` 0.039, `cnn_inpainting_prob` 0.027, `cnn_forgery_prob` 0.023).
  The SHAP values come from 49 test cases and are indicative only.
- **The Random Forest and Siamese scores depend on assumptions in the generators** (how far genuine and forged
  feature distributions overlap, how large a forger's jitter is). They show the pipeline is sound; they do not
  show real forgeries would be caught. The Siamese branch finds skilled forgeries much harder than random ones
  (flagged 65.7% against 97.9%; false-positive rate on genuine pairs 4.6%).

## What changed from the first version of this repository

| Earlier | Now |
|---|---|
| Only the CNN code was uploaded; RF and fusion were zipped, the Siamese code was missing | All four components are real folders and runnable |
| CNN trained on 36 placeholder noise images; test set shared 6 of its 8 source documents with training | 800 training images from rendered fictional bank documents; documents are split first, so no source document or splice donor crosses splits (`tests/`) |
| Siamese: all 20 identities in both train and test | Identities split first: the test set contains unseen writers |
| RF "separated classes perfectly" by construction | Genuine and forged feature ranges overlap (`--rf_difficulty`); `easy` reproduces the old behaviour |
| Fusion evaluated on probabilities drawn from assumed ranges (F1 = 1.00) | Fusion built from the three branches' real held-out predictions (`build_matched_cases.py`), plus cross-validation against baselines |
| Manifests held `C:\Users\...` paths | Relative paths; no personal paths in the repository |

**The numbers in the research report must be reconciled with this table.** In particular the earlier Siamese
F1 of 0.83 (64 test pairs) came from code that is no longer available and cannot be reproduced; the 0.859 above
comes from the rebuilt code on a different, larger split. The same applies to the RF result and the fusion
statement that its evaluation "did not complete".

## Repository layout

```
run_all.py                  one command for the whole pipeline
build_matched_cases.py      builds fusion cases from held-out branch predictions (one item used in at most one case)
RPDA8412_end_to_end_run.ipynb   notebook driver for run_all.py
document_cnn_branch/  document_rf_branch/  signature_siamese_branch/  fusion_meta_classifier/
tools/make_genuine_documents.py   renders fictional genuine bank statements / payslips
tests/                      no-leakage and portability tests (pytest)
data_manifests/             split manifests (relative paths)
results/                    logs, metrics.json, predictions.csv, shap_values.csv, summary.md
extras/ai_doc_forensics/    separate study: detecting AI-generated / AI-edited bank documents (own README)
```

## Limitations

- All data are synthetic; artefacts, difficulty settings and forger behaviour are assumptions coded by the author.
- Datasets are small, each model was run once with seed 42, and latency and memory were not measured.
- The visual CNN is weak on splicing and copy-move and was stopped at 15 epochs while still improving; ResNet-50 was not trained.
- Cases are constructed by pairing items from independent branch datasets, not observed as one application.
- Validation on real, approved banking data (with ethics and POPIA clearance) is future work. See Sections 6 and
  7 of the research report.
