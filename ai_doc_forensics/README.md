# Detecting AI use on bank documents with synthetic data

A reproducible pipeline that generates **linked synthetic cases** (fictional bank statements and payslips, with metadata, arithmetic and a signature) and trains four detection branches plus a fusion layer to flag documents that are **fully AI-generated** or **AI-edited**.

> **Read this first.** Every document is synthetic. The "AI artefacts" (double-JPEG compression, font/quality mismatches, simulated metadata and OCR traces) are *assumptions I coded*, not measurements from real AI tools or real banks. High scores show the pipeline works and the leakage controls hold. They do **not** show it would catch real AI-forged documents. Validate on real data (with ethics and POPIA clearance) before any such claim.

## Quick start
```
pip install -r requirements.txt
python run_pipeline.py --n_cases 600 --difficulty medium --out runs_trial      # ~2 min
python run_pipeline.py --n_cases 1800 --difficulty all --out runs --n_jobs 4   # full study (~10 min)
pytest tests/
```
Or open `run_project.ipynb`.

## Design
- One `case_id` ties together: rendered JPEG, fictional bank/employer (Karoo Trust Bank, Marula Logistics), metadata record, arithmetic data and signature identity.
- Labels: `genuine`, `ai_generated`, `ai_edited` (manipulation type recorded: amount edit, closing-balance edit, date edit, signature swap, with or without recalculation). Priors 60/15/25.
- Genuine and manipulated cases overlap on purpose (rescanned genuine documents, genuine arithmetic flaws, clean AI edits). Difficulty presets in `adf/config.py`: easy, medium, hard, adversarial.

## Branches (binary target: AI involved)
| Branch | Features | Model |
|---|---|---|
| Image forensics | ELA (q75/90/95), block stats, noise residual, FFT, Laplacian, blockiness | HistGradientBoosting |
| Metadata / OCR / font | producer, timestamps, text layer, font consistency, OCR mismatch | RandomForest |
| Arithmetic | balance reconciliation, totals, Benford chi2, payslip PAYE/UIF | HistGradientBoosting |
| Signature | SSIM, correlation, HOG cosine, stroke width | RandomForest (classical stand-in for a Siamese CNN) |

Fusion: logistic regression and small gradient boosting on logits of **out-of-fold** branch probabilities. Baselines: mean of branches, early fusion, best single branch (picked on train OOF AUC).

## Evaluation protocol
- Split by **signature identity** (GroupShuffleSplit 70/30); 5-fold StratifiedGroupKFold for train OOF predictions. No identity appears on both sides.
- Thresholds chosen by F1 on train OOF only; test set used once.
- McNemar (exact if discordant < 25), bootstrap CIs, ablation, permutation importance, optional SHAP, subgroup flagged rates, 3-class type task.

## Results (1800 cases per level, held-out test)
| Level | Best single branch AUC | Mean of branches | Early fusion | Fusion LR | Fusion LR F1 |
|---|---|---|---|---|---|
| easy | 0.990 | 0.998 | 0.999 | 0.999 | 0.993 |
| medium | 0.982 | 0.995 | 0.996 | 0.997 | 0.961 |
| hard | 0.952 | 0.981 | 0.990 | 0.987 | 0.939 |
| adversarial | 0.934 | 0.979 | 0.993 | 0.982 | 0.928 |

Fusion clearly beats the best single branch (adversarial McNemar p < 1e-8). Fusion LR vs mean-of-branches and vs early fusion is **not** significant at the adversarial level (p = 0.36, 0.09), and early fusion has the higher AUC there. Adversarial flagged rates: AI-generated 100%, AI-edited 87%, genuine false positives 4%. Edits with recalculated totals are hardest. Details: `runs/<level>/results/results.json`.

## Plugging in your own models
Write `case_id,split,p_ai` CSVs per branch (e.g. your Siamese/CNN) into `predictions/` and use `adf.combine.join_predictions`. Train predictions must be out-of-fold, keyed by `case_id`.

## Limitations
- Simulated artefacts and metadata; not validated against real generative tools.
- Latency and memory were not benchmarked.
- The signature branch is not a deep Siamese network.
- If your submitted report describes a different system, tell your supervisor which code backs which claims.
- Fictional entities only; no real personal data (POPIA).
