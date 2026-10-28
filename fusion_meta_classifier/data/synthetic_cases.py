"""
Synthetic case-level dataset generator for the fusion meta-classifier.

IMPORTANT DESIGN NOTE (read this first)
----------------------------------------
The three branches you already built (document_cnn_branch, document_rf_branch,
signature_siamese_branch) each generate their OWN independent synthetic
dataset, with their own sample counts and their own identity/doc IDs. They
were never designed to share a common case_id, because each was smoke-tested
in isolation.

The fusion meta-classifier, by definition, needs one row per real-world
*case* (e.g. one loan application), where that case has:
    - one document image  -> scored by the CNN branch
    - one metadata record -> scored by the RF branch
    - one signature pair  -> scored by the Siamese CNN branch
...and ONE ground-truth fraud label for the whole case.

Two ways forward:

1. (What this file does for now) Generate a synthetic CASE-LEVEL dataset
   directly -- i.e. simulate what the three branches' *outputs* would look
   like for N synthetic cases, without re-running the three branches
   themselves. This lets the fusion + SHAP pipeline be fully built and
   smoke-tested today.

2. (The real next step once you have real cases) Modify each branch's data
   generator so it accepts a shared case_id and produces its output keyed by
   that case_id, then join the three real predictions_csv files on case_id
   instead of using this synthetic generator. See combine_predictions.py's
   docstring for exactly what that join should look like.
"""

import csv
import random
from pathlib import Path

CNN_ANOMALY_TYPES = ["splicing", "copy_move", "inpainting"]
RF_ANOMALY_TYPES = ["tampered_text", "altered_figures", "metadata_anomaly", "layout_inconsistency"]

FEATURE_COLUMNS = (
    ["cnn_forgery_prob"] + [f"cnn_{t}_prob" for t in CNN_ANOMALY_TYPES]
    + ["rf_forgery_prob"] + [f"rf_{t}_prob" for t in RF_ANOMALY_TYPES]
    + ["siamese_forgery_prob"]
)


def _genuine_case(rng: random.Random) -> dict:
    row = {"cnn_forgery_prob": rng.uniform(0.0, 0.25)}
    for t in CNN_ANOMALY_TYPES:
        row[f"cnn_{t}_prob"] = rng.uniform(0.0, 0.2)
    row["rf_forgery_prob"] = rng.uniform(0.0, 0.2)
    for t in RF_ANOMALY_TYPES:
        row[f"rf_{t}_prob"] = rng.uniform(0.0, 0.2)
    row["siamese_forgery_prob"] = rng.uniform(0.0, 0.25)
    return row


def _forged_case(rng: random.Random) -> dict:
    """
    A forged case doesn't necessarily trip every branch -- e.g. a forged
    signature on an otherwise-clean document. This models realistic partial
    agreement between branches rather than all three always agreeing.
    """
    row = _genuine_case(rng)

    # Pick which branch(es) actually catch this particular forgery.
    n_branches_triggered = rng.choice([1, 1, 2, 2, 3])
    triggered = rng.sample(["cnn", "rf", "siamese"], k=n_branches_triggered)

    if "cnn" in triggered:
        row["cnn_forgery_prob"] = rng.uniform(0.6, 0.98)
        hot_type = rng.choice(CNN_ANOMALY_TYPES)
        row[f"cnn_{hot_type}_prob"] = rng.uniform(0.6, 0.95)
    if "rf" in triggered:
        row["rf_forgery_prob"] = rng.uniform(0.6, 0.98)
        hot_type = rng.choice(RF_ANOMALY_TYPES)
        row[f"rf_{hot_type}_prob"] = rng.uniform(0.6, 0.95)
    if "siamese" in triggered:
        row["siamese_forgery_prob"] = rng.uniform(0.6, 0.98)

    return row


def generate_case_level_dataset(
    output_csv: str = "fusion_manifest.csv",
    n_genuine: int = 200,
    n_forged: int = 200,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    rows = []

    for i in range(n_genuine):
        row = _genuine_case(rng)
        row["case_id"] = f"case_{i:05d}"
        row["label"] = 0
        rows.append(row)

    for i in range(n_forged):
        row = _forged_case(rng)
        row["case_id"] = f"case_{n_genuine + i:05d}"
        row["label"] = 1
        rows.append(row)

    rng.shuffle(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id"] + FEATURE_COLUMNS + ["label"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to {output_csv} ({n_genuine} genuine, {n_forged} forged).")
