"""
Join the three branches' REAL predictions_csv outputs into one case-level
feature table for the fusion meta-classifier, once you have real, matched
cases running through all three branches (see synthetic_cases.py's
docstring for why this is separate from the smoke-test path).

Prerequisite: each branch's evaluate.py must be scoring documents/pairs
that share a common case_id across all three branches -- i.e. you'll need
to add a case_id column when you generate real (or more realistic
synthetic) data for each branch, so a single case's document image,
metadata record, and signature pair are all identifiable as belonging to
the same case.

Usage
-----
python -c "
from data.combine_predictions import combine
combine(
    cnn_csv='../document_cnn_branch/results/resnet50_predictions.csv',
    rf_csv='../document_rf_branch/results/rf_predictions.csv',
    siamese_csv='../signature_siamese_branch/results/siamese_predictions.csv',
    case_labels_csv='cases_ground_truth.csv',   # case_id -> label, if not already in each branch's output
    output_csv='fusion_manifest.csv',
)
"
"""

import csv
from pathlib import Path

from data.synthetic_cases import CNN_ANOMALY_TYPES, RF_ANOMALY_TYPES


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def combine(
    cnn_csv: str,
    rf_csv: str,
    siamese_csv: str,
    case_labels_csv: str,
    output_csv: str,
    cnn_id_col: str = "case_id",
    rf_id_col: str = "case_id",
    siamese_id_col: str = "case_id",
) -> None:
    """
    Joins three branch prediction CSVs on a shared case_id column. Each
    input CSV must have been re-run with case_id substituted for its
    original doc_id/pair_id key (see this module's docstring).
    """
    cnn_rows = {r[cnn_id_col]: r for r in _read_csv(cnn_csv)}
    rf_rows = {r[rf_id_col]: r for r in _read_csv(rf_csv)}
    siamese_rows = {r[siamese_id_col]: r for r in _read_csv(siamese_csv)}
    labels = {r["case_id"]: int(r["label"]) for r in _read_csv(case_labels_csv)}

    common_ids = set(cnn_rows) & set(rf_rows) & set(siamese_rows) & set(labels)
    if not common_ids:
        raise ValueError(
            "No case_id values are shared across all three branch CSVs and the "
            "labels file. Confirm all three branches were run against the same "
            "case_id-keyed dataset."
        )

    fieldnames = ["case_id", "cnn_forgery_prob"] + [f"cnn_{t}_prob" for t in CNN_ANOMALY_TYPES] \
        + ["rf_forgery_prob"] + [f"rf_{t}_prob" for t in RF_ANOMALY_TYPES] \
        + ["siamese_forgery_prob", "label"]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(common_ids):
            row = {"case_id": cid, "label": labels[cid]}
            row["cnn_forgery_prob"] = cnn_rows[cid]["forgery_prob"]
            for t in CNN_ANOMALY_TYPES:
                row[f"cnn_{t}_prob"] = cnn_rows[cid].get(f"{t}_prob", 0.0)
            row["rf_forgery_prob"] = rf_rows[cid]["forgery_prob"]
            for t in RF_ANOMALY_TYPES:
                row[f"rf_{t}_prob"] = rf_rows[cid].get(f"{t}_prob", 0.0)
            row["siamese_forgery_prob"] = siamese_rows[cid]["forgery_prob"]
            writer.writerow(row)

    print(f"Combined {len(common_ids)} matched cases into {output_csv}")
