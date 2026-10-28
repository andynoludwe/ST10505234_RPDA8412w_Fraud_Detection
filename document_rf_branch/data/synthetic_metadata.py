"""
Synthetic generator for document metadata / OCR / font-consistency features.

Mirrors the visual CNN branch's synthetic_tamper.py pattern, but for the
tabular Random Forest baseline: instead of pixel-level splicing/copy-move,
this generates *feature rows* representing metadata-level tampering
signatures (mismatched fonts, edited PDF metadata, OCR inconsistencies,
layout shifts).

Anomaly types (multi-label, per the proposal's Section 8 breakdown):
    - tampered_text          : OCR confidence drop / char-error-rate spike
    - altered_figures        : embedded-object / compression-artifact signals
    - metadata_anomaly       : creation/modification timestamp mismatch,
                                unexpected PDF producer tool
    - layout_inconsistency   : alignment / whitespace / text-density variance

A document can carry more than one anomaly type at once (matching the CNN
branch's multi-label design), and every forged row keeps at least one
anomaly flag set so the labels stay consistent with the overall
forgery label.
"""

import csv
import random
from pathlib import Path

ANOMALY_TYPES = [
    "tampered_text",
    "altered_figures",
    "metadata_anomaly",
    "layout_inconsistency",
]

FEATURE_NAMES = [
    "font_consistency_score",      # 0-1, 1 = perfectly consistent fonts
    "num_fonts_used",              # int
    "metadata_time_diff_hours",    # |modified - created|, hours
    "known_producer_tool",         # 0/1, 1 = recognised PDF producer
    "ocr_confidence_avg",          # 0-1
    "ocr_char_error_rate",         # 0-1
    "page_count",                  # int
    "has_embedded_objects",        # 0/1
    "compression_artifact_score",  # 0-1, higher = more suspicious
    "layout_alignment_score",      # 0-1, 1 = perfectly aligned
    "whitespace_anomaly_score",    # 0-1
    "text_density_variance",       # 0-1
]


def _genuine_row(rng: random.Random) -> dict:
    return {
        "font_consistency_score": rng.uniform(0.90, 1.0),
        "num_fonts_used": rng.choice([1, 1, 1, 2]),
        "metadata_time_diff_hours": rng.uniform(0, 2),
        "known_producer_tool": 1,
        "ocr_confidence_avg": rng.uniform(0.92, 0.99),
        "ocr_char_error_rate": rng.uniform(0.0, 0.03),
        "page_count": rng.choice([1, 1, 2, 3]),
        "has_embedded_objects": rng.choice([0, 0, 0, 1]),
        "compression_artifact_score": rng.uniform(0.0, 0.15),
        "layout_alignment_score": rng.uniform(0.90, 1.0),
        "whitespace_anomaly_score": rng.uniform(0.0, 0.12),
        "text_density_variance": rng.uniform(0.0, 0.10),
    }


def _apply_tampered_text(row: dict, rng: random.Random) -> None:
    row["ocr_confidence_avg"] = rng.uniform(0.35, 0.70)
    row["ocr_char_error_rate"] = rng.uniform(0.15, 0.45)
    row["font_consistency_score"] = min(row["font_consistency_score"], rng.uniform(0.40, 0.75))
    row["num_fonts_used"] = max(row["num_fonts_used"], rng.choice([2, 3, 4]))


def _apply_altered_figures(row: dict, rng: random.Random) -> None:
    row["has_embedded_objects"] = 1
    row["compression_artifact_score"] = rng.uniform(0.55, 0.95)


def _apply_metadata_anomaly(row: dict, rng: random.Random) -> None:
    row["metadata_time_diff_hours"] = rng.uniform(48, 500)
    row["known_producer_tool"] = 0


def _apply_layout_inconsistency(row: dict, rng: random.Random) -> None:
    row["layout_alignment_score"] = rng.uniform(0.20, 0.60)
    row["whitespace_anomaly_score"] = rng.uniform(0.45, 0.90)
    row["text_density_variance"] = rng.uniform(0.40, 0.85)


_APPLIERS = {
    "tampered_text": _apply_tampered_text,
    "altered_figures": _apply_altered_figures,
    "metadata_anomaly": _apply_metadata_anomaly,
    "layout_inconsistency": _apply_layout_inconsistency,
}

# Class overlap. With "easy" genuine and forged feature distributions never overlap, so any
# model scores 1.00 and the result says nothing. The other levels add two kinds of overlap:
#   p_subtle_forgery : the forgery is only partly visible in a feature (value pulled back towards genuine)
#   p_noisy_genuine  : a genuine document that looks odd (re-scanned, re-saved by a portal, odd producer)
# All numbers are ASSUMPTIONS about how real data would behave, not measurements.
DIFFICULTY = {
    "easy":   dict(p_subtle_forgery=0.00, p_noisy_genuine=0.00, strength=(1.00, 1.00), n_noisy=(1, 1)),
    "medium": dict(p_subtle_forgery=0.50, p_noisy_genuine=0.30, strength=(0.05, 0.45), n_noisy=(1, 3)),
    "hard":   dict(p_subtle_forgery=0.80, p_noisy_genuine=0.50, strength=(0.00, 0.25), n_noisy=(2, 4)),
}


def _blend_towards_genuine(forged: dict, genuine: dict, rng: random.Random, strength_range=(0.25, 0.65)) -> None:
    """Pull every feature the forgery changed part of the way back to the genuine value."""
    strength = rng.uniform(*strength_range)  # 0 = invisible, 1 = fully tampered
    for f in FEATURE_NAMES:
        if forged[f] == genuine[f]:
            continue
        if f in ("known_producer_tool", "has_embedded_objects", "page_count", "num_fonts_used"):
            if rng.random() > strength:        # discrete feature: keep the genuine value with prob 1-strength
                forged[f] = genuine[f]
        else:
            forged[f] = genuine[f] + strength * (forged[f] - genuine[f])


def _make_noisy_genuine(row: dict, rng: random.Random, n_range=(1, 2)) -> None:
    """A genuine document that trips one or two forensic signals without being tampered with."""
    for kind in rng.sample(["rescan", "resaved", "producer", "layout"], k=rng.randint(*n_range)):
        if kind == "rescan":
            row["ocr_confidence_avg"] = rng.uniform(0.60, 0.88)
            row["ocr_char_error_rate"] = rng.uniform(0.03, 0.14)
        elif kind == "resaved":
            row["metadata_time_diff_hours"] = rng.uniform(20, 200)
        elif kind == "producer":
            row["known_producer_tool"] = 0
        else:
            row["layout_alignment_score"] = rng.uniform(0.55, 0.90)
            row["whitespace_anomaly_score"] = rng.uniform(0.12, 0.40)


def generate_labeled_dataset(
    output_csv: str = "data/rf_manifest.csv",
    n_genuine: int = 100,
    n_per_technique: int = 100,
    multi_anomaly_fraction: float = 0.25,
    seed: int = 42,
    difficulty: str = "medium",
) -> None:
    """
    Generate a labeled tabular dataset of genuine + forged document feature
    rows and write it to ``output_csv``.

    Parameters
    ----------
    output_csv : path to write the manifest CSV to.
    n_genuine : number of genuine (label=0) rows to generate.
    n_per_technique : number of forged rows to generate per anomaly type
        (before multi-anomaly combination). Mirrors the CNN branch's
        ``n_per_technique`` argument.
    multi_anomaly_fraction : fraction of forged rows that get a *second*
        random anomaly type stacked on top, so labels aren't always
        single-anomaly (matches real-world tampering, which often touches
        more than one signal at once).
    seed : RNG seed for reproducibility.
    difficulty : "easy" (classes never overlap; every model scores ~1.00),
        "medium" (default) or "hard". See DIFFICULTY.
    """
    if difficulty not in DIFFICULTY:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY)}")
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    rows = []

    doc_id = 0
    for _ in range(n_genuine):
        row = _genuine_row(rng)
        if rng.random() < cfg["p_noisy_genuine"]:
            _make_noisy_genuine(row, rng, cfg["n_noisy"])
        row["doc_id"] = f"doc_{doc_id:05d}"
        row["label"] = 0
        for a in ANOMALY_TYPES:
            row[a] = 0
        rows.append(row)
        doc_id += 1

    for technique in ANOMALY_TYPES:
        for _ in range(n_per_technique):
            row = _genuine_row(rng)
            base = dict(row)
            for a in ANOMALY_TYPES:
                row[a] = 0
            _APPLIERS[technique](row, rng)
            row[technique] = 1

            if rng.random() < multi_anomaly_fraction:
                other = rng.choice([a for a in ANOMALY_TYPES if a != technique])
                _APPLIERS[other](row, rng)
                row[other] = 1

            if rng.random() < cfg["p_subtle_forgery"]:
                _blend_towards_genuine(row, base, rng, cfg["strength"])

            row["doc_id"] = f"doc_{doc_id:05d}"
            row["label"] = 1
            rows.append(row)
            doc_id += 1

    rng.shuffle(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id"] + FEATURE_NAMES + ANOMALY_TYPES + ["label"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_forged = len(rows) - n_genuine
    print(f"Wrote {len(rows)} rows to {output_csv} "
          f"({n_genuine} genuine, {n_forged} forged).")
