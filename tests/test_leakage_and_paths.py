"""
Fast checks (no model training) that the data builders cannot leak between splits and that manifests
are portable. Run from the repository root:  pytest tests/

Each branch has its own `data` package, so every check runs in a fresh subprocess.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_in(component: str, code: str) -> None:
    r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, '.')\n{code}"],
                       cwd=ROOT / component, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def test_cnn_split_has_no_shared_source_document_or_absolute_path(tmp_path):
    gen = tmp_path / "genuine"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "make_genuine_documents.py"),
                        "--n", "20", "--out_dir", str(gen)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    run_in("document_cnn_branch", f"""
from data.synthetic_tamper import generate_split_datasets
generate_split_datasets(r'{gen}', r'{tmp_path / "out"}', r'{tmp_path / "man"}', data_root=r'{tmp_path}', seed=1)
""")
    splits = {s: read(tmp_path / "man" / f"{s}.csv") for s in ("train", "val", "test")}
    sources = {s: {r["source_file"] for r in rows} for s, rows in splits.items()}
    assert not (sources["train"] & sources["test"]) and not (sources["train"] & sources["val"]) \
        and not (sources["val"] & sources["test"]), "a source document appears in two splits"
    for rows in splits.values():
        for r in rows:
            assert not Path(r["filepath"]).is_absolute() and "\\" not in r["filepath"] and ":" not in r["filepath"]
            assert (tmp_path / r["filepath"]).exists()
        assert {r["label"] for r in rows} == {"genuine", "forged"}


def test_tampering_really_changes_the_page(tmp_path):
    import cv2
    import numpy as np
    gen = tmp_path / "genuine"
    subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "make_genuine_documents.py"), "--n", "6",
                    "--out_dir", str(gen)], check=True, capture_output=True)
    run_in("document_cnn_branch", f"""
import cv2, numpy as np, random
from data.synthetic_tamper import _TAMPER_FUNCS
random.seed(0); np.random.seed(0)
im = cv2.imread(r'{gen / "doc_0001.png"}'); dn = cv2.imread(r'{gen / "doc_0002.png"}')
for t, f in _TAMPER_FUNCS.items():
    out, _ = f(im, dn)
    assert (np.abs(out.astype(int) - im.astype(int)).max(2) > 12).sum() >= 150, t
""")


def test_siamese_split_is_identity_disjoint(tmp_path):
    run_in("signature_siamese_branch", f"""
from data.synthetic_signatures import generate_pairs_dataset
generate_pairs_dataset(r'{tmp_path / "pairs"}', r'{tmp_path / "man"}', data_root=r'{tmp_path}', n_identities=20,
                       genuine_per_id=3, forged_per_id=3, seed=1)
""")
    ids = {}
    for s in ("train", "val", "test"):
        rows = read(tmp_path / "man" / f"{s}.csv")
        ids[s] = {r["identity_a"] for r in rows} | {r["identity_b"] for r in rows}
        assert {r["label"] for r in rows} == {"0", "1"}
        for r in rows:
            assert not Path(r["img_a"]).is_absolute() and (tmp_path / r["img_a"]).exists()
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"])


def test_rf_difficulty_levels_overlap(tmp_path):
    run_in("document_rf_branch", f"""
import io, contextlib
from data.synthetic_metadata import generate_labeled_dataset
from data.dataset import stratified_split_manifest, load_manifest
from model.rf_document_branch import build_model
from sklearn.metrics import roc_auc_score
res = {{}}
for d in ("easy", "hard"):
    with contextlib.redirect_stdout(io.StringIO()):
        generate_labeled_dataset(r'{tmp_path / "m.csv"}', n_genuine=200, n_per_technique=100, difficulty=d, seed=3)
        stratified_split_manifest(r'{tmp_path / "m.csv"}', r'{tmp_path / "a.csv"}', r'{tmp_path / "b.csv"}', r'{tmp_path / "c.csv"}')
    te = load_manifest(r'{tmp_path / "c.csv"}')
    p, _ = build_model().fit(load_manifest(r'{tmp_path / "a.csv"}')).predict_proba(te)
    res[d] = roc_auc_score([r["label"] for r in te], p)
assert res["easy"] > 0.999 and res["hard"] < 0.99, res
""")


def test_matched_cases_use_each_item_once_and_find_all_columns(tmp_path):
    import random
    rnd = random.Random(0)

    def write(name, id_col, prefix, extra, n=60):
        rows = []
        for i in range(n):
            lab = i % 2
            r = {id_col: f"{prefix}{i}", "y_true": lab, "forgery_prob": rnd.random()}
            r.update({c: rnd.random() for c in extra})
            rows.append(r)
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("cnn.csv", "doc_id", "c", ["splicing_prob", "copy_move_prob", "inpainting_prob"])
    write("rf.csv", "doc_id", "r", ["tampered_text_prob", "altered_figures_prob", "metadata_anomaly_prob",
                                    "layout_inconsistency_prob"])
    write("sig.csv", "pair_id", "s", [])
    r = subprocess.run([sys.executable, "-I", str(ROOT / "build_matched_cases.py"), "--cnn_csv", str(tmp_path / "cnn.csv"),
                        "--rf_csv", str(tmp_path / "rf.csv"), "--siamese_csv", str(tmp_path / "sig.csv"),
                        "--out_dir", str(tmp_path / "f")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "were not found" not in r.stdout, "CNN type columns were not picked up"
    cmap = read(tmp_path / "f" / "case_map.csv")
    for col in ("cnn_id", "rf_id", "siamese_id"):
        ids = [c[col] for c in cmap]
        assert len(ids) == len(set(ids)), f"{col} reused across cases"
    fm = read(tmp_path / "f" / "fusion_manifest.csv")
    assert any(float(x["cnn_splicing_prob"]) != 0 for x in fm)
