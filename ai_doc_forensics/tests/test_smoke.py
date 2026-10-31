"""Quick checks: python -m pytest tests  (uses 150 cases, about 30 seconds)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adf.config import GenConfig
from adf.pipeline import run


def test_pipeline_runs_and_is_leak_free():
    with tempfile.TemporaryDirectory() as d:
        r = run(GenConfig(n_cases=150), d, n_jobs=1, n_boot=20, verbose=False)
        assert r["split"]["n_train"] > 0 and r["split"]["n_test"] > 0
        t = r["test_metrics"]
        assert 0.5 < t["fusion_lr"]["roc_auc"] <= 1.0
        assert t["fusion_lr"]["roc_auc"] < 1.0 or t["fusion_lr"]["f1"] < 1.0, "perfect score would indicate leakage or an unrealistically easy generator"


def test_generation_is_reproducible():
    from adf.generator import generate_dataset
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        da = generate_dataset(GenConfig(n_cases=12), a, verbose=False); db = generate_dataset(GenConfig(n_cases=12), b, verbose=False)
        assert da.equals(db)
        assert open(os.path.join(a, "images", "C000003.jpg"), "rb").read() == open(os.path.join(b, "images", "C000003.jpg"), "rb").read()
