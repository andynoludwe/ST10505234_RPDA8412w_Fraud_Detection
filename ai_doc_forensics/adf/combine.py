"""Join the branches' prediction files by case_id (this replaces the simulated scores of the earlier fusion step)."""
import os
import pandas as pd

BRANCHES = ["image", "metadata", "arithmetic", "signature"]


def write_branch_predictions(pred_dir, branch, case_ids, split, p):
    os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame({"case_id": case_ids, "split": split, "p_ai": p}).to_csv(os.path.join(pred_dir, f"{branch}_predictions.csv"), index=False)


def join_predictions(pred_dir, branches=BRANCHES):
    """Returns a DataFrame indexed by case_id with columns p_<branch> and a `split` column. Cases missing from any branch are dropped (reported)."""
    out = None
    for b in branches:
        d = pd.read_csv(os.path.join(pred_dir, f"{b}_predictions.csv")).rename(columns={"p_ai": f"p_{b}"}).set_index("case_id")
        if out is None:
            out = d
        else:
            if (out["split"].reindex(d.index) != d["split"]).any():
                raise ValueError(f"split disagrees between branches at {b}")
            out = out.join(d[[f"p_{b}"]], how="inner")
    return out
