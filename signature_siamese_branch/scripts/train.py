"""
Train the Siamese signature verifier.

python scripts/train.py --train_manifest <train.csv> --val_manifest <val.csv> --data_root . --output_dir checkpoints/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import SignaturePairDataset
from model.siamese_cnn import build_model
from utils.metrics import compute_forgery_metrics, format_metrics_report


def run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total, ys, ps = 0.0, [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for b in loader:
            a, c, y = b["img_a"].to(device), b["img_b"].to(device), b["label"].to(device)
            logit = model(a, c)["forgery_logit"]
            loss = crit(logit, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * a.size(0)
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    m = compute_forgery_metrics(np.concatenate(ys).ravel(), np.concatenate(ps).ravel())
    m["loss"] = total / len(loader.dataset)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="checkpoints")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    tr = SignaturePairDataset(a.train_manifest, a.data_root, augment=True, seed=a.seed)
    va = SignaturePairDataset(a.val_manifest, a.data_root, augment=False)
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=a.batch_size)

    model = build_model().to(device)
    cc = tr.class_counts()
    pos_weight = torch.tensor([cc["genuine"] / max(cc["forged"], 1)], device=device)
    print(f"device={device}  class balance={cc}  pos_weight={pos_weight.item():.2f}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)

    # Checkpoint on validation ROC-AUC (threshold-free, so it does not reward a lucky F1 at 0.5)
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        trm = run_epoch(model, tl, device, opt, pos_weight)
        vam = run_epoch(model, vl, device, None, pos_weight)
        print(f"Epoch {ep:02d}/{a.epochs}  train_loss={trm['loss']:.4f}  val_loss={vam['loss']:.4f}  "
              f"val_auc={vam['roc_auc']:.4f}  val_f1={vam['f1']:.4f}", flush=True)
        if vam["roc_auc"] > best:
            best, bad = vam["roc_auc"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_auc": best, "val_metrics": vam},
                       Path(a.output_dir) / "best_siamese.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping (no val AUC gain in {a.patience} epochs).")
                break
    print(f"\nBest val ROC-AUC: {best:.4f}")
    print(format_metrics_report(torch.load(Path(a.output_dir) / "best_siamese.pt", weights_only=False)["val_metrics"], "val (best epoch)"))


if __name__ == "__main__":
    main()
