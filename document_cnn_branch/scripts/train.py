"""
Training script for DocumentTamperCNN.

Implements the training procedure from Section 7.5 of the proposal:
  - transfer learning (frozen backbone for warm-up epochs, then fine-tune)
  - class-imbalance-aware loss (pos_weight for BCE, since genuine
    documents will outnumber forged ones)
  - validation-set monitoring for early stopping / checkpoint selection, on validation ROC-AUC
    (NOT F1: on a forged-heavy validation set "call everything forged" already scores F1 ~0.86, so
    F1-based selection keeps that degenerate epoch)
  - metrics computed at the end of each epoch: precision, recall, F1,
    ROC-AUC, PR-AUC (on the forgery head)

Usage:
    python scripts/train.py \
        --train_manifest data/train.csv \
        --val_manifest data/val.csv \
        --backbone resnet50 \
        --epochs 20 \
        --warmup_epochs 3 \
        --batch_size 16 \
        --output_dir checkpoints/
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

from data.dataset import DocumentTamperDataset
from model.cnn_document_branch import build_model
from utils.metrics import compute_forgery_metrics, format_metrics_report


def compute_pos_weight(dataset: DocumentTamperDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts()
    genuine, forged = counts["genuine"], counts["forged"]
    if forged == 0:
        return torch.tensor([1.0], device=device)
    # Standard imbalance weighting: weight positive (forged) class by ratio of negatives to positives
    weight = genuine / max(forged, 1)
    return torch.tensor([weight], device=device)


def run_epoch(model, loader, device, optimizer=None, forgery_pos_weight=None):
    """One pass over the data. If optimizer is provided, trains; otherwise evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    forgery_criterion = nn.BCEWithLogitsLoss(pos_weight=forgery_pos_weight)
    tamper_criterion = nn.BCEWithLogitsLoss()  # per-label imbalance handled less critically here

    total_loss = 0.0
    all_true, all_prob = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            forgery_label = batch["forgery_label"].to(device)
            tamper_label = batch["tamper_type_label"].to(device)

            out = model(images)
            loss_forgery = forgery_criterion(out["forgery_logit"], forgery_label)
            loss_tamper = tamper_criterion(out["tamper_type_logits"], tamper_label)
            loss = loss_forgery + 0.5 * loss_tamper  # forgery detection is the primary head

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_true.append(forgery_label.detach().cpu().numpy())
            all_prob.append(torch.sigmoid(out["forgery_logit"]).detach().cpu().numpy())

    y_true = np.concatenate(all_true).ravel()
    y_prob = np.concatenate(all_prob).ravel()
    metrics = compute_forgery_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                         help="Disable ImageNet-pretrained weights (e.g. if weight download is blocked)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Epochs with backbone frozen before fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_root", default=".", help="manifest filepaths are relative to this folder")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val ROC-AUC improvement)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_ds = DocumentTamperDataset(args.train_manifest, image_size=args.image_size, augment=True, data_root=args.data_root)
    val_ds = DocumentTamperDataset(args.val_manifest, image_size=args.image_size, augment=False, data_root=args.data_root)

    num_workers = min(2, (torch.get_num_threads() or 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(args.backbone, pretrained=args.pretrained).to(device)
    model.backbone.requires_grad_(False)  # start frozen for warm-up

    pos_weight = compute_pos_weight(train_ds, device)
    print(f"Class balance: {train_ds.class_counts()}  ->  forgery pos_weight={pos_weight.item():.2f}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            print(">>> Unfreezing backbone for fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 10)

        train_metrics = run_epoch(model, train_loader, device, optimizer, pos_weight)
        val_metrics = run_epoch(model, val_loader, device, None, pos_weight)

        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        print(f"  train_loss={train_metrics['loss']:.4f}  train_f1={train_metrics['f1']:.4f}")
        print(f"  val_loss={val_metrics['loss']:.4f}    " + format_metrics_report(val_metrics, "val").replace("\n", "\n  "), flush=True)

        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] == val_metrics["roc_auc"] else -1.0  # NaN -> -1
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            checkpoint_path = Path(args.output_dir) / f"best_{args.backbone}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "epoch": epoch,
                "val_f1": val_metrics["f1"],
                "val_auc": best_auc,
                "val_metrics": val_metrics,
            }, checkpoint_path)
            print(f"  -> saved new best checkpoint ({checkpoint_path}), val_auc={best_auc:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no val ROC-AUC improvement in {args.patience} epochs.")
                break

    print(f"\nTraining complete. Best val ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
