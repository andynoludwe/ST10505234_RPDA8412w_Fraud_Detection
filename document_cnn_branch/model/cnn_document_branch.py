"""
Document Visual Tampering CNN — "Document Stream (Visual)" component
from Table 2 of the research proposal.

Detects visual manipulation techniques in scanned/photographed financial
documents: splicing, copy-move, and inpainting. Outputs are designed to
feed directly into the late-fusion / stacking meta-classifier described
in Section 6.2 and 7.2 of the proposal, alongside:
  - the tabular model (metadata/OCR/font anomalies), and
  - the Siamese signature verification stream.

Architecture choice:
  Transfer learning with a ResNet50 or EfficientNet-B0 backbone
  (both evaluated in the proposal — Section 7.2), with the final
  classification layer replaced by a multi-label + binary head:

    - 3 sigmoid outputs: probability of splicing, copy-move, inpainting
      (multi-label, since a document can show more than one manipulation)
    - 1 sigmoid output: overall forged/genuine probability
      (this is the score consumed by the fusion layer)

Usage:
    from model.cnn_document_branch import DocumentTamperCNN

    model = DocumentTamperCNN(backbone="resnet50", pretrained=True)
    outputs = model(batch_of_images)
    outputs["tamper_type_logits"]  # shape (B, 3) -> splicing, copy_move, inpainting
    outputs["forgery_logit"]       # shape (B, 1) -> overall forged probability (pre-sigmoid)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

TAMPER_TYPES = ["splicing", "copy_move", "inpainting"]


class DocumentTamperCNN(nn.Module):
    """
    Multi-label + binary CNN for visual document forgery detection.

    Args:
        backbone: "resnet50" or "efficientnet_b0" — matches the two
            architectures compared in Section 7.2 of the proposal.
        pretrained: use ImageNet-pretrained weights (transfer learning,
            as justified in Section 10.2 for computational feasibility).
        freeze_backbone: if True, freezes all backbone weights and only
            trains the new classification head. Useful for a fast first
            pass on a small dataset; unfreeze later for fine-tuning.
        dropout: dropout applied before the classification heads.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = tv_models.resnet50(weights=weights)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net

        elif backbone == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = tv_models.efficientnet_b0(weights=weights)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.backbone = net

        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet50' or 'efficientnet_b0'.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Shared trunk before splitting into the two heads
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Multi-label head: which tampering technique(s) are present
        self.tamper_type_head = nn.Linear(256, len(TAMPER_TYPES))

        # Binary head: overall forged vs genuine probability
        # (this is the score the fusion layer / meta-classifier consumes)
        self.forgery_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        shared = self.shared(feats)
        return {
            "tamper_type_logits": self.tamper_type_head(shared),  # (B, 3), raw logits
            "forgery_logit": self.forgery_head(shared),           # (B, 1), raw logit
            "embedding": shared,                                   # (B, 256), useful for the meta-classifier / SHAP
        }

    def unfreeze_backbone(self, last_n_blocks: int | None = None):
        """
        Unfreeze the backbone for fine-tuning after an initial head-only
        training pass. If last_n_blocks is set, only unfreezes the last
        N top-level children of the backbone (cheaper fine-tuning).
        """
        children = list(self.backbone.children())
        to_unfreeze = children if last_n_blocks is None else children[-last_n_blocks:]
        for module in to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience method: returns sigmoid probabilities instead of raw logits."""
        self.eval()
        out = self.forward(x)
        return {
            "tamper_type_probs": torch.sigmoid(out["tamper_type_logits"]),
            "forgery_prob": torch.sigmoid(out["forgery_logit"]),
        }


def build_model(backbone: str = "resnet50", pretrained: bool = True) -> DocumentTamperCNN:
    """Factory function matching the pattern used elsewhere in the pipeline."""
    return DocumentTamperCNN(backbone=backbone, pretrained=pretrained)


if __name__ == "__main__":
    # Quick smoke test
    model = build_model("resnet50")
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("tamper_type_logits:", out["tamper_type_logits"].shape)
    print("forgery_logit:", out["forgery_logit"].shape)
    print("embedding:", out["embedding"].shape)

    model_eff = build_model("efficientnet_b0")
    out_eff = model_eff(dummy)
    print("\nEfficientNet-B0 variant OK:", out_eff["forgery_logit"].shape)
