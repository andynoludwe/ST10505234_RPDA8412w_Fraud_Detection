"""
Siamese CNN for signature verification.

Two signature images pass through the SAME convolutional encoder (shared weights). The pair is
described by |e_a - e_b| and e_a * e_b and a small head turns that into one logit.

    forgery_prob = sigmoid(logit)    higher = more likely the claimed signature is NOT the reference writer's

That is the same "higher means more suspicious" convention as the CNN and Random Forest branches, so
the three outputs can be fused directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SiameseSignatureNet(nn.Module):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 128),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, embed_dim))
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(64, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        ea, eb = self.embed(a), self.embed(b)
        feats = torch.cat([(ea - eb).abs(), ea * eb], dim=1)  # symmetric in (a, b)
        return {"forgery_logit": self.head(feats)}

    @torch.no_grad()
    def predict_proba(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(a, b)["forgery_logit"])


def build_model(**kw) -> SiameseSignatureNet:
    return SiameseSignatureNet(**kw)
