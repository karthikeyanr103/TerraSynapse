"""
training/loss.py
Cosine-distance triplet margin loss. Inputs must already be L2-normalized.
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = 1.0 - (anchor * positive).sum(dim=1)
        d_neg = 1.0 - (anchor * negative).sum(dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean(), d_pos.mean().item(), d_neg.mean().item()
