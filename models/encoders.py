"""Independent pretrained encoders for SAR, optical RGB, and multispectral data."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import EMBED_DIM, FREEZE_BACKBONE, BACKBONE_S1, BACKBONE_S2, BACKBONE_OPTICAL

def _load_pretrained_classifier(model_id):
    if model_id.startswith("torchvision://"):
        from torchvision import models
        name = model_id.split("://", 1)[1]
        weights = models.get_model_weights(name).DEFAULT
        return models.get_model(name, weights=weights)
    from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    return BigEarthNetv2_0_ImageClassifier.from_pretrained(model_id)

def inspect_model_structure(model_id):
    model = _load_pretrained_classifier(model_id)
    print(model)
    return model

def _extract_backbone_and_feat_dim(full_model):
    base = full_model
    for attr in ("model", "vision_encoder", "encoder", "backbone", "net"):
        candidate = getattr(full_model, attr, None)
        if isinstance(candidate, nn.Module): base = candidate; break
    if hasattr(base, "forward_features") and hasattr(base, "forward_head"):
        class TimmFeatures(nn.Module):
            def __init__(self, model): super().__init__(); self.model = model
            def forward(self, x): return self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        dim = getattr(base, "num_features", None)
        if dim: return TimmFeatures(base), int(dim)
    last = next((m for m in reversed(list(full_model.modules())) if isinstance(m, nn.Linear)), None)
    if last is None: raise RuntimeError("No final Linear layer found; inspect this backbone manually")
    captured = {}
    last.register_forward_hook(lambda _m, inp, _out: captured.update(features=inp[0]))
    class HookFeatures(nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
        def forward(self, x): self.model(x); return captured["features"]
    return HookFeatures(full_model), last.in_features

class PretrainedModalityEncoder(nn.Module):
    def __init__(self, model_id, in_channels, embed_dim=EMBED_DIM, freeze_backbone=FREEZE_BACKBONE):
        super().__init__()
        self.model_id, self.in_channels = model_id, in_channels
        self.backbone, feat_dim = _extract_backbone_and_feat_dim(_load_pretrained_classifier(model_id))
        self.freeze_backbone = freeze_backbone
        for p in self.backbone.parameters(): p.requires_grad = not freeze_backbone
        # LayerNorm remains valid when a modality appears only once in a mixed batch.
        self.proj = nn.Sequential(nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Linear(512, embed_dim))
    def forward(self, x):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad(): features = self.backbone(x)
        else: features = self.backbone(x)
        return F.normalize(self.proj(features), dim=1)
    def set_finetune_mode(self, finetune):
        self.freeze_backbone = not finetune
        for p in self.backbone.parameters(): p.requires_grad = finetune
        self.backbone.train(finetune)

def build_encoders(device):
    return {
        "SAR": PretrainedModalityEncoder(BACKBONE_S1, 2).to(device),
        "Optical": PretrainedModalityEncoder(BACKBONE_OPTICAL, 3).to(device),
        "Multispectral": PretrainedModalityEncoder(BACKBONE_S2, 10).to(device),
    }
