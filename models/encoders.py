"""Independent pretrained encoders for SAR, optical RGB, and multispectral data."""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config.config import EMBED_DIM, FREEZE_BACKBONE, BACKBONE_S1, BACKBONE_S2, BACKBONE_OPTICAL


def _configilm_from_huggingface(model_id):
    """Build ConfigILM directly and load the published pretrained weights."""
    from configilm.ConfigILM import ConfigILM, ILMConfiguration, ILMType

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")

    with open(config_path, "r", encoding="utf-8") as file:
        saved_config = json.load(file)

    supported_fields = (
        "timm_model_name",
        "hf_model_name",
        "image_size",
        "channels",
        "classes",
        "class_names",
        "visual_features_out",
        "fusion_in",
        "fusion_out",
        "fusion_hidden",
        "v_dropout_rate",
        "t_dropout_rate",
        "fusion_dropout_rate",
        "drop_rate",
        "use_pooler_output",
        "max_sequence_length",
        "load_pretrained_timm_if_available",
        "load_pretrained_hf_if_available",
    )
    config_kwargs = {
        field: saved_config[field]
        for field in supported_fields
        if field in saved_config
    }
    config_kwargs["network_type"] = ILMType(saved_config["network_type"])
    config = ILMConfiguration(**config_kwargs)
    model = ConfigILM(config)

    wrapper_state = load_file(weights_path, device="cpu")
    prefix = "model."
    if not wrapper_state or not all(key.startswith(prefix) for key in wrapper_state):
        raise RuntimeError(
            f"Unexpected checkpoint layout for {model_id}: expected all tensor "
            f"names to start with {prefix!r}."
        )
    configilm_state = {
        key[len(prefix):]: value for key, value in wrapper_state.items()
    }
    model.load_state_dict(configilm_state, strict=True)
    return model


def load_pretrained_classifier(model_id):
    if model_id.startswith("torchvision://"):
        from torchvision import models
        name = model_id.split("://", 1)[1]
        weights = models.get_model_weights(name).DEFAULT
        return models.get_model(name, weights=weights)
    return _configilm_from_huggingface(model_id)

def inspect_model_structure(model_id):
    model = load_pretrained_classifier(model_id)
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
        self.backbone, feat_dim = _extract_backbone_and_feat_dim(load_pretrained_classifier(model_id))
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
