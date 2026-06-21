"""Training loop for the three-modality encoder dictionary."""
import torch
from tqdm import tqdm
from config.config import EPOCHS_PROJ_ONLY, EPOCHS_FINETUNE, LR_PROJ, LR_ENC, MARGIN, CKPT_DIR, FREEZE_BACKBONE
from training.loss import TripletLoss

def triplet_collate_fn(samples):
    """Keep varying-channel tensors as lists; default_collate cannot stack them."""
    return {key: [sample[key] for sample in samples] for key in samples[0]}

def get_optimizer(encoders, phase):
    proj = [p for enc in encoders.values() for p in enc.proj.parameters()]
    if phase == "proj_only": return torch.optim.Adam(proj, lr=LR_PROJ)
    backbone = [p for enc in encoders.values() for p in enc.backbone.parameters()]
    return torch.optim.Adam([{"params": backbone, "lr": LR_ENC}, {"params": proj, "lr": LR_PROJ}])

def encode_batch(batch, encoders, device):
    outputs = []
    for tensor_key, modality_key in (("anchor", "anchor_mod"), ("positive", "pos_mod"), ("negative", "neg_mod")):
        encoded = [encoders[m](x.unsqueeze(0).to(device)).squeeze(0) for x, m in zip(batch[tensor_key], batch[modality_key])]
        outputs.append(torch.stack(encoded))
    return tuple(outputs)

def run_one_phase(phase, epochs, loader, encoders, criterion, device, best):
    optimizer = get_optimizer(encoders, phase)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    for epoch in range(epochs):
        for enc in encoders.values(): enc.train()
        total = 0.0
        bar = tqdm(loader, desc=f"[{phase}] Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in bar:
            optimizer.zero_grad(); a, p, n = encode_batch(batch, encoders, device)
            loss, _, _ = criterion(a, p, n); loss.backward(); optimizer.step()
            total += loss.item(); bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step(); average = total / max(len(loader), 1)
        if average < best:
            best = average
            torch.save({"encoders": {m: e.state_dict() for m, e in encoders.items()}, "phase": phase, "epoch": epoch, "loss": average}, CKPT_DIR / "best_model.pt")
    return best

def run_training(train_loader, val_loader, encoders, device):
    del val_loader  # retained in API for a future validation-loss pass
    best = run_one_phase("proj_only", EPOCHS_PROJ_ONLY, train_loader, encoders, TripletLoss(MARGIN), device, float("inf"))
    if not FREEZE_BACKBONE:
        for enc in encoders.values(): enc.set_finetune_mode(True)
        best = run_one_phase("finetune", EPOCHS_FINETUNE, train_loader, encoders, TripletLoss(MARGIN), device, best)
    print(f"Training complete; best loss {best:.4f}. Checkpoint: {CKPT_DIR / 'best_model.pt'}")
    return encoders
