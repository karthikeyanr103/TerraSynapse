# Kaggle Notebook 2 — Train three encoders

Attach the private subset, enable a GPU, and run this only after manually confirming Notebook 1's sanity grid.

```python
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt
```

```python
import sys, torch, pandas as pd
sys.path.insert(0, "/kaggle/working/repo")
from torch.utils.data import DataLoader
from config.config import METADATA, BATCH_SIZE, BACKBONE_S1, BACKBONE_S2, BACKBONE_OPTICAL
from data.dataset import BigEarthMMDataset
from models.encoders import build_encoders, inspect_model_structure
from training.trainer import run_training, triplet_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_id in (BACKBONE_S1, BACKBONE_OPTICAL, BACKBONE_S2):
    inspect_model_structure(model_id)  # manually confirm all three architectures

df = pd.read_csv(METADATA)
train_ds, val_ds = BigEarthMMDataset(df, "train"), BigEarthMMDataset(df, "validation")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=triplet_collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=triplet_collate_fn)
encoders = run_training(train_loader, val_loader, build_encoders(device), device)
```

The checkpoint at `/kaggle/working/outputs/checkpoints/best_model.pt` contains state dicts for `SAR`, `Optical`, and `Multispectral`. Commit the notebook so Notebook 3 can attach it.
