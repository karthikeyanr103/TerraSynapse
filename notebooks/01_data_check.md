# notebooks/01_data_check.md
## Kaggle Notebook 1 — Data Check & Sanity Visualization

Create a new Kaggle notebook named `01-data-check`. Attach your private
`bigearth-mm-subset-47k` dataset (Add Data -> Your Datasets). Turn ON GPU
is NOT required for this notebook — CPU is fine and saves your GPU quota
for notebooks 2 and 3.

Run these cells in order.

---

### Cell 1 — Clone your code from GitHub
```python
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt
```

### Cell 2 — Imports
```python
import sys
sys.path.insert(0, "/kaggle/working/repo")

import pandas as pd
from config.config import METADATA, KEEP_CLASSES
from data.visualize import sanity_check_grid, class_distribution_plot
```

### Cell 3 — Load and inspect the manifest
```python
df = pd.read_csv(METADATA)
print(f"Total patches: {len(df)}")
print(df['split'].value_counts())
df.head()
```

### Cell 4 — Class balance check
```python
class_distribution_plot(df)
```
**Check:** every class should have a reasonably similar bar height across
train/val/test. If one class is near-empty in any split, flag it now —
your retrieval metrics for that class will be unreliable later.

### Cell 5 — Visual sanity check (THE IMPORTANT ONE)
```python
sanity_check_grid(df, n_samples=6)
```
**Check, for every row:**
- SAR VV/VH look like grainy radar speckle, NOT black or solid gray
- Optical RGB preview actually looks like a satellite photo matching the
  printed `primary_label` (e.g. "Inland waters" should look blue/dark)
- MS B08 (NIR) shows vegetation bright, water/urban dark
- The SAR and Optical/MS images in the same row are *plausibly the same
  place* — if they look like completely different locations, your
  `s1_name` / `patch_id` matching in `01_select_subset.py` is broken and
  you must fix it before training anything.

### Cell 6 — Compute real MS normalization stats (run once)
```python
from data.preprocessing import compute_ms_stats
train_ids = df[df['split'] == 'train']['patch_id'].tolist()
mean, std = compute_ms_stats(train_ids, sample_size=2000)
# Copy the printed MS_MEAN / MS_STD values into config/config.py,
# replacing the placeholder values, then re-run from Cell 1 in future notebooks.
```

If everything in this notebook looks right, move on to Notebook 2.
