# Kaggle Notebook 3 — Evaluate all seven directions

Clone/install as in Notebook 2, attach its output, then run:

```python
import sys, torch, numpy as np, pandas as pd
from tqdm import tqdm
sys.path.insert(0, "/kaggle/working/repo")
from config.config import METADATA, N_QUERY_PER_CLS, RESULTS_DIR
from data.preprocessing import load_sar, load_optical_rgb, load_ms
from models.encoders import build_encoders
from evaluation.metrics import evaluate_retrieval, print_results_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoders = build_encoders(device)
ckpt = torch.load("/kaggle/input/02-train-retrieval-model/best_model.pt", map_location=device)
for modality, encoder in encoders.items():
    encoder.load_state_dict(ckpt["encoders"][modality]); encoder.eval()

df = pd.read_csv(METADATA).query("split == 'test'").reset_index(drop=True)
loaders = {"SAR": lambda r: load_sar(r.s1_name), "Optical": lambda r: load_optical_rgb(r.patch_id), "Multispectral": lambda r: load_ms(r.patch_id)}
def encode_rows(rows, modality, desc):
    vectors, labels = [], []
    with torch.no_grad():
        for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=desc, unit="patch"):
            vectors.append(encoders[modality](loaders[modality](row).unsqueeze(0).to(device)).cpu().numpy()[0]); labels.append(row.primary_label)
    return np.asarray(vectors, dtype="float32"), labels

galleries = {m: encode_rows(df, m, f"Gallery {m}") for m in encoders}
query_df = pd.concat([g.sample(min(N_QUERY_PER_CLS, len(g)), random_state=42) for _, g in df.groupby("primary_label")])
queries = {m: encode_rows(query_df, m, f"Queries {m}") for m in encoders}
directions = [("Optical","Optical"), ("SAR","SAR"), ("Multispectral","Multispectral"), ("Optical","SAR"), ("SAR","Optical"), ("Optical","Multispectral"), ("Multispectral","Optical")]
results = [evaluate_retrieval(*queries[src], *galleries[dst], f"{src}_to_{dst}") for src, dst in directions]
print_results_table(results)
pd.DataFrame(results).to_csv(RESULTS_DIR / "retrieval_results.csv", index=False)
```

```python
!python deploy/export_artifacts.py --checkpoint /kaggle/input/02-train-retrieval-model/best_model.pt --output /kaggle/working/artifacts
```
