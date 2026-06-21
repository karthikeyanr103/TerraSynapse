# Streamlit deployment

Artifacts are kept out of normal Git history and can be committed with Git LFS after export.

```bash
pip install -r deploy/requirements_deploy.txt
streamlit run deploy/app.py
```

Run `deploy/export_artifacts.py` in the Kaggle evaluation environment first, then extract the generated archive into `deploy/artifacts/`. For Streamlit Community Cloud, track the artifact extensions with Git LFS (`*.pt`, `*.npy`, `*.index`) before pushing.
