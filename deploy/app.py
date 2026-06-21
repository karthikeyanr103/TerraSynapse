"""Streamlit interface for seven retrieval directions."""
import streamlit as st
try:
    from deploy.inference import available_backbones, load_artifacts, encode_query, sample_embedding, search, ARTIFACT_ROOT
except ModuleNotFoundError:
    from inference import available_backbones, load_artifacts, encode_query, sample_embedding, search, ARTIFACT_ROOT

TASKS = ["Optical -> Optical", "SAR -> SAR", "Multispectral -> Multispectral", "Optical -> SAR", "SAR -> Optical", "Optical -> Multispectral", "Multispectral -> Optical"]
st.set_page_config(page_title="Satellite Image Retrieval", layout="wide")
st.title("Cross-Modal Satellite Image Retrieval")
task = st.selectbox("Choose retrieval task:", TASKS); source, target = task.split(" -> ")
backbones = available_backbones()
if not backbones: st.error("No exported artifacts found. Run deploy/export_artifacts.py first."); st.stop()
backbone = st.selectbox("Choose backbone:", backbones)

@st.cache_resource
def cached_artifacts(key): return load_artifacts(key)
@st.cache_data
def thumbnails(modality): return sorted((ARTIFACT_ROOT / "sample_thumbnails" / modality).glob("*.png"))

artifacts = cached_artifacts(backbone)
mode = st.radio("Choose query image:", ["Pick a sample image", "Upload my own"], horizontal=True)
selected, upload = None, None
if mode == "Pick a sample image":
    samples = thumbnails(source)
    if not samples: st.warning(f"No sample thumbnails are available for {source}.")
    else:
        selected = st.selectbox("Sample patch:", samples, format_func=lambda p: p.stem)
        cols = st.columns(min(5, len(samples)))
        for i, path in enumerate(samples[:10]): cols[i % len(cols)].image(str(path), caption=path.stem)
else:
    upload = st.file_uploader("Upload a query", type=["tif", "tiff", "jpg", "jpeg", "png"])

if st.button("Run Inference", type="primary"):
    try:
        if mode == "Pick a sample image":
            if selected is None: raise ValueError("Choose a sample image first.")
            query = sample_embedding(selected.stem, source, artifacts); query_preview = selected
        else:
            if upload is None: raise ValueError("Upload an image first.")
            query = encode_query(upload, source, backbone, artifacts); query_preview = upload
        with st.spinner("Encoding and searching…"): results, elapsed = search(query, target, backbone, 10, artifacts)
        left, right = st.columns([1, 4]); left.image(query_preview, caption=f"Query ({source})")
        with right:
            for tab, count in zip(st.tabs(["Top-5", "Top-10"]), [5, 10]):
                with tab:
                    cols = st.columns(5)
                    for i, (pid, score, label, thumb) in enumerate(results[:count]):
                        if thumb: cols[i % 5].image(str(thumb), caption=f"{label}\n{score:.3f}")
                        else: cols[i % 5].write(f"{pid}\n{label}\n{score:.3f}")
        st.metric("Avg retrieval time", f"{elapsed:.2f} ms")
    except (ValueError, OSError, RuntimeError) as exc: st.error(str(exc))

meta = artifacts["metadata"]
st.sidebar.header("About")
st.sidebar.write("Same-modal retrieval searches within one sensor representation; cross-modal retrieval searches across representations.")
st.sidebar.write(f"Dataset: {meta.get('dataset', 'Unknown')}  ")
st.sidebar.write(f"Backbone: {backbone}")
