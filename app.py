import streamlit as st
import numpy as np
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile

# Make src importable
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# project imports (must exist in src/)
try:
    from dataset.loader import load_volume_any
    from dataset.preprocessing import basic_brain_mask, apply_brain_mask
    from utils.mls import compute_mls_advanced
except Exception as e:
    # defer raising until run
    load_volume_any = basic_brain_mask = apply_brain_mask = compute_mls_advanced = None

import onnxruntime as ort

st.set_page_config(page_title="MLS ONNX Viewer", layout="wide")
st.title("🧠 MLS Prediction Viewer (ONNX) — Heatmap + Slice Viewer")

# UI — left panel for controls
with st.sidebar:
    st.header("Inputs & Options")
    onnx_path = st.text_input("ONNX model path", value="checkpoints/mls3d.onnx")
    default_input = "data/mls/train/vol_001.npy"
    input_path_text = st.text_input("Input volume path (optional)", value=default_input)
    uploaded = st.file_uploader("Or upload volume (.npy or .nii/.nii.gz)", type=["npy","nii","nii.gz"])
    threshold = st.slider("Mask threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    run_btn = st.button("Run Inference")
    st.markdown("---")
    st.write("Display options")
    show_heatmap = st.checkbox("Show probability heatmap overlay", value=True)
    show_mask_overlay = st.checkbox("Show binary mask overlay", value=True)
    st.write("If both checked, heatmap is shown below mask for clarity.")
    st.markdown("---")
    st.write("Save outputs")
    save_outputs = st.checkbox("Save probs/mask/overlay to outputs/", value=True)
    outdir = st.text_input("Outputs folder", value="outputs")
    st.markdown("---")
    st.write("Slice controls (after inference)")
    auto_slice = st.checkbox("Auto-select MLS slice", value=True)
    st.markdown("")

# central area for images & stats
col_img, col_stats = st.columns([2,1])

# helper loader functions
def save_overlay_png(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def load_input_volume(uploaded_file, input_path_text):
    """
    Returns (vol, affine, spacing, used_path_description)
    vol: numpy array (H,W,D)
    """
    # priority: uploaded file > typed path
    if uploaded_file is not None:
        # save to temp and load using loader if possible
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        used = tmp.name
        # if .npy
        if used.endswith(".npy"):
            vol = np.load(used)
            # loader expects H,W,D; no affine/spacing
            return vol, None, (1.0,1.0,1.0), f"uploaded:{uploaded_file.name}"
        else:
            # try nibabel
            nii = nib.load(used)
            vol = nii.get_fdata()
            # ensure orientation: convert to float32
            return vol.astype(np.float32), nii.affine, getattr(nii.header, 'get_zooms', lambda: (1.0,1.0,1.0))(), f"uploaded:{uploaded_file.name}"
    # else typed path
    if input_path_text and os.path.exists(input_path_text):
        # prefer loader that does windowing/resampling
        if load_volume_any is not None:
            vol, affine, spacing = load_volume_any(input_path_text, target_spacing=(1.0,1.0,5.0), do_window=True)
            return vol.astype(np.float32), affine, spacing, input_path_text
        # try fallback for .npy
        ext = os.path.splitext(input_path_text)[1].lower()
        if ext == ".npy":
            vol = np.load(input_path_text)
            return vol.astype(np.float32), None, (1.0,1.0,1.0), input_path_text
        else:
            nii = nib.load(input_path_text)
            vol = nii.get_fdata()
            return vol.astype(np.float32), nii.affine, getattr(nii.header, 'get_zooms', lambda: (1.0,1.0,1.0))(), input_path_text
    # not available
    return None, None, None, None

# inference core
def run_onnx_inference(onnx_path, vol, affine, spacing, prob_threshold=0.5):
    """
    vol: (H,W,D) float32
    returns: probs (H,W,D), mask (H,W,D), mls_mm, mls_slice
    """
    if vol is None:
        raise RuntimeError("No input volume provided.")

    # ensure brain mask funcs available
    if basic_brain_mask is None:
        raise RuntimeError("Project functions not importable. Ensure src is in sys.path and dataset.loader exists.")

    brain_mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=300)
    if brain_mask is None:
        raise RuntimeError("basic_brain_mask returned None; check input or intensity threshold.")
    vol_masked = apply_brain_mask(vol, brain_mask)

    if vol_masked.ndim != 3:
        raise RuntimeError("Volume must be 3D (H,W,D). Got shape: %s" % (vol_masked.shape,))

    inp = vol_masked[np.newaxis, np.newaxis, :, :, :]   # (1,1,H,W,D)
    inp = np.transpose(inp, (0,1,4,2,3)).astype(np.float32)  # (1,1,D,H,W)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: inp})[0]  # (1,1,D,H,W)
    if out.ndim != 5:
        raise RuntimeError("Unexpected model output shape: %s" % (out.shape,))
    pred = out[0,0]  # (D,H,W)
    pred_hwd = np.transpose(pred, (1,2,0))  # -> (H,W,D)

    # logits -> probs if needed
    if pred_hwd.min() < 0.0 or pred_hwd.max() > 1.0:
        z = np.clip(pred_hwd, -50.0, 50.0)
        probs = 1.0 / (1.0 + np.exp(-z))
    else:
        probs = pred_hwd

    mask = (probs > prob_threshold).astype(np.uint8)

    # compute MLS using your utility
    mls_mm, zslice = compute_mls_advanced(mask, spacing=spacing)
    return probs, mask, mls_mm, zslice

# run when button pressed
if run_btn:
    st.info("Running ONNX inference... this may take a few seconds.")
    # load input
    vol, affine, spacing, used_desc = load_input_volume(uploaded, input_path_text)
    if vol is None:
        st.error("No input volume found. Provide path or upload a .npy/.nii file.")
    else:
        try:
            probs, mask, mls_mm, zslice = run_onnx_inference(onnx_path, vol, affine, spacing, prob_threshold=threshold)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.exception(e)
            probs = mask = None
            mls_mm = zslice = None

        if probs is not None:
            H,W,D = probs.shape
            st.success("Inference completed.")

            # stats
            pmin, pmax, pmean, pstd = float(probs.min()), float(probs.max()), float(probs.mean()), float(probs.std())

            # auto-select slice
            if auto_slice and (zslice is not None):
                z_choose = int(zslice)
            else:
                z_choose = D//2

            # interactive slice selector
            z_use = st.slider("Select z-slice", 0, D-1, z_choose)

            # display images
            with col_img:
                st.subheader("Overlay / Heatmap Viewer")
                fig, ax = plt.subplots(figsize=(6,6))
                img = vol[:,:,z_use]
                ax.imshow(img, cmap="gray", interpolation='nearest')
                # heatmap
                if show_heatmap:
                    hm = probs[:,:,z_use]
                    # overlay heatmap with alpha
                    ax.imshow(hm, cmap='inferno', alpha=0.45, vmin=0.0, vmax=1.0)
                # mask overlay
                if show_mask_overlay:
                    mask_slice = mask[:,:,z_use] > 0
                    ax.imshow(np.ma.masked_where(mask_slice==0, mask_slice), cmap='Reds', alpha=0.6)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

                # small thumbnail and download links
                if save_outputs:
                    os.makedirs(outdir, exist_ok=True)
                    # save probs npz and nii
                    np.savez_compressed(os.path.join(outdir, "pred_probs_onnx.npz"), probs=probs)
                    try:
                        nib.save(nib.Nifti1Image(probs.astype(np.float32), affine if affine is not None else np.eye(4)),
                                 os.path.join(outdir, "pred_probs_onnx.nii.gz"))
                    except Exception:
                        pass
                    np.savez_compressed(os.path.join(outdir, "pred_mask_onnx.npz"), pred_mask=mask)
                    try:
                        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine if affine is not None else np.eye(4)),
                                 os.path.join(outdir, "pred_mask_onnx.nii.gz"))
                    except Exception:
                        pass
                    # save overlay png
                    overlay_path = os.path.join(outdir, f"overlay_z{z_use}.png")
                    fig2, ax2 = plt.subplots(figsize=(6,6))
                    ax2.imshow(vol[:,:,z_use], cmap='gray', interpolation='nearest')
                    if show_heatmap:
                        ax2.imshow(probs[:,:,z_use], cmap='inferno', alpha=0.45, vmin=0.0, vmax=1.0)
                    if show_mask_overlay:
                        ax2.imshow(np.ma.masked_where(mask[:,:,z_use]==0, mask[:,:,z_use]), cmap='Reds', alpha=0.6)
                    ax2.axis('off')
                    fig2.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig2)

                    st.markdown(f"Saved outputs to `{outdir}`:")
                    st.write(f"- {os.path.join(outdir,'pred_probs_onnx.npz')}")
                    st.write(f"- {os.path.join(outdir,'pred_probs_onnx.nii.gz')}")
                    st.write(f"- {os.path.join(outdir,'pred_mask_onnx.npz')}")
                    st.write(f"- {os.path.join(outdir,'pred_mask_onnx.nii.gz')}")
                    st.write(f"- {overlay_path}")

            with col_stats:
                st.subheader("Statistics")
                st.metric("MLS (mm)", f"{mls_mm:.4f}" if mls_mm is not None else "N/A")
                st.markdown("**Probability Stats**")
                st.write(f"Min: {pmin:.4f}")
                st.write(f"Max: {pmax:.4f}")
                st.write(f"Mean: {pmean:.4f}")
                st.write(f"Std: {pstd:.4f}")

                st.markdown("---")
                st.write("Volume info:")
                st.write(f"shape: {vol.shape}")
                st.write(f"spacing: {spacing}")

else:
    st.info("Provide input and press **Run Inference** (or upload a .npy/.nii file).")
    st.caption("You can change threshold, toggle heatmap/mask overlays, and save outputs.")

