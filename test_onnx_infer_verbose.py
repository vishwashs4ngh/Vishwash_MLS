# test_onnx_infer_verbose.py
import os, sys, numpy as np

# ensure src importable
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import nibabel as nib
from dataset.loader import load_volume_any
from utils.mls import compute_mls_advanced
from dataset.preprocessing import basic_brain_mask, apply_brain_mask
from matplotlib import pyplot as plt

print("=== ONNX INFERENCE TEST (VERBOSE) ===")

onnx_path = "checkpoints/mls3d.onnx"
if not os.path.exists(onnx_path):
    raise FileNotFoundError("ONNX model not found. Run export_onnx.py first.")

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime") from e

in_path = "data/mls/train/vol_001.npy"
if not os.path.exists(in_path):
    raise FileNotFoundError(f"Input not found: {in_path}")

# 1) load + preprocess (window, resample done by loader)
vol, affine, spacing = load_volume_any(in_path, target_spacing=(1.0,1.0,5.0), do_window=True)
print("Loaded vol dtype:", vol.dtype, "shape:", vol.shape)

# ensure float32
vol = vol.astype(np.float32)

print("Intensity stats: min %.4f max %.4f mean %.4f std %.4f" %
      (vol.min(), vol.max(), vol.mean(), vol.std()))

# 2) apply brain mask
brain_mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=300)
if brain_mask is None:
    raise RuntimeError("basic_brain_mask returned None - check intensity_thresh and loader output")

vol_masked = apply_brain_mask(vol, brain_mask)
print("Applied brain mask. Nonzero voxels:", np.count_nonzero(vol_masked))

# 3) prepare input for ONNX (N,C,D,H,W)
if vol_masked.ndim != 3:
    raise RuntimeError("Expected vol_masked to be 3D (H,W,D). Got shape: %s" % (vol_masked.shape,))

inp = vol_masked[np.newaxis, np.newaxis, :, :, :]   # (1,1,H,W,D)
inp = np.transpose(inp, (0,1,4,2,3)).astype(np.float32)  # (1,1,D,H,W)
print("Prepared ONNX input shape (N,C,D,H,W):", inp.shape, "dtype:", inp.dtype)

# 4) inspect ONNX model IO
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print("ONNX inputs:", sess.get_inputs())
print("ONNX outputs:", sess.get_outputs())

input_name = sess.get_inputs()[0].name
onnx_shape = sess.get_inputs()[0].shape
print("ONNX input shape signature:", onnx_shape)

# 5) run ONNX
try:
    out = sess.run(None, {input_name: inp})[0]   # (1,1,D,H,W)
except Exception as e:
    raise RuntimeError("ONNX runtime failed. Check input dtype/shape.") from e

if out.ndim != 5:
    raise RuntimeError("Unexpected ONNX output shape:", out.shape)

pred = out[0,0]   # (D,H,W)
pred_hwd = np.transpose(pred, (1,2,0))   # -> (H,W,D)
print("Model output (raw logits/prob map) shape:", pred_hwd.shape,
      "min/max:", pred_hwd.min(), pred_hwd.max())


# ============================================================
# 6) LOGITS → PROBABILITIES → THRESHOLD → SAVE
# ============================================================

print("Model output min/max before sigmoid:",
      float(pred_hwd.min()), float(pred_hwd.max()))

# If values outside [0,1] → treat as logits and apply sigmoid
if pred_hwd.min() < 0.0 or pred_hwd.max() > 1.0:
    z = np.clip(pred_hwd, -50.0, 50.0)  # stability
    probs = 1.0 / (1.0 + np.exp(-z))
    print("Applied sigmoid → probs min/max:",
          float(probs.min()), float(probs.max()))
else:
    probs = pred_hwd
    print("Already probabilities → using directly.")

os.makedirs("outputs", exist_ok=True)

# Save the probability volume (.npz)
np.savez_compressed("outputs/pred_probs_onnx.npz", probs=probs)
print("Saved outputs/pred_probs_onnx.npz")

# Save probability as NIfTI
try:
    nii_probs = nib.Nifti1Image(probs.astype(np.float32),
                                affine if affine is not None else np.eye(4))
    nib.save(nii_probs, "outputs/pred_probs_onnx.nii.gz")
    print("Saved outputs/pred_probs_onnx.nii.gz")
except Exception as e:
    print("Warning: couldn't save probability NIfTI:", e)

# Threshold at 0.5 → binary segmentation mask
pred_mask = (probs > 0.5).astype(np.uint8)

# Save mask (.npz)
np.savez_compressed("outputs/pred_mask_onnx.npz", pred_mask=pred_mask)
print("Saved outputs/pred_mask_onnx.npz")

# Save mask NIfTI
try:
    nii_mask = nib.Nifti1Image(pred_mask.astype(np.uint8),
                               affine if affine is not None else np.eye(4))
    nib.save(nii_mask, "outputs/pred_mask_onnx.nii.gz")
    print("Saved outputs/pred_mask_onnx.nii.gz")
except Exception as e:
    print("Warning: couldn't save mask NIfTI:", e)


# ============================================================
# 7) compute MLS
# ============================================================

mls_mm, zslice = compute_mls_advanced(pred_mask, spacing=spacing)
print("Computed MLS (mm):", mls_mm, "slice index:", zslice)


# ============================================================
# 8) overlay PNG
# ============================================================

z_use = int(zslice) if zslice is not None else pred_mask.shape[2] // 2
inp_slice = vol[:, :, z_use]
mask_slice = pred_mask[:, :, z_use] > 0

plt.imsave(os.path.join("outputs", f"input_slice_onnx_{z_use}.png"),
           inp_slice, cmap='gray')

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(inp_slice, cmap='gray')
ax.imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
          cmap='Reds', alpha=0.5)
ax.axis('off')
fig.tight_layout(pad=0)
fig.savefig(os.path.join("outputs", f"overlay_onnx_{z_use}.png"),
            bbox_inches='tight', pad_inches=0)
plt.close(fig)

print("Saved overlay PNGs for slice", z_use)
print("=== ONNX INFERENCE TEST (VERBOSE) COMPLETED ===")
