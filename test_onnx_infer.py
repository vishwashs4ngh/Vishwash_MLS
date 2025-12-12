# test_onnx_infer.py
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

print("=== ONNX INFERENCE TEST ===")

onnx_path = "checkpoints/mls3d.onnx"
if not os.path.exists(onnx_path):
    raise FileNotFoundError("ONNX model not found. Run export_onnx.py first.")

# minimal import of onnxruntime
try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime") from e

# choose input volume (use same one we used earlier)
in_path = "data/mls/train/vol_001.npy"
if not os.path.exists(in_path):
    raise FileNotFoundError(f"Input not found: {in_path}")

# 1) load + preprocess (window, resample done by loader)
vol, affine, spacing = load_volume_any(in_path, target_spacing=(1.0,1.0,5.0), do_window=True)
print("Loaded vol shape:", vol.shape, "spacing:", spacing)

# 2) apply brain mask
brain_mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=300)
vol_masked = apply_brain_mask(vol, brain_mask)

# 3) prepare input for ONNX (N,C,D,H,W)
# our vol is (H,W,D) -> we need (1,1,D,H,W)
inp = np.transpose(vol_masked[np.newaxis, np.newaxis, :, :, :], (0,1,4,2,3)).astype(np.float32)
print("ONNX input shape (N,C,D,H,W):", inp.shape)

# 4) run ONNX runtime
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: inp})[0]  # out shape (1,1,D,H,W)
pred = out[0,0]  # (D,H,W)
# convert back to (H,W,D)
pred_hwd = np.transpose(pred, (1,2,0))

# 5) threshold & save
pred_mask = (pred_hwd > 0.5).astype('uint8')
os.makedirs("outputs", exist_ok=True)
np.save("outputs/pred_mask_onnx.npy", pred_mask)
print("Saved outputs/pred_mask_onnx.npy")

# save nifti
try:
    nii = nib.Nifti1Image(pred_mask.astype('uint8'), affine if affine is not None else np.eye(4))
    nib.save(nii, "outputs/pred_mask_onnx.nii.gz")
    print("Saved outputs/pred_mask_onnx.nii.gz")
except Exception as e:
    print("Warning: couldn't save NIfTI:", e)

# 6) compute MLS
mls_mm, zslice = compute_mls_advanced(pred_mask, spacing=spacing)
print("Computed MLS (mm):", mls_mm, "slice:", zslice)

# 7) make overlay png
z_use = int(zslice) if zslice is not None else pred_mask.shape[2]//2
inp_slice = vol[:,:,z_use]
mask_slice = pred_mask[:,:,z_use] > 0

plt.imsave(os.path.join("outputs", f"input_slice_onnx_{z_use}.png"), inp_slice, cmap='gray')
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(inp_slice, cmap='gray')
ax.imshow(np.ma.masked_where(mask_slice==0, mask_slice), cmap='Reds', alpha=0.5)
ax.axis('off')
fig.tight_layout(pad=0)
fig.savefig(os.path.join("outputs", f"overlay_onnx_{z_use}.png"), bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved overlay PNGs for slice", z_use)

print("=== ONNX INFERENCE TEST COMPLETED ===")
