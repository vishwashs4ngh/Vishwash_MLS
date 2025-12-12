# test_step8_infer.py
import os, sys, numpy as np

# ensure src package is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
import nibabel as nib
from models.unet3d import UNet3D
from dataset.loader import load_volume_any
from utils.mls import compute_mls_advanced
from dataset.preprocessing import basic_brain_mask, apply_brain_mask
from matplotlib import pyplot as plt

print("=== STEP 8 SMOKE TEST: Full inference + MLS ===")

# choose checkpoint (prefer best if exists)
ckpt_candidates = [
    "checkpoints/best_mls3d.pth",
    "checkpoints/mls3d.pth",
    "checkpoints/mls_epoch1.pth",
]
ckpt = None
for c in ckpt_candidates:
    if os.path.exists(c):
        ckpt = c
        break
if ckpt is None:
    raise FileNotFoundError("No checkpoint found in checkpoints/. Run run_train.py first.")

print("Using checkpoint:", ckpt)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_ch=1, out_ch=1, base=16, pdrop=0.1).to(device)
ck = torch.load(ckpt, map_location=device)
if "state_dict" in ck:
    model.load_state_dict(ck["state_dict"])
else:
    model.load_state_dict(ck)
model.eval()

# input volume path (sample)
in_path = "data/mls/train/vol_001.npy"
if not os.path.exists(in_path):
    # try vol_001 from earlier steps
    raise FileNotFoundError(f"Input not found: {in_path}. Create samples with create_samples.py or run run_all.bat")

# 1) load volume (returns resampled normalized vol, affine, spacing)
vol, affine, spacing = load_volume_any(in_path, target_spacing=(1.0,1.0,5.0), do_window=True)
print("Loaded vol shape:", vol.shape, "spacing:", spacing)

# 2) apply brain mask (same as inference pipeline)
brain_mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=300)
vol_masked = apply_brain_mask(vol, brain_mask)

# 3) prepare torch input: vol is (H,W,D); our model expects (N,C,D,H,W)
inp = torch.from_numpy(vol_masked[np.newaxis, np.newaxis, :, :, :]).permute(0,1,4,2,3).contiguous().to(device).float()
# shape now (1,1,D,H,W)
print("Input tensor shape:", inp.shape)

# 4) forward
with torch.no_grad():
    logits = model(inp)
    probs = torch.sigmoid(logits).cpu().numpy()[0,0]  # (D,H,W) after permute inverse -> we arranged to (N,1,D,H,W)
    # convert back to (H,W,D)
    probs_hwd = np.transpose(probs, (1,2,0))  # (H,W,D)

# 5) threshold
pred_mask = (probs_hwd > 0.5).astype('uint8')

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# 6) save as numpy
np.save(os.path.join("outputs","pred_mask.npy"), pred_mask)
print("Saved outputs/pred_mask.npy")

# 7) save as NIfTI (use identity affine if none)
try:
    affine_save = affine if affine is not None else np.eye(4)
    # nib expects data in (X,Y,Z) = (W,H,D), but our array is (H,W,D) — nibabel will accept as is
    nii = nib.Nifti1Image(pred_mask.astype('uint8'), affine_save)
    nib.save(nii, os.path.join("outputs","pred_mask.nii.gz"))
    print("Saved outputs/pred_mask.nii.gz")
except Exception as e:
    print("Warning: could not save NIfTI:", e)

# 8) compute MLS using spacing returned by loader
mls_mm, zslice = compute_mls_advanced(pred_mask, spacing=spacing)
print(f"Computed MLS (mm): {mls_mm:.3f}, slice: {zslice}")

# 9) try to compute Dice if GT exists (gt path same prefix + _gt or *_mask)
gt_candidate = in_path.replace(".npy", "_gt.npy")
if not os.path.exists(gt_candidate):
    # try vol_001_mask.npy or vol_001_gt.nii.gz
    alt = os.path.join(os.path.dirname(in_path), os.path.basename(in_path).replace(".npy","_mask.npy"))
    if os.path.exists(alt):
        gt_candidate = alt
    elif os.path.exists(in_path.replace(".npy", "_gt.nii.gz")):
        gt_candidate = in_path.replace(".npy", "_gt.nii.gz")
    else:
        gt_candidate = None

if gt_candidate and os.path.exists(gt_candidate):
    # load gt (npy or nifti)
    if gt_candidate.endswith(".npy"):
        gt = np.load(gt_candidate).astype('uint8')
    else:
        gt = nib.load(gt_candidate).get_fdata().astype('uint8')
    inter = np.logical_and(pred_mask>0, gt>0).sum()
    denom = pred_mask.sum() + gt.sum()
    dice = 2.0 * inter / (denom + 1e-9)
    print("Dice (pred vs GT):", float(dice))
else:
    print("No GT found for Dice calculation.")

# 10) save overlay PNGs (pick slice zslice or middle)
z_use = int(zslice) if zslice is not None else pred_mask.shape[2]//2
inp_slice = vol[:,:,z_use]
mask_slice = pred_mask[:,:,z_use] > 0

# save input slice
plt.imsave(os.path.join("outputs", f"input_slice_{z_use}.png"), inp_slice, cmap='gray')
# save overlay
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(inp_slice, cmap='gray')
ax.imshow(np.ma.masked_where(mask_slice==0, mask_slice), cmap='Reds', alpha=0.5)
ax.axis('off')
fig.tight_layout(pad=0)
fig.savefig(os.path.join("outputs", f"overlay_{z_use}.png"), bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved overlay PNGs for slice", z_use)

print("=== STEP 8 COMPLETED ===")
