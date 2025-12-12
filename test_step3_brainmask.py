# test_step3_brainmask.py
import os
import numpy as np

from src.dataset.loader import load_volume_any
from src.dataset.preprocessing import basic_brain_mask, apply_brain_mask

print("=== STEP 3 SMOKE TEST: Brain Mask ===")

# 1) Ensure folder exists
os.makedirs("data/mls/train", exist_ok=True)

# 2) Create a random test volume
test_path = "data/mls/train/test_tmp_step3.npy"
arr = (np.random.randn(40,40,20)*100).astype('float32')
np.save(test_path, arr)

print("Saved test volume:", test_path)

# 3) Load with loader (window + normalization + resample)
vol, affine, spacing = load_volume_any(
    test_path,
    target_spacing=(1.0,1.0,5.0),
    do_window=True
)

print("Loaded vol shape:", vol.shape)
print("Intensity range:", float(vol.min()), float(vol.max()))
print("Returned spacing:", spacing)

# 4) Generate brain mask
mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=300)
print("Brain mask voxels:", int(mask.sum()))

# 5) Apply mask
vol_masked = apply_brain_mask(vol, mask)
print("Masked volume nonzero voxels:", int(np.count_nonzero(vol_masked)))

print("=== STEP 3 COMPLETED ===")
