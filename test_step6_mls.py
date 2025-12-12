# test_step6_mls.py  (corrected centered test)
import os, sys
import numpy as np

# ensure src is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.mls import compute_mls_advanced

print("=== STEP 6 SMOKE TEST: Advanced MLS computation (corrected) ===")

# create an empty mask with two symmetric hemorrhage blobs shifted to the right
H, W, D = 64, 80, 32
mask = np.zeros((H, W, D), dtype=np.uint8)

# create left and right blobs on a chosen slice (shifted right to create MLS)
z = 16
mask[28:36, 18:24, z] = 1         # left blob
mask[28:36, 44:50, z] = 1         # right blob (shift produces MLS)

# compute mls with standard spacing (row_mm, col_mm, slice_mm)
spacing = (1.0, 1.0, 5.0)
mls_mm, slice_idx = compute_mls_advanced(mask, spacing=spacing)

print("Computed MLS (mm):", mls_mm)
print("Slice index:", slice_idx)

# test case: truly centered mask (place it around image center)
mask2 = np.zeros_like(mask)
center_x = W // 2
mask2[28:36, center_x-4:center_x+4, z] = 1  # centered 8-px block

mls2, s2 = compute_mls_advanced(mask2, spacing=spacing)
print("Centered mask MLS (mm):", mls2, "slice:", s2)

print("=== STEP 6 COMPLETED ===")
