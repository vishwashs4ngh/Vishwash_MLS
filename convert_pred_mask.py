# convert_pred_mask.py
import os
import sys
import nibabel as nib
import numpy as np

p = os.path.join("outputs", "pred_mask.nii.gz")
if len(sys.argv) > 1:
    p = sys.argv[1]

if not os.path.exists(p):
    print("No such file:", p)
    sys.exit(1)

img = nib.load(p)
arr = img.get_fdata().astype("uint8")
out = os.path.join(os.path.dirname(p), os.path.splitext(os.path.basename(p))[0] + ".npy")
np.save(out, arr)
print("Saved", out)
