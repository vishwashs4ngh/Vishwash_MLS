import nibabel as nib
import numpy as np
import os

p = os.path.join("outputs", "pred_mask.nii.gz")
img = nib.load(p)
arr = img.get_fdata().astype("uint8")
np.save(os.path.join("outputs", "pred_mask.npy"), arr)

print("Saved outputs/pred_mask.npy")
