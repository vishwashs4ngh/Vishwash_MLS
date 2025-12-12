# view_mask.py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

p = os.path.join("outputs", "pred_mask.nii.gz")
img = nib.load(p)
mask = img.get_fdata()
print("mask shape:", mask.shape, "dtype:", mask.dtype)

# show middle slice along z
zmid = mask.shape[2] // 2
plt.imshow(mask[:, :, zmid], cmap='gray')
plt.title(f"pred_mask slice {zmid}")
plt.axis('off')
plt.show()
