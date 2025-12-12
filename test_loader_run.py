# test_loader_run.py - quick loader smoke test
import os
import numpy as np
from src.dataset import loader

print("Using loader file:", loader.__file__)

# create a random array and save as .npy
os.makedirs("data/mls/train", exist_ok=True)
p = os.path.join("data","mls","train","test_tmp.npy")
arr = (np.random.randn(40,40,20)*100).astype('float32')
np.save(p, arr)
print("Saved test file:", p)

# load via loader
try:
    vol, affine, spacing = loader.load_volume_any(p, target_spacing=(1.0,1.0,5.0), do_window=True)
    print("Loaded volume shape:", vol.shape)
    print("Returned spacing:", spacing)
    print("Min/Max after window:", float(vol.min()), float(vol.max()))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)
