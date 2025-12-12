# test_step7_augment.py
import os, sys
import numpy as np

# ensure src on path
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset.augment import augment_volume

print("=== STEP 7 SMOKE TEST: Data Augmentation ===")

# create a synthetic volume and mask
H,W,D = 64, 80, 32
vol = np.zeros((H,W,D), dtype='float32')
# put a bright square (simulate a structure)
vol[28:36, 18:26, 16] = 0.9
# normalize to 0..1 (already)
mask = np.zeros((H,W,D), dtype='uint8')
mask[28:36, 18:26, 16] = 1

print("Original center sum (vol):", float(vol[28:36,18:26,16].sum()))
print("Original mask sum:", int(mask.sum()))

# Apply augmentations multiple times and print summaries
for i in range(5):
    v_aug, m_aug = augment_volume(vol, mask, do_flip=True, do_rot=True, do_jitter=True)
    print(f"Aug {i+1}: v_aug mean:{v_aug.mean():.4f} max:{v_aug.max():.4f} mask_sum:{int(m_aug.sum())}")

# As a quick visual check save an example slice as numpy (you can convert to png later)
os.makedirs("outputs", exist_ok=True)
np.save("outputs/aug_example_vol.npy", v_aug)
np.save("outputs/aug_example_mask.npy", m_aug)
print("Saved outputs/aug_example_vol.npy and aug_example_mask.npy")

print("=== STEP 7 COMPLETED ===")
