# create_samples.py
import os, numpy as np

d = os.path.join("data", "mls", "train")
os.makedirs(d, exist_ok=True)

def make_volume(shape=(80,80,40), shift=0):
    vol = np.zeros(shape, dtype='float32')
    cx = shape[1]//2 + shift
    for z in range(shape[2]):
        vol[shape[0]//2-1:shape[0]//2+1, cx-1:cx+1, z] = 1.0
    vol += np.random.randn(*shape).astype('float32') * 0.05
    return vol

if not os.path.exists(os.path.join(d, "vol_001.npy")):
    np.save(os.path.join(d, "vol_001.npy"), make_volume(shift=0))

if not os.path.exists(os.path.join(d, "vol_002.npy")):
    np.save(os.path.join(d, "vol_002.npy"), make_volume(shift=3))

print("Sample volumes created at:", d)
