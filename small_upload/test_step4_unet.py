# test_step4_unet.py
import os
import sys

# ensure project src is on path so imports like "from models.unet3d import UNet3D" work
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from models.unet3d import UNet3D

print("=== STEP 4 SMOKE TEST: UNet import & forward ===")
m = UNet3D(in_ch=1, out_ch=1, base=16)
m.eval()
# create a dummy tensor with shape (N, C, D, H, W)
x = torch.randn(1,1,64,64,32)
with torch.no_grad():
    y = m(x)
print("input shape:", x.shape)
print("output shape:", y.shape)
print("dtype:", y.dtype)
print("min/max output:", float(y.min()), float(y.max()))
print("=== STEP 4 COMPLETED ===")
