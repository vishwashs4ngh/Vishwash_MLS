# test_step5_loss.py (self-contained)
import os, sys
import torch
import torch.nn as nn

# ensure src is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.unet3d import UNet3D

def dice_loss_logits(pred_logits, target, smooth=1e-6):
    """
    Local copy of Dice loss that accepts logits as input.
    """
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1).float()

    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)

    dice = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

print("=== STEP 5 SMOKE TEST: Dice+BCE Loss (self-contained) ===")

# fake model output
model = UNet3D(in_ch=1, out_ch=1, base=16)

x = torch.randn(1,1,64,64,32)
y = (torch.rand(1,1,64,64,32) > 0.7).float()  # random binary mask

logits = model(x)

# compute combined loss
bce = nn.BCEWithLogitsLoss()
loss_dice = dice_loss_logits(logits, y)
loss_bce = bce(logits, y)

loss = 0.5 * loss_dice + 0.5 * loss_bce

print("Loss Dice:", float(loss_dice))
print("Loss BCE :", float(loss_bce))
print("Combined :", float(loss))
print("=== STEP 5 COMPLETED ===")
