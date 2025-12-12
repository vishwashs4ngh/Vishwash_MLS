# run_train.py
# Robust training script for MLS project (handles DataLoader tensor/numpy cases)
# - Uses windowing + simple brain mask
# - Augmentations applied in dataset
# - Dice + BCE combined loss
# - Writes checkpoints/train_log.csv and saves best checkpoint

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# dataset helpers (from src/dataset)
from dataset.loader import window_and_normalize
# augment utilities
try:
    from dataset.augment import augment_volume
except Exception:
    def augment_volume(v, m=None, **k):  # fallback no-op
        return (v, m) if m is not None else v
# optional brain mask helper
try:
    from dataset.preprocessing import basic_brain_mask, apply_brain_mask
    _HAVE_BRAIN_MASK = True
except Exception:
    _HAVE_BRAIN_MASK = False

# model
from models.unet3d import UNet3D

# ---------------------------
# Loss helpers
# ---------------------------
def dice_loss_logits(pred_logits, target, smooth=1e-6):
    """
    Dice loss from logits.
    pred_logits: tensor (N,1,D,H,W) raw outputs
    target: tensor (N,1,D,H,W) 0/1
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1).float()
    inter = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2*inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()

# ---------------------------
# Dataset
# ---------------------------
class Npy3DDataset(Dataset):
    def __init__(self, folder, target_shape=(64,64,32), augment=True):
        self.folder = folder
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])
        self.target_shape = tuple(target_shape)
        self.augment = augment

    def __len__(self):
        return max(1, len(self.files))

    def _pad_or_crop(self, vol):
        ts = self.target_shape
        out = np.zeros(ts, dtype='float32')
        cz = min(ts[2], vol.shape[2])
        cy = min(ts[0], vol.shape[0])
        cx = min(ts[1], vol.shape[1])
        out[:cy, :cx, :cz] = vol[:cy, :cx, :cz]
        return out

    def __getitem__(self, idx):
        # if no files, synth sample (keeps training pipeline working)
        if len(self.files) == 0:
            vol = np.zeros(self.target_shape, dtype='float32')
            cx = self.target_shape[1] // 2
            for z in range(self.target_shape[2]):
                vol[self.target_shape[0]//2-1:self.target_shape[0]//2+1, cx-1:cx+1, z] = 1.0
            mask = (vol > 0.1).astype('float32')
            if self.augment:
                vol, mask = augment_volume(vol, mask)
            return vol[np.newaxis,...], mask[np.newaxis,...]

        p = self.files[idx % len(self.files)]
        vol = np.load(p).astype('float32')

        # window + normalize (expects HU-ish or already scaled)
        try:
            vol = window_and_normalize(vol)
        except Exception:
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)

        # optional brain mask application (helps remove skull)
        if _HAVE_BRAIN_MASK:
            try:
                # compute mask on the normalized volume
                bmask = basic_brain_mask(vol, intensity_thresh=0.12, min_volume_vox=300)
                vol = apply_brain_mask(vol, bmask)
            except Exception:
                pass

        # resample/pad/crop to target_shape if needed (assume voxel spacing 1,1,1)
        vol = self._pad_or_crop(vol)

        # create a pseudo mask from intensity (only for synthetic/weak labels)
        mask = (vol > 0.5).astype('float32')

        # augment (vol, mask)
        if self.augment:
            try:
                vol, mask = augment_volume(vol, mask, do_flip=True, do_rot=True, do_jitter=True)
            except Exception:
                # fallback to no augmentation
                pass

        vol = vol.astype('float32')
        mask = mask.astype('float32')
        return vol[np.newaxis, ...], mask[np.newaxis, ...]

# ---------------------------
# Trainer
# ---------------------------
def _to_tensor_and_permute(x_np):
    """
    Accept either numpy arrays or already-torch tensors from DataLoader.
    Convert to torch tensor and permute from (N, C, H, W, D) to (N, C, D, H, W).
    Returns a contiguous float tensor on CPU (caller will move to device).
    """
    if isinstance(x_np, np.ndarray):
        t = torch.from_numpy(x_np)
    elif torch.is_tensor(x_np):
        t = x_np
    else:
        # fallback: try converting via numpy
        t = torch.from_numpy(np.array(x_np))
    # ensure contiguous memory
    t = t.contiguous()
    # permute dims: (N, C, H, W, D) -> (N, C, D, H, W)
    t = t.permute(0,1,4,2,3)
    return t.float()

def train(train_dir="data/mls/train", epochs=1, batch_size=1, out="checkpoints/mls3d.pth"):
    os.makedirs("checkpoints", exist_ok=True)
    ds = Npy3DDataset(train_dir, target_shape=(64,64,32), augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1, out_ch=1, base=16, pdrop=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCEWithLogitsLoss()

    log_path = os.path.join("checkpoints", "train_log.csv")
    best_path = os.path.join("checkpoints", "best_mls3d.pth")
    best_metric = float("inf")
    if os.path.exists(log_path):
        try:
            import pandas as pd
            df = pd.read_csv(log_path)
            if "loss" in df.columns:
                best_metric = float(df["loss"].min())
        except Exception:
            best_metric = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_batches = 0
        for x_np, y_np in loader:
            # convert inputs to tensors and permute to (N,C,D,H,W)
            x = _to_tensor_and_permute(x_np).to(device)
            y = _to_tensor_and_permute(y_np).to(device)

            logits = model(x)
            loss_dice = dice_loss_logits(logits, y)
            loss_bce = bce(logits, y)
            loss = 0.5 * loss_dice + 0.5 * loss_bce

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}, time: {elapsed:.1f}s")

        # append to CSV
        import csv
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["epoch", "loss", "time"])
            writer.writerow([epoch+1, round(avg_loss,6), round(elapsed,3)])

        # save checkpoint each epoch
        ckpt = {"state_dict": model.state_dict(), "epoch": epoch+1, "loss": avg_loss}
        ckpt_path = out if epoch == (epochs-1) else os.path.join("checkpoints", f"mls_epoch{epoch+1}.pth")
        torch.save(ckpt, ckpt_path)

        # save best
        if avg_loss < best_metric:
            torch.save(ckpt, best_path)
            best_metric = avg_loss
            print("Saved best checkpoint:", best_path)

    # final save to out
    final_out = out
    torch.save({"state_dict": model.state_dict()}, final_out)
    print("Saved checkpoint:", final_out)

if __name__ == "__main__":
    # default small test run
    train(epochs=1, batch_size=1)
