# compute_metrics.py
import sys, os, numpy as np
import nibabel as nib
from inference import compute_mls_from_mask
from dataset.loader import load_volume_any

def load_mask(path):
    if path.endswith(".npy"):
        return np.load(path).astype(bool)
    else:
        return nib.load(path).get_fdata() > 0.5

def dice(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p,g).sum()
    denom = p.sum() + g.sum()
    return 2.0 * inter / (denom + 1e-9)

def main(pred_path, gt_path):
    if not os.path.exists(pred_path):
        print("Pred not found:", pred_path); return
    if not os.path.exists(gt_path):
        print("GT not found:", gt_path); return
    pred = load_mask(pred_path).astype(np.uint8)
    gt = load_mask(gt_path).astype(np.uint8)
    d = dice(pred, gt)
    pred_mls, p_z = compute_mls_from_mask(pred.astype(np.uint8))
    gt_mls, g_z = compute_mls_from_mask(gt.astype(np.uint8))
    mae = abs(pred_mls - gt_mls)
    print(f"Dice: {d:.4f}")
    print(f"Pred MLS: {pred_mls:.3f} mm (slice {p_z})")
    print(f"GT MLS:   {gt_mls:.3f} mm (slice {g_z})")
    print(f"MLS MAE: {mae:.3f} mm")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_metrics.py <pred_mask.(npy|nii.gz)> <gt_mask.(npy|nii.gz)>")
    else:
        main(sys.argv[1], sys.argv[2])
