import argparse, os
import torch, numpy as np, nibabel as nib
from models.unet3d import UNet3D
from dataset.loader import load_volume_any, resample_to_shape, save_nifti
from scipy.ndimage import center_of_mass
from utils.mls import compute_mls_advanced

def compute_mls_from_mask(mask, affine=None):
    H,W,D = mask.shape
    mid_x = W/2.0
    spacing_x = 1.0
    max_disp = 0.0; max_z = None
    for z in range(D):
        sl = mask[:,:,z]
        if sl.sum()==0: continue
        cy, cx = center_of_mass(sl)
        disp_px = abs(cx - mid_x)
        disp_mm = disp_px * spacing_x
        if disp_mm > max_disp:
            max_disp = disp_mm; max_z = z
    return max_disp, max_z

def predict(model_path, nifti_path, out_dir="outputs", threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # load + normalize + resample (returns vol in 0..1)
    vol, affine, spacing = load_volume_any(nifti_path, target_spacing=(1.0,1.0,5.0), do_window=True)

# apply a fast brain extraction to remove skull/air and reduce false positives
    from dataset.preprocessing import basic_brain_mask, apply_brain_mask

    brain_mask = basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=500)
    vol = apply_brain_mask(vol, brain_mask)

    vol = window_and_normalize(vol)
    vol = np.clip(vol, -100, 200)
    vol = (vol - (-100)) / (200 - (-100) + 1e-9)
    volr = resample_to_shape(vol, (64,64,32))
    x = torch.tensor(volr[np.newaxis, np.newaxis,...]).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0,0]
    mask = (probs >= threshold).astype("uint8")
    out_mask = os.path.join(out_dir, "pred_mask.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype("uint8"), np.eye(4)), out_mask)
    mls_mm, zslice = compute_mls_advanced(mask, spacing=spacing)
    print("Saved mask:", out_mask)
    print(f"Max MLS (mm): {mls_mm:.2f} (slice {z})")
    return out_mask, mls_mm, z

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='outputs')
    p.add_argument('--threshold', type=float, default=0.5)
    args = p.parse_args()
    predict(args.model, args.input, out_dir=args.out, threshold=args.threshold)
