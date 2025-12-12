# batch_infer.py
import os, glob, argparse
from src.dataset.loader import load_volume_any, resample_to_shape, save_nifti
import torch, numpy as np
from models.unet3d import UNet3D
from src.inference import compute_mls_from_mask  # or reimplement compute_mls_from_mask here

def main(model_path, folder, out_dir="outputs_batch", thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")
    model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
    ck = torch.load(model_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    files = sorted(glob.glob(os.path.join(folder, "*")))
    for f in files:
        vol, affine = load_volume_any(f)
        volr = resample_to_shape(vol, (64,64,32))
        x = torch.tensor(volr[np.newaxis, np.newaxis,...]).float().to(device)
        with torch.no_grad():
            logits = model(x)
        mask = (torch.sigmoid(logits)[0,0].cpu().numpy() > thr).astype("uint8")
        base = os.path.splitext(os.path.basename(f))[0]
        outn = os.path.join(out_dir, base + "_pred.nii.gz")
        save_nifti(mask, np.eye(4), outn)
        print("Wrote:", outn)
if __name__=="__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
