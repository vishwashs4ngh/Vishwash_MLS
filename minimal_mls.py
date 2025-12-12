# create_minimal_mls.py
# Run with: py -3 create_minimal_mls.py
import os, textwrap, sys

root = os.path.abspath(os.path.dirname(__file__))

files = {
    "requirements.txt": textwrap.dedent("""\
        torch
        numpy
        nibabel
        scipy
        tqdm
    """),
    os.path.join("src","dataset","loader.py"): textwrap.dedent("""\
        import os
        import numpy as np
        import nibabel as nib
        from scipy.ndimage import zoom

        def load_nifti(path):
            img = nib.load(path)
            return img.get_fdata().astype("float32"), img.affine

        def save_nifti(arr, affine, path):
            nib.save(nib.Nifti1Image(arr.astype("float32"), affine), path)
            return path

        def resample_to_shape(vol, target_shape=(64,64,32)):
            factors = (target_shape[0]/vol.shape[0], target_shape[1]/vol.shape[1], target_shape[2]/vol.shape[2])
            return zoom(vol, factors, order=1)

        class Simple3DDataset:
            \"""
            Minimal dataset for MLS experiments.
            __getitem__ returns: vol[np.newaxis,...], mask[np.newaxis,...], affine
            \"""
            def __init__(self, folder, target_shape=(64,64,32)):
                self.folder = folder
                self.files = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                                     if f.endswith('.nii') or f.endswith('.nii.gz')]) if os.path.exists(folder) else []
                self.target_shape = target_shape

            def __len__(self):
                return max(1, len(self.files))

            def __getitem__(self, idx):
                if len(self.files)==0:
                    # synthetic fallback: thin midline ridge slightly shifted
                    vol = np.zeros(self.target_shape, dtype='float32')
                    cx = self.target_shape[0]//2 + np.random.randint(-3,3)
                    for z in range(self.target_shape[2]):
                        vol[self.target_shape[0]//2-1:self.target_shape[0]//2+1, cx-1:cx+1, z] = 1.0
                    mask = (vol>0.1).astype('float32')
                    return vol[np.newaxis,...], mask[np.newaxis,...], np.eye(4)
                p = self.files[idx % len(self.files)]
                vol, affine = load_nifti(p)
                vol = np.clip(vol, -100, 200)
                vol = (vol - (-100)) / (200 - (-100) + 1e-9)
                volr = resample_to_shape(vol, self.target_shape)
                mask = (volr > 0.5).astype('float32')
                return volr[np.newaxis,...], mask[np.newaxis,...], affine
    """),
    os.path.join("src","models","unet3d.py"): textwrap.dedent("""\
        import torch
        import torch.nn as nn

        class DoubleConv3d(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self,x):
                return self.net(x)

        class UNet3D(nn.Module):
            def __init__(self, in_ch=1, out_ch=1, base=8):
                super().__init__()
                self.enc1 = DoubleConv3d(in_ch, base)
                self.pool = nn.MaxPool3d(2)
                self.enc2 = DoubleConv3d(base, base*2)
                self.enc3 = DoubleConv3d(base*2, base*4)
                self.up2 = nn.ConvTranspose3d(base*4, base*2, 2,2)
                self.dec2 = DoubleConv3d(base*4, base*2)
                self.up1 = nn.ConvTranspose3d(base*2, base, 2,2)
                self.dec1 = DoubleConv3d(base*2, base)
                self.final = nn.Conv3d(base, out_ch, 1)

            def forward(self,x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                d2 = self.up2(e3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                d1 = self.up1(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                out = self.final(d1)
                return out
    """),
    os.path.join("src","engine","trainer.py"): textwrap.dedent("""\
        import os, torch, time
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from dataset.loader import Simple3DDataset
        from models.unet3d import UNet3D

        def train(train_dir="data/mls/train", epochs=1, batch_size=1, out="checkpoints/mls3d.pth"):
            os.makedirs("checkpoints", exist_ok=True)
            ds = Simple3DDataset(train_dir, target_shape=(64,64,32))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                t0 = time.time()
                for x,y,aff in loader:
                    x = x.to(device).float()
                    y = y.to(device).float()
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(loader):.4f}, time: {time.time()-t0:.1f}s")
            torch.save({\"state_dict\": model.state_dict()}, out)
            print("Saved checkpoint:", out)
    """),
    os.path.join("src","inference.py"): textwrap.dedent("""\
        import argparse, os
        import torch, numpy as np, nibabel as nib
        from models.unet3d import UNet3D
        from dataset.loader import load_nifti, resample_to_shape, save_nifti
        from scipy.ndimage import center_of_mass

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

        def predict(model_path, nifti_path, out_dir=\"outputs\", threshold=0.5):
            os.makedirs(out_dir, exist_ok=True)
            device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
            model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt[\"state_dict\"])
            model.eval()
            vol, affine = load_nifti(nifti_path)
            vol = np.clip(vol, -100, 200)
            vol = (vol - (-100)) / (200 - (-100) + 1e-9)
            volr = resample_to_shape(vol, (64,64,32))
            x = torch.tensor(volr[np.newaxis, np.newaxis,...]).float().to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0,0]
            mask = (probs >= threshold).astype(\"uint8\")
            out_mask = os.path.join(out_dir, \"pred_mask.nii.gz\")
            nib.save(nib.Nifti1Image(mask.astype(\"uint8\"), np.eye(4)), out_mask)
            mls_mm, z = compute_mls_from_mask(mask)
            print(\"Saved mask:\", out_mask)
            print(f\"Max MLS (mm): {mls_mm:.2f} (slice {z})\")
            return out_mask, mls_mm, z

        if __name__ == '__main__':
            p = argparse.ArgumentParser()
            p.add_argument('--model', required=True)
            p.add_argument('--input', required=True)
            p.add_argument('--out', default='outputs')
            p.add_argument('--threshold', type=float, default=0.5)
            args = p.parse_args()
            predict(args.model, args.input, out_dir=args.out, threshold=args.threshold)
    """),
    os.path.join("src","midline_utils.py"): textwrap.dedent("""\
        from scipy.ndimage import center_of_mass
        import numpy as np

        def compute_mls_from_mask(mask, affine=None):
            if mask.dtype != np.uint8:
                mask = (mask>0.5).astype('uint8')
            H,W,D = mask.shape
            mid_x = W/2.0
            spacing_x = 1.0
            if affine is not None:
                try:
                    spacing_x = abs(affine[0,0])
                except:
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
    """)
}

# create directories & write files
for relpath, content in files.items():
    abspath = os.path.join(root, relpath)
    d = os.path.dirname(abspath)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(abspath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Wrote:", abspath)

# create folders for data/checkpoints/outputs and a synthetic nifti
os.makedirs(os.path.join(root,"data","mls","train"), exist_ok=True)
os.makedirs(os.path.join(root,"checkpoints"), exist_ok=True)
os.makedirs(os.path.join(root,"outputs"), exist_ok=True)

# create a small synthetic nifti example
try:
    import numpy as _np, nibabel as _nib
    shape = (64,64,32)
    vol = _np.zeros(shape, dtype='float32')
    cx = shape[0]//2 + 2
    for z in range(shape[2]):
        vol[shape[0]//2-1:shape[0]//2+1, cx-1:cx+1, z] = 1.0
    vol = vol + 0.05 * _np.random.randn(*shape).astype('float32')
    _nib.save(_nib.Nifti1Image(vol, _np.eye(4)), os.path.join(root,"data","mls","train","synth_001.nii.gz"))
    print("Synthetic example created at data/mls/train/synth_001.nii.gz")
except Exception as e:
    print("Could not create synthetic example (nibabel missing?):", e)

print("All files created. Next: create venv and install requirements.")
