# data.py - lightweight NIfTI dataset utilities for 3D volumes
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype('float32'), img.affine

def resample_to_shape(vol, target_shape=(128,128,64)):
    factors = (target_shape[0]/vol.shape[0], target_shape[1]/vol.shape[1], target_shape[2]/vol.shape[2])
    vol_r = zoom(vol, factors, order=1)
    return vol_r

class Simple3DDataset:
    """
    Expects a folder of NIfTI files. For MLS, we assume ground-truth masks are not present
    in this minimal example; training will use the volume intensity to create a pseudo-mask
    if needed. Replace with real mask loader if you have manual annotations.
    """
    def __init__(self, folder, target_shape=(128,128,64)):
        self.files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.target_shape = target_shape
    def __len__(self):
        return max(1, len(self.files))
    def __getitem__(self, idx):
        if len(self.files)==0:
            # synthetic fallback
            vol = np.zeros(self.target_shape, dtype='float32')
            cx = self.target_shape[0]//2 + np.random.randint(-3,3)
            for z in range(self.target_shape[2]):
                vol[self.target_shape[0]//2-1:self.target_shape[0]//2+1, cx-1:cx+1, z] = 1.0
            mask = (vol>0.1).astype('float32')
            return vol[np.newaxis,...], mask[np.newaxis,...]
        path = self.files[idx % len(self.files)]
        vol, affine = load_nifti(path)
        vol = np.clip(vol, -100, 200)
        vol = (vol - (-100)) / (200 - (-100) + 1e-9)
        volr = resample_to_shape(vol, self.target_shape)
        # create pseudo mask by thresholding (replace with your true mask loader)
        mask = (volr > 0.5).astype('float32')
        return volr[np.newaxis,...], mask[np.newaxis,...]
