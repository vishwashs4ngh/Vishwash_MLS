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
