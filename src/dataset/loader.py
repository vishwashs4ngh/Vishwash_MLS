# --- spacing-aware loader helpers (paste into src/dataset/loader.py) ---
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import os

def window_and_normalize(vol, wmin=-40, wmax=80):
    """
    Clip CT values to [wmin, wmax] then scale to [0,1].
    vol: numpy array (H,W,D)
    """
    v = np.array(vol, dtype='float32')
    v = np.clip(v, wmin, wmax)
    v = (v - wmin) / (wmax - wmin + 1e-9)
    return v

def estimate_spacing_from_nib(img):
    """
    Given a nibabel image, return voxel spacing as (row_mm, col_mm, slice_mm)
    """
    try:
        zooms = img.header.get_zooms()[:3]
        # nib returns (x, y, z) where x -> cols, y -> rows; we will use (row, col, slice) convention
        return (zooms[1], zooms[0], zooms[2])
    except Exception:
        return (1.0, 1.0, 1.0)

def resample_to_spacing(vol, current_spacing=(1.0,1.0,1.0), new_spacing=(1.0,1.0,5.0), order=1):
    """
    Resample a 3D numpy volume from current_spacing to new_spacing.
    vol: numpy array (H,W,D)
    current_spacing, new_spacing: tuples (row_mm, col_mm, slice_mm)
    order: interpolation order (1 = linear for image, 0 = nearest for masks)
    """
    cs = np.array(current_spacing, dtype=float)
    ns = np.array(new_spacing, dtype=float)
    # calculate zoom factors for (rows,cols,slices)
    factors = cs / ns
    # scipy.ndimage.zoom expects zoom factor per axis in the same order as array shape
    # ensure we pass factors matching vol.shape order
    res = ndi.zoom(vol, factors, order=order)
    return res

def load_nifti(path, target_spacing=(1.0,1.0,5.0), do_window=True):
    """
    Load a NIfTI (.nii/.nii.gz) and return (vol, affine, spacing)
    vol will be resampled to target_spacing (physical mm).
    """
    img = nib.load(path)
    vol = img.get_fdata()
    spacing = estimate_spacing_from_nib(img)  # (row_mm, col_mm, slice_mm)
    # optionally window+normalize first (works on HU)
    if do_window:
        vol = window_and_normalize(vol)
    # resample to target spacing
    vol_rs = resample_to_spacing(vol, current_spacing=spacing, new_spacing=target_spacing, order=1)
    return vol_rs.astype('float32'), img.affine, target_spacing

def load_volume_any(path, target_spacing=(1.0,1.0,5.0), do_window=True):
    """
    Load either:
      - a NumPy .npy file -> returns (vol, affine, spacing)
      - a NIfTI file (.nii / .nii.gz) using nibabel -> returns (vol, affine, spacing)
    Numpy files are assumed to be in array space; spacing will be returned as target_spacing unless metadata is provided separately.
    """
    if isinstance(path, (list, tuple)):
        path = path[0]
    path = str(path)
    if path.endswith('.npy'):
        vol = np.load(path).astype('float32')
        if do_window:
            try:
                vol = window_and_normalize(vol)
            except Exception:
                vol = vol
        # resample assuming original spacing 1,1,1 -> to target spacing
        vol_rs = resample_to_spacing(vol, current_spacing=(1.0,1.0,1.0), new_spacing=target_spacing, order=1)
        return vol_rs.astype('float32'), np.eye(4), target_spacing

    # otherwise, try nifti
    try:
        return load_nifti(path, target_spacing=target_spacing, do_window=do_window)
    except Exception as e:
        # last resort: try loading as numpy
        vol = np.load(path).astype('float32')
        if do_window:
            try:
                vol = window_and_normalize(vol)
            except Exception:
                pass
        vol_rs = resample_to_spacing(vol, current_spacing=(1.0,1.0,1.0), new_spacing=target_spacing, order=1)
        return vol_rs.astype('float32'), np.eye(4), target_spacing
# --- end of spacing-aware loader helpers ---
