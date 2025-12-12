# src/utils/mls.py
import numpy as np
from scipy import ndimage as ndi

def compute_mls_advanced(mask, spacing=(1.0,1.0,5.0)):
    """
    Compute Midline Shift (MLS) robustly.
    Approach:
      - For each axial slice, compute left and right hemisphere centroids (if present)
      - Compute the midpoint between left and right centroids and measure its deviation
        from the image centerline (in mm).
      - Return the maximum displacement (mm) across slices and its slice index.
    Args:
      mask: 3D binary numpy array (H, W, D)
      spacing: (row_mm, col_mm, slice_mm) physical spacing
    Returns:
      (best_mls_mm (float), best_z (int or None))
    """
    if mask is None:
        return 0.0, None
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D (H,W,D)")

    H, W, D = mask.shape
    mm_per_col = float(spacing[1])  # column spacing (x-direction)
    best_mls = 0.0
    best_z = None

    # Precompute midline x coordinate (center)
    midline_x = (W - 1) / 2.0

    for z in range(D):
        slice_mask = mask[:, :, z].astype(bool)
        if slice_mask.sum() < 20:
            # ignore slices with very small mask
            continue

        # split into left and right by image midline
        mid = W // 2
        left = slice_mask[:, :mid]
        right = slice_mask[:, mid:]

        if left.sum() < 10 or right.sum() < 10:
            # need both hemispheric regions for reliable centroid comparison
            continue

        try:
            ly, lx = ndi.center_of_mass(left)
            ry, rx = ndi.center_of_mass(right)
        except Exception:
            continue

        # convert right centroid x to full-image coordinates
        rx_full = rx + mid

        # centroid midpoint between left and right
        centroid_mid_x = (lx + rx_full) / 2.0

        # displacement in columns from anatomical midline
        disp_cols = abs(centroid_mid_x - midline_x)
        disp_mm = disp_cols * mm_per_col

        if disp_mm > best_mls:
            best_mls = float(disp_mm)
            best_z = int(z)

    if best_z is None:
        # fallback: try center-of-mass of entire mask projected to x-axis
        try:
            overall = ndi.center_of_mass(mask)
            if overall is not None:
                # overall returns (y,x,z)
                oc_x = overall[1]
                disp_cols = abs(oc_x - midline_x)
                best_mls = float(disp_cols * mm_per_col)
                best_z = int(round(overall[2])) if overall[2] is not None else 0
        except Exception:
            pass

    return best_mls, best_z
