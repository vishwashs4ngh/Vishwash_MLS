# src/dataset/preprocessing.py
import numpy as np
import scipy.ndimage as ndi

def basic_brain_mask(vol, intensity_thresh=0.15, min_volume_vox=500):
    """
    Very lightweight brain extraction for normalized volumes (0..1).
    Steps:
      - threshold (intensity_thresh)
      - morphological closing
      - keep largest connected component
      - fill holes
    Returns boolean mask same shape as vol.
    """
    # threshold
    bw = vol > intensity_thresh

    # small morphological closing to connect structures
    bw = ndi.binary_closing(bw, structure=np.ones((3,3,3)))

    # label and keep largest connected component
    labels, n = ndi.label(bw)
    if n == 0:
        return np.ones_like(vol, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background removed
    largest_label = np.argmax(counts)
    mask = labels == largest_label

    # remove small islands
    if mask.sum() < min_volume_vox:
        # fallback: return full mask so we do not zero everything
        return np.ones_like(vol, dtype=bool)

    # fill holes
    mask = ndi.binary_fill_holes(mask)

    return mask.astype(bool)


def apply_brain_mask(vol, mask):
    """
    Apply boolean brain mask to volume (keeps zeros outside mask).
    """
    return vol * mask
