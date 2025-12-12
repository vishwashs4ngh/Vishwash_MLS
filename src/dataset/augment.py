# src/dataset/augment.py
import numpy as np
import random
from scipy.ndimage import rotate

def random_flip(vol, mask=None, p=0.5):
    """Random left-right flip (axis=1 for (H,W,D) ordering)."""
    if random.random() < p:
        vol = np.flip(vol, axis=1).copy()
        if mask is not None:
            mask = np.flip(mask, axis=1).copy()
    return (vol, mask) if mask is not None else vol

def random_rotate_xy(vol, mask=None, max_angle=10, p=0.5):
    """Random small rotation in the axial plane (degrees)."""
    if random.random() < p:
        angle = random.uniform(-max_angle, max_angle)
        vol = rotate(vol, angle, axes=(0,1), reshape=False, order=1, mode='nearest')
        if mask is not None:
            mask = rotate(mask, angle, axes=(0,1), reshape=False, order=0, mode='nearest')
    return (vol, mask) if mask is not None else vol

def random_intensity_jitter(vol, p=0.5, scale=0.02):
    """Add small gaussian noise to intensities (vol expected 0..1)."""
    if random.random() < p:
        noise = np.random.normal(0, scale, size=vol.shape).astype('float32')
        vol = vol + noise
        vol = np.clip(vol, 0.0, 1.0)
    return vol

def augment_volume(vol, mask=None, do_flip=True, do_rot=True, do_jitter=True):
    """
    Apply a sequence of simple augmentations. Keeps types as numpy arrays.
    Returns (vol, mask) if mask provided, else vol.
    """
    if do_flip:
        res = random_flip(vol, mask)
        vol = res[0]; mask = res[1] if mask is not None else None
    if do_rot:
        res = random_rotate_xy(vol, mask)
        vol = res[0]; mask = res[1] if mask is not None else None
    if do_jitter:
        vol = random_intensity_jitter(vol)
    if mask is not None:
        return vol.astype('float32'), (mask.astype('uint8') if mask is not None else None)
    return vol.astype('float32')
