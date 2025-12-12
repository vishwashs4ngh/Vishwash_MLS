# save_overlay_pngs.py
import sys, os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_vol(path):
    if path.endswith(".npy"):
        return np.load(path)
    else:
        return nib.load(path).get_fdata()

def save_png(arr, outpath, cmap='gray'):
    plt.figure(figsize=(6,6))
    plt.imshow(arr, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):
    if len(args) < 3:
        print("Usage: python save_overlay_pngs.py <outputs_folder> <mask_filename>")
        return
    outdir = args[1]
    maskfile = args[2]
    maskpath = os.path.join(outdir, maskfile)
    if not os.path.exists(maskpath):
        print("Mask not found:", maskpath)
        return
    # try to find an input volume in same folder or data/mls/train
    sample_input = os.path.join("data","mls","train","vol_001.npy")
    if os.path.exists(sample_input):
        vol = load_vol(sample_input)
    else:
        # try to infer input next to mask
        vol = np.zeros_like(load_vol(maskpath))
    mask = load_vol(maskpath)
    # choose slice: middle or slice with max area
    if mask.ndim == 3:
        slice_idx = int(np.argmax([np.sum(mask[:,:,z]) for z in range(mask.shape[2])]))
    else:
        slice_idx = mask.shape[2]//2
    inp_slice = vol[:,:,slice_idx]
    mask_slice = mask[:,:,slice_idx] > 0.5
    os.makedirs("outputs", exist_ok=True)
    save_png(inp_slice, os.path.join(outdir, f"input_slice_{slice_idx}.png"))
    # overlay
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(inp_slice, cmap='gray')
    ax.imshow(np.ma.masked_where(mask_slice==0, mask_slice), cmap='Reds', alpha=0.5)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(outdir, f"overlay_{slice_idx}.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print("Saved input and overlay pngs at slice", slice_idx)

if __name__ == "__main__":
    main(sys.argv)
