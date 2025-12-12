# export_onnx.py
import os
import torch
import numpy as np

# ensure src is importable
THIS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(THIS, "src")
import sys
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.unet3d import UNet3D

def export(checkpoint_path="checkpoints/best_mls3d.pth", onnx_out="checkpoints/mls3d.onnx"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run run_train.py first.")
    device = torch.device("cpu")
    model = UNet3D(in_ch=1, out_ch=1, base=16, pdrop=0.1).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
    else:
        model.load_state_dict(ck)
    model.eval()

    # dummy input shape: (N, C, D, H, W) -> pick a shape matching your training/resample (e.g., D=8,H=80,W=80)
    dummy = torch.randn(1,1,8,80,80, dtype=torch.float32, device=device)

    # export
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch", 2: "depth", 3: "height", 4: "width"}
        }
    )
    print("Exported ONNX to:", onnx_out)

if __name__ == "__main__":
    export()
