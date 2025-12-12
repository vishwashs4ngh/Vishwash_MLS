import os, sys, torch

THIS = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(THIS, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if THIS not in sys.path:
    sys.path.insert(0, THIS)

try:
    from models.unet3d import UNet3D
except Exception:
    from unet3d import UNet3D

def load_checkpoint_state(path):
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict):
        for key in ("state_dict","model_state_dict","state"):
            if key in d:
                return d[key]
        for key in ("model","net"):
            if key in d and isinstance(d[key], dict):
                return d[key]
        return d
    return d

def export(checkpoint_path, onnx_out, in_ch=1, out_ch=1, base=16,
           opset=11, dummy_shape=(1,1,32,64,64)):
    print(f"Using UNet3D(in_ch={in_ch}, out_ch={out_ch}, base={base})")
    model = UNet3D(in_ch=in_ch, out_ch=out_ch, base=base)
    model.eval()

    print("Loading checkpoint:", checkpoint_path)
    state = load_checkpoint_state(checkpoint_path)
    try:
        model.load_state_dict(state)
        print("Loaded state_dict directly.")
    except Exception:
        cleaned = {k.replace('module.',''):v for k,v in state.items()}
        model.load_state_dict(cleaned)
        print("Loaded state_dict after stripping 'module.'")

    dummy = torch.randn(*dummy_shape)
    print("Exporting ONNX with dummy shape:", dummy.shape)

    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0:"batch", 2:"depth", 3:"height", 4:"width"},
            "output": {0:"batch", 2:"depth", 3:"height", 4:"width"}
        }
    )
    print("Exported ONNX to:", onnx_out)

if __name__ == "__main__":
    ck1 = os.path.join(THIS, "best_mls3d.pth")
    ck2 = os.path.join(THIS, "mls3d.pth")
    ck = ck1 if os.path.exists(ck1) else (ck2 if os.path.exists(ck2) else None)
    if ck is None:
        raise SystemExit("No checkpoint found (best_mls3d.pth or mls3d.pth)")
    out = os.path.join(THIS, "mls3d.onnx")
    export(ck, out)
