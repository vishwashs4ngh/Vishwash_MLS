import torch

d = torch.load("checkpoints/mls3d.pth", map_location="cpu")
sd = d["state_dict"]

print("Layers:", len(sd))
print("Total params:", sum(v.numel() for v in sd.values()))
