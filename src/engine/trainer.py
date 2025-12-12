import os, torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.loader import Simple3DDataset
from models.unet3d import UNet3D
def dice_loss_logits(pred_logits, target, smooth=1e-6):
    """
    Computes Dice loss using logits (before sigmoid).
    """
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1).float()

    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)

    dice = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def train(train_dir="data/mls/train", epochs=1, batch_size=1, out="checkpoints/mls3d.pth"):
    os.makedirs("checkpoints", exist_ok=True)
    ds = Simple3DDataset(train_dir, target_shape=(64,64,32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for x,y,aff in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            logits = model(x)
            # new combined loss
            bce = nn.BCEWithLogitsLoss()

            loss_dice = dice_loss_logits(logits, y)
            loss_bce  = bce(logits, y)

            loss = 0.5 * loss_dice + 0.5 * loss_bce

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(loader):.4f}, time: {time.time()-t0:.1f}s")
    torch.save({"state_dict": model.state_dict()}, out)
    print("Saved checkpoint:", out)
