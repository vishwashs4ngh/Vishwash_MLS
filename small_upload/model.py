# model.py - simple 3D UNet implementation in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.enc1 = DoubleConv3d(in_ch, base)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3d(base, base*2)
        self.enc3 = DoubleConv3d(base*2, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3d(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3d(base*2, base)
        self.final = nn.Conv3d(base, out_ch, kernel_size=1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out
