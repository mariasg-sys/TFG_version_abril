# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNET3D(nn.Module):
    """
    U-Net 3D multiclase para segmentacion volumetrica por parches.

    Entrada: [B, 1, D, H, W]
    Salida:  [B, NUM_CLASSES, D, H, W]
    """

    def __init__(self, in_channels=1, out_channels=16, features=(16, 32, 64, 128), dropout=0.15):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2, dropout=dropout)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def test():
    x = torch.randn((1, 1, 64, 128, 128))
    model = UNET3D(in_channels=1, out_channels=16)
    y = model(x)
    print(y.shape)
    assert y.shape == (1, 16, 64, 128, 128)


if __name__ == "__main__":
    test()
