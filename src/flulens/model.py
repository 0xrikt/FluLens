import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FluLensModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # ResNeSt-50 backbone
        self.backbone = timm.create_model(
            "resnest50d",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.attn = CBAM(in_channels=2048, reduction=16, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x


def build_model(num_classes=2, pretrained=False):
    return FluLensModel(num_classes=num_classes, pretrained=pretrained)
