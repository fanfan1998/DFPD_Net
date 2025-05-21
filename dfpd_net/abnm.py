import torch
import torch.nn as nn
import torch.nn.functional as F

class ABNM(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(ABNM, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.inorm = nn.InstanceNorm2d(num_channels, affine=True)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1),
            nn.ReLU(inplace=True)
        )

        self.attn_bn = nn.Conv2d(num_channels // reduction, num_channels, 1)
        self.attn_in = nn.Conv2d(num_channels // reduction, num_channels, 1)

    def forward(self, x):
        x_bn = self.bn(x)
        x_in = self.inorm(x)

        s = self.fc(x)

        b = torch.sigmoid(self.attn_bn(s))
        i = torch.sigmoid(self.attn_in(s))

        alpha_bn = b / (b + i + 1e-6)
        alpha_in = i / (b + i + 1e-6)

        out = alpha_bn * x_bn + alpha_in * x_in
        return out

class ABNMConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, abnm=True, bias=True):
        super(ABNMConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.abnm = ABNM(out_planes) if abnm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.abnm is not None:
            x = self.abnm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
