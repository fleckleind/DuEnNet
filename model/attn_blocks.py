# --------------------------------------------------------
# Attention Blocks:
# SEBlock: squeeze-and-excitation block
# CBAModule: convolution based attention module
# --------------------------------------------------------
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    # squeeze-and-excitation block
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # SE-Block
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChanAttn(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChanAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return avg_out + max_out


class SpatAttn(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super(SpatAttn, self).__init__()
        assert kernel_size in (3, 7), 'kernel size should be either 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        nax_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, nax_out], dim=1)
        return self.conv(x)


class CBAModule(nn.Module):
    def __init__(self, channel, ratio=16, in_ch=2, out_ch=1, kernel_size=7):
        super(CBAModule, self).__init__()
        self.chanAttn = ChanAttn(channel=channel, ratio=ratio)
        self.spatAttn = SpatAttn(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.chanAttn(x)
        output = out * self.spatAttn(out)
        return output
