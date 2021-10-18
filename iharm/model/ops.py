import torch
from torch import nn as nn
from math import log


class SimpleInputFusion(nn.Module):
    def __init__(self, add_ch=1, rgb_ch=3, ch=8, norm_layer=nn.BatchNorm2d):
        super(SimpleInputFusion, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=add_ch + rgb_ch, out_channels=ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            norm_layer(ch),
            nn.Conv2d(in_channels=ch, out_channels=rgb_ch, kernel_size=1),
        )

    def forward(self, image, additional_input):
        return self.fusion_conv(torch.cat((image, additional_input), dim=1))

class SEChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SEChannelAttention, self).__init__()
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(len(self.global_pools) * in_channels, intermediate_channels_count),
            nn.ReLU(),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled_x = []
        for global_pool in self.global_pools:
            pooled_x.append(global_pool(x))
        pooled_x = torch.cat(pooled_x, dim=1).flatten(start_dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention_transform = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled_x = self.avg_pool(x)
        pooled_x = pooled_x.flatten(start_dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x


class ECA_layer(nn.Module):                       # 默认1D卷积核大小为3
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, k_size=9, adaptiveK=False, gamma=2, b=1):
        super(ECA_layer, self).__init__()

        if adaptiveK:
            t = int(abs((log(channels, 2) + b) / gamma))
            k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)                                     # [batch_size, channels, 1, 1]

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))          # y.squeeze(-1): [batch_size, channels, 1]    y.squeeze(-1).transpose(-1, -2): [batch_size, 1, channels]   result y: [batch_size, 1, channels]

        y = y.transpose(-1, -2).unsqueeze(-1)                   # y.transpose(-1, -2): [batch_size, channels, 1]   y.transpose(-1, -2).unsqueeze(-1): [batch_size, channels, 1, 1]

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)                               # y.expand_as(x): [batch_size, channels, H, W]

class MaskedChannelAttention(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(MaskedChannelAttention, self).__init__()
        self.global_max_pool = MaskedGlobalMaxPool2d()
        self.global_avg_pool = FastGlobalAvgPool2d()

        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(3 * in_channels, intermediate_channels_count),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        if mask.shape[2:] != x.shape[:2]:
            mask = nn.functional.interpolate(
                mask, size=x.size()[-2:],
                mode='bilinear', align_corners=True
            )
        pooled_x = torch.cat([
            self.global_max_pool(x, mask),
            self.global_avg_pool(x)
        ], dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]

        return channel_attention_weights * x


class MaskedGlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_max_pool = FastGlobalMaxPool2d()

    def forward(self, x, mask):
        return torch.cat((
            self.global_max_pool(x * mask),
            self.global_max_pool(x * (1.0 - mask))
        ), dim=1)


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(FastGlobalAvgPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).mean(dim=2)


class FastGlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(FastGlobalMaxPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).max(dim=2)[0]


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class FeaturesConnector(nn.Module):
    def __init__(self, mode, in_channels, feature_channels, out_channels):
        super(FeaturesConnector, self).__init__()
        self.mode = mode if feature_channels else ''

        if self.mode == 'catc':
            self.reduce_conv = nn.Conv2d(in_channels + feature_channels, out_channels, kernel_size=1)
        elif self.mode == 'sum':
            self.reduce_conv = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

        self.output_channels = out_channels if self.mode != 'cat' else in_channels + feature_channels

    def forward(self, x, features):
        if self.mode == 'cat':
            return torch.cat((x, features), 1)
        if self.mode == 'catc':
            return self.reduce_conv(torch.cat((x, features), 1))
        if self.mode == 'sum':
            return self.reduce_conv(features) + x
        return x

    def extra_repr(self):
        return self.mode
