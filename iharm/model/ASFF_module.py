import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def add_conv(in_ch, out_ch, ksize, stride, norm_layer='BN', leaky=False):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    if norm_layer == 'BN':
        Normlayer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        stage.add_module('BN', Normlayer(out_ch))
    elif norm_layer == 'IN':
        Normlayer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        stage.add_module('IN', Normlayer(out_ch))

    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.2, True))
    else:
        stage.add_module('relu6', nn.ELU())
    return stage

class ASFF(nn.Module):
    def __init__(self, level=1, rfb=False, vis=False, dim=None):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = dim                    # [512, 256, 256] # 输入的两个特征层的channels
        self.inter_dim = self.dim[self.level]

        if level==0:                                 # 每个层级输出通道数需要一致
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
        elif level==1:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
        

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*2, 2, kernel_size=1, stride=1, padding=0)
        self.vis= vis

    # 尺度大小 level_0 < level_1 
    def forward(self, x_level_0, x_level_1):

        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)             # 改变通道数，保证相互融合的特征的通道数相同
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')        # 最近邻插值
            level_1_resized =x_level_1


        level_0_weight_v = self.weight_level_0(level_0_resized)          # 通道数压缩
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)   # alpha等产生

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]

        out = fused_out_reduced
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

