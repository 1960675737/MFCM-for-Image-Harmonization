import torch
from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, SeparableConv2d_X
from iharm.model.ops import MaskedChannelAttention, FeaturesConnector, ChannelAttention, ECA_layer

from iharm.model.sknet import SKUnit, SKUnit_X
from iharm.model.ASFF_module import ASFF

from math import log


class ConvEncoder(nn.Module):
    def __init__(
        self,
        depth, ch,
        norm_layer, batchnorm_from, max_channels,
        backbone_from, backbone_channels=None, backbone_mode='',
        EncoderBlock = ConvBlock
    ):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = EncoderBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = EncoderBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            self.blocks_connected[f'block{block_i}'] = EncoderBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=int(block_i < depth - 1)
            )
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False, use_initial_mask=False, dropout_depth=-1, sk_from=-1, use_SpatialAttention=False):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()
        self.encoder_blocks_channels = encoder_blocks_channels
        self.use_initial_mask = use_initial_mask

        self.sk_blocks = nn.ModuleList()
        self.sk_from = sk_from

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                use_dropout=True if 0 <= dropout_depth <= d else False,
                with_se=0 <= attend_from <= d
            ))

            # self.sk_blocks.append(SKUnit(out_channels, out_channels, 2, 16, 4, norm_layer=norm_layer))
            self.sk_blocks.append(SKUnit(out_channels, out_channels, 4, 16, 16, norm_layer=norm_layer,
                                         use_SpatialAttention=use_SpatialAttention))

            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        i = 0
        for block, sk_block, skip_output in zip(self.deconv_blocks[:-1], self.sk_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            if 0 <= self.sk_from <= i:
                skip_output = sk_block(skip_output, mask)

            output = output + skip_output
            i = i + 1
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:

            if self.use_initial_mask:
                # if use ground truth mask
                attention_map = torch.sigmoid(3.0 * mask)
                output = attention_map * self.to_rgb(output) + image * (1.0 - attention_map)
            else:
                attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
                output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)

        else:
            output = self.to_rgb(output)

        return output



class SEDeconvBlock(nn.Module):
    # def __init__(
    #     self,
    #     in_channels, out_channels,
    #     kernel_size=4, stride=2, padding=1,
    #     norm_layer=nn.BatchNorm2d, activation=nn.ELU,
    #     use_dropout=False,
    #     with_se=False
    # ):
    #     super(SEDeconvBlock, self).__init__()
    #     self.with_se = with_se
    #     self.block = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #         norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
    #         nn.Dropout(0.3) if use_dropout else nn.Dropout(0),
    #         activation(),
    #     )
    #     if self.with_se:
    #         self.se = MaskedChannelAttention(out_channels)

    ############################# use nn.LeakyReLU as activation layer instead of nn.ELU   modified by XCJ  ###################
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True),
        use_dropout=False,
        with_se=False
    ):
        super(SEDeconvBlock, self).__init__()
        self.with_se = with_se
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            nn.Dropout(0.2) if use_dropout else nn.Dropout(0),
            activation,
        )
        if self.with_se:
            self.se = MaskedChannelAttention(out_channels)

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_se:
            out = self.se(out, mask)
        return out



################################## Inception module replace standard convolution by XCJ #############################################

import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, with_ChannelAttention=False, with_ECA=False, dilation=1):
        super(InceptionModule, self).__init__()
        self.with_ChannelAttention = with_ChannelAttention
        self.with_ECA = with_ECA
        self.branch1x1 = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=2, padding=0, norm_layer=norm_layer)

        self.branch3x3_1 = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        self.branch3x3_2 = ConvBlock(int(out_channels/4), int(out_channels/4), kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)

        self.branch5x5_1 = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        if dilation>1:
            self.branch5x5_2 = ConvBlock(int(out_channels/4), int(out_channels/4), kernel_size=3, stride=2, padding=2, norm_layer=norm_layer, dilation=dilation)
        else:
            self.branch5x5_2 = ConvBlock(int(out_channels / 4), int(out_channels / 4), kernel_size=5, stride=2, padding=2, norm_layer=norm_layer)

        self.branch_pool = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)

        if self.with_ChannelAttention:
            self.ChannelAttention = ChannelAttention(out_channels)

        if self.with_ECA:
            self.ECA = ECA_layer(out_channels)                        # self.ECA = ECA_layer(out_channels, adaptiveK=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = (branch1x1, branch3x3, branch5x5, branch_pool)
        outputs = torch.cat(outputs, 1)
        if self.with_ChannelAttention:
            outputs = self.ChannelAttention(outputs)

        if self.with_ECA:
            outputs = self.ECA(outputs)

        return outputs


class InceptionModule_3and5(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, with_ChannelAttention=False, with_ECA=False, dilation=1):
        super(InceptionModule_3and5, self).__init__()
        self.with_ChannelAttention = with_ChannelAttention
        self.with_ECA = with_ECA
        # self.branch1x1 = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=2, padding=0, norm_layer=norm_layer)

        self.branch3x3_1 = ConvBlock(in_channels, int(out_channels/2), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        self.branch3x3_2 = ConvBlock(int(out_channels/2), int(out_channels/2), kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)

        self.branch5x5_1 = ConvBlock(in_channels, int(out_channels/2), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        if dilation>1:
            self.branch5x5_2 = ConvBlock(int(out_channels/2), int(out_channels/2), kernel_size=3, stride=2, padding=2, norm_layer=norm_layer, dilation=dilation)
        else:
            self.branch5x5_2 = ConvBlock(int(out_channels / 2), int(out_channels / 2), kernel_size=5, stride=2, padding=2, norm_layer=norm_layer)

        # self.branch_pool = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)

        if self.with_ChannelAttention:
            self.ChannelAttention = ChannelAttention(out_channels)

        if self.with_ECA:
            self.ECA = ECA_layer(out_channels)                        # self.ECA = ECA_layer(out_channels, adaptiveK=True)

    def forward(self, x):
        # branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        outputs = (branch3x3, branch5x5)
        outputs = torch.cat(outputs, 1)
        if self.with_ChannelAttention:
            outputs = self.ChannelAttention(outputs)

        if self.with_ECA:
            outputs = self.ECA(outputs)

        return outputs

class InceptionModule_nP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, with_ChannelAttention=False, with_ECA=False, dilation=1):
        super(InceptionModule_nP, self).__init__()
        self.with_ChannelAttention = with_ChannelAttention
        self.with_ECA = with_ECA
        self.branch1x1 = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=2, padding=0, norm_layer=norm_layer)

        self.branch3x3_1 = ConvBlock(in_channels, int(out_channels*3/8), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        self.branch3x3_2 = ConvBlock(int(out_channels*3/8), int(out_channels*3/8), kernel_size=3, stride=2, padding=1, norm_layer=norm_layer)

        self.branch5x5_1 = ConvBlock(in_channels, int(out_channels*3/8), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        if dilation>1:
            self.branch5x5_2 = ConvBlock(int(out_channels*3/8), int(out_channels*3/8), kernel_size=3, stride=2, padding=2, norm_layer=norm_layer, dilation=dilation)
        else:
            self.branch5x5_2 = ConvBlock(int(out_channels*3/8), int(out_channels*3/8), kernel_size=5, stride=2, padding=2, norm_layer=norm_layer)

        # self.branch_pool = ConvBlock(in_channels, int(out_channels/4), kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)

        if self.with_ChannelAttention:
            self.ChannelAttention = ChannelAttention(out_channels)

        if self.with_ECA:
            self.ECA = ECA_layer(out_channels)                        # self.ECA = ECA_layer(out_channels, adaptiveK=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        outputs = (branch1x1, branch3x3, branch5x5)
        outputs = torch.cat(outputs, 1)
        if self.with_ChannelAttention:
            outputs = self.ChannelAttention(outputs)

        if self.with_ECA:
            outputs = self.ECA(outputs)

        return outputs

class ConvEncoder_Inception(nn.Module):
    def __init__(
        self,
        depth, ch,
        norm_layer, batchnorm_from, max_channels,
        backbone_from, backbone_channels=None, backbone_mode='',
        with_ChannelAttention=False,
        use_Inception_inLayer2=False,
        with_ECA=False,
        dilation=1,
        no_Pool=False,
    ):
        super(ConvEncoder_Inception, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        if use_Inception_inLayer2:
            self.block1 = InceptionModule(
                out_channels, out_channels,
                norm_layer=norm_layer,
                with_ChannelAttention=with_ChannelAttention,
                with_ECA=with_ECA
            )
        else:
            self.block1 = ConvBlock(out_channels, out_channels,norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)

        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(4 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            # if no_Pool:
            #     self.blocks_connected[f'block{block_i}'] = InceptionModule_nP(
            #         in_channels, out_channels,
            #         norm_layer=norm_layer,
            #         with_ChannelAttention=with_ChannelAttention,
            #         with_ECA=with_ECA,
            #         dilation=dilation
            #     )
            # else:
            #     self.blocks_connected[f'block{block_i}'] = InceptionModule(
            #         in_channels, out_channels,
            #         norm_layer=norm_layer,
            #         with_ChannelAttention=with_ChannelAttention,
            #         with_ECA=with_ECA,
            #         dilation=dilation
            #     )

            self.blocks_connected[f'block{block_i}'] = InceptionModule_3and5(
                in_channels, out_channels,
                norm_layer=norm_layer,
                with_ChannelAttention=with_ChannelAttention,
                with_ECA=with_ECA,
                dilation=dilation
            )
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder_Inception(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False, use_dropout=False, dropout_depth=-1):
        super(DeconvDecoder_Inception, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=1,
                # use_dropout=use_dropout,
                use_dropout=True if 0 <= dropout_depth <= d else False,
                with_se=0 <= attend_from <= d
            ))                                # modify padding argument by XCJ
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output


class ConvEncoder_SKunit(nn.Module):
    def __init__(
        self,
        depth, ch,
        norm_layer, batchnorm_from, max_channels,
        backbone_from, backbone_channels=None, backbone_mode='',
        with_ChannelAttention=False
    ):
        super(ConvEncoder_SKunit, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(4 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            # self.blocks_connected[f'block{block_i}'] = SKUnit(in_channels, out_channels, 4, 16, 16, stride=2, norm_layer=norm_layer)
            self.blocks_connected[f'block{block_i}'] = SKUnit_X(in_channels, out_channels, 3, 8, 16, stride=2,
                                                              norm_layer=norm_layer)
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]

class ConvEncoder_ASFF(nn.Module):
    def __init__(
        self,
        depth, ch,
        norm_layer, batchnorm_from, max_channels,
        backbone_from, backbone_channels=None, backbone_mode='',
        EncoderBlock = ConvBlock
    ):
        super(ConvEncoder_ASFF, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = EncoderBlock(in_channels, out_channels, kernel_size=5, stride=2, padding=2, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = EncoderBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            self.blocks_connected[f'block{block_i}'] = EncoderBlock(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
            )
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder_ASFF(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False, dropout_depth=-1):
        super(DeconvDecoder_ASFF, self).__init__()
        self.image_fusion = image_fusion
        self.ASFF = nn.ModuleList()


        in_channels = encoder_blocks_channels.pop()

        self.deconv_first = SEDeconvBlock_ASFF(
            in_channels, in_channels,
            kernel_size=1, stride=1, padding=0,
            norm_layer=norm_layer
        )

        for d in range(depth-1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.ASFF.append(ASFF(dim=[in_channels, out_channels]))
            in_channels = out_channels

        self.deconv_last = SEDeconvBlock(
            in_channels, in_channels//2,
            kernel_size=4, stride=2, padding=1,
            norm_layer=norm_layer
        )

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels//2, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels//2, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        output = self.deconv_first(output)

        for block, skip_output in zip(self.ASFF, encoder_outputs[1:]):
            output = block(output, skip_output)

        output = self.deconv_last(output)

        if self.image_fusion:

            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)

        else:
            output = self.to_rgb(output)

        return output


class SEDeconvBlock_ASFF(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU
    ):
        super(SEDeconvBlock_ASFF, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    ############################# use nn.LeakyReLU as activation layer instead of nn.ELU   modified by XCJ  ###################
    # def __init__(
    #     self,
    #     in_channels, out_channels,
    #     kernel_size=3, stride=1, padding=1,
    #     norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True)
    # ):
    #     super(SEDeconvBlock_ASFF, self).__init__()
    #     self.with_se = with_se
    #     self.block = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #         norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
    #         activation,
    #     )

    def forward(self, x, mask=None):
        out = self.block(x)
        return out
################################## knowledge distill method by XCJ #############################################

class DeconvDecoder_distill(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False, dropout_depth=None):
        super(DeconvDecoder_distill, self).__init__()
        self.image_fusion = image_fusion
        # self.deconv_blocks = nn.ModuleList()                  # include only deconvolution layers
        self.Decoder_module = nn.Sequential()                 # include all Decoder layers

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.Decoder_module.add_module('Deconv_%d' % (d), SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                use_dropout=True if dropout_depth is not None and d in dropout_depth else False,
                with_se=0 <= attend_from <= d
            ))
            # self.deconv_blocks.append(SEDeconvBlock(
            #     in_channels, out_channels,
            #     norm_layer=norm_layer,
            #     padding=0 if d == 0 else 1,
            #     use_dropout=use_dropout,
            #     with_se=0 <= attend_from <= d
            # ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
            self.Decoder_module.add_module('mask_output', self.conv_attention)

        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)
        self.Decoder_module.add_module('rgb_output', self.to_rgb)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.Decoder_module[:-3], encoder_outputs[1:]):
        # for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.Decoder_module[-3](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output



class PreConvEncoder(nn.Module):                   # Encode Transfer process: a module of transfering real image into composite image
    def __init__(
        self,
        norm_layer
    ):
        super(PreConvEncoder, self).__init__()
        in_channels = 3
        out_channels = 3
        self.block0 = ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.block1 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.block2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.block3 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)
        self.block4 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.network = nn.Sequential(
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4
        )

    def forward(self, x):          # x = real_image + mask
        output = self. network(x)

        return output

