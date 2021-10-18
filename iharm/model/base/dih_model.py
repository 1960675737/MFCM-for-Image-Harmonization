import torch
import torch.nn as nn
from collections import OrderedDict

from iharm.model.modeling.conv_autoencoder import PreConvEncoder, ConvEncoder, ConvEncoder_Inception, ConvEncoder_SKunit, ConvEncoder_ASFF, DeconvDecoder_ASFF, DeconvDecoder, DeconvDecoder_Inception, DeconvDecoder_distill
from iharm.model.modeling.vid_module import get_vid_module_dict

class DeepImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode='',
        use_dropout=False,
        use_initial_mask=False,
        dropout_depth=-1,
        sk_from=-1,
        use_SpatialAttention=False,
        with_ChannelAttention=False,
        use_Inception_inLayer2=False,
        with_ECA=False,
        use_ASFF=False,
        dilation=1,
        no_Pool=False,
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth

        # if use_ASFF:
        #     self.encoder = ConvEncoder_ASFF(
        #         depth, ch,
        #         norm_layer, batchnorm_from, max_channels,
        #         backbone_from, backbone_channels, backbone_mode
        #     )
        #     self.decoder = DeconvDecoder_ASFF(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)
        # else:
        #     self.encoder = ConvEncoder(
        #         depth, ch,
        #         norm_layer, batchnorm_from, max_channels,
        #         backbone_from, backbone_channels, backbone_mode
        #     )
        #     self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion, use_initial_mask,
        #                              dropout_depth=dropout_depth, sk_from=sk_from, use_SpatialAttention=use_SpatialAttention)


        self.encoder = ConvEncoder_Inception(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode,
            with_ChannelAttention=with_ChannelAttention,
            use_Inception_inLayer2=use_Inception_inLayer2,
            with_ECA=with_ECA,
            dilation=dilation,
            no_Pool=no_Pool,
        )

        # self.encoder = ConvEncoder_SKunit(
        #     depth, ch,
        #     norm_layer, batchnorm_from, max_channels,
        #     backbone_from, backbone_channels, backbone_mode,
        # )

        self.decoder = DeconvDecoder_Inception(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion, use_dropout=use_dropout, dropout_depth=dropout_depth)



    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {'images': output}


class DeepImageHarmonization_DS(nn.Module):             # Knowledge Distill network of Student
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        vid_info=None,
        dropout_depth=None,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonization_DS, self).__init__()
        self.depth = depth

        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoder_distill(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion, dropout_depth=dropout_depth)

        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()


    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)

        decoder_dict = dict()                # storage decoder feature layer results --only include deconvolution layer
        distill_dict = dict()

        decoder_modules = self.decoder.Decoder_module._modules
        decoder_layer_names = decoder_modules.keys()

        output = intermediates[0]                                                        # encoder output

        for layer_name, skip_output in zip(decoder_layer_names, intermediates[1:]):
            output = decoder_modules[layer_name](output, mask)
            # decoder_dict[layer_name] = output
            if layer_name in self.distill_layers:

                distill_dict[layer_name] = output

                mean = self.vid_module_dict._modules[layer_name+'_mean'](output)
                var = self.vid_module_dict._modules[layer_name+'_var'](output)
                distill_dict[layer_name+'_mean'] = mean
                distill_dict[layer_name+'_var'] = var
            output = output + skip_output


        # last deconvolution layer
        layer_name = 'Deconv_%d'%(self.depth-1)
        output = decoder_modules[layer_name](output, mask)
        # decoder_dict[layer_name] = output
        if layer_name in self.distill_layers:

            distill_dict[layer_name] = output

            mean = self.vid_module_dict._modules[layer_name + '_mean'](output)
            var = self.vid_module_dict._modules[layer_name + '_var'](output)
            distill_dict[layer_name + '_mean'] = mean
            distill_dict[layer_name + '_var'] = var


        if self.decoder.image_fusion:
            attention_map = torch.sigmoid(3.0 * decoder_modules['mask_output'](output))
            output = attention_map * image + (1.0 - attention_map) * decoder_modules['rgb_output'](output)
        else:
            output = decoder_modules['rgb_output'](output)

        return {'images': output, 'distill_results': distill_dict}

    def get_vid_module_dict(self):
        self.distill_layers = []
        self.homoscedasticties = []
        for s in self.vid_info:
            layer, homoscedasticity = s.split(':')
            self.distill_layers.append(layer)
            self.homoscedasticties.append(homoscedasticity)

        vid_module_dict = get_vid_module_dict(self.decoder.Decoder_module,
                                              self.distill_layers, self.homoscedasticties)
        vid_module_dict = nn.Sequential(
            OrderedDict([(k, v) for k,v in vid_module_dict.items()])
        )
        return vid_module_dict


class DeepImageHarmonization_DT(nn.Module):  # Knowledge Distill network of Teacher
    def __init__(
            self,
            depth,
            norm_layer=nn.BatchNorm2d, batchnorm_from=0,
            attend_from=-1,
            image_fusion=False,
            ch=64, max_channels=512,
            vid_info=None,
            backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonization_DT, self).__init__()
        self.depth = depth

        self.preEncoder = PreConvEncoder(norm_layer)
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = DeconvDecoder_distill(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

    def forward(self, image, mask, backbone_features=None):       # image is real image
        # x = torch.cat((image, mask), dim=1)
        x = image
        mid_result = self.preEncoder(x)                           # predict composite image
        intermediates = self.encoder(torch.cat((mid_result, mask), dim=1), backbone_features)

        decoder_dict = dict()  # storage decoder feature layer results --only include deconvolution layer
        distill_dict = dict()

        decoder_modules = self.decoder.Decoder_module._modules
        decoder_layer_names = decoder_modules.keys()

        output = intermediates[0]  # encoder output

        for layer_name, skip_output in zip(decoder_layer_names, intermediates[1:]):
            output = decoder_modules[layer_name](output, mask)
            # decoder_dict[layer_name] = output
            if layer_name in self.distill_layers:

                distill_dict[layer_name] = output

                mean = self.vid_module_dict._modules[layer_name + '_mean'](output)
                var = self.vid_module_dict._modules[layer_name + '_var'](output)
                distill_dict[layer_name + '_mean'] = mean
                distill_dict[layer_name + '_var'] = var
            output = output + skip_output

        # last deconvolution layer
        layer_name = 'Deconv_%d' % (self.depth - 1)
        output = decoder_modules[layer_name](output, mask)
        # decoder_dict[layer_name] = output
        if layer_name in self.distill_layers:

            distill_dict[layer_name] = output

            mean = self.vid_module_dict._modules[layer_name + '_mean'](output)
            var = self.vid_module_dict._modules[layer_name + '_var'](output)
            distill_dict[layer_name + '_mean'] = mean
            distill_dict[layer_name + '_var'] = var

        if self.decoder.image_fusion:
            attention_map = torch.sigmoid(3.0 * decoder_modules['mask_output'](output))
            output = attention_map * image + (1.0 - attention_map) * decoder_modules['rgb_output'](output)
        else:
            output = decoder_modules['rgb_output'](output)

        return {'mid-images': mid_result,
                'images': output,
                'distill_results': distill_dict
                }

    def get_vid_module_dict(self):
        self.distill_layers = []
        self.homoscedasticties = []
        for s in self.vid_info:
            layer, homoscedasticity = s.split(':')
            self.distill_layers.append(layer)
            self.homoscedasticties.append(homoscedasticity)

        vid_module_dict = get_vid_module_dict(self.decoder.Decoder_module,
                                              self.distill_layers, self.homoscedasticties)
        vid_module_dict = nn.Sequential(
            OrderedDict([(k, v) for k, v in vid_module_dict.items()])
        )
        return vid_module_dict

