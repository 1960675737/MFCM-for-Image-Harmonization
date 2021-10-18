from iharm.model.base import DeepImageHarmonization, DeepImageHarmonization_DS, SSAMImageHarmonization, ISEUNetV1

###############################################################################
# Helper Functions
###############################################################################
from functools import partial
import torch.nn as nn

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='batch'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


BMCONFIGS = {
    'dih256': {
        'model': DeepImageHarmonization,
        'params': {'depth': 7}
    },
    # 'improved_dih256': {
    #     'model': DeepImageHarmonization,
    #     'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True}
    # },
    ######################### BN->IN modified by XCJ #################################

    # 'improved_dih256': {
    #     'model': DeepImageHarmonization_DS,
    #     'params': {'depth': 7, 'norm_layer':get_norm_layer(norm_type='instance'), 'batchnorm_from': 2, 'image_fusion': True}
    # },

    'improved_dih256': {
        'model': DeepImageHarmonization,
        'params': {'depth': 7, 'norm_layer':get_norm_layer(norm_type='instance'),
                   'batchnorm_from': 2, 'image_fusion': True, 'with_ChannelAttention': False,
                   'with_ECA': True, "use_Inception_inLayer2": False, 'dilation': 1, 'no_Pool': False}
    },

    # 'improved_dih256': {
    #     'model': DeepImageHarmonization,
    #     'params': {'depth': 7, 'norm_layer':get_norm_layer(norm_type='instance'), 'batchnorm_from': 2, 'image_fusion': True, 'sk_from': 0, 'use_SpatialAttention': True}
    # },

    # 'improved_dih256': {
    #     'model': DeepImageHarmonization,
    #     'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True, 'sk_from': 4}
    # },

    'improved_dih256_S': {
        'model': DeepImageHarmonization_DS,
        'params': {'depth': 7, 'norm_layer':get_norm_layer(norm_type='instance'), 'batchnorm_from': 2, 'image_fusion': True, "vid_info": ['Deconv_6:None']}
    },
    ##########################################################################
    'improved_sedih256': {
        'model': DeepImageHarmonization,
        'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True, 'attend_from': 5}
    },
    'ssam256': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 4, 'batchnorm_from': 2, 'attend_from': 2}
    },
    'improved_ssam256': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2}
    },
    'iseunetv1_256': {
        'model': ISEUNetV1,
        'params': {'depth': 4, 'batchnorm_from': 2, 'attend_from': 1, 'ch': 64}
    },
    'dih512': {
        'model': DeepImageHarmonization,
        'params': {'depth': 8}
    },
    'improved_dih512': {
        'model': DeepImageHarmonization,
        'params': {'depth': 8, 'batchnorm_from': 2, 'image_fusion': True}
    },
    'improved_ssam512': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 6, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 3}
    },
    'improved_sedih512': {
        'model': DeepImageHarmonization,
        'params': {'depth': 8, 'batchnorm_from': 2, 'image_fusion': True, 'attend_from': 6}
    },
}
