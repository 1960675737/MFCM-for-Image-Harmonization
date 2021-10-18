from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.model import initializer
from iharm.model.base import DeepImageHarmonization, DeepImageHarmonization_DS, DeepImageHarmonization_DT
# from iharm.model.losses import MaskWeightedMSE
from iharm.model import losses
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.utils.log import logger


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
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

def main(cfg):
    if cfg.model_type == 'student':
        teacher_model, model, model_cfg = init_model(cfg)
        train(model, cfg, model_cfg, teacher_model=teacher_model, start_epoch=cfg.start_epoch)
    else:
        model, model_cfg = init_model(cfg)
        train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    norm_layer = get_norm_layer(norm_type=cfg.norm)

    # model = DeepImageHarmonization(depth=7, batchnorm_from=2, image_fusion=True)                                    # initial code
    if cfg.model_type == 'teacher':
        model = DeepImageHarmonization_DT(depth=7, norm_layer=norm_layer, batchnorm_from=2, image_fusion=True)
    elif cfg.model_type == 'student':
        teacher_model = DeepImageHarmonization_DT(depth=7, norm_layer=norm_layer, batchnorm_from=2, image_fusion=True, vid_info=['Deconv_6:None'])
        teacher_model.to(cfg.device)

        model = DeepImageHarmonization_DS(depth=7, norm_layer=norm_layer, batchnorm_from=2, image_fusion=True, vid_info=['Deconv_6:None'], dropout_depth=cfg.dropout_depth)
    else:
        model = DeepImageHarmonization(depth=7, norm_layer=norm_layer, batchnorm_from=2,
                                       image_fusion=True, use_dropout=cfg.use_dropout,
                                       use_initial_mask=cfg.use_initial_mask,
                                       dropout_depth=cfg.dropout_depth,
                                       sk_from=cfg.sk_from,
                                       use_SpatialAttention=cfg.use_SpatialAttention,
                                       with_ChannelAttention=cfg.with_ChannelAttention,
                                       use_Inception_inLayer2=cfg.use_Inception_inLayer2,
                                       with_ECA=cfg.with_ECA, use_ASFF=True,
                                       dilation=cfg.dilation,
                                       no_Pool=cfg.no_Pool)             # BN->IN


    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))

    if cfg.model_type == 'student':
        return teacher_model, model, model_cfg

    return model, model_cfg


def train(model, cfg, model_cfg, teacher_model=None, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    cfg.input_normalization = model_cfg.input_normalization

    loss_cfg = edict()
    loss_cfg.pixel_loss = losses.MaskWeightedMSE(min_area=100)
    loss_cfg.pixel_loss_weight = 1.0

    if cfg.use_brightness_loss:
        loss_cfg.brightness_loss = losses.V_MSE_Loss()
        loss_cfg.brightness_loss_weight = 0.5                      # the initial weight setting is 0.1 (iDIH_brightness, iDIH_brightness_dropout56)   0.5 (iDIH_brightness_dropout_1_6)

    if cfg.use_ssim_loss:
        loss_cfg.ssim_loss = losses.SSIM_Loss()
        loss_cfg.ssim_loss_weight = 0.025

    if cfg.use_ssim_focal_loss:
        loss_cfg.ssim_focal_loss = losses.SSIM_focal_Loss()
        loss_cfg.ssim_focal_loss_weight = 0.1

    if cfg.model_type == 'teacher':                                           # imitation loss
        loss_cfg.imitation_loss = losses.MSE()
        loss_cfg.imitation_loss_weight = 0.1

    elif cfg.model_type == 'student':                                           # distill loss
        loss_cfg.distill_loss = losses.VIDLoss()
        loss_cfg.distill_loss_weight = 0.001                   # 0.0001

    ###############################  added by XCJ #############################
    if cfg.use_vgg_loss:
        loss_cfg.vgg_loss = losses.VGGLoss(cfg.device)
        loss_cfg.vgg_loss_weight = 0.1
    #############################################################################


    num_epochs = cfg.num_epochs
    train_augmentator = HCompose([
        RandomResizedCrop(256, 256, scale=(0.5, 1.0)),
        HorizontalFlip()
    ])

    val_augmentator = HCompose([
        Resize(256, 256)
    ])

    trainset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='train'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='train'),
            HDataset(cfg.HCOCO_PATH, split='train'),
            HDataset(cfg.HADOBE5K_PATH, split='train'),
        ],
        augmentator=train_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05
    )

    valset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='test'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='test'),
            HDataset(cfg.HCOCO_PATH, split='test'),
            # HDataset(cfg.HADOBE5K_PATH, split='test'),
        ],
        augmentator=val_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1
    )

    # optimizer_params = {
    #     'lr': 1e-3,
    #     'betas': (0.9, 0.999), 'eps': 1e-8
    # }

    optimizer_params = {
        'lr': cfg.lr,
        'betas': (0.9, 0.999), 'eps': 1e-8,
        'weight_decay': cfg.weight_decay
    }

    # lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
    #                        milestones=[90,120,160,180], gamma=0.5)          # allset: rugulariation  milestones=[60,100,120,140,150], gamma=0.5        dropout choice: milestones=[90,120,160,180], gamma=0.5              dropout 1-5: milestones=[90,120,160,180], gamma=0.5

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[105, 115], gamma=0.1)                      # subset: brightness [160, 185]         allset: brightness [85, 120]   allset: brightness_dropout56 [120,150]

    if cfg.model_type == 'student':
        trainer = SimpleHTrainer(
            model, cfg, model_cfg, loss_cfg,
            trainset, valset,
            teacher_model=teacher_model,
            optimizer='adam',
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            metrics=[
                DenormalizedPSNRMetric(
                    'images', 'target_images',
                    mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                    std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                ),
                DenormalizedMSEMetric(
                    'images', 'target_images',
                    mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                    std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                )
            ],
            checkpoint_interval=10,
            image_dump_interval=1000
        )
    else:
        trainer = SimpleHTrainer(
            model, cfg, model_cfg, loss_cfg,
            trainset, valset,
            optimizer='adam',
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            metrics=[
                DenormalizedPSNRMetric(
                    'images', 'target_images',
                    mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                    std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                ),
                DenormalizedMSEMetric(
                    'images', 'target_images',
                    mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                    std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                )
            ],
            checkpoint_interval=10,
            image_dump_interval=1000
        )


    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


