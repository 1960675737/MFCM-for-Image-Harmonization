import argparse
import importlib.util

import torch
from iharm.utils.exp import init_experiment

# models/fixed256/improved_dih.py --gpus 1,2 --exp-name iDIH --batch-size 128 --workers 8 --use_vgg_loss
# models/fixed256/improved_dih.py --gpus 1,2,3 --exp-name iDIH_IN_Inception --batch-size 160 --workers 8 --norm instance --resume-exp 004_iDIH_IN_Inception --start-epoch 59
# models/fixed256/improved_dih.py --gpus 0,1 --exp-name iDIH_IN_sub_distill_T --batch-size 96 --workers 8 --norm instance --model_type teacher
# models/fixed256/improved_dih.py --gpus 0,1 --exp-name iDIH_IN_sub_distill_S --batch-size 48 --workers 8 --norm instance --model_type student --weights harmonization_exps/fixed256/improved_dih/005_iDIH_IN_sub_distill_T/checkpoints/last_checkpoint.pth --teacher_weights harmonization_exps/fixed256/improved_dih/005_iDIH_IN_sub_distill_T/checkpoints/last_checkpoint.pth
# models/fixed256/improved_dih.py --gpus 1,2,3 --exp-name iDIH_IN_distill_S --batch-size 48 --workers 16--norm instance --model_type student  --teacher_weights harmonization_exps/fixed256/improved_dih/008_iDIH_IN_distill_T/checkpoints/last_checkpoint.pth --resume-exp 009_iDIH_IN_distill_S --start-epoch 167 --num_epochs 250
# models/fixed256/improved_dih.py --gpus 1,0,2,3 --exp-name iDIH_IN_distill_S_dropout --batch-size 96 --workers 12 --norm instance --model_type student  --teacher_weights harmonization_exps/fixed256/improved_dih/008_iDIH_IN_distill_T/checkpoints/last_checkpoint.pth --resume-exp 014_iDIH_IN_distill_S_dropout --num_epochs 200 --lr 2e-3 --use_dropout --start-epoch 61 --resume-prefix 060
# models/fixed256/improved_dih.py --gpus 0,1,2,3 --exp-name iDIH_sub_brightness --batch-size 192 --workers 12 --num_epochs 200 --lr 2e-3 --use_brightness_loss
# models/fixed256/improved_dih.py --gpus 0,1,2,3 --exp-name iDIH_ssim_focal_loss --batch-size 128 --workers 8 --use_ssim_focal_loss
# models/fixed256/improved_dih.py --gpus 2,1,3 --exp-name iDIH_SK_SA --batch-size 96 --norm instance --workers 6 --sk_from 0 --use_SpatialAttention
# models/fixed256/improved_dih.py --gpus 1,0,2,3 --exp-name iDIH_IN_SKunit --batch-size 128 --workers 6 --norm instance --use_dropout
# models/fixed256/improved_dih.py --gpus 1,0,2,3 --exp-name iDIH_IN_SKunit --batch-size 128 --workers 6 --norm instance --dropout_depth 4
# models/fixed256/improved_dih.py --gpus 0,1,2,3 --exp-name iDIH_IN_Inception_dropout_CA --batch-size 128 --workers 6 --norm instance --use_dropout --with_ChannelAttention
# models/fixed256/improved_dih.py --gpus 0,1,2,3 --exp-name iDIH_IN_Inception_dropout_ECA --batch-size 128 --workers 6 --norm instance --use_dropout --with_ECA
# train.py models/fixed256/improved_dih.py --gpus 1,0,2,3 --exp-name iDIH_IN_SKunitX_lr --batch-size 96 --workers 6 --norm instance --lr 3e-3
# models/fixed256/improved_dih.py --gpus 1,0,2,3 --exp-name iDIH_IN_Inception_alldropout_batchsize32 --batch-size 96 --workers 4 --norm instance --dropout_depth 0 --resume-exp iDIH_IN_Inception_alldropout_batchsize32 --start-epoch 107
# models/fixed256/improved_dih.py --gpus 0 --exp-name iDIH_IN_Inception_FCECA_batchsize32 --batch-size 32 --workers 4 --norm instance --with_ChannelAttention
# models/fixed256/improved_dih.py --gpus 0 --exp-name iDIH_IN_Inception_dilation_ECA_batchsize32 --batch-size 96 --workers 4 --norm instance --with_ECA --dilation 2
# models/fixed256/improved_dih.py --gpus 0 --exp-name iDIH_IN_Inception_nP_ECA_batchsize32 --batch-size 32 --workers 4 --norm instance --with_ECA --no_Pool

def main():
    args = parse_args()
    model_script = load_module(args.model_path)

    cfg = init_experiment(args)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--num_epochs', type=int, default=120,
                        help='The number of training epochs.')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='The parameter of weight decay in optimizer. | 0.01')

    parser.add_argument('--reg_weight_decay', type=float, default=0,
                        help='The parameter of weight decay in Regularization class. | 1000.0')

    parser.add_argument('--ngpus', type=int, default=2,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='last',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--use_vgg_loss', action='store_true', help='if use vgg loss.')

    parser.add_argument('--use_ssim_loss', action='store_true', help='if use ssim loss.')

    parser.add_argument('--use_ssim_focal_loss', action='store_true', help='if use ssim loss with focal loss weighted .')

    parser.add_argument('--use_brightness_loss', action='store_true', help='if use brightness loss.')

    parser.add_argument('--use_initial_mask', action='store_true', help='if use ground truth mask.')

    parser.add_argument('--use_dropout', action='store_true', help='if use dropout in decoder after using  Inception module replace common convolution.')

    parser.add_argument('--use_Inception_inLayer2', action='store_true',
                        help='if use ssim loss with focal loss weighted .')

    parser.add_argument('--dropout_depth', type=int, default=-1, required=False,
                        help='The dropout network layer number in decoder model.')

    parser.add_argument('--sk_from', type=int, default=-1,
                        help='the beginning layer to apply Sk unit. 0|4|')

    parser.add_argument('--use_SpatialAttention', action='store_true', help='if use SpatialAttention after SK unit.')

    parser.add_argument('--with_ChannelAttention', action='store_true', help='if use ChannelAttention after Inception module.')

    parser.add_argument('--with_ECA', action='store_true',
                        help='if use EfficientChannelAttention after Inception module.')

    parser.add_argument('--no_Pool', action='store_true',
                        help='if do not use Avg Pool branch in Inception Module.')

    parser.add_argument('--norm', type=str, default='batch',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--dilation', type=int, default=1,
                        help='dilation rate.')

    parser.add_argument('--model_type', type=str, default=None,
                        help='teacher net or student net of knowledge distill method. [teacher | student | None]')

    parser.add_argument('--teacher_weights', type=str, default=None,
                        help='Teacher model weights will be loaded from the specified path if you use this argument.')


    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()
