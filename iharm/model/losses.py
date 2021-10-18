import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from iharm.utils import misc
import math
from math import exp
import numpy as np



class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss


class MaskWeightedMSE(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedMSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)

        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss


##################################### added by XCJ ##########################################

class VIDLoss(nn.Module):
    def __init__(self, pdf='gaussian', epsilon=1e-8):
        super(VIDLoss, self).__init__()
        self.epsilon = epsilon
        self.pdf = pdf


    def forward(self, student_pred_dict, teacher_pred_dict):

        distill_loss = 0

        for k, v in student_pred_dict.items():
            if 'mean' in k:
                layer_name = k.split('_mean')[0]
                tl = teacher_pred_dict[layer_name]
                mu = student_pred_dict['%s_mean'%layer_name]
                std = student_pred_dict['%s_var'%layer_name]
                distill_loss += self.vid_loss_fn(mu, std, tl)

        return distill_loss


    def vid_loss_fn(self, mu, std, tl):
        if self.pdf == 'laplace':
            std = std * 0.1 + self.epsilon
            numerator = torch.abs(mu - tl)
            loss = mu.shape[1] * np.log(2 * math.pi) / 2 + torch.log(2 * std) + numerator / (std)
        elif self.pdf == 'gaussian':
            std = std * 0.001 + self.epsilon
            numerator = (mu - tl) ** 2
            loss = mu.shape[1] * np.log(2 * math.pi) / 2 + torch.log(std) / 2 + numerator / (2 * std)

        loss = loss.mean()
        return loss


class SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window        # shape = [3, 1, 11, 11]
            self.channel = channel

        SSIM = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        return 1 - SSIM

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window                  # shape = [3, 1, 11, 11]

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)  # mean
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # Variance
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # Covariance

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))    # shape = [batch size, 3, 256, 256]

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

class SSIM_focal_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_focal_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window        # shape = [3, 1, 11, 11]
            self.channel = channel

        SSIM = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        return SSIM

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window                  # shape = [3, 1, 11, 11]

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)  # mean
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # Variance
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # Covariance

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))    # shape = [batch size, 3, 256, 256]
        ssim_map_clamp = torch.clamp(ssim_map, 0.90001, 1.0)          # Limit element values to a fixed range [0.9, 1]     0.9 -> 0.85       # torch.clamp(ssim_map, 0.85001, 1.0)
        ssim_map_weighted = (1. - (ssim_map_clamp - 0.9)).pow(50)       # focal loss weights                                      # (1. - (ssim_map_clamp - 0.85)).pow(50)
        # ssim_map_focal = - ssim_map_weighted.mul(torch.log(ssim_map_clamp - 0.9))                                                 # ssim_map_clamp - 0.85
        ssim_map_focal = ssim_map_weighted.mul(1 - ssim_map_clamp)

        if size_average:
            # return ssim_map.mean()
            return ssim_map_focal.mean()
        else:
            # return ssim_map.mean(1).mean(1).mean(1)
            return ssim_map_focal.mean(1).mean(1).mean(1)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

class V_MSE_Loss(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(V_MSE_Loss, self).__init__(pred_outputs=(pred_name,),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        pred, label = self.rgb2v(pred, label)
        pred = torch.unsqueeze(pred, 1)
        label = torch.unsqueeze(label, 1)
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)

        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss

    def rgb2v(self, pred, label):
        pred = torch.max(pred, 1)[0]
        label = torch.max(label, 1)[0]
        return pred, label


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数, 100.0, 1000.0, 10000.0
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=2为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        # self.weight_list=self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self):
        self.weight_list = self.get_weight(self.model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = 0.5 * weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


class VGGLoss(nn.Module):             # 返回计算出的VGG感知损失值
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        # self.vgg = Vgg19().cuda()
        self.vgg = Vgg19().to(device)                # modified by XCJ
        self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):                     # VGG模型直接加载库里的模型，然后返回需要的5个特征层输出结果
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

######################################################################################
