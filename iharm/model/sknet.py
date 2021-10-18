import torch
from torch import nn
from iharm.model.modeling.basic_blocks import GaussianSmoothing
from iharm.model.SpatialAttention import SpatialGate


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1 ,L=32, norm_layer=nn.BatchNorm2d):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        # for i in range(M):                      # kernel size begin with 3 x 3
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
        #         norm_layer(features),
        #         nn.ReLU(inplace=False)
        #         # nn.LeakyReLU(0.2, False)
        #     ))

        for i in range(M):                       # kernel size begin with 1 x 1
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=1+i*2, stride=stride, padding=i, groups=G),
                norm_layer(features),
                # nn.ReLU(inplace=False)
                nn.LeakyReLU(0.2, False)
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, mid_features=None, stride=1, L=32, norm_layer=nn.BatchNorm2d, use_SpatialAttention=False):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            norm_layer(mid_features),
            SKConv(mid_features, M, G, r, stride=stride, L=L, norm_layer=norm_layer),
            norm_layer(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            norm_layer(out_features),
            # nn.ReLU(inplace=False)
            nn.LeakyReLU(0.2, False)
        )
        self.SpatialAttention = SpatialGate() if use_SpatialAttention else None
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)
    
    def forward(self, x, mask=None):
        if mask is not None:
            mask = self.mask_blurring(nn.functional.interpolate(
                mask, size=x.size()[-2:],
                mode='bilinear', align_corners=True
        ))
        fea = self.feas(x)

        if self.SpatialAttention is not None:
            fea = self.SpatialAttention(fea)

        if mask is not None:
            output = mask * fea + (1 - mask) * x
        else:
            output = fea

        return output                                            # by xie: needn't shortcut


##########################  modify by xie to be close to Inception module ############################

class SKConv_X(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, stride=1, L=32, norm_layer=nn.BatchNorm2d):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_X, self).__init__()
        d = max(int(in_features / r), L)
        self.M = M
        # self.features = in_features
        self.convs = nn.ModuleList([])
        # for i in range(M):                      # kernel size begin with 3 x 3
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
        #         norm_layer(features),
        #         nn.ReLU(inplace=False)
        #         # nn.LeakyReLU(0.2, False)
        #     ))

        for i in range(M):  # kernel size begin with 1 x 1
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1 + i * 2, stride=stride, padding=i, groups=G),
                norm_layer(out_features),
                # nn.ReLU(inplace=False)
                nn.LeakyReLU(0.2, False)
            ))
        self.fc = nn.Linear(out_features, d)
        #     nn.Sequential(
        #     nn.Linear(out_features, d),
        #     norm_layer(d),
        #     nn.LeakyReLU(0.2, False)
        # )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, out_features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit_X(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, mid_features=None, stride=1, L=32, norm_layer=nn.BatchNorm2d,
                 use_SpatialAttention=False):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit_X, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 4)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            norm_layer(mid_features),
            nn.LeakyReLU(0.2, False),
            SKConv_X(mid_features, out_features, M, G, r, stride=stride, L=L, norm_layer=norm_layer)
        )
        self.SpatialAttention = SpatialGate() if use_SpatialAttention else None
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = self.mask_blurring(nn.functional.interpolate(
                mask, size=x.size()[-2:],
                mode='bilinear', align_corners=True
            ))
        fea = self.feas(x)

        if self.SpatialAttention is not None:
            fea = self.SpatialAttention(fea)

        if mask is not None:
            output = mask * fea + (1 - mask) * x
        else:
            output = fea

        return output  # by xie: needn't shortcut



