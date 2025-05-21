import torch
import torch.nn as nn
from dfpd_net.abnm import ABNMConv

class DFPD_Net(nn.Module):
    def __init__(self, model, feature_size):
        super(DFPD_Net, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=32, stride=32)
        self.max2 = nn.MaxPool2d(kernel_size=16, stride=16)
        self.max3 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.num_ftrs = 2048 * 1 * 1

        self.S1_G1 = nn.Sequential(
            ABNMConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.S2_G1 = nn.Sequential(
            ABNMConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.S3_G1 = nn.Sequential(
            ABNMConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.S1_G2 = nn.Sequential(
            ABNMConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.S2_G2 = nn.Sequential(
            ABNMConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.S3_G2 = nn.Sequential(
            ABNMConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            ABNMConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, 2),
        )

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, 2),
        )

        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, 2),
        )

    def feature_projection_disentangle(self, f_fea, c_fea, eps=1e-12):
        """
        投影-解耦操作：从 f_fea 中剔除其在 c_fea 方向上的内容特征分量。

        Args:
            f_fea (Tensor): 伪造特征，shape (B, C)
            c_fea (Tensor): 内容特征，shape (B, C)
            eps (float): 防止除零的小常数

        Returns:
            f_proj: f_fea 在 c_fea 上的投影（内容分量）
            f_forgery: 解耦后的伪造特征
        """
        dot_prod = torch.sum(f_fea * c_fea, dim=1, keepdim=True)
        c_norm_sq = torch.sum(c_fea * c_fea, dim=1, keepdim=True).clamp(min=eps)
        f_proj = (dot_prod / c_norm_sq) * c_fea
        f_forgery = f_fea - f_proj

        return f_proj, f_forgery

    def forward(self, x):
        x3, x4, x5, M_hat1, Y_ce1, Y_un1, E1, M_hat2, Y_ce2, Y_un2, E2, M_hat3, Y_ce3, Y_un3, E3 = self.features(x)

        c_xs1 = self.S1_G1(x3)
        c_xs2 = self.S2_G1(x4)
        c_xs3 = self.S3_G1(x5)

        f_xs1 = self.S1_G2(x3)
        f_xs2 = self.S2_G2(x4)
        f_xs3 = self.S3_G2(x5)

        f_fea1 = self.max1(f_xs1).view(f_xs1.size(0), -1)
        c_fea1 = self.max1(c_xs1).view(c_xs1.size(0), -1)
        _, f_dis1 = self.feature_projection_disentangle(f_fea1, c_fea1)
        classifier_out1 = self.classifier1(f_dis1)

        f_fea2 = self.max2(f_xs2).view(f_xs2.size(0), -1)
        c_fea2 = self.max2(c_xs2).view(c_xs2.size(0), -1)
        _, f_dis2 = self.feature_projection_disentangle(f_fea2, c_fea2)
        classifier_out2 = self.classifier2(f_dis2)

        f_fea3 = self.max3(f_xs3).view(f_xs3.size(0), -1)
        c_fea3 = self.max3(c_xs3).view(c_xs3.size(0), -1)
        _, f_dis3 = self.feature_projection_disentangle(f_fea3, c_fea3)
        classifier_out3 = self.classifier3(f_dis3)

        output_dict = {}
        output_dict['pred1'] = classifier_out1
        output_dict['pred2'] = classifier_out2
        output_dict['pred3'] = classifier_out3

        output_dict['c_fea1'] = c_fea1
        output_dict['c_fea2'] = c_fea2
        output_dict['c_fea3'] = c_fea3

        output_dict['f_fea1'] = f_fea1
        output_dict['f_fea2'] = f_fea2
        output_dict['f_fea3'] = f_fea3

        output_dict['f_dis1'] = f_dis1
        output_dict['f_dis2'] = f_dis2
        output_dict['f_dis3'] = f_dis3

        output_dict['M_hat1'] = M_hat1
        output_dict['M_hat2'] = M_hat2
        output_dict['M_hat3'] = M_hat3

        output_dict['Y_ce1'] = Y_ce1
        output_dict['Y_ce2'] = Y_ce2
        output_dict['Y_ce3'] = Y_ce3

        output_dict['Y_un1'] = Y_un1
        output_dict['Y_un2'] = Y_un2
        output_dict['Y_un3'] = Y_un3

        output_dict['E1'] = E1
        output_dict['E2'] = E2
        output_dict['E3'] = E3

        return output_dict
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x