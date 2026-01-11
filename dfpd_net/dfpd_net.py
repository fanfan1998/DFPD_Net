import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== Loss Functions ====================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def mask_loss(M_hat, M):
    """
    Args:
        M_hat: 预测伪造 mask，形状 [B, 1, H, W]
        M: 原始伪造 mask，形状 [B, 1, H_0, W_0]
    Returns:
        mask_loss: MSE(M_hat, M_down)
    """
    H, W = M_hat.shape[-2:]
    M_down = F.adaptive_avg_pool2d(M, output_size=(H, W))  # 下采样
    return F.mse_loss(M_hat, M_down)


def pixel_loss(Y_ce, Y_un, M):
    """
    Args:
        Y_ce: certainty 区域特征图 [B, C, H, W]
        Y_un: uncertainty 区域特征图 [B, C, H, W]
        M: 原始伪造 mask，形状 [B, 1, H_0, W_0]
    Returns:
        pixel_loss = MSE(Y_ce, M') + MSE(Y_un, 1 - M')
    """
    H, W = Y_ce.shape[-2:]
    M_down = F.adaptive_avg_pool2d(M, output_size=(H, W))        # 下采样
    M_down_expand = M_down.expand_as(Y_ce)                       # 扩展通道数
    return F.mse_loss(Y_ce, M_down_expand) + F.mse_loss(Y_un, 1 - M_down_expand)

# ==================== ABNM Module ====================
class ABNM(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(ABNM, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.inorm = nn.InstanceNorm2d(num_channels, affine=True)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1),
            nn.ReLU(inplace=True)
        )

        self.attn_bn = nn.Conv2d(num_channels // reduction, num_channels, 1)
        self.attn_in = nn.Conv2d(num_channels // reduction, num_channels, 1)

    def forward(self, x):
        x_bn = self.bn(x)
        x_in = self.inorm(x)

        s = self.fc(x)

        b = torch.sigmoid(self.attn_bn(s))
        i = torch.sigmoid(self.attn_in(s))

        alpha_bn = b / (b + i + 1e-6)
        alpha_in = i / (b + i + 1e-6)

        out = alpha_bn * x_bn + alpha_in * x_in
        return out

class ABNMConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, abnm=True, bias=True):
        super(ABNMConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.abnm = ABNM(out_planes) if abnm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.abnm is not None:
            x = self.abnm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# ==================== LFD Module ====================
class LFDModule(nn.Module):
    def __init__(self, in_channels):
        super(LFDModule, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def compute_entropy(self, p):
        eps = 1e-6
        p = p.clamp(min=eps, max=1 - eps)  # 避免log(0)
        return -p * torch.log(p) - (1 - p) * torch.log(1 - p)

    def forward(self, x):
        """
        Args:
            x: 特征图 [B, C, H, W]
        Returns:
            M_hat: 预测伪造 mask [B, 1, H, W]
            Y_ce: certainty 区域特征图 [B, C, H, W]
            Y_un: uncertainty 区域特征图 [B, C, H, W]
        """

        # Step 2: Cls 模块预测 mask
        M_hat = self.cls_head(x)  # [B, 1, H, W]

        # Step 3: 计算像素级不确定性（信息熵）
        E = self.compute_entropy(M_hat)  # [B, 1, H, W]

        # Step 4: 加权分离特征
        Y_ce = 2 * (torch.exp(-E) - 0.5) * x  # certainty 区域
        Y_un = 2 * (1 - torch.exp(-E)) * x    # uncertainty 区域

        return M_hat, Y_ce, Y_un, E

# ==================== Discriminator Module ====================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, max_iter):
        ctx.alpha = alpha
        ctx.max_iter = max_iter
        ctx.iter_num = ctx.iter_num + 1 if hasattr(ctx, 'iter_num') else 1
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        max_iter = ctx.max_iter
        iter_num = ctx.iter_num
        coeff = np.float32(2.0 * (1.0 - 0.0) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1.0 + 0.0)
        return -coeff * grad_output, None, None

class Discriminator(nn.Module):
    def __init__(self, channels=1024, num_classes=4, alpha=10, max_iter=4000):
        super(Discriminator, self).__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, num_classes)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )

    def forward(self, x):
        adversarial_out = self.ad_net(GRL.apply(x, self.alpha, self.max_iter))
        return adversarial_out

# ==================== Core Module ====================
class DFPD_Net(nn.Module):
    def __init__(self, model=None, feature_size=512, model_name='resnet50', pretrain=True):
        """
        Args:
            model: 可选，如果提供则直接使用该模型作为特征提取器（向后兼容）
            feature_size: 特征维度，默认512
            model_name: 模型名称，默认'resnet50'（仅在model为None时使用）
            pretrain: 是否使用预训练权重，默认True（仅在model为None时使用）
        """
        super(DFPD_Net, self).__init__()

        # 如果提供了model，直接使用；否则根据model_name创建
        if model is not None:
            self.features = model
        else:
            if model_name == 'resnet50':
                # 延迟导入以避免循环导入
                from dfpd_net.resnet import resnet50
                self.features = resnet50(pretrained=pretrain)
            elif model_name == 'xception':
                # 延迟导入以避免循环导入
                from dfpd_net.xception import xception
                self.features = xception(pretrained=pretrain)
            else:
                raise ValueError(f"Unsupported model_name: {model_name}. Currently only 'resnet50' and 'xception' are supported.")
        # 使用 AdaptiveMaxPool2d 替代固定尺寸的 MaxPool，以支持不同输入尺寸
        # 这样可以兼容 Xception（padding=0 导致空间尺寸不整齐）和其他 backbone
        self.max1 = nn.AdaptiveMaxPool2d(1)
        self.max2 = nn.AdaptiveMaxPool2d(1)
        self.max3 = nn.AdaptiveMaxPool2d(1)
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

