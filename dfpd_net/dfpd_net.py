import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== Loss Functions (保持不变) ====================
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
    H, W = M_hat.shape[-2:]
    M_down = F.adaptive_avg_pool2d(M, output_size=(H, W))
    return F.mse_loss(M_hat, M_down)

def pixel_loss(Y_ce, Y_un, M):
    H, W = Y_ce.shape[-2:]
    M_down = F.adaptive_avg_pool2d(M, output_size=(H, W))
    M_down_expand = M_down.expand_as(Y_ce)
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

# ==================== 核心优化组件: ABNM_DSConv ====================
# 特点：深度可分离卷积 + 移除冗余BN + 集成ABNM
class ABNM_DSConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, reduction=16):
        super(ABNM_DSConv, self).__init__()
        
        # 1. Depthwise Convolution (groups=in_planes)
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_planes, bias=False)
        
        # 2. Pointwise Convolution
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        # 3. ABNM + ReLU (直接由 ABNM 处理归一化，不再添加额外 BN)
        self.abnm = ABNM(out_planes, reduction=reduction) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.abnm(out)
        out = self.relu(out)
        return out

# ==================== LFD & Discriminator (保持不变) ====================
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
        p = p.clamp(min=eps, max=1 - eps)
        return -p * torch.log(p) - (1 - p) * torch.log(1 - p)

    def forward(self, x):
        M_hat = self.cls_head(x)
        E = self.compute_entropy(M_hat)
        Y_ce = 2 * (torch.exp(-E) - 0.5) * x
        Y_un = 2 * (1 - torch.exp(-E)) * x
        return M_hat, Y_ce, Y_un, E

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
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, num_classes)
        self.ad_net = nn.Sequential(
            self.fc1, nn.ReLU(), nn.Dropout(0.5), self.fc2
        )
        self.alpha = alpha
        self.max_iter = max_iter

    def forward(self, x):
        return self.ad_net(GRL.apply(x, self.alpha, self.max_iter))

# ==================== DFPD_Net 主模型 ====================
class DFPD_Net(nn.Module):
    def __init__(self, model=None, feature_size=256, model_name='resnet50', pretrain=True, proj_dim=256):
        """
        Args:
            feature_size: 隐层维度 (hidden_dim)，默认 256 (原版 512)
            proj_dim:     投影解耦维度，默认 256 (原版 ~1024)
            
            说明：推荐配置为 (proj=256, hidden=256)，这是在参数量和表达能力之间的最佳平衡点。
        """
        super(DFPD_Net, self).__init__()

        # --- Backbone 加载 ---
        if model is not None:
            self.features = model
        else:
            if model_name == 'resnet50':
                from dfpd_net.resnet import resnet50
                self.features = resnet50(pretrained=pretrain)
            elif model_name == 'xception':
                from dfpd_net.xception import xception
                self.features = xception(pretrained=pretrain)
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Backbone 输出通道数 (假设 ResNet50/Xception 结构): x3(512/728), x4(1024/728), x5(2048/2048)
        # 为兼容性，我们需要获取具体的通道数。如果使用的是修改版 backbone 且 output 相同则无需调整。
        # 这里假设输入的 backbone 返回的是标准通道数。
        
        # 注意：为了让代码对 Xception 和 ResNet 都通用，最好动态获取通道数或者手动指定
        # 下面代码假设输入通道数为 ResNet50 标准 (512, 1024, 2048)。
        # 如果是 Xception，x3=728, x4=728, x5=2048。
        # 你可以在初始化时传入 channels 列表，或者在这里硬编码适配。
        # 为了稳健，建议如下：
        
        if model_name == 'xception':
            # XceptionWrapper 实际输出: x3=512 (project_s1: 256->512), x4=1024 (project_s2: 728->1024), x5=2048
            c3, c4, c5 = 512, 1024, 2048
        else: # resnet50
            c3, c4, c5 = 512, 1024, 2048

        # --- 分支构建 (使用 hidden=384, proj=256) ---
        
        # Scale 1 (Feature Map 32x32)
        self.S1_G1 = self._make_branch(c3, feature_size, proj_dim) # Content
        self.S1_G2 = self._make_branch(c3, feature_size, proj_dim) # Forgery
        
        # Scale 2 (Feature Map 16x16)
        self.S2_G1 = self._make_branch(c4, feature_size, proj_dim)
        self.S2_G2 = self._make_branch(c4, feature_size, proj_dim)
        
        # Scale 3 (Feature Map 8x8)
        self.S3_G1 = self._make_branch(c5, feature_size, proj_dim)
        self.S3_G2 = self._make_branch(c5, feature_size, proj_dim)

        # --- 共享分类器 ---
        # 输入维度固定为 proj_dim (256)，无论前端 backbone 如何变化
        self.shared_classifier = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, 2),
        )

    def _make_branch(self, in_channels, hidden_dim, out_dim):
        """
        构建轻量化分支：
        1. 1x1 Conv: 降维/对齐到 hidden_dim (384)
        2. DSConv:   提取特征并映射到 out_dim (256)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ABNM_DSConv(hidden_dim, out_dim, kernel_size=3, padding=1)
        )

    def feature_projection_disentangle(self, f_fea, c_fea, eps=1e-12):
        dot_prod = torch.sum(f_fea * c_fea, dim=1, keepdim=True)
        c_norm_sq = torch.sum(c_fea * c_fea, dim=1, keepdim=True).clamp(min=eps)
        f_proj = (dot_prod / c_norm_sq) * c_fea
        f_forgery = f_fea - f_proj
        return f_proj, f_forgery

    def forward(self, x):
        # 假设 Backbone 返回元组，前三个是特征图
        outs = self.features(x)
        x3, x4, x5 = outs[0], outs[1], outs[2]
        
        # 处理辅助 Loss 的返回值 (如果 backbone 返回了这些)
        # 为了通用性，这里做个简单的解包检查
        if len(outs) > 3:
            M_hat1, Y_ce1, Y_un1, E1 = outs[3], outs[4], outs[5], outs[6]
            M_hat2, Y_ce2, Y_un2, E2 = outs[7], outs[8], outs[9], outs[10]
            M_hat3, Y_ce3, Y_un3, E3 = outs[11], outs[12], outs[13], outs[14]
        else:
            # 如果 backbone 没有返回辅助信息（例如只是单纯的 timm 模型），则设为 None
            M_hat1 = Y_ce1 = Y_un1 = E1 = None
            M_hat2 = Y_ce2 = Y_un2 = E2 = None
            M_hat3 = Y_ce3 = Y_un3 = E3 = None

        # 1. 特征提取 & 池化
        c_fea1 = self.max_pool(self.S1_G1(x3)).view(x.size(0), -1)
        f_fea1 = self.max_pool(self.S1_G2(x3)).view(x.size(0), -1)
        
        c_fea2 = self.max_pool(self.S2_G1(x4)).view(x.size(0), -1)
        f_fea2 = self.max_pool(self.S2_G2(x4)).view(x.size(0), -1)
        
        c_fea3 = self.max_pool(self.S3_G1(x5)).view(x.size(0), -1)
        f_fea3 = self.max_pool(self.S3_G2(x5)).view(x.size(0), -1)

        # 2. 投影解耦
        _, f_dis1 = self.feature_projection_disentangle(f_fea1, c_fea1)
        _, f_dis2 = self.feature_projection_disentangle(f_fea2, c_fea2)
        _, f_dis3 = self.feature_projection_disentangle(f_fea3, c_fea3)

        # 3. 分类 (Shared)
        pred1 = self.shared_classifier(f_dis1)
        pred2 = self.shared_classifier(f_dis2)
        pred3 = self.shared_classifier(f_dis3)

        # 4. 组装输出
        output_dict = {
            'pred1': pred1, 'pred2': pred2, 'pred3': pred3,
            'c_fea1': c_fea1, 'c_fea2': c_fea2, 'c_fea3': c_fea3,
            'f_fea1': f_fea1, 'f_fea2': f_fea2, 'f_fea3': f_fea3,
            'f_dis1': f_dis1, 'f_dis2': f_dis2, 'f_dis3': f_dis3
        }
        
        # 仅当有辅助输出时添加，避免报错
        if M_hat1 is not None:
            output_dict.update({
                'M_hat1': M_hat1, 'Y_ce1': Y_ce1, 'Y_un1': Y_un1, 'E1': E1,
                'M_hat2': M_hat2, 'Y_ce2': Y_ce2, 'Y_un2': Y_un2, 'E2': E2,
                'M_hat3': M_hat3, 'Y_ce3': Y_ce3, 'Y_un3': Y_un3, 'E3': E3
            })

        return output_dict

# 兼容性保留 BasicConv，虽然这里暂时没用到
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x