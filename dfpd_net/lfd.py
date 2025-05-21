import torch
import torch.nn as nn
import torch.nn.functional as F

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