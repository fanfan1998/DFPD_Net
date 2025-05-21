import torch
import torch.nn.functional as F

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