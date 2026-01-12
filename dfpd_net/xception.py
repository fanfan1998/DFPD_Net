import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dfpd_net.dfpd_net import LFDModule
import os

__all__ = ['xception']

logger = logging.getLogger(__name__)

# =============================================================================
#  Basic Components (SeparableConv, Block)
# =============================================================================

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

# =============================================================================
#  Xception Backbone (Original Structure)
# =============================================================================

class Xception(nn.Module):
    def __init__(self, num_classes=1000, inc=3):
        super(Xception, self).__init__()
        
        # [RESTORED] Original Padding=0
        # Best for loading pre-trained weights (FaceForensics++, ImageNet)
        # Spatial dimensions will be irregular (e.g., 256 -> 127 -> 125...), 
        # but downstream AdaptiveMaxPool2d(1) handles this perfectly.
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle Flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit Flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # 【新增】分类层（Baseline 必须有这个）
        self.fc = nn.Linear(2048, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    # 【新增】Forward 函数（Baseline 必须有这个才能跑数据）
    def forward(self, x):
        # Entry Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle Flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit Flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# =============================================================================
#  Xception Wrapper for DFPD (PFDS Compliant)
# =============================================================================

class XceptionWrapper(nn.Module):
    """
    Xception Wrapper for DFPD_Net - PFDS Aligned Version
    
    Structure:
    1. Low Level (Stride 8): Block 2 -> LFD1 -> Feed Y_ce1 to Block 3
    2. Mid Level (Stride 16): Block 7 -> LFD2 -> Feed Y_ce2 to Block 8
    3. High Level Guide (Stride 16): Block 11 -> LFD3 -> Feed Y_ce3 to Exit Flow (Block 12)
    
    Returns projected Y_ce features for x3/x4 to maximize certainty utilization.
    """
    def __init__(self, pretrained=False):
        super(XceptionWrapper, self).__init__()
        
        self.xception = Xception()
        
        if pretrained:
            try:
                # 方法1: 尝试使用 pretrainedmodels 库加载
                import pretrainedmodels
                print("==> Loading Xception pretrained weights from pretrainedmodels...")
                pretrained_model = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
                pretrained_state_dict = pretrained_model.state_dict()
                
                # 获取当前模型的 state_dict
                model_state_dict = self.xception.state_dict()
                
                # 匹配并过滤权重（pretrainedmodels 的 xception 结构与我们的更接近）
                filtered_state_dict = {}
                for k, v in pretrained_state_dict.items():
                    if k in model_state_dict and v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                
                # 加载匹配的权重
                missing_keys, unexpected_keys = self.xception.load_state_dict(filtered_state_dict, strict=False)
                print(f"==> Xception pretrained weights loaded:")
                print(f"    • Loaded {len(filtered_state_dict)} layers")
                print(f"    • {len(missing_keys)} missing keys (new modules like LFD)")
                print(f"    • {len(unexpected_keys)} unexpected keys")
                
            except (ImportError, AttributeError) as e:
                try:
                    # 方法2: 尝试从本地文件加载
                    pretrained_path = 'xception-c0a72b38.pth.tar'
                    if os.path.exists(pretrained_path):
                        print(f"==> Loading Xception pretrained weights from: {pretrained_path}")
                        state_dict = torch.load(pretrained_path, map_location='cpu')
                        # 如果 state_dict 包含 'state_dict' 键（checkpoint 格式）
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        missing_keys, unexpected_keys = self.xception.load_state_dict(state_dict, strict=False)
                        print(f"==> Xception pretrained weights loaded:")
                        print(f"    • {len(missing_keys)} missing keys")
                        print(f"    • {len(unexpected_keys)} unexpected keys")
                    else:
                        print("==> Warning: Xception pretrained weights not found.")
                        print("    Please download xception-c0a72b38.pth.tar or install pretrainedmodels library.")
                        print("    Continuing with random initialization...")
                except Exception as e2:
                    print(f"==> Warning: Failed to load Xception pretrained weights: {e2}")
                    print("    Continuing with random initialization...")
            except Exception as e:
                print(f"==> Warning: Failed to load Xception pretrained weights: {e}")
                print("    Continuing with random initialization...")

        # ---------------------------------------------------------------------
        # Stage 1: Stride 8 (256 channels)
        # ---------------------------------------------------------------------
        self.LFD1 = LFDModule(256)
        self.bn_ce1 = nn.BatchNorm2d(256)
        
        # Projection for CFD: Y_ce (256) -> 512
        self.project_s1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ---------------------------------------------------------------------
        # Stage 2: Stride 16 (728 channels) - Mid-Flow Split at Block 7
        # ---------------------------------------------------------------------
        self.LFD2 = LFDModule(728)
        self.bn_ce2 = nn.BatchNorm2d(728)
        
        # Projection for CFD: Y_ce (728) -> 1024
        self.project_s2 = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # ---------------------------------------------------------------------
        # Stage 3: Stride 16 (728 channels) - Before Exit Flow
        # ---------------------------------------------------------------------
        # LFD3 uses Block 11 output to guide the Exit Flow
        self.LFD3 = LFDModule(728)
        self.bn_ce3 = nn.BatchNorm2d(728) 
        
        # Final output x5 is 2048 channels, matching DFPD expectation.

    def forward(self, x):
        # ------------------------------------------------------------------
        # 1. Entry Flow (Guidance for Stage 1)
        # ------------------------------------------------------------------
        x = self.xception.conv1(x)
        x = self.xception.bn1(x)
        x = self.xception.relu(x)
        x = self.xception.conv2(x)
        x = self.xception.bn2(x)
        x = self.xception.relu(x)

        x = self.xception.block1(x)
        x2 = self.xception.block2(x) # Stride 8 approx

        # >>> LFD 1 <<<
        M_hat1, Y_ce1, Y_un1, E1 = self.LFD1(x2)
        Y_ce1_bn = self.bn_ce1(Y_ce1)

        # ------------------------------------------------------------------
        # 2. Middle Flow Part A (Guidance for Stage 2)
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce1 guides Block 3
        x3_in = self.xception.block3(Y_ce1_bn) # Stride 16 approx
        
        x_mid = self.xception.block4(x3_in)
        x_mid = self.xception.block5(x_mid)
        x_mid = self.xception.block6(x_mid)
        x7 = self.xception.block7(x_mid) # Stride 16 approx

        # >>> LFD 2 <<<
        M_hat2, Y_ce2, Y_un2, E2 = self.LFD2(x7)
        Y_ce2_bn = self.bn_ce2(Y_ce2)

        # ------------------------------------------------------------------
        # 3. Middle Flow Part B (Guidance for Stage 3 / Exit)
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce2 guides Block 8
        x_mid_b = self.xception.block8(Y_ce2_bn)
        x_mid_b = self.xception.block9(x_mid_b)
        x_mid_b = self.xception.block10(x_mid_b)
        x11 = self.xception.block11(x_mid_b)

        # >>> LFD 3 (Placed BEFORE Exit Flow) <<<
        M_hat3, Y_ce3, Y_un3, E3 = self.LFD3(x11)
        Y_ce3_bn = self.bn_ce3(Y_ce3)

        # ------------------------------------------------------------------
        # 4. Exit Flow (Final Semantic Abstraction)
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce3 guides Exit Flow
        x_exit = self.xception.block12(Y_ce3_bn) # Stride 32 approx
        
        x_final = self.xception.conv3(x_exit)
        x_final = self.xception.bn3(x_final)
        x_final = self.xception.relu(x_final)
        
        x5 = self.xception.conv4(x_final)
        x5 = self.xception.bn4(x5) # 2048 channels

        # ------------------------------------------------------------------
        # 5. Output Formatting for CFD
        # ------------------------------------------------------------------
        # Use Y_ce features projected to correct channels
        
        # x3 (Stage 1): Project Y_ce1 (256->512)
        x3_out = self.project_s1(Y_ce1_bn)
        
        # x4 (Stage 2): Project Y_ce2 (728->1024)
        x4_out = self.project_s2(Y_ce2_bn)
        
        # x5 (Stage 3): Final Backbone Output (2048)
        # Guided by LFD3 during the Exit Flow process
        x5_out = x5

        return x3_out, x4_out, x5_out, M_hat1, Y_ce1, Y_un1, E1, M_hat2, Y_ce2, Y_un2, E2, M_hat3, Y_ce3, Y_un3, E3


def xception(pretrained=False, **kwargs):
    model = XceptionWrapper(pretrained=pretrained)
    return model