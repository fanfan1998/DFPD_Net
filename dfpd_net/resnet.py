import torch.nn as nn
from  torch.utils.model_zoo import load_url as load_state_dict_from_url
from dfpd_net.dfpd_net import LFDModule
import torch
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        print(f"==> Loading pretrained weights from: {model_urls[arch]}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"==> Pretrained loaded with:")
        print(f"    • {len(missing_keys)} missing keys")
        print(f"    • {len(unexpected_keys)} unexpected keys")

    return model


# =============================================================================
#  ResNet Wrapper for DFPD (PFDS Compliant)
# =============================================================================

class ResNetWrapper(nn.Module):
    """
    ResNet Wrapper for DFPD_Net - PFDS Aligned Version
    
    Structure:
    1. Low Level (Stride 8): Layer1 -> LFD1 -> Feed Y_ce1 to Layer2
    2. Mid Level (Stride 16): Layer2 -> LFD2 -> Feed Y_ce2 to Layer3
    3. High Level (Stride 32): Layer3 -> LFD3 -> Feed Y_ce3 to Layer4
    
    Returns x3, x4, x5 and LFD outputs for DFPD_Net compatibility.
    """
    def __init__(self, block, layers, pretrained=False, progress=True, num_classes=1000, 
                 zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetWrapper, self).__init__()
        
        # 创建原始的 ResNet backbone
        self.resnet = ResNet(block, layers, num_classes=num_classes, 
                            zero_init_residual=zero_init_residual,
                            groups=groups, width_per_group=width_per_group,
                            replace_stride_with_dilation=replace_stride_with_dilation,
                            norm_layer=norm_layer)
        
        # 加载预训练权重
        if pretrained:
            # 根据 layers 参数确定架构名称
            arch_name = None
            if layers == [2, 2, 2, 2]:
                arch_name = 'resnet18'
            elif layers == [3, 4, 6, 3]:
                arch_name = 'resnet50' if block == Bottleneck else 'resnet34'
            elif layers == [3, 4, 23, 3]:
                arch_name = 'resnet101' if block == Bottleneck else None
            elif layers == [3, 8, 36, 3]:
                arch_name = 'resnet152'
            
            if arch_name and arch_name in model_urls:
                print(f"==> Loading ResNet pretrained weights from: {model_urls[arch_name]}")
                state_dict = load_state_dict_from_url(model_urls[arch_name], progress=progress)
                
                # 自动过滤掉和当前模型结构不匹配的 key
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if k in self.resnet.state_dict() and v.shape == self.resnet.state_dict()[k].shape
                }
                
                missing_keys, unexpected_keys = self.resnet.load_state_dict(filtered_state_dict, strict=False)
                
                print(f"==> ResNet pretrained weights loaded:")
                print(f"    • Loaded {len(filtered_state_dict)} layers")
                print(f"    • {len(missing_keys)} missing keys")
                print(f"    • {len(unexpected_keys)} unexpected keys")
            else:
                print(f"==> Warning: No pretrained weights available for this ResNet configuration.")
                print("    Continuing with random initialization...")
        
        # 根据 block 类型确定通道数
        if block == BasicBlock:
            # ResNet18/34: expansion = 1
            channels_l1 = 64 * block.expansion  # 64
            channels_l2 = 128 * block.expansion  # 128
            channels_l3 = 256 * block.expansion  # 256
        else:
            # ResNet50/101/152: expansion = 4
            channels_l1 = 64 * block.expansion  # 256
            channels_l2 = 128 * block.expansion  # 512
            channels_l3 = 256 * block.expansion  # 1024
        
        # ---------------------------------------------------------------------
        # Stage 1: After Layer1 (Stride 8)
        # ---------------------------------------------------------------------
        self.LFD1 = LFDModule(channels_l1)
        self.bn_ce1 = nn.BatchNorm2d(channels_l1)
        
        # ---------------------------------------------------------------------
        # Stage 2: After Layer2 (Stride 16)
        # ---------------------------------------------------------------------
        self.LFD2 = LFDModule(channels_l2)
        self.bn_ce2 = nn.BatchNorm2d(channels_l2)
        
        # ---------------------------------------------------------------------
        # Stage 3: After Layer3 (Stride 32)
        # ---------------------------------------------------------------------
        self.LFD3 = LFDModule(channels_l3)
        self.bn_ce3 = nn.BatchNorm2d(channels_l3)
        
        # 初始化 LFD 和 BN 模块
        for m in [self.LFD1, self.LFD2, self.LFD3, self.bn_ce1, self.bn_ce2, self.bn_ce3]:
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ------------------------------------------------------------------
        # 1. Entry Flow
        # ------------------------------------------------------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)
        
        # ------------------------------------------------------------------
        # 2. Layer1 -> LFD1
        # ------------------------------------------------------------------
        x2 = self.resnet.layer1(x1)
        
        # >>> LFD 1 <<<
        M_hat1, Y_ce1, Y_un1, E1 = self.LFD1(x2)
        Y_ce1_bn = self.bn_ce1(Y_ce1)
        
        # ------------------------------------------------------------------
        # 3. Layer2 -> LFD2
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce1 guides Layer2
        x3 = self.resnet.layer2(Y_ce1_bn)
        
        # >>> LFD 2 <<<
        M_hat2, Y_ce2, Y_un2, E2 = self.LFD2(x3)
        Y_ce2_bn = self.bn_ce2(Y_ce2)
        
        # ------------------------------------------------------------------
        # 4. Layer3 -> LFD3
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce2 guides Layer3
        x4 = self.resnet.layer3(Y_ce2_bn)
        
        # >>> LFD 3 <<<
        M_hat3, Y_ce3, Y_un3, E3 = self.LFD3(x4)
        Y_ce3_bn = self.bn_ce3(Y_ce3)
        
        # ------------------------------------------------------------------
        # 5. Layer4 (Final)
        # ------------------------------------------------------------------
        # [PFDS]: Y_ce3 guides Layer4
        x5 = self.resnet.layer4(Y_ce3_bn)
        
        return x3, x4, x5, M_hat1, Y_ce1, Y_un1, E1, M_hat2, Y_ce2, Y_un2, E2, M_hat3, Y_ce3, Y_un3, E3


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model with DFPD wrapper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetWrapper(BasicBlock, [2, 2, 2, 2], pretrained=pretrained, progress=progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model with DFPD wrapper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetWrapper(BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model with DFPD wrapper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetWrapper(Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model with DFPD wrapper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetWrapper(Bottleneck, [3, 4, 23, 3], pretrained=pretrained, progress=progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model with DFPD wrapper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetWrapper(Bottleneck, [3, 8, 36, 3], pretrained=pretrained, progress=progress, **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNetWrapper(Bottleneck, [3, 4, 6, 3],
                        pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNetWrapper(Bottleneck, [3, 4, 23, 3],
                        pretrained=False, progress=True, **kwargs)

