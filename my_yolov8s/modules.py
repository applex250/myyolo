"""
YOLOv8s 基础模块实现
包含: Conv, Bottleneck, C2f, SPPF, DFL, Concat
完全兼容官方 ultralytics 权重.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def autopad(k, p=None, d=1):
    """自动计算 padding 使输出尺寸不变 (same padding)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    """确保数值能被 divisor 整除."""
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    """标准卷积模块: Conv2d + BatchNorm + SiLU (激活函数) 完全兼容 ultralytics.nn.modules.conv.Conv.
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """参数: c1: 输入通道数 c2: 输出通道数 k: 卷积核大小 s: 步长 p: padding (None 表示自动计算) g: 分组数 (groups) d: 膨胀率 (dilation) act: 激活函数
        (True=SiLU, False=无, 或传入自定义激活函数).
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合 BN 后的前向传播 (用于模型优化)."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """深度可分离卷积 (Depthwise Convolution)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """标准瓶颈块: 1x1 conv -> 3x3 conv (+ shortcut) 完全兼容 ultralytics.nn.modules.block.Bottleneck.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """参数: c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: 分组卷积的分组数 k: 两个卷积的卷积核大小元组 e: 扩展比例 (中间层通道数 = c2 * e).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8 核心模块 - CSP Bottleneck with 2 convolutions (fast) 通过将特征分成两部分，一部分直接传递，另一部分通过多个 Bottleneck 处理 完全兼容
    ultralytics.nn.modules.block.C2f.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """参数: c1: 输入通道数 c2: 输出通道数 n: Bottleneck 的数量 shortcut: Bottleneck 中是否使用残差连接 g: 分组卷积的分组数 e: 扩展比例.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 注意: C2f 中的 Bottleneck 使用 k=((3, 3), (3, 3))，即两个 3x3 卷积
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """前向传播: split -> bottleneck chain -> concat -> conv."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """快速空间金字塔池化 (Spatial Pyramid Pooling - Fast) 通过多次最大池化捕获不同尺度的上下文信息 完全兼容 ultralytics.nn.modules.block.SPPF.
    """

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        """参数: c1: 输入通道数 c2: 输出通道数 k: 最大池化核大小 n: 池化迭代次数 (默认3，输出 n+1 个特征图) shortcut: 是否使用残差连接.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # 注意: cv1 使用 act=False (无激活函数)
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """前向传播: conv -> n次池化 -> concat -> conv."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class DFL(nn.Module):
    """分布式焦点损失积分模块 (Distribution Focal Loss) 将分类分布转换为连续的边界框坐标预测 完全兼容 ultralytics.nn.modules.block.DFL.
    """

    def __init__(self, c1=16):
        """参数: c1: 分布 bins 数量 (reg_max).
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """参数: x: 形状为 (b, c1*4, a) 的张量，其中 c1=reg_max, a=anchor数 返回: 形状为 (b, 4, a) 的张量，表示积分后的边界框偏移.
        """
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Concat(nn.Module):
    """张量拼接模块 完全兼容 ultralytics.nn.modules.conv.Concat.
    """

    def __init__(self, dimension=1):
        """参数: dimension: 拼接的维度 (默认为通道维度).
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Upsample(nn.Module):
    """上采样模块 (使用 nn.Upsample)."""

    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
