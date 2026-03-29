"""
YOLOv8s 检测头实现
包含: Detect 检测头
完全兼容官方 ultralytics 权重
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Conv, DWConv, DFL


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """生成锚点坐标和步长张量"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将距离预测转换为边界框坐标"""
    lt, rb = distance.chunk(2, dim)  # left-top, right-bottom
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class Detect(nn.Module):
    """YOLOv8 检测头
    处理多尺度特征图，输出边界框和类别预测
    完全兼容 ultralytics.nn.modules.head.Detect
    """
    dynamic = False  # 强制网格重建
    export = False  # 导出模式
    format = None  # 导出格式
    max_det = 300  # 最大检测数
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # 初始化
    strides = torch.empty(0)  # 初始化
    legacy = False  # 向后兼容 v3/v5/v8/v9 模型
    xyxy = False  # xyxy 或 xywh 输出格式

    def __init__(self, nc=80, reg_max=16, ch=()):
        """
        参数:
            nc: 类别数 (number of classes)
            reg_max: DFL 分布的最大通道数
            ch: 输入通道列表，对应不同尺度的特征图
        """
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数 (通常为3)
        self.reg_max = reg_max  # DFL 分布 bins 数
        self.no = nc + self.reg_max * 4  # 每个锚点的输出维度 (类别 + 4个边界框参数)
        self.stride = torch.zeros(self.nl)  # 步长，初始化为0

        # Box 回归分支和 Class 分类分支的通道数
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        # Box 回归分支 - 3层: Conv(3x3) -> Conv(3x3) -> Conv2d(1x1)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )

        # Class 分类分支 - legacy 模式 (匹配官方 yolov8s.pt)
        # 结构: Conv(3x3) -> Conv(3x3) -> Conv2d(1x1)
        # 键名: .0=Conv, .1=Conv, .2=Conv2d
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),           # .0: 3x3 Conv
                Conv(c3, c3, 3),          # .1: 3x3 Conv
                nn.Conv2d(c3, self.nc, 1), # .2: 1x1 Conv2d
            ) for x in ch
        )

        # DFL 模块 (仅当 reg_max > 1 时使用)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 多尺度特征图列表 [P3, P4, P5]

        返回:
            训练模式: 字典包含 boxes 和 scores
            推理模式: (y, preds) 元组，y 为解码后的输出
        """
        # 训练模式
        if self.training:
            return self._forward_train(x)

        # 推理模式
        return self._forward_inference(x)

    def _forward_train(self, x):
        """训练模式前向传播"""
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {'boxes': boxes, 'scores': scores, 'feats': x}

    def _forward_inference(self, x):
        """推理模式前向传播"""
        shape = x[0].shape  # BCHW

        # 对每个尺度分别处理: 拼接 box 和 class 预测
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # 构建锚点和步长
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 拼接所有尺度的输出
        # x[i] 形状: (b, no, h_i, w_i) -> (b, no, h_i*w_i)
        # 拼接后: (b, no, total_anchors)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)

        # DFL 处理 box 预测
        dbox = self._decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # 拼接最终输出: (b, 4+nc, total_anchors)
        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y, {'boxes': box, 'scores': cls, 'feats': x}

    def _decode_bboxes(self, bboxes, anchors, xywh=True):
        """解码边界框预测"""
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.xyxy, dim=1)

    def bias_init(self):
        """初始化检测头偏置 (需要 stride 可用)"""
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0  # box 偏置
            b[-1].bias.data[:self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls 偏置 (.01 objects, 80 classes, 640 img)
