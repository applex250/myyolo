"""
独立 YOLOv8s 实现 - 可直接加载 yolov8s.pt 权重

使用方法:
    from my_yolov8s import YOLOv8s, create_model

    # 创建模型并加载预训练权重
    model = YOLOv8s(nc=80, weights_path='yolov8s.pt')

    # 或使用便捷函数
    model = create_model(nc=80, weights_path='yolov8s.pt')

    # 推理
    import torch
    x = torch.randn(1, 3, 640, 640)
    output = model(x)
"""

from .model import YOLOv8s, create_model
from .modules import Conv, C2f, SPPF, Bottleneck, DFL, Concat, DWConv, Upsample
from .head import Detect
from .config import COCO_NAMES, YOLOV8S_SCALES

__all__ = [
    'YOLOv8s',
    'create_model',
    'Conv',
    'C2f',
    'SPPF',
    'Bottleneck',
    'DFL',
    'Concat',
    'DWConv',
    'Upsample',
    'Detect',
    'COCO_NAMES',
    'YOLOV8S_SCALES',
]
__version__ = '1.0.0'
