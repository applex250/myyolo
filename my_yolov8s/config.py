"""
YOLOv8s 配置文件
包含模型架构定义和缩放参数
"""

import math

# YOLOv8s 缩放参数
# [depth_multiple, width_multiple, max_channels]
YOLOV8S_SCALES = {
    'n': [0.33, 0.25, 1024],  # YOLOv8n
    's': [0.33, 0.50, 1024],  # YOLOv8s
    'm': [0.67, 0.75, 768],   # YOLOv8m
    'l': [1.00, 1.00, 512],   # YOLOv8l
    'x': [1.00, 1.25, 512],   # YOLOv8x
}

# YOLOv8s 模型配置
# [from, repeats, module, args]
# from: -1 表示上一层，列表表示多个输入
# repeats: 模块重复次数
# module: 模块名称
# args: 模块参数
YOLOV8S_BACKBONE = [
    # [from, repeats, module, args]
    [-1, 1, 'Conv', [64, 3, 2]],      # 0-P1/2: stem
    [-1, 1, 'Conv', [128, 3, 2]],     # 1-P2/4
    [-1, 3, 'C2f', [128, True]],      # 2
    [-1, 1, 'Conv', [256, 3, 2]],     # 3-P3/8
    [-1, 6, 'C2f', [256, True]],      # 4
    [-1, 1, 'Conv', [512, 3, 2]],     # 5-P4/16
    [-1, 6, 'C2f', [512, True]],      # 6
    [-1, 1, 'Conv', [1024, 3, 2]],    # 7-P5/32
    [-1, 3, 'C2f', [1024, True]],     # 8
    [-1, 1, 'SPPF', [1024, 5]],       # 9
]

YOLOV8S_HEAD = [
    [-1, 1, 'Upsample', [None, 2, 'nearest']],  # 10
    [[-1, 6], 1, 'Concat', [1]],                # 11 cat P4
    [-1, 3, 'C2f', [512]],                      # 12
    [-1, 1, 'Upsample', [None, 2, 'nearest']],  # 13
    [[-1, 4], 1, 'Concat', [1]],                # 14 cat P3
    [-1, 3, 'C2f', [256]],                      # 15 (P3/8-small)
    [-1, 1, 'Conv', [256, 3, 2]],               # 16
    [[-1, 12], 1, 'Concat', [1]],               # 17 cat P4
    [-1, 3, 'C2f', [512]],                      # 18 (P4/16-medium)
    [-1, 1, 'Conv', [512, 3, 2]],               # 19
    [[-1, 9], 1, 'Concat', [1]],                # 20 cat P5
    [-1, 3, 'C2f', [1024]],                     # 21 (P5/32-large)
    [[15, 18, 21], 1, 'Detect', ['nc']],        # 22 Detect(P3, P4, P5)
]


def make_divisible(x, divisor=8):
    """确保数值能被 divisor 整除"""
    return math.ceil(x / divisor) * divisor


def parse_model_config(nc=80, model_scale='s'):
    """
    解析模型配置，返回构建模型所需的信息

    参数:
        nc: 类别数
        model_scale: 模型规模 ('n', 's', 'm', 'l', 'x')

    返回:
        包含模型构建信息的字典
    """
    if model_scale not in YOLOV8S_SCALES:
        raise ValueError(f"不支持的模型规模: {model_scale}，可选: {list(YOLOV8S_SCALES.keys())}")

    depth_multiple, width_multiple, max_channels = YOLOV8S_SCALES[model_scale]

    return {
        'nc': nc,
        'depth_multiple': depth_multiple,
        'width_multiple': width_multiple,
        'max_channels': max_channels,
        'backbone': YOLOV8S_BACKBONE,
        'head': YOLOV8S_HEAD,
    }


# COCO 数据集类别名称 (80类)
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
