"""
YOLOv8s 工具函数
包含: NMS, 坐标缩放, 图像处理等
"""

import torch
import torchvision
import cv2
import numpy as np


def box_iou(box1, box2):
    """计算两组边界框的 IoU

    参数:
        box1: (N, 4) 第一组边界框 (xyxy)
        box2: (M, 4) 第二组边界框 (xyxy)

    返回:
        (N, M) IoU 矩阵
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


    area1 = box_area(box1)
    area2 = box_area(box2)


    # 计算交集
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 计算 IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nc=80):
    """非极大值抑制 (NMS)

    参数:
        prediction: (B, 4+nc, N) 模型输出 (已解码的边界框和类别分数)
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        max_det: 每张图最大检测数
        nc: 类别数

    返回:
        List[(N, 6)] 检测结果 [x1, y1, x2, y2, conf, cls]
    """
    # 如果输出是元组，取第一个元素 (推理输出)

    bs = prediction.shape[0]  # batch size
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        # x 形状: (4+nc, N) = (84, 8400)
        # 前 4 行是坐标，后 nc 行是类别分数
        box = x[:4, :].t().contiguous()   # (N, 4) xywh
        scores = x[4:, :].t().contiguous()  # (N, nc)

        # xywh -> xyxy
        xyxy = torch.zeros_like(box)
        xyxy[:, 0] = box[:, 0] - box[:, 2] / 2  # x1
        xyxy[:, 1] = box[:, 1] - box[:, 3] / 2  # y1
        xyxy[:, 2] = box[:, 0] + box[:, 2] / 2  # x2
        xyxy[:, 3] = box[:, 1] + box[:, 3] / 2  # y2

        # 获取每个预测的最大类别分数和类别索引
        conf, j = scores.max(1, keepdim=True)
        conf = conf.squeeze(1)

        # 置信度过滤
        mask = conf > conf_thres
        xyxy = xyxy[mask]
        conf = conf[mask]
        j = j.squeeze(1)[mask]

        # 如果没有检测到任何目标
        if not xyxy.shape[0]:
            continue

        # 拼接结果 [x1, y1, x2, y2, conf, cls]
        detections = torch.cat((xyxy, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # NMS (按类别分别进行)
        # 这里简化为对所有类别一起做 NMS
        keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thres)

        # 限制最大检测数
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = detections[keep]

    return output


def scale_boxes(img1_shape, boxes, img0_shape):
    """将边界框坐标从 img1_shape 缩放到 img0_shape

    参数:
        img1_shape: 模型输入尺寸 (h, w)
        boxes: (N, 4) 边界框 (xyxy)
        img0_shape: 原始图像尺寸 (h, w)

    返回:
        缩放后的边界框
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (
        (img1_shape[1] - img0_shape[1] * gain) / 2,
        (img1_shape[0] - img0_shape[0] * gain) / 2,
    )

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    # 裁剪到图像边界
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])

    return boxes


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """将图像缩放到指定尺寸，保持纵横比，不足部分填充

    参数:
        img: 输入图像 (numpy array)
        new_shape: 目标尺寸 (int 或 (h, w))
        color: 填充颜色

    返回:
        img: 缩放后的图像
        ratio: 缩放比例
        pad: 填充大小 (dw, dh)
    """
    shape = img.shape  # current shape: [h, w, c]

    # 如果是 int， 转换为 (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算缩放后的尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算填充
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2

    # 缩放
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 填充
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


def preprocess_image(img, img_size=640):
    """预处理图像用于推理

    参数:
        img: 输入图像 (numpy array, BGR)
        img_size: 目标尺寸

    返回:
        tensor: (1, 3, H, W) 张量
        original_shape: 原始图像尺寸
    """
    original_shape = img.shape[:2]

    # Letterbox
    img, ratio, pad = letterbox(img, new_shape=img_size)

    # 转换颜色空间 BGR -> RGB
    img = img[:, :, ::-1]

    # 转换为 tensor
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).float() / 255.0

    # 添加 batch 维度
    tensor = tensor.unsqueeze(0)

    return tensor, original_shape
