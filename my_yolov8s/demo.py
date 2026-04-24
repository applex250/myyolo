"""
YOLOv8s 使用示例
演示如何加载预训练权重并进行推理.
"""

import torch

from my_yolov8s import YOLOv8s
from my_yolov8s.config import COCO_NAMES
from my_yolov8s.utils import non_max_suppression


def demo_basic():
    """基础使用示例."""
    print("=" * 60)
    print("YOLOv8s 基础使用示例")
    print("=" * 60)

    # 1. 创建模型并加载预训练权重
    print("\n1. 加载预训练模型...")
    model = YOLOv8s(nc=80, weights_path="yolov8s.pt", verbose=True)
    model.eval()

    # 2. 打印模型信息
    print("\n2. 模型信息:")
    model.info(detailed=False)
    print(f"   Stride: {model.stride}")

    # 3. 测试推理
    print("\n3. 测试推理...")
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    if isinstance(output, tuple):
        y = output[0]
        print(f"   推理输出形状: {y.shape}")
        print(f"   Box 坐标范围: [{y[:, :4].min():.2f}, {y[:, :4].max():.2f}]")
        print(f"   类别分数范围: [{y[:, 4:].min():.4f}, {y[:, 4:].max():.4f}]")

        # NMS
        detections = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45)
        print(f"   NMS 后检测数量: {detections[0].shape[0]}")

        if detections[0].shape[0] > 0:
            print("   前5个检测结果:")
            for i, det in enumerate(detections[0][:5]):
                x1, y1, x2, y2, conf, cls = det
                cls_name = COCO_NAMES[int(cls)]
                print(f"      [{i}] {cls_name}: conf={conf:.3f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")


def demo_training_mode():
    """训练模式示例."""
    print("\n" + "=" * 60)
    print("YOLOv8s 训练模式示例")
    print("=" * 60)

    # 加载模型
    print("\n1. 加载模型...")
    model = YOLOv8s(nc=80, weights_path="yolov8s.pt", verbose=False)

    # 切换到训练模式
    print("\n2. 切换到训练模式...")
    model.train()

    # 测试训练模式前向传播
    print("\n3. 测试训练模式前向传播...")
    x = torch.randn(2, 3, 640, 640)
    output = model(x)

    print(f"   输出类型: {type(output)}")
    print(f"   输出键: {list(output.keys())}")
    print(f"   boxes 形状: {output['boxes'].shape}")
    print(f"   scores 形状: {output['scores'].shape}")
    print(f"   feats 数量: {len(output['feats'])}")


def demo_custom_classes():
    """自定义类别数示例."""
    print("\n" + "=" * 60)
    print("YOLOv8s 自定义类别数示例")
    print("=" * 60)

    # 创建一个只有 10 个类别的模型
    print("\n1. 创建 10 类别模型 (不加载预训练权重)...")
    model = YOLOv8s(nc=10, verbose=True)
    model.eval()

    # 测试推理
    print("\n2. 测试推理...")
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    if isinstance(output, tuple):
        y = output[0]
        # 10 类别: 输出应该是 (1, 4+10, 8400) = (1, 14, 8400)
        print(f"   输出形状: {y.shape}")
        print("   预期形状: (1, 14, 8400)")


def demo_save_load():
    """保存和加载示例."""
    print("\n" + "=" * 60)
    print("YOLOv8s 保存和加载示例")
    print("=" * 60)

    # 加载模型
    print("\n1. 加载预训练模型...")
    model = YOLOv8s(nc=80, weights_path="yolov8s.pt", verbose=False)

    # 保存模型
    print("\n2. 保存模型...")
    model.save_weights("my_yolov8s_saved.pt")
    print("   已保存到 my_yolov8s_saved.pt")

    # 重新加载
    print("\n3. 重新加载模型...")
    model2 = YOLOv8s(nc=80, weights_path="my_yolov8s_saved.pt", verbose=True)

    # 验证输出一致
    print("\n4. 验证输出一致性...")
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    if isinstance(out1, tuple) and isinstance(out2, tuple):
        diff = torch.abs(out1[0] - out2[0]).max().item()
        print(f"   输出差异: {diff:.6f}")
        if diff < 1e-5:
            print("   ✓ 输出一致!")
        else:
            print("   ⚠ 输出有差异")


if __name__ == "__main__":
    import os

    # 检查 yolov8s.pt 是否存在
    if not os.path.exists("yolov8s.pt"):
        print("警告: yolov8s.pt 不存在，部分示例无法运行")
        print("请将 yolov8s.pt 放在当前目录下\n")

        # 只运行不需要预训练权重的示例
        demo_custom_classes()
    else:
        # 运行所有示例
        demo_basic()
        demo_training_mode()
        demo_custom_classes()
        demo_save_load()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
