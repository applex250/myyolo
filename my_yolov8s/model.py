"""
YOLOv8s 主模型类
完全兼容官方 ultralytics 权重命名.

State Dict 命名规则:
- model.{layer_index}.{submodule}.{param}
- 例如: model.0.conv.weight, model.2.cv1.conv.weight
"""

import torch
from torch import nn

from .head import Detect
from .modules import SPPF, C2f, Concat, Conv, make_divisible


class YOLOv8s(nn.Module):
    """独立的 YOLOv8s 实现 能够直接加载官方预训练的 yolov8s.pt 权重文件.

    架构:
        Backbone (layers 0-9):
            0: Conv(3, 32, 3, 2)      - P1/2
            1: Conv(32, 64, 3, 2)     - P2/4
            2: C2f(64, 64, 1)         -
            3: Conv(64, 128, 3, 2)    - P3/8
            4: C2f(128, 128, 2)       -
            5: Conv(128, 256, 3, 2)   - P4/16
            6: C2f(256, 256, 2)       -
            7: Conv(256, 512, 3, 2)   - P5/32
            8: C2f(512, 512, 1)       -
            9: SPPF(512, 512, 5)      -

        Head (layers 10-22):
            10: Upsample(2x)
            11: Concat([10, 6])
            12: C2f(512, 256, 1)
            13: Upsample(2x)
            14: Concat([13, 4])
            15: C2f(256, 128, 1)      - P3/8-small
            16: Conv(128, 128, 3, 2)
            17: Concat([16, 12])
            18: C2f(128, 256, 1)      - P4/16-medium
            19: Conv(256, 256, 3, 2)
            20: Concat([19, 9])
            21: C2f(256, 512, 1)      - P5/32-large
            22: Detect([15, 18, 21])
    """

    def __init__(self, nc=80, weights_path=None, verbose=True):
        """参数: nc: 类别数 weights_path: 预训练权重路径 (可选) verbose: 是否打印加载信息.
        """
        super().__init__()
        self.nc = nc
        self.verbose = verbose
        self.yaml = {"nc": nc}

        # YOLOv8s 缩放参数
        depth_multiple = 0.33
        width_multiple = 0.50
        max_channels = 1024

        # 计算实际通道数和重复次数
        def make_round(n):
            """计算实际重复次数."""
            return max(round(n * depth_multiple), 1) if n > 1 else n

        def make_channels(c):
            """计算实际通道数."""
            return min(make_divisible(c * width_multiple, 8), max_channels)

        # 构建模型
        self.model = self._build_model(nc, make_round, make_channels)
        self.stride = None

        # 初始化权重
        self._initialize_weights()

        # 加载预训练权重
        if weights_path:
            self.load_weights(weights_path)

    def _build_model(self, nc, make_round, make_channels):
        """构建 YOLOv8s 模型.

        返回 nn.ModuleList 以确保 state_dict 命名为 model.{index}.xxx
        """
        layers = []

        # ============ Backbone ============
        # Layer 0: Conv(3, 32, 3, 2) - P1/2
        layers.append(Conv(3, make_channels(64), 3, 2))

        # Layer 1: Conv(32, 64, 3, 2) - P2/4
        layers.append(Conv(make_channels(64), make_channels(128), 3, 2))

        # Layer 2: C2f(64, 64, 3, shortcut=True)
        layers.append(C2f(make_channels(128), make_channels(128), make_round(3), shortcut=True))

        # Layer 3: Conv(64, 128, 3, 2) - P3/8
        layers.append(Conv(make_channels(128), make_channels(256), 3, 2))

        # Layer 4: C2f(128, 128, 6, shortcut=True)
        layers.append(C2f(make_channels(256), make_channels(256), make_round(6), shortcut=True))

        # Layer 5: Conv(128, 256, 3, 2) - P4/16
        layers.append(Conv(make_channels(256), make_channels(512), 3, 2))

        # Layer 6: C2f(256, 256, 6, shortcut=True)
        layers.append(C2f(make_channels(512), make_channels(512), make_round(6), shortcut=True))

        # Layer 7: Conv(256, 512, 3, 2) - P5/32
        layers.append(Conv(make_channels(512), make_channels(1024), 3, 2))

        # Layer 8: C2f(512, 512, 3, shortcut=True)
        layers.append(C2f(make_channels(1024), make_channels(1024), make_round(3), shortcut=True))

        # Layer 9: SPPF(512, 512, 5)
        layers.append(SPPF(make_channels(1024), make_channels(1024), 5))

        # ============ Head ============
        # 保存各层输出通道数用于后续计算
        # Backbone outputs (after width_multiple scaling):
        # Layer 4: C2f outputs make_channels(256) = 128
        # Layer 6: C2f outputs make_channels(512) = 256
        # Layer 9: SPPF outputs make_channels(1024) = 512
        ch4 = make_channels(256)  # 128
        ch6 = make_channels(512)  # 256
        ch9 = make_channels(1024)  # 512

        # Layer 10: Upsample(2x) - input from layer 9 (512ch)
        layers.append(nn.Upsample(None, 2, "nearest"))

        # Layer 11: Concat([10, 6]) - 512 + 256 = 768
        layers.append(Concat(1))

        # Layer 12: C2f(768, 256, 3) - config says C2f [512] meaning output=512*0.5=256
        ch12 = make_channels(512)  # 256 (output of this layer)
        layers.append(C2f(ch9 + ch6, ch12, make_round(3)))

        # Layer 13: Upsample(2x) - input from layer 12 (256ch)
        layers.append(nn.Upsample(None, 2, "nearest"))

        # Layer 14: Concat([13, 4]) - 256 + 128 = 384
        layers.append(Concat(1))

        # Layer 15: C2f(384, 128, 3) - P3/8-small, config says C2f [256] meaning output=256*0.5=128
        ch15 = make_channels(256)  # 128 (output of this layer)
        layers.append(C2f(ch12 + ch4, ch15, make_round(3)))

        # Layer 16: Conv(128, 128, 3, 2) - config says Conv [256, 3, 2] meaning output=256*0.5=128
        ch16 = make_channels(256)  # 128
        layers.append(Conv(ch15, ch16, 3, 2))

        # Layer 17: Concat([16, 12]) - 128 + 256 = 384
        layers.append(Concat(1))

        # Layer 18: C2f(384, 256, 3) - P4/16-medium, config says C2f [512] meaning output=512*0.5=256
        ch18 = make_channels(512)  # 256
        layers.append(C2f(ch16 + ch12, ch18, make_round(3)))

        # Layer 19: Conv(256, 256, 3, 2) - config says Conv [512, 3, 2] meaning output=512*0.5=256
        ch19 = make_channels(512)  # 256
        layers.append(Conv(ch18, ch19, 3, 2))

        # Layer 20: Concat([19, 9]) - 256 + 512 = 768
        layers.append(Concat(1))

        # Layer 21: C2f(768, 512, 3) - P5/32-large, config says C2f [1024] meaning output=1024*0.5=512
        ch21 = make_channels(1024)  # 512
        layers.append(C2f(ch19 + ch9, ch21, make_round(3)))

        # Layer 22: Detect([15, 18, 21]) - P3: 128ch, P4: 256ch, P5: 512ch
        layers.append(Detect(nc=nc, ch=(ch15, ch18, ch21)))

        return nn.ModuleList(layers)

    def _initialize_weights(self):
        """初始化模型权重."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _initialize_strides(self):
        """初始化检测头的 stride 通过一次前向传播计算各检测层的步长.

        注意: 必须在 eval 模式下运行，且禁用梯度计算
        以防止更新 BatchNorm 的 running statistics
        """
        # 保存当前训练状态
        was_training = self.training

        # 切换到 eval 模式 (防止 BatchNorm 更新统计量)
        self.eval()

        # 创建一个 dummy 输入来计算 stride
        dummy = torch.zeros(1, 3, 640, 640)

        # 保存各层的输出尺寸
        saves = {}
        strides = []

        # 禁用梯度计算，防止 BatchNorm 更新
        with torch.no_grad():
            for i, m in enumerate(self.model):
                if i == 11:
                    dummy = m([dummy, saves[6]])
                elif i == 14:
                    dummy = m([dummy, saves[4]])
                elif i == 17:
                    dummy = m([dummy, saves[12]])
                elif i == 20:
                    dummy = m([dummy, saves[9]])
                elif i == 22:
                    # Detect 层 - 设置 stride
                    detect = m
                    # 计算每个检测层的 stride
                    # P3: 640/80 = 8, P4: 640/40 = 16, P5: 640/20 = 32
                    for j, feat in enumerate([saves[15], saves[18], saves[21]]):
                        stride = 640 / feat.shape[2]
                        strides.append(stride)
                    detect.stride = torch.tensor(strides)
                    self.stride = detect.stride
                    break
                else:
                    dummy = m(dummy)

                if i in [4, 6, 9, 12, 15, 18, 21]:
                    saves[i] = dummy

        # 初始化检测头的 bias
        self.model[22].bias_init()

        # 恢复训练状态
        if was_training:
            self.train()

    def forward(self, x):
        """前向传播.

        参数:
            x: 输入图像张量 (B, 3, H, W)

        返回:
            训练模式: 字典包含 boxes 和 scores
            推理模式: (y, preds) 元组
        """
        # 保存中间层输出用于跳跃连接
        saves = {}

        for i, m in enumerate(self.model):
            if i == 11:  # Concat([10, 6])
                x = m([x, saves[6]])
            elif i == 14:  # Concat([13, 4])
                x = m([x, saves[4]])
            elif i == 17:  # Concat([16, 12])
                x = m([x, saves[12]])
            elif i == 20:  # Concat([19, 9])
                x = m([x, saves[9]])
            elif i == 22:  # Detect([15, 18, 21])
                x = m([saves[15], saves[18], saves[21]])
            else:
                x = m(x)

            # 保存特定层输出用于后续跳跃连接
            if i in [4, 6, 9, 12, 15, 18, 21]:
                saves[i] = x

        return x

    def load_weights(self, weights_path, strict=False):
        """加载预训练权重.

        参数:
            weights_path: 权重文件路径
            strict: 是否严格匹配所有键

        返回:
            匹配的权重数量
        """
        if self.verbose:
            print(f"正在加载权重: {weights_path}")

        # 加载检查点 (weights_only=False 以支持包含自定义类的检查点)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

        # 获取 state_dict
        # 注意: 必须先调用 .float() 将 float16 转换为 float32
        # 否则 state_dict() 返回的 float16 值会导致精度问题
        if "model" in ckpt:
            if hasattr(ckpt["model"], "float"):
                state_dict = ckpt["model"].float().state_dict()
            elif hasattr(ckpt["model"], "state_dict"):
                state_dict = ckpt["model"].state_dict()
            else:
                state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # 过滤掉不匹配的键
        model_dict = self.state_dict()
        matched_dict = {}
        mismatched_shapes = []
        missing_keys = []
        unexpected_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched_dict[k] = v
                else:
                    mismatched_shapes.append((k, v.shape, model_dict[k].shape))
            else:
                unexpected_keys.append(k)

        for k in model_dict:
            if k not in state_dict:
                missing_keys.append(k)

        # 加载权重
        self.load_state_dict(matched_dict, strict=False)

        # 初始化 stride
        self._initialize_strides()

        if self.verbose:
            total_pretrained = len(state_dict)
            total_matched = len(matched_dict)
            print(f"已加载 {total_matched}/{total_pretrained} 个权重参数")

            if mismatched_shapes:
                print(f"  形状不匹配: {len(mismatched_shapes)} 个")
                for k, s1, s2 in mismatched_shapes[:5]:
                    print(f"    {k}: {s1} vs {s2}")
                if len(mismatched_shapes) > 5:
                    print(f"    ... 还有 {len(mismatched_shapes) - 5} 个")

            if missing_keys:
                print(f"  缺失键: {len(missing_keys)} 个")
                if len(missing_keys) <= 10:
                    for k in missing_keys:
                        print(f"    {k}")

        return len(matched_dict)

    def save_weights(self, save_path):
        """保存模型权重.

        参数:
            save_path: 保存路径
        """
        torch.save({"model": self.state_dict()}, save_path)
        if self.verbose:
            print(f"权重已保存到: {save_path}")

    def info(self, detailed=False, verbose=True):
        """打印模型信息."""
        n_p = sum(x.numel() for x in self.parameters())  # 参数数量
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)  # 可训练参数
        n_l = len(self.model)  # 层数

        if verbose:
            print("模型: YOLOv8s")
            print(f"  层数: {n_l}")
            print(f"  参数量: {n_p:,}")
            print(f"  可训练参数: {n_g:,}")
            print(f"  类别数: {self.nc}")

        if detailed:
            for i, m in enumerate(self.model):
                n_params = sum(x.numel() for x in m.parameters())
                print(f"  {i:2d}: {m.__class__.__name__:15s} - {n_params:>10,} params")

        return n_l, n_p, n_g


def create_model(nc=80, weights_path=None, verbose=True):
    """创建 YOLOv8s 模型的便捷函数.

    参数:
        nc: 类别数
        weights_path: 预训练权重路径 (可选)
        verbose: 是否打印信息

    返回:
        YOLOv8s 模型实例
    """
    return YOLOv8s(nc=nc, weights_path=weights_path, verbose=verbose)
