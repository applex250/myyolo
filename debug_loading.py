"""
调试脚本：检查权重加载过程
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from my_yolov8s import YOLOv8s

def debug_weight_loading():
    """详细检查权重加载过程"""
    print("=" * 60)
    print("调试权重加载过程")
    print("=" * 60)

    # 加载权重文件
    ckpt = torch.load('yolov8s.pt', map_location='cpu', weights_only=False)

    # 获取官方 state_dict
    if 'model' in ckpt:
        if hasattr(ckpt['model'], 'state_dict'):
            state_dict = ckpt['model'].state_dict()
        else:
            state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # 创建自定义模型（不加载权重）
    my_model = YOLOv8s(nc=80, weights_path=None, verbose=False)
    model_dict = my_model.state_dict()

    # 检查 running_mean 和 running_var 是否在两个 dict 中
    print("\n检查 running_mean 和 running_var 是否存在:")
    bn_keys = ['model.0.bn.running_mean', 'model.0.bn.running_var']
    for key in bn_keys:
        in_official = key in state_dict
        in_custom = key in model_dict
        print(f"  {key}:")
        print(f"    在官方 state_dict 中: {in_official}")
        print(f"    在自定义 state_dict 中: {in_custom}")

        if in_official and in_custom:
            official_shape = state_dict[key].shape
            custom_shape = model_dict[key].shape
            print(f"    官方形状: {official_shape}")
            print(f"    自定义形状: {custom_shape}")
            print(f"    形状匹配: {official_shape == custom_shape}")

    # 检查匹配逻辑
    print("\n检查匹配逻辑:")
    matched_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v

    for key in bn_keys:
        if key in matched_dict:
            print(f"  {key}: 已加入 matched_dict")
            print(f"    值: {matched_dict[key][:5]}")
            print(f"    dtype: {matched_dict[key].dtype}")
        else:
            print(f"  {key}: 未加入 matched_dict")

    # 手动加载并检查
    print("\n手动加载权重:")
    my_model.load_state_dict(matched_dict, strict=False)

    for key in bn_keys:
        loaded_val = my_model.state_dict()[key]
        original_val = state_dict[key]
        print(f"  {key}:")
        print(f"    原始值: {original_val[:5]}")
        print(f"    加载后: {loaded_val[:5]}")
        print(f"    差异: {torch.abs(original_val.float() - loaded_val).max().item():.10f}")


if __name__ == '__main__':
    debug_weight_loading()
