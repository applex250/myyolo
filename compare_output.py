"""
YOLOv8s 输出一致性验证测试
对比自定义实现与官方 ultralytics 实现的输出是否完全一致

测试场景:
1. 张量级对比 - 直接对比模型原始输出
2. 图像级对比 - 使用真实图像验证检测结果

验证标准:
- 张量输出最大差异: < 1e-5
- 检测框数量: 完全一致
- 边界框 IoU: > 0.95
- 类别分数差异: < 0.001
- 检测类别: 完全一致
"""

import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
import glob
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入自定义模型
from my_yolov8s import YOLOv8s
from my_yolov8s.utils import non_max_suppression, letterbox, scale_boxes

# COCO 评估工具 (可选依赖)
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


class COCOEvaluator:
    """COCO 数据集评估器"""

    # YOLO cls (0-79) to COCO category_id (1-90, non-contiguous)
    # COCO 类别 ID 是 1-90（不连续，跳过了一些 ID），而 YOLOv8 使用 0-79（连续）
    YOLO_TO_COCO_CAT = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]

    def __init__(self, annotation_path):
        """
        初始化 COCO 评估器

        参数:
            annotation_path: instances_val2017.json 路径
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("pycocotools 未安装，请运行: pip install pycocotools")
        self.coco_gt = COCO(annotation_path)
        self.image_ids = []

    def convert_predictions_to_coco_format(self, detections, image_id):
        """
        将检测结果转换为 COCO 格式

        参数:
            detections: (N, 6) 检测结果 [x1, y1, x2, y2, conf, cls]
            image_id: 图像 ID

        返回:
            COCO 格式的检测结果列表
        """
        coco_results = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            coco_results.append({
                'image_id': int(image_id),
                'category_id': self.YOLO_TO_COCO_CAT[int(cls)],  # 使用 YOLO -> COCO 映射表
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # xywh 格式
                'score': float(conf)
            })
        return coco_results

    def evaluate(self, predictions_dict):
        """
        评估检测结果

        参数:
            predictions_dict: {image_id: detections} 字典

        返回:
            评估结果字典
        """
        # 收集所有预测结果
        all_predictions = []
        for image_id, detections in predictions_dict.items():
            all_predictions.extend(
                self.convert_predictions_to_coco_format(detections, image_id)
            )

        # 加载预测结果
        coco_dt = self.coco_gt.loadRes(all_predictions)

        # 运行评估
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = sorted(predictions_dict.keys())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 提取关键指标 (COCOeval.stats 共 12 个元素, 索引 0-11)
        stats = coco_eval.stats

        return {
            'mAP@0.5:0.95': float(stats[0]),  # AP @ IoU=0.50:0.95 all
            'mAP@0.5': float(stats[1]),        # AP @ IoU=0.50 all
            'AP@0.75': float(stats[2]),        # AP @ IoU=0.75 all
            'AP_small': float(stats[3]),       # AP @ IoU=0.50:0.95 small
            'AP_medium': float(stats[4]),      # AP @ IoU=0.50:0.95 medium
            'AP_large': float(stats[5]),       # AP @ IoU=0.50:0.95 large
            'AR@1': float(stats[6]),           # AR maxDets=1
            'AR@10': float(stats[7]),          # AR maxDets=10
            'AR@100': float(stats[8]),         # AR maxDets=100
        }


class OutputComparator:
    """输出对比器"""

    def __init__(self, weights_path='yolov8s.pt', verbose=True):
        """
        初始化对比器

        参数:
            weights_path: 预训练权重路径
            verbose: 是否打印详细信息
        """
        self.weights_path = weights_path
        self.verbose = verbose
        self.official_model = None
        self.my_model = None

    def load_models(self):
        """加载官方模型和自定义模型"""
        if self.verbose:
            print("=" * 60)
            print("加载模型...")
            print("=" * 60)

        # 加载官方模型
        if self.verbose:
            print("\n[1/2] 加载官方 ultralytics 模型...")
        from ultralytics import YOLO
        official_yolo = YOLO(self.weights_path)
        self.official_model = official_yolo.model
        self.official_model.eval()

        # 加载自定义模型
        if self.verbose:
            print("\n[2/2] 加载自定义 my_yolov8s 模型...")
        self.my_model = YOLOv8s(nc=80, weights_path=self.weights_path, verbose=self.verbose)
        self.my_model.eval()

        if self.verbose:
            print("\n✓ 模型加载完成\n")

    def compare_tensor_output(self, input_size=(640, 640), seed=42):
        """
        场景1: 张量级对比
        直接对比模型原始输出，验证数值完全一致

        参数:
            input_size: 输入尺寸
            seed: 随机种子

        返回:
            dict: 对比结果
        """
        print("=" * 60)
        print("场景1: 张量级对比 (底层输出)")
        print("=" * 60)

        # 固定随机种子
        torch.manual_seed(seed)

        # 生成随机输入
        x = torch.randn(1, 3, input_size[0], input_size[1])
        print(f"\n输入张量形状: {x.shape}")
        print(f"随机种子: {seed}")

        # 官方模型推理
        print("\n官方模型推理...")
        with torch.no_grad():
            official_out = self.official_model(x)

        # 自定义模型推理
        print("自定义模型推理...")
        with torch.no_grad():
            my_out = self.my_model(x)

        # 提取主要输出张量
        # 官方模型输出可能是元组或单个张量
        if isinstance(official_out, tuple):
            official_tensor = official_out[0]
        else:
            official_tensor = official_out

        # 自定义模型输出是元组 (y, preds)
        if isinstance(my_out, tuple):
            my_tensor = my_out[0]
        else:
            my_tensor = my_out

        print(f"\n官方模型输出形状: {official_tensor.shape}")
        print(f"自定义模型输出形状: {my_tensor.shape}")

        # 数值对比
        diff = torch.abs(official_tensor - my_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n数值对比结果:")
        print(f"  最大差异: {max_diff:.10f}")
        print(f"  平均差异: {mean_diff:.10f}")
        print(f"  通过标准: < 1e-5")

        # 检查是否通过
        passed = max_diff < 1e-5
        status = "✓ 通过" if passed else "✗ 未通过"

        print(f"\n结果: {status}")

        return {
            'test': 'tensor_comparison',
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'passed': passed,
            'official_shape': official_tensor.shape,
            'my_shape': my_tensor.shape
        }

    def preprocess_image(self, img_path, img_size=640):
        """
        预处理图像用于推理

        参数:
            img_path: 图像路径
            img_size: 目标尺寸

        返回:
            tensor: 预处理后的张量
            original_img: 原始图像
            original_shape: 原始尺寸
        """
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        original_img = img.copy()
        original_shape = img.shape[:2]

        # Letterbox
        img, ratio, pad = letterbox(img, new_shape=img_size)

        # BGR -> RGB, HWC -> CHW
        img = img[:, :, ::-1].transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        # 转换为 tensor
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)

        return tensor, original_img, original_shape

    def get_official_detections(self, img_tensor, conf_thres=0.25, iou_thres=0.45):
        """
        获取官方模型的检测结果

        返回:
            detections: (N, 6) 张量 [x1, y1, x2, y2, conf, cls]
        """
        with torch.no_grad():
            # 官方模型原始输出
            out = self.official_model(img_tensor)
            if isinstance(out, tuple):
                out = out[0]

        # 使用自定义 NMS 以确保对比公平
        detections = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, nc=80)

        return detections[0]  # 返回第一张图的检测结果

    def get_my_detections(self, img_tensor, conf_thres=0.25, iou_thres=0.45):
        """
        获取自定义模型的检测结果

        返回:
            detections: (N, 6) 张量 [x1, y1, x2, y2, conf, cls]
        """
        with torch.no_grad():
            out = self.my_model(img_tensor)
            if isinstance(out, tuple):
                out = out[0]

        # NMS
        detections = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, nc=80)

        return detections[0]  # 返回第一张图的检测结果

    def compare_single_image(self, img_path, conf_thres=0.25, iou_thres=0.45):
        """
        场景2: 单张图像对比
        使用真实图像验证检测结果一致

        参数:
            img_path: 图像路径
            conf_thres: 置信度阈值
            iou_thres: IoU 阈值

        返回:
            dict: 对比结果
        """
        print(f"\n{'='*60}")
        print(f"测试图像: {Path(img_path).name}")
        print(f"{'='*60}")

        # 预处理图像
        img_tensor, original_img, original_shape = self.preprocess_image(img_path)
        print(f"原始图像尺寸: {original_shape}")
        print(f"输入张量尺寸: {img_tensor.shape}")

        # 获取检测结果
        official_det = self.get_official_detections(img_tensor, conf_thres, iou_thres)
        my_det = self.get_my_detections(img_tensor, conf_thres, iou_thres)

        # 统计检测数量
        official_count = official_det.shape[0]
        my_count = my_det.shape[0]

        print(f"\n检测数量:")
        print(f"  官方模型: {official_count}")
        print(f"  自定义模型: {my_count}")

        # 初始化结果
        result = {
            'image': Path(img_path).name,
            'official_count': official_count,
            'my_count': my_count,
            'count_match': official_count == my_count,
            'bbox_iou_avg': 0.0,
            'class_match_rate': 0.0,
            'score_diff_avg': 0.0,
            'passed': False
        }

        if official_count == 0 and my_count == 0:
            print("\n两个模型都没有检测到目标")
            result['passed'] = True
            return result

        if official_count == 0 or my_count == 0:
            print("\n检测数量不匹配，无法对比")
            result['passed'] = False
            return result

        # 匹配检测结果
        # 使用贪婪匹配：按置信度排序后依次匹配
        match_results = self._match_detections(official_det, my_det)

        if match_results:
            result['bbox_iou_avg'] = np.mean([m['iou'] for m in match_results])
            result['class_match_rate'] = np.mean([m['class_match'] for m in match_results])
            result['score_diff_avg'] = np.mean([m['score_diff'] for m in match_results])

            print(f"\n详细对比 (匹配 {len(match_results)} 对检测框):")
            print(f"  边界框平均 IoU: {result['bbox_iou_avg']:.4f} (标准: > 0.95)")
            print(f"  类别匹配率: {result['class_match_rate']*100:.1f}% (标准: 100%)")
            print(f"  分数平均差异: {result['score_diff_avg']:.6f} (标准: < 0.001)")

        # 判断是否通过
        passed = (
            result['count_match'] and
            result['bbox_iou_avg'] > 0.95 and
            result['class_match_rate'] == 1.0 and
            result['score_diff_avg'] < 0.001
        )
        result['passed'] = passed

        print(f"\n结果: {'✓ 通过' if passed else '✗ 未通过'}")

        return result

    def _match_detections(self, official_det, my_det):
        """
        匹配两个模型的检测结果

        使用贪婪匹配策略：按 IoU 从高到低匹配

        参数:
            official_det: 官方模型检测结果 (N, 6)
            my_det: 自定义模型检测结果 (M, 6)

        返回:
            list: 匹配结果列表
        """
        from my_yolov8s.utils import box_iou

        # 计算所有检测框之间的 IoU
        iou_matrix = box_iou(official_det[:, :4], my_det[:, :4])

        # 贪婪匹配
        matches = []
        used_official = set()
        used_my = set()

        while True:
            # 找到最大 IoU
            max_iou, max_idx = iou_matrix.max(), iou_matrix.argmax()
            if max_iou < 0.3:  # IoU 阈值
                break

            official_idx = max_idx // iou_matrix.shape[1]
            my_idx = max_idx % iou_matrix.shape[1]

            if official_idx in used_official or my_idx in used_my:
                # 标记为已使用
                iou_matrix[official_idx, my_idx] = 0
                continue

            # 记录匹配
            matches.append({
                'official_idx': official_idx.item(),
                'my_idx': my_idx.item(),
                'iou': max_iou.item(),
                'class_match': official_det[official_idx, 5].item() == my_det[my_idx, 5].item(),
                'score_diff': abs(official_det[official_idx, 4].item() - my_det[my_idx, 4].item())
            })

            used_official.add(official_idx.item())
            used_my.add(my_idx.item())

            # 标记为已使用
            iou_matrix[official_idx, my_idx] = 0

        return matches

    def run_all_tests(self, test_images=None, num_coco_images=100,
                      use_coco_eval=True, verbose=False):
        """
        运行所有测试

        参数:
            test_images: 测试图像路径列表 (None 则使用 COCO)
            num_coco_images: COCO 数据集测试图片数量
            use_coco_eval: 是否使用 COCO mAP 评估
            verbose: 是否详细输出每张图片结果
        """
        # 加载模型
        self.load_models()

        results = {
            'tensor_test': None,
            'image_tests': [],
            'coco_eval': None
        }

        # 场景1: 张量级对比
        results['tensor_test'] = self.compare_tensor_output()

        # 场景2: 图像级对比
        print("\n" + "=" * 60)
        print("场景2: 图像级对比 (COCO val2017)")
        print("=" * 60)

        # COCO 数据集路径
        coco_img_dir = Path('F:/coco17/val2017')
        coco_anno_file = Path('F:/coco17/annotations/instances_val2017.json')

        # 如果没有指定测试图片，使用 COCO 数据集
        if test_images is None:
            if coco_img_dir.exists():
                test_images = sorted(list(coco_img_dir.glob('*.jpg')))[:num_coco_images]
                print(f"使用 COCO val2017 数据集: {len(test_images)} 张图片")
            else:
                # 回退到内置图片
                assets_dir = Path(__file__).parent / 'ultralytics' / 'assets'
                test_images = [
                    str(assets_dir / 'bus.jpg'),
                    str(assets_dir / 'zidane.jpg')
                ]
                print(f"COCO 目录不存在，使用内置测试图片")

        # 存储预测结果用于 COCO 评估
        official_predictions = {}
        my_predictions = {}

        # 处理每张图片
        for i, img_path in enumerate(test_images):
            if not verbose and i > 0 and i % 10 == 0:
                print(f"进度: {i}/{len(test_images)}")

            if os.path.exists(img_path):
                # 获取检测结果
                img_tensor, _, original_shape = self.preprocess_image(str(img_path))
                official_det = self.get_official_detections(img_tensor)
                my_det = self.get_my_detections(img_tensor)

                # 将预测坐标从 letterbox (640x640) 缩放回原始图像尺寸
                if official_det.shape[0]:
                    official_det[:, :4] = scale_boxes((640, 640), official_det[:, :4], original_shape)
                if my_det.shape[0]:
                    my_det[:, :4] = scale_boxes((640, 640), my_det[:, :4], original_shape)

                # 从文件名提取 image_id (COCO 格式: 000000000139.jpg)
                image_id = int(Path(img_path).stem)

                # 存储预测结果
                official_predictions[image_id] = official_det
                my_predictions[image_id] = my_det

                if verbose:
                    result = self.compare_single_image(str(img_path), verbose=verbose)
                    results['image_tests'].append(result)
            else:
                print(f"\n警告: 图像不存在 {img_path}")

        # COCO mAP 评估
        if use_coco_eval and coco_anno_file.exists():
            print("\n" + "=" * 60)
            print("COCO mAP 评估")
            print("=" * 60)

            try:
                evaluator = COCOEvaluator(str(coco_anno_file))

                # 评估官方模型
                print("\n官方模型评估...")
                official_metrics = evaluator.evaluate(official_predictions)

                # 评估自定义模型
                print("自定义模型评估...")
                my_metrics = evaluator.evaluate(my_predictions)

                # 对比结果
                print("\n" + "=" * 60)
                print("COCO 评估结果对比")
                print("=" * 60)
                print(f"{'指标':<20} {'官方模型':>12} {'自定义模型':>12} {'差异':>12}")
                print('-' * 60)
                for key in ['mAP@0.5:0.95', 'mAP@0.5', 'AP@0.75', 'AP_small', 'AP_medium', 'AP_large', 'AR@100']:
                    off_val = official_metrics[key]
                    my_val = my_metrics[key]
                    diff = abs(off_val - my_val)
                    print(f"{key:<20} {off_val:>12.4f} {my_val:>12.4f} {diff:>12.6f}")

                results['coco_eval'] = {
                    'official': official_metrics,
                    'my': my_metrics
                }
            except Exception as e:
                print(f"\n警告: COCO 评估失败 - {e}")
        elif use_coco_eval and not coco_anno_file.exists():
            print(f"\n警告: COCO 标注文件不存在: {coco_anno_file}")

        # 汇总结果
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """打印测试汇总"""
        print("\n" + "=" * 60)
        print("测试汇总")
        print("=" * 60)

        # 张量测试
        tensor_passed = results['tensor_test']['passed']
        print(f"\n1. 张量级对比: {'✓ 通过' if tensor_passed else '✗ 未通过'}")
        print(f"   最大差异: {results['tensor_test']['max_diff']:.10f}")

        # 图像测试
        print(f"\n2. 图像级对比:")
        total = len(results['image_tests'])
        if total > 0:
            passed = sum(1 for r in results['image_tests'] if r['passed'])
            print(f"   通过: {passed}/{total}")

            for r in results['image_tests']:
                status = '✓' if r['passed'] else '✗'
                print(f"   {status} {r['image']}: 检测数 {r['official_count']}/{r['my_count']}, "
                      f"IoU {r['bbox_iou_avg']:.4f}, 分数差异 {r['score_diff_avg']:.6f}")
        else:
            print("   (使用 COCO 评估模式，无详细单图结果)")

        # COCO 评估结果
        if results.get('coco_eval'):
            print(f"\n3. COCO mAP 评估:")
            official = results['coco_eval']['official']
            my = results['coco_eval']['my']
            print(f"   {'指标':<18} {'官方':>10} {'自定义':>10} {'差异':>10}")
            print(f"   {'-'*50}")
            for key in ['mAP@0.5', 'mAP@0.5:0.95']:
                diff = abs(official[key] - my[key])
                match = '✓' if diff < 0.001 else '✗'
                print(f"   {match} {key:<16} {official[key]:>10.4f} {my[key]:>10.4f} {diff:>10.6f}")

        # 总体结果
        all_passed = tensor_passed and all(r['passed'] for r in results['image_tests'])
        print(f"\n{'='*60}")
        print(f"总体结果: {'✓ 全部通过' if all_passed else '✗ 存在未通过的测试'}")
        print(f"{'='*60}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8s 输出一致性验证')
    parser.add_argument('--num-images', type=int, default=100,
                       help='COCO 数据集测试图片数量 (默认: 100)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出每张图片结果')
    parser.add_argument('--no-coco-eval', action='store_true',
                       help='禁用 COCO mAP 评估')
    parser.add_argument('--images', type=str, nargs='+',
                       help='指定测试图片路径 (覆盖 COCO)')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8s 输出一致性验证测试 (COCO val2017)")
    print("=" * 60)
    print("\n验证标准:")
    print("  - 张量输出最大差异: < 1e-5")
    print("  - 检测框数量: 完全一致")
    print("  - 边界框 IoU: > 0.95")
    print("  - 类别分数差异: < 0.001")
    print("  - 检测类别: 完全一致")

    # 创建对比器并运行测试
    comparator = OutputComparator(weights_path='yolov8s.pt', verbose=True)
    results = comparator.run_all_tests(
        test_images=args.images,
        num_coco_images=args.num_images,
        use_coco_eval=not args.no_coco_eval,
        verbose=args.verbose
    )

    return results


if __name__ == '__main__':
    results = main()
