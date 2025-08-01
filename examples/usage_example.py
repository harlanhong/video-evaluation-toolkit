#!/usr/bin/env python3
"""
使用示例：综合视频指标计算器和LSE计算器 (VBench集成版本)

展示如何使用新的评估工具

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This example demonstrates usage of the VBench-integrated video evaluation toolkit.
"""

import sys
import os

# 添加evalutation包到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalutation.lse_calculator import LSECalculator
from evalutation.metrics_calculator import VideoMetricsCalculator


def example_lse_calculator():
    """LSE计算器使用示例"""
    print("=" * 60)
    print("🎵 LSE计算器使用示例")
    print("=" * 60)
    
    # 初始化LSE计算器
    lse_calc = LSECalculator()
    
    # 示例视频路径（请替换为实际路径）
    video_path = "path/to/your/video.mp4"
    
    if os.path.exists(video_path):
        # 计算单个视频的LSE
        print(f"📹 计算视频LSE: {video_path}")
        lse_d, lse_c = lse_calc.calculate_single_video(video_path)
        
        if lse_d is not None and lse_c is not None:
            print(f"✅ LSE结果:")
            print(f"   距离: {lse_d:.4f}")
            print(f"   置信度: {lse_c:.4f}")
        else:
            print("❌ LSE计算失败")
    else:
        print(f"⚠️ 视频文件不存在: {video_path}")
        print("请修改video_path为实际的视频文件路径")


def example_batch_lse():
    """批量LSE计算示例"""
    print("\n" + "=" * 60)
    print("📂 批量LSE计算示例")
    print("=" * 60)
    
    # 初始化LSE计算器
    lse_calc = LSECalculator()
    
    # 示例视频列表（请替换为实际路径）
    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        "path/to/video3.mp4"
    ]
    
    # 过滤存在的视频文件
    existing_videos = [v for v in video_paths if os.path.exists(v)]
    
    if existing_videos:
        print(f"📹 计算 {len(existing_videos)} 个视频的LSE")
        results = lse_calc.calculate_batch(existing_videos)
        
        print(f"\n📊 批量计算结果:")
        for video_path, (lse_d, lse_c) in results.items():
            video_name = os.path.basename(video_path)
            if lse_d is not None:
                print(f"✅ {video_name}: LSE-D={lse_d:.4f}, LSE-C={lse_c:.4f}")
            else:
                print(f"❌ {video_name}: 计算失败")
    else:
        print("⚠️ 没有找到有效的视频文件")
        print("请修改video_paths为实际的视频文件路径")


def example_metrics_calculator():
    """综合指标计算器使用示例"""
    print("\n" + "=" * 60)
    print("📊 综合指标计算器使用示例")
    print("=" * 60)
    
    # 初始化指标计算器
    metrics_calc = VideoMetricsCalculator()
    
    # 示例路径（请替换为实际路径）
    pred_video = "path/to/prediction.mp4"
    gt_video = "path/to/ground_truth.mp4"  # 可选
    
    if os.path.exists(pred_video):
        print(f"📹 计算视频指标: {pred_video}")
        
        # 只计算预测视频指标（不需要真值）
        metrics = metrics_calc.calculate_video_metrics(pred_video)
        
        print(f"\n📊 指标结果:")
        print(f"   帧数: {metrics['frame_count']}")
        print(f"   分辨率: {metrics['width']}x{metrics['height']}")
        print(f"   帧率: {metrics['fps']:.2f}")
        print(f"   人脸检测率: {metrics['face_detection_rate']:.2%}")
        print(f"   清晰度分数: {metrics['sharpness_score']:.2f}")
        
        if metrics['lse_distance'] is not None:
            print(f"   LSE距离: {metrics['lse_distance']:.4f}")
            print(f"   LSE置信度: {metrics['lse_confidence']:.4f}")
        else:
            print(f"   LSE: 计算失败")
        
        # 如果有真值，计算对比指标
        if os.path.exists(gt_video):
            print(f"\n🔍 计算对比指标...")
            metrics_with_gt = metrics_calc.calculate_video_metrics(pred_video, gt_video)
            
            if metrics_with_gt['face_psnr'] is not None:
                print(f"   人脸PSNR: {metrics_with_gt['face_psnr']:.2f}")
                print(f"   人脸SSIM: {metrics_with_gt['face_ssim']:.3f}")
                print(f"   人脸LPIPS: {metrics_with_gt['face_lpips']:.3f}")
            else:
                print(f"   对比指标: 计算失败")
    else:
        print(f"⚠️ 预测视频文件不存在: {pred_video}")
        print("请修改pred_video为实际的视频文件路径")


def example_batch_metrics():
    """批量指标计算示例"""
    print("\n" + "=" * 60)
    print("📂 批量指标计算示例")
    print("=" * 60)
    
    # 初始化指标计算器
    metrics_calc = VideoMetricsCalculator()
    
    # 示例目录（请替换为实际路径）
    pred_dir = "path/to/predictions"
    gt_dir = "path/to/ground_truth"  # 可选
    
    if os.path.exists(pred_dir):
        print(f"📂 批量计算目录: {pred_dir}")
        
        # 批量计算
        results = metrics_calc.calculate_batch_metrics(
            pred_dir=pred_dir,
            gt_dir=gt_dir if os.path.exists(gt_dir) else None,
            pattern="*.mp4"
        )
        
        if results:
            # 保存结果
            output_path = "batch_metrics_results.json"
            metrics_calc.save_results(results, output_path)
            
            # 打印统计信息
            metrics_calc.print_summary_stats(results)
        else:
            print("❌ 没有计算出任何结果")
    else:
        print(f"⚠️ 预测目录不存在: {pred_dir}")
        print("请修改pred_dir为实际的目录路径")


def main():
    """主函数"""
    print("🚀 视频评估工具使用示例")
    print("请根据需要修改示例中的文件路径")
    
    # 运行各种示例
    example_lse_calculator()
    example_batch_lse()
    example_metrics_calculator()
    example_batch_metrics()
    
    print("\n" + "=" * 60)
    print("✅ 示例演示完成！")
    print("\n💡 提示:")
    print("1. 请将示例中的路径替换为实际的视频文件或目录路径")
    print("2. LSE计算不需要真值视频或外部音频文件")
    print("3. 图像质量指标(PSNR/SSIM/LPIPS)需要真值视频")
    print("4. 其他指标（亮度、对比度、人脸检测等）不需要真值")
    print("=" * 60)


if __name__ == "__main__":
    main() 