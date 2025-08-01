#!/usr/bin/env python3
"""
VBench Direct Integration - 直接集成VBench核心逻辑
直接使用VBench类，确保100%一致性

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides direct integration with VBench for video generation quality assessment.
"""

import torch
import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional
import argparse

# 导入VBench
from vbench import VBench
from vbench.distributed import dist_init, print0


class VBenchDirect:
    """VBench Direct - 直接使用VBench类"""
    
    def __init__(self, device: str = "cuda", cache_dir: str = "./cache"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"🚀 初始化VBench Direct (设备: {self.device})")
        
        # 定义6个核心指标
        self.core_metrics = [
            'subject_consistency',
            'background_consistency', 
            'motion_smoothness',
            'dynamic_degree',
            'aesthetic_quality',
            'imaging_quality'
        ]
        
        # 初始化分布式环境
        dist_init()
        
        # 创建临时的json配置文件
        self.full_json_path = self._create_temp_json()
        
        # 创建输出目录
        self.output_path = tempfile.mkdtemp(prefix="vbench_output_")
        
        # 初始化VBench对象
        self.vbench = VBench(self.device, self.full_json_path, self.output_path)
        
        print(f"✅ VBench对象初始化完成")
    
    def _create_temp_json(self) -> str:
        """创建临时的VBench配置JSON文件"""
        # 简化的VBench配置
        config = {
            "subject_consistency": {},
            "background_consistency": {},
            "motion_smoothness": {},
            "dynamic_degree": {},
            "aesthetic_quality": {},
            "imaging_quality": {}
        }
        
        temp_json = os.path.join(self.cache_dir, "temp_vbench_config.json")
        with open(temp_json, 'w') as f:
            json.dump(config, f, indent=2)
        
        return temp_json
    
    def _prepare_video_directory(self, video_paths: List[str]) -> str:
        """准备视频目录"""
        temp_dir = tempfile.mkdtemp(prefix="vbench_videos_")
        
        print(f"🔄 准备视频目录: {temp_dir}")
        
        for i, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            _, ext = os.path.splitext(video_path)
            dest_name = f"video_{i:03d}{ext}"
            dest_path = os.path.join(temp_dir, dest_name)
            
            print(f"   📄 复制: {os.path.basename(video_path)} -> {dest_name}")
            shutil.copy2(video_path, dest_path)
        
        return temp_dir
    
    def evaluate_videos(self, video_paths: List[str], metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """评估视频，直接调用VBench核心逻辑"""
        if metrics is None:
            metrics = self.core_metrics.copy()
        
        print(f"🔍 开始评估 {len(video_paths)} 个视频，共 {len(metrics)} 个指标")
        print("=" * 60)
        
        temp_video_dir = None
        
        try:
            # 准备视频目录
            temp_video_dir = self._prepare_video_directory(video_paths)
            
            # === 核心VBench逻辑 (从官方evaluate.py移植) ===
            print0(f'🚀 开始VBench评估')
            
            current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            
            kwargs = {}
            prompt = []  # 空列表表示从文件名读取prompt
            
            # 设置参数
            kwargs['imaging_quality_preprocessing_mode'] = 'longer'
            
            # 调用VBench核心评估函数
            print0(f'📊 调用VBench.evaluate...')
            
            self.vbench.evaluate(
                videos_path=temp_video_dir,
                name=f'results_{current_time}',
                prompt_list=prompt,  # 从文件名读取prompt
                dimension_list=metrics,
                local=False,  # 不从本地加载检查点
                read_frame=False,  # 直接读取视频
                mode='custom_input',
                **kwargs
            )
            
            print0('✅ VBench评估完成')
            
            # 解析结果
            results = self._parse_results(current_time)
            
            return results
            
        finally:
            # 清理临时目录
            if temp_video_dir and os.path.exists(temp_video_dir):
                shutil.rmtree(temp_video_dir)
                print(f"🗑️  清理临时视频目录: {temp_video_dir}")
    
    def _parse_results(self, result_name: str) -> Dict[str, float]:
        """解析VBench结果"""
        print(f"📊 解析VBench结果...")
        
        # 查找eval_results文件 (这个文件包含评估结果)
        result_file = None
        
        for file in os.listdir(self.output_path):
            if file.endswith("_eval_results.json") and result_name in file:
                result_file = os.path.join(self.output_path, file)
                break
        
        if result_file is None:
            print(f"⚠️ 未找到eval_results文件在: {self.output_path}")
            print(f"目录内容: {os.listdir(self.output_path)}")
            # 返回默认值
            return {metric: 0.0 for metric in self.core_metrics}
        
        print(f"   📄 结果文件: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)
        
        print(f"   📋 原始结果键: {list(raw_results.keys())}")
        
        # 解析结果
        # VBench格式: {"metric_name": [total_score, detailed_results]}
        results = {}
        
        for metric in self.core_metrics:
            if metric in raw_results:
                metric_data = raw_results[metric]
                
                if isinstance(metric_data, list) and len(metric_data) >= 1:
                    # 取第一个元素作为总分
                    score = metric_data[0]
                    if isinstance(score, (int, float)):
                        results[metric] = float(score)
                        print(f"   ✅ {metric}: {results[metric]}")
                    else:
                        print(f"   ⚠️ {metric} 第一个元素不是数值: {type(score)}")
                        results[metric] = 0.0
                        
                elif isinstance(metric_data, (int, float)):
                    results[metric] = float(metric_data)
                    print(f"   ✅ {metric}: {results[metric]}")
                else:
                    print(f"   ⚠️ 未知的 {metric} 格式: {type(metric_data)}")
                    results[metric] = 0.0
            else:
                print(f"   ❌ 结果中未找到: {metric}")
                results[metric] = 0.0
        
        return results
    
    def save_results(self, results: Dict[str, float], output_path: str = "vbench_direct_results.json"):
        """保存结果到JSON文件"""
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                clean_results[k] = float(v)
            else:
                clean_results[k] = v
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到: {output_path}")
    
    def cleanup(self):
        """清理临时文件"""
        if hasattr(self, 'output_path') and os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            print(f"🗑️  清理输出目录: {self.output_path}")
        
        if hasattr(self, 'full_json_path') and os.path.exists(self.full_json_path):
            os.remove(self.full_json_path)
            print(f"🗑️  清理配置文件: {self.full_json_path}")


def main():
    parser = argparse.ArgumentParser(description="VBench Direct - 直接集成VBench核心逻辑")
    parser.add_argument("--videos", type=str, required=True, help="视频文件路径(用逗号分隔多个文件)")
    parser.add_argument("--metrics", type=str, nargs='+', choices=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'all'], default=['all'], help="要计算的指标")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--output", type=str, default="vbench_direct_results.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 解析视频路径
    video_paths = [path.strip() for path in args.videos.split(',')]
    
    # 验证视频文件存在
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return
    
    # 解析指标
    if 'all' in args.metrics:
        metrics = None  # 使用默认的6个核心指标
    else:
        metrics = args.metrics
    
    # 初始化VBench Direct
    vbench_direct = VBenchDirect(device=args.device)
    
    try:
        # 评估视频
        results = vbench_direct.evaluate_videos(video_paths, metrics)
        
        # 保存结果
        vbench_direct.save_results(results, args.output)
        
        # 显示结果
        print(f"\n�� VBench Direct 结果:")
        for metric, score in results.items():
            print(f"  {metric}: {score}")
            
    finally:
        # 清理
        vbench_direct.cleanup()


if __name__ == "__main__":
    main()
