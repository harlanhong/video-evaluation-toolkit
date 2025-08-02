#!/usr/bin/env python3
"""
VBench Direct Integration - Direct VBench Core Logic Integration
Direct use of VBench class to ensure 100% consistency

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

# Import VBench
from vbench import VBench
from vbench.distributed import dist_init, print0


class VBenchDirect:
    """VBench Direct - Direct use of VBench class"""
    
    def __init__(self, device: str = "cuda", cache_dir: str = "./cache"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"🚀 Initializing VBench Direct (device: {self.device})")
        
        # Define 6 core metrics
        self.core_metrics = [
            'subject_consistency',
            'background_consistency', 
            'motion_smoothness',
            'dynamic_degree',
            'aesthetic_quality',
            'imaging_quality'
        ]
        
        # Initialize distributed environment
        dist_init()
        
        # Create temporary json config file
        self.full_json_path = self._create_temp_json()
        
        # Create output directory
        self.output_path = tempfile.mkdtemp(prefix="vbench_output_")
        
        # Initialize VBench object
        self.vbench = VBench(self.device, self.full_json_path, self.output_path)
        
        print(f"✅ VBench object initialization completed")
    
    def _create_temp_json(self) -> str:
        """Create temporary VBench configuration JSON file"""
        # Simplified VBench configuration
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
        """Prepare video directory"""
        temp_dir = tempfile.mkdtemp(prefix="vbench_videos_")
        
        print(f"🔄 Preparing video directory: {temp_dir}")
        
        for i, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            _, ext = os.path.splitext(video_path)
            dest_name = f"video_{i:03d}{ext}"
            dest_path = os.path.join(temp_dir, dest_name)
            
            print(f"   📄 Copying: {os.path.basename(video_path)} -> {dest_name}")
            shutil.copy2(video_path, dest_path)
        
        return temp_dir
    
    def evaluate_videos(self, video_paths: List[str], metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate videos, directly call VBench core logic"""
        if metrics is None:
            metrics = self.core_metrics.copy()
        
        print(f"🔍 Starting evaluation of {len(video_paths)} videos with {len(metrics)} metrics")
        print("=" * 60)
        
        temp_video_dir = None
        
        try:
            # Prepare video directory
            temp_video_dir = self._prepare_video_directory(video_paths)
            
            # === Core VBench logic (ported from official evaluate.py) ===
            print0(f'🚀 Starting VBench evaluation')
            
            current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            
            kwargs = {}
            prompt = []  # Empty list means read prompt from filename
            
            # Set parameters
            kwargs['imaging_quality_preprocessing_mode'] = 'longer'
            
            # Call VBench core evaluation function
            print0(f'📊 Calling VBench.evaluate...')
            
            self.vbench.evaluate(
                videos_path=temp_video_dir,
                name=f'results_{current_time}',
                prompt_list=prompt,  # Read prompt from filename
                dimension_list=metrics,
                local=False,  # Do not load checkpoints from local
                read_frame=False,  # Read video directly
                mode='custom_input',
                **kwargs
            )
            
            print0('✅ VBench evaluation completed')
            
            # Parse results
            results = self._parse_results(current_time)
            
            return results
            
        finally:
            # Clean up temporary directory
            if temp_video_dir and os.path.exists(temp_video_dir):
                shutil.rmtree(temp_video_dir)
                print(f"🗑️  Cleaned temporary video directory: {temp_video_dir}")
    
    def _parse_results(self, result_name: str) -> Dict[str, float]:
        """Parse VBench results"""
        print(f"📊 Parsing VBench results...")
        
        # Find eval_results file (this file contains evaluation results)
        result_file = None
        
        for file in os.listdir(self.output_path):
            if file.endswith("_eval_results.json") and result_name in file:
                result_file = os.path.join(self.output_path, file)
                break
        
        if result_file is None:
            print(f"⚠️ eval_results file not found in: {self.output_path}")
            print(f"Directory contents: {os.listdir(self.output_path)}")
            # Return default values
            return {metric: 0.0 for metric in self.core_metrics}
        
        print(f"   📄 Result file: {result_file}")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)
        
        print(f"   📋 Raw result keys: {list(raw_results.keys())}")
        
        # Parse results
        # VBench format: {"metric_name": [total_score, detailed_results]}
        results = {}
        
        for metric in self.core_metrics:
            if metric in raw_results:
                metric_data = raw_results[metric]
                
                if isinstance(metric_data, list) and len(metric_data) >= 1:
                    # Take first element as total score
                    score = metric_data[0]
                    if isinstance(score, (int, float)):
                        results[metric] = float(score)
                        print(f"   ✅ {metric}: {results[metric]}")
                    else:
                        print(f"   ⚠️ {metric} first element is not numeric: {type(score)}")
                        results[metric] = 0.0
                        
                elif isinstance(metric_data, (int, float)):
                    results[metric] = float(metric_data)
                    print(f"   ✅ {metric}: {results[metric]}")
                else:
                    print(f"   ⚠️ Unknown format for {metric}: {type(metric_data)}")
                    results[metric] = 0.0
            else:
                print(f"   ❌ Not found in results: {metric}")
                results[metric] = 0.0
        
        return results
    
    def save_results(self, results: Dict[str, float], output_path: str = "vbench_direct_results.json"):
        """Save results to JSON file"""
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                clean_results[k] = float(v)
            else:
                clean_results[k] = v
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_path}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'output_path') and os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            print(f"🗑️  Cleaned output directory: {self.output_path}")
        
        if hasattr(self, 'full_json_path') and os.path.exists(self.full_json_path):
            os.remove(self.full_json_path)
            print(f"🗑️  Cleaned config file: {self.full_json_path}")


def main():
    parser = argparse.ArgumentParser(description="VBench Direct - Direct VBench Core Logic Integration")
    parser.add_argument("--videos", type=str, required=True, help="Video file paths (comma-separated for multiple files)")
    parser.add_argument("--metrics", type=str, nargs='+', choices=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'all'], default=['all'], help="Metrics to calculate")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")
    parser.add_argument("--output", type=str, default="vbench_direct_results.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Parse video paths
    video_paths = [path.strip() for path in args.videos.split(',')]
    
    # Validate video files exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
    
    # Parse metrics
    if 'all' in args.metrics:
        metrics = None  # Use default 6 core metrics
    else:
        metrics = args.metrics
    
    # Initialize VBench Direct
    vbench_direct = VBenchDirect(device=args.device)
    
    try:
        # Evaluate videos
        results = vbench_direct.evaluate_videos(video_paths, metrics)
        
        # Save results
        vbench_direct.save_results(results, args.output)
        
        # Display results
        print(f"\n📊 VBench Direct Results:")
        for metric, score in results.items():
            print(f"  {metric}: {score}")
            
    finally:
        # Cleanup
        vbench_direct.cleanup()


if __name__ == "__main__":
    main()
