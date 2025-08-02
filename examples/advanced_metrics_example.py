#!/usr/bin/env python3
"""
Advanced Metrics Usage Example
Demonstrates how to use CLIP similarity, FVD, and GIM matching metrics

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_calculator import VideoMetricsCalculator
from clip_api import CLIPVideoAPI
from clip_similarity_calculator import CLIPSimilarityCalculator  # Legacy wrapper
from fvd_calculator import FVDCalculator
from gim_matching_calculator import GIMMatchingCalculator


def example_individual_calculators():
    """Example of using individual calculators"""
    
    print("🔧 Example 1: Using Individual Calculators")
    print("=" * 50)
    
    # Example video paths (replace with your actual video paths)
    source_video = "path/to/source_video.mp4"
    target_video = "path/to/target_video.mp4"
    
    # 1. CLIP API (New Unified Interface)
    print("\n1️⃣ CLIP API (Unified Interface)")
    try:
        clip_api = CLIPVideoAPI(device="cuda", model_name="ViT-B/32")
        clip_results = clip_api.calculate_video_similarity(source_video, target_video)
        
        if clip_results.get('clip_similarity') is not None:
            print(f"   CLIP Similarity: {clip_results['clip_similarity']:.4f}")
            print(f"   Standard Deviation: {clip_results['clip_similarity_std']:.4f}")
            print(f"   Model: {clip_results.get('model_name', 'N/A')}")
        else:
            print(f"   Failed: {clip_results.get('error', 'Unknown error')}")
        
        # Additional CLIP API features
        print("\n   🔸 Text-Video Similarity Example:")
        text_results = clip_api.calculate_text_video_similarity(
            source_video, 
            ["a person walking", "someone dancing", "outdoor scene"],
            max_frames=10
        )
        if text_results.get('similarities') is not None:
            for i, sim in enumerate(text_results['similarities']):
                query = text_results['text_queries'][i]
                print(f"     '{query}': {sim:.4f}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 1b. CLIP Similarity Calculator (Legacy Wrapper)
    print("\n1️⃣b CLIP Similarity Calculator (Legacy)")
    try:
        clip_calc = CLIPSimilarityCalculator(device="cuda")
        clip_results = clip_calc.calculate_video_similarity(source_video, target_video)
        
        if clip_results.get('clip_similarity') is not None:
            print(f"   CLIP Similarity: {clip_results['clip_similarity']:.4f}")
            print(f"   Standard Deviation: {clip_results['clip_similarity_std']:.4f}")
        else:
            print(f"   Failed: {clip_results.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. GIM Matching Calculator
    print("\n2️⃣ GIM Matching Calculator")
    try:
        gim_calc = GIMMatchingCalculator(device="cuda", confidence_threshold=0.5)
        gim_results = gim_calc.calculate_video_matching(source_video, target_video)
        
        if gim_results.get('total_matching_pixels') is not None:
            print(f"   Total Matching Pixels: {gim_results['total_matching_pixels']}")
            print(f"   Average per Frame: {gim_results['avg_matching_pixels']:.2f}")
        else:
            print(f"   Failed: {gim_results.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. FVD Calculator
    print("\n3️⃣ FVD Calculator")
    try:
        fvd_calc = FVDCalculator(device="cuda")
        # FVD typically requires datasets, not single videos
        real_videos = ["path/to/real_video1.mp4", "path/to/real_video2.mp4"]
        generated_videos = ["path/to/gen_video1.mp4", "path/to/gen_video2.mp4"]
        
        fvd_results = fvd_calc.calculate_fvd(real_videos, generated_videos)
        
        if fvd_results.get('fvd_score') is not None:
            print(f"   FVD Score: {fvd_results['fvd_score']:.4f}")
        else:
            print(f"   Failed: {fvd_results.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   Error: {e}")


def example_integrated_calculator():
    """Example of using integrated metrics calculator"""
    
    print("\n\n🔧 Example 2: Using Integrated Metrics Calculator")
    print("=" * 50)
    
    # Example video paths
    pred_video = "path/to/predicted_video.mp4"
    gt_video = "path/to/ground_truth_video.mp4"
    
    # Create calculator with all advanced metrics enabled
    print("\n🚀 Initializing comprehensive calculator...")
    try:
        calculator = VideoMetricsCalculator(
            device="cuda",
            enable_vbench=True,           # VBench metrics
            enable_clip_similarity=True,  # CLIP similarity
            enable_fvd=False,            # FVD (requires dataset)
            enable_gim_matching=True     # GIM matching
        )
        
        # Calculate all metrics for a single video
        print(f"\n📊 Calculating metrics for: {os.path.basename(pred_video)}")
        metrics = calculator.calculate_video_metrics(
            pred_path=pred_video,
            gt_path=gt_video
        )
        
        # Display results
        print("\n📈 Results Summary:")
        print("-" * 30)
        
        # Basic metrics
        print(f"Frame Count: {metrics.get('frame_count', 'N/A')}")
        print(f"Resolution: {metrics.get('width', 'N/A')}x{metrics.get('height', 'N/A')}")
        print(f"FPS: {metrics.get('fps', 'N/A'):.2f}")
        
        # VBench metrics
        if metrics.get('subject_consistency') is not None:
            print(f"\nVBench Metrics:")
            print(f"  Subject Consistency: {metrics['subject_consistency']:.4f}")
            print(f"  Background Consistency: {metrics['background_consistency']:.4f}")
            print(f"  Motion Smoothness: {metrics['motion_smoothness']:.4f}")
            print(f"  Aesthetic Quality: {metrics['aesthetic_quality']:.4f}")
        
        # CLIP similarity
        if metrics.get('clip_similarity') is not None:
            print(f"\nCLIP Similarity:")
            print(f"  Similarity Score: {metrics['clip_similarity']:.4f}")
            print(f"  Standard Deviation: {metrics['clip_similarity_std']:.4f}")
        
        # GIM matching
        if metrics.get('gim_matching_pixels') is not None:
            print(f"\nGIM Matching:")
            print(f"  Total Matching Pixels: {metrics['gim_matching_pixels']}")
            print(f"  Average per Frame: {metrics['gim_avg_matching']:.2f}")
        
        # LSE metrics
        if metrics.get('lse_distance') is not None:
            print(f"\nLSE Metrics:")
            print(f"  LSE Distance: {metrics['lse_distance']:.4f}")
            print(f"  LSE Confidence: {metrics['lse_confidence']:.4f}")
        
        # Cleanup
        calculator.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_processing():
    """Example of batch processing with new metrics"""
    
    print("\n\n🔧 Example 3: Batch Processing with Advanced Metrics")
    print("=" * 50)
    
    try:
        # Create calculator
        calculator = VideoMetricsCalculator(
            device="cuda",
            enable_vbench=False,         # Disable for faster processing
            enable_clip_similarity=True,
            enable_gim_matching=True
        )
        
        # Batch process videos
        pred_dir = "path/to/predicted_videos/"
        gt_dir = "path/to/ground_truth_videos/"
        
        print(f"\n📁 Processing videos from: {pred_dir}")
        results = calculator.calculate_batch_metrics(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            pattern="*.mp4"
        )
        
        # Save results
        output_file = "advanced_metrics_results.json"
        calculator.save_results(results, output_file)
        
        # Print summary
        calculator.print_summary_stats(results)
        
        # Cleanup
        calculator.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run all examples"""
    
    print("🎬 Advanced Video Metrics Examples")
    print("=" * 60)
    print("This script demonstrates the usage of new advanced metrics:")
    print("• CLIP-V: CLIP similarity between source and target frames")
    print("• FVD-V: Fréchet Video Distance for video quality assessment")
    print("• GIM: Graph Image Matching for synchronization evaluation")
    print()
    
    # Note about file paths
    print("⚠️  Note: Update the video file paths in this script before running")
    print("   Replace 'path/to/...' with actual video file paths")
    print()
    
    # Run examples
    example_individual_calculators()
    example_integrated_calculator()
    example_batch_processing()
    
    print("\n\n✅ Examples completed!")
    print("\nNext steps:")
    print("• Update video file paths with your actual data")
    print("• Install required dependencies: pip install -r requirements.txt")
    print("• For CLIP: pip install git+https://github.com/openai/CLIP.git")
    print("• Ensure CUDA is available for GPU acceleration")


if __name__ == "__main__":
    main()