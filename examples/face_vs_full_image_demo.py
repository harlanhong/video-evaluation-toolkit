#!/usr/bin/env python3
"""
Face vs Full Image Metrics Demo
Demonstrates the difference between calculating metrics on full images vs face regions only

Usage:
    python examples/face_vs_full_image_demo.py --pred_dir /path/to/predictions --gt_dir /path/to/ground_truth
    python examples/face_vs_full_image_demo.py --single_test  # Quick test with synthetic data

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import glob
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_metrics_calculator import VideoMetricsCalculator

def create_synthetic_test_frames():
    """Create synthetic test frames for demonstration"""
    print("🎨 Creating synthetic test frames...")
    
    # Create base frame (640x480, RGB)
    h, w = 480, 640
    
    # Ground truth frame - simple face-like pattern
    gt_frame = np.zeros((h, w, 3), dtype=np.uint8)
    gt_frame[:] = [128, 128, 128]  # Gray background
    
    # Add a "face" in the center
    face_x, face_y = w//2 - 80, h//2 - 60
    face_w, face_h = 160, 120
    gt_frame[face_y:face_y+face_h, face_x:face_x+face_w] = [220, 180, 160]  # Skin color
    
    # Add eyes
    eye_y = face_y + 30
    gt_frame[eye_y:eye_y+20, face_x+30:face_x+50] = [50, 50, 50]  # Left eye
    gt_frame[eye_y:eye_y+20, face_x+110:face_x+130] = [50, 50, 50]  # Right eye
    
    # Prediction frame - slightly different
    pred_frame = gt_frame.copy()
    
    # Add some noise to background (simulating different background)
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    pred_frame = np.clip(pred_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Keep face region similar but with slight differences
    pred_frame[face_y:face_y+face_h, face_x:face_x+face_w] = [210, 170, 150]  # Slightly different skin
    pred_frame[eye_y:eye_y+20, face_x+30:face_x+50] = [40, 40, 40]  # Left eye
    pred_frame[eye_y:eye_y+20, face_x+110:face_x+130] = [40, 40, 40]  # Right eye
    
    return pred_frame, gt_frame

def extract_frame_from_video(video_path: str, frame_idx: int = 0) -> np.ndarray:
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    # Go to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def demo_single_test():
    """Demo with synthetic test data"""
    print("🚀 Face vs Full Image Metrics Demo - Synthetic Test")
    print("=" * 60)
    
    # Create test frames
    pred_frame, gt_frame = create_synthetic_test_frames()
    
    print(f"📐 Frame size: {pred_frame.shape[1]}x{pred_frame.shape[0]}")
    
    # Initialize calculator
    print("\n⚡ Initializing metrics calculator...")
    calculator = VideoMetricsCalculator(device="cuda" if sys.argv[1:] and "--gpu" in sys.argv else "cpu")
    
    # Test 1: Full image metrics
    print("\n🖼️ Calculating FULL IMAGE metrics...")
    full_metrics = calculator.calculate_frame_metrics(
        pred_frame, gt_frame, 
        region="full_image"
    )
    
    print(f"   PSNR (full): {full_metrics['psnr']:.2f} dB")
    print(f"   SSIM (full): {full_metrics['ssim']:.4f}")
    print(f"   LPIPS (full): {full_metrics['lpips']:.4f}")
    
    # Test 2: Face region metrics
    print("\n👤 Calculating FACE REGION metrics...")
    face_metrics = calculator.calculate_frame_metrics(
        pred_frame, gt_frame, 
        region="face_only",
        face_padding=0.2
    )
    
    if face_metrics['face_detected']:
        print(f"   ✅ Face detected successfully")
        print(f"   PSNR (face): {face_metrics['face_psnr']:.2f} dB")
        print(f"   SSIM (face): {face_metrics['face_ssim']:.4f}")
        print(f"   LPIPS (face): {face_metrics['face_lpips']:.4f}")
    else:
        print(f"   ❌ No face detected")
    
    # Comparison
    print("\n📊 Comparison:")
    print(f"   Face detector: {calculator.face_detection_method}")
    print(f"   Region difference shows impact of background vs face quality")
    
    if face_metrics['face_detected']:
        psnr_diff = face_metrics['face_psnr'] - full_metrics['psnr']
        ssim_diff = face_metrics['face_ssim'] - full_metrics['ssim']
        lpips_diff = face_metrics['face_lpips'] - full_metrics['lpips']
        
        print(f"   PSNR difference: {psnr_diff:+.2f} dB ({'better' if psnr_diff > 0 else 'worse'} in face)")
        print(f"   SSIM difference: {ssim_diff:+.4f} ({'better' if ssim_diff > 0 else 'worse'} in face)")
        print(f"   LPIPS difference: {lpips_diff:+.4f} ({'better' if lpips_diff < 0 else 'worse'} in face)")

def demo_video_comparison(pred_dir: str, gt_dir: str, max_videos: int = 3):
    """Demo with real video data"""
    print("🚀 Face vs Full Image Metrics Demo - Real Video Data")
    print("=" * 60)
    
    # Find video pairs
    pred_files = glob.glob(os.path.join(pred_dir, "*.mp4"))
    gt_files = glob.glob(os.path.join(gt_dir, "*.mp4"))
    
    if not pred_files or not gt_files:
        print("❌ No video files found in specified directories")
        return
    
    print(f"📊 Found {len(pred_files)} prediction videos, {len(gt_files)} ground truth videos")
    
    # Initialize calculator
    print("\n⚡ Initializing metrics calculator...")
    calculator = VideoMetricsCalculator()
    
    results = []
    processed = 0
    
    for pred_file in pred_files[:max_videos]:
        pred_name = os.path.basename(pred_file)
        
        # Find matching ground truth file
        gt_file = None
        for gt_path in gt_files:
            gt_name = os.path.basename(gt_path)
            # Simple name matching - can be made more sophisticated
            if pred_name.split('_')[0] in gt_name or gt_name.split('_')[0] in pred_name:
                gt_file = gt_path
                break
        
        if gt_file is None:
            print(f"⚠️ No matching ground truth found for {pred_name}")
            continue
        
        print(f"\n📹 Processing video pair {processed + 1}:")
        print(f"   Pred: {pred_name}")
        print(f"   GT: {os.path.basename(gt_file)}")
        
        try:
            # Extract first frame from each video
            pred_frame = extract_frame_from_video(pred_file, frame_idx=0)
            gt_frame = extract_frame_from_video(gt_file, frame_idx=0)
            
            print(f"   Frame size: {pred_frame.shape[1]}x{pred_frame.shape[0]}")
            
            # Calculate full image metrics
            print("   🖼️ Calculating full image metrics...")
            full_metrics = calculator.calculate_frame_metrics(
                pred_frame, gt_frame, region="full_image"
            )
            
            # Calculate face metrics
            print("   👤 Calculating face region metrics...")
            face_metrics = calculator.calculate_frame_metrics(
                pred_frame, gt_frame, region="face_only", face_padding=0.2
            )
            
            result = {
                'video_name': pred_name,
                'full_psnr': full_metrics['psnr'],
                'full_ssim': full_metrics['ssim'],
                'full_lpips': full_metrics['lpips'],
                'face_detected': face_metrics['face_detected']
            }
            
            if face_metrics['face_detected']:
                result.update({
                    'face_psnr': face_metrics['face_psnr'],
                    'face_ssim': face_metrics['face_ssim'],
                    'face_lpips': face_metrics['face_lpips']
                })
                
                print(f"   ✅ Full: PSNR={full_metrics['psnr']:.2f}, SSIM={full_metrics['ssim']:.4f}, LPIPS={full_metrics['lpips']:.4f}")
                print(f"   ✅ Face: PSNR={face_metrics['face_psnr']:.2f}, SSIM={face_metrics['face_ssim']:.4f}, LPIPS={face_metrics['face_lpips']:.4f}")
            else:
                print(f"   ❌ No face detected, only full image metrics available")
                result.update({
                    'face_psnr': 0.0,
                    'face_ssim': 0.0,
                    'face_lpips': 0.0
                })
            
            results.append(result)
            processed += 1
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            continue
    
    # Summary statistics
    if results:
        print(f"\n📊 Summary Statistics ({len(results)} videos):")
        print("=" * 60)
        
        face_detected_count = sum(1 for r in results if r['face_detected'])
        print(f"Face detection success rate: {face_detected_count}/{len(results)} ({face_detected_count/len(results)*100:.1f}%)")
        
        # Calculate averages
        avg_full_psnr = np.mean([r['full_psnr'] for r in results])
        avg_full_ssim = np.mean([r['full_ssim'] for r in results])
        avg_full_lpips = np.mean([r['full_lpips'] for r in results])
        
        print(f"\n🖼️ Average Full Image Metrics:")
        print(f"   PSNR: {avg_full_psnr:.2f} dB")
        print(f"   SSIM: {avg_full_ssim:.4f}")
        print(f"   LPIPS: {avg_full_lpips:.4f}")
        
        if face_detected_count > 0:
            face_results = [r for r in results if r['face_detected']]
            avg_face_psnr = np.mean([r['face_psnr'] for r in face_results])
            avg_face_ssim = np.mean([r['face_ssim'] for r in face_results])
            avg_face_lpips = np.mean([r['face_lpips'] for r in face_results])
            
            print(f"\n👤 Average Face Region Metrics:")
            print(f"   PSNR: {avg_face_psnr:.2f} dB")
            print(f"   SSIM: {avg_face_ssim:.4f}")
            print(f"   LPIPS: {avg_face_lpips:.4f}")
            
            print(f"\n🔍 Face vs Full Image Differences:")
            psnr_diff = avg_face_psnr - avg_full_psnr
            ssim_diff = avg_face_ssim - avg_full_ssim
            lpips_diff = avg_face_lpips - avg_full_lpips
            
            print(f"   PSNR: {psnr_diff:+.2f} dB ({'face better' if psnr_diff > 0 else 'full better'})")
            print(f"   SSIM: {ssim_diff:+.4f} ({'face better' if ssim_diff > 0 else 'full better'})")
            print(f"   LPIPS: {lpips_diff:+.4f} ({'face better' if lpips_diff < 0 else 'full better'})")
        
        print(f"\n💡 Recommendations:")
        print(f"   • Face detector: {calculator.face_detection_method}")
        print(f"   • Use face-only metrics for lip-sync and facial expression evaluation")
        print(f"   • Use full-image metrics for overall video quality assessment")
        print(f"   • Combine both for comprehensive evaluation")

def main():
    parser = argparse.ArgumentParser(description="Face vs Full Image Metrics Demo")
    parser.add_argument("--pred_dir", type=str, help="Directory containing prediction videos")
    parser.add_argument("--gt_dir", type=str, help="Directory containing ground truth videos")
    parser.add_argument("--single_test", action="store_true", help="Run single test with synthetic data")
    parser.add_argument("--max_videos", type=int, default=3, help="Maximum number of videos to process")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    if args.single_test:
        demo_single_test()
    elif args.pred_dir and args.gt_dir:
        if not os.path.exists(args.pred_dir):
            print(f"❌ Prediction directory not found: {args.pred_dir}")
            return
        if not os.path.exists(args.gt_dir):
            print(f"❌ Ground truth directory not found: {args.gt_dir}")
            return
        
        demo_video_comparison(args.pred_dir, args.gt_dir, args.max_videos)
    else:
        print("❌ Please specify either --single_test or both --pred_dir and --gt_dir")
        parser.print_help()

if __name__ == "__main__":
    main()