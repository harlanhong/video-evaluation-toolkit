#!/usr/bin/env python3
"""
Comprehensive Video Metrics Calculator v2.1.0 (VBench Integrated Version)
Integrates LSE calculation, VBench metrics, and other video quality metrics

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module integrates VBench metrics with comprehensive video evaluation tools.

Supported Metrics:
- Basic Video Info (no GT required): frame count, resolution, fps, duration
- Image Statistics (no GT required): brightness, contrast, saturation, sharpness
- Face Analysis (no GT required): face detection rate, average face size, face stability
- Motion Analysis (no GT required): motion intensity, frame difference
- Face Region Image Quality (GT required): face_psnr, face_ssim, face_lpips
- Lip Sync Metrics (no GT required): LSE distance, LSE confidence
- VBench Metrics (no GT required): subject_consistency, background_consistency, motion_smoothness, 
  dynamic_degree, aesthetic_quality, imaging_quality

Usage:
    from evalutation.metrics_calculator import VideoMetricsCalculator
    
    # Without VBench
    calculator = VideoMetricsCalculator()
    metrics = calculator.calculate_video_metrics("video.mp4")
    
    # With VBench
    calculator = VideoMetricsCalculator(enable_vbench=True)
    metrics = calculator.calculate_video_metrics("video.mp4")
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import tempfile
import shutil
import glob
import json
import argparse
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from tqdm import tqdm

# Image quality metrics
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import calculators
try:
    # Use relative import when imported as package
    from ..calculators.lse_calculator import LSECalculator
    from ..calculators.fvd_calculator import FVDCalculator
    from ..calculators.gim_calculator import GIMMatchingCalculator
    from ..apis.clip_api import CLIPVideoAPI
    # VBench is optional
    try:
        from ..calculators.vbench_calculator import VBenchDirect
    except ImportError:
        VBenchDirect = None
except ImportError:
    # Use absolute import when run directly
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from calculators.lse_calculator import LSECalculator
    from calculators.fvd_calculator import FVDCalculator
    from calculators.gim_calculator import GIMMatchingCalculator
    from apis.clip_api import CLIPVideoAPI
    # VBench is optional
    try:
        from calculators.vbench_calculator import VBenchDirect
    except ImportError:
        VBenchDirect = None


class VideoMetricsCalculator:
    """Comprehensive Video Metrics Calculator"""
    
    def __init__(self, 
                 device: str = "cuda", 
                 enable_vbench: bool = False,
                 enable_clip_similarity: bool = False,
                 enable_fvd: bool = False,
                 enable_gim_matching: bool = False,
                 enable_lse: bool = False):
        """
        Initialize metrics calculator
        
        Args:
            device: Computing device ("cuda" or "cpu")
            enable_vbench: Whether to enable VBench metrics calculation
            enable_clip_similarity: Whether to enable CLIP similarity calculation
            enable_fvd: Whether to enable FVD calculation  
            enable_gim_matching: Whether to enable GIM matching calculation
            enable_lse: Whether to enable LSE (Lip-Sync Error) calculation
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_vbench = enable_vbench
        self.enable_clip_similarity = enable_clip_similarity
        self.enable_fvd = enable_fvd
        self.enable_gim_matching = enable_gim_matching
        self.enable_lse = enable_lse
        
        # Initialize LPIPS model
        print("🔄 Initializing LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize face detector
        print("🔄 Initializing face detector...")
        self.face_detection_method = self._initialize_face_detector()
        self.face_detection_available = self.face_detection_method is not None
        
        if self.face_detection_available:
            print(f"✅ Face detector initialized successfully (method: {self.face_detection_method})")
        else:
            print("⚠️ All face detectors failed to initialize, face-related metrics will be skipped")
        
        # Initialize LSE calculator (only if enabled)
        if self.enable_lse:
            print("🔄 Initializing LSE calculator...")
            try:
                self.lse_calculator = LSECalculator(device=self.device)
                self.lse_available = True
            except Exception as e:
                print(f"⚠️ LSE calculator initialization failed: {e}")
                self.lse_available = False
        else:
            self.lse_calculator = None
            self.lse_available = False
        
        # Initialize VBench calculator
        if self.enable_vbench and VBenchDirect is not None:
            print("🔄 Initializing VBench calculator...")
            try:
                self.vbench_calculator = VBenchDirect(device=self.device)
                self.vbench_available = True
                print("✅ VBench calculator initialized successfully")
            except Exception as e:
                print(f"⚠️ VBench calculator initialization failed: {e}")
                self.vbench_available = False
        else:
            if self.enable_vbench and VBenchDirect is None:
                print("⚠️ VBench not available (module not installed)")
            self.vbench_available = False
        
        # Initialize CLIP API
        if self.enable_clip_similarity:
            print("🔄 Initializing CLIP API...")
            try:
                self.clip_api = CLIPVideoAPI(device=self.device, model_name="ViT-B/32")
                self.clip_available = True
                print("✅ CLIP API initialized successfully")
            except Exception as e:
                print(f"⚠️ CLIP API initialization failed: {e}")
                self.clip_available = False
        else:
            self.clip_available = False
        
        # Initialize FVD calculator
        if self.enable_fvd:
            print("🔄 Initializing FVD calculator...")
            try:
                self.fvd_calculator = FVDCalculator(device=self.device)
                self.fvd_available = True
                print("✅ FVD calculator initialized successfully")
            except Exception as e:
                print(f"⚠️ FVD calculator initialization failed: {e}")
                self.fvd_available = False
        else:
            self.fvd_available = False
        
        # Initialize GIM matching calculator
        if self.enable_gim_matching:
            print("🔄 Initializing GIM matching calculator...")
            try:
                self.gim_calculator = GIMMatchingCalculator(device=self.device)
                self.gim_available = True
                print("✅ GIM matching calculator initialized successfully")
            except Exception as e:
                print(f"⚠️ GIM matching calculator initialization failed: {e}")
                self.gim_available = False
        else:
            self.gim_available = False
        
        # Summary of initialization
        enabled_modules = []
        if self.vbench_available: enabled_modules.append("VBench")
        if self.clip_available: enabled_modules.append("CLIP")
        if self.fvd_available: enabled_modules.append("FVD")
        if self.gim_available: enabled_modules.append("GIM")
        if self.lse_available: enabled_modules.append("LSE")
        
        modules_str = ", ".join(enabled_modules) if enabled_modules else "Basic metrics only"
        print(f"🚀 Metrics calculator initialization completed (device: {self.device}, modules: {modules_str})")
    
    def calculate_video_metrics(self, 
                               pred_path: str, 
                               gt_path: Optional[str] = None,
                               audio_path: Optional[str] = None,
                               region: str = "face_only") -> Dict[str, Any]:
        """
        Calculate all metrics for a single video
        
        Args:
            pred_path: Predicted video path
            gt_path: Ground truth video path (optional)
            audio_path: Audio file path (optional, deprecated)
            region: "full_image" or "face_only" - region for PSNR/SSIM/LPIPS calculation
                   (LSE and other metrics always use full image)
            
        Returns:
            Dictionary containing all metrics
        """
        
        # Initialize metrics dictionary
        metrics = {
            'video_path': pred_path,
            'has_ground_truth': gt_path is not None and os.path.exists(gt_path) if gt_path else False,
            'has_audio': False,  # No longer need external audio files
            'vbench_enabled': self.vbench_available,
            
            # Basic info (no ground truth needed)
            'frame_count': 0,
            'width': 0,
            'height': 0,
            'fps': 0.0,
            'duration_seconds': 0.0,
            
            # Image statistics (no ground truth needed)
            'mean_brightness': 0.0,
            'mean_contrast': 0.0,
            'mean_saturation': 0.0,
            'sharpness_score': 0.0,
            
            # Face detection statistics (no ground truth needed)
            'face_detection_rate': 0.0,  # Proportion of frames with detected faces
            'avg_face_size': 0.0,        # Average face size
            'face_stability': 0.0,       # Face position stability
            
            # Motion analysis (no ground truth needed)
            'motion_intensity': 0.0,     # Motion intensity
            'frame_difference': 0.0,     # Average frame difference
            
            # Metrics requiring ground truth (region depends on 'region' parameter)
            'psnr': None,             # PSNR (full_image or face_only based on region param)
            'ssim': None,             # SSIM (full_image or face_only based on region param)  
            'lpips': None,            # LPIPS (full_image or face_only based on region param)
            
            # Lip sync metrics (using LSE calculator, no external audio needed)
            'lse_distance': None,     # LSE distance score
            'lse_confidence': None,   # LSE confidence score
            
            # VBench metrics (no ground truth needed)
            'subject_consistency': None,     # Subject consistency
            'background_consistency': None,  # Background consistency
            'motion_smoothness': None,       # Motion smoothness
            'dynamic_degree': None,          # Dynamic degree
            'aesthetic_quality': None,       # Aesthetic quality
            'imaging_quality': None,         # Imaging quality
            
            # CLIP similarity metrics (requires ground truth)
            'clip_similarity': None,         # CLIP-V similarity score
            'clip_similarity_std': None,     # Standard deviation of CLIP similarities
            
            # FVD metrics (requires ground truth or dataset)
            'fvd_score': None,              # FVD-V score
            
            # GIM matching metrics (requires ground truth)
            'gim_matching_pixels': None,     # Total matching pixels (Mat. Pix.)
            'gim_avg_matching': None,        # Average matching pixels per frame
            
            'error': None
        }
        
        try:
            # Open video file
            pred_cap = cv2.VideoCapture(pred_path)
            if not pred_cap.isOpened():
                metrics['error'] = f"Cannot open predicted video: {pred_path}"
                return metrics
            
            # Get basic video information
            fps = pred_cap.get(cv2.CAP_PROP_FPS)
            width = int(pred_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(pred_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            metrics['fps'] = fps
            metrics['width'] = width
            metrics['height'] = height
            
            pred_frames = []
            face_detections = []
            brightness_values = []
            contrast_values = []
            saturation_values = []
            sharpness_values = []
            
            while True:
                ret, frame = pred_cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred_frames.append(frame_rgb)
                
                # Calculate image statistics
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                
                # Brightness (grayscale mean)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Contrast (grayscale standard deviation)
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                # Saturation (HSV S channel mean)
                hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1])
                saturation_values.append(saturation)
                
                # Sharpness (Laplacian variance)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_values.append(sharpness)
                
                # Face detection
                if self.face_detection_available:
                    face_bbox = self.detect_face_bbox(frame_rgb)
                    if face_bbox is not None:
                        x, y, w, h = face_bbox
                        face_size = w * h
                        face_detections.append({'frame_idx': len(pred_frames)-1, 'bbox': face_bbox, 'size': face_size})
                
            pred_cap.release()
            
            frame_count = len(pred_frames)
            metrics['frame_count'] = frame_count
            metrics['duration_seconds'] = frame_count / fps if fps > 0 else 0
            
            # Calculate metrics that don't require ground truth
            if brightness_values:
                metrics['mean_brightness'] = np.mean(brightness_values)
                metrics['mean_contrast'] = np.mean(contrast_values)
                metrics['mean_saturation'] = np.mean(saturation_values)
                metrics['sharpness_score'] = np.mean(sharpness_values)
            
            # Face detection statistics
            if face_detections:
                metrics['face_detection_rate'] = len(face_detections) / frame_count
                metrics['avg_face_size'] = np.mean([fd['size'] for fd in face_detections])
                
                # Calculate face position stability (variance of center point distances between adjacent frames)
                if len(face_detections) > 1:
                    centers = []
                    for fd in face_detections:
                        x, y, w, h = fd['bbox']
                        center_x = x + w/2
                        center_y = y + h/2
                        centers.append((center_x, center_y))
                    
                    distances = []
                    for i in range(1, len(centers)):
                        dist = np.sqrt((centers[i][0] - centers[i-1][0])**2 + 
                                    (centers[i][1] - centers[i-1][1])**2)
                        distances.append(dist)
                    
                    # Stability = 1 / (1 + average distance), higher value means more stable
                    metrics['face_stability'] = 1.0 / (1.0 + np.mean(distances)) if distances else 1.0
            
            # Motion analysis
            if len(pred_frames) > 1:
                frame_diffs = []
                for i in range(1, len(pred_frames)):
                    diff = np.mean(np.abs(pred_frames[i].astype(float) - pred_frames[i-1].astype(float)))
                    frame_diffs.append(diff)
                
                metrics['motion_intensity'] = np.std(frame_diffs) if frame_diffs else 0.0
                metrics['frame_difference'] = np.mean(frame_diffs) if frame_diffs else 0.0
            
            # Calculate LSE scores (using LSE calculator, only if enabled)
            if self.enable_lse and self.lse_available:
                print(f"🎵 Calculating LSE scores for: {os.path.basename(pred_path)}")
                try:
                    lse_distance, lse_confidence = self.lse_calculator.calculate_single_video(pred_path, verbose=False)
                    metrics['lse_distance'] = lse_distance
                    metrics['lse_confidence'] = lse_confidence
                    print(f"   ✅ LSE calculation completed: distance={lse_distance:.4f}, confidence={lse_confidence:.4f}")
                except Exception as e:
                    print(f"⚠️ LSE calculation failed for {os.path.basename(pred_path)}: {e}")
                    print(f"   📁 File path: {pred_path}")
                    print(f"   📋 File exists: {os.path.exists(pred_path)}")
                    metrics['lse_distance'] = None
                    metrics['lse_confidence'] = None
            elif self.enable_lse and not self.lse_available:
                print(f"⚠️ LSE calculator not available, skipping LSE calculation")
            # If LSE is not enabled, we don't mention it at all
            
            # Calculate VBench scores
            if self.vbench_available:
                print(f"🔥 Calculating VBench metrics (6 core metrics)")
                try:
                    vbench_results = self.vbench_calculator.evaluate_videos([pred_path])
                    
                    # Merge VBench results into metrics
                    for metric_name in ['subject_consistency', 'background_consistency', 'motion_smoothness', 
                                      'dynamic_degree', 'aesthetic_quality', 'imaging_quality']:
                        if metric_name in vbench_results:
                            metrics[metric_name] = vbench_results[metric_name]
                            print(f"   ✅ {metric_name}: {metrics[metric_name]}")
                        else:
                            print(f"   ❌ Not found in VBench results: {metric_name}")
                            metrics[metric_name] = None
                    
                except Exception as e:
                    print(f"⚠️ VBench calculation failed: {e}")
                    for metric_name in ['subject_consistency', 'background_consistency', 'motion_smoothness', 
                                      'dynamic_degree', 'aesthetic_quality', 'imaging_quality']:
                        metrics[metric_name] = None
            else:
                print(f"⚠️ VBench calculator not available, skipping VBench calculation")
            
            # If ground truth video exists, calculate comparison metrics
            if gt_path and os.path.exists(gt_path):
                print(f"🔍 Calculating comparison metrics (vs ground truth)")
                gt_cap = cv2.VideoCapture(gt_path)
                if gt_cap.isOpened():
                    gt_frames = []
                    while True:
                        ret, frame = gt_cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gt_frames.append(frame_rgb)
                    gt_cap.release()
                    
                    # Calculate face region metrics
                    frame_metrics_list = []
                    min_frames = min(len(pred_frames), len(gt_frames))
                    
                    for i in range(min_frames):
                        frame_metrics = self.calculate_frame_metrics(pred_frames[i], gt_frames[i], region=region)
                        if frame_metrics:
                            frame_metrics_list.append(frame_metrics)
                    
                    if frame_metrics_list:
                        # Calculate average values for image metrics (region-dependent)
                        for key in ['psnr', 'ssim', 'lpips']:
                            values = [fm.get(key, 0.0) for fm in frame_metrics_list if fm.get(key, 0.0) > 0]
                            if values:
                                metrics[key] = np.mean(values)
                    
                    # Calculate CLIP similarity if enabled
                    if self.clip_available:
                        print(f"🎨 Calculating CLIP similarity")
                        try:
                            clip_results = self.clip_api.calculate_video_similarity(
                                source_path=pred_path, 
                                target_path=gt_path, 
                                verbose=False
                            )
                            if clip_results.get('clip_similarity') is not None:
                                metrics['clip_similarity'] = clip_results['clip_similarity']
                                metrics['clip_similarity_std'] = clip_results.get('clip_similarity_std', None)
                                print(f"   ✅ CLIP similarity: {metrics['clip_similarity']:.4f}")
                            else:
                                print(f"   ❌ CLIP similarity calculation failed")
                        except Exception as e:
                            print(f"   ⚠️ CLIP similarity calculation failed: {e}")
                    
                    # Calculate GIM matching if enabled
                    if self.gim_available:
                        print(f"🔍 Calculating GIM matching pixels")
                        try:
                            gim_results = self.gim_calculator.calculate_video_matching(
                                pred_path, gt_path, verbose=False
                            )
                            if gim_results.get('total_matching_pixels') is not None:
                                metrics['gim_matching_pixels'] = gim_results['total_matching_pixels']
                                metrics['gim_avg_matching'] = gim_results.get('avg_matching_pixels', None)
                                print(f"   ✅ GIM matching pixels: {metrics['gim_matching_pixels']}")
                            else:
                                print(f"   ❌ GIM matching calculation failed")
                        except Exception as e:
                            print(f"   ⚠️ GIM matching calculation failed: {e}")
                    
                else:
                    print(f"⚠️ Cannot open ground truth video: {gt_path}")
            else:
                print(f"⚠️ No ground truth video, skipping comparison metrics")
            
            # Calculate FVD if enabled (can work with single video against dataset)
            if self.fvd_available:
                print(f"📊 Calculating FVD score")
                try:
                    # For single video FVD, we would typically need a reference dataset
                    # This is a placeholder - in practice, you'd provide real/generated video sets
                    print(f"   ⚠️ FVD calculation requires dataset comparison (placeholder)")
                    # fvd_results = self.fvd_calculator.calculate_fvd([gt_path], [pred_path], verbose=False)
                    # if fvd_results.get('fvd_score') is not None:
                    #     metrics['fvd_score'] = fvd_results['fvd_score']
                    #     print(f"   ✅ FVD score: {metrics['fvd_score']:.4f}")
                except Exception as e:
                    print(f"   ⚠️ FVD calculation failed: {e}")
        
        except Exception as e:
            metrics['error'] = str(e)
            print(f"Error calculating video metrics: {e}")
        
        return metrics
    
    def calculate_frame_metrics(self, 
                               pred_frame: np.ndarray, 
                               gt_frame: np.ndarray, 
                               region: str = "face_only",
                               face_padding: float = 0.2) -> Dict[str, float]:
        """
        Calculate single frame metrics
        
        Args:
            pred_frame: Predicted frame in RGB format
            gt_frame: Ground truth frame in RGB format  
            region: "full_image" or "face_only" - which region to calculate metrics for
            face_padding: Padding around detected face (when region="face_only")
            
        Returns:
            Dictionary with calculated metrics
        """
        
        if region == "full_image":
            return self._calculate_full_image_metrics(pred_frame, gt_frame)
        elif region == "face_only":
            return self._calculate_face_metrics(pred_frame, gt_frame, face_padding)
        else:
            raise ValueError(f"Invalid region: {region}. Must be 'full_image' or 'face_only'")
    
    def _calculate_full_image_metrics(self, pred_frame: np.ndarray, gt_frame: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for the full image"""
        
        metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'lpips': 0.0,
            'region_used': 'full_image',
            'face_detected': False
        }
        
        # Ensure both frames have the same size
        if pred_frame.shape != gt_frame.shape:
            gt_frame = cv2.resize(gt_frame, (pred_frame.shape[1], pred_frame.shape[0]))
        
        # Calculate PSNR for full image
        try:
            psnr_val = peak_signal_noise_ratio(gt_frame, pred_frame, data_range=255)
            metrics['psnr'] = psnr_val
        except Exception as e:
            print(f"   ⚠️ PSNR calculation failed: {e}")
            metrics['psnr'] = 0.0
        
        # Calculate SSIM for full image
        try:
            ssim_val = structural_similarity(
                gt_frame, pred_frame, 
                multichannel=True, 
                data_range=255,
                channel_axis=-1
            )
            metrics['ssim'] = ssim_val
        except Exception as e:
            print(f"   ⚠️ SSIM calculation failed: {e}")
            metrics['ssim'] = 0.0
        
        # Calculate LPIPS for full image
        try:
            # Convert to tensor format [1, C, H, W]
            pred_tensor = torch.from_numpy(pred_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gt_tensor = torch.from_numpy(gt_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Normalize to [-1, 1]
            pred_tensor = pred_tensor * 2.0 - 1.0
            gt_tensor = gt_tensor * 2.0 - 1.0
            
            pred_tensor = pred_tensor.to(self.device)
            gt_tensor = gt_tensor.to(self.device)
            
            lpips_val = self.lpips_fn(pred_tensor, gt_tensor).item()
            metrics['lpips'] = lpips_val
        except Exception as e:
            print(f"   ⚠️ LPIPS calculation failed: {e}")
            metrics['lpips'] = 0.0
        
        return metrics
    
    def _calculate_face_metrics(self, 
                               pred_frame: np.ndarray, 
                               gt_frame: np.ndarray, 
                               face_padding: float = 0.2) -> Dict[str, float]:
        """Calculate metrics for face region only"""
        
        # Detect face region
        pred_face_bbox = self.detect_face_bbox(pred_frame)
        gt_face_bbox = self.detect_face_bbox(gt_frame)
        
        metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'lpips': 0.0,
            'region_used': 'face_only',
            'face_detected': False
        }
        
        # If faces are detected in both frames, calculate face region metrics
        if pred_face_bbox is not None and gt_face_bbox is not None:
            metrics['face_detected'] = True
            
            # First, ensure both frames have the same size
            if pred_frame.shape != gt_frame.shape:
                gt_frame = cv2.resize(gt_frame, (pred_frame.shape[1], pred_frame.shape[0]))
                # Re-detect face in resized GT frame to get correct coordinates
                gt_face_bbox = self.detect_face_bbox(gt_frame)
                if gt_face_bbox is None:
                    print("   ⚠️ Face lost after resizing GT frame")
                    return metrics
            
            # Use the larger bounding box to ensure the full face is included
            x1 = min(pred_face_bbox[0], gt_face_bbox[0])
            y1 = min(pred_face_bbox[1], gt_face_bbox[1])
            x2 = max(pred_face_bbox[0] + pred_face_bbox[2], gt_face_bbox[0] + gt_face_bbox[2])
            y2 = max(pred_face_bbox[1] + pred_face_bbox[3], gt_face_bbox[1] + gt_face_bbox[3])
            
            # Add padding around face region
            h, w = pred_frame.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            pad_w, pad_h = int(face_w * face_padding), int(face_h * face_padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            if x2 > x1 and y2 > y1:  # Ensure a valid region
                pred_face = pred_frame[y1:y2, x1:x2]
                gt_face = gt_frame[y1:y2, x1:x2]
                
                # Calculate face region PSNR
                try:
                    face_psnr = peak_signal_noise_ratio(gt_face, pred_face, data_range=255)
                    metrics['psnr'] = face_psnr
                except Exception as e:
                    print(f"   ⚠️ Face PSNR calculation failed: {e}")
                    metrics['psnr'] = 0.0
                
                # Calculate face region SSIM
                try:
                    face_ssim = structural_similarity(
                        gt_face, pred_face, 
                        multichannel=True, 
                        data_range=255,
                        channel_axis=-1
                    )
                    metrics['ssim'] = face_ssim
                except Exception as e:
                    print(f"   ⚠️ Face SSIM calculation failed: {e}")
                    metrics['ssim'] = 0.0
                
                # Calculate face region LPIPS
                try:
                    # Convert to tensor format [1, C, H, W]
                    pred_tensor = torch.from_numpy(pred_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    gt_tensor = torch.from_numpy(gt_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    
                    # Normalize to [-1, 1]
                    pred_tensor = pred_tensor * 2.0 - 1.0
                    gt_tensor = gt_tensor * 2.0 - 1.0
                    
                    pred_tensor = pred_tensor.to(self.device)
                    gt_tensor = gt_tensor.to(self.device)
                    
                    face_lpips = self.lpips_fn(pred_tensor, gt_tensor).item()
                    metrics['lpips'] = face_lpips
                except Exception as e:
                    print(f"   ⚠️ Face LPIPS calculation failed: {e}")
                    metrics['lpips'] = 0.0
        else:
            print("   ⚠️ No face detected in one or both frames, returning zero metrics")
        
        return metrics
    
    def _initialize_face_detector(self) -> Optional[str]:
        """Initialize the face detector, trying different methods in order of priority."""
        
        # Method 1: MediaPipe Face Detection (recommended)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detector_mp = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            print("   🚀 Using MediaPipe face detection (recommended)")
            return "mediapipe"
        except ImportError:
            print("   ⚠️ MediaPipe not installed, trying other methods...")
        except Exception as e:
            print(f"   ⚠️ MediaPipe initialization failed: {e}")
        
        # Method 2: YOLOv8 Face Detection
        try:
            from ultralytics import YOLO
            # Try to load YOLOv8n-face model (if available)
            self.yolo_model = YOLO('models/yolov8n-face.pt')
            print("   ⚡ Using YOLOv8 face detection")
            return "yolov8"
        except ImportError:
            print("   ⚠️ ultralytics not installed, trying other methods...")
        except Exception as e:
            print(f"   ⚠️ YOLOv8 initialization failed: {e}")
        
        # Method 3: OpenCV DNN Face Detection
        try:
            # Use OpenCV DNN module to load pre-trained face detection model
            # Can use SSD MobileNet or ResNet models
            self.face_net = cv2.dnn.readNetFromTensorflow(
                # Need to download model files
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            print("   🔧 Using OpenCV DNN face detection")
            return "opencv_dnn"
        except Exception as e:
            print(f"   ⚠️ OpenCV DNN initialization failed: {e}")
        
        # Method 4: Traditional Haar Cascade (fallback)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("   📰 Using traditional Haar cascade classifier (fallback)")
            return "haar_cascade"
        except Exception as e:
            print(f"   ❌ Haar cascade initialization failed: {e}")
        
        return None
    
    def detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box - supports multiple modern detectors."""
        if not self.face_detection_available:
            return None
        
        h, w = frame.shape[:2]
        
        try:
            if self.face_detection_method == "mediapipe":
                return self._detect_face_mediapipe(frame, h, w)
            elif self.face_detection_method == "yolov8":
                return self._detect_face_yolov8(frame, h, w)
            elif self.face_detection_method == "opencv_dnn":
                return self._detect_face_opencv_dnn(frame, h, w)
            elif self.face_detection_method == "haar_cascade":
                return self._detect_face_haar_cascade(frame, h, w)
        except Exception as e:
            print(f"⚠️ Face detection failed ({self.face_detection_method}): {e}")
            return None
        
        return None
    
    def _detect_face_mediapipe(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using MediaPipe."""
        # frame is already in RGB format, no conversion needed
        results = self.face_detector_mp.process(frame)
        
        if results.detections:
            # Get the face with the highest confidence score
            best_detection = max(results.detections, key=lambda d: d.score[0])
            bbox = best_detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image boundaries
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = max(1, min(width, w - x))
            height = max(1, min(height, h - y))
            
            return (x, y, width, height)
        
        return None
    
    def _detect_face_yolov8(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using YOLOv8."""
        results = self.yolo_model(frame, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the detection result with the highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            # Get bounding box coordinates (x1, y1, x2, y2)
            box = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Convert to (x, y, w, h) format
            x = int(x1)
            y = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            return (x, y, width, height)
        
        return None
    
    def _detect_face_opencv_dnn(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using OpenCV DNN."""
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2 - x1, y2 - y1)
        
        return best_box
    
    def _detect_face_haar_cascade(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using traditional Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the largest face
            areas = [face_w * face_h for (x, y, face_w, face_h) in faces]
            max_idx = np.argmax(areas)
            return tuple(faces[max_idx])
        
        return None
    
    def calculate_batch_metrics(self, 
                               pred_dir: str, 
                               gt_dir: Optional[str] = None,
                               pattern: str = "*.mp4",
                               region: str = "face_only") -> List[Dict[str, Any]]:
        """
        Batch calculate video metrics.
        
        Args:
            pred_dir: Directory of predicted videos.
            gt_dir: Directory of ground truth videos (optional).
            pattern: File matching pattern.
            region: Region for PSNR/SSIM/LPIPS calculation ("face_only" or "full_image").
            
        Returns:
            A list of metric results.
        """
        
        # Get list of predicted video files
        pred_files = sorted(glob.glob(os.path.join(pred_dir, pattern)))
        if not pred_files:
            print(f"⚠️ No files matching {pattern} found in {pred_dir}")
            return []
        
        print(f"🔍 Found {len(pred_files)} video files")
        
        results = []
        
        for pred_file in tqdm(pred_files, desc="Calculating Metrics"):
            pred_name = os.path.basename(pred_file)
            
            # Find the corresponding ground truth video
            gt_file = None
            if gt_dir:
                # Try several possible naming conventions
                possible_names = [
                    pred_name,
                    pred_name.replace('_ta2v', ''),
                    pred_name.replace('_ia2v', ''),
                    pred_name.split('_')[0] + '.mp4'
                ]
                
                # Handle special case: RD_Radio16_000_RD_Radio16_000.wav_repeat-0_ia2v.mp4 -> RD_Radio16_000.mp4
                if '_' in pred_name:
                    parts = pred_name.split('_')
                    if len(parts) >= 3:
                        base_name = f"{parts[0]}_{parts[1]}_{parts[2]}.mp4"
                        possible_names.append(base_name)
                
                for name in possible_names:
                    gt_path = os.path.join(gt_dir, name)
                    if os.path.exists(gt_path):
                        gt_file = gt_path
                        break
            
            # Calculate metrics
            metrics = self.calculate_video_metrics(pred_file, gt_file, region=region)
            results.append(metrics)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to a JSON file, including average statistics."""
        
        # Calculate averages
        if results:
            average_metrics = self._calculate_average_metrics(results)
            
            # Create the full result object including averages
            output_data = {
                "summary": {
                    "total_videos": len(results),
                    "successful_videos": sum(1 for r in results if r.get('error') is None),
                    "average_metrics": average_metrics
                },
                "individual_results": results
            }
        else:
            output_data = {
                "summary": {
                    "total_videos": 0,
                    "successful_videos": 0,
                    "average_metrics": {}
                },
                "individual_results": []
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_path}")
        
        # Print average summary
        if results and average_metrics:
            print(f"\n📊 Average Summary:")
            for category, metrics in average_metrics.items():
                if metrics:
                    print(f"  {category}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}")
                        else:
                            print(f"    {metric}: {value}")
    
    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate the average of all metrics."""
        
        # Define metric groups
        metric_groups = {
            "Basic Information": ['frame_count', 'width', 'height', 'fps', 'duration_seconds'],
            "Image Statistics": ['mean_brightness', 'mean_contrast', 'mean_saturation', 'sharpness_score'],
            "Face Analysis": ['face_detection_rate', 'avg_face_size', 'face_stability'],
            "Motion Analysis": ['motion_intensity', 'frame_difference'],
            "LSE Metrics": ['lse_distance', 'lse_confidence'],
            "VBench Metrics": ['subject_consistency', 'background_consistency', 'motion_smoothness',
                         'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
            "Comparison Metrics": ['psnr', 'ssim', 'lpips']
        }
        
        # Filter successful results
        successful_results = [r for r in results if r.get('error') is None]
        
        if not successful_results:
            return {}
        
        average_metrics = {}
        
        for group_name, metrics in metric_groups.items():
            group_averages = {}
            
            for metric in metrics:
                # Collect valid numerical values
                values = []
                for result in successful_results:
                    value = result.get(metric)
                    if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
                
                # Calculate the average
                if values:
                    group_averages[metric] = np.mean(values)
            
            if group_averages:
                average_metrics[group_name] = group_averages
        
        return average_metrics
    
    def print_summary_stats(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n📊 Metrics Summary Statistics ({len(results)} videos)")
        print("=" * 60)
        
        # Statistics on success rate
        total_videos = len(results)
        successful_videos = sum(1 for r in results if r.get('error') is None)
        
        print(f"Total videos: {total_videos}")
        print(f"Successfully processed: {successful_videos}")
        print(f"Success rate: {successful_videos/total_videos:.2%}")
        
        # Statistics on various metrics
        stats = {}
        
        metric_keys = ['face_psnr', 'face_ssim', 'face_lpips', 
                      'lse_distance', 'lse_confidence',
                      'subject_consistency', 'background_consistency', 'motion_smoothness',
                      'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
        
        for key in metric_keys:
            values = [r[key] for r in results if r.get(key) is not None and r.get('error') is None]
            if values:
                stats[key] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Print statistics
        if stats:
            print(f"\n📈 Metrics Statistics:")
            
            # Display by groups
            print(f"\n🔍 Image Quality Metrics (Face Region):")
            for key in ['psnr', 'ssim', 'lpips']:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🎵 LSE Metrics:")
            for key in ['lse_distance', 'lse_confidence']:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🔥 VBench Metrics:")
            vbench_keys = ['subject_consistency', 'background_consistency', 'motion_smoothness',
                          'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
            for key in vbench_keys:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🎨 CLIP Similarity Metrics:")
            clip_keys = ['clip_similarity', 'clip_similarity_std']
            for key in clip_keys:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n📊 FVD Metrics:")
            fvd_keys = ['fvd_score']
            for key in fvd_keys:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🔍 GIM Matching Metrics:")
            gim_keys = ['gim_matching_pixels', 'gim_avg_matching']
            for key in gim_keys:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    Count: {stat['count']}")
                    print(f"    Mean: {stat['mean']:.4f}")
                    print(f"    Std: {stat['std']:.4f}")
                    print(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vbench_calculator') and self.vbench_calculator:
            print("🗑️ Cleaning up VBench resources...")
            try:
                self.vbench_calculator.cleanup()
            except Exception as e:
                print(f"⚠️ Failed to clean up VBench resources: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Video Metrics Calculator")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted video directory")
    parser.add_argument("--gt_dir", type=str, help="Ground truth video directory")
    parser.add_argument("--output", type=str, default="metrics_results.json", help="Output JSON file")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="File matching pattern")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computing device")
    parser.add_argument("--vbench", action="store_true", help="Enable VBench metrics calculation")
    parser.add_argument("--clip", action="store_true", help="Enable CLIP similarity calculation")
    parser.add_argument("--fvd", action="store_true", help="Enable FVD calculation")
    parser.add_argument("--gim", action="store_true", help="Enable GIM matching calculation")
    parser.add_argument("--lse", action="store_true", help="Enable LSE (Lip-Sync Error) calculation")
    parser.add_argument("--all_advanced", action="store_true", help="Enable all advanced metrics (CLIP, FVD, GIM, LSE)")
    parser.add_argument("--region", type=str, default="face_only", choices=["face_only", "full_image"], help="Region for PSNR/SSIM/LPIPS calculation")
    
    args = parser.parse_args()
    
    # Handle --all_advanced flag
    if args.all_advanced:
        args.clip = True
        args.fvd = True
        args.gim = True
        args.lse = True
    
    print("🚀 Starting Comprehensive Video Metrics Calculator")
    
    # Create calculator
    calculator = VideoMetricsCalculator(
        device=args.device, 
        enable_vbench=args.vbench,
        enable_clip_similarity=args.clip,
        enable_fvd=args.fvd,
        enable_gim_matching=args.gim,
        enable_lse=args.lse
    )
    
    try:
        # Batch calculate metrics
        results = calculator.calculate_batch_metrics(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            pattern=args.pattern,
            region=args.region
        )
        
        # Save results
        calculator.save_results(results, args.output)
        
        # Print summary statistics
        calculator.print_summary_stats(results)
        
        print("\n🎉 Calculation completed!")
        
    finally:
        # Clean up resources
        calculator.cleanup()


if __name__ == "__main__":
    main()