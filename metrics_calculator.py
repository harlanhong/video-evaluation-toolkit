#!/usr/bin/env python3
"""
综合视频指标计算器 (VBench集成版本)
整合了LSE计算、VBench指标和其他视频质量指标

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module integrates VBench metrics with comprehensive video evaluation tools.

支持的指标:
- 视频基本信息 (不需要GT): 帧数、分辨率、帧率、时长
- 图像统计信息 (不需要GT): 亮度、对比度、饱和度、清晰度
- 人脸分析 (不需要GT): 人脸检测率、平均人脸大小、人脸稳定性
- 运动分析 (不需要GT): 运动强度、帧间差异
- 人脸区域图像质量 (需要GT): face_psnr, face_ssim, face_lpips
- 唇同步指标 (不需要GT): LSE distance, LSE confidence
- VBench指标 (不需要GT): subject_consistency, background_consistency, motion_smoothness, 
  dynamic_degree, aesthetic_quality, imaging_quality

使用方法:
    from evalutation.metrics_calculator import VideoMetricsCalculator
    
    # 不包含VBench
    calculator = VideoMetricsCalculator()
    metrics = calculator.calculate_video_metrics("video.mp4")
    
    # 包含VBench
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

# 图像质量指标
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# LSE计算器
try:
    # 作为包导入时使用相对导入
    from .lse_calculator import LSECalculator
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from lse_calculator import LSECalculator

# VBench计算器
try:
    # 作为包导入时使用相对导入
    from .vbench_official_final import VBenchDirect
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from vbench_official_final import VBenchDirect


class VideoMetricsCalculator:
    """综合视频指标计算器"""
    
    def __init__(self, device: str = "cuda", enable_vbench: bool = False):
        """
        初始化指标计算器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
            enable_vbench: 是否启用VBench指标计算
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_vbench = enable_vbench
        
        # 初始化LPIPS模型
        print("🔄 正在初始化LPIPS模型...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # 初始化人脸检测器
        print("🔄 正在初始化人脸检测器...")
        self.face_detection_method = self._initialize_face_detector()
        self.face_detection_available = self.face_detection_method is not None
        
        if self.face_detection_available:
            print(f"✅ 人脸检测器初始化成功 (方法: {self.face_detection_method})")
        else:
            print("⚠️ 所有人脸检测器初始化失败，将跳过人脸相关指标")
        
        # 初始化LSE计算器
        print("🔄 正在初始化LSE计算器...")
        try:
            self.lse_calculator = LSECalculator(device=self.device)
            self.lse_available = True
        except Exception as e:
            print(f"⚠️ LSE计算器初始化失败: {e}")
            self.lse_available = False
        
        # 初始化VBench计算器
        if self.enable_vbench:
            print("🔄 正在初始化VBench计算器...")
            try:
                self.vbench_calculator = VBenchDirect(device=self.device)
                self.vbench_available = True
                print("✅ VBench计算器初始化成功")
            except Exception as e:
                print(f"⚠️ VBench计算器初始化失败: {e}")
                self.vbench_available = False
        else:
            self.vbench_available = False
        
        print(f"🚀 指标计算器初始化完成 (设备: {self.device}, VBench: {self.vbench_available})")
    
    def calculate_video_metrics(self, 
                               pred_path: str, 
                               gt_path: Optional[str] = None,
                               audio_path: Optional[str] = None) -> Dict[str, Any]:
        """
        计算单个视频的所有指标
        
        Args:
            pred_path: 预测视频路径
            gt_path: 真值视频路径 (可选)
            audio_path: 音频文件路径 (可选，已弃用)
            
        Returns:
            包含所有指标的字典
        """
        
        # 初始化指标字典
        metrics = {
            'video_path': pred_path,
            'has_ground_truth': gt_path is not None and os.path.exists(gt_path) if gt_path else False,
            'has_audio': False,  # 不再需要外部音频文件
            'vbench_enabled': self.vbench_available,
            
            # 基本信息（不需要ground truth）
            'frame_count': 0,
            'width': 0,
            'height': 0,
            'fps': 0.0,
            'duration_seconds': 0.0,
            
            # 图像统计信息（不需要ground truth）
            'mean_brightness': 0.0,
            'mean_contrast': 0.0,
            'mean_saturation': 0.0,
            'sharpness_score': 0.0,
            
            # 人脸检测统计（不需要ground truth）
            'face_detection_rate': 0.0,  # 检测到人脸的帧比例
            'avg_face_size': 0.0,        # 平均人脸大小
            'face_stability': 0.0,       # 人脸位置稳定性
            
            # 运动分析（不需要ground truth）
            'motion_intensity': 0.0,     # 运动强度
            'frame_difference': 0.0,     # 平均帧间差异
            
            # 需要ground truth的指标（只计算人脸区域）
            'face_psnr': None,        # 人脸区域PSNR
            'face_ssim': None,        # 人脸区域SSIM  
            'face_lpips': None,       # 人脸区域LPIPS
            
            # 唇同步指标（使用LSE计算器，不需要外部音频）
            'lse_distance': None,     # LSE距离分数
            'lse_confidence': None,   # LSE置信度分数
            
            # VBench指标（不需要ground truth）
            'subject_consistency': None,     # 主体一致性
            'background_consistency': None,  # 背景一致性
            'motion_smoothness': None,       # 运动平滑性
            'dynamic_degree': None,          # 动态程度
            'aesthetic_quality': None,       # 美学质量
            'imaging_quality': None,         # 成像质量
            
            'error': None
        }
        
        try:
            # 打开视频文件
            pred_cap = cv2.VideoCapture(pred_path)
            if not pred_cap.isOpened():
                metrics['error'] = f"无法打开预测视频: {pred_path}"
                return metrics
            
            # 获取视频基本信息
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
                
                # 计算图像统计信息
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                
                # 亮度（灰度均值）
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # 对比度（灰度标准差）
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                # 饱和度（HSV空间的S通道均值）
                hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1])
                saturation_values.append(saturation)
                
                # 清晰度（拉普拉斯方差）
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_values.append(sharpness)
                
                # 人脸检测
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
            
            # 计算不需要ground truth的指标
            if brightness_values:
                metrics['mean_brightness'] = np.mean(brightness_values)
                metrics['mean_contrast'] = np.mean(contrast_values)
                metrics['mean_saturation'] = np.mean(saturation_values)
                metrics['sharpness_score'] = np.mean(sharpness_values)
            
            # 人脸检测统计
            if face_detections:
                metrics['face_detection_rate'] = len(face_detections) / frame_count
                metrics['avg_face_size'] = np.mean([fd['size'] for fd in face_detections])
                
                # 计算人脸位置稳定性（相邻帧人脸中心点距离的方差）
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
                    
                    # 稳定性 = 1 / (1 + 平均距离)，值越大越稳定
                    metrics['face_stability'] = 1.0 / (1.0 + np.mean(distances)) if distances else 1.0
            
            # 运动分析
            if len(pred_frames) > 1:
                frame_diffs = []
                for i in range(1, len(pred_frames)):
                    diff = np.mean(np.abs(pred_frames[i].astype(float) - pred_frames[i-1].astype(float)))
                    frame_diffs.append(diff)
                
                metrics['motion_intensity'] = np.std(frame_diffs) if frame_diffs else 0.0
                metrics['frame_difference'] = np.mean(frame_diffs) if frame_diffs else 0.0
            
            # 计算LSE分数（使用LSE计算器）
            if self.lse_available:
                print(f"🎵 计算LSE分数 (使用LSE计算器)")
                try:
                    lse_distance, lse_confidence = self.lse_calculator.calculate_single_video(pred_path, verbose=False)
                    metrics['lse_distance'] = lse_distance
                    metrics['lse_confidence'] = lse_confidence
                except Exception as e:
                    print(f"⚠️ LSE计算失败: {e}")
                    metrics['lse_distance'] = None
                    metrics['lse_confidence'] = None
            else:
                print(f"⚠️ LSE计算器不可用，跳过LSE计算")
            
            # 计算VBench分数
            if self.vbench_available:
                print(f"🔥 计算VBench指标 (6个核心指标)")
                try:
                    vbench_results = self.vbench_calculator.evaluate_videos([pred_path])
                    
                    # 将VBench结果合并到metrics中
                    for metric_name in ['subject_consistency', 'background_consistency', 'motion_smoothness', 
                                      'dynamic_degree', 'aesthetic_quality', 'imaging_quality']:
                        if metric_name in vbench_results:
                            metrics[metric_name] = vbench_results[metric_name]
                            print(f"   ✅ {metric_name}: {metrics[metric_name]}")
                        else:
                            print(f"   ❌ VBench结果中未找到: {metric_name}")
                            metrics[metric_name] = None
                    
                except Exception as e:
                    print(f"⚠️ VBench计算失败: {e}")
                    for metric_name in ['subject_consistency', 'background_consistency', 'motion_smoothness', 
                                      'dynamic_degree', 'aesthetic_quality', 'imaging_quality']:
                        metrics[metric_name] = None
            else:
                print(f"⚠️ VBench计算器不可用，跳过VBench计算")
            
            # 如果有真值视频，计算对比指标（只计算人脸区域）
            if gt_path and os.path.exists(gt_path):
                print(f"🔍 计算人脸区域对比指标 (与真值对比)")
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
                    
                    # 计算帧级别指标
                    frame_metrics_list = []
                    min_frames = min(len(pred_frames), len(gt_frames))
                    
                    for i in range(min_frames):
                        frame_metrics = self.calculate_frame_metrics(pred_frames[i], gt_frames[i])
                        if frame_metrics:
                            frame_metrics_list.append(frame_metrics)
                    
                    if frame_metrics_list:
                        # 计算平均值
                        for key in ['face_psnr', 'face_ssim', 'face_lpips']:
                            values = [fm.get(key, 0.0) for fm in frame_metrics_list if fm.get(key, 0.0) > 0]
                            if values:
                                metrics[key] = np.mean(values)
                else:
                    print(f"⚠️ 无法打开真值视频: {gt_path}")
            else:
                print(f"⚠️ 无真值视频，跳过对比指标计算")
        
        except Exception as e:
            metrics['error'] = str(e)
            print(f"计算视频指标时出错: {e}")
        
        return metrics
    
    def calculate_frame_metrics(self, pred_frame: np.ndarray, gt_frame: np.ndarray) -> Dict[str, float]:
        """计算单帧指标（只计算人脸区域）"""
        
        # 检测人脸区域
        pred_face_bbox = self.detect_face_bbox(pred_frame)
        gt_face_bbox = self.detect_face_bbox(gt_frame)
        
        metrics = {
            'face_psnr': 0.0,
            'face_ssim': 0.0,
            'face_lpips': 0.0
        }
        
        # 如果两帧都检测到人脸，计算人脸区域指标
        if pred_face_bbox is not None and gt_face_bbox is not None:
            # 使用较大的bbox确保包含完整人脸
            x1 = min(pred_face_bbox[0], gt_face_bbox[0])
            y1 = min(pred_face_bbox[1], gt_face_bbox[1])
            x2 = max(pred_face_bbox[0] + pred_face_bbox[2], gt_face_bbox[0] + gt_face_bbox[2])
            y2 = max(pred_face_bbox[1] + pred_face_bbox[3], gt_face_bbox[1] + gt_face_bbox[3])
            
            # 确保坐标在图像范围内
            h, w = pred_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:  # 确保有效区域
                pred_face = pred_frame[y1:y2, x1:x2]
                gt_face = gt_frame[y1:y2, x1:x2]
                
                # 计算人脸区域PSNR
                try:
                    face_psnr = peak_signal_noise_ratio(gt_face, pred_face, data_range=255)
                    metrics['face_psnr'] = face_psnr
                except:
                    metrics['face_psnr'] = 0.0
                
                # 计算人脸区域SSIM
                try:
                    face_ssim = structural_similarity(
                        gt_face, pred_face, 
                        multichannel=True, 
                        data_range=255,
                        channel_axis=-1
                    )
                    metrics['face_ssim'] = face_ssim
                except:
                    metrics['face_ssim'] = 0.0
                
                # 计算人脸区域LPIPS
                try:
                    # 转换为tensor格式 [1, C, H, W]
                    pred_tensor = torch.from_numpy(pred_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    gt_tensor = torch.from_numpy(gt_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    
                    # 归一化到[-1, 1]
                    pred_tensor = pred_tensor * 2.0 - 1.0
                    gt_tensor = gt_tensor * 2.0 - 1.0
                    
                    pred_tensor = pred_tensor.to(self.device)
                    gt_tensor = gt_tensor.to(self.device)
                    
                    face_lpips = self.lpips_fn(pred_tensor, gt_tensor).item()
                    metrics['face_lpips'] = face_lpips
                except:
                    metrics['face_lpips'] = 0.0
        
        return metrics
    
    def _initialize_face_detector(self) -> Optional[str]:
        """初始化人脸检测器，按优先级尝试不同方法"""
        
        # 方法1: MediaPipe Face Detection (推荐)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detector_mp = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            print("   🚀 使用MediaPipe人脸检测 (推荐)")
            return "mediapipe"
        except ImportError:
            print("   ⚠️ MediaPipe未安装，尝试其他方法...")
        except Exception as e:
            print(f"   ⚠️ MediaPipe初始化失败: {e}")
        
        # 方法2: YOLOv8 Face Detection
        try:
            from ultralytics import YOLO
            # 尝试加载YOLOv8n-face模型（如果可用）
            self.yolo_model = YOLO('yolov8n-face.pt')
            print("   ⚡ 使用YOLOv8人脸检测")
            return "yolov8"
        except ImportError:
            print("   ⚠️ ultralytics未安装，尝试其他方法...")
        except Exception as e:
            print(f"   ⚠️ YOLOv8初始化失败: {e}")
        
        # 方法3: OpenCV DNN Face Detection
        try:
            # 使用OpenCV DNN模块加载预训练的人脸检测模型
            # 这里可以使用SSD MobileNet或ResNet模型
            self.face_net = cv2.dnn.readNetFromTensorflow(
                # 需要下载模型文件
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            print("   🔧 使用OpenCV DNN人脸检测")
            return "opencv_dnn"
        except Exception as e:
            print(f"   ⚠️ OpenCV DNN初始化失败: {e}")
        
        # 方法4: 传统Haar级联 (fallback)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("   📰 使用传统Haar级联分类器 (fallback)")
            return "haar_cascade"
        except Exception as e:
            print(f"   ❌ Haar级联初始化失败: {e}")
        
        return None
    
    def detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测人脸边界框 - 支持多种现代检测器"""
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
            print(f"⚠️ 人脸检测失败 ({self.face_detection_method}): {e}")
            return None
        
        return None
    
    def _detect_face_mediapipe(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """使用MediaPipe检测人脸"""
        # frame已经是RGB格式，不需要转换
        results = self.face_detector_mp.process(frame)
        
        if results.detections:
            # 获取置信度最高的人脸
            best_detection = max(results.detections, key=lambda d: d.score[0])
            bbox = best_detection.location_data.relative_bounding_box
            
            # 转换为绝对坐标
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # 确保坐标在图像范围内
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = max(1, min(width, w - x))
            height = max(1, min(height, h - y))
            
            return (x, y, width, height)
        
        return None
    
    def _detect_face_yolov8(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """使用YOLOv8检测人脸"""
        results = self.yolo_model(frame, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # 获取置信度最高的检测结果
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            # 获取边界框坐标 (x1, y1, x2, y2)
            box = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # 转换为 (x, y, w, h) 格式
            x = int(x1)
            y = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            return (x, y, width, height)
        
        return None
    
    def _detect_face_opencv_dnn(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """使用OpenCV DNN检测人脸"""
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2 - x1, y2 - y1)
        
        return best_box
    
    def _detect_face_haar_cascade(self, frame: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        """使用传统Haar级联检测人脸"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # 返回最大的人脸
            areas = [face_w * face_h for (x, y, face_w, face_h) in faces]
            max_idx = np.argmax(areas)
            return tuple(faces[max_idx])
        
        return None
    
    def calculate_batch_metrics(self, 
                               pred_dir: str, 
                               gt_dir: Optional[str] = None,
                               pattern: str = "*.mp4") -> List[Dict[str, Any]]:
        """
        批量计算视频指标
        
        Args:
            pred_dir: 预测视频目录
            gt_dir: 真值视频目录 (可选)
            pattern: 文件匹配模式
            
        Returns:
            指标结果列表
        """
        
        # 获取预测视频文件列表
        pred_files = sorted(glob.glob(os.path.join(pred_dir, pattern)))
        if not pred_files:
            print(f"⚠️ 在 {pred_dir} 中没有找到匹配 {pattern} 的文件")
            return []
        
        print(f"🔍 找到 {len(pred_files)} 个视频文件")
        
        results = []
        
        for pred_file in tqdm(pred_files, desc="计算指标"):
            pred_name = os.path.basename(pred_file)
            
            # 查找对应的真值视频
            gt_file = None
            if gt_dir:
                # 尝试几种可能的命名方式
                possible_names = [
                    pred_name,
                    pred_name.replace('_ta2v', ''),
                    pred_name.replace('_ia2v', ''),
                    pred_name.split('_')[0] + '.mp4'
                ]
                
                for name in possible_names:
                    gt_path = os.path.join(gt_dir, name)
                    if os.path.exists(gt_path):
                        gt_file = gt_path
                        break
            
            # 计算指标
            metrics = self.calculate_video_metrics(pred_file, gt_file)
            results.append(metrics)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果到JSON文件，包含平均值统计"""
        
        # 计算平均值
        if results:
            average_metrics = self._calculate_average_metrics(results)
            
            # 创建包含平均值的完整结果
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
        print(f"✅ 结果已保存到: {output_path}")
        
        # 打印平均值摘要
        if results and average_metrics:
            print(f"\n📊 平均值摘要:")
            for category, metrics in average_metrics.items():
                if metrics:
                    print(f"  {category}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}")
                        else:
                            print(f"    {metric}: {value}")
    
    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """计算所有指标的平均值"""
        
        # 定义指标分组
        metric_groups = {
            "基本信息": ['frame_count', 'width', 'height', 'fps', 'duration_seconds'],
            "图像统计": ['mean_brightness', 'mean_contrast', 'mean_saturation', 'sharpness_score'],
            "人脸分析": ['face_detection_rate', 'avg_face_size', 'face_stability'],
            "运动分析": ['motion_intensity', 'frame_difference'],
            "LSE指标": ['lse_distance', 'lse_confidence'],
            "VBench指标": ['subject_consistency', 'background_consistency', 'motion_smoothness',
                         'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
            "对比指标": ['face_psnr', 'face_ssim', 'face_lpips']
        }
        
        # 过滤成功的结果
        successful_results = [r for r in results if r.get('error') is None]
        
        if not successful_results:
            return {}
        
        average_metrics = {}
        
        for group_name, metrics in metric_groups.items():
            group_averages = {}
            
            for metric in metrics:
                # 收集有效的数值
                values = []
                for result in successful_results:
                    value = result.get(metric)
                    if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
                
                # 计算平均值
                if values:
                    group_averages[metric] = np.mean(values)
            
            if group_averages:
                average_metrics[group_name] = group_averages
        
        return average_metrics
    
    def print_summary_stats(self, results: List[Dict[str, Any]]):
        """打印汇总统计信息"""
        
        if not results:
            print("❌ 没有结果可以统计")
            return
        
        print(f"\n📊 指标汇总统计 ({len(results)} 个视频)")
        print("=" * 60)
        
        # 统计成功率
        total_videos = len(results)
        successful_videos = sum(1 for r in results if r.get('error') is None)
        
        print(f"视频总数: {total_videos}")
        print(f"成功处理: {successful_videos}")
        print(f"成功率: {successful_videos/total_videos:.2%}")
        
        # 统计各类指标
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
        
        # 打印统计信息
        if stats:
            print(f"\n📈 指标统计:")
            
            # 分组显示
            print(f"\n🔍 人脸对比指标:")
            for key in ['face_psnr', 'face_ssim', 'face_lpips']:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    样本数: {stat['count']}")
                    print(f"    均值: {stat['mean']:.4f}")
                    print(f"    标准差: {stat['std']:.4f}")
                    print(f"    范围: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🎵 LSE指标:")
            for key in ['lse_distance', 'lse_confidence']:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    样本数: {stat['count']}")
                    print(f"    均值: {stat['mean']:.4f}")
                    print(f"    标准差: {stat['std']:.4f}")
                    print(f"    范围: [{stat['min']:.4f}, {stat['max']:.4f}]")
            
            print(f"\n🔥 VBench指标:")
            vbench_keys = ['subject_consistency', 'background_consistency', 'motion_smoothness',
                          'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
            for key in vbench_keys:
                if key in stats:
                    stat = stats[key]
                    print(f"  {key}:")
                    print(f"    样本数: {stat['count']}")
                    print(f"    均值: {stat['mean']:.4f}")
                    print(f"    标准差: {stat['std']:.4f}")
                    print(f"    范围: [{stat['min']:.4f}, {stat['max']:.4f}]")

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'vbench_calculator') and self.vbench_calculator:
            print("🗑️ 清理VBench资源...")
            try:
                self.vbench_calculator.cleanup()
            except Exception as e:
                print(f"⚠️ 清理VBench资源失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合视频指标计算器")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测视频目录")
    parser.add_argument("--gt_dir", type=str, help="真值视频目录")
    parser.add_argument("--output", type=str, default="metrics_results.json", help="输出JSON文件")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="文件匹配模式")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--vbench", action="store_true", help="启用VBench指标计算")
    
    args = parser.parse_args()
    
    print("🚀 启动综合视频指标计算器")
    
    # 创建计算器
    calculator = VideoMetricsCalculator(device=args.device, enable_vbench=args.vbench)
    
    try:
        # 批量计算指标
        results = calculator.calculate_batch_metrics(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            pattern=args.pattern
        )
        
        # 保存结果
        calculator.save_results(results, args.output)
        
        # 打印汇总统计
        calculator.print_summary_stats(results)
        
        print("\n🎉 计算完成!")
        
    finally:
        # 清理资源
        calculator.cleanup()


if __name__ == "__main__":
    main() 