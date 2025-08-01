#!/usr/bin/env python3
"""
CLIP Similarity Calculator
Calculate CLIP-V similarity between source and target frames at the same timestamp

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides CLIP-based video similarity calculation for video evaluation.

Usage:
    from evalutation.clip_similarity_calculator import CLIPSimilarityCalculator
    
    calculator = CLIPSimilarityCalculator()
    clip_similarity = calculator.calculate_video_similarity("source.mp4", "target.mp4")
"""

import os
import cv2
import torch
import numpy as np
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import clip
from PIL import Image
from tqdm import tqdm


class CLIPSimilarityCalculator:
    """CLIP Similarity Calculator for video evaluation"""
    
    def __init__(self, 
                 device: str = "cuda",
                 model_name: str = "ViT-B/32",
                 batch_size: int = 16):
        """
        Initialize CLIP similarity calculator
        
        Args:
            device: Computing device ("cuda" or "cpu")
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            batch_size: Batch size for processing frames
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        print(f"🔄 Initializing CLIP model: {model_name}")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            print(f"✅ CLIP model initialized successfully (Device: {self.device})")
        except Exception as e:
            print(f"❌ Failed to initialize CLIP model: {e}")
            raise
    
    def calculate_video_similarity(self, 
                                 source_path: str, 
                                 target_path: str,
                                 max_frames: Optional[int] = None,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate CLIP-V similarity between source and target videos
        
        Args:
            source_path: Path to source video
            target_path: Path to target video
            max_frames: Maximum number of frames to process (None for all)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing similarity metrics
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source video not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target video not found: {target_path}")
        
        if verbose:
            print(f"🎬 Calculating CLIP similarity")
            print(f"   Source: {os.path.basename(source_path)}")
            print(f"   Target: {os.path.basename(target_path)}")
        
        try:
            # Extract frames from both videos
            source_frames = self._extract_frames(source_path, max_frames, verbose)
            target_frames = self._extract_frames(target_path, max_frames, verbose)
            
            if not source_frames or not target_frames:
                return {"clip_similarity": None, "error": "Failed to extract frames"}
            
            # Align frame counts
            min_frames = min(len(source_frames), len(target_frames))
            source_frames = source_frames[:min_frames]
            target_frames = target_frames[:min_frames]
            
            if verbose:
                print(f"   Processing {min_frames} frame pairs...")
            
            # Calculate CLIP features
            source_features = self._extract_clip_features(source_frames, verbose)
            target_features = self._extract_clip_features(target_frames, verbose)
            
            # Calculate similarities
            similarities = self._calculate_similarities(source_features, target_features)
            
            # Compute statistics
            results = {
                "clip_similarity": float(np.mean(similarities)),
                "clip_similarity_std": float(np.std(similarities)),
                "clip_similarity_min": float(np.min(similarities)),
                "clip_similarity_max": float(np.max(similarities)),
                "frame_count": len(similarities),
                "frame_similarities": similarities.tolist(),
                "error": None
            }
            
            if verbose:
                print(f"   ✅ CLIP similarity calculated: {results['clip_similarity']:.4f}")
                print(f"      Standard deviation: {results['clip_similarity_std']:.4f}")
                print(f"      Range: [{results['clip_similarity_min']:.4f}, {results['clip_similarity_max']:.4f}]")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to calculate CLIP similarity: {e}"
            if verbose:
                print(f"   ❌ {error_msg}")
            return {"clip_similarity": None, "error": error_msg}
    
    def _extract_frames(self, video_path: str, max_frames: Optional[int], verbose: bool) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            step = max(1, total_frames // max_frames)
        else:
            step = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        if verbose:
            print(f"      Extracted {len(frames)} frames from {os.path.basename(video_path)}")
        
        return frames
    
    def _extract_clip_features(self, frames: List[np.ndarray], verbose: bool) -> torch.Tensor:
        """Extract CLIP features from frames"""
        features = []
        
        # Process frames in batches
        for i in tqdm(range(0, len(frames), self.batch_size), 
                      desc="Extracting CLIP features", 
                      disable=not verbose):
            batch_frames = frames[i:i + self.batch_size]
            
            # Preprocess frames
            batch_images = []
            for frame in batch_frames:
                pil_image = Image.fromarray(frame)
                preprocessed = self.preprocess(pil_image)
                batch_images.append(preprocessed)
            
            # Stack and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            features.append(batch_features.cpu())
        
        # Concatenate all features
        all_features = torch.cat(features, dim=0)
        return all_features
    
    def _calculate_similarities(self, source_features: torch.Tensor, target_features: torch.Tensor) -> np.ndarray:
        """Calculate cosine similarities between corresponding frames"""
        # Ensure same number of features
        min_features = min(len(source_features), len(target_features))
        source_features = source_features[:min_features]
        target_features = target_features[:min_features]
        
        # Calculate cosine similarity for each pair
        similarities = torch.cosine_similarity(source_features, target_features, dim=1)
        return similarities.numpy()
    
    def calculate_batch_similarity(self, 
                                 video_pairs: List[Tuple[str, str]], 
                                 max_frames: Optional[int] = None,
                                 verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Calculate CLIP similarities for multiple video pairs
        
        Args:
            video_pairs: List of (source_path, target_path) tuples
            max_frames: Maximum frames per video
            verbose: Whether to print progress
            
        Returns:
            Dictionary with pair identifiers as keys and similarity results as values
        """
        results = {}
        
        if verbose:
            print(f"🚀 Starting batch CLIP similarity calculation for {len(video_pairs)} pairs")
        
        for i, (source_path, target_path) in enumerate(video_pairs, 1):
            if verbose:
                print(f"\n[{i}/{len(video_pairs)}] Processing pair:")
            
            pair_id = f"{os.path.basename(source_path)}_vs_{os.path.basename(target_path)}"
            
            try:
                similarity_result = self.calculate_video_similarity(
                    source_path, target_path, max_frames, verbose
                )
                results[pair_id] = similarity_result
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed: {e}")
                results[pair_id] = {"clip_similarity": None, "error": str(e)}
        
        if verbose:
            print(f"\n✅ Batch calculation completed")
            success_count = sum(1 for r in results.values() if r.get("clip_similarity") is not None)
            print(f"   Success: {success_count}/{len(video_pairs)}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save similarity results to JSON file"""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_results[key] = convert_numpy(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ CLIP similarity results saved to: {output_path}")


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP Similarity Calculator")
    parser.add_argument("--source", type=str, required=True, help="Source video file")
    parser.add_argument("--target", type=str, required=True, help="Target video file")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computing device")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")
    parser.add_argument("--output", type=str, default="clip_similarity_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = CLIPSimilarityCalculator(
        device=args.device,
        model_name=args.model
    )
    
    # Calculate similarity
    results = calculator.calculate_video_similarity(
        args.source, 
        args.target, 
        max_frames=args.max_frames,
        verbose=True
    )
    
    # Save results
    calculator.save_results({"comparison": results}, args.output)
    
    # Print final results
    if results.get("clip_similarity") is not None:
        print(f"\n📊 Final Results:")
        print(f"   CLIP Similarity: {results['clip_similarity']:.4f}")
        print(f"   Standard Deviation: {results['clip_similarity_std']:.4f}")
        print(f"   Frame Count: {results['frame_count']}")
    else:
        print(f"\n❌ CLIP similarity calculation failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()