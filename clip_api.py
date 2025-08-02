#!/usr/bin/env python3
"""
CLIP API - Unified CLIP-based Video Evaluation Interface
Provides comprehensive CLIP-based metrics for video evaluation and analysis

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides a unified API for all CLIP-based video evaluation metrics.

Usage:
    from evalutation.clip_api import CLIPVideoAPI
    
    # Initialize API
    clip_api = CLIPVideoAPI(model_name="ViT-B/32")
    
    # Calculate frame similarity
    similarity = clip_api.calculate_frame_similarity(frame1, frame2)
    
    # Calculate video similarity
    video_similarity = clip_api.calculate_video_similarity("video1.mp4", "video2.mp4")
    
    # Extract video features
    features = clip_api.extract_video_features("video.mp4")
"""

import os
import cv2
import torch
import numpy as np
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import clip
from PIL import Image
from tqdm import tqdm
import json


class CLIPVideoAPI:
    """Unified CLIP API for video evaluation and analysis"""
    
    # Available CLIP models
    AVAILABLE_MODELS = [
        "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
        "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    ]
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: str = "cuda",
                 batch_size: int = 16):
        """
        Initialize CLIP Video API
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            device: Computing device ("cuda" or "cpu")
            batch_size: Batch size for processing frames
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model_name = model_name
        
        if model_name not in self.AVAILABLE_MODELS:
            print(f"⚠️ Model {model_name} not in available models: {self.AVAILABLE_MODELS}")
            print(f"   Using default model: ViT-B/32")
            self.model_name = "ViT-B/32"
        
        print(f"🔄 Initializing CLIP API with model: {self.model_name}")
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(f"✅ CLIP API initialized successfully (Device: {self.device})")
        except Exception as e:
            print(f"❌ Failed to initialize CLIP API: {e}")
            raise
    
    def extract_image_features(self, 
                              images: Union[np.ndarray, List[np.ndarray], str, List[str]],
                              normalize: bool = True) -> torch.Tensor:
        """
        Extract CLIP features from images
        
        Args:
            images: Single image, list of images, or image paths
            normalize: Whether to normalize features
            
        Returns:
            CLIP features tensor
        """
        # Handle different input types
        if isinstance(images, str):
            # Single image path
            images = [self._load_image_from_path(images)]
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            # List of image paths
            images = [self._load_image_from_path(path) for path in images]
        elif isinstance(images, np.ndarray):
            # Single image array
            images = [images]
        
        if not images:
            raise ValueError("No valid images provided")
        
        features = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess images
            batch_tensors = []
            for image in batch_images:
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image
                preprocessed = self.preprocess(pil_image)
                batch_tensors.append(preprocessed)
            
            # Stack and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                if normalize:
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            features.append(batch_features.cpu())
        
        # Concatenate all features
        all_features = torch.cat(features, dim=0)
        return all_features
    
    def extract_text_features(self, 
                             texts: Union[str, List[str]],
                             normalize: bool = True) -> torch.Tensor:
        """
        Extract CLIP features from text
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize features
            
        Returns:
            CLIP text features tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Extract features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu()
    
    def calculate_frame_similarity(self, 
                                  frame1: np.ndarray, 
                                  frame2: np.ndarray) -> float:
        """
        Calculate CLIP similarity between two frames
        
        Args:
            frame1: First frame (RGB numpy array)
            frame2: Second frame (RGB numpy array)
            
        Returns:
            Cosine similarity score
        """
        # Extract features for both frames
        features1 = self.extract_image_features([frame1])
        features2 = self.extract_image_features([frame2])
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(features1, features2, dim=1)
        return float(similarity[0])
    
    def calculate_video_similarity(self, 
                                 source_path: str, 
                                 target_path: str,
                                 max_frames: Optional[int] = None,
                                 frame_step: int = 1,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate CLIP-V similarity between source and target videos
        
        Args:
            source_path: Path to source video
            target_path: Path to target video
            max_frames: Maximum number of frames to process (None for all)
            frame_step: Step size for frame sampling
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing similarity metrics
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source video not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target video not found: {target_path}")
        
        if verbose:
            print(f"🎬 Calculating CLIP video similarity")
            print(f"   Source: {os.path.basename(source_path)}")
            print(f"   Target: {os.path.basename(target_path)}")
        
        try:
            # Extract frames from both videos
            source_frames = self._extract_video_frames(source_path, max_frames, frame_step, verbose)
            target_frames = self._extract_video_frames(target_path, max_frames, frame_step, verbose)
            
            if not source_frames or not target_frames:
                return {"clip_similarity": None, "error": "Failed to extract frames"}
            
            # Align frame counts
            min_frames = min(len(source_frames), len(target_frames))
            source_frames = source_frames[:min_frames]
            target_frames = target_frames[:min_frames]
            
            if verbose:
                print(f"   Processing {min_frames} frame pairs...")
            
            # Calculate CLIP features
            source_features = self.extract_image_features(source_frames, normalize=True)
            target_features = self.extract_image_features(target_frames, normalize=True)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(source_features, target_features, dim=1)
            similarities_np = similarities.numpy()
            
            # Compute statistics
            results = {
                "clip_similarity": float(np.mean(similarities_np)),
                "clip_similarity_std": float(np.std(similarities_np)),
                "clip_similarity_min": float(np.min(similarities_np)),
                "clip_similarity_max": float(np.max(similarities_np)),
                "clip_similarity_median": float(np.median(similarities_np)),
                "frame_count": len(similarities_np),
                "frame_similarities": similarities_np.tolist(),
                "model_name": self.model_name,
                "error": None
            }
            
            if verbose:
                print(f"   ✅ CLIP similarity calculated: {results['clip_similarity']:.4f}")
                print(f"      Standard deviation: {results['clip_similarity_std']:.4f}")
                print(f"      Range: [{results['clip_similarity_min']:.4f}, {results['clip_similarity_max']:.4f}]")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to calculate CLIP video similarity: {e}"
            if verbose:
                print(f"   ❌ {error_msg}")
            return {"clip_similarity": None, "error": error_msg}
    
    def extract_video_features(self, 
                              video_path: str,
                              max_frames: Optional[int] = None,
                              frame_step: int = 1,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Extract CLIP features from video frames
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_step: Step size for frame sampling
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing features and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if verbose:
            print(f"🎬 Extracting CLIP features from: {os.path.basename(video_path)}")
        
        try:
            # Extract frames
            frames = self._extract_video_frames(video_path, max_frames, frame_step, verbose)
            
            if not frames:
                return {"features": None, "error": "Failed to extract frames"}
            
            # Extract CLIP features
            features = self.extract_image_features(frames, normalize=True)
            
            results = {
                "features": features.numpy(),
                "feature_dim": features.shape[1],
                "frame_count": len(frames),
                "model_name": self.model_name,
                "video_path": video_path,
                "error": None
            }
            
            if verbose:
                print(f"   ✅ Extracted {results['frame_count']} features with dimension {results['feature_dim']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to extract video features: {e}"
            if verbose:
                print(f"   ❌ {error_msg}")
            return {"features": None, "error": error_msg}
    
    def calculate_text_video_similarity(self,
                                       video_path: str,
                                       text_queries: Union[str, List[str]],
                                       max_frames: Optional[int] = None,
                                       aggregation: str = "mean",
                                       verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate similarity between video and text descriptions
        
        Args:
            video_path: Path to video file
            text_queries: Text query or list of text queries
            max_frames: Maximum frames to process
            aggregation: How to aggregate frame similarities ("mean", "max", "min")
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing text-video similarity results
        """
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        if verbose:
            print(f"🎬 Calculating text-video similarity")
            print(f"   Video: {os.path.basename(video_path)}")
            print(f"   Queries: {len(text_queries)} texts")
        
        try:
            # Extract video features
            video_result = self.extract_video_features(video_path, max_frames, verbose=False)
            if video_result["features"] is None:
                return {"similarities": None, "error": video_result["error"]}
            
            video_features = torch.from_numpy(video_result["features"])
            
            # Extract text features
            text_features = self.extract_text_features(text_queries, normalize=True)
            
            # Calculate similarities
            similarities_matrix = torch.matmul(video_features, text_features.T)  # (frames, texts)
            
            # Aggregate similarities
            if aggregation == "mean":
                frame_aggregated = torch.mean(similarities_matrix, dim=0)
            elif aggregation == "max":
                frame_aggregated = torch.max(similarities_matrix, dim=0)[0]
            elif aggregation == "min":
                frame_aggregated = torch.min(similarities_matrix, dim=0)[0]
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            results = {
                "similarities": frame_aggregated.numpy().tolist(),
                "similarity_matrix": similarities_matrix.numpy().tolist(),
                "text_queries": text_queries,
                "aggregation": aggregation,
                "frame_count": video_result["frame_count"],
                "model_name": self.model_name,
                "error": None
            }
            
            if verbose:
                for i, (query, sim) in enumerate(zip(text_queries, results["similarities"])):
                    print(f"   Query {i+1}: {sim:.4f} - '{query}'")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to calculate text-video similarity: {e}"
            if verbose:
                print(f"   ❌ {error_msg}")
            return {"similarities": None, "error": error_msg}
    
    def calculate_batch_video_similarity(self, 
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
                    source_path, target_path, max_frames, verbose=verbose
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
    
    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load image from file path"""
        return Image.open(image_path).convert('RGB')
    
    def _extract_video_frames(self, 
                             video_path: str, 
                             max_frames: Optional[int], 
                             frame_step: int,
                             verbose: bool) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate actual step size if max_frames is specified
        if max_frames and total_frames > max_frames:
            actual_step = max(1, total_frames // max_frames)
        else:
            actual_step = frame_step
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % actual_step == 0:
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
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.numpy().tolist()
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
        
        print(f"✅ CLIP API results saved to: {output_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded CLIP model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "available_models": self.AVAILABLE_MODELS,
            "batch_size": self.batch_size
        }


def main():
    """Main function for testing CLIP API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP Video API")
    parser.add_argument("--task", type=str, choices=["video_similarity", "text_video", "extract_features"], 
                       default="video_similarity", help="Task to perform")
    parser.add_argument("--source", type=str, help="Source video file")
    parser.add_argument("--target", type=str, help="Target video file")
    parser.add_argument("--text", type=str, help="Text query for text-video similarity")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computing device")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")
    parser.add_argument("--output", type=str, default="clip_api_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create API instance
    clip_api = CLIPVideoAPI(
        model_name=args.model,
        device=args.device
    )
    
    # Print model info
    print(f"\n📋 Model Info:")
    model_info = clip_api.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Perform requested task
    if args.task == "video_similarity":
        if not args.source or not args.target:
            print("❌ Error: --source and --target are required for video similarity")
            return
        
        results = clip_api.calculate_video_similarity(
            args.source, 
            args.target, 
            max_frames=args.max_frames,
            verbose=True
        )
        
        clip_api.save_results({"video_similarity": results}, args.output)
        
    elif args.task == "text_video":
        if not args.source or not args.text:
            print("❌ Error: --source and --text are required for text-video similarity")
            return
        
        results = clip_api.calculate_text_video_similarity(
            args.source,
            args.text,
            max_frames=args.max_frames,
            verbose=True
        )
        
        clip_api.save_results({"text_video_similarity": results}, args.output)
        
    elif args.task == "extract_features":
        if not args.source:
            print("❌ Error: --source is required for feature extraction")
            return
        
        results = clip_api.extract_video_features(
            args.source,
            max_frames=args.max_frames,
            verbose=True
        )
        
        # Don't save features directly (too large), just metadata
        results_for_save = {k: v for k, v in results.items() if k != "features"}
        clip_api.save_results({"feature_extraction": results_for_save}, args.output)
    
    print(f"\n📊 Final Results saved to: {args.output}")


if __name__ == "__main__":
    main()