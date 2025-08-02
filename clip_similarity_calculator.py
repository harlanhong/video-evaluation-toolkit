#!/usr/bin/env python3
"""
CLIP Similarity Calculator
Legacy wrapper for CLIP-V similarity calculation using the unified CLIP API

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides backward compatibility for CLIP-based video similarity calculation.
It now uses the unified CLIP API internally.

Usage:
    from evalutation.clip_similarity_calculator import CLIPSimilarityCalculator
    
    calculator = CLIPSimilarityCalculator()
    clip_similarity = calculator.calculate_video_similarity("source.mp4", "target.mp4")
"""

import os
import sys
from typing import Optional, Dict, Any, List, Tuple

# Import the unified CLIP API
try:
    # Use relative import when imported as package
    from .clip_api import CLIPVideoAPI
except ImportError:
    # Use absolute import when run directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from clip_api import CLIPVideoAPI


class CLIPSimilarityCalculator:
    """CLIP Similarity Calculator (Legacy wrapper for CLIPVideoAPI)"""
    
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
        print(f"🔄 Initializing CLIP Similarity Calculator (using unified CLIP API)")
        
        # Initialize the unified CLIP API
        self.clip_api = CLIPVideoAPI(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        
        # Store parameters for compatibility
        self.device = self.clip_api.device
        self.batch_size = batch_size
        self.model_name = model_name
        
        print(f"✅ CLIP Similarity Calculator initialized successfully")
    
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
        # Delegate to the unified CLIP API
        return self.clip_api.calculate_video_similarity(
            source_path=source_path,
            target_path=target_path,
            max_frames=max_frames,
            frame_step=1,
            verbose=verbose
        )
    
    def extract_video_features(self, 
                              video_path: str,
                              max_frames: Optional[int] = None,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Extract CLIP features from video (wrapper for CLIP API)
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing features and metadata
        """
        return self.clip_api.extract_video_features(
            video_path=video_path,
            max_frames=max_frames,
            frame_step=1,
            verbose=verbose
        )
    
    def calculate_text_video_similarity(self,
                                       video_path: str,
                                       text_queries: str,
                                       max_frames: Optional[int] = None,
                                       verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate similarity between video and text descriptions (wrapper for CLIP API)
        
        Args:
            video_path: Path to video file
            text_queries: Text query or list of text queries
            max_frames: Maximum frames to process
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing text-video similarity results
        """
        return self.clip_api.calculate_text_video_similarity(
            video_path=video_path,
            text_queries=text_queries,
            max_frames=max_frames,
            aggregation="mean",
            verbose=verbose
        )
    
    def calculate_batch_similarity(self, 
                                 video_pairs: List[Tuple[str, str]], 
                                 max_frames: Optional[int] = None,
                                 verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Calculate CLIP similarities for multiple video pairs (wrapper for CLIP API)
        
        Args:
            video_pairs: List of (source_path, target_path) tuples
            max_frames: Maximum frames per video
            verbose: Whether to print progress
            
        Returns:
            Dictionary with pair identifiers as keys and similarity results as values
        """
        return self.clip_api.calculate_batch_video_similarity(
            video_pairs=video_pairs,
            max_frames=max_frames,
            verbose=verbose
        )
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save similarity results to JSON file (wrapper for CLIP API)"""
        self.clip_api.save_results(results, output_path)


def main():
    """Main function for testing (legacy wrapper)"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP Similarity Calculator (Legacy Wrapper)")
    parser.add_argument("--source", type=str, required=True, help="Source video file")
    parser.add_argument("--target", type=str, required=True, help="Target video file")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computing device")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")
    parser.add_argument("--output", type=str, default="clip_similarity_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    print("🔄 CLIP Similarity Calculator (Legacy wrapper for unified CLIP API)")
    print("💡 Consider using clip_api.py directly for more features")
    
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
        print(f"   Model: {results.get('model_name', 'N/A')}")
    else:
        print(f"\n❌ CLIP similarity calculation failed: {results.get('error', 'Unknown error')}")
    
    print(f"\n💡 For more advanced CLIP features, use: python clip_api.py --help")


if __name__ == "__main__":
    main()