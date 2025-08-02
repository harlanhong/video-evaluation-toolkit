#!/usr/bin/env python3
"""
CLIP API Usage Example
Demonstrates comprehensive usage of the unified CLIP API for video evaluation

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_api import CLIPVideoAPI


def example_basic_video_similarity():
    """Example 1: Basic video-to-video similarity"""
    print("🎬 Example 1: Video-to-Video Similarity")
    print("=" * 50)
    
    # Example video paths (replace with your actual video paths)
    video1 = "path/to/video1.mp4"
    video2 = "path/to/video2.mp4"
    
    try:
        # Initialize CLIP API
        clip_api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")
        
        # Calculate similarity
        print(f"\n📊 Calculating similarity between videos...")
        results = clip_api.calculate_video_similarity(
            source_path=video1,
            target_path=video2,
            max_frames=50,  # Limit frames for faster processing
            verbose=True
        )
        
        if results['error'] is None:
            print(f"\n✅ Results:")
            print(f"   Overall Similarity: {results['clip_similarity']:.4f}")
            print(f"   Standard Deviation: {results['clip_similarity_std']:.4f}")
            print(f"   Range: [{results['clip_similarity_min']:.4f}, {results['clip_similarity_max']:.4f}]")
            print(f"   Median: {results['clip_similarity_median']:.4f}")
            print(f"   Frame Count: {results['frame_count']}")
            print(f"   Model: {results['model_name']}")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


def example_text_video_similarity():
    """Example 2: Text-to-video similarity"""
    print("\n\n🎬 Example 2: Text-to-Video Similarity")
    print("=" * 50)
    
    video_path = "path/to/video.mp4"
    text_queries = [
        "a person walking on the street",
        "someone dancing energetically", 
        "a beautiful outdoor landscape",
        "people having a conversation",
        "a car driving on the road"
    ]
    
    try:
        # Initialize CLIP API
        clip_api = CLIPVideoAPI(model_name="ViT-L/14", device="cuda")
        
        print(f"\n📊 Calculating text-video similarities...")
        print(f"   Video: {os.path.basename(video_path)}")
        print(f"   Queries: {len(text_queries)} texts")
        
        results = clip_api.calculate_text_video_similarity(
            video_path=video_path,
            text_queries=text_queries,
            max_frames=30,
            aggregation="mean",
            verbose=True
        )
        
        if results['error'] is None:
            print(f"\n✅ Text-Video Similarity Results:")
            for i, (query, similarity) in enumerate(zip(text_queries, results['similarities'])):
                print(f"   {i+1}. {similarity:.4f} - '{query}'")
            
            # Find best matching text
            best_idx = results['similarities'].index(max(results['similarities']))
            print(f"\n🏆 Best Match: '{text_queries[best_idx]}' ({results['similarities'][best_idx]:.4f})")
            
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


def example_video_feature_extraction():
    """Example 3: Video feature extraction"""
    print("\n\n🎬 Example 3: Video Feature Extraction")
    print("=" * 50)
    
    video_path = "path/to/video.mp4"
    
    try:
        # Initialize CLIP API  
        clip_api = CLIPVideoAPI(model_name="ViT-B/16", device="cuda")
        
        print(f"\n📊 Extracting CLIP features from video...")
        results = clip_api.extract_video_features(
            video_path=video_path,
            max_frames=20,
            verbose=True
        )
        
        if results['error'] is None:
            print(f"\n✅ Feature Extraction Results:")
            print(f"   Feature Dimension: {results['feature_dim']}")
            print(f"   Frame Count: {results['frame_count']}")
            print(f"   Model: {results['model_name']}")
            print(f"   Features Shape: {results['features'].shape}")
            
            # Analyze feature statistics
            import numpy as np
            features = results['features']
            print(f"\n📈 Feature Statistics:")
            print(f"   Mean: {np.mean(features):.4f}")
            print(f"   Std: {np.std(features):.4f}")
            print(f"   Min: {np.min(features):.4f}")
            print(f"   Max: {np.max(features):.4f}")
            
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


def example_batch_processing():
    """Example 4: Batch video similarity processing"""
    print("\n\n🎬 Example 4: Batch Video Processing")
    print("=" * 50)
    
    # Example video pairs (replace with your actual video paths)
    video_pairs = [
        ("path/to/video1.mp4", "path/to/reference1.mp4"),
        ("path/to/video2.mp4", "path/to/reference2.mp4"),
        ("path/to/video3.mp4", "path/to/reference3.mp4"),
    ]
    
    try:
        # Initialize CLIP API
        clip_api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")
        
        print(f"\n📊 Processing {len(video_pairs)} video pairs...")
        results = clip_api.calculate_batch_video_similarity(
            video_pairs=video_pairs,
            max_frames=25,
            verbose=True
        )
        
        print(f"\n✅ Batch Processing Results:")
        for pair_id, result in results.items():
            if result['error'] is None:
                similarity = result['clip_similarity']
                print(f"   {pair_id}: {similarity:.4f}")
            else:
                print(f"   {pair_id}: Failed - {result['error']}")
        
        # Calculate average similarity
        successful_similarities = [
            r['clip_similarity'] for r in results.values() 
            if r['error'] is None
        ]
        
        if successful_similarities:
            import numpy as np
            avg_similarity = np.mean(successful_similarities)
            print(f"\n📊 Average Similarity: {avg_similarity:.4f}")
            print(f"   Successful: {len(successful_similarities)}/{len(video_pairs)}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")


def example_model_comparison():
    """Example 5: Comparing different CLIP models"""
    print("\n\n🎬 Example 5: CLIP Model Comparison")
    print("=" * 50)
    
    video1 = "path/to/video1.mp4"
    video2 = "path/to/video2.mp4"
    
    models_to_test = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    
    print(f"\n📊 Comparing CLIP models on video similarity...")
    print(f"   Videos: {os.path.basename(video1)} vs {os.path.basename(video2)}")
    
    results_by_model = {}
    
    for model_name in models_to_test:
        try:
            print(f"\n🔄 Testing model: {model_name}")
            
            # Initialize CLIP API with specific model
            clip_api = CLIPVideoAPI(model_name=model_name, device="cuda")
            
            # Calculate similarity
            result = clip_api.calculate_video_similarity(
                source_path=video1,
                target_path=video2,
                max_frames=15,  # Keep it fast for comparison
                verbose=False
            )
            
            if result['error'] is None:
                results_by_model[model_name] = result['clip_similarity']
                print(f"   ✅ Similarity: {result['clip_similarity']:.4f}")
            else:
                print(f"   ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Summary comparison
    if results_by_model:
        print(f"\n📊 Model Comparison Summary:")
        sorted_models = sorted(results_by_model.items(), key=lambda x: x[1], reverse=True)
        for i, (model, similarity) in enumerate(sorted_models, 1):
            print(f"   {i}. {model}: {similarity:.4f}")


def example_save_and_load_results():
    """Example 6: Saving and loading results"""
    print("\n\n🎬 Example 6: Saving and Loading Results")
    print("=" * 50)
    
    video1 = "path/to/video1.mp4"
    video2 = "path/to/video2.mp4"
    
    try:
        # Initialize CLIP API
        clip_api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")
        
        # Calculate similarity
        print(f"\n📊 Calculating and saving results...")
        similarity_results = clip_api.calculate_video_similarity(
            source_path=video1,
            target_path=video2,
            max_frames=20,
            verbose=False
        )
        
        # Calculate text-video similarity
        text_results = clip_api.calculate_text_video_similarity(
            video_path=video1,
            text_queries=["a person walking", "outdoor scene"],
            max_frames=10,
            verbose=False
        )
        
        # Combine results
        combined_results = {
            "video_similarity": similarity_results,
            "text_video_similarity": text_results,
            "model_info": clip_api.get_model_info()
        }
        
        # Save results
        output_file = "clip_api_example_results.json"
        clip_api.save_results(combined_results, output_file)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Load and verify results
        import json
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        print(f"✅ Results loaded successfully:")
        print(f"   Video similarity: {loaded_results['video_similarity']['clip_similarity']:.4f}")
        print(f"   Text similarities: {loaded_results['text_video_similarity']['similarities']}")
        print(f"   Model used: {loaded_results['model_info']['model_name']}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")


def main():
    """Main function to run all examples"""
    
    print("🚀 CLIP API Comprehensive Examples")
    print("=" * 60)
    print("This script demonstrates the full capabilities of the unified CLIP API:")
    print("• Video-to-Video Similarity")
    print("• Text-to-Video Similarity") 
    print("• Feature Extraction")
    print("• Batch Processing")
    print("• Model Comparison")
    print("• Result Management")
    print()
    
    # Note about file paths
    print("⚠️  Note: Update the video file paths in this script before running")
    print("   Replace 'path/to/...' with actual video file paths")
    print()
    
    # Run examples
    example_basic_video_similarity()
    example_text_video_similarity()
    example_video_feature_extraction()
    example_batch_processing()
    example_model_comparison()
    example_save_and_load_results()
    
    print("\n\n✅ All examples completed!")
    print("\nNext steps:")
    print("• Update video file paths with your actual data")
    print("• Install CLIP: pip install git+https://github.com/openai/CLIP.git")
    print("• Try different CLIP models for your specific use case")
    print("• Integrate CLIP API into your video evaluation pipeline")
    print("\n🔧 For direct CLI usage: python clip_api.py --help")


if __name__ == "__main__":
    main()