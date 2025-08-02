#!/usr/bin/env python3
"""
GIM (Graph Image Matching) Usage Example
Demonstrates how to use the official GIM integration for video matching

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalutation.calculators.gim_calculator import GIMMatchingCalculator


def example_basic_gim_matching():
    """Example 1: Basic GIM matching with different models"""
    print("🎬 Example 1: Basic GIM Matching")
    print("=" * 50)
    
    # Example video paths (replace with your actual video paths)
    source_video = "path/to/source_video.mp4"
    target_video = "path/to/target_video.mp4"
    
    # Test different GIM models
    models_to_test = [
        ("gim_roma", "Highest accuracy"),
        ("gim_lightglue", "Fast and accurate"),
        ("gim_dkm", "Dense matching"),
    ]
    
    for model_name, description in models_to_test:
        try:
            print(f"\n🔧 Testing {model_name} ({description})")
            
            # Initialize GIM calculator
            calculator = GIMMatchingCalculator(
                model_name=model_name,
                device="cuda",
                confidence_threshold=0.5
            )
            
            # Display model information
            model_info = calculator.get_model_info()
            print(f"   Model: {model_info['model_name']}")
            print(f"   GIM Available: {model_info['gim_available']}")
            print(f"   Device: {model_info['device']}")
            
            # Calculate matching (with placeholder videos)
            print(f"   📊 Calculating matching...")
            results = calculator.calculate_video_matching(
                source_path=source_video,
                target_path=target_video,
                max_frames=10,  # Small number for demo
                verbose=False
            )
            
            if results.get('total_matching_pixels') is not None:
                print(f"   ✅ Results:")
                print(f"      Total matches: {results['total_matching_pixels']}")
                print(f"      Average per frame: {results['avg_matching_pixels']:.2f}")
                print(f"      Standard deviation: {results['std_matching_pixels']:.2f}")
            else:
                print(f"   ❌ Matching failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")


def example_gim_model_comparison():
    """Example 2: Compare different GIM models on the same video pair"""
    print("\n\n🎬 Example 2: GIM Model Comparison")
    print("=" * 50)
    
    source_video = "path/to/source_video.mp4"
    target_video = "path/to/target_video.mp4"
    
    models = ["gim_roma", "gim_lightglue", "gim_dkm"]
    results_comparison = {}
    
    print(f"\n📊 Comparing GIM models on video pair...")
    print(f"   Source: {os.path.basename(source_video)}")
    print(f"   Target: {os.path.basename(target_video)}")
    
    for model_name in models:
        try:
            print(f"\n🔄 Testing {model_name}...")
            
            calculator = GIMMatchingCalculator(
                model_name=model_name,
                device="cuda",
                confidence_threshold=0.5
            )
            
            results = calculator.calculate_video_matching(
                source_video, target_video,
                max_frames=15,
                verbose=False
            )
            
            if results.get('total_matching_pixels') is not None:
                results_comparison[model_name] = {
                    'total_matches': results['total_matching_pixels'],
                    'avg_matches': results['avg_matching_pixels'],
                    'frame_count': results['frame_count']
                }
                print(f"   ✅ {model_name}: {results['total_matching_pixels']} total matches")
            else:
                print(f"   ❌ {model_name}: Failed - {results.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ {model_name}: Exception - {e}")
    
    # Summary comparison
    if results_comparison:
        print(f"\n📊 Model Comparison Summary:")
        sorted_models = sorted(results_comparison.items(), 
                             key=lambda x: x[1]['total_matches'], reverse=True)
        
        for i, (model, result) in enumerate(sorted_models, 1):
            print(f"   {i}. {model}:")
            print(f"      Total matches: {result['total_matches']}")
            print(f"      Avg per frame: {result['avg_matches']:.2f}")
            print(f"      Frame count: {result['frame_count']}")


def example_gim_configuration():
    """Example 3: GIM configuration options"""
    print("\n\n🎬 Example 3: GIM Configuration Options")
    print("=" * 50)
    
    source_video = "path/to/source_video.mp4"
    target_video = "path/to/target_video.mp4"
    
    # Test different configurations
    configurations = [
        {
            'name': 'High Accuracy',
            'model': 'gim_roma',
            'threshold': 0.2,
            'max_keypoints': 4096
        },
        {
            'name': 'Balanced',
            'model': 'gim_lightglue',
            'threshold': 0.5,
            'max_keypoints': 2048
        },
        {
            'name': 'Fast',
            'model': 'gim_lightglue',
            'threshold': 0.7,
            'max_keypoints': 1024
        }
    ]
    
    for config in configurations:
        try:
            print(f"\n🔧 Configuration: {config['name']}")
            
            calculator = GIMMatchingCalculator(
                model_name=config['model'],
                device="cuda",
                confidence_threshold=config['threshold'],
                max_keypoints=config['max_keypoints']
            )
            
            results = calculator.calculate_video_matching(
                source_video, target_video,
                max_frames=10,
                verbose=False
            )
            
            if results.get('total_matching_pixels') is not None:
                print(f"   Model: {config['model']}")
                print(f"   Threshold: {config['threshold']}")
                print(f"   Max keypoints: {config['max_keypoints']}")
                print(f"   ✅ Results: {results['total_matching_pixels']} matches")
                print(f"   Avg per frame: {results['avg_matching_pixels']:.2f}")
            else:
                print(f"   ❌ Failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def example_gim_fallback():
    """Example 4: GIM fallback mechanism"""
    print("\n\n🎬 Example 4: GIM Fallback Mechanism")
    print("=" * 50)
    
    try:
        # Initialize calculator (will use official GIM if available, fallback otherwise)
        calculator = GIMMatchingCalculator(
            model_name="gim_roma",
            device="cuda"
        )
        
        # Check what type of matcher is being used
        model_info = calculator.get_model_info()
        
        print(f"🔍 GIM Status:")
        print(f"   GIM Available: {model_info['gim_available']}")
        print(f"   Matcher Type: {model_info['matcher_type']}")
        print(f"   Model Name: {model_info['model_name']}")
        print(f"   Device: {model_info['device']}")
        
        if model_info['gim_available']:
            print(f"   ✅ Using official GIM implementation")
        else:
            print(f"   ⚠️ Using fallback matcher (simple ORB-based)")
            print(f"   💡 Install official GIM for better performance:")
            print(f"      python utils/install_gim.py")
        
        # Test functionality regardless of which matcher is used
        source_video = "path/to/source_video.mp4"
        target_video = "path/to/target_video.mp4"
        
        print(f"\n📊 Testing matcher functionality...")
        results = calculator.calculate_video_matching(
            source_video, target_video,
            max_frames=5,
            verbose=False
        )
        
        if results.get('total_matching_pixels') is not None:
            print(f"   ✅ Matcher working correctly")
            print(f"   Total matches: {results['total_matching_pixels']}")
        else:
            print(f"   ❌ Matcher test failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_save_and_load_results():
    """Example 5: Saving and loading GIM results"""
    print("\n\n🎬 Example 5: Saving and Loading Results")
    print("=" * 50)
    
    try:
        calculator = GIMMatchingCalculator(model_name="gim_lightglue")
        
        # Calculate matching
        source_video = "path/to/source_video.mp4"
        target_video = "path/to/target_video.mp4"
        
        print(f"📊 Calculating and saving results...")
        results = calculator.calculate_video_matching(
            source_video, target_video,
            max_frames=8,
            verbose=False
        )
        
        # Save results
        output_file = "gim_demo_results.json"
        calculator.save_results(results, output_file)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Load and verify results
        import json
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        print(f"✅ Results loaded successfully:")
        print(f"   Model: {loaded_results.get('model_name', 'Unknown')}")
        print(f"   Total matches: {loaded_results.get('total_matching_pixels', 'N/A')}")
        print(f"   Frame count: {loaded_results.get('frame_count', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run all examples"""
    
    print("🚀 GIM (Graph Image Matching) Examples")
    print("=" * 60)
    print("This script demonstrates the official GIM integration capabilities:")
    print("• Basic GIM matching with different models")
    print("• Model comparison and performance analysis")
    print("• Configuration options and optimization")
    print("• Fallback mechanism when GIM is not available")
    print("• Result management and persistence")
    print()
    
    # Note about file paths
    print("⚠️  Note: Update the video file paths in this script before running")
    print("   Replace 'path/to/...' with actual video file paths")
    print()
    
    # Check GIM availability
    try:
        calculator = GIMMatchingCalculator()
        model_info = calculator.get_model_info()
        
        if model_info['gim_available']:
            print("✅ Official GIM is available and ready to use")
        else:
            print("⚠️ Official GIM not found - will use fallback matcher")
            print("💡 Install GIM for best performance: python utils/install_gim.py")
        print()
    except Exception as e:
        print(f"❌ Error checking GIM availability: {e}")
        print()
    
    # Run examples
    example_basic_gim_matching()
    example_gim_model_comparison()
    example_gim_configuration()
    example_gim_fallback()
    example_save_and_load_results()
    
    print("\n\n✅ All GIM examples completed!")
    print("\nNext steps:")
    print("• Update video file paths with your actual data")
    print("• Install official GIM: python utils/install_gim.py")
    print("• Try different GIM models for your specific use case")
    print("• Read the detailed guide: docs/GIM_INTEGRATION.md")
    print("\n🔧 For direct CLI usage: python -m calculators.gim_calculator --help")


if __name__ == "__main__":
    main()