#!/usr/bin/env python3
"""
Installation Demo and Testing Script
Demonstrates various installation scenarios and validates setup

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This script demonstrates how to test and validate the one-click installation
system and provides examples of various installation scenarios.

Usage:
    python examples/installation_demo.py [--test-mode] [--verbose]
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, capture_output=True):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=capture_output, text=True, check=True)
        return result.stdout.strip() if capture_output else True
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"Command failed: {cmd}")
            print(f"Error: {e.stderr}")
        return False


def test_basic_functionality():
    """Test basic functionality after installation"""
    print("🧪 Testing Basic Functionality")
    print("-" * 40)
    
    tests = [
        ("Core Calculator Import", 
         "from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('OK')"),
        ("CLIP API Import", 
         "from evalutation.apis.clip_api import CLIPVideoAPI; print('OK')"),
        ("GIM Calculator Import", 
         "from evalutation.calculators.gim_calculator import GIMMatchingCalculator; print('OK')"),
        ("LSE Calculator Import", 
         "from evalutation.calculators.lse_calculator import LSECalculator; print('OK')"),
    ]
    
    passed = 0
    for test_name, test_code in tests:
        print(f"  • {test_name}...", end=" ")
        result = run_command(f"python -c \"{test_code}\"")
        if result and "OK" in str(result):
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
    
    print(f"\nBasic Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_advanced_functionality():
    """Test advanced functionality"""
    print("\n🔧 Testing Advanced Functionality")
    print("-" * 40)
    
    # Test GIM availability
    print("  • GIM Integration...", end=" ")
    gim_test = run_command("""
python -c "
from evalutation.calculators.gim_calculator import GIMMatchingCalculator
calc = GIMMatchingCalculator()
info = calc.get_model_info()
print(f'Available: {info[\"gim_available\"]}')
print(f'Type: {info[\"matcher_type\"]}')
"
""")
    
    if gim_test:
        print("✅ PASS")
        print(f"    {gim_test}")
    else:
        print("❌ FAIL")
    
    # Test CLIP API
    print("  • CLIP API...", end=" ")
    clip_test = run_command("""
python -c "
from evalutation.apis.clip_api import CLIPVideoAPI
try:
    api = CLIPVideoAPI()
    print('Available: True')
    print(f'Device: {api.device}')
except Exception as e:
    print(f'Error: {e}')
"
""")
    
    if clip_test and "Available: True" in str(clip_test):
        print("✅ PASS")
        print(f"    {clip_test}")
    else:
        print("❌ FAIL")
    
    # Test model files
    print("  • Model Files...", end=" ")
    models_dir = Path("models")
    expected_models = ["syncnet_v2.model", "s3fd.pth"]
    found_models = []
    
    for model in expected_models:
        model_path = models_dir / model
        if model_path.exists():
            found_models.append(model)
    
    print(f"✅ {len(found_models)}/{len(expected_models)} models found")
    for model in found_models:
        size = (models_dir / model).stat().st_size / (1024*1024)
        print(f"    {model}: {size:.1f}MB")


def demo_installation_scenarios():
    """Demonstrate different installation scenarios"""
    print("\n📋 Installation Scenarios Demo")
    print("-" * 40)
    
    scenarios = [
        {
            "name": "Quick Setup (Auto-detect)",
            "command": "python setup.py",
            "description": "Automatically detects the best installation method"
        },
        {
            "name": "Conda Environment", 
            "command": "python setup.py --mode conda --gpu",
            "description": "Creates conda environment with GPU support"
        },
        {
            "name": "Virtual Environment",
            "command": "python setup.py --mode venv",
            "description": "Creates Python virtual environment"
        },
        {
            "name": "Skip Models (Fast)",
            "command": "python setup.py --skip-models",
            "description": "Skip model downloads for faster setup"
        },
        {
            "name": "Force Reinstall",
            "command": "python setup.py --force --gpu",
            "description": "Force clean reinstallation with GPU support"
        },
        {
            "name": "Bash Installer",
            "command": "bash install.sh --gpu",
            "description": "Bash script installation with GPU support"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Command: {scenario['command']}")
        print(f"   Description: {scenario['description']}")
    
    print("\n💡 Choose the scenario that best fits your needs!")


def demo_verification_commands():
    """Demonstrate verification commands"""
    print("\n🔍 Verification Commands Demo")
    print("-" * 40)
    
    commands = [
        {
            "name": "Basic Installation Check",
            "command": "python -c \"from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working!')\"",
            "description": "Verify basic installation"
        },
        {
            "name": "GIM Status Check",
            "command": "python -c \"from evalutation.calculators.gim_calculator import GIMMatchingCalculator; print(GIMMatchingCalculator().get_model_info())\"",
            "description": "Check GIM integration status"
        },
        {
            "name": "CLIP API Test",
            "command": "python -c \"from evalutation.apis.clip_api import CLIPVideoAPI; api = CLIPVideoAPI(); print(f'CLIP ready on {api.device}')\"",
            "description": "Test CLIP API functionality"
        },
        {
            "name": "Model Files Check",
            "command": "ls -la models/",
            "description": "Check downloaded model files"
        },
        {
            "name": "Environment Info",
            "command": "python -c \"import sys, torch; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')\"",
            "description": "Display environment information"
        }
    ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['name']}")
        print(f"   Description: {cmd_info['description']}")
        print(f"   Command: {cmd_info['command']}")


def demo_usage_examples():
    """Demonstrate usage examples after installation"""
    print("\n🎬 Usage Examples After Installation")
    print("-" * 40)
    
    examples = [
        {
            "name": "Basic Video Metrics",
            "code": """
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="video.mp4",
    gt_path="reference.mp4"  # Optional
)
print(f"LSE Score: {metrics['lse_score']}")
""",
            "description": "Calculate basic video metrics"
        },
        {
            "name": "Advanced Metrics with GIM",
            "code": """
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator(
    enable_clip_similarity=True,
    enable_gim_matching=True
)
metrics = calculator.calculate_video_metrics(
    pred_path="generated.mp4",
    gt_path="reference.mp4"
)
print(f"CLIP Similarity: {metrics['clip_similarity']:.4f}")
print(f"GIM Matching: {metrics['gim_matching_pixels']}")
""",
            "description": "Use advanced synchronization metrics"
        },
        {
            "name": "Official GIM Matching",
            "code": """
from evalutation.calculators.gim_calculator import GIMMatchingCalculator

gim_calc = GIMMatchingCalculator(model_name="gim_roma")
results = gim_calc.calculate_video_matching(
    source_path="source.mp4",
    target_path="target.mp4"
)
print(f"Total matches: {results['total_matching_pixels']}")
""",
            "description": "Use official GIM implementation directly"
        },
        {
            "name": "CLIP API Usage",
            "code": """
from evalutation.apis.clip_api import CLIPVideoAPI

clip_api = CLIPVideoAPI()
similarity = clip_api.calculate_video_similarity(
    source_path="video1.mp4",
    target_path="video2.mp4"
)
print(f"CLIP Similarity: {similarity['clip_similarity']:.4f}")
""",
            "description": "Use unified CLIP API"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Code:")
        for line in example['code'].strip().split('\n'):
            print(f"     {line}")


def demo_troubleshooting():
    """Demonstrate troubleshooting steps"""
    print("\n🔧 Troubleshooting Guide")
    print("-" * 40)
    
    issues = [
        {
            "issue": "Import Error: Module not found",
            "solution": "pip install -r configs/requirements.txt --upgrade",
            "description": "Update dependencies to latest versions"
        },
        {
            "issue": "GIM not working",
            "solution": "python utils/install_gim.py --force",
            "description": "Reinstall official GIM implementation"
        },
        {
            "issue": "CUDA out of memory",
            "solution": "Use CPU mode or reduce batch sizes",
            "description": "calculator = VideoMetricsCalculator(device='cpu')"
        },
        {
            "issue": "Models not found",
            "solution": "python setup.py --force",
            "description": "Re-download model files"
        },
        {
            "issue": "Environment issues", 
            "solution": "Remove venv/gim folders and run setup.py --force",
            "description": "Clean reinstall everything"
        }
    ]
    
    for i, issue_info in enumerate(issues, 1):
        print(f"\n{i}. {issue_info['issue']}")
        print(f"   Solution: {issue_info['solution']}")
        print(f"   Description: {issue_info['description']}")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Installation Demo and Testing")
    parser.add_argument("--test-mode", action="store_true", help="Run actual tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🚀 Video Evaluation Toolkit - Installation Demo")
    print("=" * 60)
    print("This script demonstrates the one-click installation system")
    print("and provides examples of testing and verification procedures.")
    print()
    
    if args.test_mode:
        print("🧪 RUNNING ACTUAL TESTS")
        print("=" * 60)
        
        # Run actual functionality tests
        basic_ok = test_basic_functionality()
        test_advanced_functionality()
        
        if basic_ok:
            print("\n✅ Installation appears to be working correctly!")
        else:
            print("\n❌ Some issues detected. Check the output above.")
    else:
        print("📋 DEMONSTRATION MODE")
        print("=" * 60)
        print("Run with --test-mode to execute actual tests")
        print()
        
        # Show demonstrations
        demo_installation_scenarios()
        demo_verification_commands()
        demo_usage_examples()
        demo_troubleshooting()
    
    print("\n🎬 Next Steps:")
    print("1. Try the installation: python setup.py --gpu")
    print("2. Run tests: python examples/installation_demo.py --test-mode")
    print("3. Check examples: python examples/basic_usage.py")
    print("4. Read docs: docs/README.md")
    
    print("\n📧 Support: fatinghong@gmail.com")
    print("🌐 Repository: https://github.com/harlanhong/video-evaluation-toolkit")


if __name__ == "__main__":
    main()