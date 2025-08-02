#!/usr/bin/env python3
"""
Video Evaluation Toolkit - One-Click Setup Script
Automated installation script for complete environment setup

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This script provides automated setup for the video evaluation toolkit including:
- Environment creation and dependency installation
- Official GIM integration
- Model and checkpoint downloads
- VBench integration
- Installation verification

Usage:
    python setup.py [--mode MODE] [--skip-models] [--gpu] [--force]
"""

import os
import sys
import subprocess
import argparse
import json
import urllib.request
import shutil
import tempfile
from pathlib import Path
import platform
import importlib.util


class Colors:
    """Terminal colors for better output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class VideoEvaluationSetup:
    """Main setup class for video evaluation toolkit"""
    
    def __init__(self, args):
        self.args = args
        self.setup_dir = Path.cwd()
        self.models_dir = self.setup_dir / "models"
        self.cache_dir = self.setup_dir / "cache"
        
        # Environment and installation tracking
        self.conda_available = False
        self.venv_created = False
        self.dependencies_installed = False
        self.gim_installed = False
        self.models_downloaded = False
        
        # Model URLs and information
        self.models_info = {
            "syncnet": {
                "url": "https://github.com/joonson/syncnet_python/releases/download/v0.0.1/syncnet_v2.model",
                "filename": "syncnet_v2.model",
                "description": "SyncNet model for lip-sync evaluation",
                "size": "~180MB"
            },
            "s3fd": {
                "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                "filename": "s3fd.pth", 
                "description": "S3FD face detection model",
                "size": "~180MB"
            }
        }
    
    def print_header(self):
        """Print setup header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}=" * 70)
        print("🎬 VIDEO EVALUATION TOOLKIT - ONE-CLICK SETUP")
        print("=" * 70 + Colors.END)
        print(f"{Colors.GREEN}Advanced video quality assessment and synchronization evaluation{Colors.END}")
        print(f"{Colors.YELLOW}Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>{Colors.END}")
        print(f"\n📍 Setup Directory: {self.setup_dir}")
        print(f"🖥️  Platform: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print()
    
    def run_command(self, cmd, cwd=None, capture_output=False, check=True):
        """Run shell command with error handling"""
        if isinstance(cmd, list):
            cmd_str = ' '.join(cmd)
        else:
            cmd_str = cmd
            
        print(f"{Colors.BLUE}🔄 Running: {cmd_str}{Colors.END}")
        
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                                      capture_output=True, text=True, check=check)
                return result.stdout.strip() if result.returncode == 0 else None
            else:
                result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, check=check)
                return result.returncode == 0
        except subprocess.CalledProcessError as e:
            if check:
                print(f"{Colors.RED}❌ Command failed: {cmd_str}{Colors.END}")
                if capture_output and hasattr(e, 'stderr') and e.stderr:
                    print(f"{Colors.RED}   Error: {e.stderr.strip()}{Colors.END}")
                return False
            return None
        except Exception as e:
            print(f"{Colors.RED}❌ Exception running command: {e}{Colors.END}")
            return False
    
    def check_system_requirements(self):
        """Check system requirements"""
        print(f"{Colors.BOLD}📋 STEP 1: Checking System Requirements{Colors.END}")
        print("-" * 50)
        
        requirements_met = True
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            print(f"{Colors.GREEN}✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} (OK){Colors.END}")
        else:
            print(f"{Colors.RED}❌ Python {python_version.major}.{python_version.minor} (Requires ≥3.8){Colors.END}")
            requirements_met = False
        
        # Check Git
        if self.run_command("git --version", capture_output=True):
            git_version = self.run_command("git --version", capture_output=True)
            print(f"{Colors.GREEN}✅ Git available: {git_version}{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Git not found (Required for GIM installation){Colors.END}")
            requirements_met = False
        
        # Check Conda
        if self.run_command("conda --version", capture_output=True):
            conda_version = self.run_command("conda --version", capture_output=True)
            print(f"{Colors.GREEN}✅ Conda available: {conda_version}{Colors.END}")
            self.conda_available = True
        else:
            print(f"{Colors.YELLOW}⚠️ Conda not found (Will use pip){Colors.END}")
        
        # Check CUDA availability for GPU support
        if self.args.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    print(f"{Colors.GREEN}✅ CUDA available: {gpu_count} GPU(s) - {gpu_name}{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}⚠️ CUDA not available (Will use CPU){Colors.END}")
            except ImportError:
                print(f"{Colors.YELLOW}⚠️ PyTorch not installed yet (Will check after installation){Colors.END}")
        
        # Disk space check
        disk_space = shutil.disk_usage(self.setup_dir)
        free_gb = disk_space.free / (1024**3)
        if free_gb >= 5.0:  # Require at least 5GB
            print(f"{Colors.GREEN}✅ Disk space: {free_gb:.1f} GB available{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Insufficient disk space: {free_gb:.1f} GB (Requires ≥5GB){Colors.END}")
            requirements_met = False
        
        print()
        return requirements_met
    
    def setup_environment(self):
        """Setup Python environment"""
        print(f"{Colors.BOLD}🏗️  STEP 2: Setting Up Environment{Colors.END}")
        print("-" * 50)
        
        if self.args.mode == "conda" and self.conda_available:
            return self._setup_conda_environment()
        elif self.args.mode == "venv":
            return self._setup_venv_environment()
        elif self.args.mode == "pip":
            return self._setup_pip_only()
        else:
            # Auto-detect best method
            if self.conda_available:
                return self._setup_conda_environment()
            else:
                return self._setup_venv_environment()
    
    def _setup_conda_environment(self):
        """Setup conda environment"""
        env_name = "video-evaluation"
        env_file = self.setup_dir / "configs" / "environment.yaml"
        
        print(f"🐍 Setting up conda environment: {env_name}")
        
        # Check if environment already exists
        if self.run_command(f"conda env list | grep {env_name}", capture_output=True):
            if self.args.force:
                print(f"🗑️ Removing existing environment: {env_name}")
                self.run_command(f"conda env remove -n {env_name} -y")
            else:
                print(f"{Colors.YELLOW}⚠️ Environment {env_name} already exists. Use --force to recreate.{Colors.END}")
                return True
        
        # Create environment from YAML
        if env_file.exists():
            success = self.run_command(f"conda env create -f {env_file}")
        else:
            # Fallback: create basic environment
            success = self.run_command(f"conda create -n {env_name} python=3.9 -y")
        
        if success:
            print(f"{Colors.GREEN}✅ Conda environment created successfully{Colors.END}")
            print(f"{Colors.YELLOW}💡 Activate with: conda activate {env_name}{Colors.END}")
            self.venv_created = True
            return True
        else:
            print(f"{Colors.RED}❌ Failed to create conda environment{Colors.END}")
            return False
    
    def _setup_venv_environment(self):
        """Setup virtual environment"""
        venv_path = self.setup_dir / "venv"
        
        print(f"🐍 Setting up virtual environment: {venv_path}")
        
        if venv_path.exists():
            if self.args.force:
                print(f"🗑️ Removing existing virtual environment")
                shutil.rmtree(venv_path)
            else:
                print(f"{Colors.YELLOW}⚠️ Virtual environment already exists. Use --force to recreate.{Colors.END}")
                return True
        
        # Create virtual environment
        success = self.run_command(f"{sys.executable} -m venv {venv_path}")
        
        if success:
            print(f"{Colors.GREEN}✅ Virtual environment created successfully{Colors.END}")
            
            # Activate script path
            if platform.system() == "Windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                print(f"{Colors.YELLOW}💡 Activate with: {activate_script}{Colors.END}")
            else:
                activate_script = venv_path / "bin" / "activate"
                print(f"{Colors.YELLOW}💡 Activate with: source {activate_script}{Colors.END}")
            
            self.venv_created = True
            return True
        else:
            print(f"{Colors.RED}❌ Failed to create virtual environment{Colors.END}")
            return False
    
    def _setup_pip_only(self):
        """Setup using pip only (no virtual environment)"""
        print(f"🐍 Using current Python environment with pip")
        print(f"{Colors.YELLOW}⚠️ Installing directly to system Python (not recommended for production){Colors.END}")
        self.venv_created = True  # Skip venv activation checks
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print(f"{Colors.BOLD}📦 STEP 3: Installing Dependencies{Colors.END}")
        print("-" * 50)
        
        requirements_file = self.setup_dir / "configs" / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"{Colors.RED}❌ Requirements file not found: {requirements_file}{Colors.END}")
            return False
        
        print(f"📋 Installing from: {requirements_file}")
        
        # Install base requirements
        success = self.run_command(f"{sys.executable} -m pip install -r {requirements_file}")
        
        if not success:
            print(f"{Colors.RED}❌ Failed to install base requirements{Colors.END}")
            return False
        
        # Install GPU-specific packages if requested
        if self.args.gpu:
            print(f"🎮 Installing GPU-specific packages...")
            gpu_success = self.run_command(
                f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
            if gpu_success:
                print(f"{Colors.GREEN}✅ GPU packages installed{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ GPU package installation failed, continuing with CPU version{Colors.END}")
        
        # Install high priority packages for enhanced functionality
        high_priority_packages = [
            "mediapipe>=0.10.0",  # Enhanced face detection
            "ultralytics>=8.0.0",  # YOLOv8 face detection
            "numba>=0.56.0",  # Performance acceleration
        ]
        
        print(f"🎯 Installing high priority packages for enhanced functionality...")
        for package in high_priority_packages:
            print(f"   Installing {package}...")
            success = self.run_command(f"{sys.executable} -m pip install {package}", check=False)
            if success:
                print(f"   ✅ {package.split('>=')[0]} installed successfully")
            else:
                print(f"   ⚠️ {package.split('>=')[0]} installation failed (optional)")
        
        # Install additional useful packages
        extra_packages = [
            "jupyter",  # For notebooks
            "matplotlib",  # For visualization
            "seaborn",  # For better plots
            "tqdm",  # Progress bars
        ]
        
        print(f"🔧 Installing additional useful packages...")
        for package in extra_packages:
            self.run_command(f"{sys.executable} -m pip install {package}", check=False)
        
        print(f"{Colors.GREEN}✅ Dependencies installed successfully{Colors.END}")
        self.dependencies_installed = True
        return True
    
    def install_gim(self):
        """Install official GIM implementation (High Priority)"""
        print(f"{Colors.BOLD}🔍 STEP 4: Installing Official GIM (High Priority){Colors.END}")
        print("-" * 50)
        
        print(f"🎯 GIM is a high-priority component for state-of-the-art image matching")
        print(f"   Installing official GIM implementation from ICLR 2024...")
        
        gim_installer = self.setup_dir / "utils" / "install_gim.py"
        
        if gim_installer.exists():
            print(f"🚀 Using automated GIM installer...")
            force_flag = "--force" if self.args.force else ""
            success = self.run_command(f"{sys.executable} {gim_installer} {force_flag}")
        else:
            print(f"📥 Manual GIM installation...")
            success = self._manual_gim_install()
        
        if success:
            print(f"{Colors.GREEN}✅ GIM installed successfully - Enhanced image matching available!{Colors.END}")
            self.gim_installed = True
        else:
            print(f"{Colors.YELLOW}⚠️ GIM installation failed, will use fallback implementation{Colors.END}")
            print(f"   You can install GIM later with: python utils/install_gim.py")
            self.gim_installed = False
        
        return True  # Don't fail setup if GIM fails
    
    def _manual_gim_install(self):
        """Manual GIM installation with enhanced error handling"""
        gim_path = self.setup_dir / "gim"
        
        try:
            if gim_path.exists() and self.args.force:
                print(f"🗑️ Removing existing GIM installation...")
                shutil.rmtree(gim_path)
            
            if not gim_path.exists():
                print(f"📥 Cloning GIM repository from GitHub...")
                # Clone GIM repository with progress
                success = self.run_command(f"git clone --progress https://github.com/xuelunshen/gim.git {gim_path}")
                if not success:
                    print(f"❌ Failed to clone GIM repository")
                    return False
                print(f"✅ GIM repository cloned successfully")
            else:
                print(f"📁 GIM repository already exists, updating...")
                # Update existing repository
                self.run_command("git pull origin main", cwd=gim_path, check=False)
            
            # Install GIM with verbose output
            print(f"🔧 Installing GIM in development mode...")
            success = self.run_command(f"{sys.executable} -m pip install -e . --verbose", cwd=gim_path)
            
            if success:
                print(f"✅ GIM installed successfully in development mode")
                return True
            else:
                print(f"❌ GIM pip installation failed")
                return False
            
        except Exception as e:
            print(f"{Colors.RED}❌ Manual GIM installation failed: {e}{Colors.END}")
            return False
    
    def download_models(self):
        """Download required models and checkpoints"""
        if self.args.skip_models:
            print(f"{Colors.YELLOW}⏭️ Skipping model downloads (--skip-models){Colors.END}")
            return True
        
        print(f"{Colors.BOLD}🎭 STEP 5: Downloading Models and Checkpoints{Colors.END}")
        print("-" * 50)
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_models = len(self.models_info)
        
        for model_name, model_info in self.models_info.items():
            print(f"\n📥 Downloading {model_name}: {model_info['description']}")
            print(f"   Size: {model_info['size']}")
            print(f"   URL: {model_info['url']}")
            
            model_path = self.models_dir / model_info['filename']
            
            # Skip if file already exists and not forcing
            if model_path.exists() and not self.args.force:
                print(f"{Colors.GREEN}✅ {model_name} already exists{Colors.END}")
                success_count += 1
                continue
            
            try:
                # Download with progress
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
                    bar_length = 30
                    filled_length = int(bar_length * percent // 100)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f'\r   [{bar}] {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)', end='', flush=True)
                
                urllib.request.urlretrieve(model_info['url'], model_path, show_progress)
                print(f"\n{Colors.GREEN}✅ {model_name} downloaded successfully{Colors.END}")
                success_count += 1
                
            except Exception as e:
                print(f"\n{Colors.RED}❌ Failed to download {model_name}: {e}{Colors.END}")
                continue
        
        print(f"\n📊 Model Download Summary: {success_count}/{total_models} models downloaded")
        
        if success_count == total_models:
            print(f"{Colors.GREEN}✅ All models downloaded successfully{Colors.END}")
            self.models_downloaded = True
        elif success_count > 0:
            print(f"{Colors.YELLOW}⚠️ Some models downloaded, toolkit will work with reduced functionality{Colors.END}")
            self.models_downloaded = True
        else:
            print(f"{Colors.RED}❌ No models downloaded, some features may not work{Colors.END}")
            self.models_downloaded = False
        
        return True
    
    def setup_vbench_integration(self):
        """Setup VBench integration"""
        print(f"{Colors.BOLD}🎬 STEP 6: VBench Integration Setup{Colors.END}")
        print("-" * 50)
        
        print(f"🔧 VBench integration notes:")
        print(f"   • VBench requires separate environment (recommended)")
        print(f"   • Install VBench: conda create -n vbench python=3.9; conda activate vbench; pip install vbench")
        print(f"   • Our toolkit will detect VBench availability automatically")
        print(f"   • VBench can be enabled/disabled in the calculator")
        
        # Check if VBench is available in current environment
        try:
            spec = importlib.util.find_spec("vbench")
            if spec is not None:
                print(f"{Colors.GREEN}✅ VBench detected in current environment{Colors.END}")
            else:
                print(f"{Colors.YELLOW}💡 VBench not found in current environment (optional){Colors.END}")
        except ImportError:
            print(f"{Colors.YELLOW}💡 VBench not installed (optional){Colors.END}")
        
        return True
    
    def verify_installation(self):
        """Verify the installation"""
        print(f"{Colors.BOLD}🔍 STEP 7: Verifying Installation{Colors.END}")
        print("-" * 50)
        
        verification_tests = [
            ("Basic Import", "from evalutation.core.video_metrics_calculator import VideoMetricsCalculator"),
            ("CLIP API", "from evalutation.apis.clip_api import CLIPVideoAPI"),
            ("GIM Calculator", "from evalutation.calculators.gim_calculator import GIMMatchingCalculator"),
            ("LSE Calculator", "from evalutation.calculators.lse_calculator import LSECalculator"),
        ]
        
        passed_tests = 0
        
        for test_name, test_code in verification_tests:
            try:
                print(f"🧪 Testing {test_name}...")
                test_result = self.run_command(f"{sys.executable} -c \"{test_code}\"", capture_output=True)
                if test_result is not None:
                    print(f"{Colors.GREEN}   ✅ {test_name} - OK{Colors.END}")
                    passed_tests += 1
                else:
                    print(f"{Colors.RED}   ❌ {test_name} - Failed{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}   ❌ {test_name} - Exception: {e}{Colors.END}")
        
        # Additional verification tests
        print(f"\n🔧 Additional checks...")
        
        # Check GIM availability
        gim_check = self.run_command(
            f"{sys.executable} -c \"from evalutation.calculators.gim_calculator import GIMMatchingCalculator; calc = GIMMatchingCalculator(); info = calc.get_model_info(); print(f'GIM available: {{info[\\\"gim_available\\\"]}}')\"",
            capture_output=True
        )
        if gim_check:
            print(f"{Colors.GREEN}   ✅ GIM Status: {gim_check}{Colors.END}")
        
        # Check models
        models_found = sum(1 for model_info in self.models_info.values() 
                          if (self.models_dir / model_info['filename']).exists())
        print(f"   📁 Models: {models_found}/{len(self.models_info)} found")
        
        print(f"\n📊 Verification Summary: {passed_tests}/{len(verification_tests)} tests passed")
        
        if passed_tests == len(verification_tests):
            print(f"{Colors.GREEN}🎉 Installation verification successful!{Colors.END}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️ Some verification tests failed, but basic functionality should work{Colors.END}")
            return True
    
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        print(f"{Colors.BOLD}📚 STEP 8: Creating Quick Start Guide{Colors.END}")
        print("-" * 50)
        
        guide_content = f"""# Video Evaluation Toolkit - Quick Start Guide

## 🎉 Installation Completed Successfully!

### Environment Information
- Setup Directory: {self.setup_dir}
- Models Directory: {self.models_dir}
- Environment: {'Conda' if self.conda_available else 'Virtual Environment'}
- GIM Integration: {'✅ Available' if self.gim_installed else '⚠️ Using Fallback'}
- Models Downloaded: {'✅ Complete' if self.models_downloaded else '⚠️ Partial'}

### Quick Usage Examples

#### 1. Basic Video Metrics
```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="your_video.mp4",
    gt_path="reference_video.mp4"  # Optional
)
print(f"LSE Score: {{metrics['lse_score']}}")
```

#### 2. Advanced Metrics with GIM
```python
calculator = VideoMetricsCalculator(
    enable_clip_similarity=True,
    enable_gim_matching=True
)
metrics = calculator.calculate_video_metrics(
    pred_path="generated_video.mp4",
    gt_path="reference_video.mp4"
)
print(f"CLIP Similarity: {{metrics['clip_similarity']:.4f}}")
print(f"GIM Matching: {{metrics['gim_matching_pixels']}}")
```

#### 3. Command Line Usage
```bash
python -m core.video_metrics_calculator \\
    --pred generated_video.mp4 \\
    --gt reference_video.mp4 \\
    --clip --gim
```

### Available Examples
- `examples/basic_usage.py` - Basic usage examples
- `examples/advanced_metrics.py` - Advanced metrics demonstration
- `examples/clip_api_demo.py` - CLIP API examples
- `examples/gim_demo.py` - GIM integration examples

### Documentation
- `docs/README.md` - Main documentation
- `docs/GIM_INTEGRATION.md` - GIM integration guide
- `docs/MODELS_DOWNLOAD.md` - Model download instructions

### Next Steps
1. Try the examples: `python examples/basic_usage.py`
2. Read the documentation in `docs/`
3. Test with your own videos
4. Explore advanced features

### Troubleshooting
- Check installation: `python -c "from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working!')"`
- Update dependencies: `pip install -r configs/requirements.txt --upgrade`
- Reinstall GIM: `python utils/install_gim.py --force`

Happy evaluating! 🎬
"""
        
        guide_path = self.setup_dir / "QUICK_START.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📝 Quick start guide created: {guide_path}")
        return True
    
    def print_summary(self):
        """Print installation summary"""
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 SETUP COMPLETED!{Colors.END}")
        print("=" * 70)
        
        print(f"📋 Installation Summary:")
        print(f"   • Environment: {'✅' if self.venv_created else '❌'} Created")
        print(f"   • Dependencies: {'✅' if self.dependencies_installed else '❌'} Installed")
        print(f"   • GIM Integration: {'✅' if self.gim_installed else '⚠️'} {'Available' if self.gim_installed else 'Fallback'}")
        print(f"   • Models: {'✅' if self.models_downloaded else '⚠️'} {'Downloaded' if self.models_downloaded else 'Partial'}")
        
        print(f"\n🚀 Next Steps:")
        if not self.args.mode == "pip":
            env_name = "video-evaluation" if self.conda_available else "venv"
            activate_cmd = f"conda activate {env_name}" if self.conda_available else f"source venv/bin/activate"
            print(f"   1. Activate environment: {activate_cmd}")
        
        print(f"   2. Read quick start: cat QUICK_START.md")
        print(f"   3. Try examples: python examples/basic_usage.py")
        print(f"   4. Read documentation: docs/README.md")
        
        print(f"\n💡 Useful Commands:")
        print(f"   • Test installation: python -c \"from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working!')\"")
        print(f"   • Check GIM status: python -c \"from evalutation.calculators.gim_calculator import GIMMatchingCalculator; print(GIMMatchingCalculator().get_model_info())\"")
        print(f"   • Update toolkit: git pull origin main")
        
        print(f"\n{Colors.BLUE}📧 Support: fatinghong@gmail.com")
        print(f"🌐 Repository: https://github.com/harlanhong/video-evaluation-toolkit{Colors.END}")
    
    def run_setup(self):
        """Run the complete setup process"""
        try:
            self.print_header()
            
            # Step 1: Check system requirements
            if not self.check_system_requirements():
                print(f"{Colors.RED}❌ System requirements not met. Please fix the issues above.{Colors.END}")
                return False
            
            # Step 2: Setup environment
            if not self.setup_environment():
                print(f"{Colors.RED}❌ Environment setup failed.{Colors.END}")
                return False
            
            # Step 3: Install dependencies
            if not self.install_dependencies():
                print(f"{Colors.RED}❌ Dependency installation failed.{Colors.END}")
                return False
            
            # Step 4: Install GIM
            self.install_gim()
            
            # Step 5: Download models
            self.download_models()
            
            # Step 6: Setup VBench integration
            self.setup_vbench_integration()
            
            # Step 7: Verify installation
            if not self.verify_installation():
                print(f"{Colors.YELLOW}⚠️ Installation verification had issues, but setup completed.{Colors.END}")
            
            # Step 8: Create quick start guide
            self.create_quick_start_guide()
            
            # Final summary
            self.print_summary()
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️ Setup interrupted by user.{Colors.END}")
            return False
        except Exception as e:
            print(f"\n{Colors.RED}❌ Setup failed with error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Video Evaluation Toolkit - One-Click Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup.py                          # Auto-detect best setup method
    python setup.py --mode conda             # Use conda environment
    python setup.py --mode venv              # Use virtual environment
    python setup.py --mode pip               # Use pip only (no env)
    python setup.py --gpu                    # Install GPU support + high priority packages
    python setup.py --skip-models            # Skip model downloads
    python setup.py --force                  # Force reinstall everything

High Priority Packages (Auto-installed):
    • MediaPipe: Enhanced face detection and tracking
    • Ultralytics: YOLOv8-based face detection
    • NumBA: Performance acceleration
    • Official GIM: State-of-the-art image matching (ICLR 2024)
        """
    )
    
    parser.add_argument("--mode", choices=["auto", "conda", "venv", "pip"], 
                       default="auto", help="Installation mode")
    parser.add_argument("--gpu", action="store_true", 
                       help="Install GPU support (CUDA)")
    parser.add_argument("--skip-models", action="store_true", 
                       help="Skip model downloads")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstall (remove existing)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create and run setup
    setup = VideoEvaluationSetup(args)
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()