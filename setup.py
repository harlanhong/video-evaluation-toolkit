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
        
        # Python executable (will be updated to use created environment)
        self.python_executable = sys.executable
        
        # Model URLs and information
        self.models_info = {
            "syncnet": {
                "url": None,  # Already available locally
                "filename": "syncnet_v2.model",
                "description": "SyncNet model for lip-sync evaluation",
                "size": "~52MB",
                "priority": "high",
                "local_only": True
            },
            "s3fd": {
                "url": None,  # Already available locally
                "filename": "sfd_face.pth", 
                "description": "S3FD face detection model",
                "size": "~86MB",
                "priority": "high",
                "local_only": True
            },
            "yolov8_face": {
                "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
                "filename": "yolov8n-face.pt",
                "description": "YOLOv8 face detection model",
                "size": "~6MB",
                "priority": "high",
                "local_only": False
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
            
            # Set the Python executable to use the conda environment
            conda_python = self._get_conda_python_path(env_name)
            if conda_python and conda_python.exists():
                self.python_executable = str(conda_python)
                print(f"{Colors.GREEN}✅ Using conda environment Python: {self.python_executable}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ Could not find conda environment Python, using current Python{Colors.END}")
                print(f"{Colors.YELLOW}💡 Activate manually: conda activate {env_name}{Colors.END}")
            
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
            
            # Set the Python executable to use the venv environment
            if platform.system() == "Windows":
                venv_python = venv_path / "Scripts" / "python.exe"
                activate_script = venv_path / "Scripts" / "activate.bat"
                activate_hint = f"{activate_script}"
            else:
                venv_python = venv_path / "bin" / "python"
                activate_script = venv_path / "bin" / "activate"
                activate_hint = f"source {activate_script}"
            
            if venv_python.exists():
                self.python_executable = str(venv_python)
                print(f"{Colors.GREEN}✅ Using virtual environment Python: {self.python_executable}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ Could not find virtual environment Python, using current Python{Colors.END}")
                print(f"{Colors.YELLOW}💡 Activate manually: {activate_hint}{Colors.END}")
            
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
    
    def _get_conda_python_path(self, env_name):
        """Get the Python executable path for a conda environment"""
        try:
            # Try to get conda info for the environment
            result = subprocess.run(
                ["conda", "info", "--envs"], 
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.split('\n'):
                if env_name in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = Path(parts[-1])  # Last part is the path
                        if platform.system() == "Windows":
                            python_path = env_path / "python.exe"
                        else:
                            python_path = env_path / "bin" / "python"
                        return python_path
                        
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: construct path based on common conda structure
            try:
                conda_base = Path(sys.executable).parent.parent  # From current python to conda base
                envs_dir = conda_base / "envs" / env_name
                if platform.system() == "Windows":
                    python_path = envs_dir / "python.exe"
                else:
                    python_path = envs_dir / "bin" / "python"
                return python_path if python_path.exists() else None
            except:
                pass
                
        return None
    
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
        success = self.run_command(f"{self.python_executable} -m pip install -r {requirements_file}")
        
        if not success:
            print(f"{Colors.RED}❌ Failed to install base requirements{Colors.END}")
            return False
        
        # Install GPU-specific packages if requested
        if self.args.gpu:
            print(f"🎮 Installing GPU-specific packages...")
            gpu_success = self.run_command(
                f"{self.python_executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
            if gpu_success:
                print(f"{Colors.GREEN}✅ GPU packages installed{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ GPU package installation failed, continuing with CPU version{Colors.END}")
        
        # Install high priority packages for enhanced functionality
        high_priority_packages = [
            "ultralytics>=8.0.0",  # YOLOv8 face detection
            "numba>=0.56.0",  # Performance acceleration
        ]
        
        # Special packages requiring custom installation logic
        special_packages = [
            "vbench",  # Video generation evaluation benchmark (PyTorch version compatibility)
        ]
        
        print(f"🎯 Installing high priority packages for enhanced functionality...")
        for package in high_priority_packages:
            print(f"   Installing {package}...")
            success = self.run_command(f"{self.python_executable} -m pip install {package}", check=False)
            if success:
                print(f"   ✅ {package.split('>=')[0]} installed successfully")
            else:
                print(f"   ⚠️ {package.split('>=')[0]} installation failed (optional)")
        
        # Install special packages with custom logic
        print(f"\n🔧 Installing special packages with enhanced compatibility...")
        for package in special_packages:
            if package == "vbench":
                self._install_vbench()
            else:
                print(f"   Installing {package}...")
                success = self.run_command(f"{self.python_executable} -m pip install {package}", check=False)
                if success:
                    print(f"   ✅ {package} installed successfully")
                else:
                    print(f"   ⚠️ {package} installation failed")
        
        # Special handling for MediaPipe (Platform-dependent installation)
        print(f"🔧 Installing MediaPipe with platform compatibility handling...")
        self._install_mediapipe()
        
        # Install additional useful packages
        extra_packages = [
            "jupyter",  # For notebooks
            "matplotlib",  # For visualization
            "seaborn",  # For better plots
            "tqdm",  # Progress bars
        ]
        
        print(f"🔧 Installing additional useful packages...")
        for package in extra_packages:
            self.run_command(f"{self.python_executable} -m pip install {package}", check=False)
        
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
            print(f"🚀 Using automated GIM installer: {gim_installer}")
            
            # Build command with proper arguments
            cmd_parts = [self.python_executable, str(gim_installer)]
            if self.args.force:
                cmd_parts.append("--force")
                
            # Add installation path (GIM installer uses --path)
            cmd_parts.extend(["--path", str(self.setup_dir)])
            
            success = self.run_command(cmd_parts)
        else:
            print(f"📥 GIM installer not found, using manual installation...")
            success = self._manual_gim_install()
        
        if success:
            print(f"{Colors.GREEN}✅ GIM installed successfully - Enhanced image matching available.{Colors.END}")
            self.gim_installed = True
            
            # Verify GIM installation
            gim_path = self.setup_dir / "gim"
            if gim_path.exists():
                demo_file = gim_path / "demo.py"
                if demo_file.exists():
                    print(f"   📄 GIM demo available: {demo_file}")
                    print(f"   🚀 Test with: python {demo_file} --model gim_roma")
        else:
            print(f"{Colors.YELLOW}⚠️ GIM installation failed, will use fallback implementation{Colors.END}")
            print(f"   You can install GIM later with: python {gim_installer}")
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
            success = self.run_command(f"{self.python_executable} -m pip install -e . --verbose", cwd=gim_path)
            
            if success:
                print(f"✅ GIM installed successfully in development mode")
                return True
            else:
                print(f"❌ GIM pip installation failed")
                return False
            
        except Exception as e:
            print(f"{Colors.RED}❌ Manual GIM installation failed: {e}{Colors.END}")
            return False
    
    def _install_vbench(self):
        """Install VBench with PyTorch compatibility handling"""
        print(f"   Installing VBench (video generation evaluation benchmark)...")
        
        # First try normal installation
        success = self.run_command(f"{self.python_executable} -m pip install vbench", check=False)
        if success:
            print(f"   ✅ VBench installed successfully (standard method)")
            return
        
        # If failed, try with --no-deps to bypass PyTorch version conflicts
        print(f"   ⚠️ Standard installation failed, trying compatibility mode...")
        success = self.run_command(f"{self.python_executable} -m pip install vbench --no-deps", check=False)
        if success:
            print(f"   ✅ VBench installed successfully (compatibility mode)")
            # Test if VBench works
            test_success = self.run_command(f"{self.python_executable} -c 'from vbench import VBench; print(\"VBench functional\")'", check=False)
            if test_success:
                print(f"   ✅ VBench functionality verified")
            else:
                print(f"   ⚠️ VBench installed but functionality test failed")
        else:
            print(f"   ❌ VBench installation failed")
            print(f"   💡 You can try manual installation:")
            print(f"      pip install vbench --no-deps")
    
    def _install_mediapipe(self):
        """Install MediaPipe with platform-specific handling"""
        print(f"📦 MediaPipe Installation - Enhanced Face Detection & Tracking")
        
        # MediaPipe installation strategies in order of preference
        installation_strategies = [
            # Strategy 1: Try latest stable version
            {
                "name": "Latest Stable Version",
                "command": f"{self.python_executable} -m pip install mediapipe>=0.10.0",
                "description": "Standard MediaPipe installation"
            },
            # Strategy 2: Try without version constraint
            {
                "name": "Any Available Version",
                "command": f"{self.python_executable} -m pip install mediapipe",
                "description": "MediaPipe without version constraint"
            },
            # Strategy 3: Try pre-release versions
            {
                "name": "Pre-release Version",
                "command": f"{self.python_executable} -m pip install --pre mediapipe",
                "description": "MediaPipe pre-release version"
            },
            # Strategy 4: Try with specific Python version compatibility
            {
                "name": "Compatible Version",
                "command": f"{self.python_executable} -m pip install 'mediapipe>=0.8.0,<0.11.0'",
                "description": "MediaPipe with version range"
            }
        ]
        
        for i, strategy in enumerate(installation_strategies, 1):
            print(f"   🔄 Strategy {i}: {strategy['name']}")
            print(f"      {strategy['description']}")
            
            success = self.run_command(strategy["command"], check=False)
            if success:
                print(f"   ✅ MediaPipe installed successfully using {strategy['name']}")
                
                # Verify installation
                verify_success = self.run_command(
                    f"{self.python_executable} -c \"import mediapipe as mp; print(f'MediaPipe v{{mp.__version__}} ready')\"",
                    check=False
                )
                if verify_success:
                    print(f"   ✅ MediaPipe verification successful")
                    return True
                else:
                    print(f"   ⚠️ MediaPipe installed but verification failed")
                    return True
            else:
                print(f"   ❌ Strategy {i} failed, trying next approach...")
        
        # If all strategies fail, provide helpful information
        print(f"   ⚠️ MediaPipe installation failed with all strategies")
        print(f"   📋 MediaPipe Platform Notes:")
        print(f"      • Requires Python 3.8-3.12 (current: {sys.version_info.major}.{sys.version_info.minor})")
        print(f"      • Supports Windows, macOS, Linux x86_64")
        print(f"      • ARM/Apple Silicon may need special builds")
        print(f"   💡 Fallback: System will use Ultralytics or OpenCV for face detection")
        
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
            print(f"\n📦 Processing {model_name}: {model_info['description']}")
            print(f"   Size: {model_info['size']}")
            
            model_path = self.models_dir / model_info['filename']
            
            # Check if file already exists
            if model_path.exists():
                print(f"{Colors.GREEN}✅ {model_name} already exists{Colors.END}")
                success_count += 1
                continue
            
            # Handle local_only models (check if they exist, don't download)
            if model_info.get('local_only', False):
                print(f"{Colors.YELLOW}⚠️ {model_name} is local-only but not found{Colors.END}")
                print(f"   Expected at: {model_path}")
                continue
            
            # Skip models without URL
            if not model_info.get('url'):
                print(f"{Colors.YELLOW}⚠️ {model_name} has no download URL{Colors.END}")
                continue
                
            print(f"   URL: {model_info['url']}")
            
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
        
        # Copy models to appropriate locations
        print(f"\n🔄 Setting up model paths...")
        self._setup_model_paths()
        
        # Calculate priority statistics
        high_priority_models = [name for name, info in self.models_info.items() if info.get("priority") == "high"]
        high_priority_downloaded = sum(1 for name in high_priority_models 
                                     if (self.models_dir / self.models_info[name]["filename"]).exists())
        
        print(f"\n📊 Model Download Summary:")
        print(f"   Total models: {success_count}/{total_models} downloaded")
        print(f"   High priority: {high_priority_downloaded}/{len(high_priority_models)} downloaded")
        
        if success_count == total_models:
            print(f"{Colors.GREEN}✅ All models downloaded successfully - full functionality available{Colors.END}")
            self.models_downloaded = True
        elif high_priority_downloaded >= len(high_priority_models):
            print(f"{Colors.GREEN}✅ All high priority models downloaded - core functionality available{Colors.END}")
            self.models_downloaded = True
        elif success_count > 0:
            print(f"{Colors.YELLOW}⚠️ Some models downloaded, toolkit will work with reduced functionality{Colors.END}")
            self.models_downloaded = True
        else:
            print(f"{Colors.RED}❌ No models downloaded, some features may not work{Colors.END}")
            self.models_downloaded = False
        
        return True
    
    def _setup_model_paths(self):
        """Setup model paths for different components"""
        # Ensure calculators/models directory exists
        calculators_models_dir = self.setup_dir / "calculators" / "models"
        calculators_models_dir.mkdir(parents=True, exist_ok=True)
        
        # SyncNet model now uses main models/ directory directly (no copy needed)
        syncnet_src = self.models_dir / "syncnet_v2.model"
        if syncnet_src.exists():
            print(f"   ✅ SyncNet model available in models/")
        
        # Copy S3FD model 
        s3fd_src = self.models_dir / "sfd_face.pth"
        s3fd_dst = self.models_dir / "sfd_face.pth"  # Keep in models dir for S3FD
        if s3fd_src.exists():
            print(f"   ✅ S3FD model → models/")
        
        # YOLOv8 and OpenCV models stay in models directory
        yolo_model = self.models_dir / "yolov8n-face.pt"
        opencv_model = self.models_dir / "opencv_face_detector_uint8.pb"
        opencv_config = self.models_dir / "opencv_face_detector.pbtxt"
        
        if yolo_model.exists():
            print(f"   ✅ YOLOv8 face model → models/")
        if opencv_model.exists() and opencv_config.exists():
            print(f"   ✅ OpenCV DNN models → models/")
    
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
            ("Basic Import", "from core.video_metrics_calculator import VideoMetricsCalculator"),
            ("CLIP API", "from apis.clip_api import CLIPVideoAPI"),
            ("GIM Calculator", "from calculators.gim_calculator import GIMMatchingCalculator"),
            ("LSE Calculator", "from calculators.lse_calculator import LSECalculator"),
        ]
        
        passed_tests = 0
        
        for test_name, test_code in verification_tests:
            try:
                print(f"🧪 Testing {test_name}...")
                test_result = self.run_command(f"PYTHONPATH=. python -c \"{test_code}\"", capture_output=True)
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
            f"PYTHONPATH=. python -c \"from calculators.gim_calculator import GIMMatchingCalculator; calc = GIMMatchingCalculator(); info = calc.get_model_info(); print(f'GIM available: {{info[\\\"gim_available\\\"]}}')\"",
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
from core.video_metrics_calculator import VideoMetricsCalculator

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
- Check installation: `PYTHONPATH=. python -c "from core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working.')"`
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
        print(f"   • Test installation: PYTHONPATH=. python -c \"from core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working.')\"")
        print(f"   • Check GIM status: PYTHONPATH=. python -c \"from calculators.gim_calculator import GIMMatchingCalculator; print(GIMMatchingCalculator().get_model_info())\"")
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
    • MediaPipe: Google's advanced face detection and tracking framework
    • Ultralytics: YOLOv8-based face detection with superior accuracy
    • NumBA: JIT compilation for numerical performance acceleration
    • VBench: Comprehensive video generation evaluation benchmark (v0.1.5+)
    • Official GIM: State-of-the-art image matching (ICLR 2024)

VBench Features (v0.1.5+):
    • High-resolution video quality assessment
    • Customized video evaluation support
    • Enhanced preprocessing for imaging quality
    • Compatible with PyTorch 2.0+ (smart dependency handling)

MediaPipe Features:
    • Real-time face detection and landmarks
    • Multi-face tracking capabilities
    • Hand and pose estimation support
    • Cross-platform compatibility (Windows/macOS/Linux)
    • Optimized for both CPU and GPU acceleration
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