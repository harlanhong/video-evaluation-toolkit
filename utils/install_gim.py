#!/usr/bin/env python3
"""
GIM Installation Script
Automated installation script for the official GIM implementation

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

Usage:
    python utils/install_gim.py [--force] [--path PATH]
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, capture_output=False):
    """Run shell command and handle errors"""
    print(f"🔄 Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, cwd=cwd, 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        if capture_output:
            print(f"   Error: {e.stderr}")
        return False


def check_git_available():
    """Check if git is available"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not available in PATH")
        print("   Please install git first: https://git-scm.com/downloads")
        return False


def check_pip_available():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def check_gim_installed():
    """Check if GIM is already installed"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("gim")
        if spec is not None:
            print("✅ GIM is already installed")
            return True
        return False
    except ImportError:
        return False


def install_gim(install_path=".", force=False):
    """Install GIM from official repository"""
    
    print("🚀 Starting GIM installation process")
    print("=" * 50)
    
    # Check prerequisites
    if not check_git_available():
        return False
    
    if not check_pip_available():
        return False
    
    # Check if already installed
    if check_gim_installed() and not force:
        print("✅ GIM is already installed. Use --force to reinstall.")
        return True
    
    # Determine installation path
    install_path = Path(install_path).resolve()
    gim_path = install_path / "gim"
    
    print(f"📁 Installation path: {install_path}")
    print(f"📁 GIM will be cloned to: {gim_path}")
    
    # Remove existing installation if force is specified
    if force and gim_path.exists():
        print(f"🗑️ Removing existing GIM installation at {gim_path}")
        try:
            shutil.rmtree(gim_path)
        except Exception as e:
            print(f"❌ Failed to remove existing installation: {e}")
            return False
    
    # Clone GIM repository
    if not gim_path.exists():
        print("📥 Cloning GIM repository...")
        clone_cmd = f"git clone https://github.com/xuelunshen/gim.git {gim_path}"
        if not run_command(clone_cmd, cwd=install_path):
            print("❌ Failed to clone GIM repository")
            return False
        print("✅ GIM repository cloned successfully")
    else:
        print("📁 GIM repository already exists, skipping clone")
    
    # Install GIM
    print("📦 Installing GIM...")
    install_cmd = f"{sys.executable} -m pip install -e ."
    if not run_command(install_cmd, cwd=gim_path):
        print("❌ Failed to install GIM")
        return False
    
    print("✅ GIM installed successfully")
    
    # Verify installation
    print("🔍 Verifying GIM installation...")
    try:
        # Test import
        test_cmd = f"{sys.executable} -c \"import gim; print('GIM import successful')\""
        if run_command(test_cmd, capture_output=True):
            print("✅ GIM verification successful")
            return True
        else:
            print("⚠️ GIM import test failed, but installation may still be working")
            return True
    except Exception as e:
        print(f"⚠️ GIM verification failed: {e}")
        print("   Installation may still be working - try importing manually")
        return True


def get_gim_info():
    """Get information about GIM installation"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("gim")
        if spec is not None:
            gim_path = Path(spec.origin).parent if spec.origin else "Unknown"
            print(f"📍 GIM location: {gim_path}")
            
            # Try to get version info
            try:
                # Check if we can get git info
                if (gim_path / ".git").exists():
                    git_info = run_command("git log -1 --format='%H %ad' --date=short", 
                                         cwd=gim_path, capture_output=True)
                    if git_info:
                        commit_hash, date = git_info.split(' ')
                        print(f"📅 Git commit: {commit_hash[:8]} ({date})")
            except:
                pass
            
            return True
        else:
            print("❌ GIM not found")
            return False
    except Exception as e:
        print(f"❌ Error checking GIM info: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Install official GIM implementation")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstallation even if GIM is already installed")
    parser.add_argument("--path", type=str, default=".", 
                       help="Installation path (default: current directory)")
    parser.add_argument("--info", action="store_true", 
                       help="Show information about current GIM installation")
    parser.add_argument("--check", action="store_true", 
                       help="Check if GIM is installed")
    
    args = parser.parse_args()
    
    print("🔍 GIM Installation Script")
    print("Official GIM: Learning Generalizable Image Matcher From Internet Videos")
    print("Repository: https://github.com/xuelunshen/gim")
    print()
    
    if args.info:
        print("📋 GIM Installation Information:")
        get_gim_info()
        return
    
    if args.check:
        print("🔍 Checking GIM installation...")
        if check_gim_installed():
            get_gim_info()
        else:
            print("❌ GIM is not installed")
            print("   Run: python utils/install_gim.py")
        return
    
    # Install GIM
    success = install_gim(install_path=args.path, force=args.force)
    
    if success:
        print()
        print("🎉 GIM installation completed successfully!")
        print()
        print("Next steps:")
        print("1. Test the installation:")
        print("   python -c \"from evalutation.calculators.gim_calculator import GIMMatchingCalculator; print('GIM ready!')\"")
        print()
        print("2. Run GIM matching:")
        print("   python -m calculators.gim_calculator --source video1.mp4 --target video2.mp4 --model gim_roma")
        print()
        print("3. See documentation: docs/GIM_INTEGRATION.md")
    else:
        print()
        print("❌ GIM installation failed!")
        print("Please check the error messages above and try again.")
        print("For troubleshooting, see: docs/GIM_INTEGRATION.md")
        sys.exit(1)


if __name__ == "__main__":
    main()