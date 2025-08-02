#!/bin/bash

# Video Evaluation Toolkit - One-Click Installation Script
# Automated installation script for complete environment setup
#
# Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
# All rights reserved.
#
# Usage:
#   bash install.sh [OPTIONS]
#
# Options:
#   --mode MODE       Installation mode: auto, conda, venv, pip (default: auto)
#   --gpu            Install GPU support (CUDA)
#   --skip-models    Skip model downloads
#   --force          Force reinstall (remove existing)
#   --help           Show this help message

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default options
INSTALL_MODE="auto"
GPU_SUPPORT=false
SKIP_MODELS=false
FORCE_INSTALL=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_status() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${BOLD}${BLUE}$(printf '=%.0s' {1..70})"
    echo "🎬 VIDEO EVALUATION TOOLKIT - ONE-CLICK INSTALLATION"
    echo "$(printf '=%.0s' {1..70})${NC}"
    echo -e "${GREEN}Advanced video quality assessment and synchronization evaluation${NC}"
    echo -e "${YELLOW}Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>${NC}"
    echo -e "\n📍 Installation Directory: $SCRIPT_DIR"
    echo -e "🖥️  Platform: $(uname -s) $(uname -r)"
    echo -e "🐍 Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo
}

# Function to show help
show_help() {
    cat << EOF
Video Evaluation Toolkit - One-Click Installation Script

USAGE:
    bash install.sh [OPTIONS]

OPTIONS:
    --mode MODE       Installation mode (auto, conda, venv, pip)
                     auto: Auto-detect best method (default)
                     conda: Use conda environment
                     venv: Use Python virtual environment
                     pip: Use pip only (no environment)
    
    --gpu            Install GPU support (CUDA PyTorch)
    --skip-models    Skip downloading model files
    --force          Force reinstall (remove existing environments)
    --help           Show this help message

EXAMPLES:
    bash install.sh                    # Auto-detect best installation method
    bash install.sh --mode conda       # Use conda environment
    bash install.sh --gpu              # Install with GPU support + high priority packages
    bash install.sh --skip-models      # Skip model downloads
    bash install.sh --force            # Force clean reinstall

HIGH PRIORITY PACKAGES (Auto-installed):
    • MediaPipe: Enhanced face detection and tracking
    • Ultralytics: YOLOv8-based face detection
    • NumBA: Performance acceleration
    • Official GIM: State-of-the-art image matching (ICLR 2024)

REQUIREMENTS:
    - Python 3.8 or higher
    - Git (for GIM installation)
    - Internet connection (for downloads)
    - At least 5GB free disk space

For more information, see: docs/README.md
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            INSTALL_MODE="$2"
            shift 2
            ;;
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --force)
            FORCE_INSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    echo -e "${BOLD}📋 STEP 1: Checking System Requirements${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    local requirements_met=true
    
    # Check Python
    if command_exists python3; then
        python_version=$(python3 --version | cut -d' ' -f2)
        python_major=$(echo $python_version | cut -d'.' -f1)
        python_minor=$(echo $python_version | cut -d'.' -f2)
        
        if [[ $python_major -ge 3 && $python_minor -ge 8 ]]; then
            print_success "Python $python_version (OK)"
        else
            print_error "Python $python_version (Requires ≥3.8)"
            requirements_met=false
        fi
    else
        print_error "Python3 not found"
        requirements_met=false
    fi
    
    # Check Git
    if command_exists git; then
        git_version=$(git --version)
        print_success "Git available: $git_version"
    else
        print_error "Git not found (Required for GIM installation)"
        requirements_met=false
    fi
    
    # Check Conda (optional)
    if command_exists conda; then
        conda_version=$(conda --version)
        print_success "Conda available: $conda_version"
    else
        print_warning "Conda not found (Will use pip/venv)"
    fi
    
    # Check disk space
    available_space=$(df "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
    available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -ge 5 ]]; then
        print_success "Disk space: ${available_gb} GB available"
    else
        print_error "Insufficient disk space: ${available_gb} GB (Requires ≥5GB)"
        requirements_met=false
    fi
    
    echo
    
    if [[ "$requirements_met" != true ]]; then
        print_error "System requirements not met. Please fix the issues above."
        exit 1
    fi
}

# Function to detect best installation method
detect_install_method() {
    if [[ "$INSTALL_MODE" == "auto" ]]; then
        if command_exists conda; then
            INSTALL_MODE="conda"
            print_status "Auto-detected installation mode: conda"
        else
            INSTALL_MODE="venv"
            print_status "Auto-detected installation mode: venv"
        fi
    fi
}

# Function to setup environment
setup_environment() {
    echo -e "${BOLD}🏗️  STEP 2: Setting Up Environment${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    case $INSTALL_MODE in
        conda)
            setup_conda_environment
            ;;
        venv)
            setup_venv_environment
            ;;
        pip)
            setup_pip_only
            ;;
        *)
            print_error "Unknown installation mode: $INSTALL_MODE"
            exit 1
            ;;
    esac
}

# Function to setup conda environment
setup_conda_environment() {
    local env_name="video-evaluation"
    local env_file="$SCRIPT_DIR/configs/environment.yaml"
    
    print_status "Setting up conda environment: $env_name"
    
    # Check if environment exists
    if conda env list | grep -q "^$env_name "; then
        if [[ "$FORCE_INSTALL" == true ]]; then
            print_status "Removing existing environment: $env_name"
            conda env remove -n "$env_name" -y
        else
            print_warning "Environment $env_name already exists. Use --force to recreate."
            return 0
        fi
    fi
    
    # Create environment
    if [[ -f "$env_file" ]]; then
        print_status "Creating environment from: $env_file"
        conda env create -f "$env_file"
    else
        print_status "Creating basic conda environment"
        conda create -n "$env_name" python=3.9 -y
    fi
    
    print_success "Conda environment created successfully"
    print_warning "💡 Activate with: conda activate $env_name"
}

# Function to setup virtual environment
setup_venv_environment() {
    local venv_path="$SCRIPT_DIR/venv"
    
    print_status "Setting up virtual environment: $venv_path"
    
    if [[ -d "$venv_path" ]]; then
        if [[ "$FORCE_INSTALL" == true ]]; then
            print_status "Removing existing virtual environment"
            rm -rf "$venv_path"
        else
            print_warning "Virtual environment already exists. Use --force to recreate."
            return 0
        fi
    fi
    
    # Create virtual environment
    python3 -m venv "$venv_path"
    
    print_success "Virtual environment created successfully"
    print_warning "💡 Activate with: source $venv_path/bin/activate"
}

# Function to setup pip only
setup_pip_only() {
    print_status "Using current Python environment with pip"
    print_warning "Installing directly to system Python (not recommended for production)"
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BOLD}📦 STEP 3: Installing Dependencies${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    local requirements_file="$SCRIPT_DIR/configs/requirements.txt"
    
    if [[ ! -f "$requirements_file" ]]; then
        print_error "Requirements file not found: $requirements_file"
        exit 1
    fi
    
    print_status "Installing from: $requirements_file"
    
    # Install base requirements
    python3 -m pip install -r "$requirements_file"
    
    # Install GPU-specific packages if requested
    if [[ "$GPU_SUPPORT" == true ]]; then
        print_status "Installing GPU-specific packages..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
            print_warning "GPU package installation failed, continuing with CPU version"
        }
    fi
    
    # Install high priority packages for enhanced functionality
    print_status "🎯 Installing high priority packages for enhanced functionality..."
    
    local high_priority_packages=(
        "mediapipe>=0.10.0"    # Enhanced face detection
        "ultralytics>=8.0.0"   # YOLOv8 face detection  
        "numba>=0.56.0"        # Performance acceleration
    )
    
    for package in "${high_priority_packages[@]}"; do
        package_name=${package%%>=*}
        print_status "Installing $package..."
        if python3 -m pip install "$package" 2>/dev/null; then
            print_success "   $package_name installed successfully"
        else
            print_warning "   $package_name installation failed (optional)"
        fi
    done
    
    # Install additional useful packages
    print_status "Installing additional useful packages..."
    python3 -m pip install jupyter matplotlib seaborn tqdm 2>/dev/null || true
    
    print_success "Dependencies installed successfully"
}

# Function to install GIM
install_gim() {
    echo -e "${BOLD}🔍 STEP 4: Installing Official GIM (High Priority)${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    print_status "🎯 GIM is a high-priority component for state-of-the-art image matching"
    print_status "   Installing official GIM implementation from ICLR 2024..."
    
    local gim_installer="$SCRIPT_DIR/utils/install_gim.py"
    
    if [[ -f "$gim_installer" ]]; then
        print_status "🚀 Using automated GIM installer..."
        local force_flag=""
        [[ "$FORCE_INSTALL" == true ]] && force_flag="--force"
        
        if python3 "$gim_installer" $force_flag; then
            print_success "GIM installed successfully - Enhanced image matching available!"
        else
            print_warning "GIM installation failed, will use fallback implementation"
            print_warning "   You can install GIM later with: python utils/install_gim.py"
        fi
    else
        print_status "📥 Manual GIM installation..."
        manual_gim_install
    fi
}

# Function for manual GIM installation
manual_gim_install() {
    local gim_path="$SCRIPT_DIR/gim"
    
    if [[ -d "$gim_path" && "$FORCE_INSTALL" == true ]]; then
        print_status "🗑️ Removing existing GIM installation..."
        rm -rf "$gim_path"
    fi
    
    if [[ ! -d "$gim_path" ]]; then
        print_status "📥 Cloning GIM repository from GitHub..."
        if git clone --progress https://github.com/xuelunshen/gim.git "$gim_path"; then
            print_success "✅ GIM repository cloned successfully"
        else
            print_warning "❌ Failed to clone GIM repository"
            return 1
        fi
    else
        print_status "📁 GIM repository already exists, updating..."
        (cd "$gim_path" && git pull origin main 2>/dev/null) || true
    fi
    
    print_status "🔧 Installing GIM in development mode..."
    if (cd "$gim_path" && python3 -m pip install -e . --verbose); then
        print_success "✅ GIM installed successfully in development mode"
        return 0
    else
        print_warning "❌ GIM pip installation failed"
        return 1
    fi
}

# Function to download models
download_models() {
    if [[ "$SKIP_MODELS" == true ]]; then
        print_warning "⏭️ Skipping model downloads (--skip-models)"
        return 0
    fi
    
    echo -e "${BOLD}🎭 STEP 5: Downloading Models and Checkpoints${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    local models_dir="$SCRIPT_DIR/models"
    mkdir -p "$models_dir"
    
    # Model information
    declare -A models=(
        ["syncnet"]="https://github.com/joonson/syncnet_python/releases/download/v0.0.1/syncnet_v2.model:syncnet_v2.model:~180MB"
        ["s3fd"]="https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth:s3fd.pth:~180MB"
    )
    
    local success_count=0
    local total_models=${#models[@]}
    
    for model_name in "${!models[@]}"; do
        IFS=':' read -r url filename size <<< "${models[$model_name]}"
        local model_path="$models_dir/$filename"
        
        echo
        print_status "Downloading $model_name: $filename"
        echo "   Size: $size"
        echo "   URL: $url"
        
        # Skip if file exists and not forcing
        if [[ -f "$model_path" && "$FORCE_INSTALL" != true ]]; then
            print_success "$model_name already exists"
            ((success_count++))
            continue
        fi
        
        # Download with wget or curl
        if command_exists wget; then
            wget -O "$model_path" "$url" --progress=bar:force 2>&1 | {
                while IFS= read -r line; do
                    if [[ $line =~ [0-9]+% ]]; then
                        echo -ne "\r   Progress: ${line##*] }"
                    fi
                done
                echo
            }
        elif command_exists curl; then
            curl -L -o "$model_path" "$url" --progress-bar
        else
            print_error "Neither wget nor curl found. Cannot download models."
            continue
        fi
        
        if [[ -f "$model_path" ]]; then
            print_success "$model_name downloaded successfully"
            ((success_count++))
        else
            print_error "Failed to download $model_name"
        fi
    done
    
    echo
    print_status "Model Download Summary: $success_count/$total_models models downloaded"
    
    if [[ $success_count -eq $total_models ]]; then
        print_success "All models downloaded successfully"
    elif [[ $success_count -gt 0 ]]; then
        print_warning "Some models downloaded, toolkit will work with reduced functionality"
    else
        print_error "No models downloaded, some features may not work"
    fi
}

# Function to verify installation
verify_installation() {
    echo -e "${BOLD}🔍 STEP 6: Verifying Installation${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    local passed_tests=0
    local total_tests=4
    
    # Test basic imports
    local tests=(
        "Basic Import:from evalutation.core.video_metrics_calculator import VideoMetricsCalculator"
        "CLIP API:from evalutation.apis.clip_api import CLIPVideoAPI"
        "GIM Calculator:from evalutation.calculators.gim_calculator import GIMMatchingCalculator"
        "LSE Calculator:from evalutation.calculators.lse_calculator import LSECalculator"
    )
    
    for test in "${tests[@]}"; do
        IFS=':' read -r test_name test_code <<< "$test"
        print_status "Testing $test_name..."
        
        if python3 -c "$test_code" >/dev/null 2>&1; then
            print_success "   $test_name - OK"
            ((passed_tests++))
        else
            print_error "   $test_name - Failed"
        fi
    done
    
    echo
    print_status "Additional checks..."
    
    # Check GIM availability
    local gim_status=$(python3 -c "
from evalutation.calculators.gim_calculator import GIMMatchingCalculator
calc = GIMMatchingCalculator()
info = calc.get_model_info()
print(f'GIM available: {info[\"gim_available\"]}')
" 2>/dev/null || echo "GIM check failed")
    
    if [[ "$gim_status" != "GIM check failed" ]]; then
        print_success "   $gim_status"
    fi
    
    # Check models
    local models_dir="$SCRIPT_DIR/models"
    local models_found=0
    [[ -f "$models_dir/syncnet_v2.model" ]] && ((models_found++))
    [[ -f "$models_dir/s3fd.pth" ]] && ((models_found++))
    echo "   📁 Models: $models_found/2 found"
    
    echo
    print_status "Verification Summary: $passed_tests/$total_tests tests passed"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        print_success "🎉 Installation verification successful!"
    else
        print_warning "Some verification tests failed, but basic functionality should work"
    fi
}

# Function to create quick start guide
create_quick_start_guide() {
    echo -e "${BOLD}📚 STEP 7: Creating Quick Start Guide${NC}"
    echo "$(printf -- '-%.0s' {1..50})"
    
    local guide_path="$SCRIPT_DIR/QUICK_START.md"
    
    cat > "$guide_path" << 'EOF'
# Video Evaluation Toolkit - Quick Start Guide

## 🎉 Installation Completed Successfully!

### Quick Usage Examples

#### 1. Basic Video Metrics
```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="your_video.mp4",
    gt_path="reference_video.mp4"  # Optional
)
print(f"LSE Score: {metrics['lse_score']}")
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
print(f"CLIP Similarity: {metrics['clip_similarity']:.4f}")
print(f"GIM Matching: {metrics['gim_matching_pixels']}")
```

#### 3. Command Line Usage
```bash
python -m core.video_metrics_calculator \
    --pred generated_video.mp4 \
    --gt reference_video.mp4 \
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
EOF
    
    print_success "Quick start guide created: $guide_path"
}

# Function to print final summary
print_summary() {
    echo
    echo -e "${BOLD}${GREEN}🎉 INSTALLATION COMPLETED!${NC}"
    echo "$(printf '=%.0s' {1..70})"
    
    print_status "Next Steps:"
    
    case $INSTALL_MODE in
        conda)
            echo "   1. Activate environment: conda activate video-evaluation"
            ;;
        venv)
            echo "   1. Activate environment: source venv/bin/activate"
            ;;
        pip)
            echo "   1. Environment ready (using system Python)"
            ;;
    esac
    
    echo "   2. Read quick start: cat QUICK_START.md"
    echo "   3. Try examples: python examples/basic_usage.py"
    echo "   4. Read documentation: docs/README.md"
    
    echo
    print_status "Useful Commands:"
    echo "   • Test installation: python -c \"from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working!')\""
    echo "   • Check GIM status: python -c \"from evalutation.calculators.gim_calculator import GIMMatchingCalculator; print(GIMMatchingCalculator().get_model_info())\""
    echo "   • Update toolkit: git pull origin main"
    
    echo
    echo -e "${BLUE}📧 Support: fatinghong@gmail.com"
    echo -e "🌐 Repository: https://github.com/harlanhong/video-evaluation-toolkit${NC}"
}

# Main execution
main() {
    print_header
    
    # Check system requirements
    check_requirements
    
    # Detect installation method
    detect_install_method
    
    # Setup environment
    setup_environment
    
    # Install dependencies
    install_dependencies
    
    # Install GIM
    install_gim
    
    # Download models
    download_models
    
    # Verify installation
    verify_installation
    
    # Create quick start guide
    create_quick_start_guide
    
    # Print summary
    print_summary
}

# Run main function
main "$@"