# Changelog

All notable changes to the Video Evaluation Toolkit project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- One-click installation script (`setup.py`) with comprehensive environment setup
- Automated model downloading and checkpoint management
- Quick start guide generation
- Advanced installation verification and testing

## [2.0.0] - 2025-01-XX

### Added
- **🔍 Official GIM Integration**: Direct integration with the official GIM implementation (ICLR 2024)
  - Support for all GIM model variants: GIM_RoMa, GIM_LightGlue, GIM_DKM, GIM_LoFTR, GIM_SuperGlue
  - Intelligent fallback mechanism when GIM is not available
  - GPU acceleration and batch processing support
  - Comprehensive performance benchmarking (53.3% MeanAUC@5° with GIM_RoMa)
- **🎯 Advanced Synchronization Metrics**:
  - CLIP-V similarity calculation between source and target frames
  - FVD-V score for video generation quality assessment
  - GIM matching pixels with confidence threshold filtering
- **🔧 Unified CLIP API** (`apis/clip_api.py`):
  - Comprehensive interface for all CLIP-based video evaluation tasks
  - Multi-model support (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
  - Video-to-video and text-to-video similarity calculations
  - Feature extraction and batch processing capabilities
  - Legacy compatibility with existing calculators
- **📁 Project Restructuring**:
  - Organized folder hierarchy: `apis/`, `calculators/`, `core/`, `docs/`, `configs/`, `examples/`, `utils/`
  - Proper Python package structure with `__init__.py` files
  - Standardized file naming conventions
- **🛠️ Installation and Setup Tools**:
  - Automated GIM installation script (`utils/install_gim.py`)
  - One-click setup script (`setup.py`) with environment management
  - Comprehensive dependency management for pip and conda
- **📚 Enhanced Documentation**:
  - Complete GIM integration guide (`docs/GIM_INTEGRATION.md`)
  - Detailed API documentation (`apis/README.md`)
  - Updated main documentation with new features
  - Chinese documentation support (`docs/README_CN.md`)
- **💻 Usage Examples**:
  - GIM demonstration (`examples/gim_demo.py`)
  - CLIP API examples (`examples/clip_api_demo.py`)
  - Advanced metrics usage (`examples/advanced_metrics.py`)
  - Basic usage examples (`examples/basic_usage.py`)

### Changed
- **🔄 Core Architecture Refactoring**:
  - `metrics_calculator.py` moved to `core/video_metrics_calculator.py`
  - All calculators moved to `calculators/` directory
  - Configuration files moved to `configs/` directory
  - Documentation centralized in `docs/` directory
- **⚡ Performance Improvements**:
  - Optimized video frame extraction and processing
  - Enhanced GPU memory management
  - Batch processing for multiple video pairs
  - Improved error handling and logging
- **🎨 Enhanced CLI Interface**:
  - Modular metric enabling (--clip, --gim, --fvd flags)
  - Improved progress reporting and verbose output
  - Better error messages and troubleshooting guidance

### Fixed
- **🐛 Bug Fixes**:
  - Face detection reliability improvements
  - Memory leak fixes in video processing
  - Cross-platform compatibility issues
  - Import path corrections after restructuring
- **🔧 Compatibility**:
  - Python 3.8+ compatibility
  - CUDA device detection and fallback
  - Windows/Linux/macOS support
  - Dependency version conflicts resolution

### Security
- Added copyright information to all source files
- Improved error handling to prevent information leakage
- Secure model downloading with integrity checks

## [1.5.0] - 2024-12-XX

### Added
- **🎬 VBench Integration**: Complete integration with VBench video generation quality assessment
  - 6 core VBench metrics: Subject Consistency, Background Consistency, Temporal Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality
  - Flexible enablement system for performance optimization
  - Automatic VBench environment detection
- **👄 Enhanced LSE (Lip-Sync Error) Calculation**:
  - Improved accuracy with advanced face detection
  - MediaPipe integration for better performance
  - Support for multiple face detection backends
- **🔧 Advanced Face Detection**:
  - MediaPipe face detection integration
  - YOLOv8-based face detection support
  - OpenCV DNN fallback options
  - Configurable detection confidence thresholds

### Changed
- **⚡ Performance Optimization**:
  - Faster video processing with optimized frame extraction
  - Reduced memory usage for large videos
  - Parallel processing for batch calculations
- **📊 Enhanced Metrics Output**:
  - Comprehensive statistics reporting
  - Average calculation across all enabled metrics
  - Detailed progress reporting with progress bars
- **🔧 Improved Configuration**:
  - Environment-specific configuration files
  - Better dependency management
  - Conda environment support

### Fixed
- Frame extraction reliability issues
- Memory management in long video processing
- Cross-platform path handling
- Dependency conflicts with different environments

## [1.0.0] - 2024-11-XX

### Added
- **🎯 Core Video Metrics**:
  - Basic video information extraction (resolution, FPS, duration, etc.)
  - Image quality statistics (mean, std, histogram analysis)
  - Motion analysis and temporal consistency
- **👤 Face Analysis**:
  - Face detection and tracking
  - Face region extraction and analysis
  - Lip-sync error (LSE) calculation using SyncNet
- **🔍 Ground Truth Comparison**:
  - Face PSNR, SSIM, LPIPS for face regions
  - Pixel-level comparison metrics
  - Quality assessment with reference videos
- **💻 Command-Line Interface**:
  - Easy-to-use CLI for single and batch processing
  - Configurable output formats (JSON, text)
  - Progress reporting and logging
- **🐍 Python API**:
  - Object-oriented calculator classes
  - Flexible metric enabling/disabling
  - Extensible architecture for custom metrics

### Dependencies
- **Core Requirements**:
  - PyTorch ≥2.0.0 for deep learning models
  - OpenCV ≥4.5.0 for video processing
  - NumPy ≥1.21.0 for numerical computations
  - SciPy ≥1.7.0 for scientific computing
- **Face Analysis**:
  - MediaPipe ≥0.10.0 for face detection
  - LPIPS ≥0.1.4 for perceptual similarity
- **Audio Processing**:
  - librosa ≥0.9.0 for audio analysis
  - python_speech_features for speech processing
- **Video Analysis**:
  - scenedetect ≥0.6.0 for scene detection

### Documentation
- Comprehensive README with installation and usage instructions
- Model download guide (`MODELS_DOWNLOAD.md`)
- Example scripts and demonstrations
- API reference documentation

## [0.1.0] - 2024-10-XX (Initial Release)

### Added
- Basic project structure
- Initial video processing capabilities
- Simple face detection using OpenCV
- Basic LSE calculation implementation
- Command-line interface prototype

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Categories

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for security improvements

## Links

- **Repository**: [https://github.com/harlanhong/video-evaluation-toolkit](https://github.com/harlanhong/video-evaluation-toolkit)
- **Documentation**: [docs/README.md](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/harlanhong/video-evaluation-toolkit/issues)
- **Releases**: [GitHub Releases](https://github.com/harlanhong/video-evaluation-toolkit/releases)

## Acknowledgments

- **GIM**: Official implementation from [xuelunshen/gim](https://github.com/xuelunshen/gim)
- **VBench**: Video generation quality assessment from the VBench team
- **CLIP**: OpenAI's CLIP model for vision-language understanding
- **SyncNet**: Lip-sync assessment methodology
- **MediaPipe**: Google's MediaPipe for face detection

---

*For detailed installation and usage instructions, see [docs/README.md](docs/README.md)*