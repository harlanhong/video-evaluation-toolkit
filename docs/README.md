# 📊 Video Evaluation Toolkit

**Version**: 2.1.0 | **Last Updated**: 2025-08-02  
**Language**: [English](README.md) | [中文](README_CN.md)

A comprehensive, production-ready toolkit for video quality assessment with LSE, VBench, image quality metrics, and advanced face detection capabilities.

## 🚀 Quick Start

### One-Click Installation

```bash
# Clone and install everything automatically
git clone https://github.com/harlanhong/video-evaluation-toolkit.git
cd video-evaluation-toolkit
python setup.py --gpu  # or bash install.sh --gpu
```

### Basic Usage

```python
from core.video_metrics_calculator import VideoMetricsCalculator

# Fast evaluation (recommended for most cases)
calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics("video.mp4")

# Batch evaluation with face-only metrics
results = calculator.calculate_batch_metrics(
    pred_dir="/path/to/videos",
    gt_dir="/path/to/ground_truth",
    region="face_only"  # or "full_image"
)
```

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **LSE Calculation** | Lip-sync error with SyncNet | ✅ Stable |
| **Face Quality Metrics** | PSNR/SSIM/LPIPS for face regions | ✅ **Fixed v2.1.0** |
| **VBench Integration** | 6 core VBench metrics | ✅ Stable |
| **Advanced Face Detection** | MediaPipe + YOLOv8 + fallbacks | ✅ Stable |
| **Official GIM Integration** | State-of-the-art image matching | ✅ Stable |
| **One-Click Installation** | Automated environment setup | ✅ Stable |

## 📊 Supported Metrics

### Core Metrics (No Ground Truth Required)
- **Video Info**: Frame count, resolution, FPS, duration
- **Image Quality**: Brightness, contrast, saturation, sharpness  
- **Face Analysis**: Detection rate, size, stability
- **Lip Sync**: LSE distance and confidence
- **VBench**: Subject/background consistency, motion smoothness, aesthetic quality

### Comparison Metrics (Requires Ground Truth)
- **Image Quality**: PSNR, SSIM, LPIPS (full image or face region)
- **Region Selection**: `--region face_only` or `--region full_image`

## 💻 Command Line Usage

```bash
# Basic face metrics evaluation
python core/video_metrics_calculator.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --region face_only \
    --output results.json

# Full evaluation with all metrics
python core/video_metrics_calculator.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --region face_only \
    --vbench \
    --clip \
    --gim \
    --output comprehensive_results.json
```

## 🔧 Installation Options

### Automatic (Recommended)
```bash
# Python installer (cross-platform)
python setup.py --mode conda --gpu

# Bash installer (Linux/macOS)  
bash install.sh --gpu
```

### Manual Installation
```bash
# Create environment
conda env create -f configs/environment.yaml
conda activate video-evaluation

# Install high-priority packages
pip install mediapipe ultralytics vbench numba

# Install project
pip install -r configs/requirements.txt
```

## 📈 What's New in v2.1.0

### 🔧 Critical Fixes
- **Fixed face metrics calculation** for different video resolutions
- **Fixed statistics aggregation** - PSNR/SSIM/LPIPS now properly included in JSON summaries
- **Enhanced GT frame handling** with automatic resizing and face re-detection
- **Improved video file matching** for complex naming patterns

### ⚡ Performance Improvements  
- **Up to 3x faster** face detection with MediaPipe integration
- **Intelligent fallback** system for maximum compatibility
- **One-click installation** with automatic dependency resolution

### 🎯 Results Quality
```
Before v2.1.0: PSNR=9.6dB, SSIM=0.17 (❌ incorrect)
After v2.1.0:  PSNR=27.7dB, SSIM=0.87 (✅ realistic)
```

## 🏗️ Architecture

```
video-evaluation-toolkit/
├── setup.py                    # One-click Python installer
├── install.sh                  # One-click Bash installer  
├── core/
│   └── video_metrics_calculator.py  # Main evaluation engine
├── calculators/                 # Individual metric calculators
│   ├── lse_calculator.py       # LSE/lip-sync metrics
│   ├── gim_calculator.py       # Image matching (GIM)
│   └── vbench_calculator.py    # VBench integration
├── apis/
│   └── clip_api.py             # Unified CLIP interface
├── examples/                    # Usage demonstrations
├── docs/                        # Documentation
├── configs/                     # Environment configurations
└── models/                      # Pre-trained models (auto-downloaded)
```

## 📊 Output Example

```json
{
  "summary": {
    "total_videos": 20,
    "success_rate": "100.00%",
    "average_metrics": {
      "Comparison Metrics": {
        "psnr": 27.72,        // ✅ Now included in summary
        "ssim": 0.871,        // ✅ Properly calculated  
        "lpips": 0.043        // ✅ Realistic values
      },
      "LSE Metrics": {
        "lse_distance": 8.26,
        "lse_confidence": 6.52
      }
    }
  },
  "individual_results": [...]
}
```

## 🚨 Troubleshooting

### Common Issues
```bash
# Model download issues
python setup.py --download-models

# Environment activation problems  
conda activate video-evaluation
export PYTHONPATH=.

# GPU memory issues
python core/video_metrics_calculator.py --device cpu
```

### Face Detection Fallback Order
1. **MediaPipe** (recommended, 95% accuracy)
2. **YOLOv8** (fast, good accuracy) 
3. **OpenCV DNN** (moderate accuracy)
4. **Haar Cascade** (basic fallback)

## 📋 Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **2.1.0** | 2025-08-02 | 🔧 Critical face metrics fixes, resolution handling, statistics aggregation |
| **2.0.1** | 2025-08-01 | ⚡ One-click installation, MediaPipe integration, GIM support |
| **2.0.0** | 2025-07-30 | 🎯 VBench integration, modular architecture, CLIP API |
| **1.0.0** | 2025-07-25 | 🎉 Initial release with LSE calculator and basic metrics |

## 📄 Documentation

- **[Quick Start Guide](../QUICK_START.md)** - Get up and running in 5 minutes
- **[MediaPipe Integration](MEDIAPIPE_INTEGRATION.md)** - Advanced face detection setup
- **[GIM Integration](GIM_INTEGRATION.md)** - Image matching configuration  
- **[API Reference](apis/README.md)** - Detailed API documentation
- **[Changelog](../CHANGELOG.md)** - Complete version history

## 🎯 System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: 8GB+ RAM (16GB+ for VBench)
- **GPU**: NVIDIA GPU with CUDA 11.0+ (optional but recommended)
- **Storage**: 5GB+ for models and cache

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

### Quick Contribution Guide

- 🐛 **Report bugs**: [Create an issue](https://github.com/harlanhong/video-evaluation-toolkit/issues/new)
- 💡 **Request features**: [Start a discussion](https://github.com/harlanhong/video-evaluation-toolkit/discussions)
- 🔧 **Submit code**: Fork → Branch → Code → Test → Pull Request
- 📚 **Improve docs**: Documentation improvements are always welcome

For detailed guidelines, development setup, and code standards, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/harlanhong/video-evaluation-toolkit/issues)
- **Email**: fatinghong@gmail.com
- **Discussions**: [GitHub Discussions](https://github.com/harlanhong/video-evaluation-toolkit/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**🎉 Start evaluating your videos with state-of-the-art metrics in just 2 commands!**

```bash
python setup.py --gpu
python core/video_metrics_calculator.py --pred_dir /your/videos --region face_only
```