# 📊 Video Evaluation Toolkit v2.1.0

**State-of-the-art video quality assessment with LSE, VBench, and advanced face detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/harlanhong/video-evaluation-toolkit.svg)](https://github.com/harlanhong/video-evaluation-toolkit/issues)

## 🚀 Quick Start

```bash
# One-click installation
python setup.py --gpu

# Start evaluating
python core/video_metrics_calculator.py \
    --pred_dir /your/videos \
    --gt_dir /your/references \
    --region face_only \
    --output results.json
```

## ✨ Key Features

- **🔧 Fixed Face Metrics** - Realistic PSNR/SSIM/LPIPS values (v2.1.0)
- **🎯 Region Selection** - Face-only or full-image quality assessment  
- **⚡ Advanced Face Detection** - MediaPipe, YOLOv8, intelligent fallbacks
- **📊 Comprehensive Metrics** - LSE, VBench, CLIP, GIM integration
- **🚀 One-Click Setup** - Automated environment and model installation

## 📊 What's New in v2.1.0

### 🔧 Critical Fixes
- **Face metrics calculation**: Fixed resolution mismatch issues
- **Statistics aggregation**: All metrics now included in JSON summaries  
- **Realistic values**: PSNR improved from 9.6dB → 27.7dB

### 🎯 New Features  
- **Region parameter**: Choose between `face_only` or `full_image`
- **Cross-resolution support**: Automatic GT frame resizing
- **Enhanced compatibility**: Better video file matching

## 📚 Documentation

- **[📖 Complete Documentation](docs/README.md)** - Full feature guide
- **[⚡ Quick Start Guide](QUICK_START.md)** - 5-minute setup  
- **[🤝 Contributing Guidelines](CONTRIBUTING.md)** - How to contribute
- **[📋 Changelog](CHANGELOG.md)** - Version history

### Specialized Guides
- **[MediaPipe Integration](docs/MEDIAPIPE_INTEGRATION.md)** - Advanced face detection
- **[GIM Integration](docs/GIM_INTEGRATION.md)** - Image matching setup
- **[API Reference](apis/README.md)** - Detailed API docs

## 🎯 Usage Examples

### Basic Face Quality Assessment
```python
from core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="generated.mp4",
    gt_path="reference.mp4", 
    region="face_only"
)

print(f"Face PSNR: {metrics['psnr']:.1f} dB")  # ~27.7 dB
print(f"Face SSIM: {metrics['ssim']:.3f}")     # ~0.87
print(f"LSE Score: {metrics['lse_distance']:.1f}")  # ~8.3
```

### Command Line Usage
```bash
# Face-only metrics (recommended for lip-sync)
python core/video_metrics_calculator.py \
    --pred_dir /generated/videos \
    --gt_dir /reference/videos \
    --region face_only

# Full image metrics  
python core/video_metrics_calculator.py \
    --pred_dir /videos \
    --region full_image \
    --vbench --clip --gim
```

## 📈 Performance

- **3x faster** face detection with MediaPipe
- **Fixed calculations** ensuring realistic metric values
- **Cross-platform** Windows, macOS, Linux support
- **GPU acceleration** for faster processing

## 🎯 Typical Results (v2.1.0)

| Metric | Face Region | Full Image | Good Values |
|--------|-------------|------------|-------------|
| **PSNR** | 25-35 dB | 30-40 dB | Higher = Better |
| **SSIM** | 0.8-0.95 | 0.85-0.98 | Closer to 1 |
| **LPIPS** | 0.02-0.15 | 0.01-0.10 | Lower = Better |
| **LSE** | 5-12 | - | Lower = Better |

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md).

**Quick contributing steps:**
1. 🐛 [Report bugs](https://github.com/harlanhong/video-evaluation-toolkit/issues/new)
2. 💡 [Request features](https://github.com/harlanhong/video-evaluation-toolkit/discussions)  
3. 🔧 Submit code: Fork → Branch → Code → Test → Pull Request

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/harlanhong/video-evaluation-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/harlanhong/video-evaluation-toolkit/discussions)
- **Email**: fatinghong@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if it helps your research!**

📖 **[Read Full Documentation](docs/README.md)** | 🚀 **[Quick Start](QUICK_START.md)** | 🤝 **[Contribute](CONTRIBUTING.md)**