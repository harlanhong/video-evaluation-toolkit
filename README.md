# 📊 Video Evaluation Toolkit v2.1.0

**State-of-the-art video quality assessment with LSE, VBench, and advanced face detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](documents/LICENSE)
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

- **[📖 Complete Documentation](documents/README.md)** - Full feature guide
- **[⚡ Quick Start Guide](documents/QUICK_START.md)** - 5-minute setup  
- **[🤝 Contributing Guidelines](documents/CONTRIBUTING.md)** - How to contribute
- **[📋 Changelog](documents/CHANGELOG.md)** - Version history

### Specialized Guides
- **[MediaPipe Integration](documents/MEDIAPIPE_INTEGRATION.md)** - Advanced face detection
- **[GIM Integration](documents/GIM_INTEGRATION.md)** - Image matching setup
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

### 📊 Core Quality Metrics
| Metric | Face Region | Full Image | Good Values | Notes |
|--------|-------------|------------|-------------|-------|
| **PSNR** | 25-35 dB | 30-40 dB | Higher = Better | Peak signal quality |
| **SSIM** | 0.8-0.95 | 0.85-0.98 | Closer to 1 | Structural similarity |
| **LPIPS** | 0.02-0.15 | 0.01-0.10 | Lower = Better | Perceptual distance |

### 🎵 Lip-Sync Metrics  
| Metric | Typical Range | Good Values | Notes |
|--------|---------------|-------------|-------|
| **LSE Distance** | 5-12 | Lower = Better | Audio-visual sync |
| **LSE Confidence** | 6-8 | Higher = Better | Sync reliability |
| **LSE-D MSE** ⭐ | 0.1-2.0 | Lower = Better | Sync error vs GT |
| **LSE-C MSE** ⭐ | 0.1-1.5 | Lower = Better | Confidence error vs GT |

### 🎨 CLIP Similarity Metrics
| Metric | Typical Range | Good Values | Notes |
|--------|---------------|-------------|-------|
| **CLIP Similarity** | 0.85-0.98 | Higher = Better | Semantic similarity |
| **CLIP Std Dev** | 0.01-0.05 | Lower = Better | Frame consistency |

### 🔥 VBench Metrics (0-100 scale)
| Metric | Typical Range | Good Values | Focus |
|--------|---------------|-------------|-------|
| **Subject Consistency** | 80-95 | Higher = Better | Character coherence |
| **Background Consistency** | 85-98 | Higher = Better | Scene stability |
| **Motion Smoothness** | 75-90 | Higher = Better | Movement quality |
| **Dynamic Degree** | 60-85 | Depends on content | Activity level |
| **Aesthetic Quality** | 70-90 | Higher = Better | Visual appeal |
| **Imaging Quality** | 80-95 | Higher = Better | Technical quality |

## 🏗️ Architecture

```
video-evaluation-toolkit/
├── setup.py                   # 🆕 One-click installation
├── install.sh                 # 🆕 Bash installer
├── core/                      # Core evaluation engines
│   └── video_metrics_calculator.py
├── calculators/               # Individual metric calculators
│   ├── lse_calculator.py     # LSE/lip-sync metrics
│   ├── gim_calculator.py     # Image matching (GIM)
│   └── vbench_calculator.py  # VBench integration
├── apis/                      # Unified API interfaces
│   └── clip_api.py           # CLIP video similarity
├── examples/                  # Usage demonstrations
├── docs/                      # Complete documentation
├── configs/                   # Environment configurations
└── models/                    # Pre-trained models (auto-downloaded)
```

## 🧪 Supported Metrics

### 📊 Core Video Quality Metrics
| Metric | Description | Output Keys | Usage |
|--------|-------------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | `psnr` | Image quality (dB, higher=better) |
| **SSIM** | Structural Similarity Index | `ssim` | Perceptual similarity (0-1, higher=better) |
| **LPIPS** | Learned Perceptual Image Patch Similarity | `lpips` | Deep perceptual distance (lower=better) |

### 🎵 Lip-Sync Error (LSE) Metrics
| Metric | Description | Output Keys | Usage |
|--------|-------------|-------------|-------|
| **LSE Distance** | Lip-sync synchronization distance | `lse_distance` | Audio-visual sync (lower=better) |
| **LSE Confidence** | Lip-sync detection confidence | `lse_confidence` | Sync reliability score |
| **LSE-D MSE** ⭐ | MSE between pred & GT LSE distance | `lse_d_mse` | Quantitative sync comparison |
| **LSE-C MSE** ⭐ | MSE between pred & GT LSE confidence | `lse_c_mse` | Confidence difference analysis |

### 🤖 Advanced AI Metrics (Optional)

#### 🎨 CLIP Similarity Suite (`--clip`) - Multi-Modal Similarity Computing

##### 📊 Video-Video Similarity (Primary Metrics)
| Metric | Description | Output Keys | Usage |
|--------|-------------|-------------|-------|
| **CLIP Similarity** | Average semantic similarity | `clip_similarity` | Primary similarity score (0-1, higher=better) |
| **CLIP Std Dev** | Similarity standard deviation | `clip_similarity_std` | Consistency of similarity across frames |
| **CLIP Min/Max** | Minimum/Maximum similarity | `clip_similarity_min`, `clip_similarity_max` | Range analysis |
| **CLIP Median** | Median similarity | `clip_similarity_median` | Robust central tendency |

##### 🔧 Complete CLIP API Functions
| Function Type | API Method | Input | Output | Purpose |
|---------------|------------|-------|--------|---------|
| **Image-Image** | `calculate_frame_similarity()` | Two image frames | Cosine similarity | Frame semantic similarity |
| **Video-Video** | `calculate_video_similarity()` | Two video files | Statistics + frame similarities | Video semantic comparison |
| **Text-Video** | `calculate_text_video_similarity()` | Text + video | Text-video matching score | Content description matching |
| **Batch Video** | `calculate_batch_video_similarity()` | Multiple video pairs | Batch similarity results | Batch processing |
| **Feature Extract** | `extract_video_features()` | Video file | CLIP feature vectors | Feature extraction |
| **Text Features** | `extract_text_features()` | Text descriptions | CLIP text features | Text encoding |
| **Image Features** | `extract_image_features()` | Images/frames | CLIP image features | Image encoding |

#### 🔥 VBench Suite (`--vbench`) - 6 Core Metrics
| Metric | Description | Output Keys | Focus Area |
|--------|-------------|-------------|------------|
| **Subject Consistency** | Subject appearance stability | `subject_consistency` | Character/object coherence |
| **Background Consistency** | Background stability | `background_consistency` | Scene consistency |
| **Motion Smoothness** | Temporal motion quality | `motion_smoothness` | Movement fluidity |
| **Dynamic Degree** | Motion intensity analysis | `dynamic_degree` | Activity level assessment |
| **Aesthetic Quality** | Visual appeal evaluation | `aesthetic_quality` | Overall visual quality |
| **Imaging Quality** | Technical image quality | `imaging_quality` | Resolution, clarity, artifacts |

#### 📊 Other Advanced Metrics
| Metric | Description | Output Keys | Flag |
|--------|-------------|-------------|------|
| **FVD** | Fréchet Video Distance | `fvd_score` | `--fvd` |
| **GIM Matching** | Graph Image Matching pixels | `gim_matching_pixels`, `gim_avg_matching` | `--gim` |

### 📈 Video Analysis Metrics
| Category | Metrics | Description |
|----------|---------|-------------|
| **Video Info** | `frame_count`, `resolution`, `fps`, `duration` | Basic video properties |
| **Face Analysis** | `face_detection_rate`, `avg_face_size`, `face_stability` | Face region analysis |
| **Image Stats** | `brightness`, `contrast`, `saturation`, `sharpness` | Image characteristics |

### 🎯 Region Selection
- **`--region face_only`**: Calculate PSNR/SSIM/LPIPS on detected face regions only
- **`--region full_image`**: Calculate metrics on entire video frames
- **Other metrics**: Always computed on full frames (LSE, VBench, etc.)

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

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guidelines](documents/CONTRIBUTING.md).

**Quick contributing steps:**
1. 🐛 [Report bugs](https://github.com/harlanhong/video-evaluation-toolkit/issues/new)
2. 💡 [Request features](https://github.com/harlanhong/video-evaluation-toolkit/discussions)  
3. 🔧 Submit code: Fork → Branch → Code → Test → Pull Request

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/harlanhong/video-evaluation-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/harlanhong/video-evaluation-toolkit/discussions)
- **Email**: fatinghong@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](documents/LICENSE) file for details.

## 🎯 Research Applications

### Ideal for evaluating:
- **Lip-sync video generation** (talking heads, dubbing)
- **Face reenactment** and expression transfer
- **Video-to-video translation** tasks
- **General video generation** quality assessment
- **Multi-modal video synthesis** evaluation

### Used in research:
- Video generation model comparison
- Lip-sync quality benchmarking  
- Cross-modal evaluation frameworks
- Real-time video assessment systems

---

⭐ **Star this repo if it helps your research!**

📖 **[Read Full Documentation](documents/README.md)** | 🚀 **[Quick Start](documents/QUICK_START.md)** | 🤝 **[Contribute](documents/CONTRIBUTING.md)**