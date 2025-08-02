# 📚 Video Evaluation Toolkit v2.1.0 - Quick Start Guide

*Get up and running in 5 minutes!*

## 🎯 Two-Command Setup

```bash
# 1. One-click installation
python setup.py --gpu

# 2. Start evaluating  
python core/video_metrics_calculator.py --pred_dir /your/videos --region face_only
```

## ✨ What's New in v2.1.0

- **🔧 Fixed face metrics calculation**: PSNR now shows realistic values (27.7dB vs previous 9.6dB)
- **📊 Complete statistics**: All metrics properly included in JSON summaries  
- **🎯 Region selection**: Choose between `face_only` or `full_image` metrics
- **⚡ Enhanced compatibility**: Better support across different video resolutions

## 🚀 Quick Examples

### 1. Face Quality Assessment (Recommended)

```python
from core.video_metrics_calculator import VideoMetricsCalculator

# Focus on face region (best for lip-sync evaluation)
calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="generated_video.mp4",
    gt_path="reference_video.mp4",
    region="face_only"  # 🆕 New parameter
)

# Quality metrics (now fixed and realistic!)
print(f"Face PSNR: {metrics['psnr']:.2f} dB")      # ~25-30 dB is good
print(f"Face SSIM: {metrics['ssim']:.3f}")         # >0.8 is excellent  
print(f"Face LPIPS: {metrics['lpips']:.3f}")       # <0.1 is excellent
print(f"LSE Score: {metrics['lse_distance']:.2f}") # <10 is good
```

### 2. Full Image Quality Assessment

```python
# Evaluate entire image quality
metrics = calculator.calculate_video_metrics(
    pred_path="video.mp4", 
    gt_path="reference.mp4",
    region="full_image"  # Full image comparison
)
```

### 3. Batch Processing

```python
# Process multiple videos at once
results = calculator.calculate_batch_metrics(
    pred_dir="/path/to/generated_videos/",
    gt_dir="/path/to/reference_videos/", 
    region="face_only",
    output="results.json"
)
```

### 4. Command Line Interface

```bash
# Face-only metrics (recommended for lip-sync)
python core/video_metrics_calculator.py \
    --pred_dir /generated/videos \
    --gt_dir /reference/videos \
    --region face_only \
    --output face_results.json

# Full image metrics  
python core/video_metrics_calculator.py \
    --pred_dir /generated/videos \
    --gt_dir /reference/videos \
    --region full_image \
    --output full_results.json

# All advanced metrics
python core/video_metrics_calculator.py \
    --pred_dir /videos \
    --gt_dir /reference \
    --region face_only \
    --vbench --clip --gim \
    --output comprehensive_results.json
```

## 📊 Understanding Results

### Typical Good Values (v2.1.0 Fixed)

| Metric | Face Region | Full Image | Interpretation |
|--------|-------------|------------|----------------|
| **PSNR** | 25-35 dB | 30-40 dB | Higher = Better quality |
| **SSIM** | 0.8-0.95 | 0.85-0.98 | Closer to 1 = More similar |
| **LPIPS** | 0.02-0.15 | 0.01-0.10 | Lower = More similar |
| **LSE Distance** | 5-12 | - | Lower = Better lip-sync |

### JSON Output Structure

```json
{
  "summary": {
    "total_videos": 20,
    "success_rate": "100.00%",
    "average_metrics": {
      "Comparison Metrics": {
        "psnr": 27.72,     // ✅ Now included in summary!
        "ssim": 0.871,     // ✅ Realistic values
        "lpips": 0.043     // ✅ Fixed calculation
      },
      "LSE Metrics": {
        "lse_distance": 8.26,
        "lse_confidence": 6.52
      }
    }
  }
}
```

## 🛠️ Installation Verification

```bash
# Quick system check
python -c "
from core.video_metrics_calculator import VideoMetricsCalculator
print('✅ Video Evaluation Toolkit v2.1.0 Ready!')
calc = VideoMetricsCalculator()
print(f'✅ Face detection: {calc.face_detection_method}')
"
```

## 📁 Available Examples

```bash
# Basic usage
python examples/basic_usage.py

# Face vs full image comparison  
python examples/face_vs_full_image_demo.py

# CLIP API demonstration
python examples/clip_api_demo.py

# GIM matching examples
python examples/gim_demo.py
```

## 🎨 Advanced Features

### Face Detection Methods (Automatic Fallback)

1. **MediaPipe** (95% accuracy, 468 landmarks)
2. **YOLOv8** (Fast, good accuracy)  
3. **OpenCV DNN** (Moderate accuracy)
4. **Haar Cascade** (Basic fallback)

### VBench Integration

```python
# Enable VBench metrics (requires separate installation)
calculator = VideoMetricsCalculator(enable_vbench=True)
metrics = calculator.calculate_video_metrics("video.mp4")

print(f"Subject Consistency: {metrics['subject_consistency']}")
print(f"Motion Smoothness: {metrics['motion_smoothness']}")
```

## 🚨 Quick Troubleshooting

### Common Issues & Solutions

```bash
# Face detection not working
pip install mediapipe ultralytics

# GPU memory issues  
python core/video_metrics_calculator.py --device cpu

# Model download issues
python setup.py --download-models

# Import path issues
export PYTHONPATH=.
```

### Performance Tips

- **Use `--region face_only`** for lip-sync evaluation
- **Use `--region full_image`** for overall quality assessment  
- **Enable GPU** with `--device cuda` for faster processing
- **Process in batches** rather than single videos

## 📚 Documentation Links

- **[Main Documentation](docs/README.md)** - Complete feature overview
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute  
- **[MediaPipe Integration](docs/MEDIAPIPE_INTEGRATION.md)** - Advanced face detection
- **[GIM Integration](docs/GIM_INTEGRATION.md)** - Image matching setup
- **[API Reference](apis/README.md)** - Detailed API docs

## ⚡ Performance Benchmarks

### v2.1.0 Improvements

- **3x faster** face detection with MediaPipe
- **Fixed metric calculation** ensuring realistic values
- **Better memory usage** for large video batches
- **Cross-resolution support** with automatic GT frame resizing

### Typical Processing Speed

- **Face metrics**: ~2-5 seconds per video (720p, 5s)
- **Full metrics**: ~10-20 seconds per video (with VBench)
- **Batch processing**: ~80% efficiency improvement

---

## 🎉 You're Ready!

**Start with this command:**
```bash
python core/video_metrics_calculator.py \
    --pred_dir /your/videos \
    --gt_dir /your/references \
    --region face_only \
    --output results.json
```

**Questions?** Check [GitHub Issues](https://github.com/harlanhong/video-evaluation-toolkit/issues) or email: fatinghong@gmail.com

**Happy evaluating!** 🎬✨