# 📚 APIs Module Documentation v2.1.0

This module provides unified API interfaces for video evaluation tasks with enhanced face detection and region-specific metrics.

## 🎯 Overview

The APIs module contains high-level, unified interfaces that simplify complex video evaluation tasks. These APIs are designed to be user-friendly, efficient, and comprehensive.

## 📦 Available APIs

### CLIPVideoAPI

A comprehensive API for CLIP-based video evaluation tasks.

**Features:**
- Video-to-video similarity calculation
- Text-to-video semantic similarity
- Video feature extraction
- Batch processing capabilities
- Multi-model support (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
- Model comparison utilities

**Usage Example:**

```python
from evalutation.apis.clip_api import CLIPVideoAPI

# Initialize API
clip_api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")

# Video similarity
similarity_result = clip_api.calculate_video_similarity(
    source_path="video1.mp4",
    target_path="video2.mp4",
    max_frames=50
)

# Text-video similarity
text_result = clip_api.calculate_text_video_similarity(
    video_path="video.mp4",
    text_queries=["a person walking", "outdoor scene"],
    max_frames=30
)

# Feature extraction
features_result = clip_api.extract_video_features("video.mp4", max_frames=20)

# Batch processing
video_pairs = [("vid1.mp4", "ref1.mp4"), ("vid2.mp4", "ref2.mp4")]
batch_results = clip_api.calculate_batch_video_similarity(video_pairs)
```

**Supported CLIP Models:**
- RN50, RN101, RN50x4, RN50x16, RN50x64
- ViT-B/32, ViT-B/16 
- ViT-L/14, ViT-L/14@336px

**Command Line Usage:**

```bash
# Video similarity
python -m evalutation.apis.clip_api \
    --task video_similarity \
    --source video1.mp4 \
    --target video2.mp4 \
    --model ViT-L/14

# Text-video similarity
python -m evalutation.apis.clip_api \
    --task text_video \
    --source video.mp4 \
    --text "a person walking outdoors"

# Feature extraction
python -m evalutation.apis.clip_api \
    --task extract_features \
    --source video.mp4 \
    --max_frames 30
```

## 🚀 Quick Start

1. **Import the API:**
   ```python
   from evalutation.apis.clip_api import CLIPVideoAPI
   ```

2. **Initialize with your preferred model:**
   ```python
   api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")
   ```

3. **Use the API methods for your specific task**

## 📊 Performance Tips

- **GPU Acceleration**: Use `device="cuda"` for faster processing
- **Batch Processing**: Use batch methods for multiple videos
- **Frame Sampling**: Limit `max_frames` for faster processing
- **Model Selection**: Choose appropriate CLIP model based on accuracy vs. speed requirements

## 🔧 Integration

The APIs are designed to integrate seamlessly with the core video metrics calculator:

```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator(
    enable_clip_similarity=True,  # Uses CLIPVideoAPI internally
    enable_vbench=True,
    enable_gim_matching=True
)

# NEW in v2.1.0: Region-specific metrics
metrics = calculator.calculate_video_metrics(
    pred_path="video.mp4",
    gt_path="reference.mp4", 
    region="face_only"  # 🆕 Choose between "face_only" or "full_image"
)
```

## 🆕 What's New in v2.1.0

### Enhanced Face Detection API
- **Multiple fallback detectors**: MediaPipe → YOLOv8 → OpenCV DNN → Haar Cascade
- **Region-specific calculations**: Choose between face-only or full-image metrics
- **Fixed metric calculation**: Realistic PSNR/SSIM/LPIPS values (PSNR: 9.6→27.7 dB)
- **Cross-resolution support**: Automatic GT frame resizing for accurate face alignment

### Updated Method Signatures

```python
# Enhanced calculate_frame_metrics with region parameter
def calculate_frame_metrics(
    self, 
    pred_frame: np.ndarray, 
    gt_frame: np.ndarray,
    region: str = "face_only"  # 🆕 New parameter
) -> Dict[str, float]:
    """
    Calculate image quality metrics for specified region.
    
    Args:
        pred_frame: Predicted frame (H, W, 3) RGB
        gt_frame: Ground truth frame (H, W, 3) RGB  
        region: "face_only" or "full_image"
        
    Returns:
        Dictionary with psnr, ssim, lpips scores
    """
```

## 📖 More Examples

See the `examples/` directory for comprehensive usage examples:
- `clip_api_demo.py`: Detailed CLIP API demonstration
- `advanced_metrics.py`: Integration with other metrics
- `basic_usage.py`: Simple usage patterns