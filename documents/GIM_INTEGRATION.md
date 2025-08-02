# 🔍 GIM (Graph Image Matching) Integration Guide

This document provides detailed instructions for integrating the official GIM implementation into your video evaluation workflow.

## 📖 About GIM

**GIM: Learning Generalizable Image Matcher From Internet Videos (ICLR 2024 Spotlight)**

GIM is a state-of-the-art image matching method that achieves superior performance across various scenarios. The official implementation supports multiple model variants:

- **GIM_RoMa**: Highest accuracy (recommended)
- **GIM_LightGlue**: Fast and good accuracy
- **GIM_DKM**: Dense matching
- **GIM_LoFTR**: Semi-dense matching  
- **GIM_SuperGlue**: Sparse matching

**Paper**: [GIM: Learning Generalizable Image Matcher From Internet Videos](https://xuelunshen.com/gim)  
**Repository**: [https://github.com/xuelunshen/gim](https://github.com/xuelunshen/gim)

## 🚀 Installation

### Method 1: Quick Installation (Recommended)

```bash
# Navigate to your project directory
cd /path/to/your/evalutation

# Clone GIM repository
git clone https://github.com/xuelunshen/gim.git

# Install GIM
cd gim
pip install -e .
cd ..

# Verify installation
python -c "import gim; print('GIM installed successfully!')"
```

### Method 2: Side-by-Side Installation

```bash
# Install GIM adjacent to evalutation project
cd /path/to/parent/directory
git clone https://github.com/xuelunshen/gim.git
cd gim
pip install -e .
```

### Method 3: System-wide Installation

```bash
# Install in a common location
sudo git clone https://github.com/xuelunshen/gim.git /opt/gim
cd /opt/gim
sudo pip install -e .
```

## 📦 Dependencies

GIM requires the following additional dependencies:

```bash
# Core dependencies (usually auto-installed)
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib
pip install kornia
pip install einops
pip install timm

# For specific models
pip install lightglue  # For GIM_LightGlue
# Other model-specific dependencies are included in GIM
```

## 💻 Usage

### Basic Usage

```python
from evalutation.calculators.gim_calculator import GIMMatchingCalculator

# Initialize with official GIM implementation
calculator = GIMMatchingCalculator(
    model_name="gim_roma",  # or "gim_lightglue", "gim_dkm", etc.
    device="cuda",
    confidence_threshold=0.5
)

# Calculate matching for video pair
results = calculator.calculate_video_matching(
    source_path="source_video.mp4",
    target_path="target_video.mp4",
    max_frames=50,
    verbose=True
)

print(f"Total matching pixels: {results['total_matching_pixels']}")
print(f"Average per frame: {results['avg_matching_pixels']:.2f}")
```

### Command Line Usage

```bash
# Basic matching with GIM_RoMa
python -m calculators.gim_calculator \
    --source source_video.mp4 \
    --target target_video.mp4 \
    --model gim_roma \
    --output gim_results.json

# Fast matching with GIM_LightGlue
python -m calculators.gim_calculator \
    --source source_video.mp4 \
    --target target_video.mp4 \
    --model gim_lightglue \
    --threshold 0.3 \
    --max_frames 30
```

### Integration with Core Calculator

```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

# Enable GIM matching in comprehensive evaluation
calculator = VideoMetricsCalculator(
    enable_vbench=True,
    enable_clip_similarity=True,
    enable_gim_matching=True,  # Uses official GIM if available
    device="cuda"
)

# Calculate all metrics including GIM
metrics = calculator.calculate_video_metrics(
    pred_path="generated_video.mp4",
    gt_path="reference_video.mp4"
)

print(f"GIM matching pixels: {metrics['gim_matching_pixels']}")
```

## 🎯 Model Selection Guide

### Performance Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **GIM_RoMa** | Slow | **Highest** | Research, high-quality evaluation |
| **GIM_LightGlue** | **Fast** | High | Production, real-time applications |
| **GIM_DKM** | Medium | High | Dense correspondence needed |
| **GIM_LoFTR** | Medium | Medium | Semi-dense matching |
| **GIM_SuperGlue** | Fast | Medium | Sparse keypoint matching |

### Recommendations

- **Research/Evaluation**: Use `gim_roma` for highest accuracy
- **Production/Speed**: Use `gim_lightglue` for best speed/accuracy balance
- **Dense Matching**: Use `gim_dkm` when you need dense correspondence
- **Legacy Support**: Use `gim_superglue` for compatibility with SuperGlue workflows

## 🔧 Configuration

### Model Configuration

```python
# High accuracy configuration (slow)
calculator = GIMMatchingCalculator(
    model_name="gim_roma",
    confidence_threshold=0.2,  # Lower threshold for more matches
    max_keypoints=4096,        # More keypoints
    device="cuda"
)

# Fast configuration
calculator = GIMMatchingCalculator(
    model_name="gim_lightglue",
    confidence_threshold=0.5,  # Higher threshold for speed
    max_keypoints=1024,        # Fewer keypoints
    device="cuda"
)
```

### Environment Variables

```bash
# Set GIM installation path (if needed)
export GIM_PATH=/path/to/gim

# Enable GIM debug mode
export GIM_DEBUG=1
```

## 🔄 Fallback Mechanism

The GIM calculator includes an intelligent fallback system:

1. **Primary**: Official GIM implementation (if available)
2. **Fallback**: Simple ORB-based matcher with similar API

```python
# Check if official GIM is available
calculator = GIMMatchingCalculator()
model_info = calculator.get_model_info()

if model_info['gim_available']:
    print(f"✅ Using official GIM: {model_info['model_name']}")
else:
    print("⚠️ Using fallback matcher")
```

## 📊 Performance Benchmarks

Based on the [official GIM paper](https://github.com/xuelunshen/gim):

### ZEB Benchmark Results (MeanAUC@5°)

| Method | Accuracy | Note |
|--------|----------|------|
| **GIM_RoMa** | **53.3%** | Best overall |
| **GIM_DKM** | **51.2%** | Dense matching |
| **GIM_LightGlue** | **38.3%** | Speed/accuracy balance |
| **GIM_LoFTR** | **39.1%** | Semi-dense |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Error: `ModuleNotFoundError: No module named 'gim'`

**Solution**: 
```bash
# Install GIM properly
git clone https://github.com/xuelunshen/gim.git
cd gim
pip install -e .
```

#### 2. CUDA Out of Memory

**Solution**:
```python
# Reduce max_keypoints or use CPU
calculator = GIMMatchingCalculator(
    model_name="gim_lightglue",  # Use lighter model
    max_keypoints=512,           # Reduce keypoints
    device="cpu"                 # Use CPU if needed
)
```

#### 3. Slow Performance

**Solution**:
```python
# Use faster model and settings
calculator = GIMMatchingCalculator(
    model_name="gim_lightglue",  # Fastest accurate model
    confidence_threshold=0.6,    # Higher threshold
    max_keypoints=1024          # Moderate keypoints
)
```

#### 4. Model Not Found

**Solution**:
```python
# Check available models
print(GIMMatchingCalculator.AVAILABLE_MODELS)

# Use a supported model
calculator = GIMMatchingCalculator(model_name="gim_roma")
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
results = calculator.calculate_video_matching(
    source_path="video1.mp4",
    target_path="video2.mp4",
    verbose=True
)
```

## 📚 Advanced Usage

### Custom GIM Configuration

```python
# Advanced configuration example
calculator = GIMMatchingCalculator(
    model_name="gim_roma",
    device="cuda",
    confidence_threshold=0.3,
    max_keypoints=2048
)

# Get detailed model information
model_info = calculator.get_model_info()
print("Model configuration:", model_info)
```

### Batch Processing

```python
# Process multiple video pairs
video_pairs = [
    ("video1.mp4", "ref1.mp4"),
    ("video2.mp4", "ref2.mp4"),
    ("video3.mp4", "ref3.mp4")
]

for source, target in video_pairs:
    results = calculator.calculate_video_matching(source, target)
    print(f"{source} vs {target}: {results['total_matching_pixels']} matches")
```

### Result Analysis

```python
# Detailed result analysis
results = calculator.calculate_video_matching("vid1.mp4", "vid2.mp4")

if results['error'] is None:
    print(f"Model: {results['model_name']}")
    print(f"Total matches: {results['total_matching_pixels']}")
    print(f"Average per frame: {results['avg_matching_pixels']:.2f}")
    print(f"Standard deviation: {results['std_matching_pixels']:.2f}")
    
    # Analyze per-frame results
    frame_results = results['frame_results']
    for frame_result in frame_results[:5]:  # First 5 frames
        print(f"Frame {frame_result['frame_index']}: {frame_result['matching_pixels']} matches")
```

## 🔗 Related Resources

- **Official GIM Paper**: [ICLR 2024](https://openreview.net/forum?id=DkqPeXpVQ5)
- **GIM Repository**: [https://github.com/xuelunshen/gim](https://github.com/xuelunshen/gim)
- **Demo and Examples**: See `gim/demo.py` in the official repository
- **Pre-trained Models**: Available in `gim/weights/` directory

## 💡 Tips for Best Results

1. **Use GPU**: GIM performance significantly improves with CUDA
2. **Choose Right Model**: Balance speed vs. accuracy based on your needs
3. **Adjust Thresholds**: Lower confidence thresholds give more matches
4. **Limit Frames**: Use `max_frames` parameter for faster processing
5. **Monitor Memory**: Large videos may require batch processing

## 📄 Citation

If you use GIM in your research, please cite the original paper:

```bibtex
@inproceedings{
xuelun2024gim,
title={GIM: Learning Generalizable Image Matcher From Internet Videos},
author={Xuelun Shen and Zhipeng Cai and Wei Yin and Matthias Müller and Zijun Li and Kaixuan Wang and Xiaozhi Chen and Cheng Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```