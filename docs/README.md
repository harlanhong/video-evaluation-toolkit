# 📊 Video Evaluation Toolkit

**Language / 语言**: [English](README.md) | [中文](README_CN.md)

A comprehensive evaluation toolkit that integrates LSE calculation, VBench metrics, and other video quality indicators.

## 🎯 Features

### ✅ LSE (Lip-Sync Error) Calculation

  - **No External Audio Required**: Directly extracts audio from the video for calculation.
  - **High Precision**: Results are fully consistent with the original SyncNet script.
  - **Pure Python Implementation**: No dependency on external shell scripts.
  - **Batch Processing Support**: Can process multiple videos at once.

### ✅ Video Quality Metrics

  - **Basic Information**: Frame count, resolution, frame rate, duration.
  - **Image Statistics**: Brightness, contrast, saturation, sharpness.
  - **Face Analysis**: Face detection rate, average face size, face stability.
  - **Motion Analysis**: Motion intensity, inter-frame difference.
  - **Image Quality**: PSNR, SSIM, LPIPS for the face region (requires ground truth).

### ✅ VBench Metrics Integration

  - **Subject Consistency**: Consistency of the main subject in the video.
  - **Background Consistency**: Stability of the background content.
  - **Motion Smoothness**: The fluency of motion.
  - **Dynamic Degree**: The degree of dynamic change in the video.
  - **Aesthetic Quality**: The aesthetic score of the video.
  - **Imaging Quality**: Image quality assessment.
  - **Flexible Enablement**: VBench calculation can be selectively enabled to balance performance.

### ✅ Advanced Synchronization Metrics (New)

  - **CLIP-V Similarity**: Calculate CLIP similarity between source and target frames at the same timestamp.
  - **FVD-V Score**: Fréchet Video Distance for video generation quality assessment (from SV4D).
  - **GIM Matching**: Official GIM implementation for state-of-the-art image matching (ICLR 2024).
  - **Modular Design**: Each metric can be independently enabled for flexible evaluation.

### ✅ Official GIM Integration (New)

  - **State-of-the-Art Accuracy**: Direct integration with official GIM implementation from ICLR 2024.
  - **Multiple Model Support**: GIM_RoMa, GIM_LightGlue, GIM_DKM, GIM_LoFTR, GIM_SuperGlue.
  - **Intelligent Fallback**: Automatic fallback to simple matcher when GIM is not available.
  - **Performance Optimized**: GPU acceleration and efficient batch processing.
  - **Easy Installation**: Simple setup process with comprehensive documentation.

### ✅ Unified CLIP API (New)

  - **Comprehensive Interface**: Single API for all CLIP-based video evaluation tasks.
  - **Multiple CLIP Models**: Support for ViT-B/32, ViT-B/16, ViT-L/14, and more.
  - **Video-to-Video Similarity**: Frame-by-frame CLIP similarity calculation.
  - **Text-to-Video Similarity**: Semantic similarity between text descriptions and video content.
  - **Feature Extraction**: Extract CLIP features from video frames for analysis.
  - **Batch Processing**: Efficient processing of multiple video pairs.
  - **Model Comparison**: Easy comparison between different CLIP models.
  - **Legacy Compatibility**: Backward compatibility with existing code through wrapper classes.

## 📁 Directory Structure

```
evalutation/
├── setup.py                   # 🆕 One-click Python installation script
├── install.sh                 # 🆕 One-click Bash installation script  
├── CHANGELOG.md               # 🆕 Project change history
├── QUICK_START.md             # 🆕 Auto-generated quick start guide
├── __init__.py                # Package initialization
├── apis/                       # Unified API interfaces
│   ├── __init__.py
│   ├── clip_api.py            # Comprehensive CLIP API
│   └── README.md              # API documentation
├── calculators/                # Independent metric calculators
│   ├── __init__.py
│   ├── clip_calculator.py     # CLIP similarity calculator (legacy wrapper)
│   ├── fvd_calculator.py      # FVD score calculator
│   ├── gim_calculator.py      # GIM image matching calculator (official integration)
│   ├── lse_calculator.py      # LSE calculator
│   └── vbench_calculator.py   # VBench integration module
├── core/                       # Core calculation engine
│   ├── __init__.py
│   └── video_metrics_calculator.py # Comprehensive metrics calculator
├── examples/                   # Usage examples and demonstrations
│   ├── __init__.py
│   ├── basic_usage.py         # Basic usage example
│   ├── advanced_metrics.py    # Advanced metrics usage example
│   ├── clip_api_demo.py       # Comprehensive CLIP API demonstration
│   └── gim_demo.py            # 🆕 Official GIM integration demonstration
├── docs/                       # Documentation
│   ├── README.md              # Main documentation (this file)
│   ├── README_CN.md           # Chinese documentation
│   ├── MODELS_DOWNLOAD.md     # Model download instructions
│   └── GIM_INTEGRATION.md     # Official GIM integration guide
├── configs/                    # Configuration files
│   ├── requirements.txt       # pip dependency configuration
│   └── environment.yaml       # conda environment configuration
├── utils/                      # Utility functions and scripts
│   ├── __init__.py
│   └── install_gim.py         # 🆕 Automated GIM installation script
├── models/                     # Model files (auto-downloaded)
│   ├── syncnet_v2.model       # SyncNet model weights (~180MB)
│   └── s3fd.pth               # S3FD face detection model (~180MB)
├── syncnet_core/              # SyncNet core modules
│   ├── __init__.py
│   ├── model.py               # SyncNet model definition
│   └── detectors/             # Face detectors
│       ├── __init__.py
│       └── s3fd/
│           ├── __init__.py
│           ├── box_utils.py
│           └── nets.py
├── cache/                      # Video processing cache
├── gim/                        # 🆕 Official GIM repository (auto-cloned)
└── venv/                       # 🆕 Virtual environment (if using venv mode)
```

## 🚀 Quick Start

### 📋 Environment Requirements

**Python Version**: 3.8+ (3.9+ recommended)

**Hardware Requirements**:

  - **CPU**: Intel/AMD multi-core processor
  - **Memory**: 8GB+ RAM (16GB+ recommended)
  - **GPU**: NVIDIA GPU with CUDA 11.0+ (recommended for VBench and LSE acceleration)
  - **Storage**: 5GB+ available space (for model files and VBench cache)

**Operating System**:

  - Linux (recommended)
  - Windows 10/11
  - macOS 10.15+

### ⚙️ Install Dependencies

#### 🎯 One-Click Installation (Recommended)

The easiest way to get started with the Video Evaluation Toolkit:

##### Option A: Automated Python Setup Script
```bash
# Clone the repository
git clone https://github.com/harlanhong/video-evaluation-toolkit.git
cd video-evaluation-toolkit

# Run one-click setup (auto-detects best method)
python setup.py

# For GPU support and full features
python setup.py --gpu

# For conda environment (recommended)
python setup.py --mode conda --gpu

# Skip model downloads for faster setup
python setup.py --skip-models

# Force clean installation
python setup.py --force
```

##### Option B: Bash Installation Script (Linux/macOS)
```bash
# Clone the repository
git clone https://github.com/harlanhong/video-evaluation-toolkit.git
cd video-evaluation-toolkit

# Run bash installer
bash install.sh

# With GPU support
bash install.sh --gpu

# Force clean installation
bash install.sh --force --gpu

# Skip model downloads
bash install.sh --skip-models
```

**One-Click Installation Features:**
- 🔧 **Automatic environment setup** (conda/venv detection)
- 📦 **Complete dependency installation** (including GIM)
- 🎭 **Model download** (SyncNet, S3FD checkpoints)
- ✅ **Installation verification** and testing
- 📚 **Quick start guide** generation
- 🎯 **GPU support** detection and setup

#### Manual Installation Methods

#### Method 1: Using VBench Environment (Recommended)

If you already have a VBench environment, you can use it directly:

```bash
# Activate VBench environment
conda activate vbench

# Install additional dependencies
pip install lpips python_speech_features scenedetect

# Verify installation
cd evaluation
python -c "from metrics_calculator import VideoMetricsCalculator; print('✅ Installation successful!')"
```

#### Method 2: Using conda Environment Configuration

```bash
# Use the pre-configured environment.yaml file
cd evaluation
conda env create -f environment.yaml

# Activate the environment
conda activate video-evaluation

# Verify installation
python -c "from metrics_calculator import VideoMetricsCalculator; print('✅ Installation successful!')"
```

#### Method 3: Using pip Installation

```bash
# Clone or navigate to the project directory
cd evaluation

# Install all dependencies
pip install -r requirements.txt

# If you have an NVIDIA GPU, using the CUDA version is recommended
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # for CUDA 12.1
```

#### Method 4: Installation with Official GIM Integration (Recommended)

```bash
# Navigate to project directory
cd evalutation

# Create conda environment with all dependencies
conda env create -f configs/environment.yaml
conda activate video-evaluation

# Install official GIM for state-of-the-art image matching
git clone https://github.com/xuelunshen/gim.git
cd gim
pip install -e .
cd ..

# Verify installation including GIM
python -c "
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator
from evalutation.calculators.gim_calculator import GIMMatchingCalculator
calc = GIMMatchingCalculator()
print(f'✅ Installation successful with GIM: {calc.get_model_info()[\"gim_available\"]}')
"
```

For detailed GIM installation and troubleshooting, see: [`docs/GIM_INTEGRATION.md`](docs/GIM_INTEGRATION.md)

### 🔧 Dependency Details

For a complete list of dependencies, please refer to:

  - **pip users**: [`requirements.txt`](https://www.google.com/search?q=requirements.txt)
  - **conda users**: [`environment.yaml`](https://www.google.com/search?q=environment.yaml)

#### Main Dependencies

| Package Name | Version Requirement | Purpose |
|---|---|---|
| `torch` | ≥2.0.0 | Deep learning framework, SyncNet and VBench model inference |
| `torchvision` | ≥0.15.0 | Vision model support |
| `opencv-python` | ≥4.5.0 | Image/video processing, face detection |
| `numpy` | ≥1.21.0 | Numerical computation |
| `scipy` | ≥1.8.0 | Scientific computing, signal processing |
| `scikit-image` | ≥0.19.0 | PSNR/SSIM image quality metrics |
| `lpips` | ≥0.1.4 | Perceptual image quality metric |
| `python-speech-features` | ≥0.6.0 | MFCC audio feature extraction |
| `librosa` | ≥0.9.0 | Audio processing and analysis |
| `scenedetect` | ≥0.6.0 | Video scene detection |
| `ffmpeg-python` | ≥0.2.0 | Video format conversion |
| `vbench` | latest | VBench video generation quality assessment |
| `mediapipe` | ≥0.10.0 | Modern face detection (recommended) |
| `ultralytics` | ≥8.0.0 | YOLOv8 face detection (optional) |
| `tqdm` | ≥4.62.0 | Progress bar display |

#### System Requirements

  - **FFmpeg**: Requires FFmpeg to be installed on the system for video processing.
    ```bash
    # Ubuntu/Debian
    sudo apt install ffmpeg

    # macOS (using Homebrew)
    brew install ffmpeg

    # Windows
    # Download and install from https://ffmpeg.org/download.html
    ```

### ✅ Verify Installation

After installation, it is recommended to run the verification script to ensure all dependencies are installed correctly:

```bash
cd evaluation
python verify_installation.py
```

## 💻 Usage

### 1\. Python API Usage

#### Basic Metrics Calculation (Fast Mode)

```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

# Create a calculator in fast mode (without VBench)
calculator = VideoMetricsCalculator(enable_vbench=False)

# Calculate metrics for a single video
metrics = calculator.calculate_video_metrics("video.mp4")

print(f"Frame count: {metrics['frame_count']}")
print(f"LSE score: {metrics['lse_distance']}")
print(f"Face detection rate: {metrics['face_detection_rate']}")
```

#### Full Metrics Calculation (with VBench)

```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

# Create a calculator in full mode (with VBench)
calculator = VideoMetricsCalculator(enable_vbench=True)

try:
    # Calculate metrics for a single video (including 6 core VBench metrics)
    metrics = calculator.calculate_video_metrics("video.mp4")
    
    # Basic metrics
    print(f"Frame count: {metrics['frame_count']}")
    print(f"LSE score: {metrics['lse_distance']}")
    
    # VBench metrics
    print(f"Subject Consistency: {metrics['subject_consistency']}")
    print(f"Background Consistency: {metrics['background_consistency']}")
    print(f"Motion Smoothness: {metrics['motion_smoothness']}")
    print(f"Dynamic Degree: {metrics['dynamic_degree']}")
    print(f"Aesthetic Quality: {metrics['aesthetic_quality']}")
    print(f"Imaging Quality: {metrics['imaging_quality']}")
    
finally:
    # Clean up resources
    calculator.cleanup()
```

#### Batch Processing Videos

```python
from metrics_calculator import VideoMetricsCalculator

# Create the calculator
calculator = VideoMetricsCalculator(enable_vbench=True)

try:
    # Batch calculate metrics
    results = calculator.calculate_batch_metrics(
        pred_dir="/path/to/videos",
        gt_dir="/path/to/ground_truth",  # optional
        pattern="*.mp4"
    )
    
    # Save results
    calculator.save_results(results, "results.json")
    
    # Print statistics
    calculator.print_summary_stats(results)
    
finally:
    calculator.cleanup()
```

#### CLIP API Usage (New Unified Interface)

```python
from evalutation.apis.clip_api import CLIPVideoAPI

# Initialize CLIP API
clip_api = CLIPVideoAPI(model_name="ViT-B/32", device="cuda")

# 1. Video-to-Video Similarity
similarity_result = clip_api.calculate_video_similarity(
    source_path="video1.mp4",
    target_path="video2.mp4",
    max_frames=50
)
print(f"CLIP Similarity: {similarity_result['clip_similarity']:.4f}")

# 2. Text-to-Video Similarity
text_result = clip_api.calculate_text_video_similarity(
    video_path="video.mp4",
    text_queries=["a person walking", "outdoor scene", "dancing"],
    max_frames=30
)
print("Text similarities:", text_result['similarities'])

# 3. Feature Extraction
features_result = clip_api.extract_video_features("video.mp4", max_frames=20)
print(f"Features shape: {features_result['features'].shape}")

# 4. Batch Processing
video_pairs = [("vid1.mp4", "ref1.mp4"), ("vid2.mp4", "ref2.mp4")]
batch_results = clip_api.calculate_batch_video_similarity(video_pairs)

# 5. Save results
clip_api.save_results(similarity_result, "clip_results.json")
```

#### Advanced Metrics with CLIP Integration

```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

# Create calculator with advanced metrics enabled
calculator = VideoMetricsCalculator(
    enable_vbench=True,
    enable_clip_similarity=True,
    enable_gim_matching=True
)

try:
    # Calculate all metrics including CLIP similarity
    metrics = calculator.calculate_video_metrics(
        pred_path="generated_video.mp4",
        gt_path="reference_video.mp4"
    )
    
    # Display results
    print(f"VBench - Subject Consistency: {metrics['subject_consistency']}")
    print(f"CLIP Similarity: {metrics['clip_similarity']:.4f}")
    print(f"GIM Matching Pixels: {metrics['gim_matching_pixels']}")
    
finally:
    calculator.cleanup()
```

#### Official GIM Matching (New)

```python
from evalutation.calculators.gim_calculator import GIMMatchingCalculator

# Initialize with official GIM implementation
gim_calculator = GIMMatchingCalculator(
    model_name="gim_roma",        # Highest accuracy model
    device="cuda",
    confidence_threshold=0.5
)

# Calculate matching for video pair
results = gim_calculator.calculate_video_matching(
    source_path="source_video.mp4",
    target_path="target_video.mp4",
    max_frames=50,
    verbose=True
)

# Display detailed results
print(f"Model: {results['model_name']}")
print(f"Total matching pixels: {results['total_matching_pixels']}")
print(f"Average per frame: {results['avg_matching_pixels']:.2f}")
print(f"Confidence threshold: {results['confidence_threshold']}")

# Available models: gim_roma, gim_lightglue, gim_dkm, gim_loftr, gim_superglue
```

#### Comparison with Ground Truth

```python
# Comparison with ground truth (calculates PSNR, SSIM, LPIPS for the face region)
metrics = calculator.calculate_video_metrics(
    pred_path="prediction.mp4",
    gt_path="ground_truth.mp4"
)
```

#### Using the LSE Calculator Separately

```python
from evalutation.calculators.lse_calculator import LSECalculator

# Initialize
calculator = LSECalculator()

# Calculate for a single video
lse_distance, lse_confidence = calculator.calculate_single_video("video.mp4")
print(f"LSE Distance: {lse_distance:.4f}")
print(f"LSE Confidence: {lse_confidence:.4f}")

# Batch calculate
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = calculator.calculate_batch(video_paths)
```

### 2\. Command-Line Usage

#### Fast Mode (without VBench metrics)

```bash
cd evalutation
python -m core.video_metrics_calculator --pred_dir /path/to/videos
```

#### Full Mode (with VBench metrics)

```bash
cd evalutation  
python -m core.video_metrics_calculator --pred_dir /path/to/videos --vbench
```

#### Specify Ground Truth Directory for Comparison

```bash
python -m core.video_metrics_calculator \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --vbench \
    --output results.json
```

#### Custom Output and File Pattern

```bash
python -m core.video_metrics_calculator \
    --pred_dir /path/to/videos \
    --vbench \
    --output my_results.json \
    --pattern "*.avi"
```

#### Advanced Metrics (CLIP, FVD, GIM)

```bash
# Enable CLIP similarity calculation
python -m core.video_metrics_calculator \
    --pred_dir /path/to/videos \
    --gt_dir /path/to/ground_truth \
    --clip

# Enable GIM matching
python -m core.video_metrics_calculator \
    --pred_dir /path/to/videos \
    --gt_dir /path/to/ground_truth \
    --gim

# Enable all advanced metrics
python -m core.video_metrics_calculator \
    --pred_dir /path/to/videos \
    --gt_dir /path/to/ground_truth \
    --all_advanced --vbench
```

#### CLIP API Command-Line Usage

```bash
# Video-to-video similarity
python -m apis.clip_api \
    --task video_similarity \
    --source video1.mp4 \
    --target video2.mp4 \
    --model ViT-L/14

# Text-to-video similarity
python -m apis.clip_api \
    --task text_video \
    --source video.mp4 \
    --text "a person walking outdoors"

# Feature extraction
python -m apis.clip_api \
    --task extract_features \
    --source video.mp4 \
    --max_frames 30
```

#### Official GIM Matching

```bash
# High accuracy matching with GIM_RoMa
python -m calculators.gim_calculator \
    --source source_video.mp4 \
    --target target_video.mp4 \
    --model gim_roma \
    --threshold 0.5

# Fast matching with GIM_LightGlue
python -m calculators.gim_calculator \
    --source source_video.mp4 \
    --target target_video.mp4 \
    --model gim_lightglue \
    --threshold 0.6 \
    --max_frames 30

# Dense matching with GIM_DKM
python -m calculators.gim_calculator \
    --source source_video.mp4 \
    --target target_video.mp4 \
    --model gim_dkm \
    --output dense_matching_results.json
```

#### LSE Calculation for a Single Video

```bash
python -m calculators.lse_calculator --video /path/to/video.mp4
```

## 📊 Supported Metrics

### 🟢 Metrics That Do Not Require Ground Truth

| Metric Category | Metric Name | Description | Value Range |
|---|---|---|---|
| **Basic Info** | `frame_count` | Number of frames in the video | Positive Integer |
| | `width`, `height` | Video resolution | Positive Integer |
| | `fps` | Frames per second | Positive Number |
| | `duration_seconds` | Video duration in seconds | Positive Number |
| **Image Stats** | `mean_brightness` | Average brightness | 0-255 |
| | `mean_contrast` | Average contrast | ≥0 |
| | `mean_saturation` | Average saturation | 0-255 |
| | `sharpness_score` | Sharpness score | ≥0, higher is sharper |
| **Face Analysis** | `face_detection_rate` | Face detection rate | 0-1 |
| | `avg_face_size` | Average face size in pixels | ≥0 |
| | `face_stability` | Face position stability | 0-1, higher is more stable |
| **Motion Analysis** | `motion_intensity` | Motion intensity | ≥0 |
| | `frame_difference` | Average inter-frame difference | ≥0 |
| **Lip Sync** | `lse_distance` | LSE distance | ≥0, lower is better |
| | `lse_confidence` | LSE confidence | ≥0, higher is better |
| **VBench Metrics**| `subject_consistency` | Subject consistency | 0-1, higher is better |
| | `background_consistency`| Background consistency | 0-1, higher is better |
| | `motion_smoothness` | Motion smoothness | 0-1, higher is better |
| | `dynamic_degree` | Dynamic degree | 0-1, moderate is good |
| | `aesthetic_quality` | Aesthetic quality | 0-1, higher is better |
| | `imaging_quality` | Imaging quality | 0-1, higher is better |

### 🔴 Metrics That Require Ground Truth

| Metric Name | Description | Value Range | Notes |
|---|---|---|---|
| `face_psnr` | Peak Signal-to-Noise Ratio for the face region | ≥0, higher is better | \>25 is good |
| `face_ssim` | Structural Similarity for the face region | 0-1, higher is better | \>0.8 is good |
| `face_lpips` | Perceptual Similarity for the face region | ≥0, lower is better | \<0.2 is good |

## 📈 Output Example

### Output File Structure (with average statistics)

```json
{
  "summary": {
    "total_videos": 10,
    "successful_videos": 9,
    "average_metrics": {
      "Basic Information": {
        "frame_count": 125.3,
        "width": 960.0,
        "height": 544.0,
        "fps": 25.0,
        "duration_seconds": 5.01
      },
      "Image Statistics": {
        "mean_brightness": 118.45,
        "mean_contrast": 42.18,
        "mean_saturation": 125.67,
        "sharpness_score": 598.34
      },
      "Face Analysis": {
        "face_detection_rate": 0.95,
        "avg_face_size": 11800.0,
        "face_stability": 0.89
      },
      "Motion Analysis": {
        "motion_intensity": 0.62,
        "frame_difference": 7.85
      },
      "LSE Metrics": {
        "lse_distance": 8.45,
        "lse_confidence": 6.78
      },
      "VBench Metrics": {
        "subject_consistency": 0.845,
        "background_consistency": 0.798,
        "motion_smoothness": 0.723,
        "dynamic_degree": 0.612,
        "aesthetic_quality": 0.665,
        "imaging_quality": 0.689
      },
      "Comparison Metrics": {
        "face_psnr": 28.56,
        "face_ssim": 0.876,
        "face_lpips": 0.098
      }
    }
  },
  "individual_results": [
    {
      "video_path": "prediction_001.mp4",
      "has_ground_truth": false,
      "vbench_enabled": true,
      
      "frame_count": 129,
      "width": 960,
      "height": 544,
      "fps": 25.0,
      "duration_seconds": 5.16,
      
      "mean_brightness": 122.22,
      "mean_contrast": 45.14,
      "mean_saturation": 128.5,
      "sharpness_score": 615.49,
      
      "face_detection_rate": 1.0,
      "avg_face_size": 12500.0,
      "face_stability": 0.95,
      
      "motion_intensity": 0.58,
      "frame_difference": 8.23,
      
      "lse_distance": 9.2235,
      "lse_confidence": 6.5694,
      
      "subject_consistency": 0.891,
      "background_consistency": 0.802,
      "motion_smoothness": 0.756,
      "dynamic_degree": 0.634,
      "aesthetic_quality": 0.678,
      "imaging_quality": 0.712,
      
      "face_psnr": 30.12,
      "face_ssim": 0.892,
      "face_lpips": 0.089,
      
      "error": null
    }
    // ... more video results ...
  ]
}
```

## 🔧 Technical Details

### LSE Calculation Principle

1.  **Video Preprocessing**: Extract frames and audio, convert formats.
2.  **Face Detection and Tracking**: Use the S3FD detector for face detection and tracking.
3.  **Face Video Cropping**: Extract and crop the face region.
4.  **Feature Extraction**: Use SyncNet to extract video and audio features.
5.  **LSE Calculation**: Calculate the distance and confidence between audio-visual features.

### VBench Integration Principle

1.  **Direct Integration**: Directly use the official VBench library to ensure result consistency.
2.  **6 Core Metrics**: Subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality.
3.  **Optional Enablement**: Control whether to calculate VBench metrics via the `enable_vbench` parameter.
4.  **Resource Management**: Automatically manage VBench temporary files and computing resources.

### Modern Face Detection (New)

1.  **Multiple Detector Support**: Automatically selects the best available face detector.
2.  **Priority Order**: MediaPipe \> YOLOv8 \> OpenCV DNN \> Haar Cascade.
3.  **Performance Improvement**: 3-10x faster than traditional Haar cascade, with significant accuracy improvement.
4.  **Intelligent Fallback**: Automatically falls back to an available method if a higher-priority detector is unavailable.

### Model Files

  - **SyncNet Model** (`syncnet_v2.model`, 52MB): For audio-visual feature extraction.
  - **S3FD Model** (`sfd_face.pth`, 86MB): For face detection.
  - **VBench Models**: Automatically downloaded and managed, used for the 6 core metric calculations.

### Device Support

  - **CUDA**: Supports GPU acceleration (recommended).
  - **CPU**: Supports CPU computation (slower).

## 🚨 Troubleshooting

### VBench Related Issues

1.  **VBench initialization fails**: Ensure the VBench library is installed correctly and the version is compatible.
2.  **CUDA out of memory**: Try using CPU mode or reducing `batch_size`.
3.  **Network connection issues**: VBench needs to download models on its first run, ensure a stable network connection.

### LSE Calculation Fails

1.  **Check video format**: Ensure the video contains an audio track.
2.  **Check model files**: Ensure the model files exist and are not corrupted.
3.  **Check dependencies**: Ensure all required packages are installed.

### Face Detection Fails

1.  **Video quality**: Ensure there are clear, visible faces in the video.
2.  **Resolution**: Very low resolution may affect face detection.
3.  **Lighting conditions**: Extreme darkness or brightness may affect detection.

### Performance Optimization

1.  **Choose a mode**: Use fast mode (`enable_vbench=False`) for quicker results.
2.  **Use GPU**: Enabling CUDA acceleration can significantly improve computation speed.
3.  **Modern face detection**: Installing MediaPipe can provide a 3-10x speedup in face detection.
4.  **Batch processing**: Batch calculation is more efficient than processing one by one.
5.  **Resource cleanup**: Use `calculator.cleanup()` to release VBench resources.

## 📚 API Reference

### VideoMetricsCalculator

#### Initialization Parameters

  - `device` (str): Computing device ("cuda" or "cpu").
  - `enable_vbench` (bool): Whether to enable VBench metrics calculation, default is False.

#### Main Methods

  - `calculate_video_metrics(pred_path, gt_path=None)`: Calculate metrics for a single video.
  - `calculate_batch_metrics(pred_dir, gt_dir=None, pattern="*.mp4")`: Batch calculate metrics.
  - `save_results(results, output_path)`: Save results to a JSON file.
  - `print_summary_stats(results)`: Print summary statistics.
  - `cleanup()`: Clean up VBench resources.

### LSECalculator

#### Initialization Parameters

  - `model_path` (str, optional): Path to the SyncNet model.
  - `device` (str): Computing device ("cuda" or "cpu").
  - `batch_size` (int): Batch size, default is 20.
  - `vshift` (int): Video shift range, default is 15.

#### Main Methods

  - `calculate_single_video(video_path, verbose=True)`: Calculate LSE for a single video.
  - `calculate_batch(video_paths, verbose=True)`: Batch calculate LSE.

## 🎯 Best Practices

1.  **Balancing Performance and Accuracy**:

      - For quick evaluation: Use `enable_vbench=False`.
      - For complete evaluation: Use `enable_vbench=True`.

2.  **File Organization**:

      - Use the same filenames for prediction videos and ground truth videos.
      - Use a meaningful directory structure.

3.  **Performance Optimization**:

      - Prioritize using a GPU for computation.
      - For large batch tasks, consider processing in smaller chunks.
      - Call `cleanup()` promptly to release VBench resources.

4.  **Result Analysis**:

      - Use the summary statistics function to analyze overall performance.
      - Focus on the distribution of LSE, face quality, and VBench metrics.
      - VBench metrics provide a more comprehensive assessment of video generation quality.

5.  **Error Handling**:

      - Check the `error` field in the output.
      - For failed videos, check video quality and format.
      - VBench calculation will fail gracefully, allowing other metrics to still be computed.

## 🆕 Changelog

### v2.0.0 (Current Version)

  - ✅ **Added VBench Integration**: Supports 6 core VBench metrics.
  - ✅ **Flexible Enablement Mechanism**: VBench can be selectively enabled to balance performance.
  - ✅ **Unified API Design**: VBench metrics are integrated into a single metrics interface.
  - ✅ **Optimized Resource Management**: Automatically manages VBench temporary files and resources.
  - ✅ **Enhanced Error Handling**: VBench failure does not affect the calculation of other metrics.
  - ✅ **Updated Dependency Management**: Optimized `requirements.txt` and `environment.yaml`.

### v1.0.0

  - ✅ Integrated LSE calculator and video metrics calculator.
  - ✅ Refactored directory structure and optimized module organization.
  - ✅ Unified API interface for simplified usage.
  - ✅ Added detailed documentation and usage examples.
  - ✅ Verified consistency with the original SyncNet script.

## 📄 License

This project is based on the original SyncNet, VBench, and related open-source projects, and follows their respective open-source licenses.

## 👨‍💻 Author

**Fating Hong** / fatinghong@gmail.com

The VBench integrated version of the Video Evaluation Toolkit is developed and maintained by Fating Hong.

## 🤝 Contributing

Issues and Pull Requests to improve this toolkit are welcome\!

-----

**🎉 You can now use this powerful video evaluation toolkit (with VBench metrics) to comprehensively assess your video generation results\!**