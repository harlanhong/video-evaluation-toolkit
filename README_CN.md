# 📊 视频评估工具包

**Language / 语言**: [English](README.md) | [中文](README_CN.md)

集成了LSE计算、VBench指标和其他视频质量指标的综合评估工具包。

## 🎯 功能特性

### ✅ LSE (Lip-Sync Error) 计算
- **无需外部音频**：直接从视频中提取音频进行计算
- **高精度**：与原始SyncNet脚本结果完全一致
- **纯Python实现**：无需依赖外部shell脚本
- **支持批量处理**：可一次处理多个视频

### ✅ 视频质量指标
- **基本信息**：帧数、分辨率、帧率、时长
- **图像统计**：亮度、对比度、饱和度、清晰度
- **人脸分析**：人脸检测率、平均人脸大小、人脸稳定性
- **运动分析**：运动强度、帧间差异
- **图像质量**：人脸区域的PSNR、SSIM、LPIPS（需要真值）

### ✅ VBench指标集成 (新增)
- **主体一致性 (Subject Consistency)**：视频中主体对象的一致性
- **背景一致性 (Background Consistency)**：背景内容的稳定性
- **运动平滑性 (Motion Smoothness)**：运动的流畅度
- **动态程度 (Dynamic Degree)**：视频的动态变化程度
- **美学质量 (Aesthetic Quality)**：视频的美学评分
- **成像质量 (Imaging Quality)**：图像质量评估
- **灵活启用**：可选择性启用VBench计算以平衡性能

## 📁 目录结构

```
evalutation/
├── models/                     # 模型文件
│   ├── syncnet_v2.model       # SyncNet模型权重 (52MB)
│   └── sfd_face.pth           # S3FD人脸检测模型 (86MB)
├── syncnet_core/              # SyncNet核心模块
│   ├── __init__.py
│   ├── model.py               # SyncNet模型定义
│   └── detectors/             # 人脸检测器
│       ├── __init__.py
│       └── s3fd/
│           ├── __init__.py
│           ├── box_utils.py
│           └── nets.py
├── lse_calculator.py          # LSE计算器
├── metrics_calculator.py      # 综合指标计算器 (集成VBench)
├── vbench_official_final.py   # VBench直接集成模块
├── requirements.txt           # pip依赖配置
├── environment.yaml           # conda环境配置
├── verify_installation.py     # 安装验证脚本
├── examples/
│   └── example_usage.py       # 使用示例
└── README.md                  # 本文档
```

## 🚀 快速开始

### 📋 环境要求

**Python版本**: 3.8+ (推荐 3.9+)

**硬件要求**:
- **CPU**: Intel/AMD 多核处理器
- **内存**: 8GB+ RAM (推荐 16GB+)  
- **GPU**: NVIDIA GPU with CUDA 11.0+ (推荐，用于VBench和LSE加速)
- **存储**: 5GB+ 可用空间 (用于模型文件和VBench缓存)

**操作系统**: 
- Linux (推荐)
- Windows 10/11
- macOS 10.15+

### ⚙️ 安装依赖

#### 方法1: 使用VBench环境 (推荐)

如果您已有VBench环境，可直接使用：

```bash
# 激活VBench环境
conda activate vbench

# 安装额外依赖
pip install lpips python_speech_features scenedetect

# 验证安装
cd evalutation
python -c "from metrics_calculator import VideoMetricsCalculator; print('✅ 安装成功!')"
```

#### 方法2: 使用conda环境配置

```bash
# 使用预配置的environment.yaml文件
cd evalutation
conda env create -f environment.yaml

# 激活环境
conda activate video-evaluation

# 验证安装
python -c "from metrics_calculator import VideoMetricsCalculator; print('✅ 安装成功!')"
```

#### 方法3: 使用pip安装

```bash
# 克隆或进入项目目录
cd evalutation

# 安装所有依赖
pip install -r requirements.txt

# 如果有NVIDIA GPU，推荐使用CUDA版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # for CUDA 12.1
```

### 🔧 依赖说明

完整的依赖列表请参考：
- **pip用户**: [`requirements.txt`](requirements.txt)
- **conda用户**: [`environment.yaml`](environment.yaml)

#### 主要依赖包

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `torch` | ≥2.0.0 | 深度学习框架，SyncNet和VBench模型推理 |
| `torchvision` | ≥0.15.0 | 视觉模型支持 |
| `opencv-python` | ≥4.5.0 | 图像/视频处理，人脸检测 |
| `numpy` | ≥1.21.0 | 数值计算 |
| `scipy` | ≥1.8.0 | 科学计算，信号处理 |
| `scikit-image` | ≥0.19.0 | PSNR/SSIM图像质量指标 |
| `lpips` | ≥0.1.4 | 感知图像质量指标 |
| `python-speech-features` | ≥0.6.0 | MFCC音频特征提取 |
| `librosa` | ≥0.9.0 | 音频处理和分析 |
| `scenedetect` | ≥0.6.0 | 视频场景检测 |
| `ffmpeg-python` | ≥0.2.0 | 视频格式转换 |
| `vbench` | latest | VBench视频生成质量评估 |
| `mediapipe` | ≥0.10.0 | 现代化人脸检测（推荐） |
| `ultralytics` | ≥8.0.0 | YOLOv8人脸检测（可选） |
| `tqdm` | ≥4.62.0 | 进度条显示 |

#### 系统要求

- **FFmpeg**: 需要系统安装FFmpeg用于视频处理
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  
  # macOS (使用Homebrew)
  brew install ffmpeg
  
  # Windows
  # 下载并安装 https://ffmpeg.org/download.html
  ```

### ✅ 验证安装

安装完成后，建议运行验证脚本确保所有依赖都正确安装：

```bash
cd evalutation
python verify_installation.py
```

## 💻 使用方法

### 1. Python API使用

#### 基础指标计算（快速模式）
```python
from metrics_calculator import VideoMetricsCalculator

# 创建快速模式计算器（不包含VBench）
calculator = VideoMetricsCalculator(enable_vbench=False)

# 计算单个视频指标
metrics = calculator.calculate_video_metrics("video.mp4")

print(f"帧数: {metrics['frame_count']}")
print(f"LSE分数: {metrics['lse_distance']}")
print(f"人脸检测率: {metrics['face_detection_rate']}")
```

#### 完整指标计算（包含VBench）
```python
from metrics_calculator import VideoMetricsCalculator

# 创建完整模式计算器（包含VBench）
calculator = VideoMetricsCalculator(enable_vbench=True)

try:
    # 计算单个视频指标（包含VBench的6个核心指标）
    metrics = calculator.calculate_video_metrics("video.mp4")
    
    # 基础指标
    print(f"帧数: {metrics['frame_count']}")
    print(f"LSE分数: {metrics['lse_distance']}")
    
    # VBench指标
    print(f"主体一致性: {metrics['subject_consistency']}")
    print(f"背景一致性: {metrics['background_consistency']}")
    print(f"运动平滑性: {metrics['motion_smoothness']}")
    print(f"动态程度: {metrics['dynamic_degree']}")
    print(f"美学质量: {metrics['aesthetic_quality']}")
    print(f"成像质量: {metrics['imaging_quality']}")
    
finally:
    # 清理资源
    calculator.cleanup()
```

#### 批量处理视频
```python
from metrics_calculator import VideoMetricsCalculator

# 创建计算器
calculator = VideoMetricsCalculator(enable_vbench=True)

try:
    # 批量计算指标
    results = calculator.calculate_batch_metrics(
        pred_dir="/path/to/videos",
        gt_dir="/path/to/ground_truth",  # 可选
        pattern="*.mp4"
    )
    
    # 保存结果
    calculator.save_results(results, "results.json")
    
    # 打印统计信息
    calculator.print_summary_stats(results)
    
finally:
    calculator.cleanup()
```

#### 包含真值对比
```python
# 包含真值对比（计算人脸区域的PSNR, SSIM, LPIPS）
metrics = calculator.calculate_video_metrics(
    pred_path="prediction.mp4",
    gt_path="ground_truth.mp4"
)
```

#### 单独使用LSE计算器
```python
from lse_calculator import LSECalculator

# 初始化
calculator = LSECalculator()

# 计算单个视频
lse_distance, lse_confidence = calculator.calculate_single_video("video.mp4")
print(f"LSE Distance: {lse_distance:.4f}")
print(f"LSE Confidence: {lse_confidence:.4f}")

# 批量计算
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = calculator.calculate_batch(video_paths)
```

### 2. 命令行使用

#### 快速模式（不计算VBench指标）
```bash
cd evalutation
python metrics_calculator.py --pred_dir /path/to/videos
```

#### 完整模式（包含VBench指标）
```bash
cd evalutation
python metrics_calculator.py --pred_dir /path/to/videos --vbench
```

#### 指定真值目录进行对比
```bash
python metrics_calculator.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --vbench \
    --output results.json
```

#### 自定义输出和文件模式
```bash
python metrics_calculator.py \
    --pred_dir /path/to/videos \
    --vbench \
    --output my_results.json \
    --pattern "*.avi"
```

#### 单个视频LSE计算
```bash
python lse_calculator.py --video /path/to/video.mp4
```

## 📊 支持的指标

### 🟢 不需要Ground Truth的指标

| 指标类别 | 指标名称 | 描述 | 取值范围 |
|---------|---------|------|---------|
| **基本信息** | `frame_count` | 视频帧数 | 正整数 |
| | `width`, `height` | 视频分辨率 | 正整数 |
| | `fps` | 帧率 | 正数 |
| | `duration_seconds` | 视频时长(秒) | 正数 |
| **图像统计** | `mean_brightness` | 平均亮度 | 0-255 |
| | `mean_contrast` | 平均对比度 | ≥0 |
| | `mean_saturation` | 平均饱和度 | 0-255 |
| | `sharpness_score` | 清晰度分数 | ≥0，越大越清晰 |
| **人脸分析** | `face_detection_rate` | 人脸检测率 | 0-1 |
| | `avg_face_size` | 平均人脸大小(像素) | ≥0 |
| | `face_stability` | 人脸位置稳定性 | 0-1，越大越稳定 |
| **运动分析** | `motion_intensity` | 运动强度 | ≥0 |
| | `frame_difference` | 平均帧间差异 | ≥0 |
| **唇同步** | `lse_distance` | LSE距离 | ≥0，越小越好 |
| | `lse_confidence` | LSE置信度 | ≥0，越大越好 |
| **VBench指标** | `subject_consistency` | 主体一致性 | 0-1，越大越好 |
| | `background_consistency` | 背景一致性 | 0-1，越大越好 |
| | `motion_smoothness` | 运动平滑性 | 0-1，越大越好 |
| | `dynamic_degree` | 动态程度 | 0-1，适中为好 |
| | `aesthetic_quality` | 美学质量 | 0-1，越大越好 |
| | `imaging_quality` | 成像质量 | 0-1，越大越好 |

### 🔴 需要Ground Truth的指标

| 指标名称 | 描述 | 取值范围 | 备注 |
|---------|------|---------|------|
| `face_psnr` | 人脸区域峰值信噪比 | ≥0，越大越好 | >25为良好 |
| `face_ssim` | 人脸区域结构相似性 | 0-1，越大越好 | >0.8为良好 |
| `face_lpips` | 人脸区域感知相似度 | ≥0，越小越好 | <0.2为良好 |

## 📈 输出示例

### 输出文件结构 (包含平均值统计)
```json
{
  "summary": {
    "total_videos": 10,
    "successful_videos": 9,
    "average_metrics": {
      "基本信息": {
        "frame_count": 125.3,
        "width": 960.0,
        "height": 544.0,
        "fps": 25.0,
        "duration_seconds": 5.01
      },
      "图像统计": {
        "mean_brightness": 118.45,
        "mean_contrast": 42.18,
        "mean_saturation": 125.67,
        "sharpness_score": 598.34
      },
      "人脸分析": {
        "face_detection_rate": 0.95,
        "avg_face_size": 11800.0,
        "face_stability": 0.89
      },
      "运动分析": {
        "motion_intensity": 0.62,
        "frame_difference": 7.85
      },
      "LSE指标": {
        "lse_distance": 8.45,
        "lse_confidence": 6.78
      },
      "VBench指标": {
        "subject_consistency": 0.845,
        "background_consistency": 0.798,
        "motion_smoothness": 0.723,
        "dynamic_degree": 0.612,
        "aesthetic_quality": 0.665,
        "imaging_quality": 0.689
      },
      "对比指标": {
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
    // ... 更多视频结果 ...
  ]
}
```

## 🔧 技术细节

### LSE计算原理
1. **视频预处理**：提取帧和音频，转换格式
2. **人脸检测与跟踪**：使用S3FD检测器进行人脸检测和跟踪
3. **人脸视频裁剪**：提取并裁剪人脸区域
4. **特征提取**：使用SyncNet提取视频和音频特征
5. **LSE计算**：计算音视频特征间的距离和置信度

### VBench集成原理
1. **直接集成**：直接使用VBench官方库，确保结果一致性
2. **6个核心指标**：主体一致性、背景一致性、运动平滑性、动态程度、美学质量、成像质量
3. **可选启用**：通过`enable_vbench`参数控制是否计算VBench指标
4. **资源管理**：自动管理VBench临时文件和计算资源

### 现代化人脸检测 (新增)
1. **多种检测器支持**：自动选择最佳可用的人脸检测器
2. **优先级顺序**：MediaPipe > YOLOv8 > OpenCV DNN > Haar级联
3. **性能提升**：相比传统Haar级联，速度提升3-10倍，精度提升显著
4. **智能降级**：如果高级检测器不可用，自动降级到可用方法

### 模型文件
- **SyncNet模型** (`syncnet_v2.model`, 52MB)：用于音视频特征提取
- **S3FD模型** (`sfd_face.pth`, 86MB)：用于人脸检测
- **VBench模型**：自动下载和管理，用于6个核心指标计算

### 设备支持
- **CUDA**：支持GPU加速（推荐）
- **CPU**：支持CPU计算（较慢）

## 🚨 故障排除

### VBench相关问题
1. **VBench初始化失败**：确保VBench库正确安装且版本兼容
2. **CUDA内存不足**：可尝试使用CPU模式或减少batch_size
3. **网络连接问题**：VBench首次运行需要下载模型，确保网络连接

### LSE计算失败
1. **检查视频格式**：确保视频包含音频轨道
2. **检查模型文件**：确保模型文件存在且完整
3. **检查依赖**：确保所有依赖包已安装

### 人脸检测失败
1. **视频质量**：确保视频中有清晰可见的人脸
2. **分辨率**：过低的分辨率可能影响人脸检测
3. **光照条件**：过暗或过亮可能影响检测效果

### 性能优化
1. **选择模式**：使用快速模式（不启用VBench）获得更快速度
2. **使用GPU**：启用CUDA加速可显著提升计算速度
3. **现代化人脸检测**：安装MediaPipe可获得3-10倍人脸检测速度提升
4. **批量处理**：批量计算比逐个计算更高效
5. **资源清理**：使用`calculator.cleanup()`释放VBench资源

## 📚 API参考

### VideoMetricsCalculator

#### 初始化参数
- `device` (str): 计算设备 ("cuda" 或 "cpu")
- `enable_vbench` (bool): 是否启用VBench指标计算，默认False

#### 主要方法
- `calculate_video_metrics(pred_path, gt_path=None)`: 计算单个视频指标
- `calculate_batch_metrics(pred_dir, gt_dir=None, pattern="*.mp4")`: 批量计算指标
- `save_results(results, output_path)`: 保存结果到JSON文件
- `print_summary_stats(results)`: 打印汇总统计信息
- `cleanup()`: 清理VBench资源

### LSECalculator

#### 初始化参数
- `model_path` (str, optional): SyncNet模型路径
- `device` (str): 计算设备 ("cuda" 或 "cpu")
- `batch_size` (int): 批处理大小，默认20
- `vshift` (int): 视频偏移范围，默认15

#### 主要方法
- `calculate_single_video(video_path, verbose=True)`: 计算单个视频LSE
- `calculate_batch(video_paths, verbose=True)`: 批量计算LSE

## 🎯 最佳实践

1. **性能与准确性平衡**：
   - 快速评估：使用`enable_vbench=False`
   - 完整评估：使用`enable_vbench=True`

2. **文件组织**：
   - 预测视频和真值视频使用相同的文件命名
   - 使用有意义的目录结构

3. **性能优化**：
   - 优先使用GPU进行计算
   - 对于大批量任务，考虑分批处理
   - 及时调用`cleanup()`释放VBench资源

4. **结果分析**：
   - 使用汇总统计功能分析整体性能
   - 重点关注LSE、人脸质量和VBench指标的分布
   - VBench指标提供了更全面的视频生成质量评估

5. **错误处理**：
   - 检查输出中的error字段
   - 对于失败的视频，检查视频质量和格式
   - VBench计算失败时会优雅降级，其他指标仍可计算

## 🆕 更新日志

### v2.0.0 (当前版本)
- ✅ **新增VBench集成**：支持6个核心VBench指标
- ✅ **灵活启用机制**：可选择性启用VBench以平衡性能
- ✅ **统一API设计**：VBench指标集成到统一的metrics接口
- ✅ **资源管理优化**：自动管理VBench临时文件和计算资源
- ✅ **增强错误处理**：VBench失败时不影响其他指标计算
- ✅ **更新依赖管理**：优化requirements.txt和environment.yaml

### v1.0.0
- ✅ 整合LSE计算器和视频指标计算器
- ✅ 重构目录结构，优化模块组织
- ✅ 统一API接口，简化使用方式
- ✅ 添加详细文档和使用示例
- ✅ 验证与原始SyncNet脚本的一致性

## 📄 许可证

本项目基于原始SyncNet、VBench和相关开源项目，遵循相应的开源许可证。

## 👨‍💻 作者

**Fating Hong** / fatinghong@gmail.com

视频评估工具包的VBench集成版本由Fating Hong开发维护。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具包！

---

**🎉 现在您可以使用这个强大的视频评估工具包（包含VBench指标）来全面评估您的视频生成结果了！** 