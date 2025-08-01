# Evalutation Directory - 清理后的文件概览

## 📁 目录结构
```
evalutation/
├── 📄 核心脚本
│   ├── vbench_official_final.py     # 🎯 VBench官方集成脚本（主要）
│   ├── metrics_calculator.py        # 📊 统一指标计算器
│   ├── lse_calculator.py           # 🎭 LSE(Lip-Sync Error)计算API
│   └── compare_official_vs_our.py  # ✅ 官方结果对比验证脚本
│
├── 📄 配置文件
│   ├── requirements.txt            # 🐍 Python依赖包列表
│   ├── environment.yaml            # 🐍 Conda环境配置
│   └── README.md                   # 📖 主要文档说明
│
├── 📁 模型文件
│   └── models/                     # 🧠 预训练模型（SyncNet等）
│       ├── sfd_face.pth            # 👤 人脸检测模型
│       ├── syncnet_v2.model        # 💋 SyncNet V2模型
│       └── sa_0_4_vit_l_14_linear.pth # 🎨 Aesthetic质量模型
│
├── 📁 核心模块
│   └── syncnet_core/               # 💋 SyncNet核心模块
│       ├── model.py                # SyncNet模型定义
│       ├── instance.py             # SyncNet实例
│       └── detectors/              # 人脸检测器
│
├── 📁 示例代码
│   └── examples/                   # 📝 使用示例
│       └── usage_example.py        # 💡 综合使用示例
│
└── 📁 测试文件
    └── test_comparison/            # 🧪 测试对比
        └── test_video.mp4          # 📹 测试视频文件
```

## 🚀 主要功能

### 1. VBench集成 (`vbench_official_final.py`)
- 直接集成官方VBench核心逻辑
- 支持6个核心指标：subject_consistency, background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality
- 与官方版本100%一致

### 2. 统一指标计算 (`metrics_calculator.py`)
- 集成多种视频质量指标
- 支持有/无ground truth的计算
- 人脸区域特定的SSIM/PSNR/LPIPS计算

### 3. LSE计算 (`lse_calculator.py`)
- Lip-Sync Error (LSE-C, LSE-D) Python API
- 基于SyncNet的准确实现
- 与官方SyncNet脚本结果一致

## 📊 使用方法

### VBench指标计算
```bash
# 计算所有6个VBench指标
python vbench_official_final.py --videos video.mp4

# 计算特定指标
python vbench_official_final.py --videos video.mp4 --metrics subject_consistency
```

### 综合指标计算
```bash
# 有ground truth的情况
python metrics_calculator.py --generated_dir /path/to/generated --gt_dir /path/to/gt

# 无ground truth的情况  
python metrics_calculator.py --generated_dir /path/to/generated --no_gt
```

### LSE指标计算
```bash
python lse_calculator.py --video_dir /path/to/videos
```

## ✅ 验证
使用 `compare_official_vs_our.py` 验证结果与官方VBench的一致性。

## 🔧 环境设置

### 使用Conda
```bash
conda env create -f environment.yaml
conda activate vbench
```

### 使用pip
```bash
pip install -r requirements.txt
```

---
*此目录已经过清理，只保留核心功能文件，删除了所有冗余和实验性文件。* 