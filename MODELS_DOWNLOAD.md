# 🤖 模型文件下载指南

本项目需要一些模型文件才能正常运行。由于文件较大（超过100MB），它们不包含在git仓库中。

## 📋 需要的模型文件

### ✅ 必需文件
```
models/
├── syncnet_v2.model     # SyncNet模型 (52MB) - LSE计算必需
└── sfd_face.pth         # S3FD人脸检测模型 (86MB) - LSE计算必需
```

## 📥 下载方法

### 方法1: 自动下载脚本（推荐）

```bash
# 运行自动下载脚本
python download_models.py
```

### 方法2: 手动下载

#### SyncNet模型
```bash
# 创建models目录
mkdir -p models

# 下载SyncNet模型
wget -O models/syncnet_v2.model "https://github.com/joonson/syncnet_python/raw/master/data/syncnet_v2.model"
```

#### S3FD人脸检测模型
```bash
# 下载S3FD模型
wget -O models/sfd_face.pth "https://github.com/1adrianb/face-alignment/raw/master/face_alignment/models/s3fd-619a316812.pth"
```

### 方法3: 从原始源下载

- **SyncNet模型**: [官方SyncNet仓库](https://github.com/joonson/syncnet_python)
- **S3FD模型**: [Face Alignment仓库](https://github.com/1adrianb/face-alignment)

## ✅ 验证安装

下载完成后，运行验证脚本：

```bash
python verify_installation.py
```

期望输出：
```
✅ SyncNet模型 (syncnet_v2.model): 存在 (52MB)
✅ S3FD模型 (sfd_face.pth): 存在 (86MB)
🎉 所有模型文件就绪！
```

## 🚨 故障排除

### 下载失败
- 检查网络连接
- 使用VPN（如果在某些地区）
- 手动从镜像站点下载

### 文件损坏
```bash
# 检查文件大小
ls -lh models/

# 重新下载损坏的文件
rm models/syncnet_v2.model  # 删除损坏文件
# 然后重新下载
```

### 权限问题
```bash
# 确保有写权限
chmod 755 models/
chmod 644 models/*.model
chmod 644 models/*.pth
```

## 📁 最终目录结构

```
evalutation/
├── models/
│   ├── syncnet_v2.model     ✅ 已下载
│   └── sfd_face.pth         ✅ 已下载
├── metrics_calculator.py
├── vbench_official_final.py
├── lse_calculator.py
└── ...其他文件
```

## 🔐 文件校验

如果需要验证文件完整性：

```bash
# SyncNet模型 MD5 (可选)
md5sum models/syncnet_v2.model

# S3FD模型 MD5 (可选)  
md5sum models/sfd_face.pth
```

## 💡 提示

- 模型文件只需下载一次
- 如果使用不同版本的模型，可能需要调整代码
- 对于VBench功能，模型会自动下载到缓存目录
- 现代化人脸检测（MediaPipe/YOLOv8）不需要额外模型文件

---

**📧 如有问题，请联系：Fating Hong <fatinghong@gmail.com>** 