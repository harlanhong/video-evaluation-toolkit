# 🤖 Model File Download Guide

This project requires some model files to run correctly. Due to their large size (over 100MB), they are not included in the git repository.

## 📋 Required Model Files

### ✅ Required Files
```

models/
├── syncnet\_v2.model     \# SyncNet Model (52MB) - Required for LSE calculation
└── sfd\_face.pth         \# S3FD Face Detection Model (86MB) - Required for LSE calculation

````

## 📥 Download Methods

### Method 1: Automatic Download Script (Recommended)

```bash
# Run the automatic download script
python download_models.py
````

### Method 2: Manual Download

#### SyncNet Model

```bash
# Create the models directory
mkdir -p models

# Download the SyncNet model
wget -O models/syncnet_v2.model "[https://github.com/joonson/syncnet_python/raw/master/data/syncnet_v2.model](https://github.com/joonson/syncnet_python/raw/master/data/syncnet_v2.model)"
```

#### S3FD Face Detection Model

```bash
# Download the S3FD model
wget -O models/sfd_face.pth "[https://github.com/1adrianb/face-alignment/raw/master/face_alignment/models/s3fd-619a316812.pth](https://github.com/1adrianb/face-alignment/raw/master/face_alignment/models/s3fd-619a316812.pth)"
```

### Method 3: Download from Original Sources

  - **SyncNet Model**: [Official SyncNet Repository](https://github.com/joonson/syncnet_python)
  - **S3FD Model**: [Face Alignment Repository](https://github.com/1adrianb/face-alignment)

## ✅ Verify Installation

After downloading, run the verification script:

```bash
python verify_installation.py
```

Expected output:

```
✅ SyncNet Model (syncnet_v2.model): Found (52MB)
✅ S3FD Model (sfd_face.pth): Found (86MB)
🎉 All model files are ready!
```

## 🚨 Troubleshooting

### Download Failure

  - Check your network connection.
  - Use a VPN (if you are in certain regions).
  - Manually download from a mirror site.

### Corrupted Files

```bash
# Check file sizes
ls -lh models/

# Re-download the corrupted file
rm models/syncnet_v2.model  # Delete the corrupted file
# Then download it again
```

### Permission Issues

```bash
# Ensure you have write permissions
chmod 755 models/
chmod 644 models/*.model
chmod 644 models/*.pth
```

## 📁 Final Directory Structure

```
evaluation/
├── models/
│   ├── syncnet_v2.model     ✅ Downloaded
│   └── sfd_face.pth         ✅ Downloaded
├── metrics_calculator.py
├── vbench_official_final.py
├── lse_calculator.py
└── ...other files
```

## 🔐 File Integrity Check

If you need to verify file integrity:

```bash
# SyncNet Model MD5 (Optional)
md5sum models/syncnet_v2.model

# S3FD Model MD5 (Optional)
md5sum models/sfd_face.pth
```

## 💡 Tips

  - The model files only need to be downloaded once.
  - If you use different versions of the models, you may need to adjust the code.
  - For VBench functionality, models will be automatically downloaded to a cache directory.
  - Modern face detectors (MediaPipe/YOLOv8) do not require additional model files.

-----

**📧 For any issues, please contact: Fating Hong [fatinghong@gmail.com](mailto:fatinghong@gmail.com)**