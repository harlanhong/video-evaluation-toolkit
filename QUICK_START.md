# Video Evaluation Toolkit - Quick Start Guide

## 🎉 Installation Completed Successfully!

### Environment Information
- Setup Directory: /data/fating/src/video-dit-v2-token-replace/evalutation
- Models Directory: /data/fating/src/video-dit-v2-token-replace/evalutation/models
- Environment: Conda
- GIM Integration: ✅ Available
- Models Downloaded: ✅ Complete

### Quick Usage Examples

#### 1. Basic Video Metrics
```python
from evalutation.core.video_metrics_calculator import VideoMetricsCalculator

calculator = VideoMetricsCalculator()
metrics = calculator.calculate_video_metrics(
    pred_path="your_video.mp4",
    gt_path="reference_video.mp4"  # Optional
)
print(f"LSE Score: {metrics['lse_score']}")
```

#### 2. Advanced Metrics with GIM
```python
calculator = VideoMetricsCalculator(
    enable_clip_similarity=True,
    enable_gim_matching=True
)
metrics = calculator.calculate_video_metrics(
    pred_path="generated_video.mp4",
    gt_path="reference_video.mp4"
)
print(f"CLIP Similarity: {metrics['clip_similarity']:.4f}")
print(f"GIM Matching: {metrics['gim_matching_pixels']}")
```

#### 3. Command Line Usage
```bash
python -m core.video_metrics_calculator \
    --pred generated_video.mp4 \
    --gt reference_video.mp4 \
    --clip --gim
```

### Available Examples
- `examples/basic_usage.py` - Basic usage examples
- `examples/advanced_metrics.py` - Advanced metrics demonstration
- `examples/clip_api_demo.py` - CLIP API examples
- `examples/gim_demo.py` - GIM integration examples

### Documentation
- `docs/README.md` - Main documentation
- `docs/GIM_INTEGRATION.md` - GIM integration guide
- `docs/MODELS_DOWNLOAD.md` - Model download instructions

### Next Steps
1. Try the examples: `python examples/basic_usage.py`
2. Read the documentation in `docs/`
3. Test with your own videos
4. Explore advanced features

### Troubleshooting
- Check installation: `python -c "from evalutation.core.video_metrics_calculator import VideoMetricsCalculator; print('✅ Working!')"`
- Update dependencies: `pip install -r configs/requirements.txt --upgrade`
- Reinstall GIM: `python utils/install_gim.py --force`

Happy evaluating! 🎬
