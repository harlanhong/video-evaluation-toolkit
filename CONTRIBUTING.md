# Contributing to Video Evaluation Toolkit

Thank you for your interest in contributing to the Video Evaluation Toolkit! This document provides guidelines for contributing to this project.

## 🎯 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [How to Contribute](#-how-to-contribute)
- [Development Setup](#-development-setup)
- [Code Standards](#-code-standards)
- [Testing Guidelines](#-testing-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Issue Reporting](#-issue-reporting)
- [Community Guidelines](#-community-guidelines)

## 📋 Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be collaborative**: Work together to improve the project
- **Be constructive**: Provide helpful feedback and suggestions

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.8+** installed
- **Git** for version control
- **Basic knowledge** of video processing, machine learning, or related fields
- **Familiarity** with PyTorch, OpenCV, or similar libraries

### Quick Setup

```bash
# Fork and clone your fork
git clone https://github.com/YOUR-USERNAME/video-evaluation-toolkit.git
cd video-evaluation-toolkit

# Set up development environment
python setup.py --mode conda --gpu

# Install development dependencies
pip install pytest black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **🐛 Bug Reports**: Report issues you encounter
2. **💡 Feature Requests**: Suggest new features or improvements
3. **📝 Documentation**: Improve or add documentation
4. **🔧 Code Contributions**: Fix bugs or implement features
5. **🧪 Testing**: Add or improve tests
6. **🎨 Examples**: Add usage examples or tutorials

### Priority Areas

We especially welcome contributions in these areas:

- **New Metrics**: Implementation of additional video quality metrics
- **Face Detection**: Improvements to face detection accuracy/speed
- **Performance**: Optimization of existing algorithms
- **Platform Support**: Better cross-platform compatibility
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: Unit tests and integration tests

## 💻 Development Setup

### Environment Setup

```bash
# Create development environment
conda create -n video-eval-dev python=3.9
conda activate video-eval-dev

# Install in development mode
pip install -e .
pip install -r configs/requirements-dev.txt

# Install pre-commit hooks for code quality
pre-commit install
```

### Project Structure

```
video-evaluation-toolkit/
├── core/                    # Core calculation engines
├── calculators/             # Individual metric calculators
├── apis/                    # Unified API interfaces
├── examples/                # Usage examples
├── tests/                   # Test suite
├── docs/                    # Documentation
├── configs/                 # Configuration files
└── utils/                   # Utility functions
```

### Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/new-metric-xyz
   ```

2. **Make your changes** following our code standards

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   python examples/basic_usage.py  # Smoke test
   ```

4. **Commit with descriptive messages**:
   ```bash
   git commit -m "✨ Add XYZ metric calculator with CUDA support"
   ```

5. **Push and create a Pull Request**

## 📏 Code Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Good example
class VideoMetricsCalculator:
    """
    Comprehensive video metrics calculator.
    
    Args:
        device (str): Computing device ('cuda' or 'cpu')
        enable_vbench (bool): Whether to enable VBench metrics
    """
    
    def __init__(self, device: str = "cuda", enable_vbench: bool = False):
        self.device = device
        self.enable_vbench = enable_vbench
        
    def calculate_metrics(self, video_path: str) -> Dict[str, Any]:
        """Calculate comprehensive video metrics."""
        # Implementation here
        pass
```

### Code Quality Tools

We use these tools to maintain code quality:

```bash
# Code formatting
black . --line-length 88

# Import sorting
isort . --profile black

# Linting
flake8 . --max-line-length 88 --ignore E203,W503

# Type checking
mypy . --ignore-missing-imports
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type annotations for all public functions
- **Comments**: Explain complex algorithms and important decisions
- **README updates**: Update relevant documentation when adding features

```python
def calculate_face_metrics(
    self, 
    pred_frame: np.ndarray, 
    gt_frame: np.ndarray,
    region: str = "face_only"
) -> Dict[str, float]:
    """
    Calculate image quality metrics for face region.
    
    Args:
        pred_frame: Predicted frame (H, W, 3) in RGB format
        gt_frame: Ground truth frame (H, W, 3) in RGB format  
        region: Region to calculate metrics for ('face_only' or 'full_image')
        
    Returns:
        Dictionary containing PSNR, SSIM, and LPIPS scores
        
    Raises:
        ValueError: If frames have incompatible shapes
        RuntimeError: If face detection fails in face_only mode
    """
```

## 🧪 Testing Guidelines

### Test Structure

```bash
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_lse_calculator.py
│   ├── test_face_detection.py
│   └── test_metrics.py
├── integration/            # Integration tests
│   ├── test_full_pipeline.py
│   └── test_batch_processing.py
├── data/                   # Test data and fixtures
│   ├── sample_videos/
│   └── expected_outputs/
└── conftest.py            # pytest configuration
```

### Writing Tests

```python
import pytest
import numpy as np
from core.video_metrics_calculator import VideoMetricsCalculator

class TestVideoMetricsCalculator:
    """Test suite for VideoMetricsCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return VideoMetricsCalculator(device="cpu", enable_vbench=False)
    
    def test_face_metrics_calculation(self, calculator):
        """Test face metrics calculation with known inputs."""
        # Create test frames
        pred_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gt_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Calculate metrics
        metrics = calculator.calculate_frame_metrics(pred_frame, gt_frame)
        
        # Assertions
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'lpips' in metrics
        assert 0 <= metrics['ssim'] <= 1
        assert metrics['psnr'] > 0
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_lse_calculator.py

# Run with verbose output
python -m pytest -v -s
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   python -m pytest
   ```

2. **Run code quality checks**:
   ```bash
   black . --check
   flake8 .
   mypy .
   ```

3. **Test with real data** (if applicable):
   ```bash
   python examples/basic_usage.py
   ```

4. **Update documentation** if needed

### PR Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] Added/updated unit tests
- [ ] Added/updated integration tests
- [ ] Manual testing completed
- [ ] All existing tests pass

## Screenshots/Examples
(If applicable, add screenshots or example outputs)

## Checklist
- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** on multiple platforms (if applicable)
4. **Documentation review** for user-facing changes
5. **Final approval** and merge

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

```markdown
**Bug Description**
Clear description of what happened.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 2.1.0]
- GPU: [e.g., NVIDIA RTX 3080]

**Additional Context**
- Error logs/screenshots
- Sample video files (if relevant)
- Configuration used
```

### Feature Requests

For feature requests, please include:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives**: What alternatives have you considered?
- **Examples**: Similar features in other tools

## 🌟 Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Email**: fatinghong@gmail.com for private matters

### Recognition

Contributors will be recognized in:

- **README.md**: Contributor list
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

### Maintainer Responsibilities

Project maintainers will:

- **Respond** to issues and PRs within 48-72 hours
- **Provide feedback** on contributions
- **Maintain** code quality and project direction
- **Release** regular updates and improvements

## 🎉 Getting Help

### Resources

- **Documentation**: [docs/README.md](docs/README.md)
- **Examples**: [examples/](examples/) directory
- **API Reference**: Code docstrings and type hints
- **Issues**: Search existing issues for solutions

### Questions?

Don't hesitate to:

1. **Check existing issues** and discussions
2. **Open a new issue** with the "question" label
3. **Start a discussion** for broader topics
4. **Contact maintainers** directly if needed

---

## 🙏 Thank You

Thank you for contributing to the Video Evaluation Toolkit! Your contributions help make video quality assessment more accessible and reliable for everyone.

**Happy contributing!** 🚀

---

*This contributing guide is inspired by the best practices from the open-source community and will be updated as the project evolves.*