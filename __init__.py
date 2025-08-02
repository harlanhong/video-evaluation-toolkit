#!/usr/bin/env python3
"""
Video Evaluation Toolkit
A comprehensive toolkit for video quality evaluation and analysis.

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

Modules:
    - apis: Unified API interfaces for video evaluation
    - calculators: Independent metric calculators
    - core: Core video metrics calculation engine
    - utils: Utility functions and scripts
"""

__version__ = "2.0.0"
__author__ = "Fating Hong"
__email__ = "fatinghong@gmail.com"

# Import main classes for convenience
try:
    from .core.video_metrics_calculator import VideoMetricsCalculator
    from .apis.clip_api import CLIPVideoAPI
    
    __all__ = [
        'VideoMetricsCalculator',
        'CLIPVideoAPI',
    ]
except ImportError:
    # Handle cases where dependencies are not installed
    __all__ = []