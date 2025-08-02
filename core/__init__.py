#!/usr/bin/env python3
"""
Core Module - Video Metrics Calculation Engine

This module provides the core video metrics calculation engine that integrates
all individual calculators into a comprehensive evaluation system.

Available Classes:
    - VideoMetricsCalculator: Comprehensive video metrics calculator
"""

try:
    from .video_metrics_calculator import VideoMetricsCalculator
    
    __all__ = [
        'VideoMetricsCalculator',
    ]
except ImportError:
    __all__ = []