#!/usr/bin/env python3
"""
APIs Module - Unified API Interfaces

This module provides unified API interfaces for video evaluation tasks.

Available APIs:
    - CLIPVideoAPI: Comprehensive CLIP-based video evaluation
"""

try:
    from .clip_api import CLIPVideoAPI
    
    __all__ = [
        'CLIPVideoAPI',
    ]
except ImportError:
    __all__ = []