"""
SyncNet Core Module
包含SyncNet模型和人脸检测器的核心实现

Copyright (c) 2024 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module contains core SyncNet model and face detector implementations.
"""

from .model import S
from .detectors import S3FD

__version__ = "1.0.0"
__all__ = ["S", "S3FD"] 