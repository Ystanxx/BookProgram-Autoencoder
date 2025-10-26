"""
数据包
包含NIR光谱数据生成和加载功能
"""

from .data_generator import NIRProductDataGenerator, load_or_generate_data

__all__ = [
    'NIRProductDataGenerator',
    'load_or_generate_data',
]

