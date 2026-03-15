"""
CNN model architectures for step counting.
"""

from .shallow_cnn import ShallowCNN
from .deep_cnn import DeepCNN

__all__ = ['ShallowCNN', 'DeepCNN']
