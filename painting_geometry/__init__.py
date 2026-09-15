"""Interpretable multiscale descriptors for digitised paintings."""

from .appearance import strong_baseline_features
from .curvature import relative_scale_curvature_features
from .ordinal import ordinal_pattern_probabilities
from .orientation import structure_tensor_features
from .preprocessing import preprocess

__all__ = [
    "ordinal_pattern_probabilities",
    "preprocess",
    "relative_scale_curvature_features",
    "strong_baseline_features",
    "structure_tensor_features",
]
