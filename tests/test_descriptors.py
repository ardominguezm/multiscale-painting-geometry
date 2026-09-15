import numpy as np

from painting_geometry.appearance import strong_baseline_features
from painting_geometry.curvature import relative_scale_curvature_features
from painting_geometry.ordinal import dense_rank_codes_2x2, ordinal_pattern_probabilities
from painting_geometry.orientation import structure_tensor_features


def test_reported_descriptor_dimensions():
    rng = np.random.default_rng(7)
    field = rng.random((64, 64))
    appearance = strong_baseline_features(field)
    curvature = relative_scale_curvature_features(field, long_side=64)
    tensor = structure_tensor_features(field, sigma=0.25)
    assert len(appearance) == 90
    assert len(curvature) == 40
    assert len(tensor) == 4
    assert np.isfinite(list(appearance.values())).all()
    assert np.isfinite(list(curvature.values())).all()
    assert np.isfinite(list(tensor.values())).all()


def test_tie_aware_example_and_probability_vector():
    field = np.asarray([[0.2, 0.7], [0.2, 0.9]])
    assert dense_rank_codes_2x2(field).tolist() == [18]
    probabilities = ordinal_pattern_probabilities(field)
    assert len(probabilities) == 75
    assert probabilities["ord75__0102"] == 1.0
    assert np.isclose(sum(probabilities.values()), 1.0)
