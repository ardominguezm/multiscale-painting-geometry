"""Tie-aware two-dimensional ordinal-pattern descriptors."""

from __future__ import annotations

import itertools

import numpy as np
from PIL import Image
from skimage.color import rgb2gray


def all_weak_order_patterns() -> list[tuple[int, int, int, int]]:
    """Return the 75 dense weak orderings of four labelled positions."""
    patterns = []
    for pattern in itertools.product(range(4), repeat=4):
        if set(pattern) == set(range(max(pattern) + 1)):
            patterns.append(tuple(int(value) for value in pattern))
    if len(patterns) != 75:
        raise RuntimeError(f"Expected 75 weak orderings, found {len(patterns)}")
    return patterns


WEAK_ORDER_PATTERNS = all_weak_order_patterns()
WEAK_ORDER_CODES = np.asarray(
    [a * 64 + b * 16 + c * 4 + d for a, b, c, d in WEAK_ORDER_PATTERNS],
    dtype=np.int16,
)


def pattern_label(pattern: tuple[int, int, int, int]) -> str:
    return "".join(str(value) for value in pattern)


OP75_COLUMNS = [f"ord75__{pattern_label(pattern)}" for pattern in WEAK_ORDER_PATTERNS]


def load_ordinal_grayscale(path: str) -> np.ndarray:
    """Load RGB data and apply the grayscale transform used for OP75."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return np.asarray(rgb2gray(rgb), dtype=np.float64)


def dense_rank_codes_2x2(gray: np.ndarray) -> np.ndarray:
    """Encode dense ranks for every overlapping 2 x 2 window."""
    image = np.asarray(gray)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError(f"Expected a two-dimensional image at least 2 x 2; got {image.shape}")

    windows = np.stack(
        [
            image[:-1, :-1].ravel(),
            image[:-1, 1:].ravel(),
            image[1:, :-1].ravel(),
            image[1:, 1:].ravel(),
        ],
        axis=1,
    )
    order = np.argsort(windows, axis=1, kind="stable")
    sorted_values = np.take_along_axis(windows, order, axis=1)
    sorted_ranks = np.zeros(order.shape, dtype=np.uint8)
    sorted_ranks[:, 1:] = np.cumsum(
        sorted_values[:, 1:] != sorted_values[:, :-1], axis=1, dtype=np.uint8
    )
    ranks = np.empty_like(sorted_ranks)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return (
        ranks[:, 0].astype(np.int16) * 64
        + ranks[:, 1].astype(np.int16) * 16
        + ranks[:, 2].astype(np.int16) * 4
        + ranks[:, 3].astype(np.int16)
    )


def ordinal_pattern_probabilities(gray: np.ndarray) -> dict[str, float]:
    """Return the OP75 probability vector for a grayscale image."""
    codes = dense_rank_codes_2x2(gray)
    counts = np.bincount(codes, minlength=256)
    denominator = float(codes.size)
    return {
        column: float(counts[int(code)] / denominator)
        for column, code in zip(OP75_COLUMNS, WEAK_ORDER_CODES)
    }


def validate_against_ordpy() -> None:
    """Check exact numerical agreement with ordpy on tie-rich fields."""
    try:
        import ordpy
    except ImportError as exc:
        raise ImportError("Install ordpy>=1.2.0 to run the validation") from exc

    rng = np.random.default_rng(20260826)
    keys = [f"[{pattern_label(pattern)}]" for pattern in WEAK_ORDER_PATTERNS]
    for case in range(4):
        field = rng.integers(0, 7, size=(9 + case, 11 + case)).astype(float)
        observed = ordinal_pattern_probabilities(field)
        reference_raw = ordpy.two_by_two_patterns(
            field,
            taux=1,
            tauy=1,
            overlapping=True,
            tie_patterns=True,
            group_patterns=False,
        )
        reference = {key: float(reference_raw.get(key, 0.0)) for key in keys}
        left = np.asarray(list(observed.values()))
        right = np.asarray([reference[key] for key in keys])
        if not np.allclose(left, right, atol=1e-12, rtol=0):
            raise RuntimeError(
                f"OP75 validation failed in case {case}; "
                f"maximum difference={np.max(np.abs(left - right))}"
            )
