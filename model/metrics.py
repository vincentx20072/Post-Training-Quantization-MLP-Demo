"""Small deterministic metric helpers shared by evaluation and the benchmark."""
from __future__ import annotations


def accuracy(correct: int, total: int) -> float:
    """Fraction correct in [0, 1]."""
    if total <= 0:
        raise ValueError("total must be positive")
    return correct / total


def percentage_points(a: float, b: float) -> float:
    """Absolute difference between two accuracies, expressed in percentage points."""
    return abs(a - b) * 100.0


def size_reduction(baseline_bytes: int, candidate_bytes: int) -> float:
    """Fraction by which ``candidate`` is smaller than ``baseline`` (0.75 == 75% smaller)."""
    if baseline_bytes <= 0:
        raise ValueError("baseline_bytes must be positive")
    return 1.0 - candidate_bytes / baseline_bytes


def compression_ratio(baseline_bytes: int, candidate_bytes: int) -> float:
    """How many times smaller the candidate is (4.0 == a quarter of the size)."""
    if candidate_bytes <= 0:
        raise ValueError("candidate_bytes must be positive")
    return baseline_bytes / candidate_bytes
