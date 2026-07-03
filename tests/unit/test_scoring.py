"""Unit tests for the router scoring primitives.

These tests exercise the pure numerical core with hand-built matrices, so they
run in microseconds and pin down the exact scoring contract the rest of the
system relies on. No embeddings, no I/O.
"""

import numpy as np

from meridian.application.router.scoring import max_similarity, score_intent, unit_norm


def test_unit_norm_normalises_a_vector() -> None:
    """A non-zero vector is scaled to unit L2 norm."""
    result = unit_norm(np.array([3.0, 4.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_unit_norm_leaves_zero_vector_unchanged() -> None:
    """A zero vector is returned as-is rather than producing NaNs."""
    result = unit_norm(np.zeros(4, dtype=np.float32))
    assert np.all(result == 0.0)


def test_unit_norm_normalises_matrix_rows() -> None:
    """Each row of a matrix is independently normalised."""
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    result = unit_norm(matrix)
    assert np.allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])


def test_max_similarity_picks_the_closest_row() -> None:
    """The score is the dot product with the nearest example, not the average."""
    matrix = unit_norm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    query = unit_norm(np.array([1.0, 0.0], dtype=np.float32))
    assert np.isclose(max_similarity(matrix, query), 1.0)


def test_max_similarity_empty_matrix_scores_zero() -> None:
    """An intent with no examples never wins."""
    empty = np.zeros((0, 2), dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert max_similarity(empty, query) == 0.0


def test_max_similarity_shape_mismatch_scores_zero() -> None:
    """A dimension mismatch (corrupt cache) scores out rather than raising."""
    matrix = np.ones((2, 3), dtype=np.float32)
    query = np.ones(2, dtype=np.float32)
    assert max_similarity(matrix, query) == 0.0


def test_score_intent_without_negatives_is_positive_similarity() -> None:
    """With no negatives, the score equals the positive similarity."""
    positives = unit_norm(np.array([[1.0, 0.0]], dtype=np.float32))
    negatives = np.zeros((0, 2), dtype=np.float32)
    query = unit_norm(np.array([1.0, 0.0], dtype=np.float32))
    assert np.isclose(score_intent(positives, negatives, query, 0.8), 1.0)


def test_score_intent_negative_penalty_reduces_score() -> None:
    """A query resembling a negative example is penalised."""
    positives = unit_norm(np.array([[1.0, 0.0]], dtype=np.float32))
    negatives = unit_norm(np.array([[1.0, 0.0]], dtype=np.float32))
    query = unit_norm(np.array([1.0, 0.0], dtype=np.float32))
    # positive 1.0 minus penalty 0.8 * negative 1.0 = 0.2
    assert np.isclose(score_intent(positives, negatives, query, 0.8), 0.2)


def test_score_intent_negative_term_never_rewards() -> None:
    """A negative similarity below zero is clamped, so it cannot raise the score."""
    positives = unit_norm(np.array([[1.0, 0.0]], dtype=np.float32))
    negatives = unit_norm(np.array([[-1.0, 0.0]], dtype=np.float32))
    query = unit_norm(np.array([1.0, 0.0], dtype=np.float32))
    # negative similarity is -1.0, clamped to 0, so score stays at the positive 1.0
    assert np.isclose(score_intent(positives, negatives, query, 0.8), 1.0)
