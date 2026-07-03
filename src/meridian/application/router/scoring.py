"""Vector scoring primitives for the semantic router.

This module is the numerical heart of the router, isolated from any I/O so it
can be unit-tested with hand-built matrices and reasoned about independently.
Every function here is pure: given the same arrays, it returns the same result.

The scoring model is the one used in the production system this reference is
modelled on. For a query vector ``q`` and an intent with a positive example
matrix ``M_pos`` and (optional) negative example matrix ``M_neg``:

    score = max(M_pos @ q) - NEG_PENALTY * max(0, max(M_neg @ q))

The ``max`` over the rows captures similarity to the *closest* example rather
than the average, which preserves signal for intents whose examples are
multi-modal. The negative term penalises queries that look like a known
confusable - the phrases that resemble this intent but belong to another.
"""

import numpy as np


def unit_norm(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (1-D) or each row of a matrix (2-D).

    Rows with zero norm are returned unchanged rather than producing ``NaN``.
    Embedding providers are expected to return normalised vectors already; this
    is a defensive guard for hand-built test matrices and for any provider that
    does not normalise.

    :param vectors: A 1-D vector or 2-D matrix of shape ``(n, dim)``.
    :returns: The input normalised to unit L2 norm, as ``float32``.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr if norm == 0.0 else (arr / norm).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32)


def max_similarity(matrix: np.ndarray, query: np.ndarray) -> float:
    """Return the maximum dot product between ``query`` and any row of ``matrix``.

    With normalised inputs this is the maximum cosine similarity. An empty
    matrix scores ``0.0`` so that an intent with no examples never wins.

    :param matrix: Example matrix of shape ``(n, dim)``.
    :param query: Query vector of shape ``(dim,)``.
    :returns: The best similarity, or ``0.0`` if the matrix is empty.
    """
    if matrix.size == 0 or query.size == 0:
        return 0.0
    if matrix.shape[1] != query.shape[0]:
        # Shape mismatch means a corrupt cache or a dimension change; score it
        # out rather than raising, so one bad intent cannot break routing.
        return 0.0
    return float(np.max(matrix @ query))


def score_intent(
    positives: np.ndarray,
    negatives: np.ndarray,
    query: np.ndarray,
    negative_penalty: float,
) -> float:
    """Compute a single intent's score for a query.

    Implements ``max(M_pos @ q) - penalty * max(0, max(M_neg @ q))``. The
    negative term only ever subtracts (it's clamped at zero) so that a query
    unrelated to an intent's negatives is never rewarded.

    :param positives: Positive example matrix, shape ``(n_pos, dim)``.
    :param negatives: Negative example matrix, shape ``(n_neg, dim)`` (may be empty).
    :param query: Normalised query vector, shape ``(dim,)``.
    :param negative_penalty: Weight applied to the negative similarity.
    :returns: The intent's score. Not bounded to ``[0, 1]``.
    """
    positive_score = max_similarity(positives, query)
    if negatives.size == 0:
        return positive_score
    negative_score = max_similarity(negatives, query)
    return positive_score - negative_penalty * max(0.0, negative_score)
