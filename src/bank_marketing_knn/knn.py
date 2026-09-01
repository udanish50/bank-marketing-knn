"""A clear manual implementation of k-nearest neighbors."""

from __future__ import annotations

import numpy as np


class ManualKNNClassifier:
    """Binary KNN classifier implemented with NumPy.

    The class intentionally mirrors the small educational function from the
    original notebook, while adding validation, batching, and probabilities.
    """

    def __init__(self, k: int = 5, batch_size: int = 256) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.k = k
        self.batch_size = batch_size
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ManualKNNClassifier":
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)

        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must contain the same number of rows.")
        if self.k > len(y_train):
            raise ValueError("k cannot be larger than the training set size.")
        if not set(np.unique(y_train)).issubset({0, 1}):
            raise ValueError("ManualKNNClassifier currently expects binary labels 0 and 1.")

        self.x_train_ = x_train
        self.y_train_ = y_train
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Return two-column probabilities for classes 0 and 1."""

        if self.x_train_ is None or self.y_train_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        x_test = np.asarray(x_test, dtype=float)
        positive_probs: list[np.ndarray] = []

        for start in range(0, len(x_test), self.batch_size):
            batch = x_test[start : start + self.batch_size]
            distances = self._euclidean_distances(batch, self.x_train_)
            nearest = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]
            neighbor_labels = self.y_train_[nearest]
            positive_probs.append(neighbor_labels.mean(axis=1))

        p1 = np.concatenate(positive_probs)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels using a 0.5 probability threshold."""

        return (self.predict_proba(x_test)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))
