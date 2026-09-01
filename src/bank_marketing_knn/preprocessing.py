"""Preprocessing for the older 17-column Bank Marketing dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
BINARY_COLUMNS = ["default", "housing", "loan"]
CATEGORICAL_COLUMNS = ["job", "marital", "education", "contact", "month", "poutcome"]
TARGET_COLUMN = "y"


@dataclass
class BankMarketingPreprocessor:
    """Fit-on-train transformer for manual KNN experiments."""

    numeric_columns: list[str]
    binary_columns: list[str]
    categorical_columns: list[str]
    means_: pd.Series | None = None
    stds_: pd.Series | None = None
    feature_columns_: list[str] | None = None

    @classmethod
    def default(cls) -> "BankMarketingPreprocessor":
        return cls(
            numeric_columns=NUMERIC_COLUMNS.copy(),
            binary_columns=BINARY_COLUMNS.copy(),
            categorical_columns=CATEGORICAL_COLUMNS.copy(),
        )

    def fit(self, frame: pd.DataFrame) -> "BankMarketingPreprocessor":
        features = self._encode_without_scaling(frame)
        self.means_ = features[self.numeric_columns].mean()
        self.stds_ = features[self.numeric_columns].std().replace(0, 1)

        scaled = self._scale(features)
        self.feature_columns_ = scaled.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.stds_ is None or self.feature_columns_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        features = self._encode_without_scaling(frame)
        scaled = self._scale(features)
        aligned = scaled.reindex(columns=self.feature_columns_, fill_value=0)
        return aligned.to_numpy(dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _encode_without_scaling(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
        for column in self.binary_columns:
            features[column] = features[column].map({"yes": 1, "no": 0}).astype(int)
        return pd.get_dummies(features, columns=self.categorical_columns, dtype=int)

    def _scale(self, features: pd.DataFrame) -> pd.DataFrame:
        scaled = features.copy()
        scaled[self.numeric_columns] = (
            scaled[self.numeric_columns] - self.means_
        ) / self.stds_
        return scaled


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return raw features and binary target array."""

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'.")

    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].map({"yes": 1, "no": 0})
    if target.isna().any():
        bad_values = sorted(frame.loc[target.isna(), TARGET_COLUMN].unique())
        raise ValueError(f"Unexpected target values: {bad_values}")
    return features, target.to_numpy(dtype=int)
