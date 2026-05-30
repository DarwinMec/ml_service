from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class AutoCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder automático para variables categóricas.

    - Si una variable categórica tiene baja cardinalidad, usa OneHotEncoder.
    - Si tiene alta cardinalidad, usa OrdinalEncoder.

    Esta clase debe mantenerse en un módulo estable porque los modelos guardados
    con joblib necesitan encontrarla en la misma ruta al momento de cargar.
    """

    def __init__(self, max_ohe_cards: int = 10):
        self.max_ohe_cards = max_ohe_cards

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.columns_: List[str] = X.columns.tolist()

        categorical_columns = [
            col for col in self.columns_
            if X[col].dtype == "object"
            or str(X[col].dtype).startswith("category")
        ]

        self.low_cardinality_cols_ = [
            col for col in categorical_columns
            if X[col].nunique(dropna=False) <= self.max_ohe_cards
        ]

        self.high_cardinality_cols_ = [
            col for col in categorical_columns
            if col not in self.low_cardinality_cols_
        ]

        self.numeric_cols_ = [
            col for col in self.columns_
            if col not in self.low_cardinality_cols_
            and col not in self.high_cardinality_cols_
        ]

        self.ohe_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.ordinal_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        if self.low_cardinality_cols_:
            self.ohe_.fit(
                X[self.low_cardinality_cols_].astype(str).fillna("missing")
            )

        if self.high_cardinality_cols_:
            self.ordinal_.fit(
                X[self.high_cardinality_cols_].astype(str).fillna("missing")
            )

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.columns_:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.columns_]

        parts = []

        if self.numeric_cols_:
            numeric_part = (
                X[self.numeric_cols_]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .to_numpy()
            )
            parts.append(numeric_part)

        if self.low_cardinality_cols_:
            low_part = self.ohe_.transform(
                X[self.low_cardinality_cols_].astype(str).fillna("missing")
            )
            parts.append(low_part)

        if self.high_cardinality_cols_:
            high_part = self.ordinal_.transform(
                X[self.high_cardinality_cols_].astype(str).fillna("missing")
            )
            parts.append(high_part)

        if not parts:
            return np.empty((len(X), 0))

        return np.hstack(parts)