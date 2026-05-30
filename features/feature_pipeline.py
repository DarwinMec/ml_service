from typing import List, Tuple

import numpy as np
import pandas as pd

from app.config import get_settings
from features.temporal_features import add_temporal_features
from features.lag_features import add_lag_and_rolling_features, get_lag_feature_columns


TARGET_COL = "y"
DATE_COL = "week_start"
GROUP_COL = "dish_id"


def prepare_feature_dataset(
    df: pd.DataFrame,
    drop_na_lags: bool = True,
    use_dynamic_lags: bool = True,
) -> pd.DataFrame:
    """
    Prepara el dataset final para entrenamiento con XGBoost.

    Pasos:
    1. Ordena el dataset.
    2. Agrega features temporales.
    3. Agrega lags y rolling features.
    4. Convierte columnas categóricas.
    5. Elimina filas con lags incompletos si corresponde.
    """

    settings = get_settings()

    if df.empty:
        raise ValueError("El dataset recibido está vacío")

    required_cols = [
        "dish_id",
        "dish_name",
        "dish_category",
        "week_start",
        "y",
        "price",
        "avg_price_last4",
        "seasonal_factor_weekly",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    lags = settings.lag_list
    rolling_windows = settings.rolling_window_list

    if use_dynamic_lags:
        lags, rolling_windows = adjust_lags_to_dataset(
            out,
            lags=lags,
            rolling_windows=rolling_windows,
            group_col=GROUP_COL,
        )

    out = add_temporal_features(out, date_col=DATE_COL)

    out = add_lag_and_rolling_features(
        out,
        group_col=GROUP_COL,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    out = normalize_feature_types(out)

    lag_cols = get_lag_feature_columns(out, target_col=TARGET_COL)

    if drop_na_lags and lag_cols:
        out = out.dropna(subset=lag_cols).reset_index(drop=True)

    # Los std iniciales pueden quedar NaN; los reemplazamos por 0.
    roll_std_cols = [
        col for col in out.columns
        if col.startswith(f"{TARGET_COL}_roll_std_")
    ]

    for col in roll_std_cols:
        out[col] = out[col].fillna(0)

    return out


def adjust_lags_to_dataset(
    df: pd.DataFrame,
    lags: List[int],
    rolling_windows: List[int],
    group_col: str = "dish_id",
) -> Tuple[List[int], List[int]]:
    """
    Ajusta los lags y rolling windows según la cantidad de semanas disponibles.

    Esto evita que un dataset pequeño pierda demasiadas filas por usar lags como 52.
    """

    weeks_per_item = df.groupby(group_col)[DATE_COL].nunique()

    if weeks_per_item.empty:
        return lags, rolling_windows

    median_weeks = int(weeks_per_item.median())

    # Dejamos margen para que queden filas entrenables.
    max_allowed_lag = max(1, median_weeks // 2)

    adjusted_lags = [lag for lag in lags if lag <= max_allowed_lag]
    adjusted_windows = [w for w in rolling_windows if w <= max_allowed_lag]

    if not adjusted_lags:
        adjusted_lags = [1]

    if not adjusted_windows:
        adjusted_windows = [4] if median_weeks >= 4 else [2]

    return adjusted_lags, adjusted_windows


def normalize_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tipos para que el pipeline de XGBoost pueda procesarlos.
    """

    out = df.copy()

    out["dish_id"] = out["dish_id"].astype(str)
    out["dish_name"] = out["dish_name"].astype(str)
    out["dish_category"] = out["dish_category"].fillna("Sin categoría").astype(str)

    numeric_cols = [
        "y",
        "price",
        "avg_price_last4",
        "seasonal_factor_weekly",
        "weekofyear",
        "month",
        "year",
        "week_idx",
        "week_sin",
        "week_cos",
    ]

    numeric_cols += [
        col for col in out.columns
        if col.startswith("y_lag_")
        or col.startswith("y_roll_")
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Separa X e y.

    Excluye:
    - y
    - week_start

    Mantiene dish_id y dish_category como categóricas para el encoder.
    """

    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target: {target_col}")

    drop_cols = [target_col, date_col]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].astype(float).values

    return X, y


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna las columnas finales que entrarán al modelo.
    """

    return [
        col for col in df.columns
        if col not in [TARGET_COL, DATE_COL]
    ]