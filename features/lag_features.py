from typing import List

import pandas as pd


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    group_col: str = "dish_id",
    date_col: str = "week_start",
    target_col: str = "y",
    lags: List[int] | None = None,
    rolling_windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Agrega lags y rolling features por platillo.

    Importante:
    - Los lags usan shift(lag).
    - Los rolling usan shift(1) para evitar fuga de información.
    """

    if lags is None:
        lags = [1, 2, 3, 4, 8, 12, 26, 52]

    if rolling_windows is None:
        rolling_windows = [4, 8, 12]

    required_cols = [group_col, date_col, target_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"No existe la columna requerida: {col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([group_col, date_col]).reset_index(drop=True)

    def apply_group_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(date_col).copy()

        for lag in lags:
            group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)

        for window in rolling_windows:
            shifted = group[target_col].shift(1)
            group[f"{target_col}_roll_mean_{window}"] = shifted.rolling(
                window=window,
                min_periods=1,
            ).mean()
            group[f"{target_col}_roll_std_{window}"] = shifted.rolling(
                window=window,
                min_periods=2,
            ).std()

        return group

    out = (
        out.groupby(group_col, group_keys=False)
        .apply(apply_group_features)
        .reset_index(drop=True)
    )

    return out


def get_lag_feature_columns(df: pd.DataFrame, target_col: str = "y") -> list[str]:
    """
    Retorna las columnas de lags y rolling features.
    """

    return [
        col for col in df.columns
        if col.startswith(f"{target_col}_lag_")
        or col.startswith(f"{target_col}_roll_")
    ]