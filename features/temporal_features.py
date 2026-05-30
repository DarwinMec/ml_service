import numpy as np
import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "week_start",
) -> pd.DataFrame:
    """
    Agrega variables temporales al dataset semanal.

    Features:
    - weekofyear
    - month
    - year
    - week_idx
    - week_sin
    - week_cos
    """

    if date_col not in df.columns:
        raise ValueError(f"No existe la columna de fecha: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["weekofyear"] = out[date_col].dt.isocalendar().week.astype(int)
    out["month"] = out[date_col].dt.month.astype(int)
    out["year"] = out[date_col].dt.year.astype(int)

    min_week = out[date_col].min()
    out["week_idx"] = ((out[date_col] - min_week).dt.days // 7).astype(int)

    out["week_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52.0)

    return out