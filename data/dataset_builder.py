from typing import Optional

import numpy as np
import pandas as pd

from data.loader import load_active_dishes, load_sales_raw


FINAL_DATASET_COLUMNS = [
    "dish_id",
    "dish_name",
    "dish_category",
    "week_start",
    "y",
    "price",
    "avg_price_last4",
    "seasonal_factor_weekly",
]


def get_week_start(series: pd.Series) -> pd.Series:
    """
    Convierte una fecha cualquiera al lunes de su semana.
    """
    dates = pd.to_datetime(series)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def build_weekly_sales_dataset(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_missing_weeks: bool = True,
) -> pd.DataFrame:
    """
    Construye el dataset semanal de demanda para entrenamiento ML.

    Salida final:
    - dish_id
    - dish_name
    - dish_category
    - week_start
    - y
    - price
    - avg_price_last4
    - seasonal_factor_weekly
    """

    sales_df = load_sales_raw(start_date=start_date, end_date=end_date)
    dishes_df = load_active_dishes()

    if dishes_df.empty:
        raise ValueError("No existen platos activos registrados en la base de datos")

    if sales_df.empty:
        raise ValueError("No existen ventas históricas para construir el dataset de entrenamiento")

    sales_df = sales_df.copy()
    sales_df["week_start"] = get_week_start(sales_df["sale_date"])

    weekly_df = aggregate_weekly_sales(sales_df)

    if fill_missing_weeks:
        weekly_df = complete_missing_weeks(weekly_df, dishes_df)

    weekly_df = add_price_features(weekly_df)
    weekly_df = add_seasonal_factor(weekly_df)

    weekly_df = weekly_df[FINAL_DATASET_COLUMNS]
    weekly_df = weekly_df.sort_values(["dish_id", "week_start"]).reset_index(drop=True)

    return weekly_df


def aggregate_weekly_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa ventas por platillo y semana.
    Calcula:
    - y: cantidad total vendida
    - price: precio promedio ponderado por cantidad
    """

    def aggregate_group(group: pd.DataFrame) -> pd.Series:
        total_y = float(group["quantity"].sum())

        valid_prices = group["unit_price"].notna()

        if total_y > 0 and valid_prices.any():
            price = np.average(
                group.loc[valid_prices, "unit_price"].astype(float),
                weights=group.loc[valid_prices, "quantity"].astype(float),
            )
        else:
            price = group["unit_price"].mean()

        return pd.Series(
            {
                "y": total_y,
                "price": float(price) if pd.notna(price) else np.nan,
            }
        )

    weekly_df = (
        sales_df
        .groupby(
            ["dish_id", "dish_name", "dish_category", "week_start"],
            as_index=False,
            sort=False,
        )
        .apply(aggregate_group)
        .reset_index(drop=True)
    )

    weekly_df["y"] = pd.to_numeric(weekly_df["y"], errors="coerce").fillna(0).astype(float)
    weekly_df["price"] = pd.to_numeric(weekly_df["price"], errors="coerce")

    return weekly_df


def complete_missing_weeks(weekly_df: pd.DataFrame, dishes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una grilla completa:
    todos los platos activos × todas las semanas del histórico.

    Las semanas sin ventas quedan con y = 0.
    """

    min_week = weekly_df["week_start"].min()
    max_week = weekly_df["week_start"].max()

    all_weeks = pd.date_range(start=min_week, end=max_week, freq="W-MON")

    base_grid = (
        dishes_df.assign(key=1)
        .merge(pd.DataFrame({"week_start": all_weeks, "key": 1}), on="key")
        .drop(columns=["key"])
    )

    merged = base_grid.merge(
        weekly_df,
        on=["dish_id", "week_start"],
        how="left",
        suffixes=("", "_sales"),
    )

    merged["dish_name"] = merged["dish_name_sales"].fillna(merged["dish_name"])
    merged["dish_category"] = merged["dish_category_sales"].fillna(merged["dish_category"])

    merged = merged.drop(columns=["dish_name_sales", "dish_category_sales"], errors="ignore")

    merged["y"] = merged["y"].fillna(0).astype(float)

    merged["price"] = merged["price"].fillna(merged["dish_price"])
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce").fillna(0).astype(float)

    return merged[
        [
            "dish_id",
            "dish_name",
            "dish_category",
            "week_start",
            "y",
            "price",
        ]
    ]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega precio promedio de las 4 semanas anteriores.

    Usa shift(1) para evitar fuga de información.
    """

    df = df.sort_values(["dish_id", "week_start"]).copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["price"] = (
        df.groupby("dish_id")["price"]
        .transform(lambda s: s.ffill().bfill())
    )

    df["avg_price_last4"] = (
        df.groupby("dish_id")["price"]
        .transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
    )

    df["avg_price_last4"] = df["avg_price_last4"].fillna(df["price"])

    return df


def add_seasonal_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula factor estacional mensual con base en la demanda promedio mensual.

    seasonal_factor_weekly = promedio mensual / promedio global
    """

    df = df.copy()
    df["month"] = df["week_start"].dt.month

    monthly = (
        df.groupby("month", as_index=False)["y"]
        .mean()
        .rename(columns={"y": "month_mean"})
    )

    overall_mean = monthly["month_mean"].mean()

    if pd.isna(overall_mean) or overall_mean == 0:
        overall_mean = 1.0

    monthly["seasonal_factor_weekly"] = monthly["month_mean"] / overall_mean

    df = df.merge(
        monthly[["month", "seasonal_factor_weekly"]],
        on="month",
        how="left",
    )

    df["seasonal_factor_weekly"] = df["seasonal_factor_weekly"].fillna(1.0)

    df = df.drop(columns=["month"])

    return df


def filter_active_demand_items(df: pd.DataFrame, min_active_ratio: float = 0.20) -> pd.DataFrame:
    """
    Filtra platos con actividad mínima.

    active_ratio = porcentaje de semanas donde y > 0.
    """

    activity = df.groupby("dish_id")["y"].apply(lambda s: float(np.mean(s > 0)))
    keep_ids = activity[activity >= min_active_ratio].index

    return df[df["dish_id"].isin(keep_ids)].reset_index(drop=True)