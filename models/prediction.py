import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from data.database import get_engine
from data.dataset_builder import build_weekly_sales_dataset, filter_active_demand_items
from features.feature_pipeline import prepare_feature_dataset
from features.temporal_features import add_temporal_features
from features.lag_features import add_lag_and_rolling_features
from models.registry import resolve_user_id
from models.xgboost_model import clip_negative_predictions


MODEL_NAME = "xgboost_demand_forecasting"


def get_active_model_record(model_name: str = MODEL_NAME) -> Dict[str, Any]:
    """
    Obtiene el modelo activo desde la tabla ml_models.
    """
    engine = get_engine()

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    id::text AS id,
                    model_name,
                    model_type,
                    version,
                    parameters,
                    mae,
                    rmse,
                    r2,
                    trained_at
                FROM ml_models
                WHERE model_name = :model_name
                  AND is_active = true
                ORDER BY trained_at DESC
                LIMIT 1
                """
            ),
            {"model_name": model_name},
        ).mappings().first()

    if row is None:
        raise ValueError(f"No existe un modelo activo para model_name='{model_name}'")

    record = dict(row)

    params = record.get("parameters")

    if isinstance(params, str):
        record["parameters"] = json.loads(params)

    return record


def load_active_model_artifact(model_name: str = MODEL_NAME) -> Dict[str, Any]:
    """
    Carga el artifact .joblib del modelo activo.
    """
    model_record = get_active_model_record(model_name=model_name)
    parameters = model_record.get("parameters") or {}

    artifact_path = parameters.get("artifact_path")

    if not artifact_path:
        raise ValueError("El modelo activo no tiene artifact_path en parameters")

    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {path}")

    artifact = joblib.load(path)

    return {
        "model_record": model_record,
        "artifact": artifact,
    }


def predict_future_demand(
    weeks_ahead: int = 4,
    dish_id: Optional[str] = None,
    save_to_db: bool = False,
    created_by: str = "admin",
) -> Dict[str, Any]:
    """
    Genera predicciones futuras de demanda usando el modelo activo.

    Parámetros:
    - weeks_ahead: número de semanas futuras a predecir.
    - dish_id: si se envía, predice solo para ese platillo.
    - save_to_db: si es True, guarda en la tabla predictions.
    - created_by: username/email/UUID del usuario que genera la predicción.
    """

    if weeks_ahead <= 0:
        raise ValueError("weeks_ahead debe ser mayor a cero")

    loaded = load_active_model_artifact()
    model_record = loaded["model_record"]
    artifact = loaded["artifact"]

    model = artifact["model"]

    weekly_df = build_weekly_sales_dataset(fill_missing_weeks=True)
    weekly_df = filter_active_demand_items(weekly_df)

    if dish_id:
        weekly_df = weekly_df[weekly_df["dish_id"].astype(str) == str(dish_id)].copy()

        if weekly_df.empty:
            raise ValueError(f"No existen datos históricos suficientes para dish_id={dish_id}")

    predictions_df = generate_recursive_predictions(
        model=model,
        historical_df=weekly_df,
        weeks_ahead=weeks_ahead,
        model_record=model_record,
    )

    if save_to_db:
        save_predictions_to_database(
            predictions_df=predictions_df,
            model_id=model_record["id"],
            created_by=created_by,
        )

    return {
        "status": "completed",
        "model_id": model_record["id"],
        "model_name": model_record["model_name"],
        "model_version": model_record["version"],
        "weeks_ahead": weeks_ahead,
        "dish_id": dish_id,
        "saved_to_db": save_to_db,
        "predictions": predictions_df.to_dict(orient="records"),
    }


def generate_recursive_predictions(
    model,
    historical_df: pd.DataFrame,
    weeks_ahead: int,
    model_record: Dict[str, Any],
) -> pd.DataFrame:
    """
    Genera predicciones futuras de forma recursiva.

    Esto es necesario porque para predecir la semana 2 futura,
    necesitamos usar como historial la predicción de la semana 1 futura.
    """

    if historical_df.empty:
        raise ValueError("El dataset histórico está vacío")

    historical_df = historical_df.copy()
    historical_df["week_start"] = pd.to_datetime(historical_df["week_start"])
    historical_df = historical_df.sort_values(["dish_id", "week_start"]).reset_index(drop=True)

    last_week = historical_df["week_start"].max()
    future_weeks = [
        last_week + pd.Timedelta(weeks=i)
        for i in range(1, weeks_ahead + 1)
    ]

    working_df = historical_df.copy()
    all_predictions: List[Dict[str, Any]] = []

    rmse = model_record.get("rmse")

    for future_week in future_weeks:
        future_rows = build_future_rows_for_week(
            historical_or_working_df=working_df,
            future_week=future_week,
        )

        combo_df = pd.concat([working_df, future_rows], ignore_index=True)
        combo_features = prepare_future_feature_dataset(combo_df)

        current_features = combo_features[
            combo_features["week_start"] == future_week
        ].copy()

        if current_features.empty:
            continue

        X_future = current_features.drop(columns=["y", "week_start"], errors="ignore")

        y_pred = model.predict(X_future)
        y_pred = clip_negative_predictions(y_pred)

        current_features["predicted_quantity"] = y_pred

        for _, row in current_features.iterrows():
            predicted_quantity = float(row["predicted_quantity"])

            confidence = estimate_confidence(
                predicted_quantity=predicted_quantity,
                rmse=rmse,
            )

            trend_factor = estimate_trend_factor(
                dish_id=str(row["dish_id"]),
                historical_df=working_df,
                predicted_quantity=predicted_quantity,
            )

            all_predictions.append(
                {
                    "dish_id": str(row["dish_id"]),
                    "dish_name": str(row["dish_name"]),
                    "predicted_date": future_week.date().isoformat(),
                    "predicted_quantity": int(round(predicted_quantity)),
                    "predicted_quantity_raw": predicted_quantity,
                    "confidence_level": confidence,
                    "seasonal_factor": float(row.get("seasonal_factor_weekly", 1.0)),
                    "trend_factor": trend_factor,
                    "weather_factor": "normal",
                }
            )

        # Agregamos las predicciones al historial de trabajo para las siguientes semanas.
        predicted_history_rows = future_rows.copy()

        prediction_map = {
            item["dish_id"]: item["predicted_quantity_raw"]
            for item in all_predictions
            if item["predicted_date"] == future_week.date().isoformat()
        }

        predicted_history_rows["y"] = predicted_history_rows["dish_id"].map(prediction_map).fillna(0)

        working_df = pd.concat([working_df, predicted_history_rows], ignore_index=True)
        working_df = working_df.sort_values(["dish_id", "week_start"]).reset_index(drop=True)

    return pd.DataFrame(all_predictions)


def build_future_rows_for_week(
    historical_or_working_df: pd.DataFrame,
    future_week: pd.Timestamp,
) -> pd.DataFrame:
    """
    Construye filas futuras para cada platillo.

    Para la semana futura:
    - y se coloca en 0 temporalmente.
    - price toma el último precio conocido.
    - avg_price_last4 toma promedio de las últimas 4 semanas conocidas.
    - seasonal_factor_weekly se estima según el mes histórico.
    """

    df = historical_or_working_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])

    rows = []

    seasonal_by_month = calculate_seasonal_factor_by_month(df)
    month = int(future_week.month)
    seasonal_factor = float(seasonal_by_month.get(month, 1.0))

    for dish_id, group in df.groupby("dish_id"):
        group = group.sort_values("week_start")

        last_row = group.iloc[-1]

        last_price = float(last_row.get("price", 0.0))

        last4_price = (
            pd.to_numeric(group["price"], errors="coerce")
            .tail(4)
            .mean()
        )

        if pd.isna(last4_price):
            last4_price = last_price

        rows.append(
            {
                "dish_id": str(dish_id),
                "dish_name": str(last_row["dish_name"]),
                "dish_category": str(last_row["dish_category"]),
                "week_start": future_week,
                "y": 0.0,
                "price": last_price,
                "avg_price_last4": float(last4_price),
                "seasonal_factor_weekly": seasonal_factor,
            }
        )

    return pd.DataFrame(rows)


def prepare_future_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara features para predicción futura.

    A diferencia del entrenamiento, no eliminamos completamente el futuro,
    solo necesitamos que la fila futura tenga lags disponibles.
    """

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["dish_id", "week_start"]).reset_index(drop=True)

    # Features temporales
    df = add_temporal_features(df, date_col="week_start")

    # Usamos lags conservadores para predicción, alineados con entrenamiento dinámico.
    lags = [1, 2, 3, 4, 8, 12]
    rolling_windows = [4, 8, 12]

    max_weeks = int(df.groupby("dish_id")["week_start"].nunique().median())

    lags = [lag for lag in lags if lag < max_weeks]
    rolling_windows = [w for w in rolling_windows if w < max_weeks]

    if not lags:
        lags = [1]

    if not rolling_windows:
        rolling_windows = [2]

    df = add_lag_and_rolling_features(
        df,
        group_col="dish_id",
        date_col="week_start",
        target_col="y",
        lags=lags,
        rolling_windows=rolling_windows,
    )

    # Normalización de tipos
    df["dish_id"] = df["dish_id"].astype(str)
    df["dish_name"] = df["dish_name"].astype(str)
    df["dish_category"] = df["dish_category"].fillna("Sin categoría").astype(str)

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
        col for col in df.columns
        if col.startswith("y_lag_") or col.startswith("y_roll_")
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def calculate_seasonal_factor_by_month(df: pd.DataFrame) -> Dict[int, float]:
    """
    Calcula factor estacional mensual a partir del histórico.
    """

    temp = df.copy()
    temp["week_start"] = pd.to_datetime(temp["week_start"])
    temp["month"] = temp["week_start"].dt.month

    monthly = (
        temp.groupby("month", as_index=False)
        .agg(month_mean=("y", "mean"))
    )

    overall = monthly["month_mean"].mean()

    if pd.isna(overall) or overall == 0:
        overall = 1.0

    monthly["factor"] = monthly["month_mean"] / overall

    return {
        int(row["month"]): float(row["factor"])
        for _, row in monthly.iterrows()
    }


def estimate_confidence(
    predicted_quantity: float,
    rmse: Optional[float],
) -> float:
    """
    Estima una confianza aproximada usando el RMSE del modelo.

    No es una probabilidad estadística exacta, pero funciona como indicador
    operativo para frontend y reportes.
    """

    if rmse is None or predicted_quantity <= 0:
        return 0.70

    confidence = 1.0 - (float(rmse) / (predicted_quantity + float(rmse) + 1.0))

    confidence = max(0.50, min(0.95, confidence))

    return round(confidence, 4)


def estimate_trend_factor(
    dish_id: str,
    historical_df: pd.DataFrame,
    predicted_quantity: float,
) -> float:
    """
    Calcula un factor de tendencia simple comparando la predicción
    contra el promedio reciente de las últimas 4 semanas.
    """

    group = historical_df[historical_df["dish_id"].astype(str) == str(dish_id)].copy()

    if group.empty:
        return 1.0

    recent_mean = pd.to_numeric(group["y"], errors="coerce").tail(4).mean()

    if pd.isna(recent_mean) or recent_mean == 0:
        return 1.0

    trend = predicted_quantity / recent_mean

    return round(float(trend), 4)


def save_predictions_to_database(
    predictions_df: pd.DataFrame,
    model_id: str,
    created_by: str = "admin",
) -> None:
    """
    Guarda predicciones en la tabla predictions.

    Antes de insertar, elimina predicciones anteriores del mismo:
    - model_id
    - dish_id
    - rango de fechas predicho

    Esto evita duplicados sin borrar predicciones de otros platos o periodos.
    """

    if predictions_df.empty:
        return

    created_by_id = resolve_user_id(created_by)

    if created_by_id is None:
        raise ValueError(
            f"No se encontró usuario válido para created_by='{created_by}'"
        )

    engine = get_engine()

    dish_ids = predictions_df["dish_id"].astype(str).unique().tolist()
    min_date = predictions_df["predicted_date"].min()
    max_date = predictions_df["predicted_date"].max()

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM predictions
                WHERE model_id = CAST(:model_id AS uuid)
                AND dish_id::text = ANY(:dish_ids)
                AND predicted_date BETWEEN :min_date AND :max_date
                """
            ),
            {
                "model_id": model_id,
                "dish_ids": dish_ids,
                "min_date": min_date,
                "max_date": max_date,
            },
        )
        for _, row in predictions_df.iterrows():
            connection.execute(
                text(
                    """
                    INSERT INTO predictions (
                        id,
                        model_id,
                        dish_id,
                        predicted_date,
                        predicted_quantity,
                        confidence_level,
                        weather_factor,
                        seasonal_factor,
                        trend_factor,
                        created_by,
                        created_at
                    )
                    VALUES (
                        gen_random_uuid(),
                        CAST(:model_id AS uuid),
                        CAST(:dish_id AS uuid),
                        :predicted_date,
                        :predicted_quantity,
                        :confidence_level,
                        :weather_factor,
                        :seasonal_factor,
                        :trend_factor,
                        CAST(:created_by AS uuid),
                        :created_at
                    )
                    """
                ),
                {
                    "model_id": model_id,
                    "dish_id": row["dish_id"],
                    "predicted_date": row["predicted_date"],
                    "predicted_quantity": int(row["predicted_quantity"]),
                    "confidence_level": float(row["confidence_level"]),
                    "weather_factor": row.get("weather_factor", "normal"),
                    "seasonal_factor": float(row.get("seasonal_factor", 1.0)),
                    "trend_factor": float(row.get("trend_factor", 1.0)),
                    "created_by": created_by_id,
                    "created_at": datetime.now(),
                },
            )