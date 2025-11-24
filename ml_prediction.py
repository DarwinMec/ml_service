# ml_prediction.py
# Módulo de predicción que carga el modelo entrenado y genera forecasts

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy import text
from sklearn.pipeline import Pipeline

from data_prep_backend import get_engine
from ml_training import (
    add_temporal_features,
    add_lag_roll_features,
)


def get_active_model_id() -> Optional[str]:
    """
    Obtiene el ID del modelo activo más reciente desde ml_models.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id 
                FROM ml_models 
                WHERE is_active = TRUE 
                ORDER BY trained_at DESC 
                LIMIT 1
            """)
        )
        row = result.fetchone()
        return str(row[0]) if row else None


def load_model_from_disk(model_id: str) -> Pipeline:
    """
    Carga el modelo guardado en disco (.joblib).
    """
    models_dir = os.getenv("ML_MODELS_DIR", "models")
    model_path = os.path.join(models_dir, f"{model_id}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    return load(model_path)


def get_latest_data_for_dish(dish_id: str, weeks_needed: int = 52) -> pd.DataFrame:
    """
    Obtiene las últimas N semanas de datos históricos para un plato específico.
    Necesario para calcular lags y rolling features.
    """
    engine = get_engine()

    # Para simplificar, tomamos TODO el histórico del plato
    sql = text("""
        SELECT 
            si.dish_id,
            d.name AS dish_name,
            d.category AS dish_category,
            d.price AS dish_price,
            s.sale_date,
            si.quantity
        FROM sale_items si
        JOIN sales s ON si.sale_id = s.id
        JOIN dishes d ON si.dish_id = d.id
        WHERE d.id = :dish_id
          AND d.is_active = TRUE
        ORDER BY s.sale_date
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"dish_id": dish_id}
        )

    if df.empty:
        return df

    # Procesar a nivel semanal
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["week_start"] = df["sale_date"] - pd.to_timedelta(
        df["sale_date"].dt.weekday, unit="D"
    )
    df["week_start"] = df["week_start"].dt.normalize()

    weekly = (
        df.groupby(["dish_id", "dish_name", "dish_category", "week_start"], as_index=False)
        .agg(
            y=("quantity", "sum"),
            price=("dish_price", "mean")
        )
        .sort_values("week_start")
    )

    # Aseguramos tipos
    weekly["dish_id"] = weekly["dish_id"].astype(str)
    weekly["dish_category"] = weekly["dish_category"].astype(str)

    # Alias para compatibilidad con modelo entrenado en Kaggle:
    # - "meal_id" ~ identificador de plato
    weekly["meal_id"] = weekly["dish_id"]
    # - "category" ~ categoría de plato
    weekly["category"] = weekly["dish_category"]
    # - "cuisine" (si el modelo la espera) — valor fijo genérico
    weekly["cuisine"] = "Unknown"
    # - "center_id" (en tu Colab filtraste center 10)
    weekly["center_id"] = 10

    return weekly


def get_seasonal_factor(month: int) -> float:
    """
    Obtiene el factor estacional para un mes dado desde la tabla seasonal_factors.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT factor 
                FROM seasonal_factors 
                WHERE month = :month 
                  AND day_of_week = 0
                LIMIT 1
            """),
            {"month": month}
        )
        row = result.fetchone()
        return float(row[0]) if row else 1.0


def prepare_prediction_features(
    dish_id: str,
    target_week_start: datetime,
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepara las features necesarias para una predicción específica.

    Args:
        dish_id: ID del plato
        target_week_start: Fecha de inicio de la semana a predecir
        historical_data: DataFrame con datos históricos del plato

    Returns:
        DataFrame de una fila con todas las features necesarias
    """
    if historical_data.empty:
        raise ValueError(f"No hay datos históricos para el plato {dish_id}")

    # Aseguramos alias de columnas esperadas por el modelo
    hist = historical_data.copy()
    hist["dish_id"] = hist["dish_id"].astype(str)
    hist["dish_category"] = hist["dish_category"].astype(str)

    # Alias tipo Kaggle
    hist["meal_id"] = hist["dish_id"]
    if "category" not in hist.columns:
        hist["category"] = hist["dish_category"]
    if "cuisine" not in hist.columns:
        hist["cuisine"] = "Unknown"
    if "center_id" not in hist.columns:
        hist["center_id"] = 10  # center fijo

    last_row = hist.iloc[-1].copy()

    prediction_row = pd.DataFrame([{
        "dish_id": str(dish_id),
        "dish_name": last_row["dish_name"],
        "dish_category": last_row["dish_category"],
        "week_start": target_week_start,
        "y": 0,  # placeholder, no se usa en predicción
        "price": last_row["price"],
        # Alias Kaggle
        "meal_id": str(dish_id),
        "category": last_row.get("category", last_row["dish_category"]),
        "cuisine": last_row.get("cuisine", "Unknown"),
        "center_id": last_row.get("center_id", 10),
    }])

    combined = pd.concat([hist, prediction_row], ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"])

    # Precio promedio móvil 4 semanas
    combined["avg_price_last4"] = (
        combined.groupby("dish_id")["price"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )

    # Factor estacional
    combined["month"] = combined["week_start"].dt.month
    combined["seasonal_factor_weekly"] = combined["month"].apply(get_seasonal_factor)

    # Features temporales
    combined = add_temporal_features(combined)

    # Lags y rolling features
    combined = add_lag_roll_features(combined)

    # Target encoding aproximado:
    # En el modelo original se usaba "meal_id_te".
    mean_y = hist["y"].mean() if not hist.empty else 0.0
    combined["dish_id_te"] = float(mean_y)

    # Aseguramos tipos
    combined["dish_id"] = combined["dish_id"].astype(str)
    combined["dish_category"] = combined["dish_category"].astype(str)
    combined["meal_id"] = combined["meal_id"].astype(str)
    combined["category"] = combined["category"].astype(str)
    combined["cuisine"] = combined["cuisine"].astype(str)

    # Fila final para predicción
    result = combined.iloc[[-1]].copy()

    return result


def predict_single_dish(
    model: Pipeline,
    dish_id: str,
    weeks_ahead: int = 4
) -> List[Dict]:
    """
    Predice la demanda para un plato específico para las próximas N semanas.
    """
    historical = get_latest_data_for_dish(dish_id, weeks_needed=52)

    if historical.empty:
        return []

    print(f"[DEBUG] Plato {dish_id} histórico semanas={len(historical)} "
      f"y_mean={historical['y'].mean() if 'y' in historical.columns else 'NA'}")

    predictions: List[Dict] = []

    last_week = historical["week_start"].max()
    current_historical = historical.copy()

    for week_offset in range(1, weeks_ahead + 1):
        target_week = last_week + timedelta(weeks=week_offset)

        try:
            X_pred = prepare_prediction_features(
                dish_id=dish_id,
                target_week_start=target_week,
                historical_data=current_historical
            )

            # Eliminar columnas que el modelo NO usa como target
            X_pred_clean = X_pred.drop(columns=["y", "week_start"], errors="ignore")

            # ⚠️ Truco de seguridad:
            # Nos aseguramos de que existan las columnas más probables
            # que el modelo espera (no pasa nada si hay columnas extra).
            for col in ["dish_id_te", "meal_id", "category", "cuisine", "center_id"]:
                if col not in X_pred_clean.columns:
                    if col == "dish_id_te":
                        # usamos el promedio histórico del plato como target encoding
                        X_pred_clean[col] = float(current_historical["y"].mean())
                    elif col == "center_id":
                        X_pred_clean[col] = 10
                    elif col == "category":
                        X_pred_clean[col] = current_historical.iloc[-1]["dish_category"]
                    elif col == "cuisine":
                        X_pred_clean[col] = "Unknown"
                    elif col == "meal_id":
                        X_pred_clean[col] = dish_id
            
            raw_pred = model.predict(X_pred_clean)[0]
            print(f"[DEBUG] Plato {dish_id} semana {week_offset} → raw_pred={raw_pred}")
            y_pred = max(0.0, float(raw_pred))

            predictions.append({
                "dish_id": dish_id,
                "dish_name": current_historical.iloc[-1]["dish_name"],
                "week_start": target_week.strftime("%Y-%m-%d"),
                "predicted_demand": round(y_pred, 2),
                "confidence": "0.80",  # numérico en BD; se puede ajustar
            })

            # Actualizar histórico con la predicción
            new_row = pd.DataFrame([{
                "dish_id": dish_id,
                "dish_name": current_historical.iloc[-1]["dish_name"],
                "dish_category": current_historical.iloc[-1]["dish_category"],
                "week_start": target_week,
                "y": y_pred,
                "price": current_historical.iloc[-1]["price"],
                # Alias Kaggle
                "meal_id": dish_id,
                "category": current_historical.iloc[-1]["dish_category"],
                "cuisine": "Unknown",
                "center_id": 10,
            }])

            current_historical = pd.concat(
                [current_historical, new_row],
                ignore_index=True
            )

        except Exception as e:
            print(f"Error prediciendo semana {week_offset} para plato {dish_id}: {e}")
            continue

    return predictions


def predict_all_active_dishes(
    model: Pipeline,
    weeks_ahead: int = 4
) -> List[Dict]:
    """
    Predice demanda para todos los platos activos.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id FROM dishes WHERE is_active = TRUE")
        )
        dish_ids = [str(row[0]) for row in result.fetchall()]

    all_predictions: List[Dict] = []

    for dish_id in dish_ids:
        try:
            preds = predict_single_dish(model, dish_id, weeks_ahead)
            all_predictions.extend(preds)
        except Exception as e:
            print(f"Error prediciendo plato {dish_id}: {e}")
            continue

    return all_predictions


def save_predictions_to_db(predictions: List[Dict], model_id: str) -> None:
    """
    Guarda las predicciones en la tabla predictions (tu tabla real de BD).
    """
    if not predictions:
        print("⚠️ No hay predicciones para guardar")
        return

    engine = get_engine()

    with engine.begin() as conn:
        # Limpiar predicciones antiguas del mismo modelo
        conn.execute(
            text("DELETE FROM predictions WHERE model_id = :model_id"),
            {"model_id": model_id}
        )

        insert_sql = text("""
            INSERT INTO predictions (
                id,
                confidence_level,
                created_at,
                predicted_date,
                predicted_quantity,
                seasonal_factor,
                trend_factor,
                weather_factor,
                created_by,
                dish_id,
                model_id
            )
            VALUES (
                :id,
                :confidence_level,
                NOW(),
                :predicted_date,
                :predicted_quantity,
                :seasonal_factor,
                :trend_factor,
                :weather_factor,
                :created_by,
                :dish_id,
                :model_id
            )
        """)

        for pred in predictions:
            conn.execute(
                insert_sql,
                {
                    "id": str(uuid.uuid4()),
                    "confidence_level": float(pred.get("confidence", 0.8)),
                    "predicted_date": pred["week_start"],
                    "predicted_quantity": int(round(pred["predicted_demand"])),
                    "seasonal_factor": 1.0,
                    "trend_factor": 1.0,
                    "weather_factor": "normal",
                    "created_by": None,
                    "dish_id": pred["dish_id"],
                    "model_id": model_id,
                }
            )

    print(f"✅ {len(predictions)} predicciones guardadas en la BD (tabla predictions)")


def generate_predictions(
    dish_id: Optional[str] = None,
    weeks_ahead: int = 4,
    save_to_db: bool = True
) -> List[Dict]:
    """
    Función principal para generar predicciones.
    """
    print("🔮 Iniciando proceso de predicción...")

    model_id = get_active_model_id()
    if not model_id:
        raise RuntimeError("No hay ningún modelo activo en la BD")

    print(f"📦 Cargando modelo: {model_id}")

    model = load_model_from_disk(model_id)

    if dish_id:
        predictions = predict_single_dish(model, dish_id, weeks_ahead)
    else:
        predictions = predict_all_active_dishes(model, weeks_ahead)

    print(f"✅ Generadas {len(predictions)} predicciones")

    if save_to_db:
        save_predictions_to_db(predictions, model_id)

    return predictions
