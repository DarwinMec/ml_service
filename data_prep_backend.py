# data_prep_backend.py
# Construye el dataset semanal por plato a partir de la BD PostgreSQL
# y actualiza la tabla seasonal_factors.

import os
import uuid
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# 👉 Cambia esto o usa variables de entorno
DB_URL = os.getenv(
    "ML_DB_URL",
    "postgresql+psycopg2://postgres:1234@localhost:5432/db_TP1test2"
)


def get_engine() -> Engine:
    """Devuelve un engine de SQLAlchemy para conectarse a PostgreSQL."""
    return create_engine(DB_URL)


def load_raw_sales(
    engine: Engine,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Lee ventas + ítems + platos desde la BD.

    - start_date / end_date en formato 'YYYY-MM-DD' (opcionales).
    - Usa solo platos activos (dishes.is_active = true).
    """
    params = {}
    date_filter = ""

    if start_date is not None:
        date_filter += " AND s.sale_date >= :start_date"
        params["start_date"] = start_date
    if end_date is not None:
        date_filter += " AND s.sale_date <= :end_date"
        params["end_date"] = end_date

    sql = f"""
    SELECT
        si.dish_id,
        d.name        AS dish_name,
        d.category    AS dish_category,
        d.price       AS dish_price,
        s.sale_date,
        s.day_of_week,
        s.is_weekend,
        s.is_holiday,
        s.weather,
        si.quantity
    FROM sale_items si
    JOIN sales   s ON si.sale_id = s.id
    JOIN dishes  d ON si.dish_id = d.id
    WHERE d.is_active = TRUE
      {date_filter}
    """

    df = pd.read_sql(sql, engine, params=params or None)

    if df.empty:
        print("⚠️ No se encontraron ventas para el rango indicado.")
        return df

    # Normalizar tipos
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["dish_price"] = pd.to_numeric(df["dish_price"], errors="coerce")

    return df

def build_weekly_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partir de ventas diarias, construye un dataset semanal por plato
    similar al CSV limpio de Kaggle.
    Devuelve:
      - df_final: dataset semanal por plato
      - monthly: factores mensuales (para seasonal_factors)
    """

    if df_raw.empty:
        return df_raw.copy(), pd.DataFrame()

    df = df_raw.copy()

    # Lunes de la semana como "week_start" (pandas weekday: lunes=0,...,domingo=6)
    df["week_start"] = df["sale_date"] - pd.to_timedelta(
        df["sale_date"].dt.weekday, unit="D"
    )
    df["week_start"] = df["week_start"].dt.normalize()

    # --- Agregación semanal por plato ---
    weekly = (
        df.groupby(
            ["dish_id", "dish_name", "dish_category", "week_start"],
            as_index=False
        )
        .agg(
            y=("quantity", "sum"),
            price=("dish_price", "mean")  # precio promedio en la semana
        )
        .sort_values(["dish_id", "week_start"])
        .reset_index(drop=True)
    )

    # Limpieza básica
    weekly["y"] = pd.to_numeric(weekly["y"], errors="coerce").fillna(0).astype(float)
    weekly["price"] = pd.to_numeric(weekly["price"], errors="coerce")

    # Si faltan precios, los rellenamos con la media global por plato
    price_by_dish = weekly.groupby("dish_id")["price"].transform(
        lambda s: s.fillna(s.mean())
    )
    weekly["price"] = weekly["price"].fillna(price_by_dish)
    global_price_mean = weekly["price"].mean()
    weekly["price"] = weekly["price"].fillna(global_price_mean).clip(lower=0)

    # --- Precio promedio móvil 4 semanas ---
    weekly["avg_price_last4"] = (
        weekly.groupby("dish_id")["price"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )

    # --- Factor estacional semanal (versión por mes, como en Kaggle) ---
    weekly["month"] = weekly["week_start"].dt.month

    monthly = (
        weekly
        .groupby("month", as_index=False)
        .agg(m_mean=("y", "mean"))
    )

    overall = monthly["m_mean"].mean()
    overall = 1.0 if overall == 0 else overall
    monthly["seasonal_factor_weekly"] = monthly["m_mean"] / overall
    monthly = monthly[["month", "seasonal_factor_weekly"]]

    weekly = weekly.merge(monthly, on="month", how="left")
    weekly["seasonal_factor_weekly"] = weekly["seasonal_factor_weekly"].fillna(1.0)

    # Dataset final estilo df_final de tu Colab
    final_cols = [
        "dish_id",
        "dish_name",
        "dish_category",
        "week_start",
        "y",
        "price",
        "avg_price_last4",
        "seasonal_factor_weekly",
    ]
    df_final = (
        weekly[final_cols]
        .sort_values(["dish_id", "week_start"])
        .reset_index(drop=True)
    )

    return df_final, monthly

def update_seasonal_factors_table(
    engine: Engine,
    monthly_factors: pd.DataFrame
) -> None:
    """
    Actualiza la tabla seasonal_factors con los factores mensuales calculados.

    👉 NOTA DE DISEÑO:
       - Como tu tabla exige (month, day_of_week) NOT NULL, por ahora
         usamos day_of_week = 0 para indicar "factor mensual global".
       - Más adelante, si quieres factores por día de la semana,
         podemos refinar este cálculo.
    """
    if monthly_factors.empty:
        print("⚠️ monthly_factors está vacío. No se actualizará seasonal_factors.")
        return

    monthly_factors = monthly_factors.copy()

    with engine.begin() as conn:
        # Puedes cambiar DELETE por un UPDATE más fino si quieres mantener histórico
        conn.execute(text("DELETE FROM seasonal_factors"))

        insert_sql = text("""
            INSERT INTO seasonal_factors (
                id, created_at, month, day_of_week, factor, description
            )
            VALUES (
                :id, NOW(), :month, :dow, :factor, :description
            )
        """)

        for _, row in monthly_factors.iterrows():
            conn.execute(
                insert_sql,
                {
                    "id": str(uuid.uuid4()),
                    "month": int(row["month"]),
                    "dow": 0,  # 0 = factor mensual agregado
                    "factor": float(row["seasonal_factor_weekly"]),
                    "description": "Monthly seasonal factor computed from weekly dish demand",
                },
            )

    print("✅ seasonal_factors actualizado con factores mensuales.")


def build_training_dataset_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    export_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Punto de entrada principal:
      1. Conecta a la BD.
      2. Carga ventas diarias.
      3. Construye el dataset semanal por plato.
      4. Actualiza seasonal_factors en la BD.
      5. (Opcional) Exporta a CSV.
      6. Devuelve el DataFrame final.
    """
    engine = get_engine()

    df_raw = load_raw_sales(engine, start_date=start_date, end_date=end_date)
    if df_raw.empty:
        return df_raw

    df_final, monthly = build_weekly_dataset(df_raw)

    # Actualizar seasonal_factors
    update_seasonal_factors_table(engine, monthly)

    if export_csv_path is not None:
        df_final.to_csv(export_csv_path, index=False, encoding="utf-8")
        print(f"💾 Dataset semanal exportado a: {export_csv_path}")

    return df_final


if __name__ == "__main__":
    # Ejemplo de uso directo (puedes ajustarlo a tu rango de fechas real)
    df_train = build_training_dataset_from_db(
        start_date=None,   # por ejemplo "2024-01-01"
        end_date=None,     # por ejemplo "2025-12-31"
        export_csv_path="weekly_demand_from_backend.csv"
    )
    print("Shape final:", df_train.shape)
    print(df_train.head())
