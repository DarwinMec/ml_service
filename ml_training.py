# ml_training.py

import os
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from joblib import dump
from sqlalchemy import text
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from data_prep_backend import (
    build_training_dataset_from_db,
    get_engine,
)

# ==========================
# 1. Constantes y helpers de métricas
# ==========================

SEED = 42
np.random.seed(SEED)

# Lags y ventanas de rolling (similar a tu notebook)
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
ROLL_WINS = [4, 8, 12]


def rmse_safe(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_pos(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)


def metrics_on_positive(y_true, y_pred) -> Dict:
    """Métricas solo en semanas con demanda > 0 (como en tu Colab)."""
    mask = np.asarray(y_true) > 0
    if mask.sum() < 2:
        return dict(
            MAE=np.nan,
            RMSE=np.nan,
            R2=np.nan,
            MAPE=np.nan,
            n_pos=int(mask.sum()),
            n_total=int(len(y_true)),
        )
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]
    return dict(
        MAE=mean_absolute_error(yt, yp),
        RMSE=rmse_safe(yt, yp),
        R2=r2_score(yt, yp),
        MAPE=mape_pos(yt, yp),
        n_pos=int(mask.sum()),
        n_total=int(len(y_true)),
    )


# ==========================
# 2. Encoder automático categórico (igual que en tu notebook)
# ==========================

class AutoCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    - Variables numéricas → se dejan tal cual (solo se apilan).
    - Categóricas con <= max_ohe_cards → OneHotEncoder.
    - Categóricas con > max_ohe_cards → OrdinalEncoder.
    """
    def __init__(self, max_ohe_cards: int = 10):
        self.max_ohe_cards = max_ohe_cards

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns.tolist()

        # Detectar categóricas
        cat = [
            c for c in self.columns_
            if isinstance(X[c].dtype, CategoricalDtype) or X[c].dtype == "object"
        ]


        self.low_ = [c for c in cat if X[c].nunique(dropna=False) <= self.max_ohe_cards]
        self.high_ = [c for c in cat if c not in self.low_]

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ord_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        if self.low_:
            self.ohe_.fit(X[self.low_])
        if self.high_:
            self.ord_.fit(X[self.high_])

        # Numéricas = resto
        self.num_ = [c for c in self.columns_ if c not in (self.low_ + self.high_)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        parts = []
        if self.num_:
            parts.append(X[self.num_].to_numpy())
        if self.low_:
            parts.append(self.ohe_.transform(X[self.low_]))
        if self.high_:
            parts.append(self.ord_.transform(X[self.high_]))
        if not parts:
            return np.empty((len(X), 0))
        return np.hstack(parts)


# ==========================
# 3. Features temporales + lags/rolling
# ==========================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega weekofyear, month, year, week_idx y codificación cíclica."""
    df = df.copy()
    df["weekofyear"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month.astype(int)
    df["year"] = df["week_start"].dt.year.astype(int)

    min_week = df["week_start"].min()
    df["week_idx"] = ((df["week_start"] - min_week).dt.days // 7).astype(int)

    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52.0)

    return df


def _apply_lag_roll_one_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week_start").copy()
    # Lags de la demanda
    for lag in LAGS:
        g[f"y_lag_{lag}"] = g["y"].shift(lag)
    # Rolling stats desplazadas (no usan la semana actual)
    for win in ROLL_WINS:
        g[f"y_roll_mean_{win}"] = g["y"].shift(1).rolling(win).mean()
        g[f"y_roll_std_{win}"] = g["y"].shift(1).rolling(win).std()
    return g


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica lags y rolling por plato (dish_id), sin fuga de información.
    """
    if df.empty:
        return df.copy()

    df_out = (
        df.groupby("dish_id", group_keys=False)
        .apply(_apply_lag_roll_one_group)
        .reset_index(drop=True)
    )
    return df_out


# ==========================
# 4. Preparación del dataset desde la BD
# ==========================

def prepare_dataset_from_backend(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    - Carga dataset semanal desde PostgreSQL (via build_training_dataset_from_db).
    - Agrega features temporales + lags/rolling.
    - Limpia filas con NaN en lags.
    """
    # 👇 build_training_dataset_from_db devuelve SOLO un DataFrame
    df = build_training_dataset_from_db(
        start_date=start_date,
        end_date=end_date,
        export_csv_path=None,  # no exportamos CSV en modo servicio
    )

    if df.empty:
        raise RuntimeError("No hay datos suficientes para entrenar el modelo.")

    # Aseguramos tipos
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(float)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["avg_price_last4"] = pd.to_numeric(df["avg_price_last4"], errors="coerce")
    df["seasonal_factor_weekly"] = pd.to_numeric(
        df["seasonal_factor_weekly"], errors="coerce"
    ).fillna(1.0)

    # Features temporales
    df = add_temporal_features(df)

    # Lags/rolling
    df = add_lag_roll_features(df)

    # Eliminar filas sin lags completos (NaN)
    lag_cols = [c for c in df.columns if c.startswith("y_lag_") or c.startswith("y_roll_")]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    # Normalizar tipos categóricos
    df["dish_id"] = df["dish_id"].astype(str)
    df["dish_category"] = df["dish_category"].astype(str)

    return df


# ==========================
# 5. Modelo XGBoost y entrenamiento
# ==========================

# 👉 Rellena estos hiperparámetros con los de tu Nested CV definitivo
BEST_XGB_PARAMS: Dict[str, float | int] = {
    "n_estimators": 1600,
    "max_depth": 10,
    "learning_rate": 0.03,
    "min_child_weight": 1,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.10,
    "reg_lambda": 1.50,
}


def build_xgb_pipeline(params: Dict) -> Pipeline:
    """Pipeline = AutoCategoricalEncoder + XGBRegressor global."""
    mdl = XGBRegressor(
        n_estimators=params.get("n_estimators", 700),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 6),
        min_child_weight=params.get("min_child_weight", 3),
        subsample=params.get("subsample", 0.90),
        colsample_bytree=params.get("colsample_bytree", 0.80),
        reg_alpha=params.get("reg_alpha", 0.10),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    pipe = Pipeline(
        steps=[
            ("enc", AutoCategoricalEncoder(max_ohe_cards=10)),
            ("model", mdl),
        ]
    )
    return pipe


def temporal_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test_weeks: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal a nivel de semanas:
    - Train: primeras semanas
    - Test: últimas semanas (como hold-out).
    """
    df = df.sort_values("week_start").copy()
    weeks = np.array(sorted(df["week_start"].unique()))
    n_weeks = len(weeks)

    if n_weeks < 5:
        # Dataset muy pequeño, hacemos un split simple 70/30 sin mínimo
        n_test = max(1, int(round(n_weeks * test_ratio)))
    else:
        n_test = max(min_test_weeks, int(round(n_weeks * test_ratio)))
        if n_test >= n_weeks:
            n_test = max(1, n_weeks // 3)

    test_weeks = weeks[-n_test:]
    train_weeks = weeks[:-n_test]

    df_train = df[df["week_start"].isin(train_weeks)].copy()
    df_test = df[df["week_start"].isin(test_weeks)].copy()

    return df_train, df_test


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    """
    Entrena XGBoost GLOBAL con split temporal 80/20 y devuelve:
    - model_pipeline: pipeline entrenado.
    - metrics: diccionario con métricas en TEST (y tiempos).
    """
    # 1) Split temporal
    df_train, df_test = temporal_train_test_split(df, test_ratio=0.2, min_test_weeks=8)

    if df_train.empty or df_test.empty:
        raise RuntimeError("Split temporal produjo conjuntos vacíos. Revisa datos.")

    # 2) Target encoding por plato (dish_id) basado SOLO en TRAIN
    global_mean_y = df_train["y"].mean()
    mean_by_dish = df_train.groupby("dish_id")["y"].mean()

    df_train["dish_id_te"] = df_train["dish_id"].map(mean_by_dish)
    df_test["dish_id_te"] = df_test["dish_id"].map(mean_by_dish).fillna(global_mean_y)

    # 3) Asegurar tipos categóricos como string
    for d in (df_train, df_test):
        d["dish_id"] = d["dish_id"].astype(str)
        d["dish_category"] = d["dish_category"].astype(str)

    # 4) Definir columnas X / y
    DROP_COLS = ["y", "week_start"]

    X_train = df_train.drop(columns=DROP_COLS, errors="ignore")
    y_train = df_train["y"].astype(float).values

    X_test = df_test.drop(columns=DROP_COLS, errors="ignore")
    y_test = df_test["y"].astype(float).values

    # 5) Entrenar modelo
    model_pipeline = build_xgb_pipeline(BEST_XGB_PARAMS)

    t0 = time.time()
    model_pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # 6) Evaluar en TEST
    y_pred_test = model_pipeline.predict(X_test)
    m_test = metrics_on_positive(y_test, y_pred_test)

    metrics = {
        "MAE": m_test["MAE"],
        "RMSE": m_test["RMSE"],
        "MAPE": m_test["MAPE"],
        "R2": m_test["R2"],
        "n_pos": m_test["n_pos"],
        "n_total": m_test["n_total"],
        "TrainTime_s": train_time,
        "TrainTime_total_s": train_time,
    }

    return model_pipeline, metrics


# ==========================
# 6. Guardar modelo (.pkl) y metadatos en BD
# ==========================

def save_model_pickle(model_pipeline: Pipeline, model_id: str) -> str:
    """
    Guarda el modelo entrenado en la carpeta 'models' (o ML_MODELS_DIR).
    """
    models_dir = os.getenv("ML_MODELS_DIR", "models")
    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, f"{model_id}.joblib")
    dump(model_pipeline, path)
    return path


def save_model_metadata_to_db(
    model_pipeline: Pipeline,
    metrics: Dict,
    created_by: Optional[str] = None,
    training_start: Optional[datetime] = None,
) -> Tuple[str, str]:
    """
    Inserta:
    - 1 fila en ml_models
    - 1 fila en model_training_history
    y devuelve (model_id, training_history_id).

    created_by puede ser:
      - None
      - username (ej. "admin")
      - uuid en string (ej. "c1b1c6f3-...")
    """
    from sqlalchemy import text
    import uuid as uuid_lib

    engine = get_engine()
    now = datetime.utcnow()
    training_start = training_start or now

    model_id = str(uuid.uuid4())
    training_history_id = str(uuid.uuid4())

    # Obtenemos parámetros básicos del modelo XGBoost
    xgb_model: XGBRegressor = model_pipeline.named_steps["model"]
    model_params = xgb_model.get_params()

    # Reducimos un poco para guardar solo lo importante
    params_to_store = {k: model_params.get(k) for k in BEST_XGB_PARAMS.keys()}
    metrics_to_store = {
        "MAE": metrics.get("MAE"),
        "RMSE": metrics.get("RMSE"),
        "MAPE": metrics.get("MAPE"),
        "R2": metrics.get("R2"),
        "n_pos": metrics.get("n_pos"),
        "n_total": metrics.get("n_total"),
    }

    parameters_json = json.dumps(
        {
            "model_params": params_to_store,
            "metrics": metrics_to_store,
        }
    )

    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")

    with engine.begin() as conn:

        # --- Resolver created_by a UUID (o dejarlo en None) ---
        db_created_by: Optional[str] = None
        if created_by:
            try:
                # Si ya es un UUID válido, lo usamos tal cual
                uuid_lib.UUID(str(created_by))
                db_created_by = str(created_by)
            except ValueError:
                # Si no es UUID, asumimos que es username y buscamos en users
                row = conn.execute(
                    text("SELECT id FROM users WHERE username = :u LIMIT 1"),
                    {"u": created_by},
                ).fetchone()
                if row:
                    db_created_by = str(row[0])
                else:
                    # No existe usuario con ese username ⇒ lo dejamos en NULL
                    db_created_by = None

        # === Insert en ml_models ===
        insert_model_sql = text(
            """
            INSERT INTO ml_models (
                id,
                model_name,
                model_type,
                version,
                mae,
                rmse,
                accuracy,
                parameters,
                trained_at,
                created_at,
                is_active,
                created_by
            )
            VALUES (
                :id,
                :model_name,
                :model_type,
                :version,
                :mae,
                :rmse,
                :accuracy,
                :parameters,
                :trained_at,
                :created_at,
                :is_active,
                :created_by
            )
            """
        )

        version_str = now.strftime("v%Y%m%d_%H%M%S")

        conn.execute(
            insert_model_sql,
            {
                "id": model_id,
                "model_name": "XGBoost Global Demand",
                "model_type": "global",
                "version": version_str,
                "mae": mae,
                "rmse": rmse,
                "accuracy": None,  # podrías usar R2 si quieres
                "parameters": parameters_json,
                "trained_at": now,
                "created_at": now,
                "is_active": True,
                "created_by": db_created_by,
            },
        )

        # === Insert en model_training_history ===
        insert_hist_sql = text(
            """
            INSERT INTO model_training_history (
                id,
                model_id,
                status,
                accuracy_before,
                accuracy_after,
                data_points_used,
                training_start,
                training_end,
                created_at,
                created_by,
                error_message
            )
            VALUES (
                :id,
                :model_id,
                :status,
                :acc_before,
                :acc_after,
                :data_points_used,
                :training_start,
                :training_end,
                :created_at,
                :created_by,
                :error_message
            )
            """
        )

        conn.execute(
            insert_hist_sql,
            {
                "id": training_history_id,
                "model_id": model_id,
                "status": "SUCCESS",
                "acc_before": None,
                "acc_after": None,  # podrías mapear R2 aquí si quieres
                "data_points_used": int(metrics.get("n_total") or 0),
                "training_start": training_start,
                "training_end": now,
                "created_at": now,
                "created_by": db_created_by,
                "error_message": None,
            },
        )

    return model_id, training_history_id


# ==========================
# 7. Función pública que usa FastAPI
# ==========================

def train_model(created_by: Optional[str] = None) -> Dict:
    """
    Entrena el modelo GLOBAL XGBoost usando datos del backend
    y devuelve un diccionario con todo lo que FastAPI va a responder.
    """
    print("🚀 Iniciando entrenamiento GLOBAL XGBoost (FastAPI)...")

    training_start = datetime.utcnow()

    # 1) Dataset desde la BD + features + lags
    df = prepare_dataset_from_backend()
    print(f"Dataset listo para ML. Shape: {df.shape}")

    # 2) Entrenar + evaluar (hold-out temporal 80/20)
    model_pipeline, metrics = train_and_evaluate(df)

    # 3) Guardar metadatos en BD
    model_id, training_history_id = save_model_metadata_to_db(
        model_pipeline,
        metrics,
        created_by=created_by,
        training_start=training_start,
    )

    # 4) Guardar .pkl
    model_path = save_model_pickle(model_pipeline, model_id)

    # 5) Armar respuesta amigable
    response = {
        "model_id": model_id,
        "training_history_id": training_history_id,
        "model_path": model_path,
        "metrics": {
            "mae_pos": metrics.get("MAE"),
            "rmse_pos": metrics.get("RMSE"),
            "mape_pos": metrics.get("MAPE"),
            "r2_pos": metrics.get("R2"),
            "n_pos": metrics.get("n_pos"),
            "n_total": metrics.get("n_total"),
            "train_time_holdout_s": metrics.get("TrainTime_s"),
            "train_time_total_s": metrics.get("TrainTime_total_s"),
        },
        "version": datetime.utcnow().strftime("v%Y%m%d_%H%M%S"),
        "created_by": created_by,
    }

    print("✅ Entrenamiento finalizado (FastAPI).")
    return response
