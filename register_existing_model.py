# register_existing_model.py
#
# Toma el artifact guardado en global_xgb_model_center10.pkl (joblib),
# extrae el modelo (Pipeline XGBoost), lo guarda como <model_id>.joblib
# y lo registra en la tabla ml_models como modelo ACTIVO.
#
# El registro en BD se hace directamente con psycopg2 (sin SQLAlchemy).

import os
import uuid
import json
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import psycopg2

from data_prep_backend import DB_URL

# ==========================
# AutoCategoricalEncoder (copiado de ml_training.py)
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
# Hack para el módulo "Pipeline" usado en el entrenamiento original (por si acaso)
# ==========================

import types
import sys
import sklearn.pipeline as sk_pipeline

dummy_mod = types.ModuleType("Pipeline")
dummy_mod.Pipeline = sk_pipeline.Pipeline  # type: ignore[attr-defined]
sys.modules["Pipeline"] = dummy_mod

# ==========================
# Rutas
# ==========================

PKL_PATH = os.path.join("models", "global_xgb_model_center10.pkl")
MODELS_DIR = os.getenv("ML_MODELS_DIR", "models")


def get_psycopg2_conn():
    """
    Construye una conexión psycopg2 a partir de DB_URL de data_prep_backend.
    Ejemplo DB_URL:
    postgresql+psycopg2://postgres:1234@localhost:5432/db_TP1test2
    """
    parsed = urlparse(DB_URL)
    user = parsed.username
    password = parsed.password
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    dbname = parsed.path.lstrip("/")

    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
    )


def main():
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo PKL en: {PKL_PATH}")

    print(f"Cargando artifact desde {PKL_PATH} ...")

    # 1) Cargar el artifact con joblib
    artifact = joblib.load(PKL_PATH)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise RuntimeError(
            "El archivo PKL no tiene el formato esperado de 'artifact' "
            "(dict con clave 'model')."
        )

    model = artifact["model"]
    best_params = artifact.get("best_params") or {}
    training_info = artifact.get("training_info") or {}
    nested_best = training_info.get("nested_cv_best_outer") or {}
    full_metrics = training_info.get("full_history_metrics") or {}

    mae = nested_best.get("MAE_pos") or full_metrics.get("MAE")
    rmse = nested_best.get("RMSE_pos") or full_metrics.get("RMSE")
    r2 = nested_best.get("R2_pos") or full_metrics.get("R2")

    # Redondear para encajar perfecto en numeric(10,4) y numeric(5,4)
    mae_rounded = round(float(mae), 4) if mae is not None else None
    rmse_rounded = round(float(rmse), 4) if rmse is not None else None
    r2_rounded = round(float(r2), 4) if r2 is not None else None

    print("Artifact cargado correctamente.")
    print("  - Llaves:", list(artifact.keys()))
    print("  - Métricas MAE_pos:", mae_rounded, " RMSE_pos:", rmse_rounded)

    # 2) Generar model_id y guardar SOLO el modelo como .joblib
    model_id = str(uuid.uuid4())
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")

    print(f"Guardando modelo (Pipeline) como {joblib_path} ...")
    joblib.dump(model, joblib_path)
    print("Modelo guardado en formato .joblib.")

    # 3) Preparar JSON de parámetros/métricas
    parameters_json = json.dumps(
        {
            "model_params": best_params,
            "metrics": {
                "MAE_pos": mae_rounded,
                "RMSE_pos": rmse_rounded,
                "R2_pos": r2_rounded,
            },
        }
    )

    now = datetime.utcnow()
    version_str = now.strftime("v%Y%m%d_%H%M%S")

    # 4) Insertar en ml_models usando psycopg2
    conn = get_psycopg2_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Desactivar otros modelos activos
                cur.execute("UPDATE ml_models SET is_active = FALSE WHERE is_active = TRUE;")

                # Insertar el nuevo modelo
                cur.execute(
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
                        %s,  -- id (uuid)
                        %s,  -- model_name
                        %s,  -- model_type
                        %s,  -- version
                        %s,  -- mae (numeric(10,4))
                        %s,  -- rmse (numeric(10,4))
                        %s,  -- accuracy (numeric(5,4))
                        %s::jsonb, -- parameters
                        %s,  -- trained_at
                        %s,  -- created_at
                        %s,  -- is_active
                        %s   -- created_by (uuid, puede ser NULL)
                    );
                    """,
                    (
                        model_id,
                        "XGBoost Global Demand (import artifact pkl)",
                        "global",
                        version_str,
                        mae_rounded,
                        rmse_rounded,
                        r2_rounded,
                        parameters_json,
                        now,
                        now,
                        True,
                        None,  # created_by NULL para respetar el FK a users.id
                    ),
                )
        print("✅ Modelo registrado en ml_models como ACTIVO.")
        print(f"   model_id    = {model_id}")
        print(f"   joblib path = {joblib_path}")
        print("Ahora get_active_model_id() devolverá este model_id.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
