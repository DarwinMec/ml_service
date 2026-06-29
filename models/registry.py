import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from app.config import get_settings
from data.database import get_engine
from storage.s3_artifacts import upload_local_file_to_s3


def make_model_version(prefix: str = "xgb") -> str:
    """
    Genera una versión única para el modelo.
    Ejemplo: xgb_20260525_153012
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_model_artifact(
    artifact: Dict[str, Any],
    version: str,
    model_name: str = "xgboost_demand_forecasting",
) -> str:
    """
    Guarda el artefacto del modelo en formato .joblib.

    Modo local:
    - Guarda en artifacts/models y retorna una ruta local.

    Modo S3:
    - Guarda temporalmente en artifacts/models.
    - Sube el archivo a S3.
    - Retorna una ruta s3://bucket/models/modelo.joblib.
    """
    settings = get_settings()
    settings.ensure_directories()

    filename = f"{model_name}_{version}.joblib"
    local_path = settings.models_path / filename

    joblib.dump(artifact, local_path)

    if settings.storage_backend.lower() == "s3":
        return upload_local_file_to_s3(
            local_path=local_path,
            prefix=settings.s3_models_prefix,
        )

    return str(local_path)


def save_metrics_json(
    metrics_payload: Dict[str, Any],
    version: str,
    model_name: str = "xgboost_demand_forecasting",
) -> str:
    """
    Guarda métricas y metadata del entrenamiento en un archivo JSON.

    Modo local:
    - Guarda en artifacts/metrics y retorna una ruta local.

    Modo S3:
    - Guarda temporalmente en artifacts/metrics.
    - Sube el archivo a S3.
    - Retorna una ruta s3://bucket/metrics/archivo.json.
    """
    settings = get_settings()
    settings.ensure_directories()

    filename = f"{model_name}_{version}_metrics.json"
    local_path = settings.metrics_path / filename

    with open(local_path, "w", encoding="utf-8") as file:
        json.dump(
            to_json_safe(metrics_payload),
            file,
            ensure_ascii=False,
            indent=4,
        )

    if settings.storage_backend.lower() == "s3":
        return upload_local_file_to_s3(
            local_path=local_path,
            prefix=settings.s3_metrics_prefix,
        )

    return str(local_path)


def resolve_user_id(created_by: Optional[str]) -> Optional[str]:
    """
    Convierte un username/email/id en UUID válido para columnas created_by.

    Casos:
    - Si created_by ya es UUID, lo retorna.
    - Si created_by es username, busca en users.username.
    - Si created_by es email, busca en users.email.
    - Si no encuentra usuario, retorna None.
    """

    if not created_by:
        return None

    # Si ya viene como UUID válido
    try:
        return str(uuid.UUID(str(created_by)))
    except ValueError:
        pass

    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT id::text
                FROM users
                WHERE username = :value
                   OR email = :value
                LIMIT 1
                """
            ),
            {"value": created_by},
        ).scalar()

    return result


def register_model_in_database(
    model_name: str,
    model_type: str,
    version: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    created_by: str = "admin",
    is_active: bool = True,
) -> str:
    """
    Registra el modelo entrenado en la tabla ml_models.

    Antes de activar el nuevo modelo, desactiva modelos activos anteriores
    con el mismo model_name.
    """
    engine = get_engine()
    model_id = str(uuid.uuid4())

    created_by_id = resolve_user_id(created_by)

    if created_by_id is None:
        raise ValueError(
            f"No se encontró un usuario válido para created_by='{created_by}'. "
            "Usa un username, email o UUID existente en la tabla users."
        )

    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")

    parameters_json = json.dumps(to_json_safe(parameters), ensure_ascii=False)

    with engine.begin() as connection:
        if is_active:
            connection.execute(
                text(
                    """
                    UPDATE ml_models
                    SET is_active = false
                    WHERE model_name = :model_name
                      AND is_active = true
                    """
                ),
                {"model_name": model_name},
            )

        connection.execute(
            text(
                """
                INSERT INTO ml_models (
                    id,
                    model_name,
                    model_type,
                    version,
                    parameters,
                    is_active,
                    mae,
                    rmse,
                    r2,
                    trained_at,
                    created_by,
                    created_at
                )
                VALUES (
                    CAST(:id AS uuid),
                    :model_name,
                    :model_type,
                    :version,
                    CAST(:parameters AS jsonb),
                    :is_active,
                    :mae,
                    :rmse,
                    :r2,
                    :trained_at,
                    CAST(:created_by AS uuid),
                    :created_at
                )
                """
            ),
            {
                "id": model_id,
                "model_name": model_name,
                "model_type": model_type,
                "version": version,
                "parameters": parameters_json,
                "is_active": is_active,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "trained_at": datetime.now(),
                "created_by": created_by_id,
                "created_at": datetime.now(),
            },
        )

    return model_id


def register_training_history(
    model_id: str,
    status: str,
    data_points_used: int,
    training_start: datetime,
    training_end: datetime,
    r2_before: Optional[float] = None,
    r2_after: Optional[float] = None,
    error_message: Optional[str] = None,
    created_by: str = "admin",
) -> str:
    """
    Registra el historial de entrenamiento en model_training_history.
    """
    engine = get_engine()
    history_id = str(uuid.uuid4())

    created_by_id = resolve_user_id(created_by)

    if created_by_id is None:
        raise ValueError(
            f"No se encontró un usuario válido para created_by='{created_by}'. "
            "Usa un username, email o UUID existente en la tabla users."
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO model_training_history (
                    id,
                    model_id,
                    training_start,
                    training_end,
                    data_points_used,
                    r2before,
                    r2after,
                    status,
                    error_message,
                    created_by,
                    created_at
                )
                VALUES (
                    CAST(:id AS uuid),
                    CAST(:model_id AS uuid),
                    :training_start,
                    :training_end,
                    :data_points_used,
                    :r2_before,
                    :r2_after,
                    :status,
                    :error_message,
                    CAST(:created_by AS uuid),
                    :created_at
                )
                """
            ),
            {
                "id": history_id,
                "model_id": model_id,
                "training_start": training_start,
                "training_end": training_end,
                "data_points_used": data_points_used,
                "r2_before": r2_before,
                "r2_after": r2_after,
                "status": status,
                "error_message": error_message,
                "created_by": created_by_id,
                "created_at": datetime.now(),
            },
        )

    return history_id


def to_json_safe(value: Any) -> Any:
    """
    Convierte objetos de numpy, pandas, datetime y Path a tipos compatibles con JSON.
    """

    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and np.isnan(value):
        return None

    return value