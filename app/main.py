from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import TrainRequest, PredictRequest
from data.database import test_connection
from jobs.training_jobs import (
    get_training_job,
    list_training_jobs,
    start_training_job,
)
from models.prediction import (
    get_active_model_record,
    predict_future_demand,
)
from models.training import train_xgboost_model_or_raise
from models.registry import to_json_safe


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Microservicio ML para predicción de demanda en restaurantes con XGBoost.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def should_run_training_async(request: TrainRequest) -> bool:
    """
    Decide si /ml/train debe ejecutarse en segundo plano.

    Prioridad:
    1. request.async_mode, si viene explícito.
    2. Variable ML_TRAIN_MODE.
    """
    if request.async_mode is not None:
        return bool(request.async_mode)

    return settings.train_mode.strip().lower() == "async"


def start_async_training_response(request: TrainRequest, compatible_status: bool = False) -> dict:
    job = start_training_job(
        start_date=request.start_date,
        end_date=request.end_date,
        fast_mode=request.fast_mode,
        register_in_db=request.register_in_db,
        created_by=request.created_by,
    )

    # compatible_status=True se usa en /ml/train para no romper backends actuales
    # que esperan status=completed/ok/success para considerar la llamada exitosa.
    root_status = "completed" if compatible_status else "accepted"

    return {
        "status": root_status,
        "message": "Entrenamiento iniciado en segundo plano.",
        "data": json_safe(job),
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "storage_backend": settings.storage_backend,
        "train_mode": settings.train_mode,
    }


@app.get("/health")
def health_check():
    db_ok = test_connection()

    return {
        "status": "ok" if db_ok else "error",
        "database": "connected" if db_ok else "disconnected",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/ml/model/active")
def get_active_model():
    try:
        record = get_active_model_record()
        return {
            "status": "completed",
            "message": "Modelo activo encontrado",
            "data": json_safe(record),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No se pudo obtener el modelo activo: {str(exc)}",
        )


@app.post("/ml/train")
def train_model(request: TrainRequest):
    """
    Entrena el modelo.

    En local puede ejecutarse síncrono.
    En AWS/App Runner se recomienda ML_TRAIN_MODE=async para que este mismo
    endpoint responda rápido y el entrenamiento continúe en segundo plano.
    """
    try:
        if should_run_training_async(request):
            return start_async_training_response(
                request,
                compatible_status=True,
            )

        result = train_xgboost_model_or_raise(
            start_date=request.start_date,
            end_date=request.end_date,
            fast_mode=request.fast_mode,
            register_in_db=request.register_in_db,
            created_by=request.created_by,
        )

        return {
            "status": "completed",
            "message": "Modelo entrenado correctamente",
            "data": json_safe(result),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo entrenar el modelo: {str(exc)}",
        )


@app.post("/ml/train/sync")
def train_model_sync(request: TrainRequest):
    """
    Fuerza entrenamiento síncrono aunque ML_TRAIN_MODE=async.
    Recomendado solo para local o entrenamientos muy cortos.
    """
    try:
        result = train_xgboost_model_or_raise(
            start_date=request.start_date,
            end_date=request.end_date,
            fast_mode=request.fast_mode,
            register_in_db=request.register_in_db,
            created_by=request.created_by,
        )

        return {
            "status": "completed",
            "message": "Modelo entrenado correctamente",
            "data": json_safe(result),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo entrenar el modelo: {str(exc)}",
        )


@app.post("/ml/train/async")
def train_model_async(request: TrainRequest):
    """
    Fuerza entrenamiento asíncrono.
    """
    try:
        return start_async_training_response(request, compatible_status=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo iniciar el entrenamiento asíncrono: {str(exc)}",
        )


@app.get("/ml/train/status/{job_id}")
def get_training_status(job_id: str):
    job = get_training_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No existe job de entrenamiento con id={job_id}",
        )

    return {
        "status": job.get("status", "unknown"),
        "message": job.get("message", "Estado de entrenamiento obtenido."),
        "data": json_safe(job),
    }


@app.get("/ml/train/jobs")
def get_training_jobs(limit: int = 20):
    jobs = list_training_jobs(limit=limit)

    return {
        "status": "completed",
        "message": "Jobs de entrenamiento obtenidos correctamente.",
        "data": json_safe({"jobs": jobs}),
    }


@app.post("/ml/predict")
def predict_model(request: PredictRequest):
    try:
        result = predict_future_demand(
            weeks_ahead=request.weeks_ahead,
            dish_id=request.dish_id,
            save_to_db=request.save_to_db,
            created_by=request.created_by,
        )

        return {
            "status": "completed",
            "message": "Predicción generada correctamente",
            "data": json_safe(result),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo generar la predicción: {str(exc)}",
        )


def json_safe(value: Any) -> Any:
    """
    Convierte objetos no serializables a formatos seguros para JSON.
    Complementa to_json_safe() con Decimal.
    """

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    safe = to_json_safe(value)

    if isinstance(safe, Decimal):
        return float(safe)

    return safe
