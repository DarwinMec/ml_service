from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import TrainRequest, PredictRequest
from data.database import test_connection
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


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
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