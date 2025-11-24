# main.py - VERSIÓN ACTUALIZADA

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from ml_training import train_model
from ml_prediction import generate_predictions  # ⬅️ NUEVO IMPORT
from dotenv import load_dotenv
load_dotenv()

from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd

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


app = FastAPI(
    title="Inventory ML Service",
    description="Servicio de Machine Learning para predicción de demanda",
    version="1.0.0",
)

# ⬇️ NUEVO: Configurar CORS para Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== Schemas Pydantic (mantén los que ya tienes) ======

class TrainRequest(BaseModel):
    created_by: Optional[str] = None
    async_mode: bool = False


class TrainMetrics(BaseModel):
    mae_pos: float | None
    rmse_pos: float | None
    mape_pos: float | None
    r2_pos: float | None
    n_pos: int | None
    n_total: int | None
    train_time_holdout_s: float | None
    train_time_total_s: float | None


class TrainResponse(BaseModel):
    model_id: str
    training_history_id: str
    model_path: str
    metrics: TrainMetrics
    version: str
    created_by: str | None


# ⬇️ NUEVOS SCHEMAS para predicciones
class PredictionRequest(BaseModel):
    dish_id: Optional[str] = None
    weeks_ahead: int = 4
    save_to_db: bool = True


class PredictionItem(BaseModel):
    dish_id: str
    dish_name: str
    week_start: str
    predicted_demand: float
    confidence: str


class PredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionItem]
    total_predictions: int
    model_id: str | None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ====== Endpoints (mantén los que ya tienes y agrega los nuevos) ======

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Verifica que el servicio esté funcionando."""
    return {
        "status": "ok",
        "service": "ML Inventory Service",
        "version": "1.0.0"
    }


@app.post("/ml/train", response_model=TrainResponse, tags=["ml"])
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Dispara el entrenamiento global de XGBoost.
    - Usa datos desde PostgreSQL (comparte BD con tu backend de Spring).
    - Registra modelo + training_history.
    - Guarda el .pkl en disco.
    """

    if not request.async_mode:
        try:
            result = train_model(created_by=request.created_by)
            return TrainResponse(
                model_id=result["model_id"],
                training_history_id=result["training_history_id"],
                model_path=result["model_path"],
                metrics=TrainMetrics(**result["metrics"]),
                version=result["version"],
                created_by=result["created_by"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")

    def background_job(created_by: str | None):
        try:
            train_model(created_by=created_by)
        except Exception as e:
            print(f"❌ Error en entrenamiento background: {e}")

    background_tasks.add_task(background_job, request.created_by)

    return TrainResponse(
        model_id="pending",
        training_history_id="pending",
        model_path="pending",
        metrics=TrainMetrics(
            mae_pos=None,
            rmse_pos=None,
            mape_pos=None,
            r2_pos=None,
            n_pos=None,
            n_total=None,
            train_time_holdout_s=None,
            train_time_total_s=None,
        ),
        version="pending",
        created_by=request.created_by,
    )


# ⬇️ NUEVOS ENDPOINTS para predicciones

# ========================
# FIX 1 — Forzar model_id = str(model_id)
# ========================

@app.post("/ml/predict", response_model=PredictionResponse, tags=["ml"])
def trigger_prediction(request: PredictionRequest):
    try:
        predictions = generate_predictions(
            dish_id=request.dish_id,
            weeks_ahead=request.weeks_ahead,
            save_to_db=request.save_to_db
        )

        from ml_prediction import get_active_model_id
        model_id = get_active_model_id()

        return PredictionResponse(
            success=True,
            predictions=[PredictionItem(**pred) for pred in predictions],
            total_predictions=len(predictions),
            model_id=str(model_id) if model_id else None   # ← FIX
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generando predicciones: {str(e)}"
        )


@app.get("/ml/predict/{dish_id}", response_model=PredictionResponse, tags=["ml"])
def get_prediction_for_dish(
    dish_id: str,
    weeks_ahead: int = Query(default=4, ge=1, le=52),
    save_to_db: bool = Query(default=False)
):
    try:
        predictions = generate_predictions(
            dish_id=dish_id,
            weeks_ahead=weeks_ahead,
            save_to_db=save_to_db
        )

        from ml_prediction import get_active_model_id
        model_id = get_active_model_id()

        return PredictionResponse(
            success=True,
            predictions=[PredictionItem(**pred) for pred in predictions],
            total_predictions=len(predictions),
            model_id=str(model_id) if model_id else None   # ← FIX
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo predicción: {str(e)}"
        )


@app.get("/ml/predictions/latest", response_model=PredictionResponse, tags=["ml"])
def get_latest_predictions():
    """
    Obtiene las predicciones más recientes guardadas en la BD.
    Lee desde la tabla 'predictions' (estructura real de tu BD).
    """
    try:
        from ml_prediction import get_engine, get_active_model_id
        from sqlalchemy import text

        engine = get_engine()
        model_id = get_active_model_id()

        if not model_id:
            return PredictionResponse(
                success=False,
                predictions=[],
                total_predictions=0,
                model_id=None
            )

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 
                        p.dish_id,
                        d.name AS dish_name,
                        p.predicted_date AS week_start,
                        p.predicted_quantity,
                        p.confidence_level
                    FROM predictions p
                    JOIN dishes d ON p.dish_id = d.id
                    WHERE p.model_id = :model_id
                    ORDER BY p.predicted_date, d.name
                """),
                {"model_id": model_id}
            )

            rows = result.fetchall()

            predictions = [
                {
                    "dish_id": str(row[0]),
                    "dish_name": row[1],
                    "week_start": str(row[2]),
                    "predicted_demand": float(row[3]),
                    "confidence": str(row[4]) if row[4] is not None else "0.80",
                }
                for row in rows
            ]

        return PredictionResponse(
            success=True,
            predictions=[PredictionItem(**pred) for pred in predictions],
            total_predictions=len(predictions),
            model_id=model_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo predicciones: {str(e)}"
        )


@app.get("/ml/models/active", tags=["ml"])
def get_active_model_info():
    try:
        from ml_prediction import get_engine, get_active_model_id
        from sqlalchemy import text
        
        model_id = get_active_model_id()
        if not model_id:
            raise HTTPException(status_code=404, detail="No hay modelo activo")

        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 
                        id, model_name, model_type, version,
                        mae, rmse, accuracy, trained_at, created_at
                    FROM ml_models
                    WHERE id = :model_id
                """),
                {"model_id": model_id}
            )

            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Modelo no encontrado")

        return {
            "model_id": str(row[0]),   # ← FIX
            "model_name": row[1],
            "model_type": row[2],
            "version": row[3],
            "mae": float(row[4]) if row[4] else None,
            "rmse": float(row[5]) if row[5] else None,
            "accuracy": float(row[6]) if row[6] else None,
            "trained_at": str(row[7]),
            "created_at": str(row[8])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo información del modelo: {str(e)}"
        )



# ====== Ejecutar con: uvicorn main:app --reload ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)