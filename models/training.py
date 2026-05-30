from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.config import get_settings
from data.dataset_builder import (
    build_weekly_sales_dataset,
    filter_active_demand_items,
)
from features.feature_pipeline import (
    prepare_feature_dataset,
    split_features_target,
)
from models.cross_validation import nested_rolling_cv_xgboost
from models.evaluation import metrics_on_positive, metrics_all
from models.registry import (
    make_model_version,
    register_model_in_database,
    register_training_history,
    save_metrics_json,
    save_model_artifact,
)
from models.xgboost_model import build_xgboost_pipeline, clip_negative_predictions


MODEL_NAME = "xgboost_demand_forecasting"
MODEL_TYPE = "XGBoost"


def train_xgboost_model(
    start_date: str | None = None,
    end_date: str | None = None,
    fast_mode: bool = False,
    register_in_db: bool = True,
    created_by: str = "ml_service",
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de entrenamiento XGBoost.

    Pasos:
    1. Construye dataset semanal desde PostgreSQL.
    2. Filtra platillos con actividad mínima.
    3. Genera features temporales, lags y rolling.
    4. Ejecuta nested rolling cross-validation.
    5. Selecciona mejores hiperparámetros.
    6. Entrena modelo final con todo el histórico.
    7. Guarda artifact .joblib.
    8. Guarda métricas .json.
    9. Registra modelo e historial en PostgreSQL.
    """

    settings = get_settings()
    training_start = datetime.now()

    model_id = None
    history_id = None

    try:
        # 1. Dataset semanal base
        weekly_df = build_weekly_sales_dataset(
            start_date=start_date,
            end_date=end_date,
            fill_missing_weeks=True,
        )

        # 2. Filtrar platillos con actividad mínima
        filtered_df = filter_active_demand_items(
            weekly_df,
            min_active_ratio=settings.min_active_ratio,
        )

        if filtered_df.empty:
            raise ValueError(
                "Después del filtro de actividad no quedaron platillos suficientes para entrenar"
            )

        # 3. Features para ML
        feature_df = prepare_feature_dataset(
            filtered_df,
            drop_na_lags=True,
            use_dynamic_lags=True,
        )

        if feature_df.empty:
            raise ValueError(
                "Después de generar lags y rolling features no quedaron filas para entrenar"
            )

        # 4. Nested Rolling Cross-Validation
        cv_result = nested_rolling_cv_xgboost(
            feature_df,
            fast_mode=fast_mode,
        )

        best_params = cv_result["best_params"]
        cv_summary = cv_result["summary"]

        if not best_params:
            raise ValueError("No se encontraron hiperparámetros válidos para XGBoost")

        # 5. Entrenamiento final con todo el histórico disponible
        X_all, y_all = split_features_target(feature_df)

        final_model = build_xgboost_pipeline(params=best_params)
        final_model.fit(X_all, y_all)

        y_pred = final_model.predict(X_all)
        y_pred = clip_negative_predictions(y_pred)

        train_metrics_positive = metrics_on_positive(y_all, y_pred)
        train_metrics_all = metrics_all(y_all, y_pred)

        training_end = datetime.now()
        version = make_model_version(prefix="xgb")

        feature_columns = list(X_all.columns)

        # 6. Payload del artifact
        artifact = {
            "model": final_model,
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "version": version,
            "features": feature_columns,
            "best_params": best_params,
            "training_info": {
                "training_start": training_start,
                "training_end": training_end,
                "data_points_used": int(len(feature_df)),
                "fast_mode": fast_mode,
                "cv_summary": cv_summary,
                "cv_outer_results": cv_result["outer_results"],
                "train_metrics_positive": train_metrics_positive,
                "train_metrics_all": train_metrics_all,
            },
        }

        # 7. Guardar artifact .joblib
        model_path = save_model_artifact(
            artifact=artifact,
            version=version,
            model_name=MODEL_NAME,
        )

        # 8. Guardar métricas .json
        metrics_payload = {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "version": version,
            "model_path": str(model_path),
            "best_params": best_params,
            "data": {
                "weekly_rows": int(len(weekly_df)),
                "filtered_rows": int(len(filtered_df)),
                "feature_rows": int(len(feature_df)),
                "n_dishes": int(filtered_df["dish_id"].nunique()),
                "start_week": str(filtered_df["week_start"].min()),
                "end_week": str(filtered_df["week_start"].max()),
            },
            "cross_validation": cv_summary,
            "train_metrics_positive": train_metrics_positive,
            "train_metrics_all": train_metrics_all,
            "created_by": created_by,
            "created_at": training_end,
        }

        metrics_path = save_metrics_json(
            metrics_payload=metrics_payload,
            version=version,
            model_name=MODEL_NAME,
        )

        # 9. Registro en PostgreSQL
        if register_in_db:
            parameters_payload = {
                "artifact_path": str(model_path),
                "metrics_path": str(metrics_path),
                "best_params": best_params,
                "feature_columns": feature_columns,
                "fast_mode": fast_mode,
                "data_rows": int(len(feature_df)),
                "n_dishes": int(filtered_df["dish_id"].nunique()),
                "cv_summary": cv_summary,
            }

            model_id = register_model_in_database(
                model_name=MODEL_NAME,
                model_type=MODEL_TYPE,
                version=version,
                parameters=parameters_payload,
                metrics=cv_summary,
                created_by=created_by,
                is_active=True,
            )

            history_id = register_training_history(
                model_id=model_id,
                status="completed",
                data_points_used=int(len(feature_df)),
                training_start=training_start,
                training_end=training_end,
                r2_before=None,
                r2_after=cv_summary.get("r2"),
                error_message=None,
                created_by=created_by,
            )

        return {
            "status": "completed",
            "model_id": model_id,
            "history_id": history_id,
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "version": version,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "data_points_used": int(len(feature_df)),
            "n_dishes": int(filtered_df["dish_id"].nunique()),
            "best_params": best_params,
            "cv_summary": cv_summary,
            "train_metrics_positive": train_metrics_positive,
            "train_metrics_all": train_metrics_all,
        }

    except Exception as exc:
        training_end = datetime.now()

        # Si más adelante quieres registrar errores de entrenamiento fallido,
        # se puede añadir aquí un registro en model_training_history.
        return {
            "status": "failed",
            "model_id": model_id,
            "history_id": history_id,
            "error": str(exc),
            "training_start": training_start.isoformat(),
            "training_end": training_end.isoformat(),
        }


def train_xgboost_model_or_raise(
    start_date: str | None = None,
    end_date: str | None = None,
    fast_mode: bool = False,
    register_in_db: bool = True,
    created_by: str = "ml_service",
) -> Dict[str, Any]:
    """
    Variante que lanza excepción si el entrenamiento falla.
    Útil para FastAPI o jobs programados.
    """

    result = train_xgboost_model(
        start_date=start_date,
        end_date=end_date,
        fast_mode=fast_mode,
        register_in_db=register_in_db,
        created_by=created_by,
    )

    if result.get("status") == "failed":
        raise RuntimeError(result.get("error", "Error desconocido durante el entrenamiento"))

    return result