from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from app.config import get_settings
from features.encoders import AutoCategoricalEncoder


DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 1200,
    "max_depth": 8,
    "learning_rate": 0.03,
    "min_child_weight": 1,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.10,
    "reg_lambda": 1.50,
}


XGB_PARAM_GRID: List[Dict[str, Any]] = [
    {
        "n_estimators": 1200,
        "max_depth": 10,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },
    {
        "n_estimators": 1600,
        "max_depth": 10,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 1.50,
    },
    {
        "n_estimators": 2000,
        "max_depth": 8,
        "learning_rate": 0.02,
        "min_child_weight": 3,
        "subsample": 0.95,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.20,
        "reg_lambda": 1.50,
    },
    {
        "n_estimators": 2500,
        "max_depth": 6,
        "learning_rate": 0.015,
        "min_child_weight": 1,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.05,
        "reg_lambda": 0.50,
    },
]


def build_xgboost_pipeline(
    params: Optional[Dict[str, Any]] = None,
    max_ohe_cards: int = 10,
) -> Pipeline:
    """
    Construye el pipeline final:

    AutoCategoricalEncoder
    + XGBRegressor

    Este enfoque permite trabajar con variables numéricas y categóricas
    dentro de un único modelo global para todos los platillos.
    """

    settings = get_settings()

    final_params = DEFAULT_XGB_PARAMS.copy()

    if params:
        final_params.update(params)

    model = XGBRegressor(
        n_estimators=final_params["n_estimators"],
        max_depth=final_params["max_depth"],
        learning_rate=final_params["learning_rate"],
        min_child_weight=final_params["min_child_weight"],
        subsample=final_params["subsample"],
        colsample_bytree=final_params["colsample_bytree"],
        reg_alpha=final_params["reg_alpha"],
        reg_lambda=final_params["reg_lambda"],
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=settings.random_seed,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("encoder", AutoCategoricalEncoder(max_ohe_cards=max_ohe_cards)),
            ("model", model),
        ]
    )

    return pipeline


def get_xgb_param_grid(fast_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Retorna la grilla de hiperparámetros.

    fast_mode=True usa menos combinaciones para pruebas rápidas.
    """

    if fast_mode:
        return XGB_PARAM_GRID[:2]

    return XGB_PARAM_GRID


def clip_negative_predictions(y_pred) -> np.ndarray:
    """
    La demanda no puede ser negativa.
    Si el modelo predice valores negativos, se ajustan a cero.
    """

    y_pred = np.asarray(y_pred, dtype=float)

    return np.maximum(y_pred, 0.0)


def get_default_xgb_params() -> Dict[str, Any]:
    """
    Retorna una copia de los parámetros base.
    """

    return DEFAULT_XGB_PARAMS.copy()