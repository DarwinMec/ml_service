from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse_safe(y_true, y_pred) -> float:
    """
    Calcula RMSE de forma segura.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_pos(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Calcula MAPE solo para valores reales positivos.
    Evita división entre cero cuando y_true = 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true > 0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))
        ) * 100
    )


def metrics_on_positive(y_true, y_pred) -> Dict[str, Any]:
    """
    Calcula métricas considerando solo semanas con demanda real positiva.
    Esto replica la lógica usada en tu Colab de entrenamiento.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true > 0

    if mask.sum() < 2:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "n_pos": int(mask.sum()),
            "n_total": int(len(y_true)),
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": rmse_safe(yt, yp),
        "r2": float(r2_score(yt, yp)),
        "mape": mape_pos(yt, yp),
        "n_pos": int(mask.sum()),
        "n_total": int(len(y_true)),
    }


def metrics_all(y_true, y_pred) -> Dict[str, Any]:
    """
    Métricas generales sobre todas las filas, incluyendo semanas con demanda cero.
    Útil como referencia complementaria.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "n_total": int(len(y_true)),
        }

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse_safe(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape_pos(y_true, y_pred),
        "n_total": int(len(y_true)),
    }