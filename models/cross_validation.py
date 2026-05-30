from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import get_settings
from features.feature_pipeline import split_features_target
from models.evaluation import metrics_on_positive
from models.xgboost_model import (
    build_xgboost_pipeline,
    clip_negative_predictions,
    get_xgb_param_grid,
)


def get_unique_weeks(df: pd.DataFrame, date_col: str = "week_start") -> np.ndarray:
    """
    Retorna semanas únicas ordenadas cronológicamente.
    """

    if date_col not in df.columns:
        raise ValueError(f"No existe la columna de fecha: {date_col}")

    weeks = pd.to_datetime(df[date_col]).drop_duplicates().sort_values()

    return weeks.to_numpy()


def filter_by_weeks(
    df: pd.DataFrame,
    weeks: np.ndarray,
    date_col: str = "week_start",
) -> pd.DataFrame:
    """
    Filtra un DataFrame usando un conjunto de semanas.
    """

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    return out[out[date_col].isin(weeks)].copy()


def build_inner_folds(
    train_weeks: np.ndarray,
    inner_validation_weeks: int,
    min_train_weeks: int,
    max_inner_folds: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Construye folds internos tipo rolling origin.

    Estos folds se usan para seleccionar hiperparámetros.
    """

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    cursor = len(train_weeks)

    while (
        cursor >= min_train_weeks + inner_validation_weeks
        and len(folds) < max_inner_folds
    ):
        val_end = cursor
        val_start = val_end - inner_validation_weeks

        train_inner_weeks = train_weeks[:val_start]
        val_inner_weeks = train_weeks[val_start:val_end]

        folds.append((train_inner_weeks, val_inner_weeks))

        cursor -= inner_validation_weeks

    return folds


def nested_rolling_cv_xgboost(
    df: pd.DataFrame,
    fast_mode: bool = False,
    date_col: str = "week_start",
    target_col: str = "y",
    outer_folds: int = 3,
    max_inner_folds: int = 3,
) -> Dict[str, Any]:
    """
    Ejecuta Nested Rolling Cross-Validation para XGBoost.

    Estructura:
    - Outer folds: evaluación final.
    - Inner folds: búsqueda de hiperparámetros.

    Esto respeta el orden temporal de las ventas y evita fuga de información.
    """

    settings = get_settings()

    if df.empty:
        raise ValueError("El dataset recibido para cross-validation está vacío")

    required_cols = [date_col, target_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    weeks_all = get_unique_weeks(df, date_col=date_col)
    n_weeks = len(weeks_all)

    if n_weeks < settings.min_train_weeks + settings.validation_weeks + settings.test_weeks:
        raise ValueError(
            "No hay suficientes semanas para ejecutar nested rolling CV. "
            f"Semanas disponibles: {n_weeks}. "
            f"Mínimo requerido: "
            f"{settings.min_train_weeks + settings.validation_weeks + settings.test_weeks}"
        )

    param_grid = get_xgb_param_grid(fast_mode=fast_mode)

    outer_results: List[Dict[str, Any]] = []

    for outer_idx in range(outer_folds):
        test_end_idx = n_weeks - outer_idx * settings.test_weeks
        test_start_idx = test_end_idx - settings.test_weeks

        if test_start_idx <= settings.validation_weeks:
            break

        weeks_test_outer = weeks_all[test_start_idx:test_end_idx]
        weeks_train_outer = weeks_all[:test_start_idx]

        if len(weeks_train_outer) < settings.min_train_weeks:
            continue

        inner_folds = build_inner_folds(
            train_weeks=weeks_train_outer,
            inner_validation_weeks=settings.validation_weeks,
            min_train_weeks=settings.min_train_weeks,
            max_inner_folds=max_inner_folds,
        )

        if not inner_folds:
            continue

        best_params = None
        best_rmse = np.inf
        best_inner_metrics = None

        for params in param_grid:
            inner_metrics: List[Dict[str, Any]] = []

            for train_inner_weeks, val_inner_weeks in inner_folds:
                df_train_inner = filter_by_weeks(df, train_inner_weeks, date_col=date_col)
                df_val_inner = filter_by_weeks(df, val_inner_weeks, date_col=date_col)

                X_train, y_train = split_features_target(
                    df_train_inner,
                    target_col=target_col,
                    date_col=date_col,
                )
                X_val, y_val = split_features_target(
                    df_val_inner,
                    target_col=target_col,
                    date_col=date_col,
                )

                model = build_xgboost_pipeline(params=params)
                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_val)
                y_val_pred = clip_negative_predictions(y_val_pred)

                metrics = metrics_on_positive(y_val, y_val_pred)
                inner_metrics.append(metrics)

            mean_rmse = _safe_mean([m["rmse"] for m in inner_metrics])
            mean_mae = _safe_mean([m["mae"] for m in inner_metrics])
            mean_r2 = _safe_mean([m["r2"] for m in inner_metrics])
            mean_mape = _safe_mean([m["mape"] for m in inner_metrics])

            if mean_rmse is not None and mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = params
                best_inner_metrics = {
                    "mae": mean_mae,
                    "rmse": mean_rmse,
                    "r2": mean_r2,
                    "mape": mean_mape,
                    "folds": inner_metrics,
                }

        if best_params is None:
            continue

        df_train_outer = filter_by_weeks(df, weeks_train_outer, date_col=date_col)
        df_test_outer = filter_by_weeks(df, weeks_test_outer, date_col=date_col)

        X_train_outer, y_train_outer = split_features_target(
            df_train_outer,
            target_col=target_col,
            date_col=date_col,
        )
        X_test_outer, y_test_outer = split_features_target(
            df_test_outer,
            target_col=target_col,
            date_col=date_col,
        )

        final_model = build_xgboost_pipeline(params=best_params)
        final_model.fit(X_train_outer, y_train_outer)

        y_test_pred = final_model.predict(X_test_outer)
        y_test_pred = clip_negative_predictions(y_test_pred)

        test_metrics = metrics_on_positive(y_test_outer, y_test_pred)

        outer_results.append(
            {
                "outer_fold": outer_idx + 1,
                "train_weeks": int(len(weeks_train_outer)),
                "test_weeks": int(len(weeks_test_outer)),
                "best_params": best_params,
                "inner_metrics": best_inner_metrics,
                "test_metrics": test_metrics,
            }
        )

    if not outer_results:
        raise ValueError("No se pudo generar ningún fold válido para nested rolling CV")

    summary = summarize_outer_results(outer_results)

    return {
        "outer_results": outer_results,
        "summary": summary,
        "best_params": select_best_params_from_outer_results(outer_results),
    }


def summarize_outer_results(outer_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resume las métricas promedio de los outer folds.
    """

    test_metrics = [r["test_metrics"] for r in outer_results]

    return {
        "mae": _safe_mean([m["mae"] for m in test_metrics]),
        "rmse": _safe_mean([m["rmse"] for m in test_metrics]),
        "r2": _safe_mean([m["r2"] for m in test_metrics]),
        "mape": _safe_mean([m["mape"] for m in test_metrics]),
        "n_folds": len(outer_results),
        "n_pos_total": int(
            sum(m["n_pos"] for m in test_metrics if m["n_pos"] is not None)
        ),
        "n_total": int(
            sum(m["n_total"] for m in test_metrics if m["n_total"] is not None)
        ),
    }


def select_best_params_from_outer_results(
    outer_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Selecciona los mejores hiperparámetros a partir del mejor outer fold por RMSE.
    """

    best_row = None
    best_rmse = np.inf

    for row in outer_results:
        rmse = row["test_metrics"].get("rmse")

        if rmse is not None and rmse < best_rmse:
            best_rmse = rmse
            best_row = row

    if best_row is None:
        return {}

    return best_row["best_params"]


def _safe_mean(values: List[Any]) -> float | None:
    """
    Calcula promedio ignorando None y NaN.
    """

    cleaned = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, float) and np.isnan(value):
            continue

        cleaned.append(float(value))

    if not cleaned:
        return None

    return float(np.mean(cleaned))