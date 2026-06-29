from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from app.config import get_settings
from models.training import train_xgboost_model_or_raise


settings = get_settings()

_executor = ThreadPoolExecutor(max_workers=max(1, settings.max_async_training_jobs))
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = Lock()

_ACTIVE_STATUSES = {"queued", "training"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retorna una copia segura del job para exponer por API.
    """
    return deepcopy(job)


def has_active_training_job() -> bool:
    with _jobs_lock:
        return any(job.get("status") in _ACTIVE_STATUSES for job in _jobs.values())


def get_active_training_job() -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        for job in _jobs.values():
            if job.get("status") in _ACTIVE_STATUSES:
                return _public_job(job)
    return None


def start_training_job(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    fast_mode: bool = True,
    register_in_db: bool = True,
    created_by: str = "admin",
) -> Dict[str, Any]:
    """
    Crea un job de entrenamiento y lo ejecuta en segundo plano.

    El registro de jobs es en memoria. Es suficiente para App Runner/MVP,
    porque evita el timeout HTTP. El modelo entrenado se persiste en S3/RDS.
    """
    with _jobs_lock:
        active_job = next(
            (job for job in _jobs.values() if job.get("status") in _ACTIVE_STATUSES),
            None,
        )

        if active_job is not None:
            raise ValueError(
                "Ya existe un entrenamiento en ejecución. "
                f"Job activo: {active_job.get('job_id')}"
            )

        job_id = str(uuid4())
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "Entrenamiento en cola.",
            "requested_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "start_date": start_date,
            "end_date": end_date,
            "fast_mode": fast_mode,
            "register_in_db": register_in_db,
            "created_by": created_by,
            "model_id": None,
            "history_id": None,
            "version": None,
            "model_path": None,
            "metrics_path": None,
            "data_points_used": None,
            "n_dishes": None,
            "cv_summary": None,
            "error": None,
        }

        _jobs[job_id] = job

    _executor.submit(
        _run_training_job,
        job_id=job_id,
        start_date=start_date,
        end_date=end_date,
        fast_mode=fast_mode,
        register_in_db=register_in_db,
        created_by=created_by,
    )

    return get_training_job(job_id)  # type: ignore[return-value]


def _run_training_job(
    *,
    job_id: str,
    start_date: str | None,
    end_date: str | None,
    fast_mode: bool,
    register_in_db: bool,
    created_by: str,
) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "training"
        job["message"] = "Entrenamiento en ejecución."
        job["started_at"] = _now_iso()

    try:
        result = train_xgboost_model_or_raise(
            start_date=start_date,
            end_date=end_date,
            fast_mode=fast_mode,
            register_in_db=register_in_db,
            created_by=created_by,
        )

        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "completed"
            job["message"] = "Modelo entrenado correctamente."
            job["finished_at"] = _now_iso()
            job["model_id"] = result.get("model_id")
            job["history_id"] = result.get("history_id")
            job["version"] = result.get("version")
            job["model_path"] = result.get("model_path")
            job["metrics_path"] = result.get("metrics_path")
            job["data_points_used"] = result.get("data_points_used")
            job["n_dishes"] = result.get("n_dishes")
            job["cv_summary"] = result.get("cv_summary")
            job["result"] = result

    except Exception as exc:
        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["message"] = "El entrenamiento falló."
            job["finished_at"] = _now_iso()
            job["error"] = str(exc)


def get_training_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return _public_job(job) if job is not None else None


def list_training_jobs(limit: int = 20) -> list[Dict[str, Any]]:
    with _jobs_lock:
        ordered = sorted(
            _jobs.values(),
            key=lambda item: item.get("requested_at") or "",
            reverse=True,
        )
        return [_public_job(job) for job in ordered[:limit]]
