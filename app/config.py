from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración central del microservicio ML.
    Lee variables desde el archivo .env y desde variables de entorno.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = Field(default="GestRest AI ML Service", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Database
    ml_db_url: str = Field(
        default="postgresql+psycopg2://postgres:1234@localhost:5432/db_TP1test2",
        alias="ML_DB_URL",
    )

    # Backend Spring Boot
    backend_api_url: str = Field(
        default="http://localhost:8080/api",
        alias="BACKEND_API_URL",
    )

    # Artifact paths
    artifacts_dir: str = Field(default="artifacts", alias="ML_ARTIFACTS_DIR")
    models_dir: str = Field(default="artifacts/models", alias="ML_MODELS_DIR")
    metrics_dir: str = Field(default="artifacts/metrics", alias="ML_METRICS_DIR")
    logs_dir: str = Field(default="artifacts/logs", alias="ML_LOGS_DIR")

    # Storage backend
    # local: mantiene rutas locales en parameters.artifact_path
    # s3: sube modelos/métricas a S3 y guarda rutas s3://... en PostgreSQL
    storage_backend: str = Field(default="local", alias="ML_STORAGE_BACKEND")

    # S3 artifacts
    s3_bucket: Optional[str] = Field(default=None, alias="ML_S3_BUCKET")
    s3_models_prefix: str = Field(default="models", alias="ML_S3_MODELS_PREFIX")
    s3_metrics_prefix: str = Field(default="metrics", alias="ML_S3_METRICS_PREFIX")
    aws_region: str = Field(default="us-east-2", alias="AWS_REGION")

    # Training execution mode
    # sync: /ml/train espera hasta terminar.
    # async: /ml/train responde rápido y entrena en segundo plano.
    # En App Runner se recomienda async para evitar timeouts HTTP.
    train_mode: str = Field(default="sync", alias="ML_TRAIN_MODE")
    max_async_training_jobs: int = Field(default=1, alias="ML_MAX_ASYNC_TRAINING_JOBS")

    # Training config
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    min_active_ratio: float = Field(default=0.20, alias="MIN_ACTIVE_RATIO")
    min_train_weeks: int = Field(default=24, alias="MIN_TRAIN_WEEKS")
    validation_weeks: int = Field(default=6, alias="VALIDATION_WEEKS")
    test_weeks: int = Field(default=6, alias="TEST_WEEKS")

    # Feature config
    lags: str = Field(default="1,2,3,4,8,12,26,52", alias="LAGS")
    rolling_windows: str = Field(default="4,8,12", alias="ROLLING_WINDOWS")

    @property
    def lag_list(self) -> List[int]:
        return self._parse_int_list(self.lags)

    @property
    def rolling_window_list(self) -> List[int]:
        return self._parse_int_list(self.rolling_windows)

    @staticmethod
    def _parse_int_list(value: str) -> List[int]:
        if not value:
            return []
        return [int(x.strip()) for x in value.split(",") if x.strip()]

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_dir)

    @property
    def logs_path(self) -> Path:
        return Path(self.logs_dir)

    def ensure_directories(self) -> None:
        """
        Crea las carpetas necesarias para guardar modelos, métricas y logs.
        """
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
