from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings


def normalize_s3_uri(value: str | Path | None) -> str:
    """
    Normaliza rutas S3.

    Casos soportados:
    - s3://bucket/key
    - s3:/bucket/key

    El segundo caso puede aparecer si una ruta S3 fue tratada antes como Path.
    """
    if value is None:
        return ""

    value_str = str(value).strip()

    if value_str.startswith("s3://"):
        return value_str

    if value_str.startswith("s3:/"):
        clean_value = value_str.replace("s3:/", "", 1).lstrip("/")
        return f"s3://{clean_value}"

    return value_str


def is_s3_uri(value: str | Path | None) -> bool:
    """
    Indica si una ruta corresponde a un URI S3.
    """
    normalized = normalize_s3_uri(value)
    return normalized.startswith("s3://")


def parse_s3_uri(s3_uri: str | Path) -> tuple[str, str]:
    """
    Convierte s3://bucket/key en bucket y key.
    """
    normalized = normalize_s3_uri(s3_uri)
    parsed = urlparse(normalized)

    if parsed.scheme != "s3":
        raise ValueError(f"La ruta no es un URI S3 válido: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"URI S3 incompleto: {s3_uri}")

    return bucket, key


def get_s3_client():
    """
    Retorna un cliente S3 usando credenciales del entorno.

    En AWS App Runner se deben otorgar permisos mediante Instance Role.
    En local se usan las credenciales configuradas por AWS CLI.
    """
    settings = get_settings()

    return boto3.client(
        "s3",
        region_name=settings.aws_region,
    )


def upload_local_file_to_s3(
    local_path: str | Path,
    prefix: str,
) -> str:
    """
    Sube un archivo local a S3 y retorna su URI s3://bucket/key.
    """
    settings = get_settings()

    if not settings.s3_bucket:
        raise ValueError("ML_S3_BUCKET no está configurado.")

    path = Path(local_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo local para subir a S3: {path}")

    clean_prefix = prefix.strip("/")
    key = f"{clean_prefix}/{path.name}" if clean_prefix else path.name

    try:
        s3 = get_s3_client()
        s3.upload_file(str(path), settings.s3_bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"No se pudo subir el artefacto a S3: {exc}") from exc

    return f"s3://{settings.s3_bucket}/{key}"


def download_s3_uri_to_cache(
    s3_uri: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """
    Descarga un archivo S3 a una carpeta temporal/cache local.
    Si el archivo ya existe en cache, lo reutiliza.
    """
    settings = get_settings()
    bucket, key = parse_s3_uri(s3_uri)

    if cache_dir is None:
        cache_root = settings.models_path / "_s3_cache"
    else:
        cache_root = Path(cache_dir)

    local_path = cache_root / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    try:
        s3 = get_s3_client()
        s3.download_file(bucket, key, str(local_path))
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"No se pudo descargar el artefacto desde S3: {exc}") from exc

    return local_path


def resolve_artifact_to_local_path(artifact_path: str | Path) -> Path:
    """
    Resuelve una ruta de artefacto a una ruta local utilizable por joblib.

    Soporta:
    - Ruta local: artifacts/models/modelo.joblib
    - Ruta S3: s3://bucket/models/modelo.joblib
    - Ruta S3 malformada defensiva: s3:/bucket/models/modelo.joblib
    """
    artifact_path_str = normalize_s3_uri(artifact_path)

    if is_s3_uri(artifact_path_str):
        return download_s3_uri_to_cache(artifact_path_str)

    return Path(artifact_path_str)
