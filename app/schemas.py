from typing import Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Fecha inicial opcional YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="Fecha final opcional YYYY-MM-DD")
    fast_mode: bool = Field(default=True, description="Modo rápido para pruebas")
    register_in_db: bool = Field(default=True, description="Registrar modelo en PostgreSQL")
    created_by: str = Field(default="admin", description="Username, email o UUID del usuario")


class PredictRequest(BaseModel):
    weeks_ahead: int = Field(default=4, ge=1, le=12)
    dish_id: Optional[str] = Field(default=None, description="UUID del platillo opcional")
    save_to_db: bool = Field(default=True)
    created_by: str = Field(default="admin")


class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None