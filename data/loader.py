from typing import Optional

import pandas as pd

from data.database import read_sql


def load_sales_raw(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Carga las ventas históricas desde PostgreSQL.

    Retorna un DataFrame con:
    - dish_id
    - dish_name
    - dish_category
    - sale_date
    - quantity
    - unit_price
    - total_amount
    """

    query = """
        SELECT
            d.id::text AS dish_id,
            d.name AS dish_name,
            d.category AS dish_category,
            s.sale_date AS sale_date,
            si.quantity AS quantity,
            COALESCE(si.unit_price, d.price) AS unit_price,
            si.total_amount AS total_amount
        FROM sale_items si
        INNER JOIN sales s ON s.id = si.sale_id
        INNER JOIN dishes d ON d.id = si.dish_id
        WHERE d.is_active = true
    """

    params = {}

    if start_date:
        query += " AND s.sale_date >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query += " AND s.sale_date <= :end_date"
        params["end_date"] = end_date

    query += """
        ORDER BY d.id, s.sale_date
    """

    df = read_sql(query, params)

    if df.empty:
        return df

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(float)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

    df["dish_category"] = df["dish_category"].fillna("Sin categoría").astype(str)
    df["dish_name"] = df["dish_name"].astype(str)

    return df


def load_active_dishes() -> pd.DataFrame:
    """
    Carga los platos activos desde PostgreSQL.
    """

    query = """
        SELECT
            id::text AS dish_id,
            name AS dish_name,
            category AS dish_category,
            price AS dish_price
        FROM dishes
        WHERE is_active = true
        ORDER BY name
    """

    df = read_sql(query)

    if df.empty:
        return df

    df["dish_category"] = df["dish_category"].fillna("Sin categoría").astype(str)
    df["dish_price"] = pd.to_numeric(df["dish_price"], errors="coerce").fillna(0).astype(float)

    return df