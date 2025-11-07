# views_config.py
from collections import OrderedDict
import os
import json

try:
    import yaml
except ImportError:
    yaml = None

# ---- Fixed, hand-coded base tables ----
BASE_TABLES = OrderedDict({
    "Daily ES": ("daily_es", "time"),
    "Weekly ES": ("es_weekly", "time"),
    "30m ES": ("es_30m", "time"),
    "2h ES": ("es_2hr", "time"),
    "4h ES": ("es_4hr", "time"),
    "Daily Pivots": ("es_daily_pivot_levels", "date"),
    "Weekly Pivots": ("es_weekly_pivot_levels", "date"),
    "2h Pivots": ("es_2hr_pivot_levels", "time"),
    "4h Pivots": ("es_4hr_pivot_levels", "time"),
    "30m Pivots": ("es_30m_pivot_levels", "time"),
    "Range Extensions": ("es_range_extensions", "date"),
    "RTH Pivots": ("es_rth_pivot_levels", "trade_date"),   
    "ON Pivots":  ("es_on_pivot_levels",  "trade_date"),
    "Opening Range Stats": ("es_opening_range_stats", "trade_date"),
})

def _load_yaml_views(path="views.yaml"):
    if not yaml:
        return []
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Expecting a list under "views" or a dict; be flexible
    if isinstance(data, dict):
        return data.get("views", [])
    if isinstance(data, list):
        return data
    return []

def _load_db_views(sb):
    try:
        rows = (
            sb.table("dashboard_views")
              .select("*")
              .eq("is_enabled", True)
              .order("sort_order")
              .execute()
              .data
        )
        return rows or []
    except Exception:
        return []

def build_tables(sb, cache_bust: int | None = None):
    """
    Merge:
      1) BASE_TABLES (always included),
      2) views from Supabase (dashboard_views), and
      3) optional YAML views (views.yaml).
    No Streamlit cache here to avoid stale registries.
    """
    tables = OrderedDict(BASE_TABLES)

    # 2) DB views
    for r in _load_db_views(sb):
        view_name = r.get("view_name")
        table     = r.get("table_name")
        if not view_name or not table:
            continue
        tables[view_name] = {
            "table": table,
            "date_col": r.get("date_col", "date"),
            "keep": r.get("keep_columns") or None,
            "labels": r.get("labels") or {},
        }

    # 3) YAML views (optional)
    for v in _load_yaml_views():
        view_name = v.get("view_name")
        table     = v.get("table")
        if not view_name or not table:
            continue
        tables[view_name] = {
            "table": table,
            "date_col": v.get("date_col", "date"),
            "keep": v.get("keep"),
            "labels": v.get("labels", {}) or {},
        }

    return tables
