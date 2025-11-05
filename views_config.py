# views_config.py
from collections import OrderedDict

try:
    import streamlit as st
except Exception:
    st = None

def _warn(msg: str):
    if st:
        st.info(msg)

# Your fixed, hand-coded base tables stay here.
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
})

def build_tables(sb=None) -> OrderedDict:
    """
    Merge BASE_TABLES with any views stored in Supabase (dashboard_views).
    Pass the Supabase client from App.py as sb.
    """
    tables = OrderedDict(BASE_TABLES)
    if sb is None:
        return tables

    try:
        rows = (
            sb.table("dashboard_views")
              .select("*")
              .eq("is_enabled", True)
              .order("sort_order")
              .execute()
              .data
        )
    except Exception as e:
        _warn("Views table not loaded; using base tables only.")
        return tables

    for r in rows:
        view_name = r.get("view_name")
        if not view_name:
            continue
        tables[view_name] = {
            "table": r.get("table_name"),
            "date_col": r.get("date_col", "date"),
            "keep": r.get("keep_columns"),
            "labels": r.get("labels") or {},
        }
    return tables
