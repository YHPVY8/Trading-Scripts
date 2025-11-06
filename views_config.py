# views_config.py
from collections import OrderedDict
from pathlib import Path

# Optional deps (don't crash if missing)
try:
    import yaml  # PyYAML for views.yaml
except Exception:
    yaml = None

try:
    import streamlit as st
except Exception:
    st = None

def _info(msg: str):
    if st:
        st.info(msg)

def _warn(msg: str):
    if st:
        st.warning(msg)

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
})

# ---- YAML loader (optional) ----
def _load_yaml_views(config_path: str = "views.yaml") -> OrderedDict:
    out = OrderedDict()
    if not yaml:
        return out
    p = Path(config_path)
    if not p.exists():
        return out
    try:
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            _warn("views.yaml must be a mapping at top level.")
            return out
        for view_name, cfg in data.items():
            if view_name == "config":
                continue
            if not isinstance(cfg, dict) or "table" not in cfg:
                _warn(f"Skipping bad view '{view_name}' in YAML.")
                continue
            out[view_name] = {
                "table": cfg["table"],
                "date_col": cfg.get("date_col", "date"),
                "keep": cfg.get("keep"),
                "labels": cfg.get("labels", {}) or {},
            }
    except Exception:
        _warn("Could not parse views.yaml.")
    return out

# ---- Supabase loader (optional) ----
def _load_supabase_views(sb) -> OrderedDict:
    out = OrderedDict()
    if sb is None:
        return out
    try:
        rows = (
            sb.table("dashboard_views")
              .select("*")
              .eq("is_enabled", True)
              .order("sort_order")
              .execute()
              .data
        )
    except Exception:
        # Silent fallback; we keep base tables even if this fails
        return out

    for r in rows:
        name = r.get("view_name")
        if not name:
            continue
        out[name] = {
            "table": r.get("table_name"),
            "date_col": r.get("date_col", "date"),
            "keep": r.get("keep_columns"),
            "labels": r.get("labels") or {},
        }
    return out

def build_tables(arg=None) -> OrderedDict:
    """
    Unified builder:
      - If arg has `.table`, it's treated as a Supabase client and we load Supabase views.
      - We also merge YAML views (if views.yaml exists).
      - BASE_TABLES are always included.
    """
    tables = OrderedDict(BASE_TABLES)

    # Supabase client?
    sb = arg if hasattr(arg, "table") else None
    tables.update(_load_supabase_views(sb))

    # YAML views (optional; merged after Supabase in case you still want file-based views)
    tables.update(_load_yaml_views("views.yaml"))

    return tables
