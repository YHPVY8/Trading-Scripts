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

    "ES Daily Pivots": ("es_daily_pivot_levels", "date"),
    "ES Weekly Pivots": ("es_weekly_pivot_levels", "date"),
    "ES 2h Pivots": ("es_2hr_pivot_levels", "time"),
    "ES 4h Pivots": ("es_4hr_pivot_levels", "time"),
    "ES 30m Pivots": ("es_30m_pivot_levels", "time"),

    # ---- New GC pivot views ----
    "GC Daily Pivots": ("gc_daily_pivot_levels", "date"),
    "GC Weekly Pivots": ("gc_weekly_pivot_levels", "date"),

    "ES Range Extensions": ("es_range_extensions", "date"),
    "ES RTH Pivots": ("es_rth_pivot_levels", "trade_date"),
    "ES ON Pivots":  ("es_on_pivot_levels",  "trade_date"),
    "SPX Opening Range": ("spx_opening_range_stats", "trade_date"),

    # ---- SPX Daily ----
    "SPX Daily": {
        "table": "spx_daily",
        "date_col": "trade_date",
        "keep": [
            "trade_date", "day",
            "ibh", "ibl", "ib_mid",
            "am_hi", "am_lo",
            "rth_hi", "rth_lo",
            "ibh_broke_am", "ibl_broke_am", "both_ib_broke_am",
            "pm_ext_up", "pm_ext_down",
            "ib_ext",
        ],
        "labels": {
            "trade_date": "Date",
            "day": "Day",
            "ibh": "IBH",
            "ibl": "IBL",
            "ib_mid": "IB Mid",
            "am_hi": "AM High",
            "am_lo": "AM Low",
            "rth_hi": "RTH High",
            "rth_lo": "RTH Low",
            "ibh_broke_am": "IBH Broke AM",
            "ibl_broke_am": "IBL Broke AM",
            "both_ib_broke_am": "Both IB Broke AM",
            "pm_ext_up": "PM Ext Up",
            "pm_ext_down": "PM Ext Down",
            "ib_ext": "IB Ext",
        },
    },

    # ---- Euro IB ----
    "Euro IB": {
        "table": "es_eur_ib_summary",
        "date_col": "trade_date",
        "keep": [
            "trade_date", "day",
            "eur_ibh", "eur_ibl",
            "eibh_break", "eibl_break",
            "eibh12_hit", "eibl12_hit",
            "eibh15_hit", "eibl15_hit",
            "eibh20_hit", "eibl20_hit",
            "eur_ibh_rth_hit", "eur_ibl_rth_hit",
        ],
        "labels": {
            "trade_date": "Date",
            "day": "Day",
            "eur_ibh": "EUR IBH",
            "eur_ibl": "EUR IBL",
            "eibh_break": "eIBH Break",
            "eibl_break": "eIBL Break",
            "eibh12_hit": "IBH ≥1.2×",
            "eibl12_hit": "IBL ≥1.2×",
            "eibh15_hit": "IBH ≥1.5×",
            "eibl15_hit": "IBL ≥1.5×",
            "eibh20_hit": "IBH ≥2.0×",
            "eibl20_hit": "IBL ≥2.0×",
            "eur_ibh_rth_hit": "IBH → RTH Hit",
            "eur_ibl_rth_hit": "IBL → RTH Hit",
        },
    },

    # ---- GC Levels (base) ----
"GC Levels": {
    "table": "gc_levels",
    "date_col": "trade_date",
    "keep": [
        "trade_date", "day",
        "broke_onh", "broke_onl",
        "aibh_broke_premarket", "aibl_broke_premarket",
        "aib_mid_hit_premarket",
        "aibh_broke_adj_rth", "aibl_broke_adj_rth",
        "aib_mid_hit_rth",
        "aibh12x_hit_rth", "aibh15x_hit_rth", "aibh2x_hit_rth",
        "aibl12x_hit_rth", "aibl15x_hit_rth", "aibl2x_hit_rth",
    ],
    "labels": {
        "trade_date": "Date",
        "day": "Day",
        "onh": "ONH",
        "onl": "ONL",
        "broke_onh": "Hit ONH",
        "broke_onl": "Hit ONL",

        "aibh_broke_premarket": "aIBH Hit Premarket",
        "aibl_broke_premarket": "aIBL Hit Premarket",
        "aib_mid_hit_premarket": "aIB Mid Hit Premarket",
        "aibh_broke_adj_rth":   "aIBH Hit RTH",
        "aibl_broke_adj_rth":   "aIBL Hit RTH",
        "aib_mid_hit_rth":      "aIB Mid Hit RTH",
        "aibh12x_hit_rth": "aIBH1.2x - RTH",
        "aibh15x_hit_rth": "aIBH1.5x - RTH",
        "aibh2x_hit_rth":  "aIBH 2x - RTH",
        "aibl12x_hit_rth": "aIBL1.2x - RTH",
        "aibl15x_hit_rth": "aIBL1.5x - RTH",
        "aibl2x_hit_rth":  "aIBL2x - RTH",
        "aib_mid": "aIB Mid",
    },
},
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
