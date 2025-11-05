# views_config.py
from collections import OrderedDict
import yaml
from pathlib import Path

# Your existing base tables live here so App.py stays clean
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

def _load_yaml_views(path: Path) -> OrderedDict:
    if not path.exists():
        return OrderedDict()
    data = yaml.safe_load(path.read_text()) or {}
    out = OrderedDict()
    for view_name, cfg in data.items():
        out[view_name] = {
            "table": cfg["table"],
            "date_col": cfg.get("date_col", "date"),
            "keep": cfg.get("keep"),
            "labels": cfg.get("labels", {}) or {},
        }
    return out

def build_tables(config_path: str = "views.yaml") -> OrderedDict:
    """Merge base tables with YAML-defined views."""
    tables = OrderedDict(BASE_TABLES)
    tables.update(_load_yaml_views(Path(config_path)))
    return tables
