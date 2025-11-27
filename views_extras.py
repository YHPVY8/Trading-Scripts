# views_extras.py
import streamlit as st
import pandas as pd
from supabase import create_client
from datetime import date, timedelta

# -------------------- Existing content (kept) --------------------
LEVEL_ALIASES = [
    ("Pivot", ["Pivot", "pivot"]),
    ("R0.5",  ["R05", "r05"]),
    ("S0.5",  ["S05", "s05"]),
    ("R1",    ["R1",  "r1"]),
    ("S1",    ["S1",  "s1"]),
    ("R1.5",  ["R15", "r15"]),
    ("S1.5",  ["S15", "s15"]),
    ("R2",    ["R2",  "r2"]),
    ("S2",    ["S2",  "s2"]),
    ("R3",    ["R3","r3"]),
    ("S3",    ["S3","s3"]),
]

PIVOT_TABLES = {
    "es_daily_pivot_levels",
    "es_weekly_pivot_levels",
    "es_2hr_pivot_levels",
    "es_4hr_pivot_levels",
    "es_30m_pivot_levels",
    "es_rth_pivot_levels",
    "es_on_pivot_levels",
}

def render_spx_or_extras(choice: str, table_name: str, df: pd.DataFrame | None):
    """Lightweight metrics for SPX Opening Range view (used by generic path)."""
    if choice != "SPX Opening Range" or df is None or df.empty:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    if "broke_up" in df:   c2.metric("Broke Up",   f"{100*pd.to_numeric(df['broke_up'], errors='coerce').mean():.1f}%")
    if "broke_down" in df: c3.metric("Broke Down", f"{100*pd.to_numeric(df['broke_down'], errors='coerce').mean():.1f}%")
    if "broke_both" in df: c4.metric("Broke Both", f"{100*pd.to_numeric(df['broke_both'], errors='coerce').mean():.1f}%")

def _normalize_table_name(name: str) -> str:
    return (name or "").strip().split(".")[-1].lower()

def _first_present(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None

def _first_existing_datecol(rec: dict, preferred: str):
    # prefer the passed-in date_col; if missing, try common fallbacks
    candidates = [preferred, "trade_date", "date", "time"]
    for c in candidates:
        if c in rec:
            return c
    return preferred  # last resort; request will still have 1 row

def fetch_current_levels(sb, table_name: str, date_col: str) -> dict:
    try:
        # Get latest row with all columns to avoid case/quote issues
        resp = (
            sb.table(table_name)
              .select("*")
              .order(date_col, desc=True)
              .limit(1)
              .execute()
        )
        rows = resp.data or []
        if not rows:
            return {}
        rec = rows[0]

        # If the chosen date_col wasn't actually present, try again ordering by a fallback
        if date_col not in rec:
            fallback = _first_existing_datecol(rec, date_col)
            if fallback != date_col:
                resp2 = (
                    sb.table(table_name)
                      .select("*")
                      .order(fallback, desc=True)
                      .limit(1)
                      .execute()
                )
                rows2 = resp2.data or []
                if rows2:
                    rec = rows2[0]

        out = {}
        for label, cands in LEVEL_ALIASES:
            v = _first_present(rec, cands)
            if isinstance(v, str):
                try:
                    v = float(v.replace(",", ""))
                except Exception:
                    pass
            out[label] = v
        return out
    except Exception:
        return {}

def render_current_levels(sb, choice: str, table_name: str, date_col: str):
    norm = _normalize_table_name(table_name)
    is_pivot = (norm in PIVOT_TABLES) or norm.endswith("_pivot_levels")
    if not is_pivot:
        # Small diagnostic so you know why nothing prints
        st.caption(f"ℹ️ Skipping levels (table '{table_name}' not recognized as a pivot-level table).")
        return

    levels = fetch_current_levels(sb, table_name, date_col)

    if not any(levels.values()):
        # Show what keys were present in the latest record to help diagnose column-name mismatches
        try:
            probe = (
                sb.table(table_name)
                  .select("*")
                  .order(date_col, desc=True)
                  .limit(1)
                  .execute()
            )
            keys = ", ".join(sorted((probe.data or [{}])[0].keys()))
        except Exception:
            keys = "unknown"
        st.caption(f"ℹ️ No current levels found (table='{table_name}', date_col='{date_col}'). Latest row keys: {keys}")
        return

    st.markdown("#### Current period levels")
    lines = []
    for label, _ in LEVEL_ALIASES:
        v = levels.get(label)
        if v is not None:
            try:
                lines.append(f"**{label}**  {v:,.2f}")
            except Exception:
                lines.append(f"**{label}**  {v}")
    if lines:
        st.markdown("\n".join(f"- {ln}" for ln in lines))

# --- Euro IB metrics (top-of-view header) ---
def render_euro_ib_metrics(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    dff = df.copy()

    # Case-insensitive resolver
    lower = {c.lower(): c for c in dff.columns}
    def col(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    # Columns we care about
    c_eibh_break = col("eibh_break")
    c_eibl_break = col("eibl_break")
    c_eib_both   = None  # compute below

    c_ibh12 = col("eibh12_hit", "eur_ibh1.2_hit", "eur_ibh12_hit")
    c_ibl12 = col("eibl12_hit", "eur_ibl1.2_hit", "eur_ibl12_hit")
    c_ibh15 = col("eibh15_hit", "eur_ibh1.5_hit", "eur_ibh15_hit")
    c_ibl15 = col("eibl15_hit", "eur_ibl1.5_hit", "eur_ibl15_hit")
    c_ibh20 = col("eibh20_hit", "eur_ibh2_hit")
    c_ibl20 = col("eibl20_hit", "eur_ibl2_hit")

    c_ibh_rth = col("eur_ibh_rth_hit")
    c_ibl_rth = col("eur_ibl_rth_hit")

    # Coerce booleans robustly
    def to_bool_series(s):
        if s is None or s not in dff:
            return pd.Series(dtype="boolean")
        x = dff[s]
        if x.dtype == bool:
            return x.astype("boolean")
        # handle 0/1 ints too
        if pd.api.types.is_integer_dtype(x) or pd.api.types.is_float_dtype(x):
            return (x.astype(float) != 0.0).astype("boolean")
        return x.astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        ).astype("boolean")


    s_eibh = to_bool_series(c_eibh_break)
    s_eibl = to_bool_series(c_eibl_break)
    s_both = (s_eibh & s_eibl).astype("boolean")

    def rate(s: pd.Series) -> str:
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    # Top row: breaks
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{len(dff):,}")
    with c2: st.metric("% eIBH Break", rate(s_eibh))
    with c3: st.metric("% eIBL Break", rate(s_eibl))
    with c4: st.metric("% Both eIB Break", rate(s_both))

    # Second row: extension hits
    e1, e2, e3, e4, e5, e6 = st.columns(6)
    with e1: st.metric("IBH ≥1.2×", rate(to_bool_series(c_ibh12)))
    with e2: st.metric("IBL ≥1.2×", rate(to_bool_series(c_ibl12)))
    with e3: st.metric("IBH ≥1.5×", rate(to_bool_series(c_ibh15)))
    with e4: st.metric("IBL ≥1.5×", rate(to_bool_series(c_ibl15)))
    with e5: st.metric("IBH ≥2.0×", rate(to_bool_series(c_ibh20)))
    with e6: st.metric("IBL ≥2.0×", rate(to_bool_series(c_ibl20)))

    # Third row: RTH hits
    r1, r2 = st.columns(2)
    with r1: st.metric("IBH → RTH Hit", rate(to_bool_series(c_ibh_rth)))
    with r2: st.metric("IBL → RTH Hit", rate(to_bool_series(c_ibl_rth)))



# -------------------- New helpers (SPX filter + metrics) --------------------

def _sb():
    """Local Supabase client using Streamlit secrets."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def spx_opening_range_filter_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar control + metrics for SPX Opening Range that returns a filtered DF.
    We DO NOT render a custom table here—App.py's generic renderer will handle
    styling, sorting, green highlights, downloads, etc.
    """
    if df is None or df.empty:
        return df
    if "or_window" not in df.columns:
        return df

    # --- Sidebar: choose a single window (3m/5m/15m)
    win = st.sidebar.selectbox("OR Window (SPX)", ["3m", "5m", "15m"], index=0, key="spx_or_window")
    dff = df[df["or_window"] == win].copy()

    # Ensure date ordering ASC so latest is at the bottom (matches App.py)
    if "trade_date" in dff.columns:
        dff["trade_date"] = pd.to_datetime(dff["trade_date"], errors="coerce")
        dff = dff.sort_values("trade_date", ascending=True).reset_index(drop=True)

    # --- Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{len(dff):,}")
    with c2: st.metric("Broke Up",   f"{100.0 * pd.to_numeric(dff.get('broke_up',   pd.Series(dtype=bool)), errors='coerce').mean():.1f}%" if "broke_up"   in dff else "–")
    with c3: st.metric("Broke Down", f"{100.0 * pd.to_numeric(dff.get('broke_down', pd.Series(dtype=bool)), errors='coerce').mean():.1f}%" if "broke_down" in dff else "–")
    with c4: st.metric("Broke Both", f"{100.0 * pd.to_numeric(dff.get('broke_both', pd.Series(dtype=bool)), errors='coerce').mean():.1f}%" if "broke_both" in dff else "–")

    # --- Extension metrics (≥20/50/100%) — robust, case-insensitive lookup
    cols_lower = {c.lower(): c for c in dff.columns}  # map lower->actual

    def _find_col_ci(names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    # Your schema first, then fallbacks
    up20  = _find_col_ci(["hit_20_up",  "hitup20", "hit_up20", "up20",  "or_up_20",  "hit_or_up_20"])
    up50  = _find_col_ci(["hit_50_up",  "hitup50", "hit_up50", "up50",  "or_up_50",  "hit_or_up_50"])
    up100 = _find_col_ci(["hit_100_up", "hitup100","hit_up100","up100", "or_up_100", "hit_or_up_100"])

    dn20  = _find_col_ci(["hit_20_down",  "hitdn20", "hit_down20", "down20",  "or_dn_20",  "hit_or_dn_20"])
    dn50  = _find_col_ci(["hit_50_down",  "hitdn50", "hit_down50", "down50",  "or_dn_50",  "hit_or_dn_50"])
    dn100 = _find_col_ci(["hit_100_down", "hitdn100","hit_down100","down100", "or_dn_100", "hit_or_dn_100"])

    # numeric fallbacks (fractions of OR)
    max_up = _find_col_ci(["max_ext_up", "max_up_ext", "max_up_frac", "max_up_or_mult"])
    max_dn = _find_col_ci(["max_ext_down", "max_dn_ext", "max_dn_frac", "max_dn_or_mult"])

    # --- Metrics render
    st.markdown("#### Extension hits (fraction of OR after the window completes)")
    e1, e2, e3, e4, e5, e6 = st.columns(6)
    def _rate_from_bool(dfX, col):
        if not col or col not in dfX: return "–"
        s = dfX[col]
        # coerce any "True"/"False"/1/0 strings -> booleans
        s = s.map(lambda v: True if str(v).strip().lower() in {"true","1","yes"} else (False if str(v).strip().lower() in {"false","0","no"} else None))
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _rate_from_numeric(dfX, col, thr):
        if not col or col not in dfX: return "–"
        s = pd.to_numeric(dfX[col], errors="coerce")
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * (s >= thr).mean():.1f}%"

    with e1: st.metric("Up ≥20%",   _rate_from_bool(dff, up20)  if up20  else _rate_from_numeric(dff, max_up, 0.20))
    with e2: st.metric("Up ≥50%",   _rate_from_bool(dff, up50)  if up50  else _rate_from_numeric(dff, max_up, 0.50))
    with e3: st.metric("Up ≥100%",  _rate_from_bool(dff, up100) if up100 else _rate_from_numeric(dff, max_up, 1.00))
    with e4: st.metric("Down ≥20%", _rate_from_bool(dff, dn20)  if dn20  else _rate_from_numeric(dff, max_dn, 0.20))
    with e5: st.metric("Down ≥50%", _rate_from_bool(dff, dn50)  if dn50  else _rate_from_numeric(dff, max_dn, 0.50))
    with e6: st.metric("Down ≥100%",_rate_from_bool(dff, dn100) if dn100 else _rate_from_numeric(dff, max_dn, 1.00))

    # TEMP debug to verify columns present (remove after it works)
    if not any([up20, up50, up100, dn20, dn50, dn100, max_up, max_dn]):
        st.caption(f"🧪 Debug: columns available → {', '.join(dff.columns)}")

    # Return filtered DF so App.py keeps your standard styling
    return dff


# -------------------- Compatibility stub (no-op override) --------------------
def render_view_override(view_id: str) -> bool:
    """
    Deprecated: Do not use table-level override anymore.
    We filter + show metrics via 'spx_opening_range_filter_and_metrics' and
    let App.py's generic renderer handle the table/styling.
    """
    return False
