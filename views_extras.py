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
    ("R3", ["R3","r3"]),
    ("S3", ["S3","s3"]),
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
    if "broke_up" in df:   c2.metric("Broke Up",   f"{100*df['broke_up'].mean():.1f}%")
    if "broke_down" in df: c3.metric("Broke Down", f"{100*df['broke_down'].mean():.1f}%")
    if "broke_both" in df: c4.metric("Broke Both", f"{100*df['broke_both'].mean():.1f}%")

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

# -------------------- New helpers & override --------------------

def _sb():
    """Local Supabase client using Streamlit secrets."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pct(series: pd.Series) -> str:
    if series is None or series.empty:
        return "–"
    try:
        return f"{100.0 * series.mean():.1f}%"
    except Exception:
        return "–"

def _rate_from_bool(df: pd.DataFrame, colname: str) -> str:
    return _pct(df[colname]) if colname and colname in df else "–"

def _rate_from_numeric(df: pd.DataFrame, colname: str, threshold: float) -> str:
    if colname and colname in df:
        s = pd.to_numeric(df[colname], errors="coerce")
        return f"{100.0 * (s >= threshold).mean():.1f}%"
    return "–"

def render_view_override(view_id: str) -> bool:
    """
    Custom renderer for SPX Opening Range that preserves:
      - ascending date order (latest at bottom),
      - True/False green highlighting,
      - 2-decimal formatting on OR columns,
      - extension metrics with clear fallbacks/diagnostics.
    """
    if view_id != "SPX Opening Range":
        return False

    st.title("SPX Opening Range — single window")

    # ---- Sidebar: pick a single window and date range
    with st.sidebar:
        st.subheader("Filters")
        default_end = date.today()
        default_start = default_end - timedelta(days=120)
        d1, d2 = st.date_input("Date range", value=(default_start, default_end), format="YYYY-MM-DD")
        win = st.selectbox("OR Window", ["3m", "5m", "15m"], index=0)
        st.caption("Shows one row per day by filtering to the selected window.")

    # ---- Query Supabase: filter to the selected window (no ORDER here; we sort ASC client-side)
    sb = _sb()
    q = (
        sb.table("spx_opening_range_stats")
          .select("*")
          .gte("trade_date", str(d1))
          .lte("trade_date", str(d2))
          .eq("symbol", "SPX")
          .eq("or_window", win)
    )
    data = (q.execute().data) or []
    df = pd.DataFrame(data)
    if df.empty:
        st.info("No rows match the current filters.")
        return True

    # ---- Normalize/format & sort ASC (latest at bottom)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)

    # Numeric formatting (2 decimals for OR columns)
    for c in ["orh", "orl", "or_range"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # ---- Top metrics: broke up/down/both
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{len(df):,}")
    with c2: st.metric("Broke Up",   _pct(df["broke_up"])   if "broke_up"   in df else "–")
    with c3: st.metric("Broke Down", _pct(df["broke_down"]) if "broke_down" in df else "–")
    with c4: st.metric("Broke Both", _pct(df["broke_both"]) if "broke_both" in df else "–")

    # ---- Extension metrics (≥20/50/100%) — supports either boolean hit cols or numeric max-ext cols
    up20_col = _first_existing(df, ["hit_up20","up20","or_up_20","hit_or_up_20"])
    up50_col = _first_existing(df, ["hit_up50","up50","or_up_50","hit_or_up_50"])
    up100_col= _first_existing(df, ["hit_up100","up100","or_up_100","hit_or_up_100"])
    dn20_col = _first_existing(df, ["hit_dn20","down20","or_dn_20","hit_or_dn_20"])
    dn50_col = _first_existing(df, ["hit_dn50","down50","or_dn_50","hit_or_dn_50"])
    dn100_col= _first_existing(df, ["hit_dn100","down100","or_dn_100","hit_or_dn_100"])
    max_up_col = _first_existing(df, ["max_up_ext","max_up_frac","max_up_or_mult"])
    max_dn_col = _first_existing(df, ["max_dn_ext","max_dn_frac","max_dn_or_mult"])

    st.markdown("#### Extension hits (fraction of OR after the window completes)")
    e1, e2, e3, e4, e5, e6 = st.columns(6)
    with e1: st.metric("Up ≥20%",  _rate_from_bool(df, up20_col)  if up20_col  else _rate_from_numeric(df, max_up_col, 0.20))
    with e2: st.metric("Up ≥50%",  _rate_from_bool(df, up50_col)  if up50_col  else _rate_from_numeric(df, max_up_col, 0.50))
    with e3: st.metric("Up ≥100%", _rate_from_bool(df, up100_col) if up100_col else _rate_from_numeric(df, max_up_col, 1.00))
    with e4: st.metric("Down ≥20%",  _rate_from_bool(df, dn20_col)  if dn20_col  else _rate_from_numeric(df, max_dn_col, 0.20))
    with e5: st.metric("Down ≥50%",  _rate_from_bool(df, dn50_col)  if dn50_col  else _rate_from_numeric(df, max_dn_col, 0.50))
    with e6: st.metric("Down ≥100%", _rate_from_bool(df, dn100_col) if dn100_col else _rate_from_numeric(df, max_dn_col, 1.00))

    if not any([up20_col, up50_col, up100_col, dn20_col, dn50_col, dn100_col, max_up_col, max_dn_col]):
        st.caption("ℹ️ To populate these metrics, add boolean columns like "
                   "`hit_up20`, `hit_up50`, `hit_up100`, `hit_dn20`, `hit_dn50`, `hit_dn100` "
                   "or numeric `max_up_ext` / `max_dn_ext` (fractions of OR).")

    # ---- Display the filtered table with friendly headers + boolean highlighting
    rename = {
        "trade_date":"Date","symbol":"Symbol","or_window":"OR Time",
        "orh":"ORH","orl":"ORL","or_range":"OR Range",
        "first_break":"First Break","broke_up":"Broke Up",
        "broke_down":"Broke Down","broke_both":"Broke Both",
    }
    table = df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()

    # Limit display columns to a clean set (optional)
    show_cols = [c for c in ["Date","Symbol","OR Time","ORH","ORL","OR Range","First Break","Broke Up","Broke Down","Broke Both"] if c in table.columns]
    if show_cols:
        table = table[show_cols]

    # Re-round for display safety
    for c in ["ORH","ORL","OR Range"]:
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce").round(2)

    # === Match your green highlighting ===
    def _color_hits(val):
        if val is True or str(val).lower() == "true":
            return "background-color: #98FB98"
        return ""

    def _detect_bool_like_columns(dfx: pd.DataFrame):
        bool_cols = []
        for c in dfx.columns:
            s = dfx[c]
            if s.dtype == bool:
                bool_cols.append(c)
            else:
                nonnull = s.dropna().astype(str).str.lower().unique()
                if len(nonnull) > 0 and set(nonnull).issubset({"true", "false"}):
                    bool_cols.append(c)
        return bool_cols

    styler = table.style.hide(axis="index")
    table_styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {"selector": "th, td", "props": [("border", "1px solid #E5E7EB"), ("padding", "6px 8px")]},
        {"selector": "thead th", "props": [("font-weight", "700"), ("color", "#000")]}
    ]
    styler = styler.set_table_styles(table_styles)

    bool_like = _detect_bool_like_columns(table)
    if bool_like:
        styler = styler.map(_color_hits, subset=bool_like)

    # Render with same sticky header/scroll container pattern you use in app.py
    html_table = styler.to_html()
    st.markdown("""
        <style>
        .scroll-table-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #E5E7EB;
        }
        .scroll-table-container table {
            width: 100%;
            border-collapse: collapse;
        }
        .scroll-table-container thead th {
            position: sticky;
            top: 0;
            z-index: 3;
            background-color: #d0d0d0 !important;
            color: #000;
            font-weight: 700;
            background-image: linear-gradient(to bottom,
                rgba(0,0,0,0) calc(100% - 3px),
                #000 calc(100% - 3px),
                #000 100%
            ) !important;
            background-clip: padding-box;
            border-bottom: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="scroll-table-container">{html_table}</div>', unsafe_allow_html=True)

    # ---- Download (raw df so you keep all columns)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "spx_opening_range_stats_filtered.csv",
        "text/csv"
    )

    return True
