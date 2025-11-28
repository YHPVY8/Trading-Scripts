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
    if "broke_up"   in df: c2.metric("Broke Up",   f"{100*pd.to_numeric(df['broke_up'],   errors='coerce').mean():.1f}%")
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
    candidates = [preferred, "trade_date", "date", "time"]
    for c in candidates:
        if c in rec:
            return c
    return preferred

def fetch_current_levels(sb, table_name: str, date_col: str) -> dict:
    try:
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
        st.caption(f"ℹ️ Skipping levels (table '{table_name}' not recognized as a pivot-level table).")
        return

    levels = fetch_current_levels(sb, table_name, date_col)

    if not any(levels.values()):
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
def render_euro_ib_metrics(df):
    """Render Euro-IB stats in a 3-column vertical layout."""
    days = len(df)
    pct = lambda x: f"{(x / days * 100):.1f}%" if days > 0 else "0.0%"

    row1 = {
        "Days": days,
        "% eIBH Break": pct(df["eibh_break"].sum()),
        "% eIBL Break": pct(df["eibl_break"].sum()),
        "% Either Break": pct((df["eibh_break"] | df["eibl_break"]).sum()),
        "% Both Break": pct((df["eibh_break"] & df["eibl_break"]).sum()),
    }

    row2 = {
        "% 1.2× Hit": pct(df[["eibh12_hit","eibl12_hit"]].any(axis=1).sum()),
        "% 1.5× Hit": pct(df[["eibh15_hit","eibl15_hit"]].any(axis=1).sum()),
        "% 2.0× Hit": pct(df[["eibh20_hit","eibl20_hit"]].any(axis=1).sum()),
    }

    row3 = {
        "% IBH Hit RTH": pct(df["eur_ibh_rth_hit"].sum()),
        "% IBL Hit RTH": pct(df["eur_ibl_rth_hit"].sum()),
        "% Either Hit RTH": pct((df["eur_ibh_rth_hit"] | df["eur_ibl_rth_hit"]).sum()),
        "% Both Hit RTH": pct((df["eur_ibh_rth_hit"] & df["eur_ibl_rth_hit"]).sum()),
    }

    col1, col2, col3 = st.columns(3)

    def render_block(col, title, data_dict):
        with col:
            st.markdown(f"### {title}")
            for k, v in data_dict.items():
                st.markdown(
                    f"""
                    <div style='padding:4px 0; margin-bottom:2px; border-bottom:1px solid #ddd;'>
                        <strong>{k}:</strong> {v}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_block(col1, "Break Statistics", row1)
    render_block(col2, "Extension Hits", row2)
    render_block(col3, "Intraday RTH Hits", row3)

# --- Pivot tables (Daily / RTH / ON) dynamic stats ---------------------------------
def render_pivot_stats(choice: str, df: pd.DataFrame) -> None:
    """
    Compact two-column stats with tighter spacing and inline 'Days'/'Hit Pivot' values.
    Keeps original function name/signature for compatibility.
    """
    if choice not in {"Daily Pivots", "RTH Pivots", "ON Pivots"}:
        return
    if df is None or df.empty:
        return

    dff = df.copy()
    cols_lower = {c.lower(): c for c in dff.columns}

    def _col(name: str):
        return cols_lower.get(name.lower())

    def _to_bool(colname: str) -> pd.Series:
        if not colname or colname not in dff:
            return pd.Series([], dtype="boolean")
        s = dff[colname]
        if s.dtype == bool:
            return s.astype("boolean")
        return (
            s.astype(str).str.strip().str.lower().map({
                "true": True, "1": True, "yes": True, "y": True,
                "false": False, "0": False, "no": False, "n": False
            }).astype("boolean")
        )

    def _rate(series: pd.Series) -> str:
        s = series.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    days = len(dff)
    c_hit_pivot = _col("hit_pivot")

    pairs = [
        ("R0.25/S0.25", _col("hit_r025"), _col("hit_s025"), "R0.25", "S0.25"),
        ("R0.5/S0.5",   _col("hit_r05"),  _col("hit_s05"),  "R0.5",  "S0.5"),
        ("R1/S1",       _col("hit_r1"),   _col("hit_s1"),   "R1",    "S1"),
        ("R1.5/S1.5",   _col("hit_r15"),  _col("hit_s15"),  "R1.5",  "S1.5"),
        ("R2/S2",       _col("hit_r2"),   _col("hit_s2"),   "R2",    "S2"),
        ("R3/S3",       _col("hit_r3"),   _col("hit_s3"),   "R3",    "S3"),
    ]

    # ---------- CSS for tight layout and no-wrap inline rows ----------
    st.markdown("""
        <style>
          /* Outer wrapper: 2 columns with tighter gap */
          .pv-wrap {
            display: grid;
            grid-template-columns: 1fr 1fr;
            column-gap: 14px;    /* tighten distance between the two panels */
            row-gap: 0.5rem;
            margin: 0.25rem 0 0.5rem 0;
          }
          .pv-hdr { font-size: 1.15rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; }

          /* Single line label/value (Days, Hit Pivot). Prevent wrap into other cells. */
          .pv-line {
            display: grid;
            grid-template-columns: auto 1fr;
            align-items: baseline;
            column-gap: 8px;
            padding: 2px 0;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
          }
          .pv-label { font-weight: 600; }

          /* Four-cell rows: (left_label | left_value | right_label | right_value) */
          .pv-row-4 {
            display: grid;
            grid-template-columns: auto 1fr auto 1fr;
            column-gap: 8px;
            align-items: baseline;
            padding: 2px 0;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
          }
          .pv-val-right { text-align: right; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Build LEFT panel HTML ----------
    left_parts = ['<div>', '<div class="pv-hdr">Pivot &amp; Pair Hits</div>']
    # Days
    left_parts.append(
        f'<div class="pv-line"><div class="pv-label">Days:</div><div>{days:,}</div></div>'
    )
    # Hit Pivot
    if c_hit_pivot:
        left_parts.append(
            f'<div class="pv-line"><div class="pv-label">Hit Pivot:</div><div>{_rate(_to_bool(c_hit_pivot))}</div></div>'
        )
    # Either / Both pairs
    for label, rcol, scol, _, _ in pairs:
        sr = _to_bool(rcol)
        ss = _to_bool(scol)
        either = _rate(sr | ss)
        both   = _rate(sr & ss)
        left_parts.append(
            f'''
            <div class="pv-row-4">
              <div class="pv-label">{label} Either:</div><div class="pv-val-right">{either}</div>
              <div class="pv-label">Both:</div><div class="pv-val-right">{both}</div>
            </div>
            '''
        )
    left_parts.append('</div>')
    left_html = "\n".join(left_parts)

    # ---------- Build RIGHT panel HTML ----------
    right_parts = ['<div>', '<div class="pv-hdr">Individual Levels</div>']
    for _, rcol, scol, rname, sname in pairs:
        r_rate = _rate(_to_bool(rcol)) if rcol else "–"
        s_rate = _rate(_to_bool(scol)) if scol else "–"
        right_parts.append(
            f'''
            <div class="pv-row-4">
              <div class="pv-label">{rname}:</div><div class="pv-val-right">{r_rate}</div>
              <div class="pv-label">{sname}:</div><div class="pv-val-right">{s_rate}</div>
            </div>
            '''
        )
    right_parts.append('</div>')
    right_html = "\n".join(right_parts)

    # ---------- Render two columns with tight gap ----------
    st.markdown(f'''
        <div class="pv-wrap">
          {left_html}
          {right_html}
        </div>
    ''', unsafe_allow_html=True)

# -------------------- New helpers (SPX filter + metrics) --------------------
def _sb():
    """Local Supabase client using Streamlit secrets."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def spx_opening_range_filter_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar control + metrics for SPX Opening Range that returns a filtered DF.
    Renders metrics in a 2-column vertical layout:
      - Left: Break stats
      - Right: Extension hits (Up/Down ≥20/50/100%)
    """
    if df is None or df.empty:
        return df
    if "or_window" not in df.columns:
        return df

    win = st.sidebar.selectbox("OR Window (SPX)", ["3m", "5m", "15m"], index=0, key="spx_or_window")
    dff = df[df["or_window"] == win].copy()

    if "trade_date" in dff.columns:
        dff["trade_date"] = pd.to_datetime(dff["trade_date"], errors="coerce")
        dff = dff.sort_values("trade_date", ascending=True).reset_index(drop=True)

    def _render_block(col, title, items_dict):
        with col:
            st.markdown(f"### {title}")
            for k, v in items_dict.items():
                st.markdown(
                    f"""
                    <div style='padding:4px 0; margin-bottom:2px; border-bottom:1px solid #ddd;'>
                        <strong>{k}:</strong> {v}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    cols_lower = {c.lower(): c for c in dff.columns}
    def _find_col_ci(names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    def _rate_from_bool(dfX, col):
        if not col or col not in dfX: return "–"
        s = dfX[col].map(lambda v: True if str(v).strip().lower() in {"true","1","yes"}
                                   else (False if str(v).strip().lower() in {"false","0","no"} else None))
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _rate_from_numeric(dfX, col, thr):
        if not col or col not in dfX: return "–"
        s = pd.to_numeric(dfX[col], errors="coerce").dropna()
        return "–" if s.empty else f"{100.0 * (s >= thr).mean():.1f}%"

    def _pct_bool_mean(colname):
        if colname not in dff: return "–"
        s = pd.to_numeric(dff[colname], errors="coerce")
        return f"{100.0 * s.mean():.1f}%" if len(s.dropna()) else "–"

    days = len(dff)
    broke_up   = _find_col_ci(["broke_up"])
    broke_down = _find_col_ci(["broke_down"])
    broke_both = _find_col_ci(["broke_both"])

    left_data = {
        "Days": f"{days:,}",
        "Broke Up":   _pct_bool_mean(broke_up)   if broke_up   else "–",
        "Broke Down": _pct_bool_mean(broke_down) if broke_down else "–",
        "Broke Both": _pct_bool_mean(broke_both) if broke_both else "–",
    }

    up20  = _find_col_ci(["hit_20_up",  "hitup20",  "hit_up20",  "up20",  "or_up_20",  "hit_or_up_20"])
    up50  = _find_col_ci(["hit_50_up",  "hitup50",  "hit_up50",  "up50",  "or_up_50",  "hit_or_up_50"])
    up100 = _find_col_ci(["hit_100_up", "hitup100", "hit_up100", "up100", "or_up_100", "hit_or_up_100"])

    dn20  = _find_col_ci(["hit_20_down",  "hitdn20",  "hit_down20",  "down20",  "or_dn_20",  "hit_or_dn_20"])
    dn50  = _find_col_ci(["hit_50_down",  "hitdn50",  "hit_down50",  "down50",  "or_dn_50",  "hit_or_dn_50"])
    dn100 = _find_col_ci(["hit_100_down", "hitdn100", "hit_down100", "down100", "or_dn_100", "hit_or_dn_100"])

    max_up = _find_col_ci(["max_ext_up", "max_up_ext", "max_up_frac", "max_up_or_mult"])
    max_dn = _find_col_ci(["max_ext_down", "max_dn_ext", "max_dn_frac", "max_dn_or_mult"])

    right_data = {
        "Up ≥20%":   (_rate_from_bool(dff, up20)  if up20  else _rate_from_numeric(dff, max_up, 0.20)),
        "Up ≥50%":   (_rate_from_bool(dff, up50)  if up50  else _rate_from_numeric(dff, max_up, 0.50)),
        "Up ≥100%":  (_rate_from_bool(dff, up100) if up100 else _rate_from_numeric(dff, max_up, 1.00)),
        "Down ≥20%": (_rate_from_bool(dff, dn20)  if dn20  else _rate_from_numeric(dff, max_dn, 0.20)),
        "Down ≥50%": (_rate_from_bool(dff, dn50)  if dn50  else _rate_from_numeric(dff, max_dn, 0.50)),
        "Down ≥100%":(_rate_from_bool(dff, dn100) if dn100 else _rate_from_numeric(dff, max_dn, 1.00)),
    }

    col_left, col_right = st.columns(2)
    _render_block(col_left,  "Break Statistics", left_data)
    _render_block(col_right, "Extension Hits",   right_data)

    return dff

# -------------------- Compatibility stub (no-op override) --------------------
def render_view_override(view_id: str) -> bool:
    """
    Deprecated: Do not use table-level override anymore.
    We filter + show metrics via 'spx_opening_range_filter_and_metrics' and
    let App.py's generic renderer handle the table/styling.
    """
    return False
