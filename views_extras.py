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
def render_euro_ib_metrics(df):
    """Render Euro-IB stats in a 3-column vertical layout."""

    # --- Compute all probabilities dynamically ---
    days = len(df)

    pct = lambda x: f"{(x / days * 100):.1f}%" if days > 0 else "0.0%"

    # ROW 1 (Break Stats)
    row1 = {
        "Days": days,
        "% eIBH Break": pct(df["eibh_break"].sum()),
        "% eIBL Break": pct(df["eibl_break"].sum()),
        "% Either Break": pct((df["eibh_break"] | df["eibl_break"]).sum()),
        "% Both Break": pct((df["eibh_break"] & df["eibl_break"]).sum()),
    }

    # ROW 2 (Extension Hits)
    row2 = {
        "% 1.2× Hit": pct(df[["eibh12_hit","eibl12_hit"]].any(axis=1).sum()),
        "% 1.5× Hit": pct(df[["eibh15_hit","eibl15_hit"]].any(axis=1).sum()),
        "% 2.0× Hit": pct(df[["eibh20_hit","eibl20_hit"]].any(axis=1).sum()),
    }

    # ROW 3 (RTH Intraday Hits)
    row3 = {
        "% IBH Hit RTH": pct(df["eur_ibh_rth_hit"].sum()),
        "% IBL Hit RTH": pct(df["eur_ibl_rth_hit"].sum()),
        "% Either Hit RTH": pct((df["eur_ibh_rth_hit"] | df["eur_ibl_rth_hit"]).sum()),
        "% Both Hit RTH": pct((df["eur_ibh_rth_hit"] & df["eur_ibl_rth_hit"]).sum()),
    }

    # --- DISPLAY AS THREE COLUMNS ---
    col1, col2, col3 = st.columns(3)

    def render_block(col, title, data_dict):
        with col:
            st.markdown(f"### {title}")
            for k, v in data_dict.items():
                st.markdown(
                    f"""
                    <div style='padding:4px 0; margin-bottom:2px;
                                border-bottom:1px solid #ddd;'>
                        <strong>{k}:</strong> {v}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_block(col1, "Break Statistics", row1)
    render_block(col2, "Extension Hits", row2)
    render_block(col3, "Intraday RTH Hits", row3)

# --- Pivot tables (Daily / RTH / ON) dynamic stats ---------------------------------
def render_pivot_stats_compact(df, title="Stats"):
    """
    Compact 2-column layout with tight spacing:
      • Left column: Days, Hit Pivot, and the 'Either / Both' pairs (R0.25..R3)
      • Right column: Individual level pairs (R / S)
    """

    if df is None or df.empty:
        return

    # ---------- helpers ----------
    def _to_bool(s):
        if s not in df.columns:
            return None
        x = df[s]
        if x.dtype == bool:
            return x
        return x.astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        )

    def _pct_from_bool(series):
        if series is None:
            return "–"
        s = series.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _pct_from_either(col_a, col_b):
        a = _to_bool(col_a); b = _to_bool(col_b)
        if a is None and b is None:
            return "–"
        if a is None: a = pd.Series(False, index=df.index)
        if b is None: b = pd.Series(False, index=df.index)
        s = (a | b).astype("boolean").dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _pct_from_both(col_a, col_b):
        a = _to_bool(col_a); b = _to_bool(col_b)
        if a is None or b is None:
            return "–"
        s = (a & b).astype("boolean").dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _pct(col):
        return _pct_from_bool(_to_bool(col))

    # ---------- compute stats ----------
    days = len(df)
    hit_pivot = _pct("hit_pivot")

    pairs = [
        ("R0.25/S0.25", "hit_r025", "hit_s025"),
        ("R0.5/S0.5",   "hit_r05",  "hit_s05"),
        ("R1/S1",       "hit_r1",   "hit_s1"),
        ("R1.5/S1.5",   "hit_r15",  "hit_s15"),
        ("R2/S2",       "hit_r2",   "hit_s2"),
        ("R3/S3",       "hit_r3",   "hit_s3"),
    ]

    # Left column rows: Days, Hit Pivot, plus Either/Both lines
    left_rows = [("single", "Days", f"{days:,}", None),
                 ("single", "Hit Pivot", hit_pivot, None)]
    for label, r_col, s_col in pairs:
        left_rows.append(("pair", f"{label} Either", _pct_from_either(r_col, s_col),
                          ("Both", _pct_from_both(r_col, s_col))))

    # Right column rows: individual R/S on one line
    right_rows = []
    for label, r_col, s_col in pairs:
        right_rows.append(("pair", label, f"R: {_pct(r_col)}", ("S", _pct(s_col))))

    # ---------- tighter column spacing between the two big sections ----------
    # Use Streamlit's "small" gap and a slightly biased ratio to pull columns closer.
    col_left, col_right = st.columns([1.12, 0.88], gap="small")

    # ---------- CSS to fix “value pushed to the other column” & tighten gaps ----------
    st.markdown("""
        <style>
        /* Make single rows only as wide as their content, so values sit next to labels */
        .pv-row-single {
            display: inline-flex;         /* shrink to content width */
            align-items: baseline;
            gap: 6px;                     /* label ↔ value */
            padding: 2px 0;               /* tight vertical spacing */
            border-bottom: 1px solid #eee;
            white-space: nowrap;          /* keep on one line */
        }
        .pv-row-single strong { font-weight: 600; }

        /* Two tight pairs per row:
           [Either label][Either value][spacer][Both label][Both value]  */
        .pv-row-pair {
            display: grid;
            grid-template-columns: max-content max-content 12px max-content max-content;
            column-gap: 8px;              /* ≈ one tab between cells */
            align-items: baseline;
            padding: 2px 0;               /* tight vertical spacing */
            border-bottom: 1px solid #eee;
            width: fit-content;           /* shrink to content width */
        }
        .pv-leftcell  { font-weight: 600; }
        .pv-rightcell { text-align: right; }
        </style>
    """, unsafe_allow_html=True)

    def _render_single(col, label, value):
        with col:
            st.markdown(
                f"""
                <div class="pv-row-single">
                    <div><strong>{label}:</strong></div>
                    <div>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_pair(col, left_label, left_value, right_pair):
        r_lbl, r_val = right_pair if isinstance(right_pair, tuple) else ("", "")
        with col:
            st.markdown(
                f"""
                <div class="pv-row-pair">
                    <div class="pv-leftcell">{left_label}:</div>
                    <div>{left_value}</div>
                    <div></div>  <!-- small fixed spacer between groups -->
                    <div class="pv-leftcell">{r_lbl}:</div>
                    <div class="pv-rightcell">{r_val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------- render ----------
    with col_left:
        st.markdown(f"### {title}")
        for kind, a, b, c in left_rows:
            if kind == "single":
                _render_single(col_left, a, b)
            else:
                _render_pair(col_left, a, b, c)

    with col_right:
        st.markdown("### Individual Levels")
        for _, a, b, c in right_rows:
            _render_pair(col_right, a, b, c)


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

    # --- Sidebar: choose a single window (3m/5m/15m)
    win = st.sidebar.selectbox("OR Window (SPX)", ["3m", "5m", "15m"], index=0, key="spx_or_window")
    dff = df[df["or_window"] == win].copy()

    # Ensure date ordering ASC so latest is at the bottom (matches App.py)
    if "trade_date" in dff.columns:
        dff["trade_date"] = pd.to_datetime(dff["trade_date"], errors="coerce")
        dff = dff.sort_values("trade_date", ascending=True).reset_index(drop=True)

    # ---------- Helpers ----------
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
        s = dfX[col].map(lambda v: True if str(v).strip().lower() in {"true","1","yes"} else
                                   (False if str(v).strip().lower() in {"false","0","no"} else None))
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

    # ---------- Break stats (left column) ----------
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

    # ---------- Extension hits (right column) ----------
    up20  = _find_col_ci(["hit_20_up",  "hitup20",  "hit_up20",  "up20",  "or_up_20",  "hit_or_up_20"])
    up50  = _find_col_ci(["hit_50_up",  "hitup50",  "hit_up50",  "up50",  "or_up_50",  "hit_or_up_50"])
    up100 = _find_col_ci(["hit_100_up", "hitup100", "hit_up100", "up100", "or_up_100", "hit_or_up_100"])

    dn20  = _find_col_ci(["hit_20_down",  "hitdn20",  "hit_down20",  "down20",  "or_dn_20",  "hit_or_dn_20"])
    dn50  = _find_col_ci(["hit_50_down",  "hitdn50",  "hit_down50",  "down50",  "or_dn_50",  "hit_or_dn_50"])
    dn100 = _find_col_ci(["hit_100_down", "hitdn100", "hit_down100", "down100", "or_dn_100", "hit_or_dn_100"])

    # numeric fallbacks (fractions of OR)
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
