import streamlit as st
import pandas as pd
from supabase import create_client
from datetime import date, timedelta
import re
from typing import Optional, List, Dict, Tuple

# -------------------- Column label aliases for fetch_current_levels --------------------
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
    ("S3",    ["S3","r3"]),
]

# Tables we consider “pivot level” sets (ES + GC)
PIVOT_TABLES = {
    "es_daily_pivot_levels",
    "es_weekly_pivot_levels",
    "es_2hr_pivot_levels",
    "es_4hr_pivot_levels",
    "es_30m_pivot_levels",
    "es_rth_pivot_levels",
    "es_on_pivot_levels",
    "gc_daily_pivot_levels",
    "gc_weekly_pivot_levels",
}

# -------------------- Small utilities --------------------
def _normalize_table_name(name: str) -> str:
    return (name or "").strip().split(".")[-1].lower()

def _first_present(rec: dict, keys: List[str]):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None

def _first_existing_datecol(rec: dict, preferred: str) -> str:
    candidates = [preferred, "trade_date", "date", "time"]
    for c in candidates:
        if c in rec:
            return c
    return preferred

# -------------------- Current levels (Pivot/R/S…) header block --------------------
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

# -------------------- SPX Daily metrics (right-side tiles) --------------------
def render_spx_daily_metrics(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    dff = df.copy()

    # Keep trade_date display clean inside the df used for metrics (no 00:00:00)
    if "trade_date" in dff.columns:
        td = pd.to_datetime(dff["trade_date"], errors="coerce")
        dff["trade_date"] = td.dt.strftime("%Y-%m-%d")

    def _to_bool(col: str) -> pd.Series:
        if col not in dff:
            return pd.Series(dtype="boolean")
        s = dff[col]
        if s.dtype == bool:
            return s.astype("boolean")
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype("boolean")
        return (
            s.astype(str).str.strip().str.lower()
             .map({"true": True, "1": True, "1.0": True, "yes": True, "y": True,
                   "false": False, "0": False, "0.0": False, "no": False, "n": False})
             .astype("boolean")
        )

    def _num(col: str) -> pd.Series:
        if col not in dff:
            return pd.Series(dtype="float64")
        return pd.to_numeric(dff[col], errors="coerce")

    def _pct(series: pd.Series) -> str:
        s = series.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _avg_gt1(series: pd.Series) -> str:
        s = pd.to_numeric(series, errors="coerce")
        s = s[(s > 1.0) & s.notna()]
        return "–" if s.empty else f"{s.mean():.2f}"

    cols_lower = {c.lower(): c for c in dff.columns}
    get = lambda name: cols_lower.get(name.lower())

    # Breaks (AM)
    ibh_broke  = _to_bool(get("ibh_broke_am") or "ibh_broke_am")
    ibl_broke  = _to_bool(get("ibl_broke_am") or "ibl_broke_am")
    both_broke = _to_bool(get("both_ib_broke_am") or "both_ib_broke_am")
    either_broke = (ibh_broke | ibl_broke) if (len(ibh_broke) and len(ibl_broke)) else pd.Series(dtype="boolean")

    # PM Extensions
    pm_up = _to_bool(get("pm_ext_up") or "pm_ext_up")
    pm_dn = _to_bool(get("pm_ext_down") or "pm_ext_down")
    either_ext = (pm_up | pm_dn) if (len(pm_up) and len(pm_dn)) else pd.Series(dtype="boolean")
    both_ext   = (pm_up & pm_dn) if (len(pm_up) and len(pm_dn)) else pd.Series(dtype="boolean")

    # Ranges / averages
    ibh     = _num(get("ibh") or "ibh")
    ibl     = _num(get("ibl") or "ibl")
    rth_hi  = _num(get("rth_hi") or "rth_hi")
    rth_lo  = _num(get("rth_lo") or "rth_lo")
    ib_ext  = _num(get("ib_ext") or "ib_ext")

    ib_ext_up_am   = _num(get("ib_ext_up_am") or "ib_ext_up_am")
    ib_ext_down_am = _num(get("ib_ext_down_am") or "ib_ext_down_am")
    pm_ext         = _num(get("pm_ext") or "pm_ext")

    ib_range  = (ibh - ibl)
    rth_range = (rth_hi - rth_lo)
    days = len(dff)

    # Afternoon IB Stats (PM hits)
    ibh_pm_hit    = _to_bool(get("ibh_pm_hit") or "ibh_pm_hit")
    ibl_pm_hit    = _to_bool(get("ibl_pm_hit") or "ibl_pm_hit")
    ib_mid_pm_hit = _to_bool(get("ib_mid_pm_hit") or "ib_mid_pm_hit")
    either_ib_pm_hit = (ibh_pm_hit | ibl_pm_hit) if (len(ibh_pm_hit) and len(ibl_pm_hit)) else pd.Series(dtype="boolean")
    both_ib_pm_hit   = (ibh_pm_hit & ibl_pm_hit) if (len(ibh_pm_hit) and len(ibl_pm_hit)) else pd.Series(dtype="boolean")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### Break Statistics")
        st.markdown(f"**Days:** {days:,}")
        st.markdown(f"**IBH Broke AM:** {_pct(ibh_broke)}")
        st.markdown(f"**IBL Broke AM:** {_pct(ibl_broke)}")
        st.markdown(f"**Either IB Broke AM:** {_pct(either_broke)}")
        st.markdown(f"**Both IB Broke AM:** {_pct(both_broke)}")

    with c2:
        st.markdown("### PM Extensions")
        st.markdown(f"**PM Ext Up:** {_pct(pm_up)}")
        st.markdown(f"**PM Ext Down:** {_pct(pm_dn)}")
        st.markdown(f"**Either Range Extended:** {_pct(either_ext)}")
        st.markdown(f"**Both Ranges Extended:** {_pct(both_ext)}")

    with c3:
        st.markdown("### Averages")
        avg_ib_ext  = "–" if ib_ext.dropna().empty else f"{ib_ext.mean():.2f}"
        avg_ib_rng  = "–" if ib_range.dropna().empty else f"{ib_range.mean():.2f}"
        avg_rth_rng = "–" if rth_range.dropna().empty else f"{rth_range.mean():.2f}"

        # ✅ New averages requested
        avg_ib_ext_up_am   = _avg_gt1(ib_ext_up_am)
        avg_ib_ext_down_am = _avg_gt1(ib_ext_down_am)
        avg_pm_ext         = "–" if pm_ext.dropna().empty else f"{pm_ext.dropna().mean():.2f}"

        st.markdown(f"**Avg IB Ext:** {avg_ib_ext}")
        st.markdown(f"**Avg IB Ext Up AM (>1):** {avg_ib_ext_up_am}")
        st.markdown(f"**Avg IB Ext Down AM (>1):** {avg_ib_ext_down_am}")
        st.markdown(f"**Avg PM Ext:** {avg_pm_ext}")
        st.markdown(f"**Avg IB Range:** {avg_ib_rng}")
        st.markdown(f"**Avg RTH Range:** {avg_rth_rng}")

    with c4:
        st.markdown("### Afternoon IB Stats")
        st.markdown(f"**IBH Hit in PM:** {_pct(ibh_pm_hit)}")
        st.markdown(f"**IBL Hit in PM:** {_pct(ibl_pm_hit)}")
        st.markdown(f"**IB Mid Hit in PM:** {_pct(ib_mid_pm_hit)}")
        st.markdown(f"**Either IB Hit in PM:** {_pct(either_ib_pm_hit)}")
        st.markdown(f"**Both IB Hit in PM:** {_pct(both_ib_pm_hit)}")

# -------------------- Euro IB header tiles --------------------
def render_euro_ib_metrics(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

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

# -------------------- Pivot stats block (Daily / RTH / ON) --------------------
def render_pivot_stats(choice: str, df: pd.DataFrame) -> None:
    valid = {
        "Daily Pivots","ES Daily Pivots","GC Daily Pivots",
        "Weekly Pivots","ES Weekly Pivots","GC Weekly Pivots",
        "ES RTH Pivots","ES ON Pivots",
    }
    if choice not in valid or df is None or df.empty:
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
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype("boolean")
        return (
            s.astype(str).str.strip().str.lower()
             .map({"true": True, "1": True, "1.0": True, "yes": True, "y": True,
                   "false": False, "0": False, "0.0": False, "no": False, "n": False})
             .astype("boolean")
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

    left_rows = [("Days", f"{days:,}", "")]
    if c_hit_pivot:
        left_rows.append(("Hit Pivot", _rate(_to_bool(c_hit_pivot)), ""))

    for label, rcol, scol, _, _ in pairs:
        sr = _to_bool(rcol)
        ss = _to_bool(scol)
        either = _rate(sr | ss)
        both   = _rate(sr & ss)
        left_rows.append((f"{label} Either", either, both))

    left_df = pd.DataFrame(left_rows, columns=["Pair", "Either", "Both"])

    right_rows = []
    for _, rcol, scol, rname, sname in pairs:
        r_rate = _rate(_to_bool(rcol)) if rcol else "–"
        s_rate = _rate(_to_bool(scol)) if scol else "–"
        right_rows.append((f"{rname}/{sname}", r_rate, s_rate))
    right_df = pd.DataFrame(right_rows, columns=["Level", "R", "S"])

    def _df_height(n_rows: int) -> int:
        row_px, header_px, fudge = 34, 36, 0
        return header_px + n_rows * row_px - fudge

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Pivot & Pair Hits")
        st.dataframe(
            left_df, hide_index=True, use_container_width=False, height=max(120, _df_height(len(left_df))),
            column_config={
                "Pair":   st.column_config.TextColumn("Pair", width=220),
                "Either": st.column_config.TextColumn("Either", width=100),
                "Both":   st.column_config.TextColumn("Both", width=100),
            },
        )
    with right:
        st.subheader("Individual Levels")
        st.dataframe(
            right_df, hide_index=True, use_container_width=False, height=max(120, _df_height(len(right_df))),
            column_config={
                "Level": st.column_config.TextColumn("Level", width=160),
                "R":     st.column_config.TextColumn("R", width=100),
                "S":     st.column_config.TextColumn("S", width=100),
            },
        )

# ==================== GC Levels (dynamic probabilities for specified fields) ====================
_WEEKDAY_ABBR = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def _normalize_key(s: str) -> str:
    s = s.lower().replace("×", "x")
    s = re.sub(r"[^a-z0-9x]+", "", s)
    return s

def _build_ci_map(dff: pd.DataFrame) -> dict:
    return {_normalize_key(c): c for c in dff.columns}

def _find_norm(dff: pd.DataFrame, *candidates: str) -> Optional[str]:
    if dff is None or dff.empty:
        return None
    m = _build_ci_map(dff)
    for cand in candidates:
        k = _normalize_key(cand)
        if k in m:
            return m[k]
    return None

def _ensure_day_col(dff: pd.DataFrame) -> pd.DataFrame:
    if "Day" in dff.columns:
        dff["Day"] = pd.Categorical(dff["Day"], categories=_WEEKDAY_ABBR, ordered=True)
        return dff
    for cand in ["globex_date", "trade_date", "date", "session_date", "time", "timestamp"]:
        if cand in dff.columns:
            dd = pd.to_datetime(dff[cand], errors="coerce")
            dff["Day"] = dd.dt.day_name().str.slice(0, 3)
            break
    if "Day" not in dff.columns:
        dff["Day"] = None
    dff["Day"] = pd.Categorical(dff["Day"], categories=_WEEKDAY_ABBR, ordered=True)
    return dff

def _to_bool_series(dff: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in dff:
        return pd.Series([], dtype="boolean")
    s = dff[col]
    if s.dtype == bool:
        return s.astype("boolean")
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype("boolean")
    return (
        s.astype(str).str.strip().str.lower()
         .map({"true": True, "1": True, "1.0": True, "yes": True, "y": True,
               "false": False, "0": False, "0.0": False, "no": False, "n": False})
         .astype("boolean")
    )

def _pct_mean_bool(s: pd.Series) -> str:
    s = s.dropna()
    return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

def render_gc_levels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        st.info("No data in gc_levels for the current filters.")
        return df

    dff = df.copy()
    dff = _ensure_day_col(dff)

    get = lambda *names: _find_norm(dff, *names)

    c_ibh_pm  = get("aIBH Broke Premarket", "aibh broke premarket", "aibh_broke_premarket")
    c_ibl_pm  = get("aIBL Broke Premarket", "aibl broke premarket", "aibl_broke_premarket")
    c_mid_pm  = get("aIB Mid Hit Premarket", "aib mid hit premarket", "aib_mid_hit_premarket")

    c_ibh_adj = get("aIBH Broke Adj RTH", "aibh broke adj rth", "aibh broke adjusted rth", "aibh_broke_adj_rth")
    c_ibl_adj = get("aIBL Broke Adj RTH", "aibl broke adj rth", "aibl broke adjusted rth", "aibl_broke_adj_rth")

    c_mid_rth = get("aIB Mid Hit RTH", "aib mid hit rth", "aib_mid_hit_rth")

    c_ibh_12 = get("aIBH1.2 - Hit RTH", "aIBH1.2x - Hit RTH",
                   "aibh12 hit rth", "aibh12x hit rth", "aibh1_2 hit rth", "aibh1_2x hit rth", "aibh12x_hit_rth")
    c_ibh_15 = get("aIBH1.5 - Hit RTH", "aIBH1.5x - Hit RTH",
                   "aibh15 hit rth", "aibh15x hit rth", "aibh1_5 hit rth", "aibh1_5x hit rth", "aibh15x_hit_rth")
    c_ibh_2x = get("aIBH2x - Hit RTH",  "aibh2x - hit rth",  "aibh2x hit rth", "aibh2x_hit_rth")

    c_ibl_12 = get("aIBL1.2x - Hit RTH", "aibl1.2x - hit rth", "aibl12x hit rth", "aibl1_2x hit rth", "aibl12 hit rth", "aibl1_2 hit rth", "aibl12x_hit_rth")
    c_ibl_15 = get("aIBL1.5x - Hit RTH", "aibl1.5x - hit rth", "aibl15x hit rth", "aibl1_5x hit rth", "aibl15 hit rth", "aibl1_5 hit rth", "aibl15x_hit_rth")
    c_ibl_2x = get("aIBL2x - Hit RTH",   "aibl2x - hit rth",   "aibl2x hit rth", "aibl2x_hit_rth")

    c_onh = get("broke_onh", "Broke ONH", "Hit ONH")
    c_onl = get("broke_onl", "Broke ONL", "Hit ONL")

    s_ibh_pm = _to_bool_series(dff, c_ibh_pm)
    s_ibl_pm = _to_bool_series(dff, c_ibl_pm)
    s_mid_pm = _to_bool_series(dff, c_mid_pm)

    pm_either = _pct_mean_bool((s_ibh_pm | s_ibl_pm) if len(s_ibh_pm) and len(s_ibl_pm) else pd.Series(dtype="boolean"))
    pm_both   = _pct_mean_bool((s_ibh_pm & s_ibl_pm) if len(s_ibh_pm) and len(s_ibl_pm) else pd.Series(dtype="boolean"))

    row_pm = {
        "Days": f"{len(dff):,}",
        "aIBH Broke Premarket": _pct_mean_bool(s_ibh_pm),
        "aIBL Broke Premarket": _pct_mean_bool(s_ibl_pm),
        "Either Premarket": pm_either,
        "Both Premarket":   pm_both,
        "aIB Mid Hit Premarket": _pct_mean_bool(s_mid_pm),
    }

    s_ibh_adj = _to_bool_series(dff, c_ibh_adj)
    s_ibl_adj = _to_bool_series(dff, c_ibl_adj)
    s_mid_rth = _to_bool_series(dff, c_mid_rth)

    adj_either = _pct_mean_bool((s_ibh_adj | s_ibl_adj) if len(s_ibh_adj) and len(s_ibl_adj) else pd.Series(dtype="boolean"))
    adj_both   = _pct_mean_bool((s_ibh_adj & s_ibl_adj) if len(s_ibh_adj) and len(s_ibl_adj) else pd.Series(dtype="boolean"))

    row_adj = {
        "aIBH Broke Adj RTH": _pct_mean_bool(s_ibh_adj),
        "aIBL Broke Adj RTH": _pct_mean_bool(s_ibl_adj),
        "Either Adj RTH": adj_either,
        "Both Adj RTH":   adj_both,
        "aIB Mid Hit RTH": _pct_mean_bool(s_mid_rth),
    }

    s_ibh_12 = _to_bool_series(dff, c_ibh_12)
    s_ibh_15 = _to_bool_series(dff, c_ibh_15)
    s_ibh_2x = _to_bool_series(dff, c_ibh_2x)
    s_ibl_12 = _to_bool_series(dff, c_ibl_12)
    s_ibl_15 = _to_bool_series(dff, c_ibl_15)
    s_ibl_2x = _to_bool_series(dff, c_ibl_2x)

    row_hits = {
        "aIBH 1.2× — Hit RTH": _pct_mean_bool(s_ibh_12),
        "aIBH 1.5× — Hit RTH": _pct_mean_bool(s_ibh_15),
        "aIBH 2×   — Hit RTH": _pct_mean_bool(s_ibh_2x),
        "aIBL 1.2× — Hit RTH": _pct_mean_bool(s_ibl_12),
        "aIBL 1.5× — Hit RTH": _pct_mean_bool(s_ibl_15),
        "aIBL 2×   — Hit RTH": _pct_mean_bool(s_ibl_2x),
    }

    s_onh = _to_bool_series(dff, c_onh)
    s_onl = _to_bool_series(dff, c_onl)

    on_either = _pct_mean_bool((s_onh | s_onl) if len(s_onh) and len(s_onl) else pd.Series(dtype="boolean"))
    on_both   = _pct_mean_bool((s_onh & s_onl) if len(s_onh) and len(s_onl) else pd.Series(dtype="boolean"))

    row_on = {
        "Break ONH": _pct_mean_bool(s_onh),
        "Break ONL": _pct_mean_bool(s_onl),
        "Break Either ON": on_either,
        "Break Both ON":   on_both,
    }

    col1, col2, col3, col4 = st.columns(4)

    def _render_block(col, title, d):
        with col:
            st.markdown(f"### {title}")
            for k, v in d.items():
                st.markdown(
                    f"""
                    <div style='padding:4px 0; margin-bottom:2px; border-bottom:1px solid #ddd;'>
                        <strong>{k}:</strong> {v}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    _render_block(col1, "Premarket Breaks", row_pm)
    _render_block(col2, "Adjusted RTH Breaks", row_adj)
    _render_block(col3, "RTH Extension Hits", row_hits)
    _render_block(col4, "ON Extreme Stats", row_on)

    return dff

# -------------------- SPX Opening Range (filter + header tiles) --------------------
def spx_opening_range_filter_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "or_window" not in df.columns:
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
        if not col or col not in dfX:
            return "–"
        s = dfX[col]
        if pd.api.types.is_numeric_dtype(s):
            s = (pd.to_numeric(s, errors="coerce").fillna(0) > 0)
        else:
            s = s.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "1.0": True, "yes": True, "y": True,
                 "false": False, "0": False, "0.0": False, "no": False, "n": False}
            )
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _rate_from_numeric(dfX, col, thr):
        if not col or not (col in dfX):
            return "–"
        s = pd.to_numeric(dfX[col], errors="coerce").dropna()
        return "–" if s.empty else f"{100.0 * (s >= thr).mean():.1f}%"

    def _pct_bool_mean(colname):
        if not colname or colname not in dff:
            return "–"
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

    up20  = _find_col_ci(["hit_20_up"])
    up50  = _find_col_ci(["hit_50_up"])
    up100 = _find_col_ci(["hit_100_up"])

    dn20  = _find_col_ci(["hit_20_down"])
    dn50  = _find_col_ci(["hit_50_down"])
    dn100 = _find_col_ci(["hit_100_down"])

    max_up = _find_col_ci(["max_ext_up"])
    max_dn = _find_col_ci(["max_ext_down"])

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

# -------------------- GC Opening Range (filter + header tiles) --------------------
def gc_opening_range_filter_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "or_window" not in df.columns:
        return df

    win = st.sidebar.selectbox("OR Window (GC)", ["5m", "30m"], index=0, key="gc_or_window")
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
        if not col or col not in dfX:
            return "–"
        s = dfX[col]
        if pd.api.types.is_numeric_dtype(s):
            s = (pd.to_numeric(s, errors="coerce").fillna(0) > 0)
        else:
            s = s.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "1.0": True, "yes": True, "y": True,
                 "false": False, "0": False, "0.0": False, "no": False, "n": False}
            )
        s = s.dropna()
        return "–" if s.empty else f"{100.0 * s.mean():.1f}%"

    def _rate_from_numeric(dfX, col, thr):
        if not col or not (col in dfX):
            return "–"
        s = pd.to_numeric(dfX[col], errors="coerce").dropna()
        return "–" if s.empty else f"{100.0 * (s >= thr).mean():.1f}%"

    def _pct_bool_mean(colname):
        if not colname or colname not in dff:
            return "–"
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

    up20  = _find_col_ci(["hit_20_up"])
    up50  = _find_col_ci(["hit_50_up"])
    up100 = _find_col_ci(["hit_100_up"])

    dn20  = _find_col_ci(["hit_20_down"])
    dn50  = _find_col_ci(["hit_50_down"])
    dn100 = _find_col_ci(["hit_100_down"])

    max_up = _find_col_ci(["max_ext_up"])
    max_dn = _find_col_ci(["max_ext_down"])

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

# Backwards-compatibility stub (kept no-op)
def render_view_override(view_id: str) -> bool:
    return False
