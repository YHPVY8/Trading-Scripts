#!/usr/bin/env python3
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from supabase import create_client

# =========================
# Page config
# =========================
st.set_page_config(page_title="Conditions", layout="wide")
st.title("Market Conditions (30m)")

# =========================
# Supabase client
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Controls
# =========================
c1, c2, c3, c4 = st.columns([1,1,1,1.2])
with c1:
    symbol = st.text_input("Symbol label (for display only)", value="ES")
with c2:
    lookback_days = st.number_input("Load last N calendar days", 30, 3650, 180, 30)
with c3:
    trailing_window = st.number_input("Trailing window (days)", 5, 60, 10, 1)
with c4:
    mode = st.selectbox("View mode", ["Full day", "As-of time snapshot"])

asof_time = None
if mode == "As-of time snapshot":
    asof_time = st.time_input("As-of time (US/Eastern)", value=dt.time(3, 0))

rth_only = st.toggle("RTH only (09:30–16:00 ET) for charts (does not affect trade-date alignment)", value=False)

# =========================
# Fetch data
# =========================
TABLE = "es_30m"  # adjust if needed
since = (dt.date.today() - dt.timedelta(days=int(lookback_days))).isoformat()
q = sb.table(TABLE).select("*").gte("time", f"{since}T00:00:00Z").order("time", desc=False)
data = q.execute().data
if not data:
    st.warning("No rows returned. Check table name/filters.")
    st.stop()

df = pd.DataFrame(data)

# Expect columns: time, open, high, low, close, Volume, 5MA,10MA,20MA,50MA,200MA
base_cols = ["time","open","high","low","close","Volume"]
for col in base_cols:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

MA_COLS = [c for c in ["5MA","10MA","20MA","50MA","200MA"] if c in df.columns]

# Parse times
df["time"] = pd.to_datetime(df["time"], utc=True)
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")
df["hm"] = df["time_et"].dt.strftime("%H:%M")

# =========================
# Trading-day alignment (Globex → next day)
# Bars >= 18:00 ET belong to next trade date
# =========================
def compute_trade_date(ts_et: pd.Timestamp) -> dt.date:
    if ts_et.tzinfo is None:
        raise ValueError("Expected tz-aware Eastern timestamp")
    if ts_et.time() >= dt.time(18, 0):
        return (ts_et + pd.Timedelta(days=1)).date()
    return ts_et.date()

df["trade_date"] = df["time_et"].apply(compute_trade_date)

# =========================
# Session labels (ON, IB, RTH)
# ON: 18:00–09:30 next morning
# IB: 09:30–10:30
# RTH: 09:30–16:00
# =========================
t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30))).astype(bool)
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30))).astype(bool)
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0))).astype(bool)

# =========================
# As-of filtering (per trade_date)
# “As-of 03:00” means: include bars from 18:00 prev calendar day up to 03:00 of trade_date
# =========================
def mask_asof(df_: pd.DataFrame, asof_t: dt.time) -> pd.Series:
    td = df_["trade_date"]
    d = df_["time_et"].dt.date
    tm = df_["time_et"].dt.time
    cond_today = (d == td) & (tm <= asof_t)
    cond_prev  = (d == (td - pd.to_timedelta(1, unit="D"))) & (tm >= dt.time(18,0))
    return cond_today | cond_prev

if mode == "As-of time snapshot":
    df = df[mask_asof(df, asof_time)].copy()

# Optional RTH-only for CHARTS ONLY (we’ll keep full set for stats already filtered above)
df_for_chart = df.copy()
if rth_only:
    df_for_chart = df_for_chart[(df_for_chart["hm"] >= "09:30") & (df_for_chart["hm"] <= "16:00")]

# =========================
# Prior day levels (pHi/pLo) by trade_date
# =========================
daily_hi_lo = (
    df.groupby("trade_date")
      .agg(day_high=("high","max"), day_low=("low","min"))
      .sort_index()
      .reset_index()
)
daily_hi_lo["pHi"] = daily_hi_lo["day_high"].shift(1)
daily_hi_lo["pLo"] = daily_hi_lo["day_low"].shift(1)
df = df.merge(daily_hi_lo[["trade_date","pHi","pLo"]], on="trade_date", how="left")

# =========================
# Derived per-bar fields
# =========================
df["hi_op"] = df["high"] - df["open"]
df["op_lo"] = df["open"] - df["low"]
df["hi_pHi"] = df["high"] - df["pHi"]
df["lo_pLo"] = df["low"] - df["pLo"]

# =========================
# Aggregation helpers
# =========================
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean().shift(1)

def agg_intraday(df_scope: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregates across the entire (filtered) day."""
    first_last = df_scope.sort_values("time_et").groupby("trade_date").agg(
        day_open=("open","first"),
        day_close=("close","last")
    )
    hilo = df_scope.groupby("trade_date").agg(
        day_high=("high","max"),
        day_low=("low","min"),
        day_volume=("Volume","sum")
    )
    perbar = df_scope.groupby("trade_date").agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
        bars_in_day=("time","count")
    )
    out = first_last.join(hilo).join(perbar).reset_index()
    out["day_range"] = out["day_high"] - out["day_low"]
    return out.sort_values("trade_date")

def agg_by_session(df_scope: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregate per day for a given boolean session column."""
    sub = df_scope[df_scope[label]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["trade_date",
                                     f"{label}_range", f"{label}_volume",
                                     f"{label}_avg_hi_op", f"{label}_avg_op_lo",
                                     f"{label}_avg_hi_pHi", f"{label}_avg_lo_pLo",
                                     f"{label}_bars"])
    ses = sub.groupby("trade_date").agg(
        s_high=("high","max"),
        s_low=("low","min"),
        s_vol=("Volume","sum"),
        s_avg_hi_op=("hi_op","mean"),
        s_avg_op_lo=("op_lo","mean"),
        s_avg_hi_pHi=("hi_pHi","mean"),
        s_avg_lo_pLo=("lo_pLo","mean"),
        s_bars=("time","count")
    ).reset_index()
    ses.rename(columns={
        "s_high":f"{label}_high",
        "s_low":f"{label}_low",
        "s_vol":f"{label}_volume",
        "s_avg_hi_op":f"{label}_avg_hi_op",
        "s_avg_op_lo":f"{label}_avg_op_lo",
        "s_avg_hi_pHi":f"{label}_avg_hi_pHi",
        "s_avg_lo_pLo":f"{label}_avg_lo_pLo",
        "s_bars":f"{label}_bars"
    }, inplace=True)
    ses[f"{label}_range"] = ses[f"{label}_high"] - ses[f"{label}_low"]
    return ses[["trade_date",
                f"{label}_range", f"{label}_volume",
                f"{label}_avg_hi_op", f"{label}_avg_op_lo",
                f"{label}_avg_hi_pHi", f"{label}_avg_lo_pLo",
                f"{label}_bars"]]

# =========================
# Build aggregates
# =========================
daily_all = agg_intraday(df)
for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily_all.columns:
        daily_all[f"{col}_tw_avg"] = trailing_mean(daily_all[col], trailing_window)
        daily_all[f"{col}_vs_tw"] = daily_all[col] - daily_all[f"{col}_tw_avg"]
        daily_all[f"{col}_pct_vs_tw"] = daily_all[f"{col}_vs_tw"] / daily_all[f"{col}_tw_avg"]

# Session splits
on_agg  = agg_by_session(df, "ON")
ib_agg  = agg_by_session(df, "IB")
rth_agg = agg_by_session(df, "RTH")

daily = daily_all.merge(on_agg, on="trade_date", how="left")\
                 .merge(ib_agg, on="trade_date", how="left")\
                 .merge(rth_agg, on="trade_date", how="left")

# Apply trailing to session metrics too
for ses in ["ON","IB","RTH"]:
    for col in [f"{ses}_range", f"{ses}_volume", f"{ses}_avg_hi_op", f"{ses}_avg_op_lo", f"{ses}_avg_hi_pHi", f"{ses}_avg_lo_pLo"]:
        if col in daily.columns:
            daily[f"{col}_tw_avg"] = trailing_mean(daily[col], trailing_window)
            daily[f"{col}_pct_vs_tw"] = (daily[col] - daily[f"{col}_tw_avg"]) / daily[f"{col}_tw_avg"]

daily = daily.sort_values("trade_date")

# =========================
# KPI snapshot (latest trade day or partial)
# =========================
latest_td = daily["trade_date"].max()
row = daily[daily["trade_date"] == latest_td].iloc[0]

hdr = f"{symbol} — {'As-of ' + asof_time.strftime('%H:%M ET') if asof_time else 'Full day'}"
st.subheader(hdr)

k1,k2,k3,k4 = st.columns(4)
k1.metric(f"Range vs {trailing_window}d", f"{row['day_range']:.2f}",
          None if pd.isna(row.get('day_range_pct_vs_tw')) else f"{row['day_range_pct_vs_tw']*100:+.1f}%")
k2.metric(f"Volume vs {trailing_window}d", f"{row['day_volume']:,.0f}",
          None if pd.isna(row.get('day_volume_pct_vs_tw')) else f"{row['day_volume_pct_vs_tw']*100:+.1f}%")
k3.metric(f"Avg(Hi-Op) vs {trailing_window}d", f"{row['avg_hi_op']:.2f}",
          None if pd.isna(row.get('avg_hi_op_pct_vs_tw')) else f"{row['avg_hi_op_pct_vs_tw']*100:+.1f}%")
k4.metric(f"Avg(Op-Lo) vs {trailing_window}d", f"{row['avg_op_lo']:.2f}",
          None if pd.isna(row.get('avg_op_lo_pct_vs_tw')) else f"{row['avg_op_lo_pct_vs_tw']*100:+.1f}%")

st.caption("Session splits below respect the chosen mode (full day or as-of time).")

# =========================
# Session split table (recent N days)
# =========================
st.markdown("### Session Splits vs Trailing Avg")
show_days = st.slider("Show last N trade days", 10, 120, 30, 5)

cols_order = [
    "trade_date",
    # Whole day
    "day_range","day_range_tw_avg","day_range_pct_vs_tw",
    "day_volume","day_volume_tw_avg","day_volume_pct_vs_tw",
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    "avg_hi_pHi","avg_hi_pHi_tw_avg",
    "avg_lo_pLo","avg_lo_pLo_tw_avg",
    # ON
    "ON_range","ON_range_tw_avg","ON_range_pct_vs_tw",
    "ON_volume","ON_volume_tw_avg","ON_volume_pct_vs_tw",
    "ON_avg_hi_op","ON_avg_hi_op_tw_avg","ON_avg_hi_op_pct_vs_tw",
    "ON_avg_op_lo","ON_avg_op_lo_tw_avg","ON_avg_op_lo_pct_vs_tw",
    # IB
    "IB_range","IB_range_tw_avg","IB_range_pct_vs_tw",
    "IB_volume","IB_volume_tw_avg","IB_volume_pct_vs_tw",
    "IB_avg_hi_op","IB_avg_hi_op_tw_avg","IB_avg_hi_op_pct_vs_tw",
    "IB_avg_op_lo","IB_avg_op_lo_tw_avg","IB_avg_op_lo_pct_vs_tw",
    # RTH
    "RTH_range","RTH_range_tw_avg","RTH_range_pct_vs_tw",
    "RTH_volume","RTH_volume_tw_avg","RTH_volume_pct_vs_tw",
    "RTH_avg_hi_op","RTH_avg_hi_op_tw_avg","RTH_avg_hi_op_pct_vs_tw",
    "RTH_avg_op_lo","RTH_avg_op_lo_tw_avg","RTH_avg_op_lo_pct_vs_tw",
]

existing = [c for c in cols_order if c in daily.columns]
tbl = daily[existing].tail(show_days).copy()

labels = {
    "trade_date":"Trade Date",
    "day_range":"Day Range",
    "day_range_tw_avg":f"Day Range {trailing_window}d",
    "day_range_pct_vs_tw":f"Day Range vs {trailing_window}d",
    "day_volume":"Day Volume",
    "day_volume_tw_avg":f"Day Volume {trailing_window}d",
    "day_volume_pct_vs_tw":f"Day Volume vs {trailing_window}d",
    "avg_hi_op":"Avg(Hi-Op)",
    "avg_hi_op_tw_avg":f"Avg(Hi-Op) {trailing_window}d",
    "avg_hi_op_pct_vs_tw":f"Avg(Hi-Op) vs {trailing_window}d",
    "avg_op_lo":"Avg(Op-Lo)",
    "avg_op_lo_tw_avg":f"Avg(Op-Lo) {trailing_window}d",
    "avg_op_lo_pct_vs_tw":f"Avg(Op-Lo) vs {trailing_window}d",
    "avg_hi_pHi":"Avg(Hi-pHi)",
    "avg_hi_pHi_tw_avg":f"Avg(Hi-pHi) {trailing_window}d",
    "avg_lo_pLo":"Avg(Lo-pLo)",
    "avg_lo_pLo_tw_avg":f"Avg(Lo-pLo) {trailing_window}d",
    # Sessions
    "ON_range":"ON Range", "ON_range_tw_avg":f"ON Range {trailing_window}d", "ON_range_pct_vs_tw":f"ON Range vs {trailing_window}d",
    "ON_volume":"ON Volume", "ON_volume_tw_avg":f"ON Volume {trailing_window}d", "ON_volume_pct_vs_tw":f"ON Volume vs {trailing_window}d",
    "ON_avg_hi_op":"ON Avg(Hi-Op)", "ON_avg_hi_op_tw_avg":f"ON Avg(Hi-Op) {trailing_window}d", "ON_avg_hi_op_pct_vs_tw":f"ON Avg(Hi-Op) vs {trailing_window}d",
    "ON_avg_op_lo":"ON Avg(Op-Lo)", "ON_avg_op_lo_tw_avg":f"ON Avg(Op-Lo) {trailing_window}d", "ON_avg_op_lo_pct_vs_tw":f"ON Avg(Op-Lo) vs {trailing_window}d",

    "IB_range":"IB Range", "IB_range_tw_avg":f"IB Range {trailing_window}d", "IB_range_pct_vs_tw":f"IB Range vs {trailing_window}d",
    "IB_volume":"IB Volume", "IB_volume_tw_avg":f"IB Volume {trailing_window}d", "IB_volume_pct_vs_tw":f"IB Volume vs {trailing_window}d",
    "IB_avg_hi_op":"IB Avg(Hi-Op)", "IB_avg_hi_op_tw_avg":f"IB Avg(Hi-Op) {trailing_window}d", "IB_avg_hi_op_pct_vs_tw":f"IB Avg(Hi-Op) vs {trailing_window}d",
    "IB_avg_op_lo":"IB Avg(Op-Lo)", "IB_avg_op_lo_tw_avg":f"IB Avg(Op-Lo) {trailing_window}d", "IB_avg_op_lo_pct_vs_tw":f"IB Avg(Op-Lo) vs {trailing_window}d",

    "RTH_range":"RTH Range", "RTH_range_tw_avg":f"RTH Range {trailing_window}d", "RTH_range_pct_vs_tw":f"RTH Range vs {trailing_window}d",
    "RTH_volume":"RTH Volume", "RTH_volume_tw_avg":f"RTH Volume {trailing_window}d", "RTH_volume_pct_vs_tw":f"RTH Volume vs {trailing_window}d",
    "RTH_avg_hi_op":"RTH Avg(Hi-Op)", "RTH_avg_hi_op_tw_avg":f"RTH Avg(Hi-Op) {trailing_window}d", "RTH_avg_hi_op_pct_vs_tw":f"RTH Avg(Hi-Op) vs {trailing_window}d",
    "RTH_avg_op_lo":"RTH Avg(Op-Lo)", "RTH_avg_op_lo_tw_avg":f"RTH Avg(Op-Lo) {trailing_window}d", "RTH_avg_op_lo_pct_vs_tw":f"RTH Avg(Op-Lo) vs {trailing_window}d",
}
tbl = tbl.rename(columns=labels)

# Formatting helpers
def color_pos_neg(val):
    if pd.isna(val): return ""
    try: v = float(val)
    except: return ""
    return f"color: {'#16a34a' if v>0 else ('#dc2626' if v<0 else '#111827')};"

fmt = {c:"{:,.2f}" for c in tbl.columns if c not in ["Trade Date"]}
for c in tbl.columns:
    if "Volume" in c: fmt[c] = "{:,.0f}"
    if c.endswith("Bars"): fmt[c] = "{:,.0f}"

styled = tbl.style.format(fmt).applymap(color_pos_neg, subset=[c for c in tbl.columns if "vs " in c])
st.dataframe(styled, use_container_width=True)

# =========================
# Price vs MAs (recent days) — chart convenience
# =========================
st.markdown("### Recent Price vs MAs")
recent_days = int(max(10, min(20, trailing_window * 2)))
cutoff = df_for_chart["time_et"].max() - pd.Timedelta(days=recent_days)
mini = df_for_chart[df_for_chart["time_et"] >= cutoff][["time_et","close"] + MA_COLS].copy().set_index("time_et")
st.line_chart(mini, use_container_width=True)

st.caption("""
**Notes**
- Trade day rolls at **18:00 ET**; “As-of 03:00” includes last night’s Globex up to 03:00.
- Session splits respect the chosen mode (Full day vs As-of).
- Trailing averages exclude the current day to avoid look-ahead bias.
""")
