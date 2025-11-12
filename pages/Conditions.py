#!/usr/bin/env python3
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Conditions", layout="wide")
st.title("Market Conditions (30m)")

# -----------------------
# Supabase
# -----------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Controls
# -----------------------
c1, c2, c3, c4 = st.columns([1,1,1,1.4])
with c1:
    symbol = st.text_input("Symbol (display-only)", value="ES")
with c2:
    lookback_days = st.number_input("Load last N calendar days", 30, 3650, 180, 30)
with c3:
    trailing_window = st.number_input("Trailing window (days)", 5, 60, 10, 1)
with c4:
    mode = st.selectbox("View mode", ["Full day", "As-of time snapshot"])

asof_time = None
if mode == "As-of time snapshot":
    asof_time = st.time_input("As-of time (US/Eastern)", value=dt.time(3, 0))

TABLE = "es_30m"  # adjust if needed

# -----------------------
# Fetch + parse
# -----------------------
since = (dt.date.today() - dt.timedelta(days=int(lookback_days))).isoformat()
q = sb.table(TABLE).select("*").gte("time", f"{since}T00:00:00Z").order("time", desc=False)
data = q.execute().data
if not data:
    st.warning("No rows returned. Check table name/filters.")
    st.stop()

df = pd.DataFrame(data)

# Required columns
for col in ["time", "open", "high", "low", "close", "Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# Time handling
df["time"] = pd.to_datetime(df["time"], utc=True)
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")
df["date_et"] = df["time_et"].dt.date
df["hm"] = df["time_et"].dt.strftime("%H:%M")

# -----------------------
# Trade date alignment: bars >= 18:00 ET belong to next trade_date
# -----------------------
def trade_date_et(ts: pd.Timestamp) -> dt.date:
    if ts.time() >= dt.time(18, 0):
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()

df["trade_date"] = df["time_et"].apply(trade_date_et)

# -----------------------
# Session labels
#   ON : 18:00–09:30
#   IB : 09:30–10:30
#   RTH: 09:30–16:00
# -----------------------
tm = df["time_et"].dt.time
df["ON"]  = ((tm >= dt.time(18,0)) | (tm < dt.time(9,30)))
df["IB"]  = ((tm >= dt.time(9,30)) & (tm < dt.time(10,30)))
df["RTH"] = ((tm >= dt.time(9,30)) & (tm <= dt.time(16,0)))

# -----------------------
# As-of cutoff filtering
# “As-of 03:00” includes bars from prev calendar day >= 18:00 plus today <= 03:00
# -----------------------
def mask_asof(dfx: pd.DataFrame, cutoff: dt.time) -> pd.Series:
    td = dfx["trade_date"]
    d  = dfx["time_et"].dt.date
    t  = dfx["time_et"].dt.time
    prev_day = td - pd.to_timedelta(1, unit="D")
    return ((d == td) & (t <= cutoff)) | ((d == prev_day) & (t >= dt.time(18,0)))

if mode == "As-of time snapshot":
    df = df[mask_asof(df, asof_time)].copy()

# -----------------------
# Prior-day extrema per trade_date (for pHi/pLo)
# -----------------------
daily_hi_lo = (
    df.groupby("trade_date")
      .agg(day_high=("high","max"), day_low=("low","min"))
      .sort_index()
      .reset_index()
)
daily_hi_lo["pHi"] = daily_hi_lo["day_high"].shift(1)
daily_hi_lo["pLo"] = daily_hi_lo["day_low"].shift(1)
df = df.merge(daily_hi_lo[["trade_date","pHi","pLo"]], on="trade_date", how="left")

# Bar-level deltas
df["hi_op"] = df["high"] - df["open"]
df["op_lo"] = df["open"] - df["low"]
df["hi_pHi"] = df["high"] - df["pHi"]
df["lo_pLo"] = df["low"] - df["pLo"]

# -----------------------
# Compute bar index and cumulative volume since 18:00 ET
#   bar_n: 0,1,2,... within each trade_date ordered by time_et
#   cum_vol: rolling sum of Volume up to that bar
# -----------------------
df = df.sort_values(["trade_date", "time_et"]).copy()
df["bar_n"] = df.groupby("trade_date").cumcount()
df["cum_vol"] = df.groupby("trade_date")["Volume"].cumsum()

# Helper: trailing mean shifted to avoid look-ahead
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean().shift(1)

# -----------------------
# Daily aggregates (on filtered df respecting mode)
# -----------------------
def agg_daily(scope: pd.DataFrame) -> pd.DataFrame:
    first_last = scope.groupby("trade_date").agg(
        day_open=("open","first"),
        day_close=("close","last"),
    )
    hilo = scope.groupby("trade_date").agg(
        day_high=("high","max"),
        day_low=("low","min"),
        day_volume=("Volume","sum"),
        bars_in_day=("time","count")
    )
    perbar = scope.groupby("trade_date").agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
    )
    out = first_last.join(hilo).join(perbar).reset_index().sort_values("trade_date")
    out["day_range"] = out["day_high"] - out["day_low"]
    return out

daily_all = agg_daily(df)

# -----------------------
# Volume vs 10d TRAILING at the SAME CUTOFF
# Logic:
#  • For each trade_date, find last bar_n present (cutoff_n)
#  • Take that day's cum_vol at cutoff_n
#  • Build a series of cum_vol_at_cutoff per day, then 10d trailing avg (shifted)
# -----------------------
# Cum vol at cutoff for each day
cutoff_n = df.groupby("trade_date")["bar_n"].max().rename("cutoff_n")
last_rows = df.merge(cutoff_n, on="trade_date")
last_rows = last_rows[last_rows["bar_n"] == last_rows["cutoff_n"]][["trade_date","cum_vol"]]
last_rows = last_rows.sort_values("trade_date").rename(columns={"cum_vol":"cum_vol_at_cutoff"})

# Trailing average of cum volume at the *same cutoff index*
# We don't need to align by exact bar_n because each day's cutoff_n is "whatever exists today up to the as-of filter".
# To compare fairly, we compute for each day the cum_vol_at_cutoff and then rolling mean across prior N days.
last_rows["cum_vol_tw_avg"] = trailing_mean(last_rows["cum_vol_at_cutoff"], int(trailing_window))
last_rows["cum_vol_vs_tw"] = last_rows["cum_vol_at_cutoff"] - last_rows["cum_vol_tw_avg"]
last_rows["cum_vol_pct_vs_tw"] = last_rows["cum_vol_vs_tw"] / last_rows["cum_vol_tw_avg"]

# Attach to daily
daily = daily_all.merge(last_rows, on="trade_date", how="left")

# Still keep whole-day/traditional trailing metrics if you want them:
for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], int(trailing_window))
        daily[f"{col}_pct_vs_tw"] = (daily[col] - daily[f"{col}_tw_avg"]) / daily[f"{col}_tw_avg"]

# -----------------------
# % pMid hit by session (within-session prev-bar midpoint)
# For each session S in {ON, IB, RTH}:
#   prev_mid = (prev_bar_high + prev_bar_low)/2 within same S and day
#   hit = (low <= prev_mid <= high) on the CURRENT bar
#   pMid_hit_pct_S = hits / eligible_bars  (eligible_bars = count where prev exists)
# -----------------------
def session_pmid_percent(scope: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = scope[scope[label]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["trade_date", f"{label}_pMid_hit_pct"])
    sub = sub.sort_values(["trade_date","time_et"]).copy()
    # prior bar within the session/day
    sub["prev_high"] = sub.groupby("trade_date")["high"].shift(1)
    sub["prev_low"]  = sub.groupby("trade_date")["low"].shift(1)
    sub["prev_mid"]  = (sub["prev_high"] + sub["prev_low"]) / 2.0
    sub["eligible"]  = ~sub["prev_mid"].isna()
    sub["hit"]       = sub["eligible"] & (sub["low"] <= sub["prev_mid"]) & (sub["high"] >= sub["prev_mid"])
    agg = sub.groupby("trade_date").agg(
        hits=( "hit", "sum"),
        elig=("eligible","sum")
    ).reset_index()
    agg[f"{label}_pMid_hit_pct"] = np.where(agg["elig"]>0, agg["hits"]/agg["elig"], np.nan)
    return agg[["trade_date", f"{label}_pMid_hit_pct"]]

on_hit  = session_pmid_percent(df, "ON")
ib_hit  = session_pmid_percent(df, "IB")
rth_hit = session_pmid_percent(df, "RTH")

daily = (daily.merge(on_hit, on="trade_date", how="left")
              .merge(ib_hit, on="trade_date", how="left")
              .merge(rth_hit, on="trade_date", how="left"))

# Trailing averages for pMid% at same cutoff (because df already filtered by mode/as-of)
for ses in ["ON","IB","RTH"]:
    col = f"{ses}_pMid_hit_pct"
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], int(trailing_window))
        daily[f"{col}_pct_vs_tw"] = (daily[col] - daily[f"{col}_tw_avg"])

# -----------------------
# KPI (latest day in filtered set)
# -----------------------
latest_td = daily["trade_date"].max()
row = daily.loc[daily["trade_date"] == latest_td].iloc[0]

hdr = f"{symbol} — {'As-of ' + asof_time.strftime('%H:%M ET') if asof_time else 'Full day'}"
st.subheader(hdr)

k1, k2, k3, k4 = st.columns(4)
# Range vs trailing (classic)
k1.metric(f"Range vs {trailing_window}d",
          f"{row['day_range']:.2f}",
          None if pd.isna(row.get('day_range_pct_vs_tw')) else f"{row['day_range_pct_vs_tw']*100:+.1f}%")
# Cum Volume vs trailing at same cutoff  <-- NEW definition
k2.metric(f"Cum Vol vs {trailing_window}d",
          f"{row['cum_vol_at_cutoff']:,.0f}",
          None if pd.isna(row.get('cum_vol_pct_vs_tw')) else f"{row['cum_vol_pct_vs_tw']*100:+.1f}%")
# Avg(Hi-Op) vs trailing
k3.metric(f"Avg(Hi-Op) vs {trailing_window}d",
          f"{row['avg_hi_op']:.2f}",
          None if pd.isna(row.get('avg_hi_op_pct_vs_tw')) else f"{row['avg_hi_op_pct_vs_tw']*100:+.1f}%")
# Avg(Op-Lo) vs trailing
k4.metric(f"Avg(Op-Lo) vs {trailing_window}d",
          f"{row['avg_op_lo']:.2f}",
          None if pd.isna(row.get('avg_op_lo_pct_vs_tw')) else f"{row['avg_op_lo_pct_vs_tw']*100:+.1f}%")

st.caption("Cum Vol uses the cumulative volume up to the current cutoff (Full day = last bar). Trailing averages exclude the current day.")

# -----------------------
# Table (last N trade days)
# -----------------------
st.markdown("### Conditions vs Trailing (last N trade days)")
show_days = st.slider("Show last N trade days", 10, 120, 30, 5)

cols = [
    "trade_date",
    # whole-day / core
    "day_open","day_high","day_low","day_close","day_range","day_range_tw_avg","day_range_pct_vs_tw",
    # cumulative volume at cutoff
    "cum_vol_at_cutoff","cum_vol_tw_avg","cum_vol_pct_vs_tw",
    # per-bar averages (whole filtered day)
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    "avg_hi_pHi","avg_hi_pHi_tw_avg",
    "avg_lo_pLo","avg_lo_pLo_tw_avg",
    # pMid hit % by session
    "ON_pMid_hit_pct","ON_pMid_hit_pct_tw_avg","ON_pMid_hit_pct_vs_tw",
    "IB_pMid_hit_pct","IB_pMid_hit_pct_tw_avg","IB_pMid_hit_pct_vs_tw",
    "RTH_pMid_hit_pct","RTH_pMid_hit_pct_tw_avg","RTH_pMid_hit_pct_vs_tw",
    "bars_in_day",
]
existing = [c for c in cols if c in daily.columns]
tbl = daily[existing].tail(show_days).copy()

labels = {
    "trade_date":"Trade Date",
    "day_open":"Open","day_high":"High","day_low":"Low","day_close":"Close",
    "day_range":"Range","day_range_tw_avg":f"Range {trailing_window}d","day_range_pct_vs_tw":f"Range vs {trailing_window}d",
    "cum_vol_at_cutoff":"Cum Vol (cutoff)",
    "cum_vol_tw_avg":f"Cum Vol {trailing_window}d (cutoff)",
    "cum_vol_pct_vs_tw":f"Cum Vol vs {trailing_window}d",
    "avg_hi_op":"Avg(Hi-Op)","avg_hi_op_tw_avg":f"Avg(Hi-Op) {trailing_window}d","avg_hi_op_pct_vs_tw":f"Avg(Hi-Op) vs {trailing_window}d",
    "avg_op_lo":"Avg(Op-Lo)","avg_op_lo_tw_avg":f"Avg(Op-Lo) {trailing_window}d","avg_op_lo_pct_vs_tw":f"Avg(Op-Lo) vs {trailing_window}d",
    "avg_hi_pHi":"Avg(Hi-pHi)","avg_hi_pHi_tw_avg":f"Avg(Hi-pHi) {trailing_window}d",
    "avg_lo_pLo":"Avg(Lo-pLo)","avg_lo_pLo_tw_avg":f"Avg(Lo-pLo) {trailing_window}d",
    "ON_pMid_hit_pct":"ON % pMid Hit","ON_pMid_hit_pct_tw_avg":f"ON % pMid Hit {trailing_window}d",
    "ON_pMid_hit_pct_vs_tw":"ON % pMid Hit vs tw",
    "IB_pMid_hit_pct":"IB % pMid Hit","IB_pMid_hit_pct_tw_avg":f"IB % pMid Hit {trailing_window}d",
    "IB_pMid_hit_pct_vs_tw":"IB % pMid Hit vs tw",
    "RTH_pMid_hit_pct":"RTH % pMid Hit","RTH_pMid_hit_pct_tw_avg":f"RTH % pMid Hit {trailing_window}d",
    "RTH_pMid_hit_pct_vs_tw":"RTH % pMid Hit vs tw",
    "bars_in_day":"Bars"
}
tbl = tbl.rename(columns=labels)

# Formatting
fmt = {}
for name in tbl.columns:
    if name in ["Trade Date"]:
        continue
    if "Vol" in name:
        fmt[name] = "{:,.0f}"
    elif "pMid Hit" in name:
        fmt[name] = "{:.1%}"
    elif name == "Bars":
        fmt[name] = "{:,.0f}"
    else:
        fmt[name] = "{:,.2f}"

def color_pos_neg(val):
    if pd.isna(val): return ""
    try: v = float(val)
    except: return ""
    return f"color: {'#16a34a' if v>0 else ('#dc2626' if v<0 else '#111827')};"

# Apply color to “vs trailing” columns
vs_cols = [c for c in tbl.columns if "vs" in c]
styled = tbl.style.format(fmt).applymap(color_pos_neg, subset=vs_cols).set_properties(
    subset=["Trade Date"], **{"font-weight":"600"}
)
st.dataframe(styled, use_container_width=True)

st.caption("""
**Definitions**
- **Cum Vol vs 10d**: cumulative volume since **18:00 ET** up to the current cutoff (full day = last bar), compared to the **trailing {tw} days** at that same cutoff.  
- **% pMid hit**: within each session (ON / IB / RTH), share of bars whose range **touches the previous bar’s midpoint** in that *same session/day*. Trailing averages are computed on the same cutoff to avoid look-ahead.
""".format(tw=trailing_window))
