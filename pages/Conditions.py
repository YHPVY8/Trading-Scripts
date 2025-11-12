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
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.3])
with c1:
    symbol = st.text_input("Symbol (display-only)", value="ES")
with c2:
    lookback_days = st.number_input("Load last N calendar days", 30, 3650, 240, 30)
with c3:
    trailing_window = st.number_input("Trailing window (days)", 5, 60, 10, 1)
with c4:
    mode = st.selectbox("View mode", ["Full day", "As-of time snapshot"])
with c5:
    trade_days_to_keep = st.number_input("Trade days to keep (most recent)", 10, 3650, 180, 10)

asof_time = None
if mode == "As-of time snapshot":
    asof_time = st.time_input("As-of time (US/Eastern)", value=dt.time(3, 0))

# =========================
# Fetch (newest-first) + choose last N trade days like main App
# =========================
TABLE = "es_30m"  # adjust if needed

# Load enough rows to comfortably include the last N trade days
# 30m bars ~= 48 per 24h. Use a buffer (x1.5) to be safe.
rows_per_day_guess = 48
rows_to_load = int(max(2000, trade_days_to_keep * rows_per_day_guess * 1.5))

response = (
    sb.table(TABLE)
      .select("*")
      .order("time", desc=True)     # NEW: pull newest first (matches your App)
      .limit(rows_to_load)
      .execute()
)
df = pd.DataFrame(response.data)
if df.empty:
    st.error("No data returned from es_30m.")
    st.stop()

# --- Parse time + compute tz-aware trade_day (18:00 ET roll) ---
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")

# Keep these BEFORE any filtering so they reflect the raw latest
latest_bar_time = df["time_et"].max()
latest_trade_day_expected = (
    latest_bar_time.floor("D") + pd.to_timedelta(int(latest_bar_time.hour >= 18), unit="D")
)

et = df["time_et"]
midnight = et.dt.floor("D")
roll = (et.dt.hour >= 18).astype("int64")
df["trade_day"] = midnight + pd.to_timedelta(roll, unit="D")   # tz-aware midnight ET of trade-date
df["trade_date"] = df["trade_day"].dt.date                     # display helper

# --- Create session flags BEFORE any functions use them ---
t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30)))
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30)))
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0)))

# --- Pick the most recent N trade days by their latest bar timestamp (DESC), then sort ASC for display ---
per_day_last_bar = (
    df.groupby("trade_day")["time_et"]
      .max()
      .sort_values(ascending=False)
)
recent_trade_days_desc = per_day_last_bar.index[:int(trade_days_to_keep)]
recent_mask = df["trade_day"].isin(set(recent_trade_days_desc))
df = df[recent_mask].copy()

# Sort oldest->newest for downstream grouping / tables (like your App)
df = df.sort_values(["trade_day", "time_et"]).reset_index(drop=True)

# =========================
# Optional: As-of cutoff filtering (trade_day aware)
# “As-of HH:MM” keeps bars where:
#   prev_session_start = trade_day - 1 day + 18:00
#   cutoff_ts          = trade_day + HH:MM
#   prev_session_start <= time_et <= cutoff_ts
# =========================
if mode == "As-of time snapshot":
    cut_seconds = asof_time.hour*3600 + asof_time.minute*60 + asof_time.second
    cutoff_ts = df["trade_day"] + pd.to_timedelta(cut_seconds, unit="s")
    prev_start = (df["trade_day"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)
    mask_asof = (df["time_et"] >= prev_start) & (df["time_et"] <= cutoff_ts)
    df = df[mask_asof].copy()

# --- Required columns guard (kept after trimming for clarity) ---
for col in ["open", "high", "low", "close", "Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# --- Data health (sanity) ---
with st.expander("Data health (debug)"):
    st.write({
        "rows_loaded_desc": rows_to_load,
        "rows_after_trim": len(df),
        "min_time_et": str(df["time_et"].min()),
        "max_time_et": str(df["time_et"].max()),
        "first_trade_day": str(df["trade_day"].min()),
        "last_trade_day": str(df["trade_day"].max()),
        "unique_trade_days_kept": int(df["trade_day"].nunique()),
        "kept_trade_days_desc": [str(d) for d in recent_trade_days_desc.tolist()],
        "expected_latest_trade_day_from_raw": str(latest_trade_day_expected),
    })

# =========================
# Prior-day extrema by trade_day (for pHi/pLo)
# =========================
daily_hi_lo = (
    df.groupby("trade_day")
      .agg(day_high=("high","max"), day_low=("low","min"))
      .sort_index()
      .reset_index()
)
daily_hi_lo["pHi"] = daily_hi_lo["day_high"].shift(1)
daily_hi_lo["pLo"] = daily_hi_lo["day_low"].shift(1)
df = df.merge(daily_hi_lo[["trade_day","pHi","pLo"]], on="trade_day", how="left")

# Bar-level deltas
df["hi_op"] = df["high"] - df["open"]
df["op_lo"] = df["open"] - df["low"]
df["hi_pHi"] = df["high"] - df["pHi"]
df["lo_pLo"] = df["low"] - df["pLo"]

# =========================
# Intra-day bar index + cumulative volume (since 18:00 roll)
# =========================
df["bar_n"] = df.groupby("trade_day").cumcount()
df["cum_vol"] = df.groupby("trade_day")["Volume"].cumsum()

# Helper
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean().shift(1)

# =========================
# Aggregate per trade_day (respecting current filter)
# =========================
def agg_daily(scope: pd.DataFrame) -> pd.DataFrame:
    first_last = scope.groupby("trade_day").agg(
        day_open=("open","first"),
        day_close=("close","last"),
    )
    hilo = scope.groupby("trade_day").agg(
        day_high=("high","max"),
        day_low=("low","min"),
        day_volume=("Volume","sum"),
        bars_in_day=("time","count")
    )
    perbar = scope.groupby("trade_day").agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
    )
    out = first_last.join(hilo).join(perbar).reset_index().sort_values("trade_day")
    out["day_range"] = out["day_high"] - out["day_low"]
    out["trade_date"] = out["trade_day"].dt.date
    return out

daily_all = agg_daily(df)

# Cum volume at "same cutoff" for each day
cutoff_n = df.groupby("trade_day")["bar_n"].max().rename("cutoff_n")
last_rows = (
    df.merge(cutoff_n, on="trade_day")
      .query("bar_n == cutoff_n")[["trade_day","cum_vol"]]
      .sort_values("trade_day")
      .rename(columns={"cum_vol":"cum_vol_at_cutoff"})
)
last_rows["cum_vol_tw_avg"] = trailing_mean(last_rows["cum_vol_at_cutoff"], int(trailing_window))
last_rows["cum_vol_vs_tw"] = last_rows["cum_vol_at_cutoff"] - last_rows["cum_vol_tw_avg"]
last_rows["cum_vol_pct_vs_tw"] = last_rows["cum_vol_vs_tw"] / last_rows["cum_vol_tw_avg"]

# % pMid hit by session (within-session prev-bar midpoint)
def session_pmid_percent(scope: pd.DataFrame, label: str) -> pd.DataFrame:
    # robust guard: if label column missing, return empty frame
    if label not in scope.columns:
        return pd.DataFrame(columns=["trade_day", f"{label}_pMid_hit_pct"])
    sub = scope[scope[label]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["trade_day", f"{label}_pMid_hit_pct"])
    sub = sub.sort_values(["trade_day","time_et"]).copy()
    sub["prev_high"] = sub.groupby("trade_day")["high"].shift(1)
    sub["prev_low"]  = sub.groupby("trade_day")["low"].shift(1)
    sub["prev_mid"]  = (sub["prev_high"] + sub["prev_low"]) / 2.0
    sub["eligible"]  = ~sub["prev_mid"].isna()
    sub["hit"]       = sub["eligible"] & (sub["low"] <= sub["prev_mid"]) & (sub["high"] >= sub["prev_mid"])
    agg = sub.groupby("trade_day").agg(hits=("hit","sum"), elig=("eligible","sum")).reset_index()
    agg[f"{label}_pMid_hit_pct"] = np.where(agg["elig"]>0, agg["hits"]/agg["elig"], np.nan)
    return agg[["trade_day", f"{label}_pMid_hit_pct"]]

on_hit  = session_pmid_percent(df, "ON")
ib_hit  = session_pmid_percent(df, "IB")
rth_hit = session_pmid_percent(df, "RTH")

# Build final daily table
daily = (
    daily_all.merge(last_rows, on="trade_day", how="left")
             .merge(on_hit,  on="trade_day", how="left")
             .merge(ib_hit,  on="trade_day", how="left")
             .merge(rth_hit, on="trade_day", how="left")
             .sort_values("trade_day")
)

# Classic trailing metrics (full-day style; still respects as-of filter because daily_all did)
for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], int(trailing_window))
        daily[f"{col}_pct_vs_tw"] = (daily[col] - daily[f"{col}_tw_avg"]) / daily[f"{col}_tw_avg"]

# Trailing for pMid% (aligned by current filter)
for ses in ["ON","IB","RTH"]:
    col = f"{ses}_pMid_hit_pct"
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], int(trailing_window))
        daily[f"{col}_pct_vs_tw"] = daily[col] - daily[f"{col}_tw_avg"]

# =========================
# Trim to most recent N trade days (based on trade_day, not raw time)
# =========================
daily = daily.sort_values("trade_day")
unique_days = pd.Series(daily["trade_day"].unique())
if len(unique_days) > int(trade_days_to_keep):
    keep = set(unique_days.iloc[-int(trade_days_to_keep):])
    daily = daily[daily["trade_day"].isin(keep)].copy().sort_values("trade_day")

# =========================
# Assert latest trade_day presence
# =========================
if latest_trade_day_expected not in daily["trade_day"].values:
    st.warning(
        "Latest trade day derived from the raw table is missing after filters.\n"
        f"Latest bar ET: {latest_bar_time}\n"
        f"Expected trade_day (ET midnight): {latest_trade_day_expected}\n"
        "Tip: widen 'Load last N calendar days', increase 'Trade days to keep', or adjust As-of cutoff."
    )

if daily.empty:
    st.warning("No trade days after trimming—expand your lookback or lower 'Trade days to keep'.")
    st.stop()

# =========================
# KPI (latest day)
# =========================
latest_td = daily["trade_day"].max()
row = daily.loc[daily["trade_day"] == latest_td].iloc[0]

hdr = f"{symbol} — {'As-of ' + asof_time.strftime('%H:%M ET') if asof_time else 'Full day'}"
st.subheader(hdr)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Range vs {trailing_window}d",
          f"{row.get('day_range', np.nan):.2f}" if pd.notna(row.get('day_range')) else "—",
          None if pd.isna(row.get('day_range_pct_vs_tw')) else f"{row['day_range_pct_vs_tw']*100:+.1f}%")
k2.metric(f"Cum Vol vs {trailing_window}d",
          f"{row.get('cum_vol_at_cutoff', np.nan):,.0f}" if pd.notna(row.get('cum_vol_at_cutoff')) else "—",
          None if pd.isna(row.get('cum_vol_pct_vs_tw')) else f"{row['cum_vol_pct_vs_tw']*100:+.1f}%")
k3.metric(f"Avg(Hi-Op) vs {trailing_window}d",
          f"{row.get('avg_hi_op', np.nan):.2f}" if pd.notna(row.get('avg_hi_op')) else "—",
          None if pd.isna(row.get('avg_hi_op_pct_vs_tw')) else f"{row['avg_hi_op_pct_vs_tw']*100:+.1f}%")
k4.metric(f"Avg(Op-Lo) vs {trailing_window}d",
          f"{row.get('avg_op_lo', np.nan):.2f}" if pd.notna(row.get('avg_op_lo')) else "—",
          None if pd.isna(row.get('avg_op_lo_pct_vs_tw')) else f"{row['avg_op_lo_pct_vs_tw']*100:+.1f}%")

st.caption("Cum Vol uses cumulative volume up to the current cutoff (Full day = last bar). Trailing averages exclude the current day.")

# =========================
# Table (last N trade days)
# =========================
st.markdown("### Conditions vs Trailing (last N trade days)")
max_slider = int(min(120, trade_days_to_keep, len(daily)))
show_days = st.slider("Show last N trade days", 10, max_slider, min(30, max_slider), 5)

cols = [
    "trade_date",
    "day_open","day_high","day_low","day_close",
    "day_range","day_range_tw_avg","day_range_pct_vs_tw",
    "cum_vol_at_cutoff","cum_vol_tw_avg","cum_vol_pct_vs_tw",
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    "avg_hi_pHi","avg_hi_pHi_tw_avg",
    "avg_lo_pLo","avg_lo_pLo_tw_avg",
    "ON_pMid_hit_pct","ON_pMid_hit_pct_tw_avg","ON_pMid_hit_pct_pct_vs_tw",  # placeholder
    "IB_pMid_hit_pct","IB_pMid_hit_pct_tw_avg","IB_pMid_hit_pct_pct_vs_tw",  # placeholder
    "RTH_pMid_hit_pct","RTH_pMid_hit_pct_tw_avg","RTH_pMid_hit_pct_pct_vs_tw",# placeholder
    "bars_in_day",
]
# correct placeholder column names
for ses in ["ON","IB","RTH"]:
    real = f"{ses}_pMid_hit_pct_vs_tw"
    ph = f"{ses}_pMid_hit_pct_pct_vs_tw"
    if real in daily.columns and ph in cols:
        cols[cols.index(ph)] = real

existing = [c for c in cols if c in daily.columns]
tbl = daily[existing].tail(int(show_days)).copy()

labels = {
    "trade_date":"Trade Date",
    "day_open":"Open","day_high":"High","day_low":"Low","day_close":"Close",
    "day_range":"Range","day_range_tw_avg":f"Range {trailing_window}d","day_range_pct_vs_tw":f"Range vs {trailing_window}d",
    "cum_vol_at_cutoff":"Cum Vol (cutoff)","cum_vol_tw_avg":f"Cum Vol {trailing_window}d (cutoff)",
    "cum_vol_pct_vs_tw":f"Cum Vol vs {trailing_window}d",
    "avg_hi_op":"Avg(Hi-Op)","avg_hi_op_tw_avg":f"Avg(Hi-Op) {trailing_window}d","avg_hi_op_pct_vs_tw":f"Avg(Hi-Op) vs {trailing_window}d",
    "avg_op_lo":"Avg(Op-Lo)","avg_op_lo_tw_avg":f"Avg(Op-Lo) {trailing_window}d","avg_op_lo_pct_vs_tw":f"Avg(Op-Lo) vs {trailing_window}d",
    "avg_hi_pHi":"Avg(Hi-pHi)","avg_hi_pHi_tw_avg":f"Avg(Hi-pHi) {trailing_window}d",
    "avg_lo_pLo":"Avg(Lo-pLo)","avg_lo_pLo_tw_avg":f"Avg(Lo-pLo) {trailing_window}d",
    "ON_pMid_hit_pct":"ON % pMid Hit","ON_pMid_hit_pct_tw_avg":f"ON % pMid Hit {trailing_window}d","ON_pMid_hit_pct_vs_tw":"ON % pMid Hit vs tw",
    "IB_pMid_hit_pct":"IB % pMid Hit","IB_pMid_hit_pct_tw_avg":f"IB % pMid Hit {trailing_window}d","IB_pMid_hit_pct_vs_tw":"IB % pMid Hit vs tw",
    "RTH_pMid_hit_pct":"RTH % pMid Hit","RTH_pMid_hit_pct_tw_avg":f"RTH % pMid Hit {trailing_window}d","RTH_pMid_hit_pct_vs_tw":"RTH % pMid Hit vs tw",
    "bars_in_day":"Bars"
}
tbl = tbl.rename(columns=labels)

# Formatting
fmt = {}
for name in tbl.columns:
    if name == "Trade Date":
        continue
    if "Vol" in name:
        fmt[name] = "{:,.0f}"
    elif "% pMid Hit" in name:
        fmt[name] = "{:.1%}"
    elif "vs" in name and "% pMid Hit" not in name and "Vol" not in name:
        fmt[name] = "{:,.2f}"
    elif name == "Bars":
        fmt[name] = "{:,.0f}"
    else:
        fmt[name] = "{:,.2f}"

def color_pos_neg(val):
    if pd.isna(val): return ""
    try: v = float(val)
    except: return ""
    return f"color: {'#16a34a' if v>0 else ('#dc2626' if v<0 else '#111827')};"

vs_cols = [c for c in tbl.columns if "vs" in c and "% pMid Hit" not in c]
styled = tbl.style.format(fmt).applymap(color_pos_neg, subset=vs_cols).set_properties(
    subset=["Trade Date"], **{"font-weight":"600"}
)
st.dataframe(styled, use_container_width=True)

st.caption(f"""
**Alignment guards in place**
- Latest bar ET: **{latest_bar_time}**
- Expected latest trade_day (ET midnight, 18:00 roll): **{latest_trade_day_expected}**
- Page groups by **tz-aware `trade_day`**, selects the most recent trade days from a **DESC query**, then re-sorts ASC for display.
""")
