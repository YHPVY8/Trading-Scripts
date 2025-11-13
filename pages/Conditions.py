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
TABLE = "es_30m"  # base price table

rows_per_day_guess = 48  # 30m bars
rows_to_load = int(max(2000, trade_days_to_keep * rows_per_day_guess * 1.5))

response = (
    sb.table(TABLE)
      .select("*")
      .order("time", desc=True)   # newest first (matches your main App)
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
df["trade_day"] = midnight + pd.to_timedelta(roll, unit="D")  # tz-aware midnight ET of trade-date
df["trade_date"] = df["trade_day"].dt.date

# --- Create session flags (needed later for pMid stats) ---
t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30)))
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30)))
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0)))

# --- Pick the most recent N trade days by latest bar timestamp (DESC), then sort ASC for display ---
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
# Backfill: ensure the latest trade_day has ALL bars from 18:00 ET
# (prevents wrong lows/highs/cum-vol when DESC limit clipped early ON bars)
# =========================
latest_td = df["trade_day"].max()
latest_day_df = df[df["trade_day"] == latest_td]
if not latest_day_df.empty:
    earliest_bar_et = latest_day_df["time_et"].min()

    # Start of session = previous calendar day 18:00 ET = trade_day - 6h
    latest_start_et = latest_td - pd.Timedelta(hours=6)     # 18:00 ET of prior calendar day
    latest_end_et   = latest_td + pd.Timedelta(hours=24)    # generous cap

    # If first bar we have begins after 18:00 ET, fetch the missing early bars
    if earliest_bar_et > latest_start_et:
        start_iso = latest_start_et.tz_convert("UTC").isoformat()
        end_iso   = latest_end_et.tz_convert("UTC").isoformat()
        backfill_resp = (
            sb.table(TABLE)
              .select("*")
              .gte("time", start_iso)
              .lt("time", end_iso)
              .order("time", desc=False)
              .execute()
        )
        missing_df = pd.DataFrame(backfill_resp.data)
        if not missing_df.empty:
            missing_df["time"] = pd.to_datetime(missing_df["time"], utc=True, errors="coerce")
            missing_df["time_et"] = missing_df["time"].dt.tz_convert("US/Eastern")

            et2 = missing_df["time_et"]
            midnight2 = et2.dt.floor("D")
            roll2 = (et2.dt.hour >= 18).astype("int64")
            missing_df["trade_day"] = midnight2 + pd.to_timedelta(roll2, unit="D")
            missing_df["trade_date"] = missing_df["trade_day"].dt.date

            # add the flags we use later
            tt = missing_df["time_et"].dt.time
            missing_df["ON"]  = ((tt >= dt.time(18,0)) | (tt < dt.time(9,30)))
            missing_df["IB"]  = ((tt >= dt.time(9,30)) & (tt < dt.time(10,30)))
            missing_df["RTH"] = ((tt >= dt.time(9,30)) & (tt <= dt.time(16,0)))

            # keep only rows that aren't already present
            missing_df = missing_df[~missing_df["time"].isin(df["time"])]
            if not missing_df.empty:
                df = pd.concat([df, missing_df], ignore_index=True)
                df = df.sort_values(["trade_day","time_et"]).reset_index(drop=True)

# =========================
# Optional: As-of cutoff filtering (trade_day aware)
# Keeps bars: (trade_day-1 @ 18:00) → (trade_day @ HH:MM)
# =========================
if mode == "As-of time snapshot":
    cut_seconds = asof_time.hour*3600 + asof_time.minute*60 + asof_time.second
    cutoff_ts = df["trade_day"] + pd.to_timedelta(cut_seconds, unit="s")
    prev_start = (df["trade_day"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)
    mask_asof = (df["time_et"] >= prev_start) & (df["time_et"] <= cutoff_ts)
    df = df[mask_asof].copy()

# --- Required columns guard ---
for col in ["open", "high", "low", "close", "Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# =========================
# Join enriched 30m data for session-level features
# =========================
enriched_cols = [
    "time",
    "cum_vol",
    "session_high",
    "session_low",
    "hi_op",
    "op_lo",
    "pHi",
    "pLo",
    "hi_phi",
    "lo_plo",
]

try:
    min_time = df["time"].min()
    max_time = df["time"].max()
    if pd.notna(min_time) and pd.notna(max_time):
        enr_resp = (
            sb.table("es_30m_enriched")
              .select(",".join(enriched_cols))
              .gte("time", min_time.isoformat())
              .lte("time", max_time.isoformat())
              .execute()
        )
        enr_df = pd.DataFrame(enr_resp.data)
        if not enr_df.empty:
            enr_df["time"] = pd.to_datetime(enr_df["time"], utc=True, errors="coerce")
            enr_df = enr_df.dropna(subset=["time"])
            # Drop any duplicate times in enriched just in case
            enr_df = (
                enr_df.sort_values("time")
                      .drop_duplicates(subset=["time"], keep="last")
            )
            df = df.merge(enr_df, on="time", how="left", suffixes=("", "_enriched"))
except Exception as e:
    st.warning(f"Could not join es_30m_enriched; falling back to local calculations: {e}")

# =========================
# Ensure session-level fields exist (prefer enriched, fallback to local)
# =========================

# cum_vol
if "cum_vol" not in df.columns or df["cum_vol"].isna().all():
    df["cum_vol"] = df.groupby("trade_day")["Volume"].cumsum()

# session_high / session_low
if "session_high" not in df.columns or df["session_high"].isna().all():
    df["session_high"] = df.groupby("trade_day")["high"].cummax()
if "session_low" not in df.columns or df["session_low"].isna().all():
    df["session_low"]  = df.groupby("trade_day")["low"].cummin()

# hi_op / op_lo
if "hi_op" not in df.columns or df["hi_op"].isna().all():
    df["hi_op"]  = df["high"] - df["open"]
if "op_lo" not in df.columns or df["op_lo"].isna().all():
    df["op_lo"]  = df["open"] - df["low"]

# pHi / pLo – prefer enriched, fallback to prior-day extrema
if ("pHi" not in df.columns) or df["pHi"].isna().all() or \
   ("pLo" not in df.columns) or df["pLo"].isna().all():
    daily_hi_lo = (
        df.groupby("trade_day")
          .agg(day_high=("high","max"), day_low=("low","min"))
          .sort_index()
          .reset_index()
    )
    daily_hi_lo["pHi_fallback"] = daily_hi_lo["day_high"].shift(1)
    daily_hi_lo["pLo_fallback"] = daily_hi_lo["day_low"].shift(1)
    df = df.merge(
        daily_hi_lo[["trade_day","pHi_fallback","pLo_fallback"]],
        on="trade_day",
        how="left"
    )
    if "pHi" not in df.columns:
        df["pHi"] = df["pHi_fallback"]
    else:
        df["pHi"] = df["pHi"].where(df["pHi"].notna(), df["pHi_fallback"])
    if "pLo" not in df.columns:
        df["pLo"] = df["pLo_fallback"]
    else:
        df["pLo"] = df["pLo"].where(df["pLo"].notna(), df["pLo_fallback"])
    df = df.drop(columns=[c for c in ["pHi_fallback","pLo_fallback"] if c in df.columns])

# hi_pHi / lo_pLo from enriched hi_phi / lo_plo where available; fallback to computed deltas
if "hi_phi" in df.columns and df["hi_phi"].notna().any():
    if "hi_pHi" not in df.columns:
        df["hi_pHi"] = df["hi_phi"]
    else:
        df["hi_pHi"] = df["hi_pHi"].where(df["hi_pHi"].notna(), df["hi_phi"])
else:
    if "hi_pHi" not in df.columns:
        df["hi_pHi"] = df["high"] - df["pHi"]
    else:
        mask = df["hi_pHi"].isna()
        df.loc[mask, "hi_pHi"] = df.loc[mask, "high"] - df.loc[mask, "pHi"]

if "lo_plo" in df.columns and df["lo_plo"].notna().any():
    if "lo_pLo" not in df.columns:
        df["lo_pLo"] = df["lo_plo"]
    else:
        df["lo_pLo"] = df["lo_pLo"].where(df["lo_pLo"].notna(), df["lo_plo"])
else:
    if "lo_pLo" not in df.columns:
        df["lo_pLo"] = df["low"] - df["pLo"]
    else:
        mask = df["lo_pLo"].isna()
        df.loc[mask, "lo_pLo"] = df.loc[mask, "low"] - df.loc[mask, "pLo"]

# Helper: trailing mean (shifted to avoid look-ahead)
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean().shift(1)

# =========================
# Aggregate per trade_day (respects current filter/as-of)
# =========================
def agg_daily(scope: pd.DataFrame) -> pd.DataFrame:
    # Basic opens/closes
    first_last = scope.groupby("trade_day").agg(
        day_open=("open","first"),
        day_close=("close","last"),
    )

    # Highs/Lows/Volume
    hilo = scope.groupby("trade_day").agg(
        day_high=("high","max"),
        day_low=("low","min"),
        day_volume=("Volume","sum"),
    )
    bars = scope.groupby("trade_day").size().rename("bars_in_day")

    # Per-bar averages
    perbar = scope.groupby("trade_day").agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
    )

    # Last values at the current cutoff
    lasts = scope.groupby("trade_day").agg(
        session_high_at_cutoff=("session_high","last"),
        session_low_at_cutoff=("session_low","last"),
        cum_vol_at_cutoff=("cum_vol","last"),
    )

    out = (
        first_last.join(hilo)
                  .join(bars)
                  .join(perbar)
                  .join(lasts)
                  .reset_index()
                  .sort_values("trade_day")
    )
    out["day_range"]  = out["day_high"] - out["day_low"]
    out["trade_date"] = out["trade_day"].dt.date
    return out

daily = agg_daily(df)

# Trailing comparisons (exclude current day via shift inside trailing_mean)
for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], trailing_window)
        daily[f"{col}_pct_vs_tw"] = (daily[col] - daily[f"{col}_tw_avg"]) / daily[f"{col}_tw_avg"]

# % pMid hit by session (prev-bar midpoint within the session)
def session_pmid_percent(scope: pd.DataFrame, label: str) -> pd.DataFrame:
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

daily = (
    daily.merge(on_hit,  on="trade_day", how="left")
         .merge(ib_hit,  on="trade_day", how="left")
         .merge(rth_hit, on="trade_day", how="left")
         .sort_values("trade_day")
)

for ses in ["ON","IB","RTH"]:
    col = f"{ses}_pMid_hit_pct"
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], trailing_window)
        daily[f"{col}_pct_vs_tw"] = daily[col] - daily[f"{col}_tw_avg"]

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
    st.warning("No trade days after processing—adjust controls.")
    st.stop()

# =========================
# KPI (latest day)
# =========================
latest_td = daily["trade_day"].max()
row = daily.loc[daily["trade_day"] == latest_td].iloc[0]

hdr = f"{symbol} — {'As-of ' + asof_time.strftime('%H:%M ET') if asof_time else 'Full day'}"
st.subheader(hdr)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"Range vs {trailing_window}d",
    f"{row.get('day_range', np.nan):.2f}" if pd.notna(row.get('day_range')) else "—",
    None if pd.isna(row.get('day_range_pct_vs_tw')) else f"{row['day_range_pct_vs_tw']*100:+.1f}%"
)
k2.metric(
    f"Cum Vol vs {trailing_window}d",
    f"{row.get('cum_vol_at_cutoff', np.nan):,.0f}" if pd.notna(row.get('cum_vol_at_cutoff')) else "—",
    None  # you can add % vs trailing later if you want
)
k3.metric(
    "Session High (cutoff)",
    f"{row.get('session_high_at_cutoff', np.nan):.2f}" if pd.notna(row.get('session_high_at_cutoff')) else "—"
)
k4.metric(
    "Session Low (cutoff)",
    f"{row.get('session_low_at_cutoff',  np.nan):.2f}" if pd.notna(row.get('session_low_at_cutoff'))  else "—"
)

st.caption(
    "Session metrics come from **es_30m_enriched** and reset at 18:00 ET. "
    "Cumulative volume and session high/low are taken at the last bar in the current filter "
    "(full-day = last bar of the session; as-of snapshot = last bar before the cutoff). "
    "Trailing averages exclude the current day."
)

# =========================
# Table (last N trade days)
# =========================
st.markdown("### Conditions vs Trailing (last N trade days)")
max_slider = int(min(120, trade_days_to_keep, len(daily)))
show_days = st.slider("Show last N trade days", 10, max_slider, min(30, max_slider), 5)

cols = [
    "trade_date",
    "day_open","day_high","day_low","day_close","day_range",
    "session_high_at_cutoff","session_low_at_cutoff",
    "cum_vol_at_cutoff","day_volume","day_volume_tw_avg",
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    "avg_hi_pHi","avg_hi_pHi_tw_avg",
    "avg_lo_pLo","avg_lo_pLo_tw_avg",
    "ON_pMid_hit_pct","ON_pMid_hit_pct_tw_avg",
    "IB_pMid_hit_pct","IB_pMid_hit_pct_tw_avg",
    "RTH_pMid_hit_pct","RTH_pMid_hit_pct_tw_avg",
    "bars_in_day",
]
existing = [c for c in cols if c in daily.columns]
tbl = daily[existing].tail(int(show_days)).copy()

labels = {
    "trade_date":"Trade Date",
    "day_open":"Open","day_high":"High","day_low":"Low","day_close":"Close","day_range":"Range",
    "session_high_at_cutoff":"Session High (cutoff)","session_low_at_cutoff":"Session Low (cutoff)",
    "cum_vol_at_cutoff":"Cum Vol (cutoff)","day_volume":"Vol (day)","day_volume_tw_avg":f"Vol {trailing_window}d",
    "avg_hi_op":"Avg(Hi-Op)","avg_hi_op_tw_avg":f"Avg(Hi-Op) {trailing_window}d","avg_hi_op_pct_vs_tw":f"Avg(Hi-Op) vs {trailing_window}d",
    "avg_op_lo":"Avg(Op-Lo)","avg_op_lo_tw_avg":f"Avg(Op-Lo) {trailing_window}d","avg_op_lo_pct_vs_tw":f"Avg(Op-Lo) vs {trailing_window}d",
    "avg_hi_pHi":"Avg(Hi-pHi)","avg_hi_pHi_tw_avg":f"Avg(Hi-pHi) {trailing_window}d",
    "avg_lo_pLo":"Avg(Lo-pLo)","avg_lo_pLo_tw_avg":f"Avg(Lo-pLo) {trailing_window}d",
    "ON_pMid_hit_pct":"ON % pMid Hit","ON_pMid_hit_pct_tw_avg":f"ON % pMid Hit {trailing_window}d",
    "IB_pMid_hit_pct":"IB % pMid Hit","IB_pMid_hit_pct_tw_avg":f"IB % pMid Hit {trailing_window}d",
    "RTH_pMid_hit_pct":"RTH % pMid Hit","RTH_pMid_hit_pct_tw_avg":f"RTH % pMid Hit {trailing_window}d",
    "bars_in_day":"Bars"
}
tbl = tbl.rename(columns=labels)

# Formatting
fmt = {}
for name in tbl.columns:
    if name == "Trade Date":
        continue
    if "Vol" in name or "Vol (" in name:
        fmt[name] = "{:,.0f}"
    elif "% pMid Hit" in name:
        fmt[name] = "{:.1%}"
    elif "vs" in name:
        fmt[name] = "{:,.2f}"
    elif name in ["Bars"]:
        fmt[name] = "{:,.0f}"
    else:
        fmt[name] = "{:,.2f}"

def color_pos_neg(val):
    if pd.isna(val): 
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    return f"color: {'#16a34a' if v>0 else ('#dc2626' if v<0 else '#111827')};"

vs_cols = [c for c in tbl.columns if "vs" in c and "% pMid Hit" not in c]
styled = (
    tbl.style
       .format(fmt)
       .applymap(color_pos_neg, subset=vs_cols)
       .set_properties(subset=["Trade Date"], **{"font-weight":"600"})
)
st.dataframe(styled, use_container_width=True)

with st.expander("Data health (debug)"):
    st.write({
        "latest_bar_time_et": str(latest_bar_time),
        "expected_latest_trade_day": str(latest_trade_day_expected),
        "kept_trade_days_desc": [str(d) for d in recent_trade_days_desc.tolist()],
        "rows_total_after_filters": int(len(df)),
    })
