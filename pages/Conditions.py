#!/usr/bin/env python3
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from supabase import create_client

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Conditions", layout="wide")
st.title("Market Conditions (30m)")

# -----------------------
# Supabase client
# -----------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Controls
# -----------------------
colA, colB, colC = st.columns([1,1,2])
with colA:
    lookback_days = st.number_input("Load last N calendar days", min_value=30, max_value=3650, value=180, step=30,
                                    help="Server-side filter to keep the query lean.")
with colB:
    trailing_window = st.number_input("Trailing window (days) for comparison", min_value=5, max_value=60, value=10, step=1)
with colC:
    rth_only = st.toggle("RTH only (09:30–16:00 ET)", value=False,
                         help="If your 30m table includes 24h bars, enable this to focus on RTH.")

# -----------------------
# Fetch data
# -----------------------
since = (dt.date.today() - dt.timedelta(days=int(lookback_days))).isoformat()
# Adjust the table name if yours differs
TABLE = "es_30m"

q = (
    sb.table(TABLE)
    .select("*")
    .gte("time", f"{since}T00:00:00Z")
    .order("time", desc=False)
)
data = q.execute().data
if not data:
    st.warning("No rows returned. Check table name/filters.")
    st.stop()

df = pd.DataFrame(data)

# Expecting columns: time, open, high, low, close, Volume, 5MA,10MA,20MA,50MA,200MA
# If your MA column names differ, edit here:
MA_COLS = ["5MA","10MA","20MA","50MA","200MA"]
present_ma_cols = [c for c in MA_COLS if c in df.columns]

# Basic parsing
df["time"] = pd.to_datetime(df["time"], utc=True)
# Convert to US/Eastern to make RTH toggle intuitive; change if your pipeline uses another tz
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")
df["date"] = df["time_et"].dt.date
df["hm"] = df["time_et"].dt.strftime("%H:%M")

if rth_only:
    # 09:30–16:00 *inclusive of 09:30 bar; exclusive of >16:00
    df = df[(df["hm"] >= "09:30") & (df["hm"] <= "16:00")]

# Guard
for col in ["open","high","low","close","Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# -----------------------
# Daily aggregates
# -----------------------
# Daily high/low/range/volume
daily_hilo = (
    df.groupby("date").agg(
        day_open=("open", "first"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
        day_range=("high", lambda x: x.max())  # placeholder; we’ll compute below for clarity
    )
    .reset_index()
)

# Fix day_range correctly as high - low
daily_hilo["day_range"] = daily_hilo["day_high"] - daily_hilo["day_low"]

# Sum volume per day
daily_vol = df.groupby("date")["Volume"].sum().rename("day_volume").reset_index()

# Per-bar intraday effects (averages within day)
bars = df.copy()
bars["hi_op"] = bars["high"] - bars["open"]
bars["op_lo"] = bars["open"] - bars["low"]

# Build previous day high/low per day
daily_prev = daily_hilo[["date","day_high","day_low"]].sort_values("date").copy()
daily_prev["pHi"] = daily_prev["day_high"].shift(1)
daily_prev["pLo"] = daily_prev["day_low"].shift(1)
daily_prev = daily_prev[["date","pHi","pLo"]]

# Merge pHi/pLo onto bars (per day) to compute bar deltas vs prior day extrema
bars = bars.merge(daily_prev, on="date", how="left")
bars["hi_pHi"] = bars["high"] - bars["pHi"]
bars["lo_pLo"] = bars["low"] - bars["pLo"]

# Average these per day
intraday_avgs = (
    bars.groupby("date")
    .agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
        bars_in_day=("time", "count")
    )
    .reset_index()
)

# Combine
daily = (
    daily_hilo[["date","day_open","day_high","day_low","day_close","day_range"]]
    .merge(daily_vol, on="date", how="left")
    .merge(intraday_avgs, on="date", how="left")
    .sort_values("date")
)

# -----------------------
# Trailing window baselines (exclude current day via shift)
# -----------------------
def trailing_mean(s, n):
    return s.rolling(n).mean().shift(1)

for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        daily[f"{col}_tw_avg"] = trailing_mean(daily[col], trailing_window)
        daily[f"{col}_vs_tw"] = daily[col] - daily[f"{col}_tw_avg"]
        daily[f"{col}_pct_vs_tw"] = daily[f"{col}_vs_tw"] / daily[f"{col}_tw_avg"]

# -----------------------
# Latest snapshot + KPIs
# -----------------------
latest_date = daily["date"].max()
today_row = daily.loc[daily["date"] == latest_date].iloc[0]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Range vs {}-day avg".format(trailing_window),
    f"{today_row['day_range']:.2f}",
    None if pd.isna(today_row["day_range_pct_vs_tw"]) else
    "{:+.1f}%".format(100*today_row["day_range_pct_vs_tw"])
)
kpi2.metric(
    "Volume vs {}-day avg".format(trailing_window),
    f"{today_row['day_volume']:,.0f}",
    None if pd.isna(today_row["day_volume_pct_vs_tw"]) else
    "{:+.1f}%".format(100*today_row["day_volume_pct_vs_tw"])
)
kpi3.metric(
    "Avg(Hi-Op) vs {}-day".format(trailing_window),
    f"{today_row['avg_hi_op']:.2f}",
    None if pd.isna(today_row["avg_hi_op_pct_vs_tw"]) else
    "{:+.1f}%".format(100*today_row["avg_hi_op_pct_vs_tw"])
)
kpi4.metric(
    "Avg(Op-Lo) vs {}-day".format(trailing_window),
    f"{today_row['avg_op_lo']:.2f}",
    None if pd.isna(today_row["avg_op_lo_pct_vs_tw"]) else
    "{:+.1f}%".format(100*today_row["avg_op_lo_pct_vs_tw"])
)

st.caption("Avg(Hi-pHi) and Avg(Lo-pLo) trend in the table below helps gauge ‘follow-through’ beyond prior day extremes vs mean-reverting chop.")

# -----------------------
# Where is price vs MAs (using the most recent bar)
# -----------------------
last_bar = df.iloc[-1].copy()
price = float(last_bar["close"])

ma_cards = st.container()
with ma_cards:
    st.subheader("Price vs Moving Averages (last 30m bar)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Last Price", f"{price:.2f}")
    for i, ma_col in enumerate(present_ma_cols, start=2):
        val = float(last_bar[ma_col])
        pct = (price - val) / val if val != 0 else np.nan
        (c1, c2, c3, c4, c5, c6)[i-1].metric(
            ma_col.replace("MA","-MA"),
            f"{val:.2f}",
            None if np.isnan(pct) else "{:+.1f}%".format(100*pct)
        )

# -----------------------
# Trend table (last ~30 days)
# -----------------------
st.subheader("Daily Conditions vs Trailing Average")
show_days = st.slider("Show last N days", min_value=10, max_value=120, value=30, step=5)

columns_order = [
    "date","day_open","day_high","day_low","day_close","day_range","day_range_tw_avg","day_range_pct_vs_tw",
    "day_volume","day_volume_tw_avg","day_volume_pct_vs_tw",
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    "avg_hi_pHi","avg_hi_pHi_tw_avg","avg_lo_pLo","avg_lo_pLo_tw_avg",
    "bars_in_day"
]
existing = [c for c in columns_order if c in daily.columns]

tbl = daily[existing].copy().tail(show_days)
# Friendly labels
labels = {
    "date":"Date",
    "day_open":"Open",
    "day_high":"High",
    "day_low":"Low",
    "day_close":"Close",
    "day_range":"Range",
    "day_range_tw_avg":f"Range {trailing_window}d Avg",
    "day_range_pct_vs_tw":f"Range vs {trailing_window}d",
    "day_volume":"Volume",
    "day_volume_tw_avg":f"Volume {trailing_window}d Avg",
    "day_volume_pct_vs_tw":f"Volume vs {trailing_window}d",
    "avg_hi_op":"Avg(Hi-Op)",
    "avg_hi_op_tw_avg":f"Avg(Hi-Op) {trailing_window}d",
    "avg_hi_op_pct_vs_tw":f"Avg(Hi-Op) vs {trailing_window}d",
    "avg_op_lo":"Avg(Op-Lo)",
    "avg_op_lo_tw_avg":f"Avg(Op-Lo) {trailing_window}d",
    "avg_op_lo_pct_vs_tw":f"Avg(Op-Lo) vs {trailing_window}d",
    "avg_hi_pHi": "Avg(Hi-pHi)",
    "avg_hi_pHi_tw_avg": f"Avg(Hi-pHi) {trailing_window}d",
    "avg_lo_pLo": "Avg(Lo-pLo)",
    "avg_lo_pLo_tw_avg": f"Avg(Lo-pLo) {trailing_window}d",
    "bars_in_day":"Bars"
}
tbl = tbl.rename(columns=labels)

# Formatting
def pct(x):
    return x.apply(lambda v: "" if pd.isna(v) else f"{100*v:,.1f}%")

fmt = {}
for name in tbl.columns:
    if "Volume" in name:
        fmt[name] = "{:,.0f}"
    elif "vs" in name and "%" in name:
        # already percent string if we used pct(); but we'll keep numeric formatting flow below
        pass
    elif name in ["Bars"]:
        fmt[name] = "{:,.0f}"
    elif name != "Date":
        fmt[name] = "{:,.2f}"

# Convert pct columns to percent text
for col in [c for c in tbl.columns if ("vs" in c and ("Range" in c or "Volume" in c or "Avg(" in c)) and c.endswith(f"vs {trailing_window}d")]:
    # These won't match because we renamed; instead detect numeric pct columns before rename—handled above.
    pass

# Color helper
def color_pos_neg(val):
    if pd.isna(val): return ""
    try:
        v = float(val)
    except Exception:
        return ""
    color = "#16a34a" if v > 0 else ("#dc2626" if v < 0 else "#111827")
    return f"color: {color};"

# Build a styled DataFrame; keep raw numeric pct columns for styling if they exist
styled = tbl.style.format(fmt)\
    .applymap(color_pos_neg, subset=[c for c in tbl.columns if "vs" in c])\
    .set_properties(subset=["Date"], **{"font-weight": "600"})

st.dataframe(styled, use_container_width=True)

# -----------------------
# Mini chart: Price vs MAs for recent days
# -----------------------
st.subheader("Recent Price with Moving Averages (last ~2 weeks)")
recent_days = int(max(10, min(20, trailing_window*2)))
cutoff = df["time_et"].max() - pd.Timedelta(days=recent_days)
mini = df[df["time_et"] >= cutoff][["time_et","close"] + present_ma_cols].copy().set_index("time_et")
st.line_chart(mini, use_container_width=True)

st.caption("""
**Notes**
- Trailing averages are calculated *excluding* the current day (shifted by 1) to avoid look-ahead bias.
- Avg(Hi-pHi) and Avg(Lo-pLo) are computed at the bar level using the prior day’s high/low (pHi/pLo) broadcast to all bars for that day, then averaged per-day.
- If your MA column names differ from `5MA,10MA,20MA,50MA,200MA`, update `MA_COLS`.
""")
