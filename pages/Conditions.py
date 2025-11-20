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

# =========================
# Supabase client
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Daily performance snapshot from daily_es
# =========================

def _period_return_open_to_close(sub_df: pd.DataFrame) -> float:
    """
    Period return using first period_open vs last period_close in a subset.
    If there's only one row, it's (close / open) - 1 for that day.
    Returns NaN if no data or bad values.
    """
    if sub_df is None or sub_df.empty:
        return np.nan
    first_open = sub_df["period_open"].iloc[0]
    last_close = sub_df["period_close"].iloc[-1]
    if pd.isna(first_open) or pd.isna(last_close) or first_open == 0:
        return np.nan
    return (last_close / first_open) - 1.0

day_ret = wtd_ret = mtd_ret = ytd_ret = np.nan
daily_agg = None
y_subset = m_subset = w_subset = None  # (kept for clarity, not shown)

try:
    # IMPORTANT: get the MOST RECENT rows, then sort ascending in pandas
    perf_resp = (
        sb.table("daily_es")
          .select("time, open, close")
          .order("time", desc=True)  # newest -> oldest in DB
          .limit(260)                # most recent ~260 rows
          .execute()
    )
    perf_df = pd.DataFrame(perf_resp.data)

    if not perf_df.empty:
        if not {"time", "open", "close"}.issubset(perf_df.columns):
            st.warning(
                "Expected columns 'time', 'open', 'close' in `daily_es` for performance tiles. "
                f"Columns available: {list(perf_df.columns)}"
            )
        else:
            # Convert to date and aggregate; now these are the most recent dates
            perf_df["time"] = pd.to_datetime(perf_df["time"]).dt.date
            perf_df = perf_df.dropna(subset=["time", "open", "close"])

            if not perf_df.empty:
                daily_agg = (
                    perf_df
                    .groupby("time")
                    .agg(
                        period_open=("open", "first"),
                        period_close=("close", "last"),
                    )
                    .reset_index()
                    .sort_values("time")   # oldest -> newest of the most recent 260 days
                )

                if len(daily_agg) >= 1:
                    # Latest date in the table (true latest because we just sorted ascending)
                    current_date = daily_agg["time"].iloc[-1]

                    # --- Day performance: latest close vs previous day's close ---
                    if len(daily_agg) >= 2:
                        last_two = daily_agg.tail(2)
                        prev_close = last_two["period_close"].iloc[0]
                        last_close = last_two["period_close"].iloc[1]
                        if prev_close and not pd.isna(prev_close) and prev_close != 0:
                            day_ret = (last_close / prev_close) - 1.0
                        else:
                            day_ret = np.nan
                    else:
                        day_ret = np.nan  # only 1 day

                    # Prepare datetime series for filters
                    dt_series = pd.to_datetime(daily_agg["time"])

                    # === YTD: latest close vs first OPEN of the calendar year ===
                    current_year = current_date.year
                    y_mask = dt_series.dt.year == current_year
                    y_subset = daily_agg.loc[y_mask].copy()
                    ytd_ret = _period_return_open_to_close(y_subset)

                    # === MTD: latest close vs first OPEN of the calendar month ===
                    current_month = current_date.month
                    m_mask = (dt_series.dt.year == current_year) & (dt_series.dt.month == current_month)
                    m_subset = daily_agg.loc[m_mask].copy()
                    mtd_ret = _period_return_open_to_close(m_subset)

                    # === WTD (ISO week): latest close vs first OPEN of the ISO week ===
                    iso_all = dt_series.dt.isocalendar()
                    curr_iso_year, curr_iso_week, _ = pd.Timestamp(current_date).isocalendar()
                    w_mask = (iso_all["year"] == curr_iso_year) & (iso_all["week"] == curr_iso_week)
                    w_subset = daily_agg.loc[w_mask].copy()
                    wtd_ret = _period_return_open_to_close(w_subset)
except Exception as e:
    st.warning(f"Could not load daily_es for performance tiles: {e}")

# =========================
# Title + top tiles
# =========================
st.title("Market Conditions")

c_day, c_week, c_month, c_year = st.columns(4)

def _perf_tile(container, label: str, ret: float):
    """
    Clean Streamlit tile with:
    - green/red/gray background
    - centered layout
    """
    if pd.isna(ret):
        display = "—"
        bg = "#E5E7EB"   # gray-200
    else:
        display = f"{ret*100:+.1f}%"
        if ret > 0:
            bg = "#86EFAC"  # green-300
        elif ret < 0:
            bg = "#FCA5A5"  # red-300
        else:
            bg = "#E5E7EB"  # gray-200

    html = f"""
        <div style="background-color:{bg};
                    border:1px solid #9CA3AF;
                    border-radius:12px;
                    height:90px;
                    padding:8px 12px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;">
            <div style="color:#374151;
                        font-size:0.75rem;
                        text-transform:uppercase;
                        letter-spacing:0.05em;
                        text-align:center;
                        margin-bottom:2px;">
                {label}
            </div>
            <div style="color:#111827;
                        font-size:1.6rem;
                        font-weight:600;
                        text-align:center;">
                {display}
            </div>
        </div>
    """

    container.markdown(html, unsafe_allow_html=True)

_perf_tile(c_day,   "Day performance",   day_ret)
_perf_tile(c_week,  "Week-to-date",      wtd_ret)
_perf_tile(c_month, "Month-to-date",     mtd_ret)
_perf_tile(c_year,  "Year-to-date",      ytd_ret)

# =========================
# Controls
# =========================
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.3])
with c1:
    symbol = st.text_input("Symbol", value="ES")
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
# Keeps bars: (trade_day-1 @ 18:00) → (trade_day @ HH:MM) for EVERY trade_day
# So prior days are also truncated at the same intraday cutoff -> apples-to-apples.
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

# hi_pHi / lo_pLo
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
        df.loc[mask, "lo_pLo"] = df.loc[mask, "low"] - df["pLo"]

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

# Trailing comparisons (apples-to-apples because scope already respects Full/As-of mode)
for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        avg_col = f"{col}_tw_avg"
        pct_col = f"{col}_pct_vs_tw"
        daily[avg_col] = trailing_mean(daily[col], trailing_window)
        # guard against divide-by-zero so you don't see crazy values when avg ~ 0
        daily[pct_col] = np.where(
            daily[avg_col].abs() > 0,
            (daily[col] - daily[avg_col]) / daily[avg_col],
            np.nan,
        )

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
        avg_col = f"{col}_tw_avg"
        pct_col = f"{col}_pct_vs_tw"
        daily[avg_col] = trailing_mean(daily[col], trailing_window)
        # For hit pct, pct_vs_tw is a difference (not ratio) so you see +/- absolute change
        daily[pct_col] = daily[col] - daily[avg_col]

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
    None
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
    "Trailing averages exclude the current day and are computed on the same filtered data "
    "(full-day or as-of), so comparisons are apples-to-apples."
)

# ======================================================
# Range vs prior RTH (Option A) + Moving averages (MA cards only)
# ======================================================
st.markdown("### Intraday Context: Prior RTH Range & Moving Averages")

col_range, col_ma_cards = st.columns([2, 2])

# ----- helper: range position bar (Option A, fixed) -----
def _render_range_position_bar(container, last_price, prev_rth_low, prev_rth_high,
                               sess_low, sess_high):
    # Need all key pieces
    if any(pd.isna(x) for x in [last_price, prev_rth_low, prev_rth_high, sess_low, sess_high]):
        container.info("Not enough data to show prior RTH range vs current session.")
        return
    if prev_rth_high <= prev_rth_low:
        container.info("Prior RTH range invalid (High <= Low).")
        return

    # Combined span so extensions beyond prior RTH are visible
    base_min = min(prev_rth_low, sess_low)
    base_max = max(prev_rth_high, sess_high)
    span = base_max - base_min
    if span <= 0:
        container.info("Range span is zero; cannot render bar.")
        return

    def _pos(value: float) -> float:
        return max(0.0, min(1.0, (value - base_min) / span)) * 100.0

    prior_low_pos  = _pos(prev_rth_low)
    prior_high_pos = _pos(prev_rth_high)
    sess_low_pos   = _pos(sess_low)
    sess_high_pos  = _pos(sess_high)
    last_pos       = _pos(last_price)

    html = f"""
        <div style="font-size:0.9rem; margin-bottom:4px; color:#111827;">
            Prior RTH range vs current session
        </div>
        <div style="font-size:0.8rem; color:#4B5563; margin-bottom:6px;">
            Prev RTH Lo: {prev_rth_low:.2f} &nbsp;&nbsp; Prev RTH Hi: {prev_rth_high:.2f}<br/>
            Session Lo: {sess_low:.2f} &nbsp;&nbsp; Last: {last_price:.2f} &nbsp;&nbsp; Session Hi: {sess_high:.2f}
        </div>
        <div style="position:relative;
                    height:32px;
                    border-radius:999px;
                    background-color:#E5E7EB;
                    border:1px solid #D1D5DB;
                    overflow:hidden;
                    margin-bottom:4px;">
            <!-- prior RTH band -->
            <div style="position:absolute;
                        top:0;
                        bottom:0;
                        left:{prior_low_pos:.1f}%;
                        right:{100.0-prior_high_pos:.1f}%;
                        background:linear-gradient(90deg,#FCA5A5,#BBF7D0);
                        opacity:0.8;">
            </div>

            <!-- LAST price marker -->
            <div style="position:absolute;
                        top:4px;
                        bottom:4px;
                        left:{last_pos:.1f}%;
                        width:2px;
                        background-color:#111827;">
            </div>

            <!-- Session Low marker -->
            <div style="position:absolute;
                        top:0;
                        bottom:0;
                        left:{sess_low_pos:.1f}%;
                        width:2px;
                        background-color:#DC2626;
                        opacity:0.9;">
            </div>

            <!-- Session High marker -->
            <div style="position:absolute;
                        top:0;
                        bottom:0;
                        left:{sess_high_pos:.1f}%;
                        width:2px;
                        background-color:#16A34A;
                        opacity:0.9;">
            </div>
        </div>
        <div style="display:flex;
                    justify-content:space-between;
                    font-size:0.75rem;
                    color:#6B7280;
                    margin-top:2px;">
            <span>Min of (Prev RTH / Session)</span>
            <span>Max of (Prev RTH / Session)</span>
        </div>
    """

    container.markdown(html, unsafe_allow_html=True)

# ----- helper: MA cards (Option B) -----
def _render_ma_cards(container, ma_rows):
    if not ma_rows:
        container.info("No moving average columns found in es_30m.")
        return

    container.markdown("**MA snapshot (cards)**")
    c1, c2 = container.columns(2)
    cols = [c1, c2]

    for i, ma in enumerate(ma_rows):
        label = ma["label"]
        val = ma["value"]
        dist = ma["dist"]
        if pd.isna(val) or pd.isna(dist):
            text = "N/A"
            bg = "#E5E7EB"
        else:
            text = f"{val:.2f} ({dist:+.1f} pts)"
            if dist > 0:
                bg = "#BBF7D0"  # green-200
            elif dist < 0:
                bg = "#FECACA"  # red-200
            else:
                bg = "#E5E7EB"

        html = f"""
            <div style="background-color:{bg};
                        border:1px solid #D1D5DB;
                        border-radius:10px;
                        padding:8px 10px;
                        margin-bottom:6px;">
                <div style="font-size:0.8rem; color:#4B5563; margin-bottom:2px;">
                    {label}
                </div>
                <div style="font-size:1rem; font-weight:600; color:#111827;">
                    {text}
                </div>
            </div>
        """
        cols[i % 2].markdown(html, unsafe_allow_html=True)

# ----- build data needed for these visuals -----
current_close = row.get("day_close", np.nan)
sess_high_at = row.get("session_high_at_cutoff", np.nan)
sess_low_at  = row.get("session_low_at_cutoff", np.nan)

# Prior RTH range (from es_trade_day_summary)
prev_rth_low = prev_rth_high = np.nan
try:
    # Use quoted column names from your schema: "RTH Hi", "RTH Lo"
    summ_resp = (
        sb.table("es_trade_day_summary")
          .select('trade_date,"RTH Hi","RTH Lo"')
          .lte("trade_date", row["trade_date"].isoformat())
          .order("trade_date", desc=True)
          .limit(10)
          .execute()
    )
    summ_df = pd.DataFrame(summ_resp.data)
    if not summ_df.empty:
        summ_df["trade_date"] = pd.to_datetime(summ_df["trade_date"]).dt.date
        # Rename to simpler keys
        summ_df = summ_df.rename(columns={"RTH Hi": "rth_hi", "RTH Lo": "rth_lo"})
        latest_date = row["trade_date"]
        prev_mask = summ_df["trade_date"] < latest_date
        if prev_mask.any():
            prev_row = summ_df.loc[prev_mask].sort_values("trade_date").iloc[-1]
        else:
            # fallback: use latest available as "prior"
            prev_row = summ_df.sort_values("trade_date").iloc[-1]
        prev_rth_low = prev_row.get("rth_lo", np.nan)
        prev_rth_high = prev_row.get("rth_hi", np.nan)
except Exception as e:
    st.warning(f"Could not load es_trade_day_summary for prior RTH range: {e}")

# Last price from latest bar in df for latest_td (respects as-of filter)
last_price = np.nan
try:
    latest_bars = df[df["trade_day"] == latest_td]
    if not latest_bars.empty:
        last_price = latest_bars.iloc[-1].get("close", np.nan)
except Exception as e:
    st.warning(f"Could not determine last price for range bar: {e}")

# Render the range bar
_render_range_position_bar(
    col_range,
    last_price,
    prev_rth_low,
    prev_rth_high,
    sess_low_at,
    sess_high_at,
)

# Moving averages for the latest bar of the latest trade_day (MA cards only)
ma_rows = []
try:
    latest_bars = df[df["trade_day"] == latest_td]
    if not latest_bars.empty:
        last_bar = latest_bars.iloc[-1]
        price = last_bar.get("close", np.nan)

        ma_candidates = ["5MA", "10MA", "20MA", "50MA", "200MA"]
        ma_cols = [c for c in ma_candidates if c in df.columns]

        for col in ma_cols:
            val = last_bar.get(col, np.nan)
            dist = np.nan
            if pd.notna(price) and pd.notna(val):
                dist = price - val
            ma_rows.append({"label": col, "value": val, "dist": dist})

except Exception as e:
    st.warning(f"Could not compute MA context: {e}")

_render_ma_cards(col_ma_cards, ma_rows)

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
