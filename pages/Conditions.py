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
# Helpers
# =========================
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean().shift(1)

# Fixed config (replaces removed filters)
SYMBOL = "ES"
TRAILING_WINDOW = 10
TRADE_DAYS_TO_KEEP = 180

# =========================
# Title + placeholders
# =========================
st.title("Market Conditions")

# (1) Performance tiles
c_day, c_week, c_month, c_year = st.columns(4)

def _perf_tile(container, label: str, ret: float):
    if pd.isna(ret):
        display = "—"
        bg = "#E5E7EB"
    else:
        display = f"{ret*100:+.1f}%"
        bg = "#86EFAC" if ret > 0 else ("#FCA5A5" if ret < 0 else "#E5E7EB")

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
            <div style="color:#374151; font-size:0.75rem; text-transform:uppercase;
                        letter-spacing:0.05em; text-align:center; margin-bottom:2px;">
                {label}
            </div>
            <div style="color:#111827; font-size:1.6rem; font-weight:600; text-align:center;">
                {display}
            </div>
        </div>
    """
    container.markdown(html, unsafe_allow_html=True)

# -------- PLACEHOLDERS so visuals render ABOVE the rest --------
ma_placeholder = st.container()      # (2) MA snapshot here
range_placeholder = st.container()   # (3) Current vs prior RTH range here

# =========================
# Fetch (newest-first) + choose last N trade days
# =========================
TABLE = "es_30m"
rows_per_day_guess = 48
rows_to_load = int(max(2000, TRADE_DAYS_TO_KEEP * rows_per_day_guess * 1.5))

response = (
    sb.table(TABLE)
      .select("*")
      .order("time", desc=True)
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

et = df["time_et"]
midnight = et.dt.floor("D")
roll = (et.dt.hour >= 18).astype("int64")
df["trade_day"] = midnight + pd.to_timedelta(roll, unit="D")
df["trade_date"] = df["trade_day"].dt.date

# --- Session flags ---
t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30)))
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30)))
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0)))

# --- Keep most recent N trade days (sort ASC later) ---
per_day_last_bar = (
    df.groupby("trade_day")["time_et"]
      .max()
      .sort_values(ascending=False)
)
recent_trade_days_desc = per_day_last_bar.index[:int(TRADE_DAYS_TO_KEEP)]
recent_mask = df["trade_day"].isin(set(recent_trade_days_desc))
df = df[recent_mask].copy()
df = df.sort_values(["trade_day", "time_et"]).reset_index(drop=True)

# --- Required columns guard ---
for col in ["open", "high", "low", "close", "Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# =========================
# Join enriched 30m data (optional)
# =========================
enriched_cols = [
    "time","cum_vol","session_high","session_low",
    "hi_op","op_lo","pHi","pLo","hi_phi","lo_plo",
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
            enr_df = enr_df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
            df = df.merge(enr_df, on="time", how="left", suffixes=("", "_enriched"))
except Exception:
    pass  # suppress warnings per your request

# =========================
# Ensure session-level fields exist (fallbacks)
# =========================
if "cum_vol" not in df.columns or df["cum_vol"].isna().all():
    df["cum_vol"] = df.groupby("trade_day")["Volume"].cumsum()

if "session_high" not in df.columns or df["session_high"].isna().all():
    df["session_high"] = df.groupby("trade_day")["high"].cummax()
if "session_low" not in df.columns or df["session_low"].isna().all():
    df["session_low"]  = df.groupby("trade_day")["low"].cummin()

if "hi_op" not in df.columns or df["hi_op"].isna().all():
    df["hi_op"]  = df["high"] - df["open"]
if "op_lo" not in df.columns or df["op_lo"].isna().all():
    df["op_lo"]  = df["open"] - df["low"]

# pHi / pLo fallbacks from previous day
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
        df.loc[mask, "hi_pHi"] = df.loc[mask, "high"] - df["pHi"]

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

# =========================
# Aggregate per trade_day (18:00 ET-based)
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
    )
    bars = scope.groupby("trade_day").size().rename("bars_in_day")
    perbar = scope.groupby("trade_day").agg(
        avg_hi_op=("hi_op","mean"),
        avg_op_lo=("op_lo","mean"),
        avg_hi_pHi=("hi_pHi","mean"),
        avg_lo_pLo=("lo_pLo","mean"),
    )
    lasts = scope.groupby("trade_day").agg(
        session_high_at_cutoff=("session_high","last"),
        session_low_at_cutoff=("session_low","last"),
        cum_vol_at_cutoff=("cum_vol","last"),
    )
    out = (
        first_last.join(hilo).join(bars).join(perbar).join(lasts)
        .reset_index().sort_values("trade_day")
    )
    out["day_range"]  = out["day_high"] - out["day_low"]
    out["trade_date"] = out["trade_day"].dt.date
    return out

daily = agg_daily(df)

for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        avg_col = f"{col}_tw_avg"
        pct_col = f"{col}_pct_vs_tw"
        daily[avg_col] = trailing_mean(daily[col], TRAILING_WINDOW)
        daily[pct_col] = np.where(
            daily[avg_col].abs() > 0,
            (daily[col] - daily[avg_col]) / daily[avg_col],
            np.nan,
        )

# =========================
# Compute returns with the SAME 18:00 ET roll
# =========================
# 1) Last intraday price within the latest trade_day
latest_td = daily["trade_day"].max()
latest_rows = df.loc[df["trade_day"] == latest_td]
last_price = latest_rows["close"].iloc[-1] if not latest_rows.empty else np.nan

# 2) Prior daily close & period opens from daily_es (calendar dates)
day_ret = wtd_ret = mtd_ret = ytd_ret = np.nan
try:
    perf_resp = (
        sb.table("daily_es")
          .select("time, open, close")
          .order("time", desc=True)
          .limit(400)
          .execute()
    )
    perf_df = pd.DataFrame(perf_resp.data)
    if not perf_df.empty and {"time","open","close"}.issubset(perf_df.columns):
        perf_df["time"] = pd.to_datetime(perf_df["time"]).dt.date
        perf_df = perf_df.dropna(subset=["time","open","close"]).sort_values("time")

        # Calendar date that corresponds to our trade_day (after 6pm this is "next" calendar date)
        curr_cal_date = latest_td.date()

        # Previous calendar day close (yesterday's daily close)
        prev_mask = perf_df["time"] < curr_cal_date
        if prev_mask.any() and pd.notna(last_price):
            prev_close = perf_df.loc[prev_mask, "close"].iloc[-1]
            if pd.notna(prev_close) and prev_close != 0:
                day_ret = (last_price / prev_close) - 1.0

        # Period opens vs SAME last_price
        if pd.notna(last_price):
            # YTD
            y_mask = perf_df["time"].apply(lambda d: d.year == curr_cal_date.year)
            if y_mask.any():
                y_open = perf_df.loc[y_mask, "open"].iloc[0]
                ytd_ret = (last_price / y_open) - 1.0 if (pd.notna(y_open) and y_open != 0) else np.nan

            # MTD
            m_mask = perf_df["time"].apply(lambda d: d.year == curr_cal_date.year and d.month == curr_cal_date.month)
            if m_mask.any():
                m_open = perf_df.loc[m_mask, "open"].iloc[0]
                mtd_ret = (last_price / m_open) - 1.0 if (pd.notna(m_open) and m_open != 0) else np.nan

            # WTD (ISO week)
            curr_iso_year, curr_iso_week, _ = pd.Timestamp(curr_cal_date).isocalendar()
            iso = perf_df["time"].apply(lambda d: pd.Timestamp(d).isocalendar())
            w_mask = iso.apply(lambda tup: tup.year == curr_iso_year and tup.week == curr_iso_week)
            if w_mask.any():
                w_open = perf_df.loc[w_mask, "open"].iloc[0]
                wtd_ret = (last_price / w_open) - 1.0 if (pd.notna(w_open) and w_open != 0) else np.nan
except Exception:
    pass  # keep silent per your pattern

# =========================
# Render tiles (top of page)
# =========================
_perf_tile(c_day,   "Day performance",   day_ret)
_perf_tile(c_week,  "Week-to-date",      wtd_ret)
_perf_tile(c_month, "Month-to-date",     mtd_ret)
_perf_tile(c_year,  "Year-to-date",      ytd_ret)

# =========================
# KPI (latest day)
# =========================
row = daily.loc[daily["trade_day"] == latest_td].iloc[0]
hdr = f"{SYMBOL} — Full day"
st.subheader(hdr)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"Range vs {TRAILING_WINDOW}d",
    f"{row.get('day_range', np.nan):.2f}" if pd.notna(row.get('day_range')) else "—",
    None if pd.isna(row.get('day_range_pct_vs_tw')) else f"{row['day_range_pct_vs_tw']*100:+.1f}%"
)
k2.metric(
    f"Cum Vol vs {TRAILING_WINDOW}d",
    f"{row.get('cum_vol_at_cutoff', np.nan):,.0f}" if pd.notna(row.get('cum_vol_at_cutoff')) else "—",
    None
)
k3.metric("Session High (cutoff)",
          f"{row.get('session_high_at_cutoff', np.nan):.2f}" if pd.notna(row.get('session_high_at_cutoff')) else "—")
k4.metric("Session Low (cutoff)",
          f"{row.get('session_low_at_cutoff',  np.nan):.2f}" if pd.notna(row.get('session_low_at_cutoff'))  else "—")

# ======================================================
# (2) MA snapshot — ONE ROW, directly under tiles (via placeholder)
# ======================================================
def _render_ma_cards_horizontal(container, ma_rows):
    """Render MA cards in a single horizontal row."""
    if not ma_rows:
        container.info("No moving average columns found in es_30m.")
        return
    cols = container.columns(len(ma_rows))
    for i, ma in enumerate(ma_rows):
        label = ma["label"]
        val = ma["value"]
        dist = ma["dist"]
        if pd.isna(val) or pd.isna(dist):
            text = "N/A"
            bg = "#E5E7EB"
        else:
            text = f"{val:.2f} ({dist:+.1f} pts)"
            bg = "#BBF7D0" if dist > 0 else ("#FECACA" if dist < 0 else "#E5E7EB")

        html = f"""
            <div style="background-color:{bg};
                        border:1px solid #D1D5DB;
                        border-radius:10px;
                        padding:10px 12px;
                        margin-bottom:8px;">
                <div style="font-size:0.8rem; color:#4B5563; margin-bottom:2px;">
                    {label}
                </div>
                <div style="font-size:1rem; font-weight:600; color:#111827;">
                    {text}
                </div>
            </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)

# Build MA rows
ma_rows = []
try:
    latest_bars = df[df["trade_day"] == latest_td]
    if not latest_bars.empty:
        last_bar = latest_bars.iloc[-1]
        price = last_bar.get("close", np.nan)
        for col_name in ["5MA", "10MA", "20MA", "50MA", "200MA"]:
            if col_name in df.columns:
                val = last_bar.get(col_name, np.nan)
                dist = (price - val) if (pd.notna(price) and pd.notna(val)) else np.nan
                ma_rows.append({"label": col_name, "value": val, "dist": dist})
except Exception:
    pass

with ma_placeholder:
    st.markdown("### MA snapshot")
    _render_ma_cards_horizontal(st, ma_rows)

# ======================================================
# (3) Current range vs prior RTH range — EXACT working snippet, under MA snapshot
# ======================================================
def _render_range_position_bar(container, current_price, prev_low, prev_high,
                               sess_low, sess_high):
    # Validate
    vals = [v for v in [current_price, prev_low, prev_high, sess_low, sess_high] if pd.notna(v)]
    if len(vals) < 3:
        container.info("Not enough data for range visualization.")
        return
    scale_min = min(vals)
    scale_max = max(vals)
    if not (scale_max > scale_min):
        container.info("Invalid scale.")
        return

    # Side padding so pLo/pHi labels never clip
    LEFT_PAD_PCT  = 2.0
    RIGHT_PAD_PCT = 2.0
    INNER_MIN = LEFT_PAD_PCT
    INNER_MAX = 100.0 - RIGHT_PAD_PCT
    inner_span = INNER_MAX - INNER_MIN

    def to_pct(x):
        return INNER_MIN + inner_span * (x - scale_min) / (scale_max - scale_min)

    # Positions
    pLo = to_pct(prev_low)
    pHi = to_pct(prev_high)
    prior_left  = pLo
    prior_width = max(0.5, pHi - pLo)

    sLo = to_pct(sess_low)
    sHi = to_pct(sess_high)
    sess_left  = min(sLo, sHi)
    sess_width = max(0.5, abs(sHi - sLo))

    last_pct = to_pct(current_price)

    html = f"""
    <div style="position:relative; width:100%; height:82px;
                margin-top:6px; margin-bottom:6px;">

      <!-- Base track -->
      <div style="position:absolute; left:0; right:0; top:36px; height:12px;
                  background-color:#EEF2F7; border-radius:999px;"></div>

      <!-- Prior RTH hollow rectangle -->
      <div style="position:absolute; top:30px; height:24px; left:{prior_left:.2f}%;
                  width:{prior_width:.2f}%; border-radius:12px; border:2px solid #9CA3AF;
                  background-color:transparent;"></div>

      <!-- Current session filled pill (darker blue) -->
      <div style="position:absolute; top:36px; height:12px; left:{sess_left:.2f}%;
                  width:{sess_width:.2f}%; border-radius:999px; background-color:#1D4ED8;"></div>

      <!-- Last price marker -->
      <div title="Last" style="position:absolute; top:26px; bottom:26px; left:{last_pct:.2f}%;">
        <div style="width:2px; height:100%; background-color:#111827;"></div>
      </div>

      <!-- Prior labels (top row, inside, with padding-aware positions) -->
      <div style="position:absolute; top:6px; left:{pLo:.2f}%;
                  transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
        {prev_low:.2f} <span style="opacity:0.8;">pLo</span>
      </div>
      <div style="position:absolute; top:6px; left:{pHi:.2f}%;
                  transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
        {prev_high:.2f} <span style="opacity:0.8;">pHi</span>
      </div>

      <!-- Session labels (bottom row, inside) -->
      <div style="position:absolute; bottom:6px; left:{sLo:.2f}%;
                  transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
        {sess_low:.2f} <span style="opacity:0.8;">Lo</span>
      </div>
      <div style="position:absolute; bottom:6px; left:{sHi:.2f}%;
                  transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
        {sess_high:.2f} <span style="opacity:0.8;">Hi</span>
      </div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)

# Pull inputs for the bar using the SAME 6pm-roll day as tiles
current_price       = last_price
session_low_cutoff  = row.get("session_low_at_cutoff", np.nan)
session_high_cutoff = row.get("session_high_at_cutoff", np.nan)

prev_rth_low = prev_rth_high = np.nan
try:
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
        latest_date = row["trade_date"]
        prev_mask = summ_df["trade_date"] < latest_date
        prev_row = (summ_df.loc[prev_mask].sort_values("trade_date").iloc[-1]
                    if prev_mask.any() else summ_df.sort_values("trade_date").iloc[-1])
        prev_rth_low = prev_row.get("RTH Lo", np.nan)
        prev_rth_high = prev_row.get("RTH Hi", np.nan)
except Exception:
    pass

with range_placeholder:
    st.markdown("### Current range vs prior RTH range")
    _render_range_position_bar(
        st, current_price, prev_rth_low, prev_rth_high, session_low_cutoff, session_high_cutoff
    )

# (table section previously removed; file ends here)
