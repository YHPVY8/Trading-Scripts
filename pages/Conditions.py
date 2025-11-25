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
def _period_return_open_to_close(sub_df: pd.DataFrame) -> float:
    if sub_df is None or sub_df.empty:
        return np.nan
    first_open = sub_df["period_open"].iloc[0]
    last_close = sub_df["period_close"].iloc[-1]
    if pd.isna(first_open) or pd.isna(last_close) or first_open == 0:
        return np.nan
    return (last_close / first_open) - 1.0

def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean().shift(1)

# =========================
# Daily performance snapshot from daily_es
# =========================
day_ret = wtd_ret = mtd_ret = ytd_ret = np.nan
daily_agg = None

try:
    perf_resp = (
        sb.table("daily_es")
          .select("time, open, close")
          .order("time", desc=True)
          .limit(260)
          .execute()
    )
    perf_df = pd.DataFrame(perf_resp.data)

    if not perf_df.empty and {"time","open","close"}.issubset(perf_df.columns):
        perf_df["time"] = pd.to_datetime(perf_df["time"]).dt.date
        perf_df = perf_df.dropna(subset=["time","open","close"])

        daily_agg = (
            perf_df.groupby("time")
                   .agg(period_open=("open","first"), period_close=("close","last"))
                   .reset_index()
                   .sort_values("time")
        )

        if len(daily_agg) >= 1:
            current_date = daily_agg["time"].iloc[-1]

            if len(daily_agg) >= 2:
                last_two = daily_agg.tail(2)
                prev_close = last_two["period_close"].iloc[0]
                last_close = last_two["period_close"].iloc[1]
                day_ret = (last_close / prev_close) - 1.0 if prev_close else np.nan

            dt_series = pd.to_datetime(daily_agg["time"])
            curr_year = current_date.year
            curr_month = current_date.month

            y_mask = dt_series.dt.year == curr_year
            m_mask = (dt_series.dt.year == curr_year) & (dt_series.dt.month == curr_month)
            iso_all = dt_series.dt.isocalendar()
            curr_iso_year, curr_iso_week, _ = pd.Timestamp(current_date).isocalendar()
            w_mask = (iso_all["year"] == curr_iso_year) & (iso_all["week"] == curr_iso_week)

            ytd_ret = _period_return_open_to_close(daily_agg.loc[y_mask])
            mtd_ret = _period_return_open_to_close(daily_agg.loc[m_mask])
            wtd_ret = _period_return_open_to_close(daily_agg.loc[w_mask])
except Exception as e:
    st.warning(f"Could not load daily_es for performance tiles: {e}")

# =========================
# Title + top tiles
# =========================
st.title("Market Conditions")

c_day, c_week, c_month, c_year = st.columns(4)

def _perf_tile(container, label: str, ret: float):
    if pd.isna(ret):
        display = "—"
        bg = "#E5E7EB"
    else:
        display = f"{ret*100:+.1f}%"
        if ret > 0:  bg = "#86EFAC"
        elif ret < 0: bg = "#FCA5A5"
        else:         bg = "#E5E7EB"

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

_perf_tile(c_day,   "Day performance",   day_ret)
_perf_tile(c_week,  "Week-to-date",      wtd_ret)
_perf_tile(c_month, "Month-to-date",     mtd_ret)
_perf_tile(c_year,  "Year-to-date",      ytd_ret)

# =========================
# Fetch 30m data (newest-first) and prep day
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

TABLE = "es_30m"
rows_per_day_guess = 48
rows_to_load = int(max(2000, trade_days_to_keep * rows_per_day_guess * 1.5))

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

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")

latest_bar_time = df["time_et"].max()
latest_trade_day_expected = (
    latest_bar_time.floor("D") + pd.to_timedelta(int(latest_bar_time.hour >= 18), unit="D")
)

et = df["time_et"]
midnight = et.dt.floor("D")
roll = (et.dt.hour >= 18).astype("int64")
df["trade_day"] = midnight + pd.to_timedelta(roll, unit="D")
df["trade_date"] = df["trade_day"].dt.date

t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30)))
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30)))
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0)))

per_day_last_bar = (
    df.groupby("trade_day")["time_et"].max().sort_values(ascending=False)
)
recent_trade_days_desc = per_day_last_bar.index[:int(trade_days_to_keep)]
df = df[df["trade_day"].isin(set(recent_trade_days_desc))].copy()
df = df.sort_values(["trade_day","time_et"]).reset_index(drop=True)

latest_td = df["trade_day"].max()
latest_day_df = df[df["trade_day"] == latest_td]
if not latest_day_df.empty:
    earliest_bar_et = latest_day_df["time_et"].min()
    latest_start_et = latest_td - pd.Timedelta(hours=6)
    latest_end_et   = latest_td + pd.Timedelta(hours=24)
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
            tt = missing_df["time_et"].dt.time
            missing_df["ON"]  = ((tt >= dt.time(18,0)) | (tt < dt.time(9,30)))
            missing_df["IB"]  = ((tt >= dt.time(9,30)) & (tt < dt.time(10,30)))
            missing_df["RTH"] = ((tt >= dt.time(9,30)) & (tt <= dt.time(16,0)))
            missing_df = missing_df[~missing_df["time"].isin(df["time"])]
            if not missing_df.empty:
                df = pd.concat([df, missing_df], ignore_index=True)
                df = df.sort_values(["trade_day","time_et"]).reset_index(drop=True)

# As-of mode (optional)
asof_time = None
if mode == "As-of time snapshot":
    asof_time = st.time_input("As-of time (US/Eastern)", value=dt.time(3, 0))
    cut_seconds = asof_time.hour*3600 + asof_time.minute*60 + asof_time.second
    cutoff_ts = df["trade_day"] + pd.to_timedelta(cut_seconds, unit="s")
    prev_start = (df["trade_day"] - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)
    mask_asof = (df["time_et"] >= prev_start) & (df["time_et"] <= cutoff_ts)
    df = df[mask_asof].copy()

# Guards
for col in ["open","high","low","close","Volume"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in {TABLE}.")
        st.stop()

# Join enriched (optional)
enriched_cols = ["time","cum_vol","session_high","session_low","hi_op","op_lo","pHi","pLo","hi_phi","lo_plo"]
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
except Exception as e:
    pass  # silent

# Ensure derived cols
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
        on="trade_day", how="left"
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

# =========================
# Aggregate per trade_day
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

    out = (first_last.join(hilo).join(bars).join(perbar).join(lasts)).reset_index().sort_values("trade_day")
    out["day_range"]  = out["day_high"] - out["day_low"]
    out["trade_date"] = out["trade_day"].dt.date
    return out

daily = agg_daily(df)

for col in ["day_range","day_volume","avg_hi_op","avg_op_lo","avg_hi_pHi","avg_lo_pLo"]:
    if col in daily.columns:
        avg_col = f"{col}_tw_avg"
        pct_col = f"{col}_pct_vs_tw"
        daily[avg_col] = trailing_mean(daily[col], trailing_window)
        daily[pct_col] = np.where(
            daily[avg_col].abs() > 0, (daily[col] - daily[avg_col]) / daily[avg_col], np.nan
        )

if daily.empty:
    st.warning("No trade days after processing—adjust controls.")
    st.stop()

latest_td = daily["trade_day"].max()
row = daily.loc[daily["trade_day"] == latest_td].iloc[0]

# =========================
# KPI (latest day)
# =========================
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

# =========================
# MA snapshot (ONE horizontal row)
# =========================
def _render_ma_snapshot_row(container, ma_rows):
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
                        text-align:center;">
                <div style="font-size:0.8rem; color:#4B5563; margin-bottom:2px;">
                    {label}
                </div>
                <div style="font-size:1rem; font-weight:600; color:#111827;">
                    {text}
                </div>
            </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)

# Build MA rows from the very latest bar of latest trade day
ma_rows = []
try:
    latest_bars = df[df["trade_day"] == latest_td]
    if not latest_bars.empty:
        last_bar = latest_bars.iloc[-1]
        price = last_bar.get("close", np.nan)
        ma_candidates = ["5MA","10MA","20MA","50MA","200MA"]
        ma_cols = [c for c in ma_candidates if c in df.columns]
        for col_name in ma_cols:
            val = last_bar.get(col_name, np.nan)
            dist = price - val if (pd.notna(price) and pd.notna(val)) else np.nan
            ma_rows.append({"label": col_name, "value": val, "dist": dist})
except Exception as e:
    pass

st.markdown("### MA snapshot")
_render_ma_snapshot_row(st, ma_rows)

# =========================
# Current range vs prior RTH range (EXACT HTML snippet you liked)
# =========================
def _render_range_position_bar_exact(container, last_px, p_rth_lo, p_rth_hi, sess_lo, sess_hi):
    vals = [v for v in [last_px, p_rth_lo, p_rth_hi, sess_lo, sess_hi] if pd.notna(v)]
    if len(vals) < 3:
        container.info("Not enough data for range visualization.")
        return
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        container.info("Invalid range for visualization.")
        return
    span = vmax - vmin
    def pct(x): return 100.0*(x - vmin)/span

    prior_left = pct(p_rth_lo); prior_right = pct(p_rth_hi); prior_width = max(0.01, prior_right - prior_left)
    sess_left  = pct(sess_lo);   sess_right  = pct(sess_hi);   sess_width  = max(0.01, sess_right - sess_left)
    last_pct   = pct(last_px)

    html = f"""
    <div style="margin-top:8px;">
      <div style="position:relative; width:100%; height:80px; margin-top:4px;">

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

        <!-- Prior labels (top row, inside) -->
        <div style="position:absolute; top:6px; left:{prior_left:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {p_rth_lo:.2f} <span style="opacity:0.8;">pLo</span>
        </div>
        <div style="position:absolute; top:6px; left:{prior_right:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {p_rth_hi:.2f} <span style="opacity:0.8;">pHi</span>
        </div>

        <!-- Session labels (bottom row, inside) -->
        <div style="position:absolute; bottom:6px; left:{sess_left:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {sess_lo:.2f} <span style="opacity:0.8;">Lo</span>
        </div>
        <div style="position:absolute; bottom:6px; left:{sess_right:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {sess_hi:.2f} <span style="opacity:0.8;">Hi</span>
        </div>
      </div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)

# Gather inputs for the bar
current_close = row.get("day_close", np.nan)
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
        if prev_mask.any():
            prev_row = summ_df.loc[prev_mask].sort_values("trade_date").iloc[-1]
        else:
            prev_row = summ_df.sort_values("trade_date").iloc[-1]
        prev_rth_low  = prev_row.get("RTH Lo", np.nan)
        prev_rth_high = prev_row.get("RTH Hi", np.nan)
except Exception:
    pass

st.markdown("### Current range vs prior RTH range")
_render_range_position_bar_exact(
    st,
    current_close,
    prev_rth_low,
    prev_rth_high,
    session_low_cutoff,
    session_high_cutoff,
)

# =========================
# Conditions table (trim columns as requested)
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
    # REMOVED: avg_hi_pHi, avg_hi_pHi_tw_avg, avg_lo_pLo, avg_lo_pLo_tw_avg,
    # ON_pMid_hit_pct, ON_pMid_hit_pct_tw_avg,
    # IB_pMid_hit_pct, IB_pMid_hit_pct_tw_avg,
    # RTH_pMid_hit_pct, RTH_pMid_hit_pct_tw_avg,
    # bars_in_day
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
}
tbl = tbl.rename(columns=labels)

fmt = {}
for name in tbl.columns:
    if name == "Trade Date":
        continue
    if "Vol" in name or "Vol (" in name:
        fmt[name] = "{:,.0f}"
    elif "vs" in name:
        fmt[name] = "{:,.2f}"
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

vs_cols = [c for c in tbl.columns if "vs" in c]
styled = (
    tbl.style
       .format(fmt)
       .applymap(color_pos_neg, subset=vs_cols)
       .set_properties(subset=["Trade Date"], **{"font-weight":"600"})
)
st.dataframe(styled, use_container_width=True)
