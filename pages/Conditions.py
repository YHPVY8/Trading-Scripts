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
    if sub_df is None or sub_df.empty:
        return np.nan
    first_open = sub_df["period_open"].iloc[0]
    last_close = sub_df["period_close"].iloc[-1]
    if pd.isna(first_open) or pd.isna(last_close) or first_open == 0:
        return np.nan
    return (last_close / first_open) - 1.0

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
        if not perf_df.empty:
            daily_agg = (
                perf_df.groupby("time")
                       .agg(period_open=("open","first"),
                            period_close=("close","last"))
                       .reset_index()
                       .sort_values("time")
            )
            if len(daily_agg) >= 1:
                current_date = daily_agg["time"].iloc[-1]

                # Day: last close vs prev close
                if len(daily_agg) >= 2:
                    prev_close = daily_agg["period_close"].iloc[-2]
                    last_close = daily_agg["period_close"].iloc[-1]
                    day_ret = (last_close / prev_close) - 1.0 if prev_close else np.nan

                dt_series = pd.to_datetime(daily_agg["time"])

                # YTD: latest close vs first OPEN of calendar year
                y_mask = dt_series.dt.year == current_date.year
                y_subset = daily_agg.loc[y_mask].copy()
                ytd_ret = _period_return_open_to_close(y_subset)

                # MTD
                m_mask = (dt_series.dt.year == current_date.year) & (dt_series.dt.month == current_date.month)
                m_subset = daily_agg.loc[m_mask].copy()
                mtd_ret = _period_return_open_to_close(m_subset)

                # WTD
                iso_all = dt_series.dt.isocalendar()
                curr_iso_year, curr_iso_week, _ = pd.Timestamp(current_date).isocalendar()
                w_mask = (iso_all["year"] == curr_iso_year) & (iso_all["week"] == curr_iso_week)
                w_subset = daily_agg.loc[w_mask].copy()
                wtd_ret = _period_return_open_to_close(w_subset)
except Exception:
    pass  # keep tiles blank instead of warning

# =========================
# Title + top tiles
# =========================
st.title("Market Conditions")

c_day, c_week, c_month, c_year = st.columns(4)

def _perf_tile(container, label: str, ret: float):
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
            bg = "#E5E7EB"
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
            <div style="color:#374151;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">
                {label}
            </div>
            <div style="color:#111827;font-size:1.6rem;font-weight:600;">
                {display}
            </div>
        </div>
    """
    container.markdown(html, unsafe_allow_html=True)

_perf_tile(c_day,   "Day performance",   day_ret)
_perf_tile(c_week,  "Week-to-date",      wtd_ret)
_perf_tile(c_month, "Month-to-date",     mtd_ret)
_perf_tile(c_year,  "Year-to-date",      ytd_ret)

# ======================================================
# Fetch base intraday table (do BEFORE controls so visuals appear right under tiles)
# ======================================================
TABLE = "es_30m"
rows_per_day_guess = 48
trade_days_to_keep_default = 180
rows_to_load = int(max(2000, trade_days_to_keep_default * rows_per_day_guess * 1.5))

response = (
    sb.table(TABLE)
      .select("*")
      .order("time", desc=True)
      .limit(rows_to_load)
      .execute()
)
df = pd.DataFrame(response.data)
if df.empty:
    st.stop()

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df["time_et"] = df["time"].dt.tz_convert("US/Eastern")
et = df["time_et"]
df["trade_day"] = et.dt.floor("D") + pd.to_timedelta((et.dt.hour >= 18).astype("int64"), unit="D")
df["trade_date"] = df["trade_day"].dt.date

# Session flags (used later)
t = df["time_et"].dt.time
df["ON"]  = ((t >= dt.time(18,0)) | (t < dt.time(9,30)))
df["IB"]  = ((t >= dt.time(9,30)) & (t < dt.time(10,30)))
df["RTH"] = ((t >= dt.time(9,30)) & (t <= dt.time(16,0)))

# Enriched join (quietly optional)
enriched_cols = ["time","cum_vol","session_high","session_low","hi_op","op_lo","pHi","pLo","hi_phi","lo_plo"]
try:
    min_time = df["time"].min(); max_time = df["time"].max()
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
            enr_df = enr_df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
            df = df.merge(enr_df, on="time", how="left", suffixes=("", "_enriched"))
except Exception:
    pass

# Fallbacks
if "cum_vol" not in df.columns or df["cum_vol"].isna().all():
    df["cum_vol"] = df.groupby("trade_day")["Volume"].cumsum()
if "session_high" not in df.columns or df["session_high"].isna().all():
    df["session_high"] = df.groupby("trade_day")["high"].cummax()
if "session_low" not in df.columns or df["session_low"].isna().all():
    df["session_low"] = df.groupby("trade_day")["low"].cummin()
if "hi_op" not in df.columns or df["hi_op"].isna().all():
    df["hi_op"] = df["high"] - df["open"]
if "op_lo" not in df.columns or df["op_lo"].isna().all():
    df["op_lo"] = df["open"] - df["low"]

# Latest day row for KPI/visuals
per_day_last_bar = df.groupby("trade_day")["time_et"].max().sort_values(ascending=False)
recent_trade_days_desc = per_day_last_bar.index[:trade_days_to_keep_default]
df = df[df["trade_day"].isin(set(recent_trade_days_desc))].copy()
df = df.sort_values(["trade_day","time_et"]).reset_index(drop=True)

latest_td = df["trade_day"].max()
latest_day_df = df[df["trade_day"] == latest_td]
row_close = np.nan
session_low_cutoff = np.nan
session_high_cutoff = np.nan
if not latest_day_df.empty:
    last_bar = latest_day_df.iloc[-1]
    row_close = last_bar.get("close", np.nan)
    session_low_cutoff = latest_day_df["session_low"].iloc[-1]
    session_high_cutoff = latest_day_df["session_high"].iloc[-1]

# Pull prior RTH Hi/Lo (quoted column names)
prev_rth_low = prev_rth_high = np.nan
try:
    # Use latest trade_date from latest_td - 1 (prior trading day)
    latest_trade_date = latest_td.date()
    summ_resp = (
        sb.table("es_trade_day_summary")
          .select('trade_date,"RTH Hi","RTH Lo"')
          .lt("trade_date", latest_trade_date.isoformat())
          .order("trade_date", desc=True)
          .limit(1)
          .execute()
    )
    summ_df = pd.DataFrame(summ_resp.data)
    if not summ_df.empty:
        prev_rth_low = float(summ_df.iloc[0]["RTH Lo"]) if pd.notna(summ_df.iloc[0]["RTH Lo"]) else np.nan
        prev_rth_high = float(summ_df.iloc[0]["RTH Hi"]) if pd.notna(summ_df.iloc[0]["RTH Hi"]) else np.nan
except Exception:
    pass

# ======================================================
# VISUALS DIRECTLY UNDER THE TILES
# ======================================================

# ---- MA snapshot (cards) ----
st.markdown("### MA snapshot")
def _render_ma_cards_hstack(container, ma_rows):
    # simple 1-row layout: each card in its own column
    if not ma_rows:
        container.info("No moving average columns found in es_30m.")
        return
    cols = container.columns(len(ma_rows))
    for i, ma in enumerate(ma_rows):
        label, val, dist = ma["label"], ma["value"], ma["dist"]
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
                <div style="font-size:0.8rem; color:#4B5563; margin-bottom:4px;">
                    {label}
                </div>
                <div style="font-size:1rem; font-weight:600; color:#111827;">
                    {text}
                </div>
            </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)

# Build MA rows for latest bar
ma_rows = []
try:
    if not latest_day_df.empty:
        last_bar = latest_day_df.iloc[-1]
        price = last_bar.get("close", np.nan)
        ma_candidates = ["5MA", "10MA", "20MA", "50MA", "200MA"]
        for colname in [c for c in ma_candidates if c in df.columns]:
            val = last_bar.get(colname, np.nan)
            dist = (price - val) if (pd.notna(price) and pd.notna(val)) else np.nan
            ma_rows.append({"label": colname, "value": val, "dist": dist})
except Exception:
    pass

_render_ma_cards_hstack(st, ma_rows)

# ---- Current range vs prior RTH range (full-width, darker blue session bar) ----
st.markdown("### Current range vs prior RTH range")

def _render_range_position_bar(container, current_price, p_low, p_high, s_low, s_high):
    vals = [v for v in [current_price, p_low, p_high, s_low, s_high] if pd.notna(v)]
    if len(vals) < 3:
        container.info("Not enough data for range visualization.")
        return
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return
    span = vmax - vmin
    def pct(x): return 100.0 * (x - vmin) / span

    prior_left = pct(p_low); prior_right = pct(p_high); prior_w = max(0.5, prior_right - prior_left)
    sess_left = pct(s_low);  sess_right  = pct(s_high);  sess_w  = max(0.5, sess_right - sess_left)
    last_pct  = pct(current_price)

    # A single full-width container; labels inside; darker session blue #1D4ED8
    html = f"""
    <div style="position:relative; width:100%; height:90px; margin-top:6px; margin-bottom:2px;">
        <!-- Base track -->
        <div style="position:absolute; left:0; right:0; top:36px; height:12px;
                    background-color:#EEF2F7; border-radius:999px;"></div>

        <!-- Prior RTH hollow rectangle -->
        <div style="position:absolute; top:30px; height:24px; left:{prior_left:.2f}%;
                    width:{prior_w:.2f}%; border-radius:12px; border:2px solid #9CA3AF;
                    background-color:transparent;"></div>

        <!-- Current session filled pill (darker blue) -->
        <div style="position:absolute; top:36px; height:12px; left:{sess_left:.2f}%;
                    width:{sess_w:.2f}%; border-radius:999px; background-color:#1D4ED8;"></div>

        <!-- Last price marker -->
        <div title="Last" style="position:absolute; top:26px; bottom:26px; left:{last_pct:.2f}%;">
          <div style="width:2px; height:100%; background-color:#111827;"></div>
        </div>

        <!-- Prior labels (top row, inside) -->
        <div style="position:absolute; top:6px; left:{prior_left:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {p_low:.2f} <span style="opacity:0.8;">pLo</span>
        </div>
        <div style="position:absolute; top:6px; left:{prior_right:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {p_high:.2f} <span style="opacity:0.8;">pHi</span>
        </div>

        <!-- Session labels (bottom row, inside) -->
        <div style="position:absolute; bottom:6px; left:{sess_left:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {s_low:.2f} <span style="opacity:0.8;">Lo</span>
        </div>
        <div style="position:absolute; bottom:6px; left:{sess_right:.2f}%;
                    transform:translateX(-50%); font-size:0.75rem; color:#374151; white-space:nowrap;">
          {s_high:.2f} <span style="opacity:0.8;">Hi</span>
        </div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)

_render_range_position_bar(
    st,
    row_close,
    prev_rth_low,
    prev_rth_high,
    session_low_cutoff,
    session_high_cutoff,
)

# =========================
# Controls (moved BELOW visuals)
# =========================
st.markdown("---")
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

# =========================
# Aggregate table (unchanged logic, but we'll hide verbose captions and remove columns you requested)
# =========================
def trailing_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean().shift(1)

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
        daily[avg_col] = trailing_mean(daily[col], 10)  # default shown; UI slider still controls KPIs above
        daily[pct_col] = np.where(daily[avg_col].abs()>0, (daily[col]-daily[avg_col])/daily[avg_col], np.nan)

# Final table selection (remove the fields you asked to drop)
st.markdown("### Conditions vs Trailing (last N trade days)")
max_slider = int(min(120, len(daily)))
show_days = st.slider("Show last N trade days", 10, max_slider, min(30, max_slider), 5)

cols_keep = [
    "trade_date",
    "day_open","day_high","day_low","day_close","day_range",
    "session_high_at_cutoff","session_low_at_cutoff",
    "cum_vol_at_cutoff","day_volume","day_volume_tw_avg",
    "avg_hi_op","avg_hi_op_tw_avg","avg_hi_op_pct_vs_tw",
    "avg_op_lo","avg_op_lo_tw_avg","avg_op_lo_pct_vs_tw",
    # removed: avg_hi_pHi, avg_lo_pLo, their 10d avgs, ON/IB/RTH % pMid hits, Bars
]
existing = [c for c in cols_keep if c in daily.columns]
tbl = daily[existing].tail(int(show_days)).copy()

labels = {
    "trade_date":"Trade Date",
    "day_open":"Open","day_high":"High","day_low":"Low","day_close":"Close","day_range":"Range",
    "session_high_at_cutoff":"Session High (cutoff)","session_low_at_cutoff":"Session Low (cutoff)",
    "cum_vol_at_cutoff":"Cum Vol (cutoff)","day_volume":"Vol (day)","day_volume_tw_avg":"Vol 10d",
    "avg_hi_op":"Avg(Hi-Op)","avg_hi_op_tw_avg":"Avg(Hi-Op) 10d","avg_hi_op_pct_vs_tw":"Avg(Hi-Op) vs 10d",
    "avg_op_lo":"Avg(Op-Lo)","avg_op_lo_tw_avg":"Avg(Op-Lo) 10d","avg_op_lo_pct_vs_tw":"Avg(Op-Lo) vs 10d",
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

with st.expander("Data health (debug)"):
    st.write({
        "latest_trade_day": str(latest_td),
        "rows_total_after_filters": int(len(df)),
    })
