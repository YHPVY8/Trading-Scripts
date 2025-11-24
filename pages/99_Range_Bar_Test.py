#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from supabase import create_client

# ---------- Page config ----------
st.set_page_config(page_title="Range Bar Test Page", layout="wide")
st.title("Current range vs prior range")
st.caption(
    "Test intraday range visualizations (prior RTH range vs current session range + last price). "
    "If live data is unavailable, switch to Manual inputs."
)

# ---------- Supabase ----------
@st.cache_resource
def _get_sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

sb = _get_sb()

# ---------- Time helpers ----------
def _to_utc(ts_series: pd.Series) -> pd.Series:
    # Ensure tz-aware UTC
    s = pd.to_datetime(ts_series, errors="coerce", utc=True)
    return s

def _et(series_utc: pd.Series) -> pd.Series:
    # Convert tz-aware UTC -> US/Eastern
    return series_utc.dt.tz_convert("US/Eastern")

def _et_trade_day(ts_utc_series: pd.Series) -> pd.Series:
    # 18:00 ET roll -> trade_day is ET-midnight with roll if hour>=18
    ts_utc = _to_utc(ts_utc_series)
    ts_et = _et(ts_utc)
    midnight = ts_et.dt.floor("D")
    roll = (ts_et.dt.hour >= 18).astype("int64")
    return midnight + pd.to_timedelta(roll, unit="D")  # tz-aware ET

# ---------- Live data fetch ----------
def _fetch_current_session_from_es_30m():
    """
    Returns dict:
      {
        'prior_low': float,
        'prior_high': float,
        'session_low': float,
        'session_high': float,
        'last_price': float,
        'latest_trade_date': date
      }
    """
    # Pull a safe window of recent rows
    resp = (
        sb.table("es_30m")
          .select("time,open,high,low,close")
          .order("time", desc=True)
          .limit(1200)   # ~25 trade days of 30m bars
          .execute()
    )
    df = pd.DataFrame(resp.data)
    if df.empty:
        raise RuntimeError("No rows from es_30m")

    # Parse time -> tz-aware UTC
    df["time"] = _to_utc(df["time"])
    df = df.dropna(subset=["time"])

    # Compute trade_day in ET
    df["trade_day"] = _et_trade_day(df["time"])
    df["time_et"] = _et(df["time"])
    df["date"] = df["trade_day"].dt.date

    # Latest trade_day
    latest_td = df["trade_day"].max()
    latest_date = latest_td.date()

    # Current session slice
    cur = df[df["trade_day"] == latest_td].sort_values("time")
    if cur.empty:
        raise RuntimeError("No rows for latest trade_day in es_30m")

    last_price = float(cur["close"].iloc[-1])
    session_low = float(cur["low"].min())
    session_high = float(cur["high"].max())

    # Get prior RTH range from summary table; columns have spaces and must be quoted
    prior_low = np.nan
    prior_high = np.nan
    try:
        summ = (
            sb.table("es_trade_day_summary")
              .select('trade_date,"RTH Hi","RTH Lo"')
              .lte("trade_date", latest_date.isoformat())
              .order("trade_date", desc=True)
              .limit(3)
              .execute()
        )
        sdf = pd.DataFrame(summ.data)
        if not sdf.empty:
            sdf["trade_date"] = pd.to_datetime(sdf["trade_date"]).dt.date
            # Prior = most recent date strictly before latest_date
            prev = sdf[sdf["trade_date"] < latest_date]
            if not prev.empty:
                prev = prev.sort_values("trade_date").iloc[-1]
            else:
                # Fallback: use the row at latest_date if available (better than empty)
                prev = sdf.sort_values("trade_date").iloc[-1]
            prior_high = float(prev['RTH Hi']) if pd.notna(prev['RTH Hi']) else np.nan
            prior_low  = float(prev['RTH Lo']) if pd.notna(prev['RTH Lo']) else np.nan
    except Exception as e:
        st.warning(f"Could not read es_trade_day_summary; will fallback if needed. Error: {e}")

    # Fallback if summary missing: compute previous day RTH from es_30m directly (9:30–16:00 ET)
    if not pd.notna(prior_low) or not pd.notna(prior_high):
        prior_td = sorted(df["trade_day"].unique())
        if len(prior_td) >= 2:
            prev_td = prior_td[-2]
            prev_slice = df[df["trade_day"] == prev_td].copy()
            t = prev_slice["time_et"].dt.time
            rth = prev_slice[(t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("16:00").time())]
            if not rth.empty:
                prior_low = float(rth["low"].min())
                prior_high = float(rth["high"].max())

    return {
        "prior_low": prior_low,
        "prior_high": prior_high,
        "session_low": session_low,
        "session_high": session_high,
        "last_price": last_price,
        "latest_trade_date": latest_date,
    }

# ---------- Inputs / Mode ----------
st.markdown("### Data Source")
use_live = st.toggle("Use LIVE latest session from Supabase", value=True)

if use_live:
    try:
        live = _fetch_current_session_from_es_30m()
        prior_low = float(live["prior_low"])
        prior_high = float(live["prior_high"])
        session_low = float(live["session_low"])
        session_high = float(live["session_high"])
        last_price = float(live["last_price"])
        st.success(f"Live pulled — latest trade date: {live['latest_trade_date']}")
    except Exception as e:
        st.warning(f"Live fetch failed ({e}). Switch to Manual to test.")
        use_live = False

if not use_live:
    c1, c2, c3 = st.columns(3)
    with c1:
        prior_low = st.number_input("Prior RTH Low", value=6539.00, step=0.25)
        prior_high = st.number_input("Prior RTH High", value=6677.50, step=0.25)
    with c2:
        session_low = st.number_input("Current Session Low", value=6625.00, step=0.25)
        session_high = st.number_input("Current Session High", value=6669.25, step=0.25)
    with c3:
        last_price = st.number_input("Last Price", value=6636.75, step=0.25)

# Guards
if prior_high <= prior_low:
    st.error("Prior RTH High must be greater than Prior RTH Low.")
    st.stop()
if session_high <= session_low:
    st.error("Current Session High must be greater than Current Session Low.")
    st.stop()

# ---------- Scaling helpers ----------
# We pad left/right so labels never clip
pad_pts = max(1.0, (prior_high - prior_low) * 0.02)  # 2% or 1pt min
all_vals = [prior_low, prior_high, session_low, session_high, last_price]
min_all = float(np.nanmin(all_vals)) - pad_pts
max_all = float(np.nanmax(all_vals)) + pad_pts
if max_all <= min_all:
    max_all = min_all + 1.0
span_all = max_all - min_all

def _pct(p: float) -> float:
    return (p - min_all) / span_all * 100.0

# Precompute positions
prior_left = _pct(prior_low)
prior_width = _pct(prior_high) - prior_left
sess_left = _pct(session_low)
sess_width = _pct(session_high) - sess_left
last_pos = _pct(last_price)
sess_lo_pos = _pct(session_low)
sess_hi_pos = _pct(session_high)
prior_lo_pos = _pct(prior_low)
prior_hi_pos = _pct(prior_high)

# =====================================================
# Example 1 – HTML Overlay Range Bar (labels inside)
# =====================================================
st.markdown("### Example 1 — Overlay Range Bar (HTML)")

# Single HTML block; avoid leaking text by keeping all CSS inline
html_bar = f"""
<div style="position:relative; width:100%; height:90px; margin:8px 0;
            border-radius:10px; background-color:#F8FAFC; border:1px solid #CBD5E1; overflow:visible;">

  <!-- Hollow PRIOR RTH range -->
  <div style="position:absolute; top:30%; height:40%;
              left:{prior_left:.2f}%; width:{prior_width:.2f}%;
              border:2px solid #9CA3AF; border-radius:8px; background-color:rgba(156,163,175,0.08);">
  </div>

  <!-- Filled CURRENT SESSION range -->
  <div style="position:absolute; top:40%; height:20%;
              left:{sess_left:.2f}%; width:{sess_width:.2f}%;
              border-radius:6px; background-color:rgba(37,99,235,0.45);">
  </div>

  <!-- LAST price marker -->
  <div title="Last" style="position:absolute; top:20%; bottom:20%;
                           left:{last_pos:.2f}%; width:2px; background-color:#111827;"></div>

  <!-- Session Low / High markers -->
  <div title="Session Low" style="position:absolute; top:30%; height:40%;
                                  left:{sess_lo_pos:.2f}%; width:2px; background-color:#DC2626;"></div>
  <div title="Session High" style="position:absolute; top:30%; height:40%;
                                   left:{sess_hi_pos:.2f}%; width:2px; background-color:#16A34A;"></div>

  <!-- Labels INSIDE the viz (below numbers, compact) -->
  <div style="position:absolute; top:68%; left:{prior_lo_pos:.2f}%; transform:translateX(-50%); font-size:11px; color:#374151;">
    {prior_low:.2f}<div style="font-size:10px; color:#6B7280;">pLo</div>
  </div>
  <div style="position:absolute; top:68%; left:{prior_hi_pos:.2f}%; transform:translateX(-50%); font-size:11px; color:#374151;">
    {prior_high:.2f}<div style="font-size:10px; color:#6B7280;">pHi</div>
  </div>

  <div style="position:absolute; top:8px; left:{sess_lo_pos:.2f}%; transform:translateX(-50%); font-size:11px; color:#DC2626;">
    {session_low:.2f}<div style="font-size:10px;">sLo</div>
  </div>
  <div style="position:absolute; top:8px; left:{sess_hi_pos:.2f}%; transform:translateX(-50%); font-size:11px; color:#16A34A;">
    {session_high:.2f}<div style="font-size:10px;">sHi</div>
  </div>

  <div style="position:absolute; top:8px; left:{last_pos:.2f}%; transform:translateX(-50%); font-size:11px; color:#111827;">
    {last_price:.2f}<div style="font-size:10px;">Last</div>
  </div>
</div>
"""
st.markdown(html_bar, unsafe_allow_html=True)

st.caption(
    "Hollow grey = prior RTH; blue = current session; black rule = last price. "
    "Labels are rendered inside the visualization with padding so they don’t clip."
)

# =====================================================
# Example 2 – Altair Range Bars (narrow scale + rule)
# =====================================================
st.markdown("### Example 2 — Horizontal Range Bars (Altair)")

range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low,    "end": prior_high,   "y": "Range"},
        {"label": "Current Session", "start": session_low, "end": session_high, "y": "Range"},
    ]
)
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})

domain_min = min_all
domain_max = max_all

base = (
    alt.Chart(range_df)
    .mark_bar(height=26)
    .encode(
        x=alt.X("start:Q", title="Price", scale=alt.Scale(domain=[domain_min, domain_max])),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Prior RTH", "Current Session"], range=["#9CA3AF", "#2563EB"]),
            legend=alt.Legend(title="Range"),
        ),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("start:Q"), alt.Tooltip("end:Q")],
    )
)

last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(x=alt.X("price:Q", scale=alt.Scale(domain=[domain_min, domain_max])), y="y:N")
)

# Text labels BELOW the numbers (dy=6)
labels = (
    alt.Chart(range_df)
    .mark_text(baseline="top", dy=6, fontSize=11, color="#374151")
    .encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        y="y:N",
        text=alt.Text("start:Q", format=".2f")
    )
) + (
    alt.Chart(range_df)
    .mark_text(baseline="top", dy=6, fontSize=11, color="#374151")
    .encode(
        x=alt.X("end:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        y="y:N",
        text=alt.Text("end:Q", format=".2f")
    )
)

chart = (base + last_rule + labels).properties(height=110, width="container").configure_view(strokeWidth=0)
st.altair_chart(chart, use_container_width=True)

# ---------- Notes ----------
with st.expander("Notes"):
    st.write(
        "- Example 1 is pure HTML/CSS (fast, flexible for labels/markers). "
        "If something ever shows up as raw text, it means Streamlit escaped a block; "
        "keeping styles inline (as above) prevents that.\n"
        "- Example 2 uses Altair with an explicit domain narrowed to the relevant prices "
        "and a vertical rule for last price."
    )
