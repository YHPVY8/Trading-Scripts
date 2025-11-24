#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from supabase import create_client
import datetime as dt

st.set_page_config(page_title="Range Bar Test Page", layout="wide")

st.title("Range Bar Test Page")
st.caption(
    "Testing intraday range visualizations (prior RTH range vs current session range and last price)."
)

# =========================
# Supabase client (live data)
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Helpers
# =========================
def _et_trade_day(ts_utc: pd.Timestamp) -> pd.Timestamp:
    """Return ET-midnight timestamp representing the trade_day (rolls at 18:00 ET)."""
    ts_et = ts_utc.tz_convert("US/Eastern")
    return (ts_et.floor("D") + pd.to_timedelta(int(ts_et.hour >= 18), "D"))

def _fetch_current_session_from_es_30m():
    """
    Get latest session (by 18:00 ET roll) low/high + last price from es_30m.
    Returns (trade_date, session_low, session_high, last_price)
    """
    resp = (
        sb.table("es_30m")
          .select("time, low, high, close")
          .order("time", desc=True)
          .limit(3000)   # plenty to cover multiple days
          .execute()
    )
    df = pd.DataFrame(resp.data)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["time_et"] = df["time"].dt.tz_convert("US/Eastern")
    td = _et_trade_day(df["time"])
    df["trade_day"] = td
    latest_td = df["trade_day"].max()
    cur = df[df["trade_day"] == latest_td].sort_values("time")
    if cur.empty:
        return None

    session_low = cur["low"].min()
    session_high = cur["high"].max()
    last_price = cur["close"].iloc[-1]
    trade_date = latest_td.date()
    return trade_date, float(session_low), float(session_high), float(last_price)

def _fetch_prior_rth_from_summary(cur_trade_date: dt.date):
    """
    From es_trade_day_summary, fetch prior day's RTH Hi/Lo using the correct quoted names:
    "RTH Hi", "RTH Lo".
    Returns (prior_trade_date, rth_low, rth_high)
    """
    # Pull a small window up to the current trade_date
    resp = (
        sb.table("es_trade_day_summary")
          .select('trade_date,"RTH Hi","RTH Lo"')
          .lte("trade_date", cur_trade_date.isoformat())
          .order("trade_date", desc=True)
          .limit(10)
          .execute()
    )
    df = pd.DataFrame(resp.data)
    if df.empty:
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    # Prior = max trade_date < cur_trade_date
    prev = df[df["trade_date"] < cur_trade_date].sort_values("trade_date")
    if prev.empty:
        # fallback: if we don't have a prior, use the latest available row as prior
        prev_row = df.sort_values("trade_date").iloc[-1]
    else:
        prev_row = prev.iloc[-1]

    rth_hi = prev_row.get("RTH Hi", np.nan)
    rth_lo = prev_row.get("RTH Lo", np.nan)
    return prev_row["trade_date"], float(rth_lo), float(rth_hi)

def _pct(value, lo, hi):
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo) * 100.0

# =========================
# Pull live data
# =========================
live = _fetch_current_session_from_es_30m()
if live is None:
    st.error("Could not load current session from es_30m.")
    st.stop()

cur_trade_date, session_low, session_high, last_price = live
prior_info = _fetch_prior_rth_from_summary(cur_trade_date)
if prior_info is None:
    st.error("Could not load prior RTH range from es_trade_day_summary.")
    st.stop()

prior_trade_date, prior_low, prior_high = prior_info

# =========================
# Scale and padding (tight domain around ranges)
# =========================
lo_core = min(prior_low, session_low)
hi_core = max(prior_high, session_high)
span_core = max(hi_core - lo_core, 1e-6)

# add ~1.5% padding on both sides to avoid label clipping
pad = span_core * 0.015
scale_min = lo_core - pad
scale_max = hi_core + pad

# Positions for HTML example
prior_left = _pct(prior_low, scale_min, scale_max)
prior_width = _pct(prior_high, scale_min, scale_max) - prior_left
sess_left = _pct(session_low, scale_min, scale_max)
sess_width = _pct(session_high, scale_min, scale_max) - sess_left
last_pos = _pct(last_price, scale_min, scale_max)

# =========================
# Shared Title
# =========================
st.subheader("Current range vs prior range")

# ============================================
# Example 1 – HTML overlay range bar (labels INSIDE)
# ============================================
st.markdown("### Example 1 – Overlay Range Bar (HTML)")

# One container with enough height; labels sit inside the bar
html1 = f"""
<div style="position:relative;
            width:100%;
            height:90px;
            margin-top:8px;
            margin-bottom:8px;
            border-radius:12px;
            background-color:#F8FAFC;
            border:1px solid #E5E7EB;">

  <!-- prior RTH hollow range -->
  <div style="
        position:absolute;
        top:30%;
        height:40%;
        left:{prior_left:.2f}%;
        width:{prior_width:.2f}%;
        border-radius:6px;
        border:2px solid #9CA3AF;
        background-color:rgba(209,213,219,0.10);
  "></div>

  <!-- current session filled range -->
  <div style="
        position:absolute;
        top:40%;
        height:20%;
        left:{sess_left:.2f}%;
        width:{sess_width:.2f}%;
        border-radius:6px;
        background-color:rgba(37,99,235,0.55);
  "></div>

  <!-- LAST price marker -->
  <div title="Last"
       style="position:absolute;
              top:25%;
              bottom:25%;
              left:{last_pos:.2f}%;
              width:2px;
              background-color:#111827;">
  </div>

  <!-- label: prior low (inside, bottom) -->
  <div style="
        position:absolute;
        left:{prior_left:.2f}%;
        bottom:6px;
        transform:translate(-0%, 0);
        font-size:11px; color:#374151;">
    {prior_low:.2f} <span style="opacity:0.8;">pLo</span>
  </div>

  <!-- label: prior high (inside, bottom, aligned right) -->
  <div style="
        position:absolute;
        left:{prior_left + prior_width:.2f}%;
        bottom:6px;
        transform:translate(-100%, 0);
        font-size:11px; color:#374151; text-align:right;">
    {prior_high:.2f} <span style="opacity:0.8;">pHi</span>
  </div>

  <!-- label: session low (inside, top) -->
  <div style="
        position:absolute;
        left:{sess_left:.2f}%;
        top:6px;
        transform:translate(-0%, 0);
        font-size:11px; color:#1F2937;">
    {session_low:.2f} <span style="opacity:0.8;">sLo</span>
  </div>

  <!-- label: session high (inside, top, aligned right) -->
  <div style="
        position:absolute;
        left:{sess_left + sess_width:.2f}%;
        top:6px;
        transform:translate(-100%, 0);
        font-size:11px; color:#1F2937; text-align:right;">
    {session_high:.2f} <span style="opacity:0.8;">sHi</span>
  </div>

  <!-- label: last price (just above the session bar) -->
  <div style="
        position:absolute;
        left:{last_pos:.2f}%;
        top:22%;
        transform:translate(-50%, -100%);
        font-size:11px; color:#111827; font-weight:600;">
    {last_price:.2f}
  </div>
</div>
"""
st.markdown(html1, unsafe_allow_html=True)
st.caption(
    f"Prior from {prior_trade_date} — hollow grey; current session — solid blue; black rule = last price. "
    "Labels are **inside** the visualization with padding so they don’t clip."
)

# ============================================
# Example 2 – Altair overlapping horizontal bars (tight scale)
# ============================================
st.markdown("### Example 2 – Overlapping Bars (Altair)")

range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low, "end": prior_high, "y": "Range"},
        {"label": "Current Session", "start": session_low, "end": session_high, "y": "Range"},
    ]
)

base = (
    alt.Chart(range_df)
    .mark_bar(height=24)
    .encode(
        x=alt.X("start:Q",
                title="Price",
                scale=alt.Scale(domain=[scale_min, scale_max])),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["Prior RTH", "Current Session"],
                range=["#9CA3AF", "#2563EB"],
            ),
            legend=alt.Legend(title="Range"),
        ),
        tooltip=["label:N","start:Q","end:Q"]
    )
)

# Vertical rule for last price
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})
last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(
        x=alt.X("price:Q", scale=alt.Scale(domain=[scale_min, scale_max])),
        y="y:N"
    )
)

# Text annotations BELOW the ends and last price (dy > 0)
text_df = pd.DataFrame(
    [
        {"x": prior_low,   "y": "Range", "txt": f"{prior_low:.2f} pLo", "dy": 18, "align": "left"},
        {"x": prior_high,  "y": "Range", "txt": f"{prior_high:.2f} pHi", "dy": 18, "align": "right"},
        {"x": session_low, "y": "Range", "txt": f"{session_low:.2f} sLo", "dy": 34, "align": "left"},
        {"x": session_high,"y": "Range", "txt": f"{session_high:.2f} sHi", "dy": 34, "align": "right"},
        {"x": last_price,  "y": "Range", "txt": f"{last_price:.2f}",     "dy": -6, "align": "center"},
    ]
)

text_layer = (
    alt.Chart(text_df)
    .mark_text(fontSize=11)
    .encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[scale_min, scale_max])),
        y=alt.Y("y:N"),
        text="txt:N",
        align=alt.Condition(
            alt.datum.align == "left", alt.value("left"),
            alt.Condition(alt.datum.align == "right", alt.value("right"), alt.value("center"))
        ),
        dy="dy:Q"
    )
)

chart = (
    (base + last_rule + text_layer)
    .properties(height=110, width="container", title="Current range vs prior range")
    .configure_view(strokeWidth=0)
    .configure_axis(labelFontSize=11, titleFontSize=12)
)

st.altair_chart(chart, use_container_width=True)

# =========================
# Debug block (optional)
# =========================
with st.expander("Debug values"):
    st.write({
        "current_trade_date": str(cur_trade_date),
        "prior_trade_date": str(prior_trade_date),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "session_low": session_low,
        "session_high": session_high,
        "last_price": last_price,
        "scale_min": scale_min,
        "scale_max": scale_max,
    })
