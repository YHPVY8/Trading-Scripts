#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Range Bar Test Page", layout="wide")

st.title("Range Bar Test Page")
st.caption(
    "Play with the inputs below to test intraday range visualizations "
    "(prior RTH range vs current session range and last price)."
)

# ============================================
# Inputs (no DB – just manual for testing)
# ============================================
st.markdown("### Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    prior_low = st.number_input("Prior RTH Low", value=6850.0, step=0.25)
    prior_high = st.number_input("Prior RTH High", value=6900.0, step=0.25)
with c2:
    session_low = st.number_input("Current Session Low", value=6865.0, step=0.25)
    session_high = st.number_input("Current Session High", value=6935.0, step=0.25)
with c3:
    last_price = st.number_input("Last Price", value=6925.0, step=0.25)

# Sanity guard: make sure highs > lows to avoid weird math
if prior_high <= prior_low:
    st.error("Prior RTH High must be greater than Prior RTH Low.")
    st.stop()
if session_high <= session_low:
    st.error("Current Session High must be greater than Current Session Low.")
    st.stop()

# Global min/max to scale 0–100% across ALL prices involved
all_vals = [prior_low, prior_high, session_low, session_high, last_price]
min_all = float(np.nanmin(all_vals))
max_all = float(np.nanmax(all_vals))
if max_all == min_all:
    max_all = min_all + 1.0  # avoid div/0

span_all = max_all - min_all

def _pct(p: float) -> float:
    """Map price -> 0–100% across full combined range."""
    return (p - min_all) / span_all * 100.0

# positions for HTML example
prior_left = _pct(prior_low)
prior_width = _pct(prior_high) - prior_left

sess_left = _pct(session_low)
sess_width = _pct(session_high) - sess_left

last_pos = _pct(last_price)

# ============================================
# Example 1 – HTML overlay range bar
# ============================================

st.markdown("### Example 1 – Overlay Range Bar (HTML)")

st.write(
    "• Grey hollow bar = **prior RTH range**  \n"
    "• Blue filled bar = **current session range**  \n"
    "• Black line = **last price**"
)

# Labels above / below to keep HTML simple
c_top1, c_top2, c_top3 = st.columns(3)
c_top1.markdown(f"**Prior RTH Low:** {prior_low:.2f}")
c_top2.markdown(f"**Prior RTH High:** {prior_high:.2f}")
c_top3.markdown(f"**Last Price:** {last_price:.2f}")

c_mid1, c_mid2 = st.columns(2)
c_mid1.markdown(f"**Session Low:** {session_low:.2f}")
c_mid2.markdown(f"**Session High:** {session_high:.2f}")

# HTML bar
html_bar = f"""
<div style="position:relative;
            width:100%;
            height:60px;
            margin-top:8px;
            margin-bottom:8px;
            border-radius:10px;
            background-color:#F3F4F6;
            border:1px solid #D1D5DB;
            overflow:hidden;">

  <!-- prior RTH hollow range -->
  <div style="
        position:absolute;
        top:20%;
        height:60%;
        left:{prior_left:.1f}%;
        width:{prior_width:.1f}%;
        border-radius:6px;
        border:2px solid #9CA3AF;
        background-color:rgba(209,213,219,0.15);
  "></div>

  <!-- current session filled range -->
  <div style="
        position:absolute;
        top:35%;
        height:30%;
        left:{sess_left:.1f}%;
        width:{sess_width:.1f}%;
        border-radius:6px;
        background-color:rgba(37,99,235,0.50);
  "></div>

  <!-- last price marker -->
  <div style="
        position:absolute;
        top:10%;
        bottom:10%;
        left:{last_pos:.1f}%;
        width:2px;
        background-color:#111827;
  "></div>

</div>
"""

st.markdown(html_bar, unsafe_allow_html=True)

st.caption(
    "This bar is built with plain HTML/CSS. All positioning is scaled between the "
    "minimum and maximum of: prior RTH low/high, session low/high, and last price."
)

# ============================================
# Example 2 – Altair horizontal range “candle”
# ============================================

st.markdown("### Example 2 – Horizontal Range Bars (Altair)")

st.write(
    "Here we use Altair to draw overlapping horizontal bars:  \n"
    "• Grey bar = **prior RTH range**  \n"
    "• Blue bar = **current session range**  \n"
    "• Black rule = **last price**"
)

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
        x=alt.X("start:Q", title="Price"),
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
    )
)

last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})
last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(x="price:Q", y="y:N")
)

chart = (
    (base + last_rule)
    .properties(height=80, width="container")
    .configure_view(strokeWidth=0)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "This Altair example is very robust and easy to extend (add text annotations, colors "
    "for expansions beyond prior range, etc.). Once you pick the style you prefer, we can "
    "plug real values in from es_trade_day_summary + es_30m."
)
