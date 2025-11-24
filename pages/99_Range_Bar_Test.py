#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Range Bar Test Page", layout="wide")

st.title("Current Range vs Prior Range — Test Page")
st.caption("Testing visualization styles for comparing the current session range to the prior RTH range.")

# ============================================
# Inputs (manual testing)
# ============================================
st.markdown("### Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    prior_low = st.number_input("Prior RTH Low", value=6539.00, step=0.25)
    prior_high = st.number_input("Prior RTH High", value=6677.50, step=0.25)
with c2:
    session_low = st.number_input("Session Low", value=6625.00, step=0.25)
    session_high = st.number_input("Session High", value=6669.25, step=0.25)
with c3:
    last_price = st.number_input("Last Price", value=6636.75, step=0.25)

# Sanity checks
if prior_high <= prior_low:
    st.error("Prior RTH High must be > Prior RTH Low.")
    st.stop()
if session_high <= session_low:
    st.error("Session High must be > Session Low.")
    st.stop()

# ============================================
# Shared scaling for Example 1
# ============================================
all_vals = [prior_low, prior_high, session_low, session_high, last_price]
min_all = min(all_vals)
max_all = max(all_vals)
span_all = max_all - min_all

def _pct(p):
    return (p - min_all) / span_all * 100.0

prior_left = _pct(prior_low)
prior_width = _pct(prior_high) - prior_left

sess_left = _pct(session_low)
sess_width = _pct(session_high) - sess_left

last_pos = _pct(last_price)

# Add extra padding so labels don't get cut off
pad_left_pct = 3
pad_right_pct = 3

# ============================================
# Example 1 — HTML overlay version
# ============================================
st.markdown("## Example 1 — HTML Overlay Range Bar")

# Labels
top = st.columns(3)
top[0].markdown(f"**{prior_low:.2f} (pLo)**")
top[1].markdown(f"**Last: {last_price:.2f}**")
top[2].markdown(f"**{prior_high:.2f} (pHi)**")

mid = st.columns(2)
mid[0].markdown(f"**{session_low:.2f} (sLo)**")
mid[1].markdown(f"**{session_high:.2f} (sHi)**")

html_bar = f"""
<div style="position:relative;
            width:100%;
            height:70px;
            margin-top:8px;
            margin-bottom:8px;
            border-radius:10px;
            background-color:#F3F4F6;
            border:1px solid #D1D5DB;
            overflow:hidden;
            padding-left:{pad_left_pct}%;
            padding-right:{pad_right_pct}%">

  <!-- PRIOR RANGE (hollow rectangle) -->
  <div style="
        position:absolute;
        top:22%;
        height:56%;
        left:{prior_left:.1f}%;
        width:{prior_width:.1f}%;
        border-radius:6px;
        border:2px solid #6B7280;
        background-color:rgba(209,213,219,0.05);
  "></div>

  <!-- CURRENT SESSION RANGE (solid) -->
  <div style="
        position:absolute;
        top:36%;
        height:28%;
        left:{sess_left:.1f}%;
        width:{sess_width:.1f}%;
        border-radius:6px;
        background-color:rgba(37,99,235,0.55);
  "></div>

  <!-- LAST PRICE MARKER -->
  <div style="
        position:absolute;
        top:15%;
        bottom:15%;
        left:{last_pos:.1f}%;
        width:2px;
        background-color:#111827;
  "></div>

</div>
"""

st.markdown(html_bar, unsafe_allow_html=True)
st.caption("HTML version — hollow prior range, filled current range, and vertical last-price marker.")

# ============================================
# Example 2 — Altair version (REVERTED VERSION YOU LIKED)
# ============================================
st.markdown("## Example 2 — Altair Horizontal Range Bars")

# Data for the bars
range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low, "end": prior_high, "y": "Range"},
        {"label": "Current Session", "start": session_low, "end": session_high, "y": "Range"},
    ]
)

# Base bars
base = (
    alt.Chart(range_df)
    .mark_bar(height=28)
    .encode(
        x=alt.X("start:Q", title=None),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Prior RTH", "Current Session"],
                            range=["#9CA3AF", "#2563EB"]),
            legend=alt.Legend(title=""),
        ),
    )
)

# Vertical last price line
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})
lp = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=3)
    .encode(x="price:Q", y="y:N")
)

# Text below references
text_df = pd.DataFrame([
    {"x": prior_low, "y": 0, "text": f"{prior_low:.2f} pLo"},
    {"x": prior_high, "y": 0, "text": f"{prior_high:.2f} pHi"},
    {"x": session_low, "y": 0, "text": f"{session_low:.2f} sLo"},
    {"x": session_high, "y": 0, "text": f"{session_high:.2f} sHi"},
    {"x": last_price, "y": 0, "text": f"{last_price:.2f} Last"},
])

labels = (
    alt.Chart(text_df)
    .mark_text(dy=20, fontSize=12)
    .encode(x="x:Q", y=alt.value(0), text="text:N")
)

chart = (
    (base + lp + labels)
    .properties(height=110, width="container")
    .configure_view(strokeWidth=0)
)

st.altair_chart(chart, use_container_width=True)
st.caption("Altair version — overlapping bars + last-price line + text labels below.")

