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
# Defaults = your example values
# ============================================
st.markdown("### Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    prior_low = st.number_input("Prior RTH Low", value=6539.00, step=0.25, format="%.2f")
    prior_high = st.number_input("Prior RTH High", value=6677.50, step=0.25, format="%.2f")
with c2:
    session_low = st.number_input("Current Session Low", value=6625.00, step=0.25, format="%.2f")
    session_high = st.number_input("Current Session High", value=6669.25, step=0.25, format="%.2f")
with c3:
    last_price = st.number_input("Last Price", value=6636.75, step=0.25, format="%.2f")

# Sanity guards
if prior_high <= prior_low:
    st.error("Prior RTH High must be greater than Prior RTH Low.")
    st.stop()
if session_high <= session_low:
    st.error("Current Session High must be greater than Current Session Low.")
    st.stop()

# Global min/max for consistent scaling (tight, with a small pad)
all_vals = [prior_low, prior_high, session_low, session_high, last_price]
min_all = float(np.nanmin(all_vals))
max_all = float(np.nanmax(all_vals))
if max_all == min_all:
    max_all = min_all + 1.0
span_all = max_all - min_all
pad = max(1.0, span_all * 0.03)  # ~3% or at least 1 point
domain_min = min_all - pad
domain_max = max_all + pad
domain_span = domain_max - domain_min

def _pct(p: float) -> float:
    """Map price -> 0–100% across full combined range (with padding)."""
    return (p - domain_min) / domain_span * 100.0

# positions for HTML example
prior_left = _pct(prior_low)
prior_width = _pct(prior_high) - prior_left

sess_left = _pct(session_low)
sess_width = _pct(session_high) - sess_left

last_pos = _pct(last_price)

# ============================================
# Example 1 – HTML overlay range bar (robust)
# ============================================

st.markdown("### Example 1 – Overlay Range Bar (HTML)")

st.write(
    "• Grey hollow = **prior RTH range**  \n"
    "• Blue filled = **current session range**  \n"
    "• Black line = **last price**  \n"
    "• Green/Red ticks below = **session High/Low markers**"
)

# Labels top/bottom (simple, stable)
c_top1, c_top2, c_top3 = st.columns(3)
c_top1.markdown(f"**Prior RTH Low:** {prior_low:.2f}")
c_top2.markdown(f"**Prior RTH High:** {prior_high:.2f}")
c_top3.markdown(f"**Last Price:** {last_price:.2f}")

c_mid1, c_mid2 = st.columns(2)
c_mid1.markdown(f"**Session Low:** {session_low:.2f}")
c_mid2.markdown(f"**Session High:** {session_high:.2f}")

# HTML bar — keep styles compact and inline; avoid HTML comments
html_bar = f"""
<div style="position:relative;
            width:100%;
            height:84px;
            margin-top:8px;
            margin-bottom:8px;
            border-radius:10px;
            background-color:#F8FAFC;
            border:1px solid #CBD5E1;
            overflow:visible;">

  <!-- lane background -->
  <div style="position:absolute; left:0; right:0; top:28px; height:28px; background-color:#F3F4F6;"></div>

  <!-- prior RTH hollow range -->
  <div style="position:absolute;
              top:30px; height:24px;
              left:{prior_left:.3f}%; width:{prior_width:.3f}%;
              border-radius:6px;
              border:2px solid #9CA3AF;
              background-color:rgba(209,213,219,0.10);"></div>

  <!-- current session filled range -->
  <div style="position:absolute;
              top:34px; height:16px;
              left:{sess_left:.3f}%; width:{sess_width:.3f}%;
              border-radius:6px;
              background-color:rgba(37,99,235,0.55);"></div>

  <!-- LAST price marker -->
  <div title="Last {last_price:.2f}" style="
        position:absolute; top:24px; height:36px;
        left:{last_pos:.3f}%;
        width:2px; background-color:#111827;"></div>

  <!-- Prior range labels above (aligned to ends) -->
  <div style="position:absolute; top:6px; left:{prior_left:.3f}%;
              transform:translateX(-50%); font-size:0.75rem; color:#475569;">
      {prior_low:.2f}
  </div>
  <div style="position:absolute; top:6px; left:{prior_left+prior_width:.3f}%;
              transform:translateX(-50%); font-size:0.75rem; color:#475569;">
      {prior_high:.2f}
  </div>

  <!-- Session Low/High ticks and labels below -->
  <div title="Session Low {session_low:.2f}" style="
        position:absolute; top:62px; height:14px;
        left:{_pct(session_low):.3f}%;
        width:2px; background-color:#DC2626;"></div>
  <div style="position:absolute; top:74px; left:{_pct(session_low):.3f}%;
              transform:translateX(-50%); font-size:0.72rem; color:#DC2626;">
      {session_low:.2f}
  </div>

  <div title="Session High {session_high:.2f}" style="
        position:absolute; top:62px; height:14px;
        left:{_pct(session_high):.3f}%;
        width:2px; background-color:#16A34A;"></div>
  <div style="position:absolute; top:74px; left:{_pct(session_high):.3f}%;
              transform:translateX(-50%); font-size:0.72rem; color:#16A34A;">
      {session_high:.2f}
  </div>
</div>
"""
st.markdown(html_bar, unsafe_allow_html=True)

st.caption(
    "Positioning is computed from a tight domain spanning the min/max of prior range, session range, and last price (with a small pad)."
)

# ============================================
# Example 2 – Altair horizontal range bars (tight scale, labels)
# ============================================

st.markdown("### Example 2 – Horizontal Range Bars (Altair)")

# Data for ranges
range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low,    "end": prior_high,   "y": "Range"},
        {"label": "Session",   "start": session_low,  "end": session_high, "y": "Range"},
    ]
)
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})

# Base bars
base = (
    alt.Chart(range_df)
    .mark_bar(height=26)
    .encode(
        x=alt.X("start:Q",
                title="Price",
                scale=alt.Scale(domain=[domain_min, domain_max])),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Prior RTH", "Session"],
                            range=["#9CA3AF", "#2563EB"]),
            legend=alt.Legend(title="Range"),
        ),
        tooltip=["label:N", "start:Q", "end:Q"]
    )
)

# Last price rule + label
last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(
        x=alt.X("price:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
        y=alt.Y("y:N", axis=None)
    )
)
last_text = (
    alt.Chart(last_df)
    .mark_text(dy=-12, fontSize=11, color="#111827")
    .encode(
        x=alt.X("price:Q"),
        y=alt.Y("y:N"),
        text=alt.value(f"Last {last_price:.2f}")
    )
)

# End labels for prior & session ranges
endpoints_df = pd.DataFrame(
    [
        {"label": "Prior Low", "price": prior_low,   "y": "Range"},
        {"label": "Prior High","price": prior_high,  "y": "Range"},
        {"label": "Sess Low",  "price": session_low, "y": "Range"},
        {"label": "Sess High", "price": session_high,"y": "Range"},
    ]
)
end_text = (
    alt.Chart(endpoints_df)
    .mark_text(dy=18, fontSize=10, color="#334155")
    .encode(
        x=alt.X("price:Q"),
        y=alt.Y("y:N"),
        text=alt.Text("label:N")
    )
)

chart = (
    (base + last_rule + last_text + end_text)
    .properties(height=100, width="container")
    .configure_view(strokeWidth=0)
    .configure_axis(labelFontSize=11, titleFontSize=12)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Altair version with a tight x-axis domain and text labels for prior/session endpoints and last price. "
    "Easy to style and extend (e.g., color changes when the session exceeds prior range, add tooltips, etc.)."
)
