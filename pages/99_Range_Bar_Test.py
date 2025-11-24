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
    prior_low = st.number_input("Prior RTH Low", value=6539.00, step=0.25)
    prior_high = st.number_input("Prior RTH High", value=6677.50, step=0.25)
with c2:
    session_low = st.number_input("Current Session Low", value=6625.00, step=0.25)
    session_high = st.number_input("Current Session High", value=6669.25, step=0.25)
with c3:
    last_price = st.number_input("Last Price", value=6636.75, step=0.25)

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
# Example 1 – HTML overlay range bar (robust)
# ============================================

st.markdown("## Current range vs prior range — Example 1 (HTML overlay)")

# We’ll keep labels simple: number on the first line, tag on second line.
def _label_block(left_pct: float, top_px: int, text_num: str, text_tag: str, color="#111827"):
    return (
        f'<div style="position:absolute;left:{left_pct:.2f}%;'
        f'transform:translateX(-50%);top:{top_px}px;'
        f'font-size:12px;line-height:14px;text-align:center;color:{color};">'
        f'<div style="font-weight:600;">{text_num}</div>'
        f'<div style="opacity:0.9;">{text_tag}</div>'
        f'</div>'
    )

# HTML bar container (single-line styles to avoid escaping issues)
html_bar = f"""
<div style="position:relative;width:100%;height:120px;margin:8px 0;border-radius:10px;
            background-color:#F9FAFB;border:1px solid #E5E7EB;overflow:hidden;">

  <!-- prior RTH hollow range (center lane) -->
  <div style="position:absolute;top:46px;height:28px;left:{prior_left:.2f}%;
              width:{prior_width:.2f}%;border-radius:6px;border:2px solid #9CA3AF;
              background-color:rgba(209,213,219,0.12);"></div>

  <!-- current session filled range (slimmer, overlay) -->
  <div style="position:absolute;top:52px;height:16px;left:{sess_left:.2f}%;
              width:{sess_width:.2f}%;border-radius:6px;background-color:rgba(37,99,235,0.55);"></div>

  <!-- last price marker (vertical rule) -->
  <div style="position:absolute;top:36px;bottom:36px;left:{last_pos:.2f}%;
              width:2px;background-color:#111827;"></div>

  <!-- Labels: numbers + tags -->
  { _label_block(prior_left, 16, f"{prior_low:.2f}", "pLo") }
  { _label_block(prior_left + prior_width, 16, f"{prior_high:.2f}", "pHi") }
  { _label_block(sess_left, 88, f"{session_low:.2f}", "sLo") }
  { _label_block(sess_left + sess_width, 88, f"{session_high:.2f}", "sHi") }
  { _label_block(last_pos, 16, f"{last_price:.2f}", "Last") }

</div>
"""

st.markdown(html_bar, unsafe_allow_html=True)

st.caption(
    "Grey hollow = prior RTH range. Blue filled = current session range. "
    "Black rule = last. Labels show price (top line) and tag (bottom line)."
)

# ============================================
# Example 2 – Altair horizontal range “candle”
# ============================================

st.markdown("## Current range vs prior range — Example 2 (Altair)")

# Create simple two-row DF for ranges
range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low,    "end": prior_high,    "y": "Range"},
        {"label": "Session",   "start": session_low,  "end": session_high,  "y": "Range"},
    ]
)

# Base bars (overlapping)
base = (
    alt.Chart(range_df)
    .mark_bar(height=28)
    .encode(
        x=alt.X("start:Q", title="Price",
                scale=alt.Scale(domain=[min_all, max_all])),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Prior RTH", "Session"],
                            range=["#9CA3AF", "#2563EB"]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
)

# Last price vertical rule
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})
last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(x=alt.X("price:Q", scale=alt.Scale(domain=[min_all, max_all])), y="y:N")
)

# Text annotations UNDER each bar end (numbers, with tag on next line via \n)
text_df = pd.DataFrame([
    {"x": prior_low,    "y": "Range", "txt": f"{prior_low:.2f}\npLo"},
    {"x": prior_high,   "y": "Range", "txt": f"{prior_high:.2f}\npHi"},
    {"x": session_low,  "y": "Range", "txt": f"{session_low:.2f}\nsLo"},
    {"x": session_high, "y": "Range", "txt": f"{session_high:.2f}\nsHi"},
    {"x": last_price,   "y": "Range", "txt": f"{last_price:.2f}\nLast"},
])

text_anno = (
    alt.Chart(text_df)
    .mark_text(baseline="top", dy=10, fontSize=11, fontWeight="bold", color="#111827", align="center")
    .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[min_all, max_all])), y="y:N", text="txt:N")
)

chart = (
    (base + last_rule + text_anno)
    .properties(height=110, width="container")
    .configure_view(strokeWidth=0)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Altair version: grey bar = prior RTH range, blue bar = current session, black rule = last. "
    "Labels appear below each reference."
)
