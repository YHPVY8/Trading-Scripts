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

# Defaults reflect the example values you provided most recently
c1, c2, c3 = st.columns(3)
with c1:
    prior_low = st.number_input("Prior RTH Low", value=6539.00, step=0.25, format="%.2f")
    prior_high = st.number_input("Prior RTH High", value=6677.50, step=0.25, format="%.2f")
with c2:
    session_low = st.number_input("Current Session Low", value=6625.00, step=0.25, format="%.2f")
    session_high = st.number_input("Current Session High", value=6669.25, step=0.25, format="%.2f")
with c3:
    last_price = st.number_input("Last Price", value=6636.75, step=0.25, format="%.2f")
    pad_pts = st.number_input("Scale padding (pts)", value=6.0, step=0.5)

# Sanity guard
if prior_high <= prior_low:
    st.error("Prior RTH High must be greater than Prior RTH Low.")
    st.stop()
if session_high <= session_low:
    st.error("Current Session High must be greater than Current Session Low.")
    st.stop()

# Global min/max to scale everything consistently (with slight padding)
all_vals = [prior_low, prior_high, session_low, session_high, last_price]
min_all = float(np.nanmin(all_vals)) - float(pad_pts)
max_all = float(np.nanmax(all_vals)) + float(pad_pts)
if max_all <= min_all:
    max_all = min_all + 1.0
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

# A small helper to place labels at a % from the left edge
def _abs_label(left_pct: float, text: str, top_px: int, align: str = "center", color: str = "#111827", size: str = "0.80rem"):
    # translateX to align by center / left / right
    if align == "center":
        tx = "-50%"
    elif align == "right":
        tx = "-100%"
    else:
        tx = "0%"
    return f"""
    <div style="
        position:absolute;
        left:{left_pct:.2f}%;
        top:{top_px}px;
        transform:translateX({tx});
        font-size:{size};
        color:{color};
        background:rgba(255,255,255,0.65);
        padding:2px 6px;
        border-radius:6px;
        border:1px solid #E5E7EB;
        white-space:nowrap;
    ">{text}</div>
    """

# ============================================
# Title bar for both examples
# ============================================
st.markdown("## Current range vs prior range")

# ============================================
# Example 1 – HTML overlay range bar (with labels)
# ============================================
st.markdown("### Example 1 – Overlay Range Bar (HTML)")

st.write(
    "• **Grey hollow bar** = prior RTH range (pLo → pHi)  \n"
    "• **Blue filled bar** = current session range (sLo → sHi)  \n"
    "• **Black line** = last price"
)

# HTML bar + labels
html_bar = f"""
<div style="position:relative;
            width:100%;
            height:96px;
            margin-top:8px;
            margin-bottom:8px;
            border-radius:10px;
            background-color:#F9FAFB;
            border:1px solid #E5E7EB;
            overflow:hidden;">

  <!-- PRIOR RTH hollow range -->
  <div style="
        position:absolute;
        top:36px;
        height:24px;
        left:{prior_left:.2f}%;
        width:{prior_width:.2f}%;
        border-radius:8px;
        border:2px solid #9CA3AF;
        background-color:rgba(209,213,219,0.12);
  "></div>

  <!-- CURRENT SESSION filled range -->
  <div style="
        position:absolute;
        top:42px;
        height:12px;
        left:{sess_left:.2f}%;
        width:{sess_width:.2f}%;
        border-radius:6px;
        background-color:rgba(37,99,235,0.55);
  "></div>

  <!-- LAST price marker -->
  <div title="Last" style="
        position:absolute;
        top:24px;
        bottom:24px;
        left:{last_pos:.2f}%;
        width:2px;
        background-color:#111827;
  "></div>

  <!-- Labels ABOVE prior bar ends -->
  { _abs_label(prior_left,  f"{prior_low:.2f} • pLo", 8,  "right") }
  { _abs_label(prior_left + prior_width, f"{prior_high:.2f} • pHi", 8, "left") }

  <!-- Labels BELOW current session bar ends -->
  { _abs_label(sess_left,  f"{session_low:.2f} • sLo", 72, "right", "#1F2937") }
  { _abs_label(sess_left + sess_width, f"{session_high:.2f} • sHi", 72, "left", "#1F2937") }

  <!-- Label near LAST -->
  { _abs_label(last_pos, f"{last_price:.2f} • Last", 56, "center", "#111827", "0.85rem") }

</div>
"""
st.markdown(html_bar, unsafe_allow_html=True)

st.caption(
    "HTML/CSS overlay. Prior RTH is a hollow rectangle; current session overlays as a thinner filled bar. "
    "Labels are anchored to ends and the last-price line."
)

# ============================================
# Example 2 – Altair horizontal range “candle” (tight scale + last line)
# ============================================
st.markdown("### Example 2 – Horizontal Range Bars (Altair)")

st.write(
    "• **Grey bar** = prior RTH range  \n"
    "• **Blue bar** = current session range  \n"
    "• **Black rule** = last price"
)

range_df = pd.DataFrame(
    [
        {"label": "Prior RTH", "start": prior_low, "end": prior_high, "y": "Range"},
        {"label": "Current Session", "start": session_low, "end": session_high, "y": "Range"},
    ]
)

# Bars
base = (
    alt.Chart(range_df)
    .mark_bar(height=26)
    .encode(
        x=alt.X(
            "start:Q",
            title="Price",
            scale=alt.Scale(domain=[min_all, max_all], nice=False),  # tighten scale
        ),
        x2="end:Q",
        y=alt.Y("y:N", axis=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Prior RTH", "Current Session"], range=["#9CA3AF", "#2563EB"]),
            legend=alt.Legend(title="Range"),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Leg"),
            alt.Tooltip("start:Q", title="Start", format=",.2f"),
            alt.Tooltip("end:Q", title="End", format=",.2f"),
        ],
    )
)

# Last price rule
last_df = pd.DataFrame({"price": [last_price], "y": ["Range"]})
last_rule = (
    alt.Chart(last_df)
    .mark_rule(color="black", strokeWidth=2)
    .encode(
        x=alt.X("price:Q", scale=alt.Scale(domain=[min_all, max_all], nice=False)),
        y="y:N",
        tooltip=[alt.Tooltip("price:Q", title="Last", format=",.2f")],
    )
)

# Optional small point + text at ends (labels)
ends_df = pd.DataFrame(
    [
        {"pos": prior_low, "label": f"{prior_low:.2f} • pLo"},
        {"pos": prior_high, "label": f"{prior_high:.2f} • pHi"},
        {"pos": session_low, "label": f"{session_low:.2f} • sLo"},
        {"pos": session_high, "label": f"{session_high:.2f} • sHi"},
    ]
).assign(y="Range")

end_points = (
    alt.Chart(ends_df)
    .mark_point(filled=True, size=35, color="#374151")
    .encode(x=alt.X("pos:Q", scale=alt.Scale(domain=[min_all, max_all], nice=False)), y="y:N")
)

end_labels = (
    alt.Chart(ends_df)
    .mark_text(dy=-14, fontSize=12, color="#111827")
    .encode(
        x=alt.X("pos:Q", scale=alt.Scale(domain=[min_all, max_all], nice=False)),
        y="y:N",
        text="label:N",
    )
)

chart = (
    (base + last_rule + end_points + end_labels)
    .properties(height=120, width="container")
    .configure_view(strokeWidth=0)
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Altair layering with a tight domain around the inputs (+ adjustable padding). "
    "Easy to extend with additional marks or conditional colors for expansions."
)
