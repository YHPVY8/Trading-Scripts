#!/usr/bin/env python3
import streamlit as st
import numpy as np

st.set_page_config(page_title="Range Bar Test", layout="wide")
st.title("Range Bar Testbed")

st.caption(
    "Use the controls below to adjust the prior RTH range, current session range, "
    "and last price. The 10 examples show different ways to visualize the same idea."
)

# =========================
# Inputs
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    prior_low = st.number_input("Prior RTH Low", value=6850.0, step=0.25)
with c2:
    prior_high = st.number_input("Prior RTH High", value=6900.0, step=0.25)
with c3:
    sess_low = st.number_input("Session Low", value=6880.0, step=0.25)
with c4:
    sess_high = st.number_input("Session High", value=6940.0, step=0.25)
with c5:
    last_price = st.number_input("Last Price", value=6935.0, step=0.25)

# Guard so high >= low
if prior_high < prior_low:
    prior_high, prior_low = prior_low, prior_high
if sess_high < sess_low:
    sess_high, sess_low = sess_low, sess_high

# =========================
# Normalization helpers
# =========================
all_vals = [prior_low, prior_high, sess_low, sess_high, last_price]
min_val = np.nanmin(all_vals)
max_val = np.nanmax(all_vals)
span = max(max_val - min_val, 1e-6)

def pct_pos(value: float) -> float:
    """Return percentage [0, 100] across the full extent of prior+session+last."""
    return (value - min_val) / span * 100.0

# Positions in %
prior_start = pct_pos(prior_low)
prior_end = pct_pos(prior_high)
prior_width = max(1.0, prior_end - prior_start)

sess_start = pct_pos(sess_low)
sess_end = pct_pos(sess_high)
sess_width = max(1.0, sess_end - sess_start)

last_pct = pct_pos(last_price)

# Also useful: relative position within prior range (for some examples)
prior_span = max(prior_high - prior_low, 1e-6)
last_within_prior = (last_price - prior_low) / prior_span  # can be <0 or >1


# =========================
# Small helper to render example blocks
# =========================
def block(container, title: str, subtitle: str, inner_html: str):
    """Render a titled card with the given inner HTML."""
    html = f"""
    <div style="border:1px solid #D1D5DB;
                border-radius:10px;
                padding:8px 10px;
                margin-bottom:12px;">
        <div style="font-size:0.9rem;
                    font-weight:600;
                    color:#111827;
                    margin-bottom:2px;">
            {title}
        </div>
        <div style="font-size:0.78rem;
                    color:#4B5563;
                    margin-bottom:6px;">
            {subtitle}
        </div>
        {inner_html}
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


# Convenience label string used in all subtitles
info_line = (
    f"Prior RTH: {prior_low:.2f} – {prior_high:.2f} | "
    f"Session: {sess_low:.2f} – {sess_high:.2f} | "
    f"Last: {last_price:.2f}"
)

# =========================
# Layout: 2 columns of examples
# =========================
left_col, right_col = st.columns(2)

# -------------------------------------------------
# Example 1 – Hollow prior, filled session, last line
# -------------------------------------------------
inner1 = f"""
<div style="position:relative; height:34px; background-color:#F9FAFB;
            border-radius:999px; padding:0 2px;">
    <!-- prior hollow range -->
    <div style="position:absolute;
                top:8px;
                bottom:8px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:999px;
                border:2px solid #6B7280;
                background-color:transparent;">
    </div>

    <!-- session filled range -->
    <div style="position:absolute;
                top:11px;
                bottom:11px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                background-color:#93C5FD;">
    </div>

    <!-- last price marker -->
    <div style="position:absolute;
                top:5px;
                bottom:5px;
                left:{last_pct:.1f}%;
                width:2px;
                background-color:#111827;">
    </div>
</div>
<div style="margin-top:4px;
            display:flex;
            justify-content:space-between;
            font-size:0.75rem;
            color:#6B7280;">
    <span>Full extent min: {min_val:.2f}</span>
    <span>Full extent max: {max_val:.2f}</span>
</div>
"""
block(
    left_col,
    "Example 1 – Hollow prior, filled session, last marker",
    info_line,
    inner1,
)

# -------------------------------------------------
# Example 2 – Prior as band, session as darker band, last dot
# -------------------------------------------------
inner2 = f"""
<div style="position:relative; height:34px;
            border-radius:999px;
            background:linear-gradient(90deg,#F3F4F6,#E5E7EB);">
    <!-- prior band -->
    <div style="position:absolute;
                top:9px;
                bottom:9px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:999px;
                background-color:#E5E7EB;">
    </div>

    <!-- session darker band -->
    <div style="position:absolute;
                top:11px;
                bottom:11px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                background-color:#60A5FA;">
    </div>

    <!-- last as a small circle -->
    <div style="position:absolute;
                top:8px;
                left:calc({last_pct:.1f}% - 6px);
                width:12px;
                height:12px;
                border-radius:999px;
                border:2px solid #111827;
                background-color:#FFFFFF;">
    </div>
</div>
<div style="margin-top:4px;
            font-size:0.75rem;
            color:#6B7280;
            text-align:center;">
    Prior band (light) vs Session band (blue) with Last as dot
</div>
"""
block(
    right_col,
    "Example 2 – Overlapping bands with last as dot",
    info_line,
    inner2,
)

# -------------------------------------------------
# Example 3 – Horizontal “candlestick” for session, prior as frame
# -------------------------------------------------
# Here we treat session as a horizontal candle: thin wick from low–high, thicker body
session_mid = (sess_low + sess_high) / 2.0
mid_pct = pct_pos(session_mid)

inner3 = f"""
<div style="position:relative; height:40px; background-color:#F9FAFB;
            border-radius:8px;">

    <!-- prior frame -->
    <div style="position:absolute;
                top:12px;
                bottom:12px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:4px;
                border:2px dashed #9CA3AF;
                background-color:transparent;">
    </div>

    <!-- session 'wick' -->
    <div style="position:absolute;
                top:18px;
                bottom:18px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                background-color:#4B5563;">
    </div>

    <!-- session 'body' around mid -->
    <div style="position:absolute;
                top:14px;
                bottom:14px;
                left:{min(mid_pct-3, sess_end):.1f}%;
                width:6%;
                border-radius:4px;
                background-color:#60A5FA;">
    </div>

    <!-- last marker -->
    <div style="position:absolute;
                top:10px;
                bottom:10px;
                left:{last_pct:.1f}%;
                width:2px;
                background-color:#111827;">
    </div>
</div>
<div style="margin-top:4px;
            font-size:0.75rem;
            color:#6B7280;
            text-align:center;">
    Prior as dashed frame, Session as horizontal candle, Last as line
</div>
"""
block(
    left_col,
    "Example 3 – Horizontal candlestick style",
    info_line,
    inner3,
)

# -------------------------------------------------
# Example 4 – Inside / outside prior range zones
# -------------------------------------------------
# We color the bar segments: below prior low, inside, above prior high.
full_width = 100.0
below_width = max(0.0, prior_start)
inside_width = max(0.0, prior_end - prior_start)
above_width = max(0.0, full_width - prior_end)

inner4 = f"""
<div style="position:relative; height:30px; border-radius:999px; overflow:hidden;
            border:1px solid #D1D5DB;">

    <!-- below prior low -->
    <div style="position:absolute;
                top:0;
                bottom:0;
                left:0%;
                width:{below_width:.1f}%;
                background-color:#FEE2E2;">
    </div>

    <!-- inside prior -->
    <div style="position:absolute;
                top:0;
                bottom:0;
                left:{prior_start:.1f}%;
                width:{inside_width:.1f}%;
                background-color:#E5E7EB;">
    </div>

    <!-- above prior high -->
    <div style="position:absolute;
                top:0;
                bottom:0;
                left:{prior_end:.1f}%;
                width:{above_width:.1f}%;
                background-color:#DBEAFE;">
    </div>

    <!-- session range -->
    <div style="position:absolute;
                top:8px;
                bottom:8px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                background-color:#60A5FA;">
    </div>

    <!-- last marker -->
    <div style="position:absolute;
                top:4px;
                bottom:4px;
                left:{last_pct:.1f}%;
                width:2px;
                background-color:#111827;">
    </div>
</div>
<div style="margin-top:4px;
            display:flex;
            justify-content:space-between;
            font-size:0.75rem;
            color:#6B7280;">
    <span>Below prior low</span>
    <span>Inside prior</span>
    <span>Above prior high</span>
</div>
"""
block(
    right_col,
    "Example 4 – Zoned bar (below / inside / above)",
    info_line,
    inner4,
)

# -------------------------------------------------
# Example 5 – Double row: prior on top, session on bottom
# -------------------------------------------------
inner5 = f"""
<div style="display:flex; flex-direction:column; gap:4px;">

  <!-- prior row -->
  <div style="position:relative; height:18px;">
    <div style="position:absolute;
                top:4px;
                bottom:4px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:999px;
                background-color:#D1D5DB;">
    </div>
    <div style="position:absolute;
                top:0;
                left:{prior_start:.1f}%;
                font-size:0.7rem;
                color:#4B5563;">
        {prior_low:.2f}
    </div>
    <div style="position:absolute;
                top:0;
                right:{100-prior_end:.1f}%;
                font-size:0.7rem;
                color:#4B5563;">
        {prior_high:.2f}
    </div>
  </div>

  <!-- session row -->
  <div style="position:relative; height:18px;">
    <div style="position:absolute;
                top:4px;
                bottom:4px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                background-color:#93C5FD;">
    </div>
    <div style="position:absolute;
                top:0;
                left:{sess_start:.1f}%;
                font-size:0.7rem;
                color:#1F2937;">
        {sess_low:.2f}
    </div>
    <div style="position:absolute;
                top:0;
                right:{100-sess_end:.1f}%;
                font-size:0.7rem;
                color:#1F2937;">
        {sess_high:.2f}
    </div>

    <!-- last marker -->
    <div style="position:absolute;
                top:2px;
                bottom:2px;
                left:{last_pct:.1f}%;
                width:2px;
                background-color:#111827;">
    </div>
  </div>

</div>
"""
block(
    left_col,
    "Example 5 – Double row (prior vs session)",
    info_line,
    inner5,
)

# -------------------------------------------------
# Example 6 – Prior as hollow pill, session as outline only, last as big marker
# -------------------------------------------------
inner6 = f"""
<div style="position:relative; height:34px; background-color:#FFFFFF;">
    <!-- full extent faint -->
    <div style="position:absolute;
                top:13px;
                bottom:13px;
                left:0%;
                width:100%;
                border-radius:999px;
                background-color:#F3F4F6;">
    </div>

    <!-- prior hollow pill -->
    <div style="position:absolute;
                top:10px;
                bottom:10px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:999px;
                border:2px solid #9CA3AF;
                background-color:transparent;">
    </div>

    <!-- session outline -->
    <div style="position:absolute;
                top:14px;
                bottom:14px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                border:2px solid #60A5FA;
                background-color:transparent;">
    </div>

    <!-- last price marker (larger circle) -->
    <div style="position:absolute;
                top:8px;
                left:calc({last_pct:.1f}% - 7px);
                width:14px;
                height:14px;
                border-radius:999px;
                border:2px solid #111827;
                background-color:#FFFFFF;">
    </div>
</div>
<div style="margin-top:4px;
            font-size:0.75rem;
            color:#6B7280;
            text-align:center;">
    Hollow prior & session outlines, last as bold circle
</div>
"""
block(
    right_col,
    "Example 6 – Outlined prior and session",
    info_line,
    inner6,
)

# -------------------------------------------------
# Example 7 – “Thermometer” style inside prior range
# -------------------------------------------------
# Here we show last position *within prior only*; outside prior clamps at ends.
last_pct_prior = (last_price - prior_low) / prior_span * 100.0
last_pct_prior = max(0.0, min(100.0, last_pct_prior))

inner7 = f"""
<div style="position:relative; height:40px;">

  <!-- prior 'thermometer' tube -->
  <div style="position:absolute;
              top:6px;
              bottom:6px;
              left:{prior_start:.1f}%;
              width:{prior_width:.1f}%;
              border-radius:999px;
              background-color:#E5E7EB;">
  </div>

  <!-- fill from low to last (within prior) -->
  <div style="position:absolute;
              top:9px;
              bottom:9px;
              left:{prior_start:.1f}%;
              width:{last_pct_prior:.1f}%;
              border-radius:999px;
              background:linear-gradient(90deg,#6EE7B7,#22C55E);">
  </div>

  <!-- labels at prior extremes -->
  <div style="position:absolute;
              top:0;
              left:{prior_start:.1f}%;
              font-size:0.7rem;
              color:#4B5563;">
      {prior_low:.2f}
  </div>
  <div style="position:absolute;
              top:0;
              left:calc({prior_end:.1f}% - 3rem);
              font-size:0.7rem;
              color:#4B5563;">
      {prior_high:.2f}
  </div>

  <!-- last label -->
  <div style="position:absolute;
              bottom:0;
              left:calc({prior_start + last_pct_prior * prior_width / 100.0:.1f}% - 2rem);
              font-size:0.7rem;
              color:#111827;">
      Last: {last_price:.2f}
  </div>

</div>
"""
block(
    left_col,
    "Example 7 – Thermometer within prior range",
    info_line,
    inner7,
)

# -------------------------------------------------
# Example 8 – Split bar: left = prior, right = session
# -------------------------------------------------
inner8 = f"""
<div style="display:flex; flex-direction:column; gap:6px;">

  <div style="font-size:0.75rem; color:#4B5563;">Prior RTH range</div>
  <div style="position:relative; height:20px; background-color:#F9FAFB; border-radius:999px;">
    <div style="position:absolute;
                top:5px;
                bottom:5px;
                left:{prior_start:.1f}%;
                width:{prior_width:.1f}%;
                border-radius:999px;
                background-color:#D1D5DB;">
    </div>
  </div>

  <div style="font-size:0.75rem; color:#4B5563;">Current session range (+ Last)</div>
  <div style="position:relative; height:20px; background-color:#F9FAFB; border-radius:999px;">
    <div style="position:absolute;
                top:5px;
                bottom:5px;
                left:{sess_start:.1f}%;
                width:{sess_width:.1f}%;
                border-radius:999px;
                background-color:#93C5FD;">
    </div>
    <div style="position:absolute;
                top:2px;
                bottom:2px;
                left:{last_pct:.1f}%;
                width:2px;
                background-color:#111827;">
    </div>
  </div>

</div>
"""
block(
    right_col,
    "Example 8 – Split prior vs session rows",
    info_line,
    inner8,
)

# -------------------------------------------------
# Example 9 – Prior rectangle with current “extension tails”
# -------------------------------------------------
# Show how far session extends beyond prior (if at all).
below_ext = max(0.0, prior_start - sess_start)
above_ext = max(0.0, sess_end - prior_end)

inner9 = f"""
<div style="position:relative; height:34px; background-color:#FFFFFF;">

  <!-- prior hollow rectangle -->
  <div style="position:absolute;
              top:10px;
              bottom:10px;
              left:{prior_start:.1f}%;
              width:{prior_width:.1f}%;
              border-radius:4px;
              border:2px solid #9CA3AF;
              background-color:transparent;">
  </div>

  <!-- extension below prior low (if any) -->
  <div style="position:absolute;
              top:13px;
              bottom:13px;
              left:{max(0.0, sess_start):.1f}%;
              width:{below_ext:.1f}%;
              border-radius:999px;
              background-color:#F97316;">
  </div>

  <!-- extension above prior high (if any) -->
  <div style="position:absolute;
              top:13px;
              bottom:13px;
              left:{prior_end:.1f}%;
              width:{above_ext:.1f}%;
              border-radius:999px;
              background-color:#22C55E;">
  </div>

  <!-- last marker -->
  <div style="position:absolute;
              top:6px;
              bottom:6px;
              left:{last_pct:.1f}%;
              width:2px;
              background-color:#111827;">
  </div>

</div>
<div style="margin-top:4px;
            font-size:0.75rem;
            color:#6B7280;
            text-align:center;">
    Orange = extension below prior low, Green = extension above prior high
</div>
"""
block(
    left_col,
    "Example 9 – Prior box with extension tails",
    info_line,
    inner9,
)

# -------------------------------------------------
# Example 10 – Mini horizontal “OHLC” bar for session vs prior
# -------------------------------------------------
# Treat prior as faint and session as bold OHLC-ish bar.
inner10 = f"""
<div style="position:relative; height:36px; background-color:#F9FAFB; border-radius:8px;">

  <!-- prior faint bar -->
  <div style="position:absolute;
              top:20px;
              bottom:20px;
              left:{prior_start:.1f}%;
              width:{prior_width:.1f}%;
              background-color:#E5E7EB;">
  </div>

  <!-- prior extremes labels -->
  <div style="position:absolute;
              top:0;
              left:{prior_start:.1f}%;
              font-size:0.7rem;
              color:#4B5563;">
      pLow {prior_low:.1f}
  </div>
  <div style="position:absolute;
              top:0;
              left:calc({prior_end:.1f}% - 3rem);
              font-size:0.7rem;
              color:#4B5563;">
      pHigh {prior_high:.1f}
  </div>

  <!-- session main bar (like OHLC) -->
  <div style="position:absolute;
              top:16px;
              bottom:16px;
              left:{sess_start:.1f}%;
              width:{sess_width:.1f}%;
              background-color:#60A5FA;">
  </div>

  <!-- session low tick -->
  <div style="position:absolute;
              top:14px;
              left:{sess_start:.1f}%;
              width:6px;
              height:2px;
              background-color:#111827;">
  </div>

  <!-- session high tick -->
  <div style="position:absolute;
              top:14px;
              left:calc({sess_end:.1f}% - 6px);
              width:6px;
              height:2px;
              background-color:#111827;">
  </div>

  <!-- last marker tick -->
  <div style="position:absolute;
              top:24px;
              left:{last_pct:.1f}%;
              width:2px;
              height:8px;
              background-color:#111827;">
  </div>

</div>
<div style="margin-top:4px;
            font-size:0.75rem;
            color:#6B7280;
            text-align:center;">
    Prior = faint band, Session = bold bar with low/high ticks, Last = vertical tick
</div>
"""
block(
    right_col,
    "Example 10 – Mini horizontal OHLC style",
    info_line,
    inner10,
)
