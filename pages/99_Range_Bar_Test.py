#!/usr/bin/env python3
import streamlit as st

st.set_page_config(page_title="Range Bar Test", layout="wide")
st.title("Range Bar Test Page")

# ============================================================
# The sample data you provided
# ============================================================
prior_low = 6539.00
prior_high = 6677.50
last_price = 6636.75
session_low = 6625.00
session_high = 6669.25

st.write("### Sample Data Being Used")
st.write({
    "Prior RTH Low": prior_low,
    "Prior RTH High": prior_high,
    "Last Price": last_price,
    "Session Low": session_low,
    "Session High": session_high,
})

# Helper to normalize any value 0–100% along the bar
def scale(v):
    return (v - prior_low) / (prior_high - prior_low) * 100


# ===================================================================
# EXAMPLE 1 — Hollow prior range box + current session overlay + labels
# ===================================================================
st.markdown("## Example 1 — Hollow Box With Current Session Overlay")

# Compute positions
p_low = scale(prior_low)
p_high = scale(prior_high)
s_low = scale(session_low)
s_high = scale(session_high)
last = scale(last_price)

html_1 = f"""
<div style="position:relative; height:90px; margin-top:20px;">

    <!-- Prior Range Hollow Box -->
    <div style="
        position:absolute;
        top:30px;
        left:{p_low}%;
        width:{p_high - p_low}%;
        height:30px;
        border:2px solid #4B5563;
        background-color:rgba(0,0,0,0);
        border-radius:6px;
    "></div>

    <!-- Current Session Fill -->
    <div style="
        position:absolute;
        top:30px;
        left:{s_low}%;
        width:{s_high - s_low}%;
        height:30px;
        background-color:rgba(37,99,235,0.35);
        border-radius:6px;
    "></div>

    <!-- Last Price Marker -->
    <div style="
        position:absolute;
        top:26px;
        left:{last}%;
        width:3px;
        height:38px;
        background-color:#111827;
    "></div>

    <!-- Labels above for prior range -->
    <div style="position:absolute; top:0; left:{p_low}%; transform:translateX(-50%); font-size:0.8rem;">
        Prior Low<br>{prior_low}
    </div>
    <div style="position:absolute; top:0; left:{p_high}%; transform:translateX(-50%); font-size:0.8rem;">
        Prior High<br>{prior_high}
    </div>

    <!-- Labels below for session range -->
    <div style="position:absolute; top:70px; left:{s_low}%; transform:translateX(-50%); font-size:0.8rem; color:#DC2626;">
        Session Low<br>{session_low}
    </div>
    <div style="position:absolute; top:70px; left:{s_high}%; transform:translateX(-50%); font-size:0.8rem; color:#16A34A;">
        Session High<br>{session_high}
    </div>

    <!-- Label for last price -->
    <div style="position:absolute; top:70px; left:{last}%; transform:translateX(-50%); font-size:0.9rem; font-weight:600;">
        Last<br>{last_price}
    </div>

</div>
"""
st.markdown(html_1, unsafe_allow_html=True)


# ===================================================================
# EXAMPLE 2 — Thick horizontal bar with clear labels and session overlay
# ===================================================================
st.markdown("## Example 2 — Solid Backbone Bar With Overlays")

backbone_top = 40
backbone_height = 10

html_2 = f"""
<div style="position:relative; height:110px; margin-top:20px;">

    <!-- Backbone bar (prior range) -->
    <div style="
        position:absolute;
        top:{backbone_top}px;
        left:{p_low}%;
        width:{p_high - p_low}%;
        height:{backbone_height}px;
        background-color:#D1D5DB;
        border-radius:4px;
    "></div>

    <!-- Session Range Overlay -->
    <div style="
        position:absolute;
        top:{backbone_top - 6}px;
        left:{s_low}%;
        width:{s_high - s_low}%;
        height:{backbone_height + 12}px;
        background-color:rgba(59,130,246,0.3);
        border-radius:4px;
    "></div>

    <!-- Last Price marker -->
    <div style="
        position:absolute;
        top:{backbone_top - 8}px;
        left:{last}%;
        width:3px;
        height:{backbone_height + 16}px;
        background-color:#111827;
    "></div>

    <!-- Labels -->
    <div style="position:absolute; top:0; left:{p_low}%; transform:translateX(-50%); font-size:0.8rem;">
        Prior Low<br>{prior_low}
    </div>
    <div style="position:absolute; top:0; left:{p_high}%; transform:translateX(-50%); font-size:0.8rem;">
        Prior High<br>{prior_high}
    </div>

    <div style="position:absolute; top:80px; left:{s_low}%; transform:translateX(-50%); font-size:0.8rem; color:#DC2626;">
        Session Low<br>{session_low}
    </div>
    <div style="position:absolute; top:80px; left:{s_high}%; transform:translateX(-50%); font-size:0.8rem; color:#16A34A;">
        Session High<br>{session_high}
    </div>

    <div style="position:absolute; top:80px; left:{last}%; transform:translateX(-50%); font-size:0.9rem; font-weight:600;">
        Last<br>{last_price}
    </div>

</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
