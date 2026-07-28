#!/usr/bin/env python3
"""Session-only premarket planning page for the trading dashboard."""

from datetime import date

import streamlit as st


st.set_page_config(page_title="Daily Cockpit", layout="wide")

CHECKLIST_ITEMS = {
    "daily_cockpit_rest_check": "I am rested, focused, and emotionally ready to trade.",
    "daily_cockpit_calendar_check": "I reviewed the economic calendar and scheduled news.",
    "daily_cockpit_levels_check": "I marked the key overnight and regular-session levels.",
    "daily_cockpit_setup_check": "I will wait for the A+ setup described in this plan.",
    "daily_cockpit_stop_check": "I will stop when the maximum daily loss is reached.",
}


def _reset_cockpit() -> None:
    """Restore all cockpit widgets to their initial session values."""
    defaults = {
        "daily_cockpit_date": date.today(),
        "daily_cockpit_bias": "Neutral",
        "daily_cockpit_plan": "",
        "daily_cockpit_max_loss": 0.0,
        "daily_cockpit_position_size": 0,
        "daily_cockpit_setup": "",
        "daily_cockpit_trade_type": "Base hit",
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    for key in CHECKLIST_ITEMS:
        st.session_state[key] = False


st.title("Daily Cockpit")
st.caption(
    "Build the plan before the session. This prototype keeps entries only in "
    "the current Streamlit session and does not write to Supabase."
)

date_col, bias_col, trade_type_col = st.columns(3)
with date_col:
    trading_date = st.date_input(
        "Trading date",
        value=date.today(),
        key="daily_cockpit_date",
    )
with bias_col:
    daily_bias = st.selectbox(
        "Daily market bias",
        options=["Bullish", "Neutral", "Bearish"],
        index=1,
        key="daily_cockpit_bias",
    )
with trade_type_col:
    trade_classification = st.radio(
        "Trade classification",
        options=["Base hit", "Home run"],
        horizontal=True,
        key="daily_cockpit_trade_type",
        help="Choose the intended management style before entering a trade.",
    )

plan_col, setup_col = st.columns(2)
with plan_col:
    premarket_plan = st.text_area(
        "Premarket plan",
        placeholder="Key levels, expected scenarios, invalidation, and when to stand aside...",
        height=180,
        key="daily_cockpit_plan",
    )
with setup_col:
    a_plus_setup = st.text_area(
        "A+ setup description",
        placeholder="Describe the context, trigger, stop placement, and target...",
        height=180,
        key="daily_cockpit_setup",
    )

st.subheader("Risk limits")
loss_col, size_col = st.columns(2)
with loss_col:
    maximum_daily_loss = st.number_input(
        "Maximum daily loss ($)",
        min_value=0.0,
        value=0.0,
        step=50.0,
        key="daily_cockpit_max_loss",
        help="Trading remains stopped until this is greater than zero.",
    )
with size_col:
    starting_position_size = st.number_input(
        "Starting position size",
        min_value=0,
        value=0,
        step=1,
        key="daily_cockpit_position_size",
        help="Trading remains stopped until this is at least one.",
    )

st.subheader("Process checklist")
check_cols = st.columns(2)
checklist_complete = True
for index, (key, label) in enumerate(CHECKLIST_ITEMS.items()):
    with check_cols[index % 2]:
        checked = st.checkbox(label, key=key)
        checklist_complete = checklist_complete and checked

risk_limits_valid = maximum_daily_loss > 0 and starting_position_size > 0
trading_permitted = checklist_complete and risk_limits_valid

st.divider()
status_col, summary_col = st.columns([1, 2])
with status_col:
    st.subheader("Session status")
    if trading_permitted:
        st.success("TRADING PERMITTED", icon="✅")
        st.caption("All process checks are complete and both risk limits are valid.")
    else:
        st.error("STOP TRADING", icon="🛑")
        if not checklist_complete:
            st.caption("Complete every process check before trading.")
        if not risk_limits_valid:
            st.caption("Set a positive maximum daily loss and starting position size.")

with summary_col:
    st.subheader("Plan summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Date", trading_date.strftime("%Y-%m-%d"))
    metric_cols[1].metric("Bias", daily_bias)
    metric_cols[2].metric("Trade type", trade_classification)
    metric_cols[3].metric(
        "Checklist",
        f"{sum(bool(st.session_state.get(key)) for key in CHECKLIST_ITEMS)}/{len(CHECKLIST_ITEMS)}",
    )
    if not premarket_plan.strip() or not a_plus_setup.strip():
        st.warning(
            "The premarket plan and A+ setup description are still incomplete. "
            "Document them before the session even when the required controls are satisfied."
        )

st.button("Reset cockpit", on_click=_reset_cockpit)
