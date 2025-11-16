#!/usr/bin/env python3
# pages/03_Overtrading_Tilt.py

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from supabase import create_client

# ---- CONFIG ----
st.set_page_config(page_title="Overtrading & Tilt", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# IMPORTANT: set this in your .streamlit/secrets.toml
# [general]
# TJ_USER_ID = "your-uuid-here"
USER_ID = st.secrets.get("TJ_USER_ID")


# ========= Helpers =========

def load_trades() -> pd.DataFrame:
    """
    Load tj_trades for the current user from Supabase.
    """
    if USER_ID is None:
        st.error(
            "TJ_USER_ID is not set in secrets. "
            "Add your user UUID to .streamlit/secrets.toml as TJ_USER_ID."
        )
        return pd.DataFrame()

    resp = (
        sb.table("tj_trades")
        .select("*")
        .eq("user_id", USER_ID)
        .order("entry_ts_est", desc=False)
        .execute()
    )

    data = resp.data or []
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Parse timestamps
    df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
    df["exit_ts_est"] = pd.to_datetime(df["exit_ts_est"])

    # Make sure numerics are numeric
    for col in ["pnl_gross", "fees", "pnl_net", "planned_risk", "r_multiple", "qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_prev_trade_info(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous trade info and time gap between trades (in minutes).
    Uses:
      entry_ts_est, exit_ts_est, side, r_multiple
    """
    if trades.empty:
        return trades

    df = trades.sort_values("entry_ts_est").copy()

    df["prev_exit_ts_est"] = df["exit_ts_est"].shift(1)
    df["prev_r_multiple"] = df["r_multiple"].shift(1)
    df["prev_side"] = df["side"].shift(1)

    df["mins_since_prev_exit"] = (
        (df["entry_ts_est"] - df["prev_exit_ts_est"])
        .dt.total_seconds() / 60.0
    )

    df["prev_was_loss"] = df["prev_r_multiple"] < 0

    return df


def basic_performance_metrics(df: pd.DataFrame) -> dict:
    """
    Returns simple metrics for a trade subset.
    Uses r_multiple.
    """
    df = df.copy()
    df = df[~df["r_multiple"].isna()]

    if df.empty:
        return {
            "trades": 0,
            "win_rate_%": 0.0,
            "avg_R": 0.0,
            "total_R": 0.0,
        }

    trades = len(df)
    wins = (df["r_multiple"] > 0).sum()

    return {
        "trades": trades,
        "win_rate_%": round(100 * wins / trades, 1),
        "avg_R": round(df["r_multiple"].mean(), 2),
        "total_R": round(df["r_multiple"].sum(), 2),
    }


def loss_streaks_same_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds sequences of consecutive losing trades in the same direction (side).
    Returns a DataFrame where each row is a streak.
    Uses:
      entry_ts_est, exit_ts_est, side, r_multiple
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "streak_id", "side", "length",
            "total_R", "avg_R", "start_time", "end_time"
        ])

    d = df.sort_values("entry_ts_est").copy()

    is_loss = d["r_multiple"] < 0
    same_side_as_prev = d["side"] == d["side"].shift(1)

    # new streak whenever NOT (loss & prev was loss & same side)
    new_streak = ~(is_loss & is_loss.shift(1) & same_side_as_prev)

    d["streak_id"] = new_streak.cumsum()

    streaks = (
        d[is_loss]
        .groupby("streak_id")
        .agg(
            side=("side", "first"),
            length=("r_multiple", "size"),
            total_R=("r_multiple", "sum"),
            avg_R=("r_multiple", "mean"),
            start_time=("entry_ts_est", "first"),
            end_time=("exit_ts_est", "last"),
        )
        .reset_index()
    )

    streaks["avg_R"] = streaks["avg_R"].round(2)
    streaks["total_R"] = streaks["total_R"].round(2)

    return streaks


def build_equity_curve(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Build cumulative R equity curve for plotting.
    """
    if df.empty:
        return pd.DataFrame(columns=["entry_ts_est", "cum_R", "series"])

    d = df.sort_values("entry_ts_est").copy()
    d = d[~d["r_multiple"].isna()]
    d["cum_R"] = d["r_multiple"].cumsum()
    d["series"] = label
    return d[["entry_ts_est", "cum_R", "series"]]


# ========= Layout =========

st.title("Overtrading & Tilt")

trades_df = load_trades()

if trades_df.empty:
    st.info("No trades found in tj_trades for this user.")
    st.stop()

# ---- Sidebar filters ----
with st.sidebar:
    st.header("Filters")

    # Date range
    min_date = trades_df["entry_ts_est"].min().date()
    max_date = trades_df["entry_ts_est"].max().date()

    date_start, date_end = st.date_input(
        "Entry date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    mask_date = (
        (trades_df["entry_ts_est"].dt.date >= date_start) &
        (trades_df["entry_ts_est"].dt.date <= date_end)
    )

    # Symbol filter
    symbols = sorted([s for s in trades_df["symbol"].dropna().unique()])
    selected_symbols = st.multiselect(
        "Symbol(s)",
        options=symbols,
        default=symbols,
    )
    if selected_symbols:
        mask_symbol = trades_df["symbol"].isin(selected_symbols)
    else:
        mask_symbol = True

    # Side filter
    sides = sorted([s for s in trades_df["side"].dropna().unique()])
    selected_sides = st.multiselect(
        "Side(s)",
        options=sides,
        default=sides,
    )
    if selected_sides:
        mask_side = trades_df["side"].isin(selected_sides)
    else:
        mask_side = True

    filtered = trades_df[mask_date & mask_symbol & mask_side].copy()

st.caption(
    f"Showing {len(filtered)} trades "
    f"from {date_start} to {date_end}"
    + (f" | Symbols: {', '.join(selected_symbols)}" if selected_symbols else "")
)

if filtered.empty:
    st.warning("No trades match the current filter selection.")
    st.stop()

# Add previous trade info
tilt_df = add_prev_trade_info(filtered)

# ---- Cooldown /Tilt analysis ----

st.subheader("Cooldown after a Losing Trade")

cooldown_minutes = st.slider(
    "Cooldown window (minutes) after a LOSS to consider a re-entry as 'tilt'",
    min_value=1,
    max_value=60,
    value=5,
    step=1,
)

# Tilt = previous trade was loss AND entry within cooldown window
tilt_trades = tilt_df[
    (tilt_df["prev_was_loss"]) &
    (tilt_df["mins_since_prev_exit"].notna()) &
    (tilt_df["mins_since_prev_exit"] >= 0) &
    (tilt_df["mins_since_prev_exit"] <= cooldown_minutes)
]

# Respected = everything else (or first trades / no previous loss)
respected_trades = tilt_df[
    tilt_df.index.difference(tilt_trades.index)
]

all_metrics = basic_performance_metrics(tilt_df)
tilt_metrics = basic_performance_metrics(tilt_trades)
respected_metrics = basic_performance_metrics(respected_trades)

col_all, col_tilt, col_respected = st.columns(3)

with col_all:
    st.caption("All trades (filtered)")
    st.metric("Trades", all_metrics["trades"])
    st.metric("Win rate (%)", all_metrics["win_rate_%"])
    st.metric("Avg R", all_metrics["avg_R"])
    st.metric("Total R", all_metrics["total_R"])

with col_tilt:
    st.caption(f"‘Tilt’ trades (loss → re-entry ≤ {cooldown_minutes} min)")
    st.metric("Trades", tilt_metrics["trades"])
    st.metric("Win rate (%)", tilt_metrics["win_rate_%"])
    st.metric("Avg R", tilt_metrics["avg_R"])
    st.metric("Total R", tilt_metrics["total_R"])

with col_respected:
    st.caption(f"Trades respecting cooldown > {cooldown_minutes} min")
    st.metric("Trades", respected_metrics["trades"])
    st.metric("Win rate (%)", respected_metrics["win_rate_%"])
    st.metric("Avg R", respected_metrics["avg_R"])
    st.metric("Total R", respected_metrics["total_R"])

# ---- Equity curve with vs without tilt trades ----

st.markdown("### Equity Curve: With vs Without Tilt Trades")

curve_all = build_equity_curve(tilt_df, "All trades")
curve_no_tilt = build_equity_curve(respected_trades, "Exclude tilt trades")

curve = pd.concat([curve_all, curve_no_tilt], ignore_index=True)

if not curve.empty:
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x="entry_ts_est:T",
            y="cum_R:Q",
            color="series:N",
            tooltip=["entry_ts_est:T", "cum_R:Q", "series:N"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough data to build an equity curve.")

# ---- Recent tilt candidates table ----

st.markdown("### Recent Tilt Candidates")

if tilt_trades.empty:
    st.info("No tilt trades found for the current cooldown setting.")
else:
    cols_to_show = [
        "entry_ts_est",
        "exit_ts_est",
        "symbol",
        "side",
        "qty",
        "r_multiple",
        "pnl_net",
        "mins_since_prev_exit",
        "prev_r_multiple",
        "prev_side",
    ]
    cols_to_show = [c for c in cols_to_show if c in tilt_trades.columns]

    st.dataframe(
        tilt_trades.sort_values("entry_ts_est", ascending=False)
        .head(50)[cols_to_show]
    )

st.divider()

# ---- Same-direction loss streaks ----

st.subheader("Same-Direction Loss Streaks")

streak_df = loss_streaks_same_direction(tilt_df)

min_streak_len = st.slider(
    "Show streaks with at least N consecutive losses (same side)",
    min_value=2,
    max_value=10,
    value=2,
    step=1,
)

filtered_streaks = streak_df[streak_df["length"] >= min_streak_len]

c1, c2 = st.columns(2)
with c1:
    st.metric("Total loss streaks (all)", int(streak_df.shape[0]))
with c2:
    st.metric(f"Streaks with ≥ {min_streak_len} losses", int(filtered_streaks.shape[0]))

if filtered_streaks.empty:
    st.info("No loss streaks match the current minimum length.")
else:
    st.dataframe(
        filtered_streaks.sort_values("start_time", ascending=False)
    )
