#!/usr/bin/env python3
# pages/Overtrading_Tilt.py

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from supabase import create_client

# ===== CONFIG =====
st.set_page_config(page_title="Overtrading & Tilt", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")


# ===== Helper: classify session for a timestamp (same as Performance Stats) =====
def _classify_session(ts):
    """
    Session based on entry_ts_est (EST):
      - Overnight: 18:00:00–03:59:59
      - Premarket: 04:00:00–09:29:59
      - RTH IB:   09:30:00–10:29:59
      - RTH Morning: 10:30:00–11:59:59
      - Lunch:    12:00:00–13:29:59
      - Afternoon:13:30:00–17:59:59
    """
    if pd.isna(ts):
        return "Unknown"

    t = ts.time()
    h, m, s = t.hour, t.minute, t.second
    seconds = h * 3600 + m * 60 + s

    # Overnight 18:00:00–23:59:59 or 00:00:00–03:59:59
    if seconds >= 18 * 3600 or seconds <= 3 * 3600 + 59 * 60 + 59:
        return "Overnight (18:00–03:59:59)"

    # Premarket 04:00:00–09:29:59
    if 4 * 3600 <= seconds <= 9 * 3600 + 29 * 60 + 59:
        return "Premarket (04:00–09:29:59)"

    # RTH IB 09:30:00–10:29:59
    if 9 * 3600 + 30 * 60 <= seconds <= 10 * 3600 + 29 * 60 + 59:
        return "RTH IB (09:30–10:29:59)"

    # RTH Morning 10:30:00–11:59:59
    if 10 * 3600 + 30 * 60 <= seconds <= 11 * 3600 + 59 * 60 + 59:
        return "RTH Morning (10:30–11:59:59)"

    # Lunch 12:00:00–13:29:59
    if 12 * 3600 <= seconds <= 13 * 3600 + 29 * 60 + 59:
        return "Lunch (12:00–13:29:59)"

    # Afternoon 13:30:00–17:59:59
    if 13 * 3600 + 30 * 60 <= seconds <= 17 * 3600 + 59 * 60 + 59:
        return "Afternoon (13:30–17:59:59)"

    return "Other"


# ========= Helpers =========

def load_trades() -> pd.DataFrame:
    """
    Load tj_trades for the current user from Supabase.
    """
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

    # Ensure numerics are numeric
    for col in ["pnl_gross", "fees", "pnl_net", "planned_risk", "r_multiple", "qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_result_col(df: pd.DataFrame) -> str | None:
    """
    Decide which column to use as 'result' of a trade.
    Priority: pnl_net -> r_multiple -> None.
    """
    if "pnl_net" in df.columns:
        return "pnl_net"
    if "r_multiple" in df.columns:
        return "r_multiple"
    return None


def add_prev_trade_info(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous trade info and time gap between trades (in minutes).
    Uses:
      entry_ts_est, exit_ts_est, side, and pnl_net (or r_multiple fallback).
    """
    if trades.empty:
        return trades

    df = trades.sort_values("entry_ts_est").copy()

    # Time relationship
    df["prev_exit_ts_est"] = df["exit_ts_est"].shift(1)
    df["mins_since_prev_exit"] = (
        (df["entry_ts_est"] - df["prev_exit_ts_est"])
        .dt.total_seconds() / 60.0
    )

    df["prev_side"] = df["side"].shift(1)

    # Determine loss basis (pnl_net first, then r_multiple)
    result_col = get_result_col(df)
    if result_col is None:
        df["prev_result"] = np.nan
        df["prev_was_loss"] = False
        return df

    df["prev_result"] = df[result_col].shift(1)
    df["prev_was_loss"] = df["prev_result"] < 0

    return df


def basic_performance_metrics(df: pd.DataFrame) -> dict:
    """
    Returns simple metrics for a trade subset.
    Uses pnl_net if available; otherwise r_multiple.
    """
    if df.empty:
        return {
            "trades": 0,
            "win_rate_%": 0.0,
            "avg_result": 0.0,
            "total_result": 0.0,
        }

    result_col = get_result_col(df)
    if result_col is None:
        return {
            "trades": len(df),
            "win_rate_%": 0.0,
            "avg_result": 0.0,
            "total_result": 0.0,
        }

    d = df[~df[result_col].isna()].copy()
    if d.empty:
        return {
            "trades": 0,
            "win_rate_%": 0.0,
            "avg_result": 0.0,
            "total_result": 0.0,
        }

    trades = len(d)
    wins = (d[result_col] > 0).sum()

    return {
        "trades": trades,
        "win_rate_%": round(100 * wins / trades, 1),
        "avg_result": round(d[result_col].mean(), 2),
        "total_result": round(d[result_col].sum(), 2),
    }


def loss_streaks_same_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds sequences of consecutive losing trades in the same direction (side).
    Uses pnl_net if available; otherwise r_multiple.
    Returns a DataFrame where each row is a streak.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "streak_id", "side", "length",
            "total_result", "avg_result", "start_time", "end_time"
        ])

    result_col = get_result_col(df)
    if result_col is None:
        return pd.DataFrame(columns=[
            "streak_id", "side", "length",
            "total_result", "avg_result", "start_time", "end_time"
        ])

    d = df.sort_values("entry_ts_est").copy()

    is_loss = d[result_col] < 0
    same_side_as_prev = d["side"] == d["side"].shift(1)

    # new streak whenever NOT (loss & prev was loss & same side)
    new_streak = ~(is_loss & is_loss.shift(1) & same_side_as_prev)

    d["streak_id"] = new_streak.cumsum()

    streaks = (
        d[is_loss]
        .groupby("streak_id")
        .agg(
            side=("side", "first"),
            length=(result_col, "size"),
            total_result=(result_col, "sum"),
            avg_result=(result_col, "mean"),
            start_time=("entry_ts_est", "first"),
            end_time=("exit_ts_est", "last"),
        )
        .reset_index()
    )

    streaks["avg_result"] = streaks["avg_result"].round(2)
    streaks["total_result"] = streaks["total_result"].round(2)

    return streaks


def build_equity_curve(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Build cumulative equity curve based on pnl_net (or r_multiple fallback).
    """
    if df.empty:
        return pd.DataFrame(columns=["entry_ts_est", "cum_result", "series"])

    result_col = get_result_col(df)
    if result_col is None:
        return pd.DataFrame(columns=["entry_ts_est", "cum_result", "series"])

    d = df.sort_values("entry_ts_est").copy()
    d = d[~d[result_col].isna()]
    if d.empty:
        return pd.DataFrame(columns=["entry_ts_est", "cum_result", "series"])

    d["cum_result"] = d[result_col].cumsum()
    d["series"] = label
    return d[["entry_ts_est", "cum_result", "series"]]


def simulate_daily_stop(df: pd.DataFrame, max_loss_per_session: float) -> pd.DataFrame:
    """
    Apply a max loss per session (trading day) to a trade stream.

    - Uses pnl_net (or r_multiple) as result.
    - Groups by entry date.
    - For each day, keeps trades in chronological order until
      cumulative result < -max_loss_per_session. The trade that
      breaches the level is kept; all later trades that day are dropped.
    """
    if df.empty or max_loss_per_session <= 0:
        return df

    result_col = get_result_col(df)
    if result_col is None:
        return df

    df_sorted = df.sort_values("entry_ts_est").copy()

    out_parts = []
    # Group by "session" = calendar day of entry
    for _, g in df_sorted.groupby(df_sorted["entry_ts_est"].dt.date):
        g = g.copy()
        cum = g[result_col].cumsum()

        breach = cum < -max_loss_per_session
        if not breach.any():
            # Never hit the stop → keep full day
            out_parts.append(g)
        else:
            # First index where we breach the stop
            first_breach_pos = int(np.argmax(breach.to_numpy()))
            # Keep up to and including that trade
            out_parts.append(g.iloc[: first_breach_pos + 1])

    if not out_parts:
        return df_sorted.iloc[0:0]

    return pd.concat(out_parts, ignore_index=True)


# ========= MAIN UI =========

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

# ---- Cooldown / Tilt analysis ----

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

# Respected = everything else (non-tilt trades)
respected_trades = tilt_df.loc[
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
    st.metric("Avg Result", all_metrics["avg_result"])
    st.metric("Total Result", all_metrics["total_result"])

with col_tilt:
    st.caption(f"‘Tilt’ trades (loss → re-entry ≤ {cooldown_minutes} min)")
    st.metric("Trades", tilt_metrics["trades"])
    st.metric("Win rate (%)", tilt_metrics["win_rate_%"])
    st.metric("Avg Result", tilt_metrics["avg_result"])
    st.metric("Total Result", tilt_metrics["total_result"])

with col_respected:
    st.caption(f"Trades respecting cooldown > {cooldown_minutes} min")
    st.metric("Trades", respected_metrics["trades"])
    st.metric("Win rate (%)", respected_metrics["win_rate_%"])
    st.metric("Avg Result", respected_metrics["avg_result"])
    st.metric("Total Result", respected_metrics["total_result"])

# ---- Max loss per session slider (after cooldown) ----

st.subheader("Session Max Loss (after Cooldown)")

result_col_for_slider = get_result_col(tilt_df)
max_loss_per_session = 0.0

if result_col_for_slider is not None:
    # Use tilt_df (all filtered trades) to calibrate slider range
    daily_sums = (
        tilt_df
        .groupby(tilt_df["entry_ts_est"].dt.date)[result_col_for_slider]
        .sum()
    )

    if not daily_sums.empty:
        worst_loss = float(daily_sums.min())  # most negative day
        # slider max: 50 or 120% of worst absolute loss, whichever is larger
        slider_max = max(50.0, abs(worst_loss) * 1.2)
    else:
        slider_max = 50.0

    max_loss_per_session = st.slider(
        "Max loss per session (after cooldown; 0 = no stop)",
        min_value=0.0,
        max_value=float(round(slider_max, 2)),
        value=0.0,
        step=50.0 if slider_max > 100 else 10.0,
    )
else:
    st.info("No numeric result column (pnl_net / r_multiple) found for daily stop simulation.")

# ---- Equity curve: With vs Without Tilt Trades / Daily Stop ----

st.markdown("### Equity Curve: With vs Without Tilt Trades / Daily Stop")

series_frames = []

# 1) All trades (no cooldown, no stop)
curve_all = build_equity_curve(tilt_df, "All trades")
if not curve_all.empty:
    series_frames.append(curve_all)

# 2) Exclude tilt trades (cooldown only)
curve_no_tilt = build_equity_curve(respected_trades, "Exclude tilt trades")
if not curve_no_tilt.empty:
    series_frames.append(curve_no_tilt)

# 3) Exclude tilt + daily max loss stop
stopped_trades = respected_trades
if max_loss_per_session > 0 and result_col_for_slider is not None:
    stopped_trades = simulate_daily_stop(respected_trades, max_loss_per_session)
    curve_stop = build_equity_curve(
        stopped_trades,
        f"Exclude tilt + stop {max_loss_per_session:.0f}"
    )
    if not curve_stop.empty:
        series_frames.append(curve_stop)

curve = pd.concat(series_frames, ignore_index=True) if series_frames else pd.DataFrame()

if not curve.empty:
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x="entry_ts_est:T",
            y="cum_result:Q",
            color="series:N",
            tooltip=["entry_ts_est:T", "cum_result:Q", "series:N"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Final result per series (metrics under the chart) ---
    final_vals = (
        curve.sort_values("entry_ts_est")
             .groupby("series")
             .tail(1)               # last point per series
             .set_index("series")["cum_result"]
    )

    cols = st.columns(len(final_vals))
    for col, (series_name, val) in zip(cols, final_vals.items()):
        col.metric(f"Final result – {series_name}", round(val, 2))

else:
    st.info("Not enough data to build an equity curve.")

# ---- Session Performance (after Cooldown & Session Max Loss) ----

st.subheader("Session Performance (after Cooldown & Session Max Loss)")

# Use the same effective trade set that feeds the "Exclude tilt + stop" path
if max_loss_per_session > 0 and result_col_for_slider is not None:
    effective_trades = stopped_trades
else:
    effective_trades = respected_trades  # cooldown only

if effective_trades.empty:
    st.info("No trades remain after applying cooldown / session max loss.")
else:
    df_sess = effective_trades.copy()
    # make sure pnl_net is numeric and non-null
    df_sess["pnl_net"] = pd.to_numeric(df_sess["pnl_net"], errors="coerce").fillna(0.0)

    df_sess["session"] = df_sess["entry_ts_est"].apply(_classify_session)
    session_stats = (
        df_sess.groupby("session")
        .agg(
            n_trades=("pnl_net", "size"),
            wins=("pnl_net", lambda s: (s > 0).sum()),
            losses=("pnl_net", lambda s: (s < 0).sum()),
            pnl=("pnl_net", "sum"),
            avg_win=(
                "pnl_net",
                lambda s: s[s > 0].mean() if (s > 0).any() else 0.0,
            ),
            avg_loss=(
                "pnl_net",
                lambda s: s[s < 0].mean() if (s < 0).any() else 0.0,
            ),
        )
        .reset_index()
    )
    session_stats["win_rate"] = session_stats.apply(
        lambda r: (r["wins"] / (r["wins"] + r["losses"]))
        if (r["wins"] + r["losses"]) > 0
        else 0.0,
        axis=1,
    )

    session_order = [
        "Overnight (18:00–03:59:59)",
        "Premarket (04:00–09:29:59)",
        "RTH IB (09:30–10:29:59)",
        "RTH Morning (10:30–11:59:59)",
        "Lunch (12:00–13:29:59)",
        "Afternoon (13:30–17:59:59)",
        "Other",
    ]
    session_stats["Session"] = pd.Categorical(
        session_stats["session"],
        categories=session_order,
        ordered=True,
    )
    session_stats["Win rate (%)"] = (session_stats["win_rate"] * 100).round(1)
    session_display = (
        session_stats.rename(
            columns={
                "n_trades": "# Trades",
                "pnl": "PnL",
                "avg_win": "Avg Win",
                "avg_loss": "Avg Loss",
            }
        )[["Session", "# Trades", "PnL", "Avg Win", "Avg Loss", "Win rate (%)"]]
        .sort_values("Session")
    )
    st.dataframe(session_display, use_container_width=False)

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
