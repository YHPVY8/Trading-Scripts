#!/usr/bin/env python3
# pages/02_Performance_Stats.py

from datetime import date, timedelta
import textwrap

import pandas as pd
import streamlit as st
import altair as alt
from supabase import create_client
import streamlit.components.v1 as components

st.set_page_config(page_title="Performance Stats (EST)", layout="wide")

# ===== CONFIG =====
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")


# ---------- Load (simple view) ----------
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = (
        sb.table("tj_trades")
        .select(
            "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,qty,"
            "pnl_gross,fees,pnl_net"
        )
        .eq("user_id", USER_ID)
        .order("entry_ts_est", desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"] = pd.to_datetime(df["exit_ts_est"])
    return df


# ===== Helper: classify session for a timestamp =====
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


# ===== Calendar renderer (HTML table – medium size, centered) =====
def _render_pnl_calendar(month_daily: pd.DataFrame, period: pd.Period):
    """
    Render a month calendar with:
      - Columns: Mo, Tu, We, Th, Fr, Week
      - Square tiles per weekday
      - Day number top-left
      - PnL + trades vertically centered
    Uses components.html so HTML is never escaped.
    """
    if month_daily.empty:
        st.info("No trades for this month.")
        return

    # Map date -> stats
    month_daily = month_daily.copy()
    month_daily["date_only"] = month_daily["trade_date"].dt.date
    by_date = month_daily.set_index("date_only")[["pnl_day", "n_trades"]].to_dict("index")

    # Calendar bounds
    first_ts = period.to_timestamp()
    year, month = first_ts.year, first_ts.month
    first_day = date(year, month, 1)

    next_month_first = date(year + (month == 12), (month % 12) + 1, 1)
    last_day = next_month_first - timedelta(days=1)

    weekday_mon0 = first_day.weekday()
    start_date = first_day - timedelta(days=weekday_mon0)

    # CSS — IMPORTANT: cal-cell stays a table cell (NOT flex!)
    css = """
    <style>
    .cal-table-wrapper {
        display: flex;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .cal-table {
        border-collapse: separate;
        border-spacing: 4px;
    }
    .cal-header-cell {
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .cal-cell {
        width: 120px;
        height: 120px;
        border-radius: 4px;
        border: 1px solid #b0b0b0;
        padding: 4px;
        vertical-align: top;
        position: relative; /* needed for absolute PnL centering */
        font-size: 0.85rem;
    }
    .cal-day-label {
        font-size: 1.0rem;
        font-weight: 600;
        color: #000;
        opacity: 0.9;
        margin-bottom: 2px;
    }

    /* NEW: Center PnL + trades vertically */
    .cal-center-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -30%);  /* visually balanced */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .cal-pnl {
        font-size: 1.15rem;
        font-weight: 700;
        color: #000;
    }
    .cal-trades {
        font-size: 0.95rem;
        opacity: 0.85;
    }
    .cal-week-summary {
        font-size: 0.95rem;
        text-align: center;
        color: #000;
    }
    .cal-empty {
        border: none;
        background-color: transparent;
    }
    </style>
    """

    # Build HTML
    rows_html = []

    header_cells = "".join(
        f"<th class='cal-header-cell'>{name}</th>"
        for name in ["Mo", "Tu", "We", "Th", "Fr", "Week"]
    )
    rows_html.append(f"<tr>{header_cells}</tr>")

    week_counter = 0

    for week_idx in range(6):
        row_start = start_date + timedelta(days=7 * week_idx)
        week_dates = [row_start + timedelta(days=d) for d in range(7)]

        # Does this row include a day of the selected month?
        has_month_day = any(d.month == month and d.year == year for d in week_dates[:5])
        if not has_month_day and row_start > last_day:
            break
        if not has_month_day:
            continue

        week_counter += 1

        in_week = month_daily[
            (month_daily["trade_date"].dt.date >= week_dates[0])
            & (month_daily["trade_date"].dt.date <= week_dates[4])
        ]
        week_pnl = float(in_week["pnl_day"].sum()) if not in_week.empty else 0.0
        week_trades = int(in_week["n_trades"].sum()) if not in_week.empty else 0

        row_cells = []

        # Mon–Fri
        for i_day in range(5):
            d = week_dates[i_day]

            if d.month != month or d.year != year:
                row_cells.append("<td class='cal-cell cal-empty'></td>")
                continue

            stats = by_date.get(d)
            pnl = stats["pnl_day"] if stats else 0.0
            n_trades = stats["n_trades"] if stats else 0

            bg_color = "#8fd98f" if pnl > 0 else "#e08b8b" if pnl < 0 else "#d0d0d0"

            cell_html = textwrap.dedent(f"""
                <td class="cal-cell" style="background-color:{bg_color};">
                    <div class="cal-day-label">{d.day}</div>

                    <div class="cal-center-content">
                        <div class="cal-pnl">${pnl:,.2f}</div>
                        <div class="cal-trades">{n_trades} trades</div>
                    </div>
                </td>
            """)
            row_cells.append(cell_html)

        # Weekly summary cell
        week_bg = "#7fcf7f" if week_pnl > 0 else "#d16f6f" if week_pnl < 0 else "#c4c4c4"
        week_color = "#004d00" if week_pnl > 0 else "#550000" if week_pnl < 0 else "#000"

        week_cell_html = textwrap.dedent(f"""
            <td class="cal-cell" style="background-color:{week_bg};">
                <div class="cal-day-label">Week {week_counter}</div>

                <div class="cal-center-content">
                    <div class="cal-week-summary" style="color:{week_color}; font-weight:700;">
                        ${week_pnl:,.2f}
                    </div>
                    <div class="cal-trades">{week_trades} trades</div>
                </div>
            </td>
        """)
        row_cells.append(week_cell_html)

        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    full_html = (
        "<div class='cal-table-wrapper'>"
        "<table class='cal-table'>"
        + "".join(rows_html)
        + "</table></div>"
    )

    components.html(css + full_html, height=620, scrolling=False)


# ========== MAIN UI ==========
st.title("Performance Stats (EST)")

df_stats = _load_trades()
if df_stats.empty:
    st.info("No trades yet.")
else:
    df_stats["pnl_net"] = pd.to_numeric(
        df_stats["pnl_net"], errors="coerce"
    ).fillna(0.0)

    # ---- High-level performance metrics ----
    wins_mask = df_stats["pnl_net"] > 0
    losses_mask = df_stats["pnl_net"] < 0
    n_wins = int(wins_mask.sum())
    n_losses = int(losses_mask.sum())

    win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0.0
    avg_win = df_stats.loc[wins_mask, "pnl_net"].mean() if n_wins > 0 else 0.0
    avg_loss = df_stats.loc[losses_mask, "pnl_net"].mean() if n_losses > 0 else 0.0
    total_pnl = df_stats["pnl_net"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win rate", f"{win_rate:.1%}")
    c2.metric("Avg Win", f"{avg_win:,.2f}")
    c3.metric("Avg Loss", f"{avg_loss:,.2f}")
    c4.metric("Total PnL", f"{total_pnl:,.2f}")

    # ---- Equity curve ----
    st.markdown("### Equity Curve (Cumulative PnL)")
    eq = df_stats.sort_values("entry_ts_est").copy()
    eq["cum_pnl"] = eq["pnl_net"].cumsum()
    eq_chart = (
        alt.Chart(eq)
        .mark_line()
        .encode(
            x=alt.X("entry_ts_est:T", title="Entry time (EST)"),
            y=alt.Y("cum_pnl:Q", title="Cumulative PnL"),
            tooltip=["entry_ts_est:T", "cum_pnl:Q"],
        )
        .properties(height=250)
    )
    st.altair_chart(eq_chart, use_container_width=True)

    # ---- Daily summary ----
    df_stats["trade_date"] = df_stats["entry_ts_est"].dt.normalize()
    daily = (
        df_stats.groupby("trade_date")
        .agg(
            pnl_day=("pnl_net", "sum"),
            n_trades=("pnl_net", "size"),
            wins=("pnl_net", lambda s: (s > 0).sum()),
            losses=("pnl_net", lambda s: (s < 0).sum()),
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
    daily["win_rate"] = daily.apply(
        lambda r: (r["wins"] / (r["wins"] + r["losses"]))
        if (r["wins"] + r["losses"]) > 0
        else 0.0,
        axis=1,
    )

    # ---- Month selector + calendar ----
    daily["month_period"] = daily["trade_date"].dt.to_period("M")
    months = sorted(daily["month_period"].unique())
    selected_period = st.selectbox(
        "Month",
        months,
        index=len(months) - 1,
        format_func=lambda p: p.strftime("%b %Y"),
    )
    month_daily = daily[daily["month_period"] == selected_period].copy()

    monthly_pnl = (
        float(month_daily["pnl_day"].sum()) if not month_daily.empty else 0.0
    )
    monthly_pnl_str = f"${monthly_pnl:,.2f}"
    monthly_color = (
        "red" if monthly_pnl < 0 else ("green" if monthly_pnl > 0 else "black")
    )
    st.markdown(
        f"<h4 style='text-align:center; margin-bottom:0.25rem;'>Monthly PnL: "
        f"<span style='color:{monthly_color};'>{monthly_pnl_str}</span></h4>",
        unsafe_allow_html=True,
    )

    _render_pnl_calendar(month_daily, selected_period)

    # ---- Daily table: # Trades, PnL, Avg Win, Avg Loss, Win rate (%), Date mm/dd/yy ----
    st.markdown("#### Daily Stats")
    if month_daily.empty:
        st.info("No trades for this month.")
    else:
        daily_display = month_daily.copy()
        daily_display["Win rate (%)"] = (daily_display["win_rate"] * 100).round(1)
        daily_display["Date"] = daily_display["trade_date"].dt.strftime("%m/%d/%y")
        daily_display = daily_display.rename(
            columns={
                "n_trades": "# Trades",
                "pnl_day": "PnL",
                "avg_win": "Avg Win",
                "avg_loss": "Avg Loss",
            }
        )[
            ["Date", "# Trades", "PnL", "Avg Win", "Avg Loss", "Win rate (%)"]
        ].sort_values("Date", ascending=False)
        st.dataframe(daily_display, use_container_width=False)

    # ---- Session stats ----
    st.markdown("### Session Performance")
    df_stats["session"] = df_stats["entry_ts_est"].apply(_classify_session)
    session_stats = (
        df_stats.groupby("session")
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

    # ---- Symbol stats ----
    st.markdown("### Symbol Performance")
    df_stats_symbol = df_stats.copy()
    df_stats_symbol["symbol"] = df_stats_symbol["symbol"].fillna("Unknown")
    symbol_stats = (
        df_stats_symbol.groupby("symbol")
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
    symbol_stats["win_rate"] = symbol_stats.apply(
        lambda r: (r["wins"] / (r["wins"] + r["losses"]))
        if (r["wins"] + r["losses"]) > 0
        else 0.0,
        axis=1,
    )
    symbol_stats["Win rate (%)"] = (symbol_stats["win_rate"] * 100).round(1)
    symbol_display = (
        symbol_stats.rename(
            columns={
                "symbol": "Symbol",
                "n_trades": "# Trades",
                "pnl": "PnL",
                "avg_win": "Avg Win",
                "avg_loss": "Avg Loss",
            }
        )[["Symbol", "# Trades", "PnL", "Avg Win", "Avg Loss", "Win rate (%)"]]
        .sort_values("Symbol")
    )
    st.dataframe(symbol_display, use_container_width=False)
