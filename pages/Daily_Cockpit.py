#!/usr/bin/env python3
"""Session-only planning, trade-log, and review workflow."""

from datetime import date, time
import hashlib
import io

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Daily Cockpit", layout="wide")

CHECKLIST_ITEMS = {
    "daily_cockpit_rest_check": "I am rested, focused, and emotionally ready to trade.",
    "daily_cockpit_calendar_check": "I reviewed the economic calendar and scheduled news.",
    "daily_cockpit_levels_check": "I marked the key overnight and RTH levels.",
    "daily_cockpit_setup_check": "I will wait for the A+ setup described in this plan.",
    "daily_cockpit_stop_check": "I will stop when the maximum daily loss is reached.",
    "daily_cockpit_rth_check": "I confirm this is RTH-only: no trading before 09:30 New York time.",
}
ANNOTATION_DEFAULTS = {
    "classification": "Base hit",
    "setup_grade": "A+",
    "planned_risk": 0.0,
    "trend_alignment": "With trend",
    "followed_plan": "Yes",
    "added_winner": "No",
    "added_loser": "No",
    "notes": "",
}


def calculate_r_multiple(net_pnl, planned_risk):
    """Return net P&L divided by positive planned risk, otherwise None."""
    try:
        pnl = float(net_pnl)
        risk = float(planned_risk)
    except (TypeError, ValueError):
        return None
    if pd.isna(pnl) or pd.isna(risk) or risk <= 0:
        return None
    return pnl / risk


def detect_rapid_reentries(trades: pd.DataFrame, minutes: int = 5) -> list[bool]:
    """Flag trades beginning 0..minutes after the preceding chronological exit."""
    if trades.empty:
        return []
    ordered = trades.sort_values("entry_time")
    previous_exit = ordered["exit_time"].shift(1)
    gaps = (ordered["entry_time"] - previous_exit).dt.total_seconds() / 60
    flags = gaps.between(0, minutes, inclusive="both")
    return flags.reindex(trades.index, fill_value=False).fillna(False).tolist()


def detect_daily_loss_violations(net_pnls, maximum_daily_loss: float) -> tuple[list[bool], list[bool]]:
    """Return flags for the first/all threshold breaches and trades after first breach."""
    values = pd.to_numeric(pd.Series(net_pnls), errors="coerce").fillna(0.0)
    if maximum_daily_loss is None or maximum_daily_loss <= 0:
        return [False] * len(values), [False] * len(values)
    violations = values.cumsum() <= -float(maximum_daily_loss)
    continued = pd.Series(False, index=values.index)
    if violations.any():
        first_position = int(violations.to_numpy().argmax())
        continued.iloc[first_position + 1 :] = True
    return violations.tolist(), continued.tolist()


def calculate_base_hit_percentage(classifications) -> float:
    """Return the percentage of annotated trades classified as base hits."""
    values = [value for value in classifications if value in ("Base hit", "Home run attempt")]
    if not values:
        return 0.0
    return 100.0 * sum(value == "Base hit" for value in values) / len(values)


def _clean_header(name: str) -> str:
    return " ".join(str(name).replace("\ufeff", "").strip().lower().split())


def _as_float(value):
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_numeric(str(value).replace(",", "").replace("$", ""), errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _parse_timestamp(value):
    """Parse platform wall-clock timestamps as America/New_York times."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return timestamp.tz_convert("America/New_York")


def parse_platform_csv(uploaded_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Normalize the Topstep/Tradovate completed-trades export used by Upload.py."""
    raw = pd.read_csv(
        io.BytesIO(uploaded_bytes), sep=None, engine="python", dtype=str, keep_default_na=False
    )
    original_headers = list(raw.columns)
    raw.columns = [_clean_header(column) for column in raw.columns]
    aliases = {
        "id": ["id", "trade id", "tradeid", "external id"],
        "symbol": ["contractname", "contract", "market", "symbol"],
        "entry_time": ["enteredat", "entry time", "entry"],
        "exit_time": ["exitedat", "exit time", "exit"],
        "qty": ["size", "quantity", "qty"],
        "side": ["type", "side"],
        "pnl_gross": ["pnl", "p&l", "profit"],
        "fees": ["fees", "fee"],
        "commissions": ["commissions", "commission"],
    }
    mapped = {}
    for canonical, candidates in aliases.items():
        mapped[canonical] = next((name for name in candidates if name in raw.columns), None)
    required = ["symbol", "entry_time", "exit_time", "side"]
    missing = [name for name in required if mapped[name] is None]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))

    normalized = pd.DataFrame(index=raw.index)
    for canonical, source in mapped.items():
        normalized[canonical] = raw[source] if source else None
    normalized["entry_time"] = normalized["entry_time"].map(_parse_timestamp)
    normalized["exit_time"] = normalized["exit_time"].map(_parse_timestamp)
    for column in ("qty", "pnl_gross", "fees", "commissions"):
        normalized[column] = normalized[column].map(_as_float)
    normalized["fees"] = normalized["fees"].fillna(0) + normalized["commissions"].fillna(0)
    normalized["net_pnl"] = normalized["pnl_gross"] - normalized["fees"]
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()
    side = normalized["side"].astype(str).str.strip().str.lower()
    normalized["direction"] = side.map(
        lambda value: "Long" if value in ("long", "buy", "b") else (
            "Short" if value in ("short", "sell", "s") else value.title()
        )
    )
    invalid = normalized["entry_time"].isna() | normalized["exit_time"].isna()
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} row(s) have invalid entry or exit timestamps.")
    debug = {"headers": original_headers, "mapped": {k: v for k, v in mapped.items() if v}}
    return normalized, debug


def group_logical_trades(rows: pd.DataFrame) -> pd.DataFrame:
    """Roll overlapping partial fills into logical trades using app rollup conventions."""
    if rows.empty:
        return pd.DataFrame()
    rows = rows.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    groups = []
    active = None
    for _, row in rows.iterrows():
        same_position = (
            active is not None
            and row["symbol"] == active["symbol"]
            and row["direction"] == active["direction"]
            and row["entry_time"] <= active["exit_time"]
        )
        if not same_position:
            if active is not None:
                groups.append(active)
            active = {
                "symbol": row["symbol"], "direction": row["direction"],
                "entry_time": row["entry_time"], "exit_time": row["exit_time"],
                "quantity": abs(row["qty"]) if pd.notna(row["qty"]) else None,
                "net_pnl": row["net_pnl"], "legs": 1,
            }
        else:
            active["exit_time"] = max(active["exit_time"], row["exit_time"])
            active["quantity"] = (active["quantity"] or 0) + (
                abs(row["qty"]) if pd.notna(row["qty"]) else 0
            )
            if pd.notna(row["net_pnl"]):
                active["net_pnl"] = (active["net_pnl"] or 0) + row["net_pnl"]
            active["legs"] += 1
    if active is not None:
        groups.append(active)
    result = pd.DataFrame(groups).sort_values("entry_time").reset_index(drop=True)
    result["trade_id"] = result.apply(
        lambda row: hashlib.sha1(
            f"{row['symbol']}|{row['direction']}|{row['entry_time']}|{row['exit_time']}".encode()
        ).hexdigest()[:12], axis=1
    )
    return result


def _start_session() -> None:
    st.session_state.daily_cockpit_started = True


st.session_state.setdefault("daily_cockpit_started", False)
st.session_state.setdefault("daily_cockpit_trades", pd.DataFrame())
st.session_state.setdefault("daily_cockpit_annotations", {})

st.title("Daily Cockpit")
st.caption("Premarket plan, trade log, and daily review. All data remains in this session.")

st.header("1. Premarket Plan")
started = st.session_state.daily_cockpit_started
col_date, col_bias = st.columns(2)
with col_date:
    trading_date = st.date_input("Trading date", value=date.today(), key="daily_cockpit_date", disabled=started)
with col_bias:
    daily_bias = st.selectbox(
        "Daily market bias", ["Bullish", "Neutral", "Bearish"], index=1,
        key="daily_cockpit_bias", disabled=started,
    )
col_plan, col_setup = st.columns(2)
with col_plan:
    premarket_plan = st.text_area("Premarket plan", height=160, key="daily_cockpit_plan", disabled=started)
with col_setup:
    a_plus_setup = st.text_area("A+ setup description", height=160, key="daily_cockpit_setup", disabled=started)
col_loss, col_size = st.columns(2)
with col_loss:
    maximum_daily_loss = st.number_input(
        "Maximum daily loss ($)", min_value=0.0, step=50.0,
        key="daily_cockpit_max_loss", disabled=started,
    )
with col_size:
    starting_position_size = st.number_input(
        "Starting position size", min_value=0, step=1,
        key="daily_cockpit_position_size", disabled=started,
    )

check_cols = st.columns(2)
checklist_complete = True
for index, (key, label) in enumerate(CHECKLIST_ITEMS.items()):
    with check_cols[index % 2]:
        checked = st.checkbox(label, key=key, disabled=started)
        checklist_complete = checklist_complete and checked

plan_complete = bool(premarket_plan.strip() and a_plus_setup.strip())
risk_valid = maximum_daily_loss > 0 and starting_position_size > 0
ready = checklist_complete and plan_complete and risk_valid
if not started:
    if not ready:
        st.warning("Complete both plans, enter positive risk limits, and confirm every checklist item.")
    st.button("Start Trading Session", type="primary", disabled=not ready, on_click=_start_session)
else:
    st.success("Trading session started. The premarket plan is locked and remains visible above.")

if started:
    st.header("2. Trade Log")
    upload = st.file_uploader("Drop Topstep/Tradovate completed-trades CSV", type=["csv"])
    if upload is not None:
        upload_digest = hashlib.sha1(upload.getvalue()).hexdigest()
        if upload_digest != st.session_state.get("daily_cockpit_upload_digest"):
            try:
                normalized, debug = parse_platform_csv(upload.getvalue())
                st.session_state.daily_cockpit_trades = group_logical_trades(normalized)
                st.session_state.daily_cockpit_upload_digest = upload_digest
                st.session_state.daily_cockpit_annotations = {}
                st.success(f"Parsed {len(normalized)} row(s) into {len(st.session_state.daily_cockpit_trades)} logical trade(s).")
                st.caption("Mapped columns: " + ", ".join(f"{v} → {k}" for k, v in debug["mapped"].items()))
            except (ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not parse CSV: {error}")

    trades = st.session_state.daily_cockpit_trades
    if trades.empty:
        st.info("Upload a supported completed-trades CSV to populate the trade log and review.")
    else:
        display = trades[["symbol", "direction", "entry_time", "exit_time", "quantity", "net_pnl", "legs"]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
        annotations = st.session_state.daily_cockpit_annotations
        for position, trade in trades.iterrows():
            trade_id = trade["trade_id"]
            annotations.setdefault(trade_id, ANNOTATION_DEFAULTS.copy())
            annotation = annotations[trade_id]
            label = f"Trade {position + 1}: {trade['symbol']} {trade['direction']} | Net P&L ${trade['net_pnl'] or 0:,.2f}"
            with st.expander(label, expanded=position == 0):
                row1 = st.columns(3)
                annotation["classification"] = row1[0].selectbox(
                    "Trade classification", ["Base hit", "Home run attempt"],
                    index=["Base hit", "Home run attempt"].index(annotation["classification"]), key=f"class_{trade_id}",
                )
                annotation["setup_grade"] = row1[1].selectbox(
                    "Setup grade", ["A+", "A", "B", "No valid setup"], key=f"grade_{trade_id}"
                )
                annotation["planned_risk"] = row1[2].number_input(
                    "Planned risk ($)", min_value=0.0, step=25.0, key=f"risk_{trade_id}"
                )
                row2 = st.columns(3)
                annotation["trend_alignment"] = row2[0].selectbox(
                    "Trend alignment", ["With trend", "Countertrend", "Unclear"], key=f"trend_{trade_id}"
                )
                annotation["followed_plan"] = row2[1].radio(
                    "Followed planned setup", ["Yes", "No"], horizontal=True, key=f"plan_{trade_id}"
                )
                annotation["added_winner"] = row2[2].radio(
                    "Added to winner", ["Yes", "No"], horizontal=True, index=1, key=f"winner_{trade_id}"
                )
                annotation["added_loser"] = st.radio(
                    "Added to loser", ["Yes", "No"], horizontal=True, index=1, key=f"loser_{trade_id}"
                )
                annotation["notes"] = st.text_area("Trade notes", key=f"notes_{trade_id}")
                r_multiple = calculate_r_multiple(trade["net_pnl"], annotation["planned_risk"])
                st.metric("R multiple", "—" if r_multiple is None else f"{r_multiple:.2f}R")
        st.session_state.daily_cockpit_annotations = annotations

        st.header("3. Daily Review")
        classifications = [annotations[row.trade_id]["classification"] for row in trades.itertuples()]
        r_values = [calculate_r_multiple(row.net_pnl, annotations[row.trade_id]["planned_risk"]) for row in trades.itertuples()]
        base_pct = calculate_base_hit_percentage(classifications)
        net_pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0)
        review = pd.DataFrame({"classification": classifications, "net_pnl": net_pnl, "r": r_values})
        base = review[review["classification"] == "Base hit"]
        home = review[review["classification"] == "Home run attempt"]
        metrics = st.columns(4)
        metrics[0].metric("Total trades", len(trades))
        metrics[1].metric("Net P&L", f"${net_pnl.sum():,.2f}")
        metrics[2].metric("Total R", f"{pd.Series(r_values, dtype=float).sum():.2f}R")
        metrics[3].metric("Base-hit %", f"{base_pct:.1f}%", delta=f"{base_pct - 80:.1f} vs target")
        summary = pd.DataFrame([
            {"Trade type": "Base hit", "Count": len(base), "Percentage": base_pct,
             "Win rate": 100 * (base["net_pnl"] > 0).mean() if len(base) else 0,
             "Average R": base["r"].dropna().mean() if len(base) else None},
            {"Trade type": "Home run attempt", "Count": len(home), "Percentage": 100 - base_pct,
             "Win rate": 100 * (home["net_pnl"] > 0).mean() if len(home) else 0,
             "Average R": home["r"].dropna().mean() if len(home) else None},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        countertrend = sum(annotations[row.trade_id]["trend_alignment"] == "Countertrend" for row in trades.itertuples())
        off_plan = sum(annotations[row.trade_id]["followed_plan"] == "No" for row in trades.itertuples())
        detail_cols = st.columns(2)
        detail_cols[0].metric("Countertrend trades", countertrend)
        detail_cols[1].metric("Did not follow planned setup", off_plan)

        rapid = detect_rapid_reentries(trades)
        loss_violations, continued = detect_daily_loss_violations(net_pnl, maximum_daily_loss)
        premarket = [timestamp.time() < time(9, 30) for timestamp in trades["entry_time"]]
        flags = []
        flags.extend(f"Rapid re-entry on trade {i + 1}" for i, flag in enumerate(rapid) if flag)
        if len(trades) > 5:
            flags.append(f"Potential overtrading: {len(trades)} grouped trades (limit 5)")
        flags.extend(f"Premarket trade {i + 1}: entry before 09:30 New York" for i, flag in enumerate(premarket) if flag)
        flags.extend(f"Daily loss violation reached on trade {i + 1}" for i, flag in enumerate(loss_violations) if flag)
        flags.extend(f"Continued trading after daily loss violation: trade {i + 1}" for i, flag in enumerate(continued) if flag)
        if base_pct < 80:
            flags.append(f"Base-hit percentage below 80% target: {base_pct:.1f}%")
        st.subheader("Automatic process flags")
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No automatic process flags detected.")
