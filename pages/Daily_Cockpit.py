#!/usr/bin/env python3
"""Session-only planning, flat-to-flat trade log, and concise daily review."""

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
    "daily_cockpit_rth_check": "RTH only: I will not trade before 09:30 New York time.",
}
ISSUES = [
    "Countertrend", "Did not follow planned setup", "Added to loser",
    "Revenge or impulse entry", "Held too long", "Exited too early", "Other",
]


def calculate_r_multiple(net_pnl, planned_risk):
    """Return net P&L divided by positive planned risk, otherwise None."""
    try:
        pnl, risk = float(net_pnl), float(planned_risk)
    except (TypeError, ValueError):
        return None
    if pd.isna(pnl) or pd.isna(risk) or risk <= 0:
        return None
    return pnl / risk


def effective_planned_risk(default_risk, override_enabled=False, override_risk=None):
    """Resolve a safe per-trade override, falling back to the session default."""
    candidate = override_risk if override_enabled else default_risk
    try:
        candidate = float(candidate)
    except (TypeError, ValueError):
        return None
    return candidate if not pd.isna(candidate) and candidate > 0 else None


def detect_rapid_reentries(trades: pd.DataFrame, minutes: int = 5) -> list[bool]:
    """Flag entries made within minutes after the prior idea returned flat."""
    if trades.empty:
        return []
    ordered = trades.sort_values("entry_time")
    gaps = (ordered["entry_time"] - ordered["exit_time"].shift()).dt.total_seconds() / 60
    flags = gaps.between(0, minutes, inclusive="both")
    return flags.reindex(trades.index, fill_value=False).fillna(False).tolist()


def detect_daily_loss_violations(net_pnls, maximum_daily_loss):
    """Return cumulative loss-threshold and post-first-breach flags."""
    values = pd.to_numeric(pd.Series(net_pnls), errors="coerce").fillna(0.0)
    if maximum_daily_loss is None or maximum_daily_loss <= 0:
        return [False] * len(values), [False] * len(values)
    violations = values.cumsum() <= -float(maximum_daily_loss)
    continued = pd.Series(False, index=values.index)
    if violations.any():
        continued.iloc[int(violations.to_numpy().argmax()) + 1 :] = True
    return violations.tolist(), continued.tolist()


def calculate_base_hit_percentage(classifications):
    values = [value for value in classifications if value in ("Base hit", "Home-run attempt")]
    return 100 * values.count("Base hit") / len(values) if values else 0.0


def _clean_header(name):
    return " ".join(str(name).replace("\ufeff", "").strip().lower().split())


def _as_float(value):
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_numeric(str(value).replace(",", "").replace("$", ""), errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _number_or_zero(value):
    return 0.0 if value is None or pd.isna(value) else float(value)


def _parse_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return timestamp.tz_convert("America/New_York")


def _trade_day(timestamp):
    """Use the application's 18:00 New York Globex trading-day roll."""
    return (timestamp + pd.Timedelta(days=1)).date() if timestamp.time() >= time(18) else timestamp.date()


def _map_columns(columns, aliases):
    return {key: next((alias for alias in options if alias in columns), None) for key, options in aliases.items()}


def parse_platform_csv(uploaded_bytes):
    """Detect and normalize either execution-level or completed-trade exports."""
    raw = pd.read_csv(io.BytesIO(uploaded_bytes), sep=None, engine="python", dtype=str, keep_default_na=False)
    original = list(raw.columns)
    raw.columns = [_clean_header(column) for column in raw.columns]
    execution_aliases = {
        "symbol": ["contract", "contractname", "symbol", "market", "instrument"],
        "timestamp": ["filled at", "filledat", "execution time", "executed at", "timestamp", "time", "date/time"],
        "side": ["action", "buy/sell", "buysell", "side"],
        "quantity": ["filled qty", "filled quantity", "quantity", "qty", "size"],
        "price": ["fill price", "execution price", "price", "avg price"],
        "net_pnl": ["net pnl", "net p&l", "realized pnl", "realized p&l", "pnl", "p&l", "profit"],
        "fees": ["fees", "fee", "commission", "commissions"],
        "order_id": ["order id", "orderid", "position id", "positionid", "execution id", "fill id"],
    }
    execution_map = _map_columns(raw.columns, execution_aliases)
    execution_required = ["symbol", "timestamp", "side", "quantity"]
    execution_missing = [field for field in execution_required if not execution_map[field]]
    if not execution_missing:
        frame = pd.DataFrame({key: raw[source] if source else None for key, source in execution_map.items()})
        frame["timestamp"] = frame["timestamp"].map(_parse_timestamp)
        frame["quantity"] = frame["quantity"].map(_as_float)
        frame["price"] = frame["price"].map(_as_float)
        frame["net_pnl"] = frame["net_pnl"].map(_as_float)
        frame["fees"] = frame["fees"].map(_as_float).fillna(0.0)
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        frame["side"] = frame["side"].astype(str).str.strip().str.lower().map(
            lambda value: "Buy" if value in ("buy", "b", "bot", "long") else (
                "Sell" if value in ("sell", "s", "sold", "short") else None
            )
        )
        invalid = frame["timestamp"].isna() | frame["quantity"].isna() | (frame["quantity"] <= 0) | frame["side"].isna()
        if invalid.any():
            raise ValueError(f"{int(invalid.sum())} execution row(s) have invalid time, side, or quantity.")
        return frame, {"mode": "execution", "headers": original, "mapped": execution_map, "missing_execution": []}

    completed_aliases = {
        "symbol": ["contractname", "contract", "market", "symbol"],
        "entry_time": ["enteredat", "entry time", "entry"],
        "exit_time": ["exitedat", "exit time", "exit"],
        "quantity": ["size", "quantity", "qty"],
        "direction": ["type", "side"],
        "net_pnl": ["net pnl", "net p&l"],
        "pnl_gross": ["pnl", "p&l", "profit"],
        "fees": ["fees", "fee"],
        "commissions": ["commissions", "commission"],
    }
    completed_map = _map_columns(raw.columns, completed_aliases)
    completed_required = ["symbol", "entry_time", "exit_time", "direction"]
    completed_missing = [field for field in completed_required if not completed_map[field]]
    if completed_missing:
        raise ValueError(
            "Execution reconstruction requires symbol, timestamp, side, and quantity; missing: "
            + ", ".join(execution_missing)
            + ". Completed-trade fallback also missing: " + ", ".join(completed_missing)
        )
    frame = pd.DataFrame({key: raw[source] if source else None for key, source in completed_map.items()})
    frame["entry_time"] = frame["entry_time"].map(_parse_timestamp)
    frame["exit_time"] = frame["exit_time"].map(_parse_timestamp)
    for column in ("quantity", "net_pnl", "pnl_gross", "fees", "commissions"):
        frame[column] = frame[column].map(_as_float)
    frame["fees"] = frame["fees"].fillna(0) + frame["commissions"].fillna(0)
    frame["net_pnl"] = frame["net_pnl"].where(frame["net_pnl"].notna(), frame["pnl_gross"] - frame["fees"])
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["direction"] = frame["direction"].astype(str).str.strip().str.lower().map(
        lambda value: "Long" if value in ("long", "buy", "b") else (
            "Short" if value in ("short", "sell", "s") else None
        )
    )
    invalid = frame["entry_time"].isna() | frame["exit_time"].isna() | frame["direction"].isna()
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} completed-trade row(s) have invalid entry, exit, or direction.")
    return frame, {
        "mode": "completed", "headers": original, "mapped": completed_map,
        "missing_execution": execution_missing,
    }


def _new_idea(symbol, timestamp, signed_qty, price, net_pnl, fees):
    quantity = abs(signed_qty)
    is_buy = signed_qty > 0
    return {
        "symbol": symbol, "direction": "Long" if is_buy else "Short",
        "entry_time": timestamp, "exit_time": pd.NaT, "trade_day": _trade_day(timestamp),
        "initial_quantity": quantity, "maximum_quantity": quantity,
        "total_bought": quantity if is_buy else 0.0, "total_sold": quantity if not is_buy else 0.0,
        "entry_value": quantity * price if pd.notna(price) else 0.0,
        "entry_price_qty": quantity if pd.notna(price) else 0.0,
        "exit_value": 0.0, "exit_price_qty": 0.0,
        "net_pnl": _number_or_zero(net_pnl) - _number_or_zero(fees), "executions": 1,
        "adds": 0, "partial_exits": 0, "position": signed_qty,
        "crossed_session_boundary": False,
    }


def reconstruct_flat_to_flat_trades(executions: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct per-instrument ideas with running signed position quantity."""
    if executions.empty:
        return pd.DataFrame()
    active, finished = {}, []
    for row in executions.sort_values("timestamp").itertuples(index=False):
        delta = abs(float(row.quantity)) * (1 if row.side == "Buy" else -1)
        idea = active.get(row.symbol)
        if idea is None:
            active[row.symbol] = _new_idea(row.symbol, row.timestamp, delta, row.price, row.net_pnl, row.fees)
            continue
        position = idea["position"]
        if _trade_day(row.timestamp) != idea["trade_day"]:
            idea["crossed_session_boundary"] = True
        same_direction = position * delta > 0
        if same_direction:
            idea["adds"] += 1
            idea["executions"] += 1
            idea["position"] += delta
            idea["maximum_quantity"] = max(idea["maximum_quantity"], abs(idea["position"]))
            quantity = abs(delta)
            if delta > 0:
                idea["total_bought"] += quantity
            else:
                idea["total_sold"] += quantity
            if pd.notna(row.price):
                idea["entry_value"] += quantity * row.price
                idea["entry_price_qty"] += quantity
            idea["net_pnl"] += _number_or_zero(row.net_pnl) - _number_or_zero(row.fees)
            continue

        close_qty = min(abs(position), abs(delta))
        remainder = abs(delta) - close_qty
        idea["executions"] += 1
        if close_qty < abs(position):
            idea["partial_exits"] += 1
        if delta > 0:
            idea["total_bought"] += close_qty
        else:
            idea["total_sold"] += close_qty
        if pd.notna(row.price):
            idea["exit_value"] += close_qty * row.price
            idea["exit_price_qty"] += close_qty
        idea["net_pnl"] += _number_or_zero(row.net_pnl) - _number_or_zero(row.fees)
        idea["position"] += close_qty * (1 if delta > 0 else -1)
        if abs(idea["position"]) < 1e-9:
            idea["position"] = 0.0
            idea["exit_time"] = row.timestamp
            finished.append(idea)
            active.pop(row.symbol)
            if remainder > 0:
                reversed_delta = remainder * (1 if delta > 0 else -1)
                active[row.symbol] = _new_idea(row.symbol, row.timestamp, reversed_delta, row.price, 0.0, 0.0)
        # otherwise the execution was only a partial exit

    for idea in active.values():
        idea["open_position"] = True
        finished.append(idea)
    if not finished:
        return pd.DataFrame()
    result = pd.DataFrame(finished)
    if "open_position" not in result:
        result["open_position"] = False
    else:
        result["open_position"] = result["open_position"].fillna(False)
    result["weighted_entry_price"] = result["entry_value"] / result["entry_price_qty"].replace(0, pd.NA)
    result["weighted_exit_price"] = result["exit_value"] / result["exit_price_qty"].replace(0, pd.NA)
    result["trade_id"] = result.apply(lambda row: hashlib.sha1(
        f"{row['symbol']}|{row['direction']}|{row['entry_time']}|{row['exit_time']}".encode()
    ).hexdigest()[:12], axis=1)
    return result.sort_values("entry_time").reset_index(drop=True)


def completed_trade_fallback(rows: pd.DataFrame) -> pd.DataFrame:
    """Safely preserve completed rows as trades; never infer overlapping positions."""
    result = rows.sort_values("entry_time").reset_index(drop=True).copy()
    result["initial_quantity"] = result["quantity"].abs()
    result["maximum_quantity"] = result["quantity"].abs()
    result["total_bought"] = pd.NA
    result["total_sold"] = pd.NA
    result["weighted_entry_price"] = pd.NA
    result["weighted_exit_price"] = pd.NA
    result["executions"] = 1
    result["adds"] = pd.NA
    result["partial_exits"] = pd.NA
    result["open_position"] = False
    result["crossed_session_boundary"] = False
    result["trade_day"] = result["entry_time"].map(_trade_day)
    result["trade_id"] = result.apply(lambda row: hashlib.sha1(
        f"{row['symbol']}|{row['direction']}|{row['entry_time']}|{row['exit_time']}".encode()
    ).hexdigest()[:12], axis=1)
    return result


def _start_session():
    st.session_state.daily_cockpit_started = True


def _mark_classification_reviewed(trade_id):
    st.session_state.daily_cockpit_annotations[trade_id]["classification_reviewed"] = True


st.session_state.setdefault("daily_cockpit_started", False)
st.session_state.setdefault("daily_cockpit_trades", pd.DataFrame())
st.session_state.setdefault("daily_cockpit_annotations", {})
st.session_state.setdefault("daily_cockpit_import_mode", None)

st.title("Daily Cockpit")
st.caption("Plan, review flat-to-flat trade ideas, and focus manual input on exceptions.")
st.header("1. Premarket Plan")
started = st.session_state.daily_cockpit_started
col_date, col_bias = st.columns(2)
with col_date:
    trading_date = st.date_input("Trading date", value=date.today(), key="daily_cockpit_date", disabled=started)
with col_bias:
    daily_bias = st.selectbox("Daily market bias", ["Bullish", "Neutral", "Bearish"], index=1, key="daily_cockpit_bias", disabled=started)
col_plan, col_setup = st.columns(2)
with col_plan:
    premarket_plan = st.text_area("Premarket plan", height=140, key="daily_cockpit_plan", disabled=started)
with col_setup:
    a_plus_setup = st.text_area("A+ setup description", height=140, key="daily_cockpit_setup", disabled=started)
risk_cols = st.columns(3)
with risk_cols[0]:
    maximum_daily_loss = st.number_input("Maximum daily loss ($)", min_value=0.0, step=50.0, key="daily_cockpit_max_loss", disabled=started)
with risk_cols[1]:
    starting_position_size = st.number_input("Starting position size", min_value=0, step=1, key="daily_cockpit_position_size", disabled=started)
with risk_cols[2]:
    default_planned_risk = st.number_input("Default planned risk per trade ($)", min_value=0.0, step=25.0, key="daily_cockpit_default_risk", disabled=started)
check_cols = st.columns(2)
checklist_complete = True
for index, (key, label) in enumerate(CHECKLIST_ITEMS.items()):
    with check_cols[index % 2]:
        checklist_complete = st.checkbox(label, key=key, disabled=started) and checklist_complete
ready = checklist_complete and bool(premarket_plan.strip() and a_plus_setup.strip()) and all(
    value > 0 for value in (maximum_daily_loss, starting_position_size, default_planned_risk)
)
if not started:
    if not ready:
        st.warning("Complete both plans, positive session risk fields, and every checklist item.")
    st.button("Start Trading Session", type="primary", disabled=not ready, on_click=_start_session)
else:
    st.success("Session started. The plan is locked and remains visible.")

if started:
    st.header("2. Trade Log")
    upload = st.file_uploader("Drop Topstep/Tradovate CSV", type=["csv"])
    if upload is not None:
        digest = hashlib.sha1(upload.getvalue()).hexdigest()
        if digest != st.session_state.get("daily_cockpit_upload_digest"):
            try:
                normalized, details = parse_platform_csv(upload.getvalue())
                if details["mode"] == "execution":
                    trades = reconstruct_flat_to_flat_trades(normalized)
                else:
                    trades = completed_trade_fallback(normalized)
                st.session_state.daily_cockpit_trades = trades
                st.session_state.daily_cockpit_annotations = {}
                st.session_state.daily_cockpit_upload_digest = digest
                st.session_state.daily_cockpit_import_mode = details["mode"]
                st.session_state.daily_cockpit_missing_execution = details["missing_execution"]
            except (ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not parse CSV: {error}")
    mode = st.session_state.daily_cockpit_import_mode
    if mode == "execution":
        st.success("Execution-level export detected. Trades were reconstructed flat-to-flat by instrument.")
    elif mode == "completed":
        missing = ", ".join(st.session_state.get("daily_cockpit_missing_execution", []))
        st.warning(
            "Completed-trade export detected, not execution-level data. Each completed row is preserved as one trade; "
            f"no overlap grouping was inferred. Missing execution fields: {missing}."
        )

    trades = st.session_state.daily_cockpit_trades
    if trades.empty:
        st.info("Upload a supported CSV to populate the trade log.")
    else:
        annotations = st.session_state.daily_cockpit_annotations
        for position, trade in trades.iterrows():
            trade_id = trade["trade_id"]
            annotations.setdefault(trade_id, {
                "classification": "Base hit", "classification_reviewed": False,
                "override_enabled": False, "override_risk": default_planned_risk,
                "issues": [], "notes": "",
            })
            annotation = annotations[trade_id]
            risk = effective_planned_risk(default_planned_risk, annotation["override_enabled"], annotation["override_risk"])
            r_multiple = calculate_r_multiple(trade["net_pnl"], risk)
            with st.container(border=True):
                title_col, class_col = st.columns([2, 1])
                with title_col:
                    st.markdown(f"**Trade {position + 1} · {trade['symbol']} · {trade['direction']}**")
                    st.caption(f"{trade['entry_time']:%Y-%m-%d %H:%M:%S} → {trade['exit_time']:%H:%M:%S}" if pd.notna(trade["exit_time"]) else f"{trade['entry_time']:%Y-%m-%d %H:%M:%S} → OPEN")
                with class_col:
                    annotation["classification"] = st.radio(
                        "Classification", ["Base hit", "Home-run attempt"], horizontal=True,
                        key=f"class_{trade_id}", on_change=_mark_classification_reviewed, args=(trade_id,),
                    )
                    if not annotation["classification_reviewed"]:
                        st.caption("Base hit is the unreviewed default")
                card = st.columns(6)
                card[0].metric("Initial qty", f"{trade['initial_quantity']:g}")
                card[1].metric("Max qty", f"{trade['maximum_quantity']:g}")
                card[2].metric("Executions", int(trade["executions"]))
                card[3].metric("Net P&L", "—" if pd.isna(trade["net_pnl"]) else f"${trade['net_pnl']:,.2f}")
                card[4].metric("Risk", "—" if risk is None else f"${risk:,.2f}")
                card[5].metric("R", "—" if r_multiple is None else f"{r_multiple:.2f}R")
                annotation["override_enabled"] = st.checkbox("Edit planned risk", key=f"override_{trade_id}")
                if annotation["override_enabled"]:
                    annotation["override_risk"] = st.number_input("Trade planned risk ($)", min_value=0.0, step=25.0, value=float(annotation["override_risk"]), key=f"override_risk_{trade_id}")
                with st.expander("Add issue"):
                    selected = []
                    issue_cols = st.columns(2)
                    for issue_index, issue in enumerate(ISSUES):
                        if issue_cols[issue_index % 2].checkbox(issue, key=f"issue_{trade_id}_{issue}"):
                            selected.append(issue)
                    annotation["issues"] = selected
                    add_notes = st.checkbox("Add notes", key=f"add_notes_{trade_id}")
                    if "Other" in selected or add_notes:
                        annotation["notes"] = st.text_area("Notes", key=f"notes_{trade_id}")
        st.session_state.daily_cockpit_annotations = annotations

        st.header("3. Daily Review")
        classifications = [annotations[row.trade_id]["classification"] for row in trades.itertuples()]
        risks = [effective_planned_risk(default_planned_risk, annotations[row.trade_id]["override_enabled"], annotations[row.trade_id]["override_risk"]) for row in trades.itertuples()]
        r_values = [calculate_r_multiple(row.net_pnl, risk) for row, risk in zip(trades.itertuples(), risks)]
        net_pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0)
        base_pct = calculate_base_hit_percentage(classifications)
        headline = st.columns(5)
        headline[0].metric("Flat-to-flat ideas", len(trades))
        headline[1].metric("Net P&L", f"${net_pnl.sum():,.2f}")
        headline[2].metric("Total R", f"{pd.Series(r_values, dtype=float).sum():.2f}R")
        headline[3].metric("Base-hit %", f"{base_pct:.1f}%", f"{base_pct - 80:.1f} vs target")
        headline[4].metric("Home-run attempts", classifications.count("Home-run attempt"))
        for classification in ("Base hit", "Home-run attempt"):
            values = [r for r, label in zip(r_values, classifications) if label == classification and r is not None]
            st.caption(f"Average R · {classification}: " + (f"{sum(values) / len(values):.2f}R" if values else "—"))

        rapid = detect_rapid_reentries(trades)
        loss_breach, continued = detect_daily_loss_violations(net_pnl, maximum_daily_loss)
        flags = []
        flags += [f"Trade {i + 1}: rapid re-entry within 5 minutes after flat" for i, flag in enumerate(rapid) if flag]
        short_ideas = detect_rapid_reentries(trades, minutes=15)
        if sum(short_ideas) >= 2:
            flags.append(f"Multiple new trade ideas in short succession: {sum(short_ideas)} within 15 minutes")
        if len(trades) > 5:
            flags.append(f"Potential overtrading: {len(trades)} ideas exceeds 5")
        flags += [f"Trade {i + 1}: position size {row.maximum_quantity:g} exceeded starting-size plan {starting_position_size:g}" for i, row in enumerate(trades.itertuples()) if row.maximum_quantity > starting_position_size]
        flags += [f"Trade {i + 1}: premarket entry before 09:30 New York" for i, timestamp in enumerate(trades["entry_time"]) if timestamp.time() < time(9, 30)]
        flags += [f"Daily-loss threshold reached on trade {i + 1}" for i, flag in enumerate(loss_breach) if flag]
        flags += [f"Trade {i + 1}: continued trading after daily-loss breach" for i, flag in enumerate(continued) if flag]
        if base_pct < 80:
            flags.append(f"Base-hit percentage {base_pct:.1f}% is below the 80% target")
        manual = [(i + 1, issue) for i, row in enumerate(trades.itertuples()) for issue in annotations[row.trade_id]["issues"]]
        flags += [f"Trade {number}: {issue}" for number, issue in manual]
        if mode == "execution":
            open_count = int(trades["open_position"].sum())
            if open_count:
                flags.append(f"{open_count} instrument position(s) did not return flat in the export")
            crossed = int(trades["crossed_session_boundary"].sum())
            if crossed:
                flags.append(f"{crossed} position(s) remained open across the 18:00 New York session boundary")
            st.caption(
                f"Execution detail: {int(trades['adds'].sum())} add(s), "
                f"{int(trades['partial_exits'].sum())} partial exit(s)."
            )
        st.subheader("Exceptions and automatic flags")
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No exceptions or automatic process flags detected.")
