#!/usr/bin/env python3
import io
import re
import hashlib
from datetime import timedelta, date

import pandas as pd
import streamlit as st
from supabase import create_client
import altair as alt

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ===== CONFIG =====
USE_USER_SCOPING = True  # your DB requires user_id NOT NULL
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ---------- Utilities ----------
def _clean_header(name: str) -> str:
    if name is None:
        return ""
    s = str(name).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s.lower()


def _to_iso_est(ts_str: str):
    """Keep EST wall-clock: strip trailing timezone token like '-03:00' if present; return 'YYYY-MM-DD HH:MM:SS' or None."""
    if not ts_str:
        return None
    s = str(ts_str).strip()
    parts = s.rsplit(" ", 1)
    if (
        len(parts) == 2
        and len(parts[1]) == 6
        and parts[1][3] == ":"
        and (parts[1][0] in "+-")
    ):
        s = parts[0]
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _synth_id(row: dict) -> str:
    key = "|".join(
        [
            str(row.get("contractname", "")).strip().upper(),
            str(row.get("enteredat", "")).strip(),
            str(row.get("exitedat", "")).strip(),
            str(row.get("entryprice", "")).strip(),
            str(row.get("exitprice", "")).strip(),
            str(row.get("size", "")).strip(),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"SYN-{digest}"


def _as_float(x):
    if x is None or x == "":
        return None
    x_str = str(x).replace(",", "").strip()
    try:
        return float(x_str)
    except Exception:
        v = pd.to_numeric(x_str, errors="coerce")
        return None if pd.isna(v) else float(v)


# ---------- CSV -> canonical rows ----------
def _read_csv_to_rows(uploaded_bytes: bytes):
    df = pd.read_csv(
        io.BytesIO(uploaded_bytes),
        sep=None,  # auto-detect comma or tab
        engine="python",
        dtype=str,
        keep_default_na=False,  # keep empty cells as "", not NaN
    )
    original_headers = list(df.columns)
    norm_headers = [_clean_header(h) for h in original_headers]
    df.columns = norm_headers

    alias_map = {
        "id": "id",
        "trade id": "id",
        "tradeid": "id",
        "external id": "id",
        "contractname": "contractname",
        "contract": "contractname",
        "market": "contractname",
        "symbol": "contractname",
        "enteredat": "enteredat",
        "entry time": "enteredat",
        "entry": "enteredat",
        "exitedat": "exitedat",
        "exit time": "exitedat",
        "exit": "exitedat",
        "entryprice": "entryprice",
        "entry price": "entryprice",
        "exitprice": "exitprice",
        "exit price": "exitprice",
        "size": "size",
        "quantity": "size",
        "qty": "size",
        "type": "type",
        "side": "type",
        "pnl": "pnl",
        "p&l": "pnl",
        "profit": "pnl",
        "fees": "fees",
        "fee": "fees",
        "commissions": "commissions",
        "commission": "commissions",
        "tradeday": "tradeday",
        "tradeduration": "tradeduration",
    }

    canonical_cols = {}
    for src, canon in alias_map.items():
        if src in df.columns and canon not in canonical_cols:
            canonical_cols[canon] = src

    rows = []
    for _, r in df.iterrows():
        row = {}
        for canon, src in canonical_cols.items():
            row[canon] = r.get(src, "")
        rows.append(row)

    debug = {
        "original_headers": original_headers,
        "normalized_headers": norm_headers,
        "canonical_mapped_pairs": sorted(
            [(k, v) for k, v in canonical_cols.items()]
        ),
        "row_count": len(rows),
    }
    return rows, debug


# ---------- insert-or-update (manual upsert keyed by user_id + external_trade_id) ----------
def _insert_or_update_trade(payload: dict):
    q = (
        sb.table("tj_trades")
        .select("id", count="exact")
        .eq("external_trade_id", payload["external_trade_id"])
        .eq("user_id", payload["user_id"])
        .limit(1)
    )
    res = q.execute()
    existing = res.data or []
    if existing:
        tid = existing[0]["id"]
        sb.table("tj_trades").update(payload).eq("id", tid).execute()
        return "updated", tid
    else:
        r = sb.table("tj_trades").insert(payload).execute()
        tid = (r.data or [{}])[0].get("id")
        return "inserted", tid


# ---------- Upsert loop ----------
def _upsert_trades_from_rows(rows):
    inserted = skipped = 0
    errs, samples = [], []
    inserted_external_ids = []
    inserted_trade_ids = []

    for r in rows:
        try:
            ext_id = (r.get("id") or "").strip() or _synth_id(r)
            t = (r.get("type", "") or "").strip().lower()
            side = (
                "long"
                if t in ("long", "buy", "b")
                else ("short" if t in ("short", "sell", "s") else None)
            )

            qty_val = r.get("size")
            symbol = (r.get("contractname") or "").strip().upper() or None

            f1 = _as_float(r.get("fees")) or 0.0
            f2 = _as_float(r.get("commissions")) or 0.0
            fees_val = float(f1) + float(f2)

            entry_iso = _to_iso_est(r.get("enteredat"))
            exit_iso = _to_iso_est(r.get("exitedat"))
            if not entry_iso:
                raise ValueError(f"Missing/Bad EnteredAt: {r.get('enteredat')}")

            payload = {
                "user_id": USER_ID,  # <- REQUIRED (tj_trades.user_id NOT NULL)
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": entry_iso,
                "exit_ts_est": exit_iso,  # may be None for open trades
                "entry_px": _as_float(r.get("entryprice")),
                "exit_px": _as_float(r.get("exitprice")),
                "qty": _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees": fees_val,
                "source": "csv",
            }

            if len(samples) < 5:
                samples.append(dict(payload))

            status, tid = _insert_or_update_trade(payload)
            if status in ("inserted", "updated"):
                inserted += 1
                inserted_external_ids.append(ext_id)
                if tid:
                    inserted_trade_ids.append(tid)
            else:
                skipped += 1

        except Exception as e:
            skipped += 1
            if len(errs) < 25:
                errs.append(
                    {
                        "id": r.get("id"),
                        "enteredat": r.get("enteredat"),
                        "exitedat": r.get("exitedat"),
                        "error": str(e),
                    }
                )

    return inserted, skipped, errs, samples, inserted_external_ids, inserted_trade_ids


# ---------- Load ----------
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = (
        sb.table("tj_trades")
        .select(
            "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
            "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
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


def _fetch_tags_for(trade_ids):
    if not trade_ids:
        return {}
    data = (
        sb.table("tj_trade_tags")
        .select("trade_id,tag")
        .in_("trade_id", trade_ids)
        .eq("user_id", USER_ID)
        .execute()
        .data
    )
    tagmap = {}
    for r in data or []:
        tagmap.setdefault(r["trade_id"], []).append(r["tag"])
    return tagmap


def _save_comments(trade_ids, body):
    if not body or not trade_ids:
        return
    rows = [{"trade_id": tid, "user_id": USER_ID, "body": body} for tid in trade_ids]
    sb.table("tj_trade_comments").insert(rows).execute()


def _save_tags(trade_ids, tags):
    """Robust tag save:
       1) Try upsert on (trade_id,tag,user_id)
       2) Fallback to (trade_id,tag)
       3) Fallback to manual dedupe: fetch existing then insert only missing
    """
    if not tags or not trade_ids:
        return

    rows = [
        {"trade_id": tid, "user_id": USER_ID, "tag": t}
        for tid in trade_ids
        for t in tags
    ]
    # Attempt #1
    try:
        sb.table("tj_trade_tags").upsert(
            rows, on_conflict="trade_id,tag,user_id"
        ).execute()
        st.toast("Tags saved (conflict: trade_id,tag,user_id)")
        return
    except Exception:
        pass
    # Attempt #2
    try:
        sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag").execute()
        st.toast("Tags saved (conflict: trade_id,tag)")
        return
    except Exception:
        pass
    # Attempt #3: manual dedupe
    try:
        existing = (
            sb.table("tj_trade_tags")
            .select("trade_id,tag")
            .in_("trade_id", trade_ids)
            .eq("user_id", USER_ID)
            .execute()
            .data
            or []
        )
        existing_set = {(e["trade_id"], e["tag"]) for e in existing}
        to_insert = [r for r in rows if (r["trade_id"], r["tag"]) not in existing_set]
        if to_insert:
            sb.table("tj_trade_tags").insert(to_insert).execute()
            st.toast(f"Tags saved (manual insert {len(to_insert)})")
        else:
            st.toast("Tags already present")
    except Exception as e:
        st.error(f"Saving tags failed: {e}")


def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    """Create a group if name provided; then attach trade_ids."""
    if new_group_name:
        g = (
            sb.table("tj_trade_groups")
            .insert({"user_id": USER_ID, "name": new_group_name, "notes": notes})
            .execute()
            .data[0]
        )
        group_id = g["id"]
    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        try:
            sb.table("tj_trade_group_members").upsert(
                rows, on_conflict="group_id,trade_id"
            ).execute()
        except Exception:
            (
                sb.table("tj_trade_group_members")
                .delete()
                .in_("trade_id", trade_ids)
                .eq("group_id", group_id)
                .execute()
            )
            sb.table("tj_trade_group_members").insert(rows).execute()
    return group_id


def _remove_from_groups(trade_ids):
    """Ungroup: remove selected trades from any groups they belong to."""
    if not trade_ids:
        return
    (
        sb.table("tj_trade_group_members")
        .delete()
        .in_("trade_id", trade_ids)
        .execute()
    )


def _get_groups():
    res = (
        sb.table("tj_trade_groups")
        .select("id,name,notes,created_at")
        .eq("user_id", USER_ID)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def _next_group_name():
    """Auto-increment group name like 'Group 1', 'Group 2', ..."""
    groups = _get_groups()
    nums = []
    for g in groups:
        name = (g.get("name") or "").strip()
        m = re.search(r"(\d+)$", name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    n = (max(nums) + 1) if nums else 1
    return f"Group {n}"


# ===== Group collapsed helpers =====
def _fetch_all_groups_with_members():
    """
    Returns (groups, members_df).
    groups: list of group rows {id,name,notes,created_at}
    members_df: DataFrame of joined group members with trade columns for this USER_ID
    """
    groups = (
        sb.table("tj_trade_groups")
        .select("id,name,notes,created_at")
        .eq("user_id", USER_ID)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )

    mem = (
        sb.table("tj_trade_group_members")
        .select("group_id, trade_id, tj_trades(*)")
        .execute()
        .data
        or []
    )

    rows = []
    for m in mem:
        t = m.get("tj_trades") or {}
        if t and t.get("user_id") == USER_ID:
            t["group_id"] = m["group_id"]
            rows.append(t)

    df = pd.DataFrame(rows or [])
    if not df.empty:
        for c in ("entry_ts_est", "exit_ts_est"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        for c in ("qty", "entry_px", "exit_px", "pnl_gross", "fees", "pnl_net"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return groups, df


def _rollup_by_group(members_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse trades to one row per group_id with:
    group_id, first_entry, last_exit, legs, total_qty, vwap_entry, vwap_exit,
    symbol(mode), side(mode), pnl_net_sum
    """
    if members_df.empty:
        return pd.DataFrame()

    def _mode(series):
        s = series.dropna()
        return s.mode().iloc[0] if not s.empty else None

    out = []
    for gid, gdf in members_df.groupby("group_id"):
        legs = len(gdf)
        qty = gdf["qty"].fillna(0).abs()

        # VWAP Entry
        ok_e = gdf["entry_px"].notna()
        w_entry = qty.where(ok_e, 0)
        denom_e = w_entry.sum()
        vwap_entry = None
        if denom_e and denom_e != 0:
            vwap_entry = float(
                (gdf["entry_px"].where(ok_e, 0) * qty).sum() / denom_e
            )

        # VWAP Exit
        ok_x = gdf["exit_px"].notna()
        w_exit = qty.where(ok_x, 0)
        denom_x = w_exit.sum()
        vwap_exit = None
        if denom_x and denom_x != 0:
            vwap_exit = float(
                (gdf["exit_px"].where(ok_x, 0) * qty).sum() / denom_x
            )

        first_entry = gdf["entry_ts_est"].min()
        last_exit = gdf["exit_ts_est"].max()
        total_qty = float(gdf["qty"].fillna(0).sum())

        # Net PnL
        pnl_sum = float(
            (gdf["pnl_net"].fillna(0) if "pnl_net" in gdf else 0).sum()
        )
        if pnl_sum == 0 and ("pnl_gross" in gdf or "fees" in gdf):
            pnl_sum = float(
                gdf["pnl_gross"].fillna(0).sum()
                - gdf["fees"].fillna(0).sum()
            )

        out.append(
            {
                "group_id": gid,
                "first_entry": first_entry,
                "last_exit": last_exit,
                "legs": int(legs),
                "total_qty": total_qty,
                "vwap_entry": vwap_entry,
                "vwap_exit": vwap_exit,
                "symbol": _mode(gdf["symbol"]) if "symbol" in gdf else None,
                "side": _mode(gdf["side"]) if "side" in gdf else None,
                "pnl_net_sum": pnl_sum,
            }
        )
    return pd.DataFrame(out)


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


# ===== Calendar renderer (medium balanced size) =====
def _render_pnl_calendar(month_daily: pd.DataFrame, period: pd.Period):
    """
    Render a month calendar with:
      - Columns: Mo, Tu, We, Th, Fr, Week
      - One square tile per weekday in the month with PnL + # trades
      - Week column is also a square tile with weekly PnL + # trades
    """
    if month_daily.empty:
        st.info("No trades for this month.")
        return

    # Map date -> row dict
    month_daily = month_daily.copy()
    month_daily["date_only"] = month_daily["trade_date"].dt.date
    by_date = month_daily.set_index("date_only")[["pnl_day", "n_trades"]].to_dict(
        "index"
    )

    # Calendar bounds
    first_ts = period.to_timestamp()  # first day of month
    year, month = first_ts.year, first_ts.month
    first_day = date(year, month, 1)
    # compute last day manually
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)

    # Start on the Monday on/before first_day
    weekday_mon0 = first_day.weekday()  # Mon=0..Sun=6
    start_date = first_day - timedelta(days=weekday_mon0)

    # Calendar CSS (square tiles, medium size, black font)
    st.markdown(
        """
        <style>
        .cal-wrapper {
            max-width: 420px;   /* medium balanced width: tweak this number */
            margin: 0 auto;
        }
        .cal-cell {
            border: 1px solid #b0b0b0;
            border-radius: 4px;
            padding: 2px 2px;
            aspect-ratio: 1 / 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .cal-day-label {
            font-size: 0.85rem;
            color: #000000;
            opacity: 0.8;
        }
        .cal-pnl {
            font-size: 0.85rem;
            font-weight: 700;
            color: #000000;
        }
        .cal-trades {
            font-size: 0.75rem;
            color: #000000;
            opacity: 0.85;
        }
        .cal-week-summary {
            font-size: 0.8rem;
            color: #000000;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wrap calendar in width-limited div
    st.markdown("<div class='cal-wrapper'>", unsafe_allow_html=True)

    header_cols = st.columns(6)
    for col, name in zip(header_cols, ["Mo", "Tu", "We", "Th", "Fr", "Week"]):
        col.markdown(
            f"<div style='text-align:center; font-weight:600;'>{name}</div>",
            unsafe_allow_html=True,
        )

    week_counter = 0
    # up to 6 rows (weeks)
    for week_idx in range(6):
        row_start = start_date + timedelta(days=7 * week_idx)
        week_dates = [row_start + timedelta(days=d) for d in range(7)]  # Mon..Sun

        # Does this week touch the selected month (Mon–Fri)?
        has_month_day = any(
            (d.month == month and d.year == year) for d in week_dates[:5]
        )
        if not has_month_day and row_start > last_day:
            break
        if not has_month_day:
            continue
        week_counter += 1

        # Weekly totals only consider Mon–Fri in this month
        in_week = month_daily[
            (month_daily["trade_date"].dt.date >= week_dates[0])
            & (month_daily["trade_date"].dt.date <= week_dates[4])
        ]
        week_pnl = float(in_week["pnl_day"].sum()) if not in_week.empty else 0.0
        week_trades = int(in_week["n_trades"].sum()) if not in_week.empty else 0

        cols = st.columns(6)
        # Mon–Fri cells
        for i_day in range(5):
            col = cols[i_day]
            d = week_dates[i_day]  # Mon..Fri

            if d.month != month or d.year != year:
                col.markdown("&nbsp;", unsafe_allow_html=True)
                continue

            stats = by_date.get(d)
            pnl = stats["pnl_day"] if stats else 0.0
            n_trades = stats["n_trades"] if stats else 0

            # Darker green/red/flat backgrounds
            if pnl > 0:
                bg_color = "#8fd98f"
            elif pnl < 0:
                bg_color = "#e08b8b"
            else:
                bg_color = "#d0d0d0"

            pnl_str = f"${pnl:,.2f}"
            trades_str = f"{int(n_trades)} trades"

            html = f"""
            <div class="cal-cell" style="background-color:{bg_color};">
                <div class="cal-day-label">{d.day}</div>
                <div style="text-align:center;">
                    <div class="cal-pnl">{pnl_str}</div>
                    <div class="cal-trades">{trades_str}</div>
                </div>
            </div>
            """
            col.markdown(html, unsafe_allow_html=True)

        # Week total column (square tile)
        week_col = cols[5]
        week_pnl_str = f"${week_pnl:,.2f}"
        if week_pnl > 0:
            week_bg = "#7fcf7f"
            week_pnl_color = "#004d00"
        elif week_pnl < 0:
            week_bg = "#d16f6f"
            week_pnl_color = "#550000"
        else:
            week_bg = "#c4c4c4"
            week_pnl_color = "#000000"

        week_html = f"""
        <div class="cal-cell" style="background-color:{week_bg};">
            <div class="cal-day-label">Week {week_counter}</div>
            <div class="cal-week-summary">
                <div style="color:{week_pnl_color}; font-weight:700;">{week_pnl_str}</div>
                <div>{week_trades} trades</div>
            </div>
        </div>
        """
        week_col.markdown(week_html, unsafe_allow_html=True)

    # Close wrapper
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- UI ----------
st.title("Trading Performance (EST)")

tab_upload, tab_trades, tab_stats, tab_groups, tab_guards = st.tabs(
    ["Upload", "Trades", "Stats", "Groups", "Guardrails"]
)

# Session state
if "just_imported" not in st.session_state:
    st.session_state.just_imported = False
if "last_imported_external_ids" not in st.session_state:
    st.session_state.last_imported_external_ids = []
if "auto_select_after_upload" not in st.session_state:
    st.session_state.auto_select_after_upload = False  # default OFF

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader("Drop your trade export", type=["csv"])
    if up and not st.session_state.just_imported:
        rows, dbg = _read_csv_to_rows(up.read())
        st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
        st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
        st.caption(
            "Canonical mapped pairs: "
            + ", ".join(
                [f"{pair[1]} -> {pair[0]}" for pair in dbg["canonical_mapped_pairs"]]
            )
        )
        st.caption(f"Row count detected: {dbg['row_count']}")
        if rows:
            (
                ins,
                skip,
                errs,
                samples,
                new_ext_ids,
                new_trade_ids,
            ) = _upsert_trades_from_rows(rows)
            if ins > 0:
                st.success(f"Imported/updated {ins} trades (skipped {skip}).")
            else:
                st.error(f"Imported/updated {ins} trades (skipped {skip}).")
            if samples:
                with st.expander("Sample payloads attempted (first 5)"):
                    st.json(samples)
            if errs:
                with st.expander("Errors (first 25)"):
                    st.json(errs)
            st.cache_data.clear()
            st.session_state.last_imported_external_ids = new_ext_ids
            st.session_state.just_imported = True
            st.info("Switch to the Trades tab to tag/group the new trades.")
        else:
            st.error("No rows found.")
    elif st.session_state.just_imported:
        st.info("Upload complete. Switch to the Trades tab to tag/group the new trades.")
        if up is None:
            st.session_state.just_imported = False

# ---- Trades ----
with tab_trades:
    st.subheader("Trades (legs)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        # Toggle whether to auto-select newly uploaded rows
        st.checkbox("Auto-select newly imported trades", key="auto_select_after_upload")

        # Attach tags
        tagmap = _fetch_tags_for(df["id"].tolist())
        df["tags"] = df["id"].map(lambda i: ", ".join(sorted(tagmap.get(i, []))))

        # Display-friendly names
        rename_map = {
            "external_trade_id": "Trade ID",
            "entry_ts_est": "Entry (EST)",
            "exit_ts_est": "Exit (EST)",
            "pnl_net": "PnL (Net)",
        }

        # Insert selection column
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        # Only preselect after upload if toggle is ON
        if (
            st.session_state.auto_select_after_upload
            and st.session_state.last_imported_external_ids
        ):
            df.loc[
                df["external_trade_id"].isin(
                    st.session_state.last_imported_external_ids
                ),
                "selected",
            ] = True

        df_display = df.rename(columns=rename_map)

        view_cols = [
            "selected",
            "Trade ID",
            "symbol",
            "side",
            "Entry (EST)",
            "Exit (EST)",
            "qty",
            "pnl_gross",
            "fees",
            "PnL (Net)",
            "r_multiple",
            "review_status",
            "tags",
        ]
        view_cols = [c for c in view_cols if c in df_display.columns]

        edited = st.data_editor(
            df_display[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "pnl_gross": st.column_config.NumberColumn(
                    "PnL (Gross)", step=0.01, format="%.2f"
                ),
                "fees": st.column_config.NumberColumn(
                    "Fees", step=0.01, format="%.2f"
                ),
                "PnL (Net)": st.column_config.NumberColumn(
                    "PnL (Net)", step=0.01, format="%.2f"
                ),
                "r_multiple": st.column_config.NumberColumn(
                    "R Multiple", step=0.01
                ),
                "review_status": st.column_config.SelectboxColumn(
                    "Review Status",
                    options=["unreviewed", "flagged", "reviewed"],
                ),
                "tags": st.column_config.TextColumn("Tags (read-only)"),
            },
            disabled=[c for c in view_cols if c not in ("selected", "r_multiple", "review_status")],
            num_rows="fixed",
        )

        # Map back to original names and bring internal id for persistence
        edited_back = edited.rename(columns={v: k for k, v in rename_map.items()})
        edited_back = edited_back.merge(
            df[["external_trade_id", "id", "r_multiple", "review_status"]],
            on="external_trade_id",
            how="left",
            suffixes=("", "_old"),
        )

        # Persist inline edits (r_multiple / review_status)
        diff_cols = ["r_multiple", "review_status"]
        to_update = []
        for _, r in edited_back.iterrows():
            rec = {}
            changed = False
            for c in diff_cols:
                new = r.get(c)
                old = r.get(f"{c}_old")
                if pd.isna(new) and pd.isna(old):
                    continue
            # noqa: E501
                if (pd.isna(new) and not pd.isna(old)) or (
                    not pd.isna(new) and pd.isna(old)
                ) or (new != old):
                    rec[c] = new
                    changed = True
            if changed and r.get("id"):
                rec["id"] = r["id"]
                to_update.append(rec)

        if to_update:
            sb.table("tj_trades").upsert(to_update, on_conflict="id").execute()
            st.toast(f"Saved {len(to_update)} edits")
            st.cache_data.clear()

        st.markdown("---")
        # Bulk actions: comments, tags, grouping, ungroup
        selected_ids = edited_back.loc[
            edited_back.get("selected", False) == True, "id"
        ].tolist()
        st.write(f"Selected: {len(selected_ids)}")

        with st.form("bulk_actions", clear_on_submit=True):
            comment = st.text_area("Add comment (markdown)")
            tag_str = st.text_input("Add tags (comma-separated)")

            st.markdown("**Grouping**")
            gmode = st.radio(
                "Action",
                ["None", "Add to existing", "Create new (auto-name)", "Remove from group(s)"],
                horizontal=False,
            )

            existing = None
            new_group_name = None
            notes = st.text_input("Group notes (optional)")

            if gmode == "Add to existing":
                groups = _get_groups()
                if groups:
                    labels = [
                        f"{g['name']} ({g['id'][:6]})" for g in groups
                    ]
                    idx = st.selectbox(
                        "Pick group",
                        list(range(len(labels))) if labels else [],
                        format_func=lambda i: labels[i] if labels else None,
                    )
                    if groups and len(groups) > 0:
                        existing = groups[idx]["id"]
                else:
                    st.info("No groups yet — choose 'Create new (auto-name)'.")
            elif gmode == "Create new (auto-name)":
                suggested = _next_group_name()
                st.caption(f"Suggested name: **{suggested}**")
                new_group_name = suggested  # use suggested automatically

            do = st.form_submit_button("Apply")

        if do and selected_ids:
            if comment:
                _save_comments(selected_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(selected_ids, tags)

            if gmode == "Add to existing" and existing:
                _add_to_group(selected_ids, group_id=existing, notes=notes if notes else None)
            elif gmode == "Create new (auto-name)":
                _add_to_group(selected_ids, new_group_name=new_group_name, notes=notes if notes else None)
            elif gmode == "Remove from group(s)":
                _remove_from_groups(selected_ids)

            st.success("Saved")
            st.cache_data.clear()
            st.session_state.last_imported_external_ids = []
            st.rerun()

# ==== STATS TAB ====
with tab_stats:
    st.subheader("Performance Statistics")

    df_stats = _load_trades()
    if df_stats.empty:
        st.info("No trades yet.")
    else:
        df_stats["pnl_net"] = pd.to_numeric(
            df_stats["pnl_net"], errors="coerce"
        ).fillna(0.0)

        wins_mask = df_stats["pnl_net"] > 0
        losses_mask = df_stats["pnl_net"] < 0
        n_wins = int(wins_mask.sum())
        n_losses = int(losses_mask.sum())

        win_rate = (
            n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0.0
        )
        avg_win = (
            df_stats.loc[wins_mask, "pnl_net"].mean() if n_wins > 0 else 0.0
        )
        avg_loss = (
            df_stats.loc[losses_mask, "pnl_net"].mean()
            if n_losses > 0
            else 0.0
        )
        total_pnl = df_stats["pnl_net"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win rate", f"{win_rate:.1%}")
        c2.metric("Avg Win", f"{avg_win:,.2f}")
        c3.metric("Avg Loss", f"{avg_loss:,.2f}")
        c4.metric("Total PnL", f"{total_pnl:,.2f}")

        # Equity curve
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

        # Daily summary
        df_stats["trade_date"] = df_stats["entry_ts_est"].dt.normalize()
        daily = (
            df_stats.groupby("trade_date")
            .agg(
                pnl_day=("pnl_net", "sum"),
                n_trades=("pnl_net", "size"),
                wins=("pnl_net", lambda s: (s > 0).sum()),
                losses=("pnl_net", lambda s: (s < 0).sum()),
                avg_win=("pnl_net", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                avg_loss=("pnl_net", lambda s: s[s < 0].mean() if (s < 0).any() else 0.0),
            )
            .reset_index()
        )
        daily["win_rate"] = daily.apply(
            lambda r: (r["wins"] / (r["wins"] + r["losses"]))
            if (r["wins"] + r["losses"]) > 0
            else 0.0,
            axis=1,
        )

        # Month selector + calendar
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
            float(month_daily["pnl_day"].sum())
            if not month_daily.empty
            else 0.0
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

        # Daily table: # Trades, PnL, Avg Win, Avg Loss, Win rate (%)
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
        )[["Date", "# Trades", "PnL", "Avg Win", "Avg Loss", "Win rate (%)"]].sort_values(
            "Date", ascending=False
        )

        st.markdown("#### Daily Stats")
        st.dataframe(daily_display, use_container_width=False)

        # Session stats
        st.markdown("### Session Performance")
        df_stats["session"] = df_stats["entry_ts_est"].apply(_classify_session)
        session_stats = (
            df_stats.groupby("session")
            .agg(
                n_trades=("pnl_net", "size"),
                wins=("pnl_net", lambda s: (s > 0).sum()),
                losses=("pnl_net", lambda s: (s < 0).sum()),
                pnl=("pnl_net", "sum"),
                avg_win=("pnl_net", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                avg_loss=("pnl_net", lambda s: s[s < 0].mean() if (s < 0).any() else 0.0),
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

        # Symbol stats
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
                avg_win=("pnl_net", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
                avg_loss=("pnl_net", lambda s: s[s < 0].mean() if (s < 0).any() else 0.0),
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

# ---- Groups (collapsed + details, with hashtag filter) ----
with tab_groups:
    st.subheader("Groups (collapsed positions)")

    groups, mem_df = _fetch_all_groups_with_members()
    if not groups:
        st.info("No groups yet. Create them from the Trades tab after importing.")
    else:
        # COLLAPSED TABLE (one line per group)
        roll = _rollup_by_group(mem_df)

        # join names/notes
        name_map = {g["id"]: g["name"] for g in groups}
        notes_map = {g["id"]: g.get("notes") for g in groups}
        if not roll.empty:
            roll["name"] = roll["group_id"].map(name_map)
            roll["notes"] = roll["group_id"].map(notes_map)

            # Filters
            c1, c2, c3 = st.columns([1, 2, 3])
            with c1:
                day = st.date_input("Filter by day", value=None)
            with c2:
                hashtag_str = st.text_input(
                    "Filter by #hashtags in notes (comma-separated)", value=""
                )
            with c3:
                sym = st.text_input("Symbol filter (optional)", value="").strip().upper()

            rshow = roll.copy()
            # day filter
            if day is not None:
                try:
                    rshow = rshow[rshow["first_entry"].dt.date == day]
                except Exception:
                    pass
            # hashtag filter (all hashtags must appear in notes)
            if hashtag_str.strip():
                tags = [
                    t.strip().lstrip("#").lower()
                    for t in hashtag_str.split(",")
                    if t.strip()
                ]

                def _has_all_hashtags(txt):
                    s = (txt or "").lower()
                    return all(("#" + t) in s for t in tags)

                rshow = rshow[rshow["notes"].apply(_has_all_hashtags)]
            # symbol filter
            if sym:
                rshow = rshow[(rshow["symbol"].fillna("") == sym)]

            show_cols = [
                "name",
                "symbol",
                "side",
                "first_entry",
                "last_exit",
                "legs",
                "total_qty",
                "vwap_entry",
                "vwap_exit",
                "pnl_net_sum",
                "notes",
            ]
            show_cols = [c for c in show_cols if c in rshow.columns]
            st.markdown("**Collapsed (1 row per group)**")
            st.dataframe(
                rshow[show_cols].sort_values(["first_entry"], ascending=[False]),
                use_container_width=False,
            )

            st.divider()

            # DETAILS for a selected group
            st.markdown("**Group details**")
            label_to_id = {
                f"{name_map.get(g['id'],'(unnamed)')} ({g['id'][:6]})": g["id"]
                for g in groups
            }
            choice = st.selectbox("Pick a group", list(label_to_id.keys()))
            gid = label_to_id[choice]

            gdf = mem_df[mem_df["group_id"] == gid].copy()
            if gdf.empty:
                st.info("No member trades in this group.")
            else:
                # summary metrics for the chosen group
                groll = _rollup_by_group(gdf.assign(group_id=gid))
                if not groll.empty:
                    row = groll.iloc[0]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Legs", f"{int(row['legs'])}")
                    m2.metric("Total Qty", f"{row['total_qty']:.0f}")
                    m3.metric(
                        "VWAP Entry",
                        "-"
                        if row["vwap_entry"] is None
                        else f"{row['vwap_entry']:.2f}",
                    )
                    m4.metric(
                        "VWAP Exit",
                        "-"
                        if row["vwap_exit"] is None
                        else f"{row['vwap_exit']:.2f}",
                    )
                    st.metric("Net PnL (group)", f"{row['pnl_net_sum']:.2f}")

                trade_cols = [
                    "external_trade_id",
                    "symbol",
                    "side",
                    "entry_ts_est",
                    "exit_ts_est",
                    "qty",
                    "entry_px",
                    "exit_px",
                    "pnl_net",
                    "r_multiple",
                    "review_status",
                ]
                trade_cols = [c for c in trade_cols if c in gdf.columns]
                st.dataframe(
                    gdf[trade_cols].sort_values("entry_ts_est"),
                    use_container_width=False,
                )
        else:
            st.info("You have groups, but no member trades yet.")

# ---- Guardrails ----
with tab_guards:
    st.subheader("Guardrails (quick checks)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        df = df.sort_values("entry_ts_est")
        df["pnl_net"] = pd.to_numeric(df["pnl_net"], errors="coerce").fillna(0.0)
        st.metric("Win rate", f"{(df['pnl_net'] > 0).mean():.0%}")
        st.metric(
            "Trades (7d)",
            str(
                (df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum()
            ),
        )
        st.metric("Net PnL (sum)", f"{df['pnl_net'].sum():,.2f}")

        hits = []
        ts = df["entry_ts_est"].tolist()
        i = j = 0
        while i < len(ts):
            while j < len(ts) and (ts[j] - ts[i]) <= timedelta(minutes=60):
                j += 1
            if j - i > 6:
                hits.append(
                    {
                        "rule": "trades_per_60m_gt_6",
                        "start": ts[i],
                        "end": ts[j - 1],
                        "count": j - i,
                    }
                )
            i += 1

        streak = 0
        start_loss = None
        for _, r in df.iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                if streak == 0:
                    start_loss = r["entry_ts_est"]
                streak += 1
                if streak > 3:
                    hits.append(
                        {
                            "rule": "consecutive_losses_gt_3",
                            "start": start_loss,
                            "end": r["exit_ts_est"],
                            "count": streak,
                        }
                    )
            else:
                streak = 0
                start_loss = None

        last_loss_exit = None
        for _, r in df.sort_values("exit_ts_est").iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                last_loss_exit = r["exit_ts_est"]
            else:
                if last_loss_exit is not None and (
                    r["entry_ts_est"] - last_loss_exit
                ).total_seconds() <= 5 * 60:
                    hits.append(
                        {
                            "rule": "reentry_under_5m_after_loss",
                            "loss_exit": last_loss_exit,
                            "reentry": r["entry_ts_est"],
                        }
                    )
                last_loss_exit = None

        if hits:
            st.dataframe(pd.DataFrame(hits), use_container_width=False)
        else:
            st.success("No guardrail hits.")
