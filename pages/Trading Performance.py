# pages/Trading Performance.py
#!/usr/bin/env python3
import io
import re
import hashlib
from datetime import timedelta, date

import pandas as pd
import streamlit as st
from supabase import create_client
import altair as alt  # for charts

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
    if len(parts) == 2 and len(parts[1]) == 6 and parts[1][3] == ":" and (parts[1][0] in "+-"):
        s = parts[0]
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _synth_id(row: dict) -> str:
    key = "|".join([
        str(row.get("contractname", "")).strip().upper(),
        str(row.get("enteredat", "")).strip(),
        str(row.get("exitedat", "")).strip(),
        str(row.get("entryprice", "")).strip(),
        str(row.get("exitprice", "")).strip(),
        str(row.get("size", "")).strip(),
    ])
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
        sep=None,                 # auto-detect comma or tab
        engine="python",
        dtype=str,
        keep_default_na=False,    # keep empty cells as "", not NaN
    )
    original_headers = list(df.columns)
    norm_headers = [_clean_header(h) for h in original_headers]
    df.columns = norm_headers

    alias_map = {
        "id": "id", "trade id": "id", "tradeid": "id", "external id": "id",
        "contractname": "contractname", "contract": "contractname", "market": "contractname", "symbol": "contractname",
        "enteredat": "enteredat", "entry time": "enteredat", "entry": "enteredat",
        "exitedat": "exitedat", "exit time": "exitedat", "exit": "exitedat",
        "entryprice": "entryprice", "entry price": "entryprice",
        "exitprice": "exitprice", "exit price": "exitprice",
        "size": "size", "quantity": "size", "qty": "size",
        "type": "type", "side": "type",
        "pnl": "pnl", "p&l": "pnl", "profit": "pnl",
        "fees": "fees", "fee": "fees",
        "commissions": "commissions", "commission": "commissions",
        "tradeday": "tradeday", "tradeduration": "tradeduration",
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
        "canonical_mapped_pairs": sorted([(k, v) for k, v in canonical_cols.items()]),
        "row_count": len(rows),
    }
    return rows, debug

# ---------- insert-or-update (manual upsert keyed by user_id + external_trade_id) ----------
def _insert_or_update_trade(payload: dict):
    q = sb.table("tj_trades").select("id", count="exact") \
        .eq("external_trade_id", payload["external_trade_id"]) \
        .eq("user_id", payload["user_id"]) \
        .limit(1)
    res = q.execute()
    existing = (res.data or [])
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
            side = "long" if t in ("long", "buy", "b") else ("short" if t in ("short", "sell", "s") else None)

            qty_val = r.get("size")
            symbol = (r.get("contractname") or "").strip().upper() or None

            f1 = _as_float(r.get("fees")) or 0.0
            f2 = _as_float(r.get("commissions")) or 0.0
            fees_val = float(f1) + float(f2)

            entry_iso = _to_iso_est(r.get("enteredat"))
            exit_iso  = _to_iso_est(r.get("exitedat"))
            if not entry_iso:
                raise ValueError(f"Missing/Bad EnteredAt: {r.get('enteredat')}")

            payload = {
                "user_id": USER_ID,                 # <- REQUIRED (tj_trades.user_id NOT NULL)
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": entry_iso,
                "exit_ts_est":  exit_iso,           # may be None for open trades
                "entry_px":  _as_float(r.get("entryprice")),
                "exit_px":   _as_float(r.get("exitprice")),
                "qty":       _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees":      fees_val,
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
                errs.append({
                    "id": r.get("id"),
                    "enteredat": r.get("enteredat"),
                    "exitedat": r.get("exitedat"),
                    "error": str(e),
                })

    return inserted, skipped, errs, samples, inserted_external_ids, inserted_trade_ids

# ---------- Load ----------
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = sb.table("tj_trades").select(
        "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
        "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
    ).eq("user_id", USER_ID).order("entry_ts_est", desc=True).limit(limit).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"]  = pd.to_datetime(df["exit_ts_est"])
    return df

def _fetch_tags_for(trade_ids):
    if not trade_ids:
        return {}
    data = sb.table("tj_trade_tags").select("trade_id,tag").in_("trade_id", trade_ids).eq("user_id", USER_ID).execute().data
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

    rows = [{"trade_id": tid, "user_id": USER_ID, "tag": t} for tid in trade_ids for t in tags]
    # Attempt #1
    try:
        sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag,user_id").execute()
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
        existing = sb.table("tj_trade_tags").select("trade_id,tag").in_("trade_id", trade_ids).eq("user_id", USER_ID).execute().data or []
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
        g = sb.table("tj_trade_groups").insert({"user_id": USER_ID, "name": new_group_name, "notes": notes}).execute().data[0]
        group_id = g["id"]
    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        # also robust upsert in case unique index differs
        try:
            sb.table("tj_trade_group_members").upsert(rows, on_conflict="group_id,trade_id").execute()
        except Exception:
            # fallback: delete duplicates then insert
            sb.table("tj_trade_group_members").delete().in_("trade_id", trade_ids).eq("group_id", group_id).execute()
            sb.table("tj_trade_group_members").insert(rows).execute()
    return group_id

def _remove_from_groups(trade_ids):
    """Ungroup: remove selected trades from any groups they belong to."""
    if not trade_ids:
        return
    sb.table("tj_trade_group_members").delete().in_("trade_id", trade_ids).execute()

def _get_groups():
    res = sb.table("tj_trade_groups").select("id,name,notes,created_at").eq("user_id", USER_ID).order("created_at", desc=True).execute()
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
    groups = sb.table("tj_trade_groups")\
        .select("id,name,notes,created_at")\
        .eq("user_id", USER_ID)\
        .order("created_at", desc=True)\
        .execute().data or []

    mem = sb.table("tj_trade_group_members")\
        .select("group_id, trade_id, tj_trades(*)")\
        .execute().data or []

    rows = []
    for m in mem or []:
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
            vwap_entry = float((gdf["entry_px"].where(ok_e, 0) * qty).sum() / denom_e)

        # VWAP Exit
        ok_x = gdf["exit_px"].notna()
        w_exit = qty.where(ok_x, 0)
        denom_x = w_exit.sum()
        vwap_exit = None
        if denom_x and denom_x != 0:
            vwap_exit = float((gdf["exit_px"].where(ok_x, 0) * qty).sum() / denom_x)

        first_entry = gdf["entry_ts_est"].min()
        last_exit   = gdf["exit_ts_est"].max()
        total_qty   = float(gdf["qty"].fillna(0).sum())

        # Net PnL
        pnl_sum = float((gdf["pnl_net"].fillna(0) if "pnl_net" in gdf else 0).sum())
        if pnl_sum == 0 and ("pnl_gross" in gdf or "fees" in gdf):
            pnl_sum = float(gdf["pnl_gross"].fillna(0).sum() - gdf["fees"].fillna(0).sum())

        out.append({
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
        })
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
      - Afternoon:13:30:00–16:59:59
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

    # Afternoon 13:30:00–16:59:59
    if 13 * 3600 + 30 * 60 <= seconds <= 16 * 3600 + 59 * 60 + 59:
        return "Afternoon (13:30–16:59:59)"

    return "Other"

# ===== Helper: PnL calendar renderer =====
def _render_pnl_calendar(month_daily: pd.DataFrame, period):
    """
    Render a month calendar with one tile per day:
      - Big PnL number (green/red)
      - # trades
      - On Saturdays: weekly PnL + # trades (Week 1, Week 2, ...)
    `month_daily` must have columns: trade_date (datetime), pnl_day, n_trades.
    `period` is a pandas Period('YYYY-MM').
    """
    if month_daily.empty:
        st.info("No trades for this month.")
        return

    month_daily = month_daily.copy()
    month_daily["date_only"] = month_daily["trade_date"].dt.date
    by_date = month_daily.set_index("date_only")[["pnl_day", "n_trades"]].to_dict("index")

    # Calendar bounds
    first_ts = period.to_timestamp()
    year, month = first_ts.year, first_ts.month
    first_day = date(year, month, 1)
    # last_day is last day of that month
    last_day = (first_ts + first_ts.freq - pd.Timedelta(days=1)).date()

    # Start on Sunday before or equal to first_day
    weekday_mon0 = first_day.weekday()          # Mon=0..Sun=6
    sunday_offset = (weekday_mon0 + 1) % 7      # days back to Sunday
    start_date = first_day - timedelta(days=sunday_offset)

    # Styles
    st.markdown(
        """
        <style>
        .cal-cell {
            border: 1px solid #333;
            border-radius: 6px;
            padding: 4px 6px;
            min-height: 90px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background-color: #141414;
        }
        .cal-day-label {
            font-size: 0.7rem;
            opacity: 0.7;
        }
        .cal-pnl {
            font-size: 0.9rem;
            font-weight: 700;
        }
        .cal-trades {
            font-size: 0.7rem;
            opacity: 0.8;
        }
        .cal-week-summary {
            margin-top: 4px;
            font-size: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header row Su..Sa
    header_cols = st.columns(7)
    for col, name in zip(header_cols, ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]):
        col.markdown(f"<div style='text-align:center; font-weight:600;'>{name}</div>", unsafe_allow_html=True)

    week_counter = 0
    # Up to 6 calendar weeks
    for week_idx in range(6):
        row_start = start_date + timedelta(days=7 * week_idx)
        row_end = row_start + timedelta(days=6)
        week_dates = [row_start + timedelta(days=d) for d in range(7)]

        # Does this week touch the selected month?
        has_month_day = any((d.month == month and d.year == year) for d in week_dates)
        if not has_month_day and row_start > last_day:
            break
        if has_month_day:
            week_counter += 1

        in_week = month_daily[
            (month_daily["trade_date"].dt.date >= row_start)
            & (month_daily["trade_date"].dt.date <= row_end)
        ]
        week_pnl = float(in_week["pnl_day"].sum()) if not in_week.empty else 0.0
        week_trades = int(in_week["n_trades"].sum()) if not in_week.empty else 0

        cols = st.columns(7)
        for i, (col, d) in enumerate(zip(cols, week_dates)):
            if d.month != month or d.year != year:
                col.markdown("<div class='cal-cell'></div>", unsafe_allow_html=True)
                continue

            stats = by_date.get(d)
            pnl = stats["pnl_day"] if stats else 0.0
            n_trades = stats["n_trades"] if stats else 0

            if pnl > 0:
                bg_color = "rgba(0, 128, 0, 0.35)"
                pnl_color = "#7CFC00"
            elif pnl < 0:
                bg_color = "rgba(139, 0, 0, 0.45)"
                pnl_color = "#FF6A6A"
            else:
                bg_color = "rgba(80, 80, 80, 0.45)"
                pnl_color = "#CCCCCC"

            pnl_str = f"${pnl:,.2f}"
            trades_str = f"{int(n_trades)} trades"

            week_summary_html = ""
            if i == 6 and has_month_day:
                week_pnl_str = f"${week_pnl:,.2f}"
                week_pnl_color = "#7CFC00" if week_pnl > 0 else ("#FF6A6A" if week_pnl < 0 else "#CCCCCC")
                week_summary_html = f"""
                    <div class='cal-week-summary'>
                        <div><b>Week {week_counter}</b></div>
                        <div style="color:{week_pnl_color};">{week_pnl_str}</div>
                        <div>{week_trades} trades</div>
                    </div>
                """

            html = f"""
            <div class="cal-cell" style="background-color:{bg_color};">
                <div class="cal-day-label">{d.day}</div>
                <div style="text-align:center;">
                    <div class="
