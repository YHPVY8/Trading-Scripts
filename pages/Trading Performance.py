# pages/Trading Performance.py
#!/usr/bin/env python3
import io
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ===== CONFIG =====
USE_USER_SCOPING = False  # <- set True only if you want per-user rows
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

# ---------- CSV -> canonical rows (via pandas) ----------
def _read_csv_to_rows(uploaded_bytes: bytes):
    df = pd.read_csv(
        io.BytesIO(uploaded_bytes),
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
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

# ---------- Upsert ----------
def _upsert_trades_from_rows(rows):
    inserted = skipped = 0
    errs, samples = [], []

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

            payload = {
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": _to_iso_est(r.get("enteredat")),
                "exit_ts_est":  _to_iso_est(r.get("exitedat")),
                "entry_px":  _as_float(r.get("entryprice")),
                "exit_px":   _as_float(r.get("exitprice")),
                "qty":       _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees":      fees_val,
                "source": "csv",
            }
            if USE_USER_SCOPING:
                payload["user_id"] = USER_ID

            if len(samples) < 5:
                samples.append(dict(payload))

            if not payload["entry_ts_est"] or not payload["exit_ts_est"]:
                raise ValueError(f"Bad timestamps: {r.get('enteredat')} / {r.get('exitedat')}")

            # Choose on_conflict target based on mode
            conflict_target = "user_id,external_trade_id" if USE_USER_SCOPING else "external_trade_id"
            sb.table("tj_trades").upsert(payload, on_conflict=conflict_target).execute()
            inserted += 1

        except Exception as e:
            skipped += 1
            if len(errs) < 15:
                errs.append({"external_trade_id": r.get("id"), "error": repr(e)})

    return inserted, skipped, errs, samples

@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    sel = sb.table("tj_trades").select(
        "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
        "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
    )
    if USE_USER_SCOPING:
        sel = sel.eq("user_id", USER_ID)
    res = sel.order("entry_ts_est", desc=True).limit(limit).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"]  = pd.to_datetime(df["exit_ts_est"])
    return df

# ---------- UI ----------
st.title("Trading Performance (EST)")

tab_upload, tab_trades, tab_groups, tab_guards = st.tabs(["Upload", "Trades", "Groups", "Guardrails"])

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    mode_text = "Scoped by USER_ID" if USE_USER_SCOPING else "Global (no user_id)"
    st.caption(f"Mode: {mode_text}" + (f" — USER_ID: {USER_ID}" if USE_USER_SCOPING else ""))

    up = st.file_uploader("Drop your trade export (tabs or commas OK; headers auto-detected)", type=["csv"])
    if up:
        rows, dbg = _read_csv_to_rows(up.read())
        st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
        st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
        st.caption("Canonical mapped pairs: " + ", ".join([f"{pair[1]} -> {pair[0]}" for pair in dbg["canonical_mapped_pairs"]]))
        st.caption(f"Row count detected: {dbg['row_count']}")

        if not rows:
            st.error("No data rows found after parsing.")
        else:
            ins, skip, errs, samples = _upsert_trades_from_rows(rows)
            if ins > 0:
                st.success(f"Imported {ins} rows (skipped {skip}).")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Imported {ins} rows (skipped {skip}).")
                if samples:
                    with st.expander("Sample payloads attempted (first 5)"):
                        st.json(samples)
                if errs:
                    with st.expander("Sample errors (first 15)"):
                        st.json(errs)

# ---- Trades ----
with tab_trades:
    st.subheader("Trades")
    df = _load_trades()
    if df.empty:
        hint = "Upload trades in the Upload tab."
        if USE_USER_SCOPING:
            hint += " (Check that your USER_ID in secrets matches the one used at insert.)"
        st.info(hint)
    else:
        orig = df.copy()
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        rename_map = {"entry_ts_est": "Entry (EST)", "exit_ts_est": "Exit (EST)"}
        df_display = df.rename(columns=rename_map)

        editable_cols = ["selected", "planned_risk", "r_multiple", "review_status"]
        edited = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("Select", help="Select for bulk actions"),
                "planned_risk": st.column_config.NumberColumn("Planned Risk", step=0.01),
                "r_multiple": st.column_config.NumberColumn("R Multiple", step=0.01),
                "review_status": st.column_config.SelectboxColumn(
                    "Review Status", options=["unreviewed", "flagged", "reviewed"]
                ),
            },
            disabled=[c for c in df_display.columns if c not in editable_cols],
            num_rows="fixed",
        )
        edited = edited.rename(columns={v: k for k, v in rename_map.items()})

        # Persist inline edits
        diff_cols = ["planned_risk", "r_multiple", "review_status"]
        to_update = []
        merged = edited.merge(orig[["id"] + diff_cols], on="id", suffixes=("", "_old"))
        for _, r in merged.iterrows():
            changed = any(
                (pd.isna(r[c]) and not pd.isna(r[f"{c}_old"])) or
                (not pd.isna(r[c]) and pd.isna(r[f"{c}_old"])) or
                (r[c] != r[f"{c}_old"])
                for c in diff_cols
            )
            if changed:
                to_update.append({
                    "id": r["id"],
                    "planned_risk": r["planned_risk"],
                    "r_multiple": r["r_multiple"],
                    "review_status": r["review_status"],
                })
        if to_update:
            sb.table("tj_trades").upsert(to_update, on_conflict="id").execute()
            st.toast(f"Saved {len(to_update)} inline edit(s)")
            st.cache_data.clear()

# ---- Groups ----
with tab_groups:
    st.subheader("Groups")
    def _get_groups():
        q = sb.table("tj_trade_groups").select("id,name,created_at")
        if USE_USER_SCOPING:
            q = q.eq("user_id", USER_ID)
        return (q.order("created_at", desc=True).execute().data) or []
    groups = _get_groups()
    if not groups:
        st.info("No groups yet. Create one from the Trades tab.")
    else:
        gmap = {f"{g['name']} ({g['id'][:6]})": g["id"] for g in groups}
        choice = st.selectbox("Choose a group", list(gmap.keys()))
        gid = gmap[choice]
        mem = sb.table("tj_trade_group_members").select("trade_id, tj_trades(*)").eq("group_id", gid).execute().data
        rows = [m["tj_trades"] for m in mem if m.get("tj_trades")]
        gdf = pd.DataFrame(rows or [])
        if gdf.empty:
            st.info("No trades in this group yet.")
        else:
            show_cols = ["external_trade_id","symbol","side","entry_ts_est","exit_ts_est","qty","pnl_net","r_multiple","review_status"]
            st.dataframe(gdf[show_cols], use_container_width=True)
            st.write(f"Group PnL (net): {gdf['pnl_net'].sum():,.2f}")

# ---- Guardrails ----
with tab_guards:
    st.subheader("Guardrails (quick checks)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        df = df.sort_values("entry_ts_est")
        st.metric("Win rate", f"{(df['pnl_net']>0).mean():.0%}")
        st.metric("Trades (7d)", str((df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum()))
        st.metric("Net P&L (sum)", f"{df['pnl_net'].sum():,.2f}")

        hits = []
        ts = df["entry_ts_est"].tolist()
        i = j = 0
        while i < len(ts):
            while j < len(ts) and (ts[j] - ts[i]) <= timedelta(minutes=60):
                j += 1
            if j - i > 6:
                hits.append({"rule": "trades_per_60m_gt_6", "start": ts[i], "end": ts[j-1], "count": j - i})
            i += 1

        streak = 0; start = None
        for _, r in df.iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                if streak == 0: start = r["entry_ts_est"]
                streak += 1
                if streak > 3:
                    hits.append({"rule": "consecutive_losses_gt_3", "start": start, "end": r["exit_ts_est"], "count": streak})
            else:
                streak = 0; start = None

        last_loss_exit = None
        for _, r in df.sort_values("exit_ts_est").iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                last_loss_exit = r["exit_ts_est"]
            else:
                if last_loss_exit is not None and (r["entry_ts_est"] - last_loss_exit).total_seconds() <= 5 * 60:
                    hits.append({"rule": "reentry_under_5m_after_loss", "loss_exit": last_loss_exit, "reentry": r["entry_ts_est"]})
                last_loss_exit = None

        if hits:
            st.dataframe(pd.DataFrame(hits), use_container_width=True)
        else:
            st.success("No guardrail hits.")
